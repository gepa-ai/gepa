"""RemoteAdapter is a GEPAAdapter whose evaluate/make_reflective_dataset calls
are proxied across a gRPC stream to a connected client.

The adapter is owned by the server-side RunOptimization handler. It places
outbound ServerMessages onto a queue (drained by the handler's response
generator) and blocks on Future objects keyed by request_id. The handler's
reader thread fulfills those futures when the matching ClientMessage arrives.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.rpc.conversions import RemoteExample, RemoteTrajectory, reflective_data_to_python
from gepa.rpc.generated import gepa_pb2 as pb

# Default ceiling on how long the server waits for a client to answer a single
# EvaluateBatchRequest / ReflectiveDatasetRequest / OmniEvaluateBatchRequest.
# Without this, a hung or malicious client blocks the run's runner thread --
# and, transitively, the gRPC pool slot its stream occupies -- forever.
DEFAULT_CALLBACK_TIMEOUT_SECONDS = 300.0


class RemoteAdapterCancelledError(RuntimeError):
    """Raised inside the optimizer thread when the client stream is gone."""


class ClientEvaluatorError(RuntimeError):
    """The client reported that its own evaluate / reflective-dataset callback
    raised, via an EvaluatorFailure message.

    Distinct from EvaluatorResponseError (the client answered, but the answer
    was malformed): here the client never produced an answer at all and told
    us why. Treated as a client-side fault for reporting purposes, so the
    detail it carries is forwarded rather than sanitized -- the client
    authored that text about its own code.
    """


class EvaluatorResponseError(ValueError):
    """Raised when the client's own response is malformed -- the wrong number
    of scores/outputs/trajectories for the batch it was handed.

    A distinct type so servicer.py can report this as the client's own
    mistake (actionable without server logs) rather than an internal failure.
    """


class RemoteAdapterTimeoutError(RuntimeError):
    """Raised inside the optimizer thread when a client callback times out.

    Distinct from RemoteAdapterCancelledError: the client is still connected,
    it just didn't answer in time.
    """


class RunCancelledError(RuntimeError):
    """Raised inside the optimizer thread after an explicit Cancel RPC.

    Distinct from RemoteAdapterCancelledError (client disconnected) and
    RemoteAdapterTimeoutError (client too slow): the run was cancelled on
    purpose, by request, while the client was still connected and responsive.
    """


class RemoteAdapter(GEPAAdapter[RemoteExample, RemoteTrajectory, str]):
    def __init__(
        self,
        outbound: queue.Queue[pb.ServerMessage | None],
        callback_timeout: float = DEFAULT_CALLBACK_TIMEOUT_SECONDS,
    ):
        self._outbound = outbound
        self._callback_timeout = callback_timeout
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._cancelled = False
        self._cancel_exc: BaseException | None = None

    # Wiring
    def deliver_evaluate_response(self, resp: pb.EvaluateBatchResponse) -> None:
        self._resolve(resp.request_id, resp)

    def deliver_reflective_response(self, resp: pb.ReflectiveDatasetResponse) -> None:
        self._resolve(resp.request_id, resp)

    def cancel(self, exc: BaseException | None = None) -> None:
        """Fail every in-flight call. Called on stream-close or explicit Cancel.

        Safe to call more than once (e.g. once from the reader thread on
        stream-close and once from an explicit Cancel RPC): `self._pending`
        is only ever drained once, under `self._lock`, so a second call finds
        nothing left to fail.

        This pop-under-lock hand-off is also why cancellation can't race a
        normally-completing future: whichever of `cancel`/`_resolve` pops a
        given request_id first is the only one that ever touches its future,
        so `set_exception`/`set_result` can never both fire on the same
        future and raise InvalidStateError.

        Note this only unblocks what this adapter's caller is waiting on
        (effect 1 of cancellation) -- it does not by itself stop the
        optimizer loop from issuing more work (effect 2). See servicer.py's
        Cancel handler for how the two are wired together.
        """
        err = exc or RemoteAdapterCancelledError("client stream closed")
        with self._lock:
            self._cancelled = True
            self._cancel_exc = err
            pending = list(self._pending.values())
            self._pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(err)

    def _resolve(self, request_id: str, payload: Any) -> None:
        with self._lock:
            fut = self._pending.pop(request_id, None)
        if fut is not None and not fut.done():
            fut.set_result(payload)

    def fail(self, request_id: str, exc: BaseException) -> None:
        """Fail one pending request (or every pending one, if request_id is
        empty or unknown). Used when the client reports its own callback
        raised, so the run surfaces that instead of a bare stream close."""
        with self._lock:
            if request_id and request_id in self._pending:
                futs = [self._pending.pop(request_id)]
            else:
                futs = list(self._pending.values())
                self._pending.clear()
        for fut in futs:
            if not fut.done():
                fut.set_exception(exc)

    def _new_pending(self) -> tuple[str, Future]:
        request_id = str(uuid.uuid4())
        fut: Future = Future()
        with self._lock:
            if self._cancelled:
                # Don't queue work after cancel.
                raise self._cancel_exc or RemoteAdapterCancelledError("adapter cancelled")
            self._pending[request_id] = fut
        return request_id, fut

    def _await(self, request_id: str, fut: Future) -> Any:
        """Block on `fut`, bounded by the configured callback timeout.

        On timeout, drop the now-abandoned future from `self._pending` under
        the same lock `cancel`/`_resolve` use, so a later `cancel()` can't
        try to act on a future this call already gave up on.
        """
        try:
            return fut.result(timeout=self._callback_timeout)
        except FutureTimeoutError:
            with self._lock:
                self._pending.pop(request_id, None)
            raise RemoteAdapterTimeoutError(
                f"client did not respond to request {request_id} within "
                f"{self._callback_timeout}s"
            ) from None

    # GEPAAdapter
    def evaluate(
        self,
        batch: list[RemoteExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[RemoteTrajectory, str]:
        request_id, fut = self._new_pending()
        request = pb.EvaluateBatchRequest(
            request_id=request_id,
            candidate=dict(candidate),
            batch=[ex.to_proto() for ex in batch],
            capture_traces=capture_traces,
        )
        self._outbound.put(pb.ServerMessage(evaluate_batch_request=request))
        resp: pb.EvaluateBatchResponse = self._await(request_id, fut)

        outputs = list(resp.outputs)
        scores = list(resp.scores)
        if len(outputs) != len(batch) or len(scores) != len(batch):
            raise EvaluatorResponseError(
                f"client returned mismatched evaluate response: "
                f"got {len(outputs)} outputs, {len(scores)} scores for batch of {len(batch)}"
            )

        trajectories: list[RemoteTrajectory] | None = None
        if capture_traces:
            if len(resp.trajectories) != len(batch):
                raise EvaluatorResponseError(
                    f"capture_traces=True but client returned {len(resp.trajectories)} trajectories "
                    f"for batch of {len(batch)}"
                )
            trajectories = [RemoteTrajectory.from_proto(t) for t in resp.trajectories]

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[RemoteTrajectory, str],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            raise ValueError("trajectories are required to build a reflective dataset")

        request_id, fut = self._new_pending()
        request = pb.ReflectiveDatasetRequest(
            request_id=request_id,
            candidate=dict(candidate),
            components_to_update=list(components_to_update),
            trajectories=[t.to_proto() for t in eval_batch.trajectories],
        )
        self._outbound.put(pb.ServerMessage(reflective_dataset_request=request))
        resp: pb.ReflectiveDatasetResponse = self._await(request_id, fut)
        return reflective_data_to_python(resp)


def _opt_state_to_proto(opt_state: Any) -> pb.OmniOptimizationState:
    """Convert an OptimizationState (or dict with the same shape) to proto."""
    best_evals = getattr(opt_state, "best_example_evals", None)
    if best_evals is None and isinstance(opt_state, dict):
        best_evals = opt_state.get("best_example_evals")
    entries = []
    for e in best_evals or []:
        score = e.get("score", 0.0) if isinstance(e, dict) else getattr(e, "score", 0.0)
        side_info = e.get("side_info", {}) if isinstance(e, dict) else getattr(e, "side_info", {})
        try:
            side_info_json = json.dumps(side_info)
        except TypeError:
            # side_info may hold non-JSON values (e.g. an Image); fall back to a string repr.
            side_info_json = json.dumps({"raw": str(side_info)})
        entries.append(pb.OmniBestEval(score=float(score), side_info=side_info_json))
    return pb.OmniOptimizationState(best_example_evals=entries)


def _example_to_proto(ex: Any) -> pb.Example:
    """Convert a dataset example to a proto Example."""
    if isinstance(ex, dict):
        if "fields" in ex:
            return pb.Example(id=str(ex.get("id", "")), fields={k: str(v) for k, v in ex["fields"].items()})
        return pb.Example(id="", fields={k: str(v) for k, v in ex.items()})
    if hasattr(ex, "id") and hasattr(ex, "fields"):
        return pb.Example(id=str(ex.id), fields={k: str(v) for k, v in ex.fields.items()})
    return pb.Example(id="", fields={"value": str(ex)})


class OmniRemoteEvaluator:
    """batch_evaluator passed to optimize_anything.

    Groups (candidate, example) pairs by candidate and sends one
    OmniEvaluateBatchRequest per unique candidate. The reader thread
    resolves futures via deliver_response().

    Tracks metric_calls_used directly (one increment per example scored,
    across every group) rather than relying on a GEPA-specific progress
    callback, because this evaluator -- unlike _OmniProgressCallback --
    runs for every engine (gepa, autoresearch, best_of_n, meta_harness).
    Without it, non-gepa engines always reported metric_calls_used=0.
    """

    def __init__(
        self,
        outbound: queue.Queue[pb.OmniServerMessage | None],
        callback_timeout: float = DEFAULT_CALLBACK_TIMEOUT_SECONDS,
        run_status: dict[str, Any] | None = None,
    ):
        self._outbound = outbound
        self._callback_timeout = callback_timeout
        self._run_status = run_status
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._cancelled = False
        self._cancel_exc: BaseException | None = None

    # Wiring
    def deliver_response(self, resp: pb.OmniEvaluateBatchResponse) -> None:
        self._resolve(resp.request_id, resp)

    def cancel(self, exc: BaseException | None = None) -> None:
        """See RemoteAdapter.cancel -- identical hand-off, same race-free
        reasoning, same two-distinct-effects caveat."""
        err = exc or RemoteAdapterCancelledError("client stream closed")
        with self._lock:
            self._cancelled = True
            self._cancel_exc = err
            pending = list(self._pending.values())
            self._pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(err)

    def _resolve(self, request_id: str, payload: Any) -> None:
        with self._lock:
            fut = self._pending.pop(request_id, None)
        if fut is not None and not fut.done():
            fut.set_result(payload)

    def fail(self, request_id: str, exc: BaseException) -> None:
        """See RemoteAdapter.fail."""
        with self._lock:
            if request_id and request_id in self._pending:
                futs = [self._pending.pop(request_id)]
            else:
                futs = list(self._pending.values())
                self._pending.clear()
        for fut in futs:
            if not fut.done():
                fut.set_exception(exc)

    def _new_pending(self) -> tuple[str, Future]:
        request_id = str(uuid.uuid4())
        fut: Future = Future()
        with self._lock:
            if self._cancelled:
                raise self._cancel_exc or RemoteAdapterCancelledError("adapter cancelled")
            self._pending[request_id] = fut
        return request_id, fut

    def _await(self, request_id: str, fut: Future) -> Any:
        """See RemoteAdapter._await."""
        try:
            return fut.result(timeout=self._callback_timeout)
        except FutureTimeoutError:
            with self._lock:
                self._pending.pop(request_id, None)
            raise RemoteAdapterTimeoutError(
                f"client did not respond to request {request_id} within "
                f"{self._callback_timeout}s"
            ) from None

    # batch_evaluator
    def __call__(
        self,
        pairs: list[tuple[Any, Any]],
        opt_states: list[Any] | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        if not pairs:
            return []

        # Group by candidate. GEPA typically sends one candidate per call
        # but the contract allows mixed batches. opt_states, when present,
        # is aligned 1:1 with pairs -- carry each entry's state along.
        groups: dict[str, list[tuple[int, Any, Any]]] = defaultdict(list)
        for idx, (candidate, example) in enumerate(pairs):
            state = opt_states[idx] if opt_states is not None else None
            groups[str(candidate)].append((idx, example, state))

        results: list[tuple[float, dict[str, Any]]] = [(0.0, {}) for _ in range(len(pairs))]

        # Dispatch every group's request before awaiting any of them, so a
        # client that evaluates candidate groups concurrently on its own side
        # isn't forced to serialize behind us. Previously this put one
        # group's request, blocked on its response, and only then put the
        # next group's -- making multi-group calls strictly sequential no
        # matter how parallel the client was.
        inflight: list[tuple[str, Future, list[int], list[Any]]] = []
        for candidate_str, indexed_examples in groups.items():
            indices = [i for i, _, _ in indexed_examples]
            examples = [ex for _, ex, _ in indexed_examples]
            states = [st for _, _, st in indexed_examples]

            request_kwargs: dict[str, Any] = {
                "candidate": candidate_str,
                "batch": [_example_to_proto(ex) for ex in examples],
            }
            if opt_states is not None:
                request_kwargs["opt_states"] = [_opt_state_to_proto(st) for st in states]

            request_id, fut = self._new_pending()
            self._outbound.put(pb.OmniServerMessage(
                evaluate_batch_request=pb.OmniEvaluateBatchRequest(request_id=request_id, **request_kwargs)
            ))
            inflight.append((request_id, fut, indices, examples))

        for request_id, fut, indices, examples in inflight:
            resp: pb.OmniEvaluateBatchResponse = self._await(request_id, fut)

            if len(resp.scores) != len(examples):
                raise EvaluatorResponseError(
                    f"client returned {len(resp.scores)} scores for batch of {len(examples)}"
                )

            for i, score in enumerate(resp.scores):
                side_info_json = resp.side_infos[i] if i < len(resp.side_infos) else ""
                try:
                    side_info = json.loads(side_info_json) if side_info_json else {}
                except json.JSONDecodeError:
                    side_info = {"raw": side_info_json}
                results[indices[i]] = (float(score), side_info)

            if self._run_status is not None:
                self._run_status["metric_calls_used"] = (
                    self._run_status.get("metric_calls_used", 0) + len(examples)
                )

        return results
