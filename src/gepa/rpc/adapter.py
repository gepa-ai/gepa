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
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.rpc.conversions import RemoteExample, RemoteTrajectory, reflective_data_to_python
from gepa.rpc.generated import gepa_pb2 as pb


class RemoteAdapterCancelledError(RuntimeError):
    """Raised inside the optimizer thread when the client stream is gone."""


class RemoteAdapter(GEPAAdapter[RemoteExample, RemoteTrajectory, str]):
    def __init__(self, outbound: queue.Queue[pb.ServerMessage | None]):
        self._outbound = outbound
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
        """Called when the client stream ends. Fail every in-flight call."""
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

    def _new_pending(self) -> tuple[str, Future]:
        request_id = str(uuid.uuid4())
        fut: Future = Future()
        with self._lock:
            if self._cancelled:
                # Don't queue work after cancel.
                raise self._cancel_exc or RemoteAdapterCancelledError("adapter cancelled")
            self._pending[request_id] = fut
        return request_id, fut

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
        resp: pb.EvaluateBatchResponse = fut.result()

        outputs = list(resp.outputs)
        scores = list(resp.scores)
        if len(outputs) != len(batch) or len(scores) != len(batch):
            raise ValueError(
                f"client returned mismatched evaluate response: "
                f"got {len(outputs)} outputs, {len(scores)} scores for batch of {len(batch)}"
            )

        trajectories: list[RemoteTrajectory] | None = None
        if capture_traces:
            if len(resp.trajectories) != len(batch):
                raise ValueError(
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
        resp: pb.ReflectiveDatasetResponse = fut.result()
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
    OmniEvaluateBatchRequest per unique candidate, blocking on a Future
    until the client responds. The reader thread resolves futures via
    deliver_response().
    """

    def __init__(self, outbound: queue.Queue[pb.OmniServerMessage | None]):
        self._outbound = outbound
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._cancelled = False
        self._cancel_exc: BaseException | None = None

    # Wiring
    def deliver_response(self, resp: pb.OmniEvaluateBatchResponse) -> None:
        self._resolve(resp.request_id, resp)

    def cancel(self, exc: BaseException | None = None) -> None:
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

    def _new_pending(self) -> tuple[str, Future]:
        request_id = str(uuid.uuid4())
        fut: Future = Future()
        with self._lock:
            if self._cancelled:
                raise self._cancel_exc or RemoteAdapterCancelledError("adapter cancelled")
            self._pending[request_id] = fut
        return request_id, fut

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
            resp: pb.OmniEvaluateBatchResponse = fut.result()

            if len(resp.scores) != len(examples):
                raise ValueError(
                    f"client returned {len(resp.scores)} scores for batch of {len(examples)}"
                )

            for i, score in enumerate(resp.scores):
                side_info_json = resp.side_infos[i] if i < len(resp.side_infos) else ""
                try:
                    side_info = json.loads(side_info_json) if side_info_json else {}
                except json.JSONDecodeError:
                    side_info = {"raw": side_info_json}
                results[indices[i]] = (float(score), side_info)

        return results
