"""gRPC servicer for GEPAService.

Threading model for RunOptimization:
- handler thread: reads first message (StartRequest), then drains the outbound
  queue and yields ServerMessages back to the client.
- reader thread: pulls subsequent ClientMessages and resolves adapter futures.
- runner thread: invokes gepa.optimize(adapter=RemoteAdapter(...)) synchronously.

Checkpointing: gepa.optimize persists state under run_dir. Re-issuing
RunOptimization with the same run_id on a fresh stream will resume from disk.
"""

from __future__ import annotations

import logging
import os
import pathlib
import queue
import re
import threading
import uuid
from collections.abc import Collection
from typing import Any

import grpc

import gepa
from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything
from gepa.rpc.adapter import (
    DEFAULT_CALLBACK_TIMEOUT_SECONDS,
    EvaluatorResponseError,
    OmniRemoteEvaluator,
    RemoteAdapter,
    RemoteAdapterTimeoutError,
    RunCancelledError,
)
from gepa.rpc.conversions import RemoteExample
from gepa.rpc.generated import gepa_pb2 as pb
from gepa.rpc.generated import gepa_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)

DEFAULT_REFLECTION_LM = "openai/gpt-5.1"
DEFAULT_RUNS_DIR = os.environ.get("GEPA_RPC_RUNS_DIR", "./runs")

# Caps how many RunOptimization/RunOptimizationOmni streams may be in flight
# at once. Each such stream pins one gRPC worker thread for its entire life
# (gRPC-python's sync streaming model requires this), so leaving it unbounded
# means enough long runs starve every other RPC -- GetStatus and Cancel
# included -- which is exactly the reviewer's max_workers=2 repro. server.py
# validates that max_workers leaves headroom above this for control-plane
# calls to always have a free thread.
DEFAULT_MAX_CONCURRENT_RUNS = 8

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


def _validate_run_id(run_id: str, runs_dir: str, context: grpc.ServicerContext) -> bool:
    """Return True if run_id is safe; otherwise set gRPC error and return False."""
    if not _RUN_ID_RE.match(run_id):
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("run_id must be 1-128 alphanumeric/hyphen/underscore characters")
        return False
    # Guard against path traversal even if the regex were somehow bypassed.
    runs_root = pathlib.Path(runs_dir).resolve()
    candidate = (runs_root / run_id).resolve()
    if not str(candidate).startswith(str(runs_root)):
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("invalid run_id")
        return False
    return True


class _ProgressCallback:
    """Bridges gepa callback events to ProgressUpdate messages on the stream."""

    def __init__(
        self,
        outbound: queue.Queue[pb.ServerMessage | None],
        max_metric_calls: int,
        run_status: dict[str, Any],
    ):
        self._outbound = outbound
        self._max_metric_calls = max_metric_calls
        self._run_status = run_status
        self._best_score = float("-inf")
        self._best_candidate: dict[str, str] = {}
        self._evals_used = 0

    def on_budget_updated(self, event: dict[str, Any]) -> None:
        used = int(event["metric_calls_used"])
        self._evals_used = used
        self._run_status["metric_calls_used"] = used
        if self._best_score != float("-inf"):
            self._emit(used)

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        avg = float(event["average_score"])
        if avg > self._best_score:
            self._best_score = avg
            self._best_candidate = dict(event["candidate"])
            self._emit(self._evals_used)

    def _emit(self, metric_calls_used: int) -> None:
        update = pb.ProgressUpdate(
            metric_calls_used=metric_calls_used,
            max_metric_calls=self._max_metric_calls,
            best_score=self._best_score if self._best_score != float("-inf") else 0.0,
            best_candidate=self._best_candidate,
        )
        self._outbound.put(pb.ServerMessage(progress_update=update))


class _OmniProgressCallback:
    """Bridges gepa callback events to OmniProgressUpdate messages on the stream."""

    def __init__(
        self,
        outbound: queue.Queue[pb.OmniServerMessage | None],
        max_evals: int,
        run_status: dict[str, Any],
    ):
        self._outbound = outbound
        self._max_evals = max_evals
        self._run_status = run_status
        self._best_score = float("-inf")
        self._best_candidate = ""
        self._evals_used = 0

    def on_budget_updated(self, event: dict[str, Any]) -> None:
        used = int(event["metric_calls_used"])
        self._evals_used = used
        self._run_status["metric_calls_used"] = used
        if self._best_score != float("-inf"):
            self._emit(used)

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        avg = float(event["average_score"])
        if avg > self._best_score:
            self._best_score = avg
            # Gepa internally wraps a string seed as {"current_candidate": "..."},
            # so unwrap single-key dicts back to their string value.
            candidate = event["candidate"]
            if isinstance(candidate, dict):
                self._best_candidate = next(iter(candidate.values()), "") if len(candidate) == 1 else str(candidate)
            else:
                self._best_candidate = str(candidate)
            self._emit(self._evals_used)

    def _emit(self, evals_used: int) -> None:
        update = pb.OmniProgressUpdate(
            evals_used=evals_used,
            max_evals=self._max_evals,
            best_score=self._best_score if self._best_score != float("-inf") else 0.0,
            best_candidate=self._best_candidate,
        )
        self._outbound.put(pb.OmniServerMessage(progress_update=update))


_MAX_RUNS = 1000

# Terminal: once a run reaches one of these its status never changes again.
# "cancelling" is deliberately NOT terminal -- cancellation has been requested
# but the optimizer may still be working, and the run ends up "cancelled",
# "complete" or "failed" depending on what actually happens.
_FINISHED_STATUSES = ("complete", "failed", "cancelled")


def _finish(run_status: dict[str, Any], status: str, message: str = "") -> bool:
    """Record a terminal status, refusing to overwrite an existing one.

    Without this guard the runner's success path could clobber a cancellation
    (and vice versa), which is exactly what produced an observable
    CANCELLED -> COMPLETE flip: Cancel marked the run cancelled, then the
    optimizer finished its current step and unconditionally set "complete".

    Returns True if this call set the status, False if the run was already
    terminal and nothing changed.
    """
    if run_status.get("status") in _FINISHED_STATUSES:
        return False
    run_status["status"] = status
    if message:
        run_status["message"] = message
    return True


def _sanitized_internal_error(run_id: str, exc: BaseException) -> str:
    """Message for an *unexpected* server-side failure.

    Forwarding str(exc) for these leaks whatever the exception happens to
    carry -- provider URLs, filesystem paths, and in the observed case an API
    key echoed by a connection error. Clients get the exception type (useful
    for deciding whether to retry, and low-risk since it's a class name) plus
    the run_id to correlate with the server log, where logger.exception has
    already recorded the full traceback.

    Errors the *client* caused and can act on -- a malformed evaluate
    response, a callback timeout -- are raised by our own code with messages
    we control, and are forwarded verbatim; they do not come through here.
    """
    return (
        f"internal server error ({type(exc).__name__}) in run {run_id}; "
        "see the server log for details"
    )


class GEPAServicer(pb_grpc.GEPAServiceServicer):
    def __init__(
        self,
        runs_dir: str = DEFAULT_RUNS_DIR,
        max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_RUNS,
        callback_timeout: float = DEFAULT_CALLBACK_TIMEOUT_SECONDS,
        allowed_engines: Collection[str] | None = None,
        allowed_reflection_lms: Collection[str] | None = None,
        default_reflection_lm: str | None = None,
    ):
        self._runs_dir = runs_dir
        self._runs: dict[str, dict[str, Any]] = {}
        self._runs_lock = threading.Lock()
        self._max_concurrent_runs = max_concurrent_runs
        self._callback_timeout = callback_timeout
        # None means no restriction (every engine / reflection_lm allowed),
        # preserving prior behavior. Non-None restricts to that explicit
        # allowlist: without it, any client reaching the port could select
        # engine="autoresearch"/"meta_harness" (which spawn a Claude Code
        # agent subprocess server-side) or point reflection_lm at an
        # arbitrary model, spending the operator's LLM budget.
        self._allowed_engines = frozenset(allowed_engines) if allowed_engines is not None else None
        self._allowed_reflection_lms = (
            frozenset(allowed_reflection_lms) if allowed_reflection_lms is not None else None
        )
        self._default_reflection_lm = default_reflection_lm or DEFAULT_REFLECTION_LM
        # Acquired non-blocking at the very top of each streaming RPC, before
        # any other work, so a burst of excess connection attempts is
        # rejected cheaply rather than queueing behind real work. Released
        # exactly once per admitted run, in that run's runner() finally
        # block, however the run ends.
        self._run_slots = threading.Semaphore(max_concurrent_runs)

    def _check_reflection_lm(self, reflection_lm: str, context: grpc.ServicerContext) -> bool:
        if self._allowed_reflection_lms is not None and reflection_lm not in self._allowed_reflection_lms:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(f"reflection_lm {reflection_lm!r} is not on this server's allowlist")
            return False
        return True

    def _check_engine(self, engine: str, context: grpc.ServicerContext) -> bool:
        if self._allowed_engines is not None and engine not in self._allowed_engines:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(f"engine {engine!r} is not on this server's allowlist")
            return False
        return True

    def _register_run(
        self,
        run_id: str,
        request_fingerprint: Any,
        run_status: dict[str, Any],
        context: grpc.ServicerContext,
    ) -> bool:
        with self._runs_lock:
            existing = self._runs.get(run_id)
            if existing is not None and existing.get("status") == "running":
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details(f"run {run_id!r} is already active")
                return False
            if existing is not None and existing.get("fingerprint") != request_fingerprint:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    f"run_id {run_id!r} was previously started with a different "
                    "trainset/valset/seed_candidate/budget; a resumed run must "
                    "re-send the exact same request"
                )
                return False
            if len(self._runs) >= _MAX_RUNS:
                # Only ever evict a finished run. Evicting a still-running
                # run's status record would make GetStatus wrongly report
                # UNKNOWN for a run that is actually still in progress.
                evict_key = next(
                    (k for k, e in self._runs.items() if e.get("status") in _FINISHED_STATUSES),
                    None,
                )
                if evict_key is None:
                    context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                    context.set_details(
                        f"server is tracking {_MAX_RUNS} runs, all still running; try again later"
                    )
                    return False
                del self._runs[evict_key]
            run_status["fingerprint"] = request_fingerprint
            self._runs[run_id] = run_status
            return True

    # ------------------------------------------------------------------ status
    def GetStatus(self, request: pb.StatusRequest, context: grpc.ServicerContext) -> pb.StatusResponse:
        with self._runs_lock:
            entry = self._runs.get(request.run_id)
            if entry is None:
                return pb.StatusResponse(run_id=request.run_id, status=pb.StatusResponse.UNKNOWN)
            status_str = entry.get("status", "")
            message = entry.get("message", "")
            metric_calls_used = entry.get("metric_calls_used", 0)
        status_map = {
            "running": pb.StatusResponse.RUNNING,
            "complete": pb.StatusResponse.COMPLETE,
            "failed": pb.StatusResponse.FAILED,
            "cancelled": pb.StatusResponse.CANCELLED,
            "cancelling": pb.StatusResponse.CANCELLING,
        }
        return pb.StatusResponse(
            run_id=request.run_id,
            status=status_map.get(status_str, pb.StatusResponse.UNKNOWN),
            message=message,
            metric_calls_used=metric_calls_used,
        )

    # ------------------------------------------------------------------ cancel
    def Cancel(self, request: pb.CancelRequest, context: grpc.ServicerContext) -> pb.CancelResponse:
        with self._runs_lock:
            entry = self._runs.get(request.run_id)
            if entry is None:
                return pb.CancelResponse(accepted=False, message=f"unknown run_id {request.run_id!r}")
            status_str = entry.get("status", "")
            if status_str == "cancelling":
                return pb.CancelResponse(
                    accepted=False,
                    message=f"cancellation was already requested for run {request.run_id!r}",
                )
            if status_str != "running":
                return pb.CancelResponse(
                    accepted=False,
                    message=f"run {request.run_id!r} is not running (status={status_str!r})",
                )
            cancel_fn = entry.get("cancel")
        # Invoke the run's own cancel hook outside self._runs_lock: it does
        # I/O (writing the gepa.stop sentinel) and takes the adapter's
        # separate lock, neither of which needs the servicer-wide lock held.
        if cancel_fn is None:
            return pb.CancelResponse(accepted=False, message="run has no cancel handler (internal error)")
        cancel_fn()
        # "requested", not "cancelled": the run is now CANCELLING and only
        # becomes CANCELLED once the optimizer actually stops. Poll GetStatus
        # for the terminal state.
        return pb.CancelResponse(accepted=True, message="cancellation requested")

    # ----------------------------------------------------------- optimization
    def RunOptimization(self, request_iterator, context: grpc.ServicerContext):
        if not self._run_slots.acquire(blocking=False):
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(
                f"server at capacity: {self._max_concurrent_runs} optimization runs already in progress"
            )
            return

        released = False

        def release_slot() -> None:
            nonlocal released
            if not released:
                released = True
                self._run_slots.release()

        try:
            first = next(request_iterator)
        except StopIteration:
            release_slot()
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("client closed stream before sending start_request")
            return

        if not first.HasField("start_request"):
            release_slot()
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("first ClientMessage must contain start_request")
            return

        start_req = first.start_request
        run_id = start_req.run_id or uuid.uuid4().hex
        if not _validate_run_id(run_id, self._runs_dir, context):
            release_slot()
            return
        run_dir = os.path.join(self._runs_dir, run_id)
        reflection_lm = start_req.reflection_lm or self._default_reflection_lm
        if not self._check_reflection_lm(reflection_lm, context):
            release_slot()
            return

        fingerprint = (
            dict(start_req.seed_candidate),
            tuple(e.id for e in start_req.trainset),
            tuple(e.id for e in start_req.valset),
            start_req.max_metric_calls,
        )
        run_status: dict[str, Any] = {
            "status": "running",
            "metric_calls_used": 0,
            "message": "",
        }
        if not self._register_run(run_id, fingerprint, run_status, context):
            release_slot()
            return

        outbound: queue.Queue[pb.ServerMessage | None] = queue.Queue()
        adapter = RemoteAdapter(outbound, callback_timeout=self._callback_timeout)

        cancel_requested = threading.Event()

        def do_cancel() -> None:
            # Idempotent: Cancel may be retried, or race the run finishing on
            # its own -- only the first call should do anything.
            if cancel_requested.is_set():
                return
            cancel_requested.set()
            # Non-terminal: the optimizer has only been *asked* to stop and
            # may still be mid-step. The runner records the terminal
            # "cancelled" when it actually unwinds -- or the run may still
            # legitimately finish as complete/failed if it got there first.
            if run_status.get("status") == "running":
                run_status["status"] = "cancelling"
            # Effect 1: unblock whatever this run's adapter is waiting on
            # right now. This alone does NOT stop gepa.optimize from issuing
            # more work -- that's effect 2 below.
            adapter.cancel(RunCancelledError(f"run {run_id!r} cancelled by client request"))
            # Effect 2: ask the optimizer to actually stop. gepa_launcher
            # auto-registers a FileStopper(os.path.join(run_dir, "gepa.stop"))
            # stop-callback whenever run_dir is set (always, here), so writing
            # that file makes gepa.optimize exit at its next checkpoint --
            # reusing an existing cooperative-stop primitive rather than
            # inventing a parallel mechanism.
            try:
                os.makedirs(run_dir, exist_ok=True)
                pathlib.Path(run_dir, "gepa.stop").touch()
            except OSError:
                logger.warning("failed to write gepa.stop for run %s", run_id, exc_info=True)

        run_status["cancel"] = do_cancel

        def reader() -> None:
            try:
                for msg in request_iterator:
                    if msg.HasField("evaluate_batch_response"):
                        adapter.deliver_evaluate_response(msg.evaluate_batch_response)
                    elif msg.HasField("reflective_dataset_response"):
                        adapter.deliver_reflective_response(msg.reflective_dataset_response)
                    elif msg.HasField("start_request"):
                        logger.warning("ignoring extra start_request after run started")
            except Exception as e:
                logger.info("client stream closed: %s", e)
                adapter.cancel(e)
            else:
                adapter.cancel()

        def runner() -> None:
            try:
                trainset = [RemoteExample.from_proto(e) for e in start_req.trainset]
                valset_proto = list(start_req.valset)
                valset = [RemoteExample.from_proto(e) for e in valset_proto] if valset_proto else None

                max_metric_calls = start_req.max_metric_calls or None
                callback = _ProgressCallback(outbound, start_req.max_metric_calls, run_status)

                os.makedirs(run_dir, exist_ok=True)
                result = gepa.optimize(  # type: ignore[attr-defined]
                    seed_candidate=dict(start_req.seed_candidate),
                    trainset=trainset,
                    valset=valset,
                    adapter=adapter,
                    reflection_lm=reflection_lm,
                    max_metric_calls=max_metric_calls,
                    run_dir=run_dir,
                    callbacks=[callback],  # type: ignore[arg-type]
                    raise_on_exception=True,
                )

                best_idx = result.best_idx
                best_candidate = result.candidates[best_idx]
                best_score = result.val_aggregate_scores[best_idx]
                # Guarded: if a Cancel already drove this run terminal, don't
                # clobber it (the old code's unconditional write is what made
                # status flip CANCELLED -> COMPLETE).
                _finish(run_status, "complete")
                outbound.put(
                    pb.ServerMessage(
                        optimization_complete=pb.OptimizationComplete(
                            run_id=run_id,
                            best_candidate=dict(best_candidate),
                            best_score=float(best_score),
                        )
                    )
                )
            except RunCancelledError as exc:
                logger.info("optimization run %s cancelled: %s", run_id, exc)
                # Only now is the run really stopped, so record the terminal
                # state here rather than when cancellation was requested.
                _finish(run_status, "cancelled", str(exc))
                outbound.put(
                    pb.ServerMessage(
                        optimization_error=pb.OptimizationError(
                            run_id=run_id,
                            message=str(exc),
                            error_kind=pb.OptimizationError.CANCELLED,
                        )
                    )
                )
            except (EvaluatorResponseError, RemoteAdapterTimeoutError) as exc:
                # Raised by our own adapter code with messages we control, and
                # caused by the client -- safe and useful to forward verbatim.
                logger.exception("optimization run %s failed (client-side fault)", run_id)
                message = f"{type(exc).__name__}: {exc}"
                _finish(run_status, "failed", message)
                outbound.put(
                    pb.ServerMessage(
                        optimization_error=pb.OptimizationError(
                            run_id=run_id,
                            message=message,
                            error_kind=pb.OptimizationError.EVALUATOR_ERROR,
                        )
                    )
                )
            except Exception as exc:
                # Unexpected: sanitize. str(exc) on an arbitrary internal
                # failure can carry provider URLs, filesystem paths, even an
                # API key echoed back by a connection error. The traceback is
                # already in the server log via logger.exception.
                logger.exception("optimization run %s failed", run_id)
                message = _sanitized_internal_error(run_id, exc)
                _finish(run_status, "failed", message)
                outbound.put(
                    pb.ServerMessage(
                        optimization_error=pb.OptimizationError(
                            run_id=run_id,
                            message=message,
                            error_kind=pb.OptimizationError.INTERNAL_ERROR,
                        )
                    )
                )
            finally:
                outbound.put(None)
                adapter.cancel()
                release_slot()

        threading.Thread(target=reader, name=f"gepa-rpc-reader-{run_id}", daemon=True).start()
        threading.Thread(target=runner, name=f"gepa-rpc-runner-{run_id}", daemon=True).start()

        while True:
            msg = outbound.get()
            if msg is None:
                return
            yield msg

    # ------------------------------------------------------- omni optimization
    def RunOptimizationOmni(self, request_iterator, context: grpc.ServicerContext):
        if not self._run_slots.acquire(blocking=False):
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(
                f"server at capacity: {self._max_concurrent_runs} optimization runs already in progress"
            )
            return

        released = False

        def release_slot() -> None:
            nonlocal released
            if not released:
                released = True
                self._run_slots.release()

        try:
            first = next(request_iterator)
        except StopIteration:
            release_slot()
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("client closed stream before sending start_request")
            return

        if not first.HasField("start_request"):
            release_slot()
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("first OmniClientMessage must contain start_request")
            return

        start_req = first.start_request
        run_id = start_req.run_id or uuid.uuid4().hex
        if not _validate_run_id(run_id, self._runs_dir, context):
            release_slot()
            return
        run_dir = os.path.join(self._runs_dir, run_id)
        engine = start_req.engine or "gepa"
        if not self._check_engine(engine, context):
            release_slot()
            return
        reflection_lm = start_req.reflection_lm or self._default_reflection_lm
        if not self._check_reflection_lm(reflection_lm, context):
            release_slot()
            return

        fingerprint = (
            start_req.seed_candidate,
            tuple(e.id for e in start_req.dataset),
            tuple(e.id for e in start_req.valset),
            start_req.max_evals,
            engine,
        )
        run_status: dict[str, Any] = {"status": "running", "metric_calls_used": 0, "message": ""}
        if not self._register_run(run_id, fingerprint, run_status, context):
            release_slot()
            return

        outbound: queue.Queue[pb.OmniServerMessage | None] = queue.Queue()
        # For engine="gepa", _OmniProgressCallback below already tracks
        # metric_calls_used from gepa's own ledger, so handing run_status to
        # the evaluator too would double-count. Every other engine has no
        # such callback, which is why they always reported 0 -- give the
        # evaluator the reference only when nothing else is tracking it.
        evaluator = OmniRemoteEvaluator(
            outbound,
            callback_timeout=self._callback_timeout,
            run_status=None if engine == "gepa" else run_status,
        )

        cancel_requested = threading.Event()

        def do_cancel() -> None:
            if cancel_requested.is_set():
                return
            cancel_requested.set()
            # See RunOptimization.do_cancel: non-terminal on purpose.
            if run_status.get("status") == "running":
                run_status["status"] = "cancelling"
            # Effect 1 (see RunOptimization.do_cancel for the full
            # explanation): unblock whatever the evaluator is waiting on.
            evaluator.cancel(RunCancelledError(f"run {run_id!r} cancelled by client request"))
            # Effect 2 is only wired for the gepa engine, which runs through
            # gepa_launcher's FileStopper convention. autoresearch/best_of_n/
            # meta_harness run as their own subprocesses with no equivalent
            # hook exposed to this servicer today, so cancelling those still
            # unblocks effect 1 but does NOT stop the subprocess. Flagged
            # rather than silently claimed: wiring real subprocess
            # termination needs a hook into gepa.oa.engines.* that doesn't
            # exist yet, which is outside this PR's scope (it adds the rpc
            # layer on top of gepa, not changes to gepa's own engines).
            if engine == "gepa":
                try:
                    os.makedirs(run_dir, exist_ok=True)
                    pathlib.Path(run_dir, "gepa.stop").touch()
                except OSError:
                    logger.warning("failed to write gepa.stop for omni run %s", run_id, exc_info=True)
            else:
                logger.warning(
                    "run %s cancelled, but engine %r has no server-side stop hook -- "
                    "its subprocess (if any) keeps running until it exits on its own",
                    run_id, engine,
                )

        run_status["cancel"] = do_cancel

        def reader() -> None:
            try:
                for msg in request_iterator:
                    if msg.HasField("evaluate_batch_response"):
                        evaluator.deliver_response(msg.evaluate_batch_response)
                    elif msg.HasField("start_request"):
                        logger.warning("ignoring extra start_request after omni run started")
            except Exception as e:
                logger.info("omni client stream closed: %s", e)
                evaluator.cancel(e)
            else:
                evaluator.cancel()

        def runner() -> None:
            try:
                dataset = [
                    {"id": e.id, "fields": dict(e.fields)}
                    for e in start_req.dataset
                ]
                valset = [
                    {"id": e.id, "fields": dict(e.fields)}
                    for e in start_req.valset
                ] or None

                max_evals = start_req.max_evals or None

                # Only the gepa engine's Config accepts {reflection, callbacks};
                # autoresearch/best_of_n/meta_harness each have their own
                # engine_config dataclass with unrelated fields, so passing this
                # shape to them raises TypeError. Non-gepa runs fall back to
                # each engine's own defaults and, for now, don't stream
                # OmniProgressUpdate -- only the final OmniOptimizationComplete.
                engine_config: dict[str, Any] = {}
                if engine == "gepa":
                    callback = _OmniProgressCallback(outbound, start_req.max_evals, run_status)
                    engine_config = {
                        "reflection": {
                            "reflection_lm": reflection_lm,
                        },
                        "callbacks": [callback],
                    }

                os.makedirs(run_dir, exist_ok=True)
                result = optimize_anything(
                    seed_candidate=start_req.seed_candidate or None,
                    batch_evaluator=evaluator,
                    dataset=dataset or None,
                    valset=valset,
                    objective=start_req.objective or None,
                    background=start_req.background or None,
                    config=OptimizeAnythingConfig(
                        engine=engine,
                        max_evals=max_evals,
                        run_dir=run_dir,
                        # OptimizeAnythingConfig has two distinct directory
                        # fields: run_dir (the engine's workspace, set above)
                        # and output_dir (where the eval server writes
                        # per-eval JSON, progress_log.jsonl and summary.json,
                        # defaulting to a timestamped path under the server's
                        # CWD when left None). Only run_dir was ever passed,
                        # so Omni runs wrote eval output under the server's
                        # CWD regardless of --runs-dir. Route it into this
                        # run's own directory, under a subdirectory so it
                        # can't collide with what gepa's engine writes
                        # directly into run_dir.
                        output_dir=os.path.join(run_dir, "eval_output"),
                        # sandbox left at gepa's default (True): autoresearch/meta_harness
                        # need it to jail their Claude Code agent subprocess.
                        engine_config=engine_config,
                    ),
                )

                _finish(run_status, "complete")
                raw_best = result.best_candidate
                if isinstance(raw_best, dict):
                    best_str = next(iter(raw_best.values()), "") if len(raw_best) == 1 else str(raw_best)
                else:
                    best_str = raw_best
                outbound.put(pb.OmniServerMessage(
                    optimization_complete=pb.OmniOptimizationComplete(
                        run_id=run_id,
                        best_candidate=best_str,
                        best_score=float(result.best_score),
                        total_evals=result.total_evals,
                    )
                ))
            except RunCancelledError as exc:
                logger.info("omni optimization run %s cancelled: %s", run_id, exc)
                _finish(run_status, "cancelled", str(exc))
                outbound.put(pb.OmniServerMessage(
                    optimization_error=pb.OptimizationError(
                        run_id=run_id, message=str(exc),
                        error_kind=pb.OptimizationError.CANCELLED,
                    )
                ))
            except (EvaluatorResponseError, RemoteAdapterTimeoutError) as exc:
                logger.exception("omni optimization run %s failed (client-side fault)", run_id)
                message = f"{type(exc).__name__}: {exc}"
                _finish(run_status, "failed", message)
                outbound.put(pb.OmniServerMessage(
                    optimization_error=pb.OptimizationError(
                        run_id=run_id, message=message,
                        error_kind=pb.OptimizationError.EVALUATOR_ERROR,
                    )
                ))
            except Exception as exc:
                # See RunOptimization's handler: sanitized on purpose.
                logger.exception("omni optimization run %s failed", run_id)
                message = _sanitized_internal_error(run_id, exc)
                _finish(run_status, "failed", message)
                outbound.put(pb.OmniServerMessage(
                    optimization_error=pb.OptimizationError(
                        run_id=run_id, message=message,
                        error_kind=pb.OptimizationError.INTERNAL_ERROR,
                    )
                ))
            finally:
                outbound.put(None)
                evaluator.cancel()
                release_slot()

        threading.Thread(target=reader, name=f"gepa-rpc-omni-reader-{run_id}", daemon=True).start()
        threading.Thread(target=runner, name=f"gepa-rpc-omni-runner-{run_id}", daemon=True).start()

        while True:
            msg = outbound.get()
            if msg is None:
                return
            yield msg
