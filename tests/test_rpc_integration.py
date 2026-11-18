"""Integration tests for GEPAServicer.

These tests spin up a real gRPC server and drive it with a Python stub client.
The gepa optimizer and reflection LM are monkey-patched so no real LLM calls
are made — the tests verify the gRPC wire protocol and adapter pipeline only.
"""

from __future__ import annotations

import contextlib
import json
import queue
import socket
from concurrent import futures
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# The rpc extra (grpcio) isn't part of gepa's default dev/test dependency
# closure, so skip this whole module rather than error out when it's absent
# (CI runs an rpc-specific job separately, with the extra installed).
grpc = pytest.importorskip("grpc")

from gepa.rpc.generated import gepa_pb2 as pb  # noqa: E402
from gepa.rpc.generated import gepa_pb2_grpc as pb_grpc  # noqa: E402
from gepa.rpc.servicer import GEPAServicer  # noqa: E402

# ------------------------------------------------------------------ fixtures


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def stub(tmp_path):
    """Start a GEPAServicer on a random port; yield a connected stub."""
    port = _free_port()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_GEPAServiceServicer_to_server(
        GEPAServicer(runs_dir=str(tmp_path / "runs")), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    yield pb_grpc.GEPAServiceStub(channel)

    channel.close()
    server.stop(grace=0)


@contextlib.contextmanager
def _custom_stub(
    tmp_path, *, max_workers=4, max_concurrent_runs=8, callback_timeout=300.0,
    handshake_timeout=30.0,
    allowed_engines=None, allowed_reflection_lms=None, default_reflection_lm=None,
):
    """Like the `stub` fixture, but exposing GEPAServicer's concurrency and
    timeout knobs. A plain contextmanager, not a fixture, since different
    tests need different combinations."""
    port = _free_port()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_GEPAServiceServicer_to_server(
        GEPAServicer(
            runs_dir=str(tmp_path / "runs"),
            max_concurrent_runs=max_concurrent_runs,
            callback_timeout=callback_timeout,
            handshake_timeout=handshake_timeout,
            allowed_engines=allowed_engines,
            allowed_reflection_lms=allowed_reflection_lms,
            default_reflection_lm=default_reflection_lm,
        ),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    try:
        yield pb_grpc.GEPAServiceStub(channel)
    finally:
        channel.close()
        server.stop(grace=0)


# ------------------------------------------------------------------ helpers


_TRAINSET = [
    pb.Example(id="1", fields={"input": "2+2", "answer": "4"}),
    pb.Example(id="2", fields={"input": "3+3", "answer": "6"}),
]

_SEED = {"instructions": "Answer the question."}


def _run_optimize(stub, *, run_id: str, seed=None, trainset=None, max_metric_calls=20, reflection_lm="fake"):
    """Drive a full RunOptimization round-trip with a mock optimizer.

    The test client automatically responds to EvaluateBatchRequest messages
    with score=1.0 for every example.

    Does NOT patch gepa.optimize itself -- the caller is responsible for that
    (via `_run_optimize_patched` for single-threaded use, or its own patch
    applied once outside any thread it spawns). This keeps "what's patched"
    and "who's reading the stream" decoupled, which matters for concurrent
    callers: `unittest.mock.patch` context managers are not reentrant or
    thread-safe, so having this helper apply its own patch meant every
    concurrent caller raced to install and restore the same patch on the
    same shared object.
    """
    seed = seed if seed is not None else _SEED
    trainset = trainset if trainset is not None else _TRAINSET

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(
        start_request=pb.StartRequest(
            run_id=run_id,
            seed_candidate=seed,
            trainset=trainset,
            max_metric_calls=max_metric_calls,
            reflection_lm=reflection_lm,
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    final = None
    call = stub.RunOptimization(gen())
    for msg in call:
        if msg.HasField("evaluate_batch_request"):
            req = msg.evaluate_batch_request
            req_q.put(pb.ClientMessage(
                evaluate_batch_response=pb.EvaluateBatchResponse(
                    request_id=req.request_id,
                    outputs=["ok"] * len(req.batch),
                    scores=[1.0] * len(req.batch),
                )
            ))
        elif msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
            final = msg
            req_q.put(None)
            break

    return final


def _run_optimize_patched(stub, *, fake_optimize, **kwargs):
    """`_run_optimize`, patching gepa.optimize for the duration of the call.

    For single-threaded call sites only -- see `_run_optimize`'s docstring
    for why concurrent callers must patch once, outside their threads,
    instead of using this wrapper.
    """
    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        return _run_optimize(stub, **kwargs)


def _run_optimize_omni(stub, *, run_id: str, fake_optimize_anything, dataset=None, engine=None):
    """Drive a full RunOptimizationOmni round-trip with a mock optimizer."""
    if dataset is None:
        dataset = [
            pb.Example(id="1", fields={"x": "hello"}),
            pb.Example(id="2", fields={"x": "world"}),
        ]

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.OmniClientMessage(
        start_request=pb.OmniStartRequest(
            run_id=run_id,
            seed_candidate="Classify the input.",
            dataset=dataset,
            max_evals=10,
            reflection_lm="fake",
            engine=engine or "",
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    final = None
    with patch("gepa.rpc.servicer.optimize_anything", side_effect=fake_optimize_anything):
        call = stub.RunOptimizationOmni(gen())
        for msg in call:
            if msg.HasField("evaluate_batch_request"):
                req = msg.evaluate_batch_request
                req_q.put(pb.OmniClientMessage(
                    evaluate_batch_response=pb.OmniEvaluateBatchResponse(
                        request_id=req.request_id,
                        scores=[0.8] * len(req.batch),
                        side_infos=["{}"] * len(req.batch),
                    )
                ))
            elif msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                final = msg
                req_q.put(None)
                break

    return final


# ------------------------------------------------------------------ tests: RunOptimization


def test_optimize_returns_complete(stub):
    """Happy path: mock optimizer returns a result → OptimizationComplete."""
    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(
            candidates=[dict(seed_candidate)],
            best_idx=0,
            val_aggregate_scores=[0.75],
        )

    msg = _run_optimize_patched(stub, run_id="opt-happy", fake_optimize=fake_optimize)

    assert msg is not None
    assert msg.HasField("optimization_complete")
    c = msg.optimization_complete
    assert c.run_id == "opt-happy"
    assert c.best_score == pytest.approx(0.75)
    assert dict(c.best_candidate) == _SEED


def test_optimize_evaluate_proxied(stub):
    """adapter.evaluate is called → EvaluateBatchRequest flows to client → scores return."""
    received_batches: list = []

    def fake_optimize(*, seed_candidate, trainset, adapter, **_):
        batch = list(trainset)
        result = adapter.evaluate(batch, dict(seed_candidate), capture_traces=False)
        received_batches.append((batch, result.scores))
        return SimpleNamespace(
            candidates=[dict(seed_candidate)],
            best_idx=0,
            val_aggregate_scores=[sum(result.scores) / len(result.scores)],
        )

    _run_optimize_patched(stub, run_id="opt-proxy", fake_optimize=fake_optimize)

    assert len(received_batches) == 1
    batch, scores = received_batches[0]
    assert len(batch) == len(_TRAINSET)
    assert scores == [1.0] * len(_TRAINSET)


def test_optimize_unexpected_error_is_sanitized(stub):
    """An *unexpected* server-side exception must not forward str(exc).

    str(exc) on an arbitrary internal failure can carry provider URLs,
    filesystem paths, even an API key echoed back by a connection error. The
    client gets the exception type (useful, low-risk) plus the run_id to
    correlate against the server log, which holds the full traceback.
    Contrast test_optimize_client_fault_error_is_forwarded, where the client
    caused the failure and the detail is safe and actionable.
    """
    secret = "sk-proj-SUPERSECRET123456"

    def fake_optimize(**_):
        raise ConnectionError(
            f"HTTPSConnectionPool(host='api.openai.com', port=443): "
            f"auth failure for key {secret} /Users/operator/secrets/creds.json"
        )

    msg = _run_optimize_patched(stub, run_id="opt-err", fake_optimize=fake_optimize)

    assert msg is not None
    assert msg.HasField("optimization_error")
    body = msg.optimization_error.message
    assert secret not in body
    assert "/Users/operator" not in body
    assert "api.openai.com" not in body
    assert "ConnectionError" in body and "opt-err" in body
    assert msg.optimization_error.error_kind == pb.OptimizationError.INTERNAL_ERROR
    # GetStatus must not become a side channel for the same payload.
    status_msg = stub.GetStatus(pb.StatusRequest(run_id="opt-err")).message
    assert secret not in status_msg and "/Users/operator" not in status_msg


def test_optimize_client_fault_error_is_forwarded(stub):
    """A fault the *client* caused is still forwarded in full detail -- the
    actionable-errors behavior, which sanitizing internals must not regress.
    These messages originate in our own adapter code, not arbitrary
    third-party exception text."""
    def fake_optimize(*, seed_candidate, trainset, adapter, **_):
        adapter.evaluate(list(trainset), dict(seed_candidate), capture_traces=False)
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.0])

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(start_request=pb.StartRequest(
        run_id="client-fault", seed_candidate=_SEED, trainset=_TRAINSET, max_metric_calls=5)))

    def gen():
        while True:
            m = req_q.get()
            if m is None:
                return
            yield m

    final = None
    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        for msg in stub.RunOptimization(gen()):
            if msg.HasField("evaluate_batch_request"):
                req = msg.evaluate_batch_request
                # 1 score for a 2-example batch -> EvaluatorResponseError.
                req_q.put(pb.ClientMessage(evaluate_batch_response=pb.EvaluateBatchResponse(
                    request_id=req.request_id, outputs=["ok"], scores=[1.0])))
            elif msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                final = msg
                req_q.put(None)
                break

    assert final is not None
    assert final.HasField("optimization_error")
    assert final.optimization_error.message.startswith("EvaluatorResponseError: ")
    assert final.optimization_error.error_kind == pb.OptimizationError.EVALUATOR_ERROR


def test_get_status_complete(stub):
    """GetStatus returns COMPLETE after a successful RunOptimization."""
    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(
            candidates=[dict(seed_candidate)],
            best_idx=0,
            val_aggregate_scores=[0.5],
        )

    run_id = "status-test"
    _run_optimize_patched(stub, run_id=run_id, fake_optimize=fake_optimize)

    resp = stub.GetStatus(pb.StatusRequest(run_id=run_id))
    assert resp.status == pb.StatusResponse.COMPLETE


def test_get_status_unknown(stub):
    """GetStatus for an unknown run_id returns UNKNOWN."""
    resp = stub.GetStatus(pb.StatusRequest(run_id="no-such-run"))
    assert resp.status == pb.StatusResponse.UNKNOWN


# ------------------------------------------------------------------ tests: RunOptimizationOmni


def test_optimize_omni_returns_complete(stub):
    """Happy path: Omni mock returns a result → OmniOptimizationComplete."""
    from gepa.optimize_anything import Result

    def fake_optimize_anything(*, seed_candidate, **_):
        return Result(best_candidate=seed_candidate or "", best_score=0.9, total_evals=4)

    msg = _run_optimize_omni(stub, run_id="omni-happy", fake_optimize_anything=fake_optimize_anything)

    assert msg is not None
    assert msg.HasField("optimization_complete")
    c = msg.optimization_complete
    assert c.run_id == "omni-happy"
    assert c.best_score == pytest.approx(0.9)
    assert c.total_evals == 4


def test_optimize_omni_evaluator_proxied(stub):
    """batch_evaluator is called → OmniEvaluateBatchRequest flows to client → scores return."""
    from gepa.optimize_anything import Result

    received: list = []

    def fake_optimize_anything(*, seed_candidate, batch_evaluator, dataset, **_):
        pairs = [(seed_candidate, ex) for ex in (dataset or [])[:2]]
        results = batch_evaluator(pairs)
        received.append(results)
        best_score = max(r[0] for r in results) if results else 0.0
        return Result(best_candidate=seed_candidate or "", best_score=best_score, total_evals=len(pairs))

    _run_optimize_omni(stub, run_id="omni-proxy", fake_optimize_anything=fake_optimize_anything)

    assert len(received) == 1
    scores = [r[0] for r in received[0]]
    # proto float is 32-bit; compare with tolerance
    assert scores == pytest.approx([0.8, 0.8], rel=1e-5)


def test_optimize_omni_default_engine_uses_gepa_config(stub):
    """No engine set → config.engine == 'gepa' and reflection/callbacks are wired."""
    from gepa.optimize_anything import Result

    received_configs: list = []

    def fake_optimize_anything(*, seed_candidate, config, **_):
        received_configs.append(config)
        return Result(best_candidate=seed_candidate or "", best_score=1.0, total_evals=1)

    _run_optimize_omni(stub, run_id="omni-engine-default", fake_optimize_anything=fake_optimize_anything)

    assert len(received_configs) == 1
    config = received_configs[0]
    assert config.engine == "gepa"
    assert "reflection" in config.engine_config
    assert "callbacks" in config.engine_config


def test_optimize_omni_explicit_engine_skips_gepa_config(stub):
    """A non-gepa engine gets an empty engine_config (avoids TypeError from mismatched fields)."""
    from gepa.optimize_anything import Result

    received_configs: list = []

    def fake_optimize_anything(*, seed_candidate, config, **_):
        received_configs.append(config)
        return Result(best_candidate=seed_candidate or "", best_score=1.0, total_evals=1)

    _run_optimize_omni(
        stub, run_id="omni-engine-best-of-n", fake_optimize_anything=fake_optimize_anything, engine="best_of_n"
    )

    assert len(received_configs) == 1
    config = received_configs[0]
    assert config.engine == "best_of_n"
    assert config.engine_config == {}


def test_optimize_omni_opt_states_proxied(stub):
    """opt_states passed to batch_evaluator are forwarded on OmniEvaluateBatchRequest."""
    from gepa.optimize_anything import Result

    dataset = [
        pb.Example(id="1", fields={"x": "hello"}),
        pb.Example(id="2", fields={"x": "world"}),
    ]
    opt_states = [
        SimpleNamespace(best_example_evals=[{"score": 0.9, "side_info": {"note": "prior best 1"}}]),
        SimpleNamespace(best_example_evals=[]),
    ]

    def fake_optimize_anything(*, seed_candidate, batch_evaluator, dataset, **_):
        pairs = [(seed_candidate, ex) for ex in dataset[:2]]
        results = batch_evaluator(pairs, opt_states=opt_states)
        best_score = max(r[0] for r in results) if results else 0.0
        return Result(best_candidate=seed_candidate or "", best_score=best_score, total_evals=len(pairs))

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.OmniClientMessage(
        start_request=pb.OmniStartRequest(
            run_id="omni-opt-states",
            seed_candidate="Classify the input.",
            dataset=dataset,
            max_evals=10,
            reflection_lm="fake",
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    received_requests: list = []
    with patch("gepa.rpc.servicer.optimize_anything", side_effect=fake_optimize_anything):
        call = stub.RunOptimizationOmni(gen())
        for msg in call:
            if msg.HasField("evaluate_batch_request"):
                req = msg.evaluate_batch_request
                received_requests.append(req)
                req_q.put(pb.OmniClientMessage(
                    evaluate_batch_response=pb.OmniEvaluateBatchResponse(
                        request_id=req.request_id,
                        scores=[0.5] * len(req.batch),
                        side_infos=["{}"] * len(req.batch),
                    )
                ))
            elif msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                req_q.put(None)
                break

    assert len(received_requests) == 1
    req = received_requests[0]
    assert len(req.opt_states) == 2
    assert len(req.opt_states[0].best_example_evals) == 1
    assert req.opt_states[0].best_example_evals[0].score == pytest.approx(0.9)
    assert json.loads(req.opt_states[0].best_example_evals[0].side_info) == {"note": "prior best 1"}
    assert len(req.opt_states[1].best_example_evals) == 0


def test_optimize_omni_opt_states_omitted_backward_compat(stub):
    """When batch_evaluator is called without opt_states, the request field stays empty."""
    from gepa.optimize_anything import Result

    def fake_optimize_anything(*, seed_candidate, batch_evaluator, dataset, **_):
        pairs = [(seed_candidate, ex) for ex in (dataset or [])[:1]]
        results = batch_evaluator(pairs)
        best_score = max(r[0] for r in results) if results else 0.0
        return Result(best_candidate=seed_candidate or "", best_score=best_score, total_evals=len(pairs))

    received_requests: list = []
    dataset = [pb.Example(id="1", fields={"x": "hello"})]

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.OmniClientMessage(
        start_request=pb.OmniStartRequest(
            run_id="omni-opt-states-none",
            seed_candidate="Classify the input.",
            dataset=dataset,
            max_evals=10,
            reflection_lm="fake",
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    with patch("gepa.rpc.servicer.optimize_anything", side_effect=fake_optimize_anything):
        call = stub.RunOptimizationOmni(gen())
        for msg in call:
            if msg.HasField("evaluate_batch_request"):
                req = msg.evaluate_batch_request
                received_requests.append(req)
                req_q.put(pb.OmniClientMessage(
                    evaluate_batch_response=pb.OmniEvaluateBatchResponse(
                        request_id=req.request_id,
                        scores=[0.5] * len(req.batch),
                        side_infos=["{}"] * len(req.batch),
                    )
                ))
            elif msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                req_q.put(None)
                break

    assert len(received_requests) == 1
    assert len(received_requests[0].opt_states) == 0


def test_optimize_omni_error_propagated(stub):
    """An exception in the Omni optimizer thread becomes an actionable OptimizationError."""
    def fake_optimize_anything(**_):
        raise ValueError("omni boom")

    msg = _run_optimize_omni(stub, run_id="omni-err", fake_optimize_anything=fake_optimize_anything)

    assert msg is not None
    assert msg.HasField("optimization_error")
    assert "omni boom" not in msg.optimization_error.message
    assert "ValueError" in msg.optimization_error.message
    assert msg.optimization_error.error_kind == pb.OptimizationError.INTERNAL_ERROR


# ------------------------------------------------------------------ edge case tests


def test_invalid_run_id_rejected(stub):
    """run_id with path traversal characters is rejected with INVALID_ARGUMENT."""
    import grpc as grpc_module

    def fake_optimize(**_):
        return SimpleNamespace(candidates=[{}], best_idx=0, val_aggregate_scores=[0.0])

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(
        start_request=pb.StartRequest(
            run_id="../../../etc/passwd",
            seed_candidate=_SEED,
            trainset=_TRAINSET,
            max_metric_calls=5,
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        call = stub.RunOptimization(gen())
        try:
            list(call)
            req_q.put(None)
            pytest.fail("expected RpcError")
        except grpc_module.RpcError as e:
            req_q.put(None)
            assert e.code() == grpc_module.StatusCode.INVALID_ARGUMENT


def test_optimize_empty_trainset(stub):
    """RunOptimization with an empty trainset completes without error."""
    def fake_optimize(**_):
        return SimpleNamespace(candidates=[{"instructions": "x"}], best_idx=0, val_aggregate_scores=[0.0])

    msg = _run_optimize_patched(stub, run_id="opt-empty", fake_optimize=fake_optimize, trainset=[])
    assert msg is not None
    assert msg.HasField("optimization_complete")


def test_optimize_omni_no_seed(stub):
    """RunOptimizationOmni with no seed_candidate (empty string) completes."""
    from gepa.optimize_anything import Result

    def fake_optimize_anything(*, seed_candidate, **_):
        return Result(best_candidate="generated", best_score=0.5, total_evals=2)

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.OmniClientMessage(
        start_request=pb.OmniStartRequest(
            run_id="omni-noseed",
            seed_candidate="",
            max_evals=5,
            reflection_lm="fake",
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    final = None
    with patch("gepa.rpc.servicer.optimize_anything", side_effect=fake_optimize_anything):
        call = stub.RunOptimizationOmni(gen())
        for msg in call:
            if msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                final = msg
                req_q.put(None)
                break

    assert final is not None
    assert final.HasField("optimization_complete")
    assert final.optimization_complete.best_candidate == "generated"


def test_optimize_omni_invalid_run_id_rejected(stub):
    """Omni run_id with path traversal is rejected."""
    import grpc as grpc_module

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.OmniClientMessage(
        start_request=pb.OmniStartRequest(
            run_id="../../bad",
            seed_candidate="test",
            max_evals=5,
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    def fake(**_):
        from gepa.optimize_anything import Result
        return Result(best_candidate="x", best_score=1.0, total_evals=1)

    with patch("gepa.rpc.servicer.optimize_anything", side_effect=fake):
        call = stub.RunOptimizationOmni(gen())
        try:
            list(call)
            req_q.put(None)
            pytest.fail("expected RpcError")
        except grpc_module.RpcError as e:
            req_q.put(None)
            assert e.code() == grpc_module.StatusCode.INVALID_ARGUMENT


def test_get_status_failed(stub):
    """GetStatus for a failed run returns FAILED with the real error forwarded."""
    def fake_optimize(**_):
        raise RuntimeError("internal detail that should not escape")

    run_id = "status-fail"
    _run_optimize_patched(stub, run_id=run_id, fake_optimize=fake_optimize)

    resp = stub.GetStatus(pb.StatusRequest(run_id=run_id))
    assert resp.status == pb.StatusResponse.FAILED
    assert "internal detail that should not escape" not in resp.message
    assert "RuntimeError" in resp.message


def test_duplicate_run_id_rejected(stub):
    """Submitting the same run_id while a run is active returns ALREADY_EXISTS."""
    import threading

    import grpc as grpc_module

    started = threading.Event()
    released = threading.Event()

    def fake_optimize(**_):
        started.set()
        released.wait(timeout=5)
        return SimpleNamespace(candidates=[{}], best_idx=0, val_aggregate_scores=[0.0])

    # Start first run in background.
    first_result: list = []

    def run_first():
        msg = _run_optimize_patched(stub, run_id="dup-run", fake_optimize=fake_optimize)
        first_result.append(msg)

    t = threading.Thread(target=run_first)
    t.start()
    started.wait(timeout=5)

    # Second run with the same run_id while first is still active.
    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(
        start_request=pb.StartRequest(
            run_id="dup-run",
            seed_candidate=_SEED,
            trainset=_TRAINSET,
            max_metric_calls=5,
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        call = stub.RunOptimization(gen())
        try:
            list(call)
            req_q.put(None)
            pytest.fail("expected RpcError")
        except grpc_module.RpcError as e:
            req_q.put(None)
            assert e.code() == grpc_module.StatusCode.ALREADY_EXISTS

    released.set()
    t.join(timeout=5)


def test_optimize_omni_dict_best_candidate(stub):
    """When optimize_anything returns a dict best_candidate it is unwrapped to str."""
    class _FakeResult:
        @property
        def best_candidate(self):
            return {"__seed__": "optimized prompt text"}

        @property
        def best_score(self):
            return 0.7

        @property
        def total_evals(self):
            return 3

    def fake_optimize_anything(**_):
        return _FakeResult()

    msg = _run_optimize_omni(stub, run_id="omni-dict", fake_optimize_anything=fake_optimize_anything)

    assert msg is not None
    assert msg.HasField("optimization_complete")
    c = msg.optimization_complete
    # Single-key dict {"__seed__": "optimized prompt text"} should unwrap to its value.
    assert c.best_candidate == "optimized prompt text"
    assert c.best_score == pytest.approx(0.7)
    assert c.total_evals == 3


def test_optimize_reflective_dataset_proxied(stub):
    """make_reflective_dataset sends ReflectiveDatasetRequest and returns data to the optimizer."""
    received_reflective: list = []

    def fake_optimize(*, seed_candidate, trainset, adapter, **_):
        # First evaluate with capture_traces=True to get trajectories.
        batch = list(trainset)
        eval_batch = adapter.evaluate(batch, dict(seed_candidate), capture_traces=True)
        # Then request a reflective dataset.
        reflective = adapter.make_reflective_dataset(
            dict(seed_candidate), eval_batch, ["instructions"]
        )
        received_reflective.append(reflective)
        return SimpleNamespace(
            candidates=[dict(seed_candidate)],
            best_idx=0,
            val_aggregate_scores=[1.0],
        )

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(
        start_request=pb.StartRequest(
            run_id="reflect-test",
            seed_candidate=_SEED,
            trainset=_TRAINSET,
            max_metric_calls=5,
            reflection_lm="fake",
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        call = stub.RunOptimization(gen())
        for msg in call:
            if msg.HasField("evaluate_batch_request"):
                req = msg.evaluate_batch_request
                # Respond with scores and trajectories (capture_traces=True).
                req_q.put(pb.ClientMessage(
                    evaluate_batch_response=pb.EvaluateBatchResponse(
                        request_id=req.request_id,
                        outputs=["ok"] * len(req.batch),
                        scores=[1.0] * len(req.batch),
                        trajectories=[
                            pb.Trajectory(
                                input_fields=dict(ex.fields),
                                output="ok",
                                feedback="correct",
                            )
                            for ex in req.batch
                        ],
                    )
                ))
            elif msg.HasField("reflective_dataset_request"):
                req = msg.reflective_dataset_request
                req_q.put(pb.ClientMessage(
                    reflective_dataset_response=pb.ReflectiveDatasetResponse(
                        request_id=req.request_id,
                        reflective_data={
                            "instructions": pb.ReflectiveComponentData(
                                entries=[
                                    pb.ReflectiveEntry(
                                        inputs={"q": "2+2"},
                                        generated_output="4",
                                        feedback="correct",
                                    )
                                ]
                            )
                        },
                    )
                ))
            elif msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                req_q.put(None)
                break

    assert len(received_reflective) == 1
    reflective = received_reflective[0]
    assert "instructions" in reflective
    entries = reflective["instructions"]
    assert len(entries) == 1
    assert entries[0]["Inputs"] == {"q": "2+2"}
    assert entries[0]["Generated Outputs"] == "4"
    assert entries[0]["Feedback"] == "correct"

# ------------------------------------------------------------------ more edge cases


def test_optimize_malformed_response_mismatched_scores(stub):
    """Client returning the wrong number of scores/outputs surfaces as an actionable error."""
    def fake_optimize(*, seed_candidate, trainset, adapter, **_):
        batch = list(trainset)
        adapter.evaluate(batch, dict(seed_candidate), capture_traces=False)
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.0])

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(
        start_request=pb.StartRequest(
            run_id="malformed-response",
            seed_candidate=_SEED,
            trainset=_TRAINSET,
            max_metric_calls=5,
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    final = None
    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        call = stub.RunOptimization(gen())
        for msg in call:
            if msg.HasField("evaluate_batch_request"):
                req = msg.evaluate_batch_request
                # Batch has 2 examples; return only 1 score/output -> mismatch.
                req_q.put(pb.ClientMessage(
                    evaluate_batch_response=pb.EvaluateBatchResponse(
                        request_id=req.request_id,
                        outputs=["ok"],
                        scores=[1.0],
                    )
                ))
            elif msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                final = msg
                req_q.put(None)
                break

    assert final is not None
    assert final.HasField("optimization_error")
    assert final.optimization_error.message.startswith("EvaluatorResponseError: ")
    assert "mismatched" in final.optimization_error.message


def test_concurrent_runs_different_ids_succeed(stub):
    """Two different run_ids running concurrently both complete correctly, independently."""
    import threading
    import time

    def fake_optimize(*, seed_candidate, **_):
        time.sleep(0.05)
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.5])

    results: dict = {}

    def run_one(run_id):
        results[run_id] = _run_optimize(stub, run_id=run_id)

    # Patch exactly once, before any thread starts, and patch the *name*
    # `gepa` inside servicer's module namespace rather than `.optimize` on
    # the real shared `gepa` package object. `gepa.rpc.servicer.gepa` IS that
    # shared module object (servicer.py does `import gepa` then calls
    # `gepa.optimize(...)`), so patching its `.optimize` attribute is exactly
    # equivalent to patching the process-global `gepa.optimize` -- doing that
    # from 3 threads at once is what made this test flaky (mock.patch context
    # managers aren't thread-safe, so the 3 enter/exit calls can restore the
    # attribute out of order and leave `gepa.optimize` permanently mocked for
    # the rest of the session, breaking unrelated tests depending on order).
    # Rebinding the `gepa` name itself, once, before the threads start, means
    # they only ever read an already-stable reference.
    fake_gepa = SimpleNamespace(optimize=fake_optimize)
    with patch("gepa.rpc.servicer.gepa", fake_gepa):
        threads = [threading.Thread(target=run_one, args=(f"concurrent-{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert len(results) == 3
    for run_id, msg in results.items():
        assert msg is not None
        assert msg.HasField("optimization_complete")
        assert msg.optimization_complete.run_id == run_id


def test_optimize_max_metric_calls_omitted_passes_none(stub):
    """Omitting max_metric_calls (proto default 0) passes max_metric_calls=None (unbounded) to gepa.optimize."""
    captured: dict = {}

    def fake_optimize(*, seed_candidate, max_metric_calls, **_):
        captured["max_metric_calls"] = max_metric_calls
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.0])

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(
        start_request=pb.StartRequest(
            run_id="unbounded-metric-calls",
            seed_candidate=_SEED,
            trainset=_TRAINSET,
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        call = stub.RunOptimization(gen())
        for msg in call:
            if msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                req_q.put(None)
                break

    assert captured["max_metric_calls"] is None


def test_optimize_omni_max_evals_omitted_passes_none(stub):
    """Omitting max_evals passes max_evals=None (unbounded) to OptimizeAnythingConfig."""
    from gepa.optimize_anything import Result

    captured: dict = {}

    def fake_optimize_anything(*, seed_candidate, config, **_):
        captured["max_evals"] = config.max_evals
        return Result(best_candidate=seed_candidate or "", best_score=0.0, total_evals=0)

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.OmniClientMessage(
        start_request=pb.OmniStartRequest(
            run_id="omni-unbounded-evals",
            seed_candidate="test",
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    with patch("gepa.rpc.servicer.optimize_anything", side_effect=fake_optimize_anything):
        call = stub.RunOptimizationOmni(gen())
        for msg in call:
            if msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                req_q.put(None)
                break

    assert captured["max_evals"] is None


def test_max_runs_eviction_prefers_finished(stub):
    """When _MAX_RUNS is reached, a finished run is evicted before a running one."""
    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.0])

    with patch("gepa.rpc.servicer._MAX_RUNS", 2):
        _run_optimize_patched(stub, run_id="evict-1", fake_optimize=fake_optimize)
        _run_optimize_patched(stub, run_id="evict-2", fake_optimize=fake_optimize)

        assert stub.GetStatus(pb.StatusRequest(run_id="evict-1")).status == pb.StatusResponse.COMPLETE
        assert stub.GetStatus(pb.StatusRequest(run_id="evict-2")).status == pb.StatusResponse.COMPLETE

        # A third run should evict the oldest finished entry (evict-1), not error out.
        _run_optimize_patched(stub, run_id="evict-3", fake_optimize=fake_optimize)

        assert stub.GetStatus(pb.StatusRequest(run_id="evict-1")).status == pb.StatusResponse.UNKNOWN
        assert stub.GetStatus(pb.StatusRequest(run_id="evict-3")).status == pb.StatusResponse.COMPLETE


# ------------------------------------------------------------------ tests: Cancel


def _blocking_run(stub, run_id, final_out, *, never_respond=True):
    """Start a run whose evaluate request is never answered, so the only way
    it ends is cancellation (or timeout). Appends the final message to
    `final_out`."""
    def fake_optimize(*, seed_candidate, trainset, adapter, **_):
        adapter.evaluate(list(trainset), dict(seed_candidate), capture_traces=False)
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.0])

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(start_request=pb.StartRequest(
        run_id=run_id, seed_candidate=_SEED, trainset=_TRAINSET, max_metric_calls=5,
    )))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        for msg in stub.RunOptimization(gen()):
            if msg.HasField("evaluate_batch_request") and never_respond:
                continue
            if msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                final_out.append(msg)
                req_q.put(None)
                break


def _await_status(stub, run_id, want, tries=300):
    import time
    for _ in range(tries):
        if stub.GetStatus(pb.StatusRequest(run_id=run_id)).status == want:
            return True
        time.sleep(0.01)
    return False


def test_cancel_unblocks_pending_wait_and_marks_cancelled(stub):
    """Cancel unblocks an in-flight adapter.evaluate wait and marks the run
    CANCELLED -- the two effects of cancellation, both observable."""
    import threading

    run_id = "cancel-me"
    final: list = []
    t = threading.Thread(target=_blocking_run, args=(stub, run_id, final))
    t.start()

    assert _await_status(stub, run_id, pb.StatusResponse.RUNNING), "run never reached RUNNING"
    resp = stub.Cancel(pb.CancelRequest(run_id=run_id), timeout=5)
    assert resp.accepted is True

    t.join(timeout=10)
    assert not t.is_alive(), "run did not unblock after Cancel"
    assert len(final) == 1
    assert final[0].HasField("optimization_error")
    assert "cancelled" in final[0].optimization_error.message
    assert stub.GetStatus(pb.StatusRequest(run_id=run_id)).status == pb.StatusResponse.CANCELLED


def test_cancel_unknown_run_id_not_accepted(stub):
    """Cancelling a run_id the server never saw is a no-op, not an error."""
    resp = stub.Cancel(pb.CancelRequest(run_id="never-existed"))
    assert resp.accepted is False


def test_cancel_already_finished_run_not_accepted(stub):
    """Cancelling an already-complete run is a no-op, not an error."""
    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.5])

    run_id = "already-done"
    _run_optimize_patched(stub, run_id=run_id, fake_optimize=fake_optimize)
    assert stub.GetStatus(pb.StatusRequest(run_id=run_id)).status == pb.StatusResponse.COMPLETE
    assert stub.Cancel(pb.CancelRequest(run_id=run_id)).accepted is False


# ------------------------------------------------------------------ tests: resume validation


def test_resume_mismatched_request_rejected(stub):
    """Re-issuing a finished run_id with a different trainset is rejected."""
    import grpc as grpc_module

    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.5])

    run_id = "resume-mismatch"
    _run_optimize_patched(stub, run_id=run_id, fake_optimize=fake_optimize)

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(start_request=pb.StartRequest(
        run_id=run_id, seed_candidate=_SEED,
        trainset=[pb.Example(id="different", fields={"input": "9+9", "answer": "18"})],
        max_metric_calls=5,
    )))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        try:
            list(stub.RunOptimization(gen()))
            req_q.put(None)
            pytest.fail("expected RpcError")
        except grpc_module.RpcError as e:
            req_q.put(None)
            assert e.code() == grpc_module.StatusCode.INVALID_ARGUMENT


def test_resume_matching_request_succeeds(stub):
    """Re-issuing a finished run_id with the *same* request is a legitimate
    resume and must still be allowed."""
    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.5])

    run_id = "resume-match"
    _run_optimize_patched(stub, run_id=run_id, fake_optimize=fake_optimize)
    msg = _run_optimize_patched(stub, run_id=run_id, fake_optimize=fake_optimize)
    assert msg is not None
    assert msg.HasField("optimization_complete")


# ------------------------------------------------------------------ tests: callback timeout


def test_callback_timeout_fails_run(tmp_path):
    """A client that never answers an EvaluateBatchRequest fails the run after
    callback_timeout instead of hanging forever."""
    with _custom_stub(tmp_path, callback_timeout=0.2) as stub:
        run_id = "times-out"
        final: list = []
        _blocking_run(stub, run_id, final)
        assert len(final) == 1
        assert final[0].HasField("optimization_error")
        assert final[0].optimization_error.message.startswith("RemoteAdapterTimeoutError: ")
        assert stub.GetStatus(pb.StatusRequest(run_id=run_id)).status == pb.StatusResponse.FAILED


# ------------------------------------------------------- tests: eviction vs running runs


@contextlib.contextmanager
def _saturating_runs(stub, count, prefix):
    """Start `count` runs that block inside gepa.optimize until released, so
    every one occupies a run slot. Yields once all are RUNNING."""
    import threading
    import time

    released = threading.Event()

    def fake_optimize(**_):
        released.wait(timeout=10)
        return SimpleNamespace(candidates=[{}], best_idx=0, val_aggregate_scores=[0.0])

    threads = []
    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        for i in range(count):
            req_q: queue.Queue = queue.Queue()
            req_q.put(pb.ClientMessage(start_request=pb.StartRequest(
                run_id=f"{prefix}-{i}", seed_candidate=_SEED, trainset=_TRAINSET, max_metric_calls=5,
            )))

            def drive(q=req_q):
                def gen():
                    while True:
                        msg = q.get()
                        if msg is None:
                            return
                        yield msg
                for msg in stub.RunOptimization(gen()):
                    if msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                        q.put(None)
                        break

            t = threading.Thread(target=drive)
            t.start()
            threads.append(t)

        for i in range(count):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if stub.GetStatus(pb.StatusRequest(run_id=f"{prefix}-{i}")).status == pb.StatusResponse.RUNNING:
                    break
                time.sleep(0.01)
        assert all(
            stub.GetStatus(pb.StatusRequest(run_id=f"{prefix}-{i}")).status == pb.StatusResponse.RUNNING
            for i in range(count)
        ), "not all runs reached RUNNING"

        try:
            yield
        finally:
            released.set()
            for t in threads:
                t.join(timeout=10)


def _attempt_run(stub, run_id):
    """Try to start one run; return the gRPC status code it was rejected
    with, or None if it was admitted."""
    import grpc as grpc_module

    def fake_optimize(**_):
        return SimpleNamespace(candidates=[{}], best_idx=0, val_aggregate_scores=[0.0])

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(start_request=pb.StartRequest(
        run_id=run_id, seed_candidate=_SEED, trainset=_TRAINSET, max_metric_calls=5,
    )))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        try:
            list(stub.RunOptimization(gen()))
            req_q.put(None)
            return None
        except grpc_module.RpcError as e:
            req_q.put(None)
            return e.code()


def test_max_runs_eviction_rejects_when_all_running(tmp_path):
    """At _MAX_RUNS with every tracked run still running, a new run is
    rejected with RESOURCE_EXHAUSTED rather than evicting a live run's status
    record (which would make GetStatus wrongly report UNKNOWN for it)."""
    import grpc as grpc_module

    with _custom_stub(tmp_path, max_concurrent_runs=4) as stub, patch("gepa.rpc.servicer._MAX_RUNS", 2):
        with _saturating_runs(stub, 2, "blocked"):
            assert _attempt_run(stub, "blocked-2") == grpc_module.StatusCode.RESOURCE_EXHAUSTED
        # Both originals survived eviction and completed normally.
        for i in range(2):
            assert stub.GetStatus(pb.StatusRequest(run_id=f"blocked-{i}")).status == pb.StatusResponse.COMPLETE


# ------------------------------------------------- tests: build_server headroom validation


def test_build_server_rejects_insufficient_headroom():
    """build_server refuses to start when max_workers leaves no headroom
    above max_concurrent_runs for control-plane RPCs."""
    from gepa.rpc.server import CONTROL_PLANE_RESERVE, build_server

    with pytest.raises(ValueError, match="max_workers"):
        build_server(port=_free_port(), max_workers=8, max_concurrent_runs=8)
    # Exactly at the boundary is allowed.
    server = build_server(
        port=_free_port(), max_workers=8 + CONTROL_PLANE_RESERVE, max_concurrent_runs=8,
    )
    server.stop(grace=0)


# ------------------------------------------------- tests: concurrency under saturation


def test_control_plane_responsive_when_runs_saturated(tmp_path):
    """With every permitted run slot occupied by a long-running stream,
    GetStatus and Cancel still respond promptly -- i.e. the reserved pool
    headroom actually protects the control plane."""
    import time

    import grpc as grpc_module

    with _custom_stub(tmp_path, max_workers=6, max_concurrent_runs=2) as stub:
        with _saturating_runs(stub, 2, "sat"):
            # A third run is correctly rejected: slots are full.
            assert _attempt_run(stub, "sat-overflow") == grpc_module.StatusCode.RESOURCE_EXHAUSTED

            # ...but control-plane RPCs on existing runs stay fast.
            for i in range(2):
                start = time.monotonic()
                resp = stub.GetStatus(pb.StatusRequest(run_id=f"sat-{i}"), timeout=2.0)
                assert time.monotonic() - start < 2.0
                assert resp.status == pb.StatusResponse.RUNNING

            start = time.monotonic()
            cancel_resp = stub.Cancel(pb.CancelRequest(run_id="sat-0"), timeout=2.0)
            assert time.monotonic() - start < 2.0
            assert cancel_resp.accepted is True


def test_excess_incoming_traffic_rejected_cheaply(tmp_path):
    """A burst of connection attempts well beyond max_concurrent_runs is
    rejected quickly (acquire-and-reject before any heavyweight setup) and
    doesn't itself exhaust the pool enough to block GetStatus."""
    import threading
    import time

    import grpc as grpc_module

    with _custom_stub(tmp_path, max_workers=6, max_concurrent_runs=2) as stub:
        with _saturating_runs(stub, 2, "excess-admitted"):
            results: dict = {}

            def attempt(idx):
                results[idx] = _attempt_run(stub, f"excess-{idx}")

            threads = [threading.Thread(target=attempt, args=(i,)) for i in range(10)]
            start = time.monotonic()
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            elapsed = time.monotonic() - start

            assert len(results) == 10
            assert all(code == grpc_module.StatusCode.RESOURCE_EXHAUSTED for code in results.values()), results
            # Rejection is cheap -- 10 of them shouldn't queue behind real work.
            assert elapsed < 10.0

            # Control plane still responsive despite the burst.
            start = time.monotonic()
            resp = stub.GetStatus(pb.StatusRequest(run_id="excess-admitted-0"), timeout=2.0)
            assert time.monotonic() - start < 2.0
            assert resp.status == pb.StatusResponse.RUNNING


def test_run_slot_released_on_every_exit_path(tmp_path):
    """The max_concurrent_runs permit must come back however a run ends --
    normal completion, mid-run exception, explicit cancellation, and client
    disconnect. A leak on any path would silently shrink capacity until the
    server refused all work.

    Run with max_concurrent_runs=1 so a leaked permit is immediately fatal:
    the *next* run would get RESOURCE_EXHAUSTED instead of succeeding.
    """
    import threading

    def ok_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[1.0])

    def boom_optimize(**_):
        raise RuntimeError("boom")

    with _custom_stub(tmp_path, max_workers=6, max_concurrent_runs=1) as stub:
        # (a) normal completion
        msg = _run_optimize_patched(stub, run_id="slot-ok", fake_optimize=ok_optimize)
        assert msg.HasField("optimization_complete")

        # (b) exception mid-run -- permit must still be freed
        msg = _run_optimize_patched(stub, run_id="slot-boom", fake_optimize=boom_optimize)
        assert msg.HasField("optimization_error")
        assert _attempt_run(stub, "slot-after-boom") is None, "permit leaked after an exception"

        # (c) explicit cancellation
        final: list = []
        t = threading.Thread(target=_blocking_run, args=(stub, "slot-cancel", final))
        t.start()
        assert _await_status(stub, "slot-cancel", pb.StatusResponse.RUNNING)
        assert stub.Cancel(pb.CancelRequest(run_id="slot-cancel"), timeout=5).accepted is True
        t.join(timeout=10)
        assert not t.is_alive()
        assert _attempt_run(stub, "slot-after-cancel") is None, "permit leaked after cancellation"

        # (d) client disconnects mid-stream (generator ends without a result)
        def disconnecting():
            def fake_optimize(*, seed_candidate, trainset, adapter, **_):
                adapter.evaluate(list(trainset), dict(seed_candidate), capture_traces=False)
                return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.0])

            req_q: queue.Queue = queue.Queue()
            req_q.put(pb.ClientMessage(start_request=pb.StartRequest(
                run_id="slot-disconnect", seed_candidate=_SEED, trainset=_TRAINSET, max_metric_calls=5,
            )))

            def gen():
                msg = req_q.get()
                yield msg
                # End the request stream immediately: simulates the client
                # going away without ever answering the evaluate request.

            with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
                with contextlib.suppress(Exception):
                    for _ in stub.RunOptimization(gen()):
                        pass

        d = threading.Thread(target=disconnecting)
        d.start()
        d.join(timeout=10)
        assert not d.is_alive()
        assert _await_status(stub, "slot-disconnect", pb.StatusResponse.FAILED, tries=500)
        assert _attempt_run(stub, "slot-after-disconnect") is None, "permit leaked after disconnect"


# ------------------------------------------------------- tests: security surface


@contextlib.contextmanager
def _real_server(tmp_path, **kwargs):
    """Spin up a server via gepa.rpc.server.build_server (not a bare
    GEPAServicer) so the host/TLS/auth-interceptor wiring is really
    exercised. Yields the bound port."""
    from gepa.rpc.server import build_server

    port = _free_port()
    kwargs.setdefault("runs_dir", str(tmp_path / "runs"))
    kwargs.setdefault("max_workers", 8)
    kwargs.setdefault("max_concurrent_runs", 4)
    server = build_server(port=port, **kwargs)
    server.start()
    try:
        yield port
    finally:
        server.stop(grace=0)


def test_default_host_is_loopback():
    """Binding without an explicit --host must stay off the wildcard."""
    from gepa.rpc.server import DEFAULT_HOST

    assert DEFAULT_HOST == "127.0.0.1"


def test_auth_token_rejects_missing_or_wrong_token(tmp_path):
    """Without the configured token every RPC is UNAUTHENTICATED -- unary and
    streaming alike (the interceptor must match the RPC's cardinality)."""
    import grpc as grpc_module

    with _real_server(tmp_path, auth_token="s3cret") as port:
        channel = grpc_module.insecure_channel(f"localhost:{port}")
        stub = pb_grpc.GEPAServiceStub(channel)
        try:
            with pytest.raises(grpc_module.RpcError) as ei:
                stub.GetStatus(pb.StatusRequest(run_id="whatever"))
            assert ei.value.code() == grpc_module.StatusCode.UNAUTHENTICATED

            with pytest.raises(grpc_module.RpcError) as ei:
                stub.GetStatus(pb.StatusRequest(run_id="x"), metadata=(("authorization", "wrong"),))
            assert ei.value.code() == grpc_module.StatusCode.UNAUTHENTICATED

            req_q: queue.Queue = queue.Queue()
            req_q.put(pb.ClientMessage(start_request=pb.StartRequest(
                run_id="r", seed_candidate=_SEED, trainset=_TRAINSET)))

            def gen():
                while True:
                    msg = req_q.get()
                    if msg is None:
                        return
                    yield msg

            with pytest.raises(grpc_module.RpcError) as ei:
                list(stub.RunOptimization(gen()))
            req_q.put(None)
            assert ei.value.code() == grpc_module.StatusCode.UNAUTHENTICATED
        finally:
            channel.close()


def test_auth_token_allows_correct_token(tmp_path):
    """The correct token in `authorization` metadata is accepted."""
    import grpc as grpc_module

    with _real_server(tmp_path, auth_token="s3cret") as port:
        channel = grpc_module.insecure_channel(f"localhost:{port}")
        stub = pb_grpc.GEPAServiceStub(channel)
        try:
            resp = stub.GetStatus(
                pb.StatusRequest(run_id="whatever"), metadata=(("authorization", "s3cret"),)
            )
            # No such run, but the request itself authenticated.
            assert resp.status == pb.StatusResponse.UNKNOWN
        finally:
            channel.close()


def test_tls_handshake_succeeds_with_configured_cert(tmp_path):
    """A server given --tls-cert/--tls-key accepts a TLS connection from a
    client that trusts that certificate."""
    import shutil
    import subprocess

    import grpc as grpc_module

    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI unavailable to generate a test certificate")

    cert_path = tmp_path / "server.crt"
    key_path = tmp_path / "server.key"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key_path), "-out", str(cert_path),
         "-days", "1", "-subj", "/CN=localhost"],
        check=True, capture_output=True,
    )

    with _real_server(tmp_path, tls_cert_path=str(cert_path), tls_key_path=str(key_path)) as port:
        creds = grpc_module.ssl_channel_credentials(root_certificates=cert_path.read_bytes())
        channel = grpc_module.secure_channel(f"localhost:{port}", creds)
        stub = pb_grpc.GEPAServiceStub(channel)
        try:
            assert stub.GetStatus(pb.StatusRequest(run_id="x"), timeout=5).status == pb.StatusResponse.UNKNOWN
        finally:
            channel.close()


def test_build_server_requires_both_tls_args():
    """One half of the TLS pair is a config error, not a silently-plaintext
    server."""
    from gepa.rpc.server import build_server

    with pytest.raises(ValueError, match="tls_cert_path and tls_key_path"):
        build_server(port=_free_port(), tls_cert_path="/tmp/irrelevant.crt")


def test_engine_allowlist_rejects_disallowed_engine(tmp_path):
    """An Omni request for an engine off the allowlist is PERMISSION_DENIED
    before any work starts."""
    import grpc as grpc_module

    with _custom_stub(tmp_path, allowed_engines=["gepa"]) as stub:
        req_q: queue.Queue = queue.Queue()
        req_q.put(pb.OmniClientMessage(start_request=pb.OmniStartRequest(
            run_id="disallowed-engine", seed_candidate="x", engine="autoresearch")))

        def gen():
            while True:
                msg = req_q.get()
                if msg is None:
                    return
                yield msg

        try:
            list(stub.RunOptimizationOmni(gen()))
            req_q.put(None)
            pytest.fail("expected RpcError")
        except grpc_module.RpcError as e:
            req_q.put(None)
            assert e.code() == grpc_module.StatusCode.PERMISSION_DENIED


def test_engine_allowlist_allows_listed_engine(tmp_path):
    """An engine that *is* allowlisted proceeds normally."""
    from gepa.optimize_anything import Result

    def fake(*, seed_candidate, **_):
        return Result(best_candidate=seed_candidate or "", best_score=1.0, total_evals=1)

    with _custom_stub(tmp_path, allowed_engines=["gepa"]) as stub:
        msg = _run_optimize_omni(stub, run_id="allowed-engine", fake_optimize_anything=fake)
        assert msg is not None
        assert msg.HasField("optimization_complete")


def test_reflection_lm_allowlist_rejects_disallowed_value(tmp_path):
    """A reflection_lm off the allowlist is PERMISSION_DENIED before any work."""
    import grpc as grpc_module

    with _custom_stub(tmp_path, allowed_reflection_lms=["openai/gpt-5.1"]) as stub:
        req_q: queue.Queue = queue.Queue()
        req_q.put(pb.ClientMessage(start_request=pb.StartRequest(
            run_id="disallowed-lm", seed_candidate=_SEED, trainset=_TRAINSET,
            reflection_lm="some/other-model")))

        def gen():
            while True:
                msg = req_q.get()
                if msg is None:
                    return
                yield msg

        try:
            list(stub.RunOptimization(gen()))
            req_q.put(None)
            pytest.fail("expected RpcError")
        except grpc_module.RpcError as e:
            req_q.put(None)
            assert e.code() == grpc_module.StatusCode.PERMISSION_DENIED


def test_reflection_lm_allowlist_applies_to_configured_default(tmp_path):
    """When the client omits reflection_lm, the server's configured default is
    what gets allowlist-checked -- and passes when it's listed."""
    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.5])

    with _custom_stub(
        tmp_path, allowed_reflection_lms=["openai/gpt-5.1"], default_reflection_lm="openai/gpt-5.1",
    ) as stub:
        with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
            msg = _run_optimize(stub, run_id="allowed-lm", reflection_lm="")
        assert msg is not None
        assert msg.HasField("optimization_complete")


# ------------------------------------------------------- tests: Omni output_dir


def test_omni_output_dir_routes_under_run_dir(stub):
    """OptimizeAnythingConfig.output_dir must land under the server's runs_dir
    rather than being left None, which defaults it to a path under the
    server's CWD regardless of --runs-dir."""
    from gepa.optimize_anything import Result

    captured: dict = {}

    def fake(*, seed_candidate, config, **_):
        captured["output_dir"] = config.output_dir
        captured["run_dir"] = config.run_dir
        return Result(best_candidate=seed_candidate or "", best_score=0.0, total_evals=0)

    with patch("gepa.rpc.servicer.optimize_anything", side_effect=fake):
        req_q: queue.Queue = queue.Queue()
        req_q.put(pb.OmniClientMessage(start_request=pb.OmniStartRequest(
            run_id="output-dir-check", seed_candidate="x")))

        def gen():
            while True:
                msg = req_q.get()
                if msg is None:
                    return
                yield msg

        for msg in stub.RunOptimizationOmni(gen()):
            if msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                req_q.put(None)
                break

    assert captured["output_dir"] is not None
    assert captured["run_dir"] is not None
    assert str(captured["output_dir"]).startswith(str(captured["run_dir"]))


# ------------------------------------------------- tests: cancellation state machine


def test_cancel_reports_cancelling_until_optimizer_actually_stops(tmp_path):
    """Cancel must not claim a terminal CANCELLED while the optimizer is
    still running.

    The gepa engine only honors <run_dir>/gepa.stop at its next checkpoint,
    so there is a real window where cancellation is requested but work
    continues. Previously GetStatus reported terminal CANCELLED immediately,
    which told a client the run was over while it was still consuming its
    run slot and budget. That window is now CANCELLING (non-terminal).
    """
    import threading
    import time

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_optimize(*, seed_candidate, **_):
        started.set()
        release.wait(timeout=10)  # ignores gepa.stop: work between checkpoints
        finished.set()
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[1.0])

    with _custom_stub(tmp_path, max_workers=8, max_concurrent_runs=2) as stub:
        run_id = "cancelling-window"

        def drive():
            with patch("gepa.rpc.servicer.gepa.optimize", side_effect=slow_optimize):
                _run_optimize(stub, run_id=run_id)

        t = threading.Thread(target=drive)
        t.start()
        assert started.wait(5)

        assert stub.Cancel(pb.CancelRequest(run_id=run_id), timeout=5).accepted is True
        # The optimizer has NOT returned yet.
        assert not finished.is_set()
        assert stub.GetStatus(pb.StatusRequest(run_id=run_id)).status == pb.StatusResponse.CANCELLING

        # A second Cancel is a no-op, reported as such rather than re-issued.
        again = stub.Cancel(pb.CancelRequest(run_id=run_id), timeout=5)
        assert again.accepted is False
        assert "already" in again.message

        release.set()
        t.join(timeout=10)
        assert not t.is_alive()
        # It completed on its own before honoring the stop, so COMPLETE is the
        # honest terminal state -- and it must not flip afterwards.
        final = stub.GetStatus(pb.StatusRequest(run_id=run_id)).status
        assert final in (pb.StatusResponse.COMPLETE, pb.StatusResponse.CANCELLED)
        time.sleep(0.3)
        assert stub.GetStatus(pb.StatusRequest(run_id=run_id)).status == final


def test_terminal_status_is_never_overwritten(tmp_path):
    """A completed run's status must survive a racing Cancel, and vice versa.

    Regression for an observed CANCELLED -> COMPLETE flip: Cancel marked the
    run cancelled, then the runner's success path unconditionally wrote
    "complete" over it. Hammering the race is the only way to catch it, so
    this drives many short runs with a Cancel fired alongside each.
    """
    import threading

    def fast_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[1.0])

    with _custom_stub(tmp_path, max_workers=10, max_concurrent_runs=4) as stub:
        for i in range(25):
            run_id = f"race-terminal-{i}"
            with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fast_optimize):
                t = threading.Thread(target=lambda r=run_id: _run_optimize(stub, run_id=r))
                t.start()
                try:
                    stub.Cancel(pb.CancelRequest(run_id=run_id), timeout=2)
                except Exception:
                    pass
                t.join(timeout=10)
                assert not t.is_alive()

            first = stub.GetStatus(pb.StatusRequest(run_id=run_id)).status
            # Whatever it settled on must be terminal and stable -- never
            # CANCELLING, and never changing on a later read.
            assert first in (
                pb.StatusResponse.COMPLETE,
                pb.StatusResponse.CANCELLED,
                pb.StatusResponse.FAILED,
            ), f"{run_id} settled on non-terminal {first}"
            assert stub.GetStatus(pb.StatusRequest(run_id=run_id)).status == first


def test_cancelled_run_reports_cancelled_error_kind(stub):
    """A cancelled run's OptimizationError carries error_kind=CANCELLED, so a
    client can tell a deliberate stop from a genuine failure."""
    import threading

    run_id = "cancel-kind"
    final: list = []
    t = threading.Thread(target=_blocking_run, args=(stub, run_id, final))
    t.start()
    assert _await_status(stub, run_id, pb.StatusResponse.RUNNING)
    assert stub.Cancel(pb.CancelRequest(run_id=run_id), timeout=5).accepted is True
    t.join(timeout=10)
    assert not t.is_alive()

    assert len(final) == 1
    assert final[0].HasField("optimization_error")
    assert final[0].optimization_error.error_kind == pb.OptimizationError.CANCELLED
    assert stub.GetStatus(pb.StatusRequest(run_id=run_id)).status == pb.StatusResponse.CANCELLED


# ------------------------------------------- tests: silent-stream admission (DoS)


def _open_silent_stream(stub, stop_event):
    """Open a RunOptimization stream and never send anything, until told."""
    q: queue.Queue = queue.Queue()

    def gen():
        while not stop_event.is_set():
            try:
                m = q.get(timeout=0.1)
            except Exception:
                continue
            if m is None:
                return
            yield m

    try:
        for _ in stub.RunOptimization(gen()):
            pass
    except Exception:
        pass


def test_silent_streams_do_not_starve_the_server(tmp_path):
    """A client that opens a stream and sends nothing must not hold a run slot
    indefinitely.

    The slot is acquired before the first message can be read (gRPC hands us
    an iterator, not a message), so without a bounded handshake N silent
    connections take the server down with no auth and no payload -- which is
    exactly what this reproduced before the fix: two silent streams at
    max_concurrent_runs=2 made a legitimate run fail RESOURCE_EXHAUSTED.
    """
    import threading
    import time

    def fast_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[1.0])

    stop = threading.Event()
    with _custom_stub(
        tmp_path, max_workers=8, max_concurrent_runs=2, handshake_timeout=1.0,
    ) as stub:
        threads = [threading.Thread(target=_open_silent_stream, args=(stub, stop), daemon=True)
                   for _ in range(2)]
        for t in threads:
            t.start()
        time.sleep(0.3)  # both silent streams are open and holding slots

        # Before the handshake deadline, capacity is genuinely consumed.
        assert _attempt_run(stub, "during-handshake") is not None

        # After it, the slots are reclaimed and real work is admitted again.
        time.sleep(1.5)
        try:
            with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fast_optimize):
                msg = _run_optimize(stub, run_id="after-handshake-timeout")
            assert msg is not None
            assert msg.HasField("optimization_complete"), "slot was not reclaimed from silent streams"
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=5)


def test_handshake_timeout_returns_deadline_exceeded(tmp_path):
    """The abandoned handshake is reported as DEADLINE_EXCEEDED, not a generic
    failure, so a slow client can tell what happened."""
    import threading

    import grpc as grpc_module

    stop = threading.Event()
    with _custom_stub(tmp_path, max_workers=6, max_concurrent_runs=2, handshake_timeout=0.5) as stub:
        q: queue.Queue = queue.Queue()

        def gen():
            while not stop.is_set():
                try:
                    m = q.get(timeout=0.1)
                except Exception:
                    continue
                if m is None:
                    return
                yield m

        try:
            with pytest.raises(grpc_module.RpcError) as ei:
                list(stub.RunOptimization(gen()))
            assert ei.value.code() == grpc_module.StatusCode.DEADLINE_EXCEEDED
            assert "start_request" in (ei.value.details() or "")
        finally:
            stop.set()


# ------------------------------------------- tests: resume fingerprint completeness


def test_resume_rejected_when_example_contents_change(stub):
    """Keeping every example id but changing their fields must be rejected.

    Fingerprinting ids alone let a resume continue a checkpoint against
    different data while looking identical to the server.
    """
    import grpc as grpc_module

    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.5])

    run_id = "resume-content-change"
    _run_optimize_patched(stub, run_id=run_id, fake_optimize=fake_optimize)

    # Same ids as _TRAINSET, different field values.
    mutated = [
        pb.Example(id="1", fields={"input": "99+99", "answer": "198"}),
        pb.Example(id="2", fields={"input": "7+7", "answer": "14"}),
    ]
    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(start_request=pb.StartRequest(
        run_id=run_id, seed_candidate=_SEED, trainset=mutated,
        max_metric_calls=20, reflection_lm="fake")))

    def gen():
        while True:
            m = req_q.get()
            if m is None:
                return
            yield m

    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        try:
            list(stub.RunOptimization(gen()))
            req_q.put(None)
            pytest.fail("expected RpcError: example contents changed")
        except grpc_module.RpcError as e:
            req_q.put(None)
            assert e.code() == grpc_module.StatusCode.INVALID_ARGUMENT


def test_resume_rejected_when_reflection_lm_changes(stub):
    """reflection_lm changes what the optimizer does, so it belongs in the
    fingerprint -- it was omitted before."""
    import grpc as grpc_module

    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.5])

    run_id = "resume-lm-change"
    _run_optimize_patched(stub, run_id=run_id, fake_optimize=fake_optimize)

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.ClientMessage(start_request=pb.StartRequest(
        run_id=run_id, seed_candidate=_SEED, trainset=_TRAINSET,
        max_metric_calls=20, reflection_lm="a-different-model")))

    def gen():
        while True:
            m = req_q.get()
            if m is None:
                return
            yield m

    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
        try:
            list(stub.RunOptimization(gen()))
            req_q.put(None)
            pytest.fail("expected RpcError: reflection_lm changed")
        except grpc_module.RpcError as e:
            req_q.put(None)
            assert e.code() == grpc_module.StatusCode.INVALID_ARGUMENT


def test_resume_still_allowed_when_request_is_identical(stub):
    """The stricter fingerprint must not break legitimate resume: field
    ordering is normalized, so an identical request still matches."""
    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(candidates=[dict(seed_candidate)], best_idx=0, val_aggregate_scores=[0.5])

    run_id = "resume-identical"
    _run_optimize_patched(stub, run_id=run_id, fake_optimize=fake_optimize)
    msg = _run_optimize_patched(stub, run_id=run_id, fake_optimize=fake_optimize)
    assert msg is not None
    assert msg.HasField("optimization_complete")
