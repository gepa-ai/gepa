"""Integration tests for GEPAServicer.

These tests spin up a real gRPC server and drive it with a Python stub client.
The gepa optimizer and reflection LM are monkey-patched so no real LLM calls
are made — the tests verify the gRPC wire protocol and adapter pipeline only.
"""

from __future__ import annotations

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


# ------------------------------------------------------------------ helpers


_TRAINSET = [
    pb.Example(id="1", fields={"input": "2+2", "answer": "4"}),
    pb.Example(id="2", fields={"input": "3+3", "answer": "6"}),
]

_SEED = {"instructions": "Answer the question."}


def _run_optimize(stub, *, run_id: str, fake_optimize, seed=None, trainset=None, max_metric_calls=20):
    """Drive a full RunOptimization round-trip with a mock optimizer.

    The fake_optimize callable receives the same kwargs the servicer passes to
    gepa.optimize. It must return a GEPAResult-shaped namespace.

    The test client automatically responds to EvaluateBatchRequest messages
    with score=1.0 for every example.
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
    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=fake_optimize):
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

    msg = _run_optimize(stub, run_id="opt-happy", fake_optimize=fake_optimize)

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

    _run_optimize(stub, run_id="opt-proxy", fake_optimize=fake_optimize)

    assert len(received_batches) == 1
    batch, scores = received_batches[0]
    assert len(batch) == len(_TRAINSET)
    assert scores == [1.0] * len(_TRAINSET)


def test_optimize_error_propagated(stub):
    """An exception in the optimizer thread becomes a generic OptimizationError (no leak)."""
    def fake_optimize(**_):
        raise RuntimeError("boom")

    msg = _run_optimize(stub, run_id="opt-err", fake_optimize=fake_optimize)

    assert msg is not None
    assert msg.HasField("optimization_error")
    assert "boom" not in msg.optimization_error.message
    assert msg.optimization_error.message == "optimization failed"


def test_get_status_complete(stub):
    """GetStatus returns COMPLETE after a successful RunOptimization."""
    def fake_optimize(*, seed_candidate, **_):
        return SimpleNamespace(
            candidates=[dict(seed_candidate)],
            best_idx=0,
            val_aggregate_scores=[0.5],
        )

    run_id = "status-test"
    _run_optimize(stub, run_id=run_id, fake_optimize=fake_optimize)

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
    """An exception in the Omni optimizer thread becomes a generic OptimizationError (no leak)."""
    def fake_optimize_anything(**_):
        raise ValueError("omni boom")

    msg = _run_optimize_omni(stub, run_id="omni-err", fake_optimize_anything=fake_optimize_anything)

    assert msg is not None
    assert msg.HasField("optimization_error")
    assert "omni boom" not in msg.optimization_error.message
    assert msg.optimization_error.message == "optimization failed"


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

    msg = _run_optimize(stub, run_id="opt-empty", fake_optimize=fake_optimize, trainset=[])
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
    """GetStatus for a failed run returns FAILED with sanitized message."""
    def fake_optimize(**_):
        raise RuntimeError("internal details that must not leak")

    run_id = "status-fail"
    _run_optimize(stub, run_id=run_id, fake_optimize=fake_optimize)

    resp = stub.GetStatus(pb.StatusRequest(run_id=run_id))
    assert resp.status == pb.StatusResponse.FAILED
    assert "internal details" not in resp.message
    assert resp.message == "optimization failed"


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
        msg = _run_optimize(stub, run_id="dup-run", fake_optimize=fake_optimize)
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
