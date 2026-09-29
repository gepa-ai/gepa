"""Real end-to-end test for gepa.rpc: runs an actual GEPA optimization
through the real gepa-rpc server subprocess and the real gRPC wire protocol,
with no mocking of gepa.optimize/optimize_anything anywhere in the path.

The only faked component is the outbound LLM traffic (both the task-solving
call this test's evaluate() makes and GEPA's own internal reflection call),
which is replayed from a local record/replay stub (see
gepa.rpc.testing.fake_llm_server) instead of hitting a real provider. Set
RECORD_TESTS=true to regenerate the cache and golden file against a real
OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

grpc = pytest.importorskip("grpc")

from gepa.rpc.generated import gepa_pb2 as pb  # noqa: E402
from gepa.rpc.generated import gepa_pb2_grpc as pb_grpc  # noqa: E402

RECORD = os.environ.get("RECORD_TESTS", "false").lower() == "true"

_DIR = Path(__file__).parent
CACHE_FILE = _DIR / "llm_cache.json"
GOLDEN_FILE = _DIR / "optimized_candidate.txt"

TASK_MODEL = "openai/gpt-4.1-nano"
REFLECTION_MODEL = "openai/gpt-4.1-nano"

DATASET = [
    {"id": "1", "text": "I love this product!", "label": "positive"},
    {"id": "2", "text": "This is terrible.", "label": "negative"},
    {"id": "3", "text": "Oh great, another Monday.", "label": "negative"},
    {"id": "4", "text": "Well, that could have gone better.", "label": "negative"},
    {"id": "5", "text": "Not bad at all, actually.", "label": "positive"},
    {"id": "6", "text": "I guess it's fine, whatever.", "label": "negative,neutral"},
]

# Deliberately underspecified: says nothing about output format, so
# gpt-4.1-nano answers in a full sentence ("The sentiment of the text is
# positive."). Grading requires the first word to exactly match, so this
# seed scores near zero despite understanding the task correctly, giving
# GEPA's reflection loop a real, fixable failure (tighten the output format)
# rather than a seed that already scores perfectly and never triggers
# reflection.
SEED_CANDIDATE = "Classify the sentiment of the following text."
OBJECTIVE = "Maximize accuracy of sentiment classification."


def _grade(answer: str, accepted_labels: str) -> float:
    first_word = answer.strip().split()[0].strip(".,!\"'").lower() if answer.strip() else ""
    accepted = accepted_labels.lower().split(",")
    return 1.0 if first_word in accepted else 0.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"nothing listening on 127.0.0.1:{port} after {timeout}s")


def _call_task_lm(stub_port: int, candidate: str, text: str) -> str:
    """The evaluate() side of this test: a real HTTP call to the fake-LLM
    stub to attempt the task, mirroring AIME's task_lm role."""
    prompt = f"{candidate}\n\nText: {text}"
    body = json.dumps(
        {"model": TASK_MODEL, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{stub_port}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]["content"]


@pytest.fixture()
def e2e_servers(tmp_path):
    """Spawn a real fake-LLM stub and a real gepa-rpc server, each as a
    genuine OS subprocess (not in-process), and tear both down afterward."""
    if not RECORD and not CACHE_FILE.exists():
        pytest.fail(f"Cache file not found: {CACHE_FILE}. Run with 'RECORD_TESTS=true pytest' to generate it.")

    stub_port = _free_port()
    server_port = _free_port()

    stub_cmd = [
        sys.executable, "-m", "gepa.rpc.testing.fake_llm_server",
        "--port", str(stub_port),
        "--cache-file", str(CACHE_FILE),
    ]
    if RECORD:
        stub_cmd.append("--record")
    stub_proc = subprocess.Popen(stub_cmd)

    server_env = dict(os.environ)
    server_env["OPENAI_API_BASE"] = f"http://127.0.0.1:{stub_port}"
    server_env.setdefault("OPENAI_API_KEY", "sk-fake-e2e-test-key")
    server_cmd = [
        sys.executable, "-m", "gepa.rpc.cli",
        "--port", str(server_port),
        "--runs-dir", str(tmp_path / "runs"),
    ]
    server_proc = subprocess.Popen(server_cmd, env=server_env)

    try:
        _wait_for_port(stub_port)
        _wait_for_port(server_port)
        yield stub_port, server_port
    finally:
        server_proc.terminate()
        stub_proc.terminate()
        for p in (server_proc, stub_proc):
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)


def test_real_optimization_through_omni(e2e_servers):
    """Drives a real RunOptimizationOmni call end to end and asserts the
    final candidate against a committed golden file, mirroring
    tests/test_aime_prompt_optimization's structure."""
    stub_port, server_port = e2e_servers

    channel = grpc.insecure_channel(f"localhost:{server_port}")
    stub = pb_grpc.GEPAServiceStub(channel)

    req_q: queue.Queue = queue.Queue()
    req_q.put(pb.OmniClientMessage(
        start_request=pb.OmniStartRequest(
            run_id="rpc-e2e-sentiment",
            seed_candidate=SEED_CANDIDATE,
            dataset=[
                pb.Example(id=d["id"], fields={"text": d["text"], "label": d["label"]})
                for d in DATASET
            ],
            objective=OBJECTIVE,
            reflection_lm=REFLECTION_MODEL,
            max_evals=30,
        )
    ))

    def gen():
        while True:
            msg = req_q.get()
            if msg is None:
                return
            yield msg

    final = None
    call = stub.RunOptimizationOmni(gen())
    try:
        for msg in call:
            if msg.HasField("evaluate_batch_request"):
                req = msg.evaluate_batch_request
                scores = []
                for ex in req.batch:
                    answer = _call_task_lm(stub_port, req.candidate, ex.fields["text"])
                    scores.append(_grade(answer, ex.fields["label"]))
                req_q.put(pb.OmniClientMessage(
                    evaluate_batch_response=pb.OmniEvaluateBatchResponse(
                        request_id=req.request_id,
                        scores=scores,
                        side_infos=["{}"] * len(req.batch),
                    )
                ))
            elif msg.HasField("optimization_complete") or msg.HasField("optimization_error"):
                final = msg
                req_q.put(None)
                break
    finally:
        channel.close()

    assert final is not None
    assert final.HasField("optimization_complete"), (
        final.optimization_error.message if final.HasField("optimization_error") else "no result received"
    )

    best_candidate = final.optimization_complete.best_candidate

    if RECORD:
        GOLDEN_FILE.write_text(best_candidate)
        assert isinstance(best_candidate, str) and len(best_candidate) > 0
    else:
        if not GOLDEN_FILE.exists():
            pytest.fail(f"Golden file not found: {GOLDEN_FILE}. Run with 'RECORD_TESTS=true pytest' to generate it.")
        expected = GOLDEN_FILE.read_text()
        assert best_candidate == expected
