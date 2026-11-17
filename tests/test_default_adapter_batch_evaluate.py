# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""DefaultAdapter.batch_evaluate against a local OpenAI-compatible stub server.

The stub sits at the network boundary, so the real LM -> litellm -> HTTP path
runs. It echoes the user message back and records how many requests were in
flight at once, which is what batching across (candidate, batch) items changes.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
from gepa.core.adapter import invoke_batch_evaluate

STUB_LATENCY_S = 0.3


class _StubServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _StubHandler)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.num_requests = 0


class _StubHandler(BaseHTTPRequestHandler):
    server: _StubServer

    def log_message(self, format, *args):
        return

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.in_flight += 1
            self.server.num_requests += 1
            self.server.max_in_flight = max(self.server.max_in_flight, self.server.in_flight)
        try:
            time.sleep(STUB_LATENCY_S)
            user = next(m["content"] for m in body["messages"] if m["role"] == "user")
            payload = json.dumps(
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": user}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        finally:
            with self.server.lock:
                self.server.in_flight -= 1


@pytest.fixture
def stub_server():
    server = _StubServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _adapter(server: _StubServer) -> DefaultAdapter:
    return DefaultAdapter(
        model="openai/stub-model",
        max_litellm_workers=16,
        litellm_batch_completion_kwargs={
            "api_base": f"http://127.0.0.1:{server.server_address[1]}/v1",
            "api_key": "stub",
        },
    )


def _items():
    batch = [
        {"input": "answer is 1", "additional_context": {}, "answer": "1"},
        {"input": "answer is 2", "additional_context": {}, "answer": "x"},
    ]
    return [({"system_prompt": f"candidate {i}"}, batch) for i in range(3)]


def test_batch_evaluate_sends_all_items_concurrently(stub_server):
    results = invoke_batch_evaluate(_adapter(stub_server), _items(), capture_traces=True)

    assert stub_server.num_requests == 6
    assert stub_server.max_in_flight == 6
    for result in results:
        assert result.scores == [1.0, 0.0]
        assert [o["full_assistant_response"] for o in result.outputs] == ["answer is 1", "answer is 2"]
        assert result.trajectories is not None
        assert [t["feedback"].startswith("The generated response is correct") for t in result.trajectories] == [
            True,
            False,
        ]


def test_batch_evaluate_matches_per_item_evaluate(stub_server):
    adapter = _adapter(stub_server)
    items = _items()

    batched = invoke_batch_evaluate(adapter, items, capture_traces=False)
    single = [adapter.evaluate(batch, candidate, capture_traces=False) for candidate, batch in items]

    assert batched == single
