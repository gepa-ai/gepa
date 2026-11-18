"""Local record/replay stand-in for an OpenAI-compatible LLM endpoint, used by
the gepa.rpc end-to-end tests so a real gepa-rpc server subprocess can run a
real optimization loop without live LLM network access in CI. Unlike gepa's
own mocked_lms fixture, which injects a Python callable in-process, this
intercepts at the network boundary (litellm is pointed at it via
OPENAI_API_BASE) since non-Python SDK clients require the server to run as a
separate process. Cache keys are content-addressed by (model, messages),
matching gepa's own convention.

Usage::

    python -m gepa.rpc.testing.fake_llm_server --port 8000 \\
        --cache-file tests/test_rpc_e2e/llm_cache.json [--record]
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _cache_key(model: str, messages: list[dict[str, Any]]) -> str:
    return json.dumps({"model": model, "messages": messages}, sort_keys=True)


class _Cache:
    def __init__(self, cache_file: Path, record: bool):
        self._cache_file = cache_file
        self._record = record
        self._lock = threading.Lock()
        if cache_file.exists():
            self._data: dict[str, str] = json.loads(cache_file.read_text())
        else:
            self._data = {}

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    @property
    def record(self) -> bool:
        return self._record


def _make_handler(cache: _Cache) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                request = json.loads(raw)
                model = request["model"]
                messages = request["messages"]
            except (json.JSONDecodeError, KeyError) as e:
                self._send_json(400, {"error": {"message": f"malformed request: {e}"}})
                return

            key = _cache_key(model, messages)
            content = cache.get(key)

            if content is None:
                if not cache.record:
                    self._send_json(
                        500,
                        {"error": {"message": f"fake_llm_server: unseen input in replay mode. key={key!r}"}},
                    )
                    return
                import litellm

                try:
                    response = litellm.completion(model=model, messages=messages)
                except Exception as e:
                    self._send_json(502, {"error": {"message": f"fake_llm_server: real provider call failed: {e}"}})
                    return
                content = response.choices[0].message.content or ""  # type: ignore[union-attr]
                cache.put(key, content)

            self._send_json(
                200,
                {
                    "id": "fake-llm-response",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                },
            )

    return Handler


def serve(port: int, cache_file: Path, record: bool) -> None:
    cache = _Cache(cache_file, record)
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(cache))
    mode = "record" if record else "replay"
    logger.info("fake_llm_server listening on 127.0.0.1:%d (%s mode, cache=%s)", port, mode, cache_file)
    server.serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--record", action="store_true", help="On a cache miss, call the real provider.")
    args = parser.parse_args()
    serve(args.port, args.cache_file, args.record)


if __name__ == "__main__":
    main()
