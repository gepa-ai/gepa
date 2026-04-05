"""Eval server: the single choke point for all evaluation and budget enforcement.

Every adapter — in-process (GEPA) or external (Claude Code) — goes through
the EvalServer. It provides two interfaces to the same budget-controlled eval:

1. ``server.evaluate(candidate)`` — direct Python call (no HTTP overhead).
2. ``POST server.url/evaluate`` — HTTP endpoint for external/black-box systems.

The runner always creates the EvalServer. Adapters cannot construct their own.

HTTP Protocol
-------------
POST /evaluate
    Body: {"candidate": "<text>", "example_id": "<optional>"}
    Response: {"score": 1.23, "info": {...}, "budget": {"used": 5, "remaining": 95, ...}}
    Status 429 when budget exhausted.

GET /status
    Response: {"budget": {...}, "task": "<name>", "best_score": 1.23}

GET /task
    Response: {"name": "...", "description": "...", "initial_candidate": "...", ...}
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from terrarium.budget import BudgetExhausted, BudgetTracker
from terrarium.task import Example, Task


class EvalServer:
    """Eval server with budget enforcement — the single source of truth.

    In-process adapters call :meth:`evaluate` directly.
    External adapters POST to the HTTP endpoint at :attr:`url`.
    Both go through the same budget counter.
    """

    def __init__(self, task: Task, budget: BudgetTracker) -> None:
        self.task = task
        self.budget = budget
        self.best_score: float = float("-inf")
        self.best_candidate: str = task.initial_candidate
        self.eval_log: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        # Build example lookup for dataset tasks
        self._examples: dict[str, Example] = {}
        for dataset in [task.train_set, task.test_set]:
            if dataset:
                for ex in dataset:
                    self._examples[ex.id] = ex

        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── Direct Python API (used by in-process adapters like GEPA) ───────

    def evaluate(self, candidate: str, example: Example | None = None) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate with budget enforcement.

        This is the primary API for in-process adapters. Same budget counter
        as the HTTP endpoint — there is only one counter.

        Raises:
            BudgetExhausted: When the budget has been used up.
        """
        self.budget.check()

        if example is not None:
            score, info = self.task.eval_fn(candidate, example)
        else:
            score, info = self.task.eval_fn(candidate)

        self.budget.record(score)
        self._track(candidate, score)

        info["_budget"] = self.budget.status()
        return score, info

    # ── HTTP server (used by external/black-box adapters) ───────────────

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("Server not started")
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def start(self, port: int = 0) -> int:
        """Start the HTTP server. Returns the port number."""
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/evaluate":
                    server_ref._handle_evaluate(self)
                else:
                    self.send_error(404)

            def do_GET(self):
                if self.path == "/status":
                    server_ref._handle_status(self)
                elif self.path == "/task":
                    server_ref._handle_task_info(self)
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass  # suppress request logging

        self._server = HTTPServer(("localhost", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None

    # ── Internal ────────────────────────────────────────────────────────

    def _track(self, candidate: str, score: float) -> None:
        with self._lock:
            self.eval_log.append({"eval": self.budget.used, "score": score, "candidate_len": len(candidate)})
            if score > self.best_score:
                self.best_score = score
                self.best_candidate = candidate

    def _handle_evaluate(self, handler: BaseHTTPRequestHandler) -> None:
        content_length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(content_length)) if content_length else {}

        candidate = body.get("candidate", "")
        example_id = body.get("example_id")

        if self.budget.exhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)
            return

        try:
            example = self._examples.get(example_id) if example_id else None
            score, info = self.evaluate(candidate, example)
            self._send_json(handler, {"score": score, "info": info, "budget": self.budget.status()})

        except BudgetExhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)

        except Exception as e:
            self._send_json(handler, {"error": str(e), "budget": self.budget.status()}, status=500)

    def _handle_status(self, handler: BaseHTTPRequestHandler) -> None:
        self._send_json(handler, {
            "budget": self.budget.status(),
            "task": self.task.name,
            "best_score": self.best_score,
            "total_evals": self.budget.used,
        })

    def _handle_task_info(self, handler: BaseHTTPRequestHandler) -> None:
        self._send_json(handler, {
            "name": self.task.name,
            "description": self.task.description,
            "initial_candidate": self.task.initial_candidate,
            "has_dataset": self.task.has_dataset,
            "train_size": len(self.task.train_set) if self.task.train_set else 0,
            "test_size": len(self.task.test_set) if self.task.test_set else 0,
            "metadata": {k: v for k, v in self.task.metadata.items() if isinstance(v, (str, int, float, bool))},
        })

    def _send_json(self, handler: BaseHTTPRequestHandler, data: dict, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
