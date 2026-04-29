"""Eval server: the single choke point for evaluation, budget, and tracking.

Every backend — in-process or external — goes through the :class:`EvalServer`.
It exposes two interfaces to the same budget-controlled eval:

1. ``server.evaluate(candidate)`` — direct Python call (no HTTP overhead).
2. ``POST server.url/evaluate`` — HTTP endpoint for external/black-box backends.

The api always creates the EvalServer; backends never construct their own.

HTTP Protocol
-------------
POST /evaluate
    Body: {"candidate": "<text>", "example_id": "<optional>"}
    Response: {"score": 1.23, "info": {...}, "budget": {"used": 5, ...}}
    Status 429 when budget exhausted.

POST /evaluate_examples
    Body: {"candidate": "<text>", "example_ids": ["a","b","c"]}
       or {"candidate": "<text>", "split": "train"|"val"|"test"|"all"}
    Evaluates the candidate on the requested examples in parallel,
    respecting the concurrency limit. Each example = 1 budget tick.

POST /validate
    Body: {"candidate": "<text>"}
    Aggregates over the hidden ``val_set`` (individual scores hidden).

GET /status   → {"budget": {...}, "task": "...", "best_score": ..., ...}
GET /task     → task metadata (name/objective/background/initial_candidate/...)
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from gepa.omni.budget import BudgetExhausted, BudgetTracker
from gepa.omni.task import EvalFn, Example, Task

DEFAULT_MAX_CONCURRENCY = 8


class EvalServer:
    """Eval server with budget enforcement — the single source of truth.

    In-process backends call :meth:`evaluate` directly. External backends POST
    to :attr:`url`. Both go through the same budget counter.

    Args:
        task: Task definition (name, datasets, objective/background).
        evaluate: Scoring function. Required.
        budget: Budget tracker for limiting evaluations.
        tracker: Optional experiment tracker (wandb/mlflow) for logging.
        max_concurrency: Max parallel evaluations. Defaults to 8.
        output_dir: If set, each eval is persisted as ``<dir>/evals/<i>.json``
            and a rolling ``summary.json`` is written.
    """

    def __init__(
        self,
        task: Task,
        evaluate: EvalFn,
        budget: BudgetTracker,
        *,
        tracker: Any | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        output_dir: str | Path | None = None,
    ) -> None:
        self.task = task
        self.eval_fn: Callable[..., tuple[float, dict[str, Any]]] = evaluate
        self.budget = budget
        self.tracker = tracker
        self.max_concurrency = max_concurrency
        self.best_score: float = float("-inf")
        self.best_candidate: str = task.initial_candidate
        self.total_cost: float = 0.0
        self.eval_log: list[dict[str, Any]] = []
        self._start_time: float = time.time()
        self._best_val_score: float = float("-inf")
        self._progress_log: list[dict[str, Any]] = []
        self._candidate_registry: dict[str, int] = {}
        self._next_candidate_id: int = 0
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self.output_dir: Path | None = None
        self._evals_dir: Path | None = None
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._evals_dir = self.output_dir / "evals"
            self._evals_dir.mkdir(exist_ok=True)
        self._eval_semaphore = threading.Semaphore(max_concurrency)

        self._examples: dict[str, Example] = {}
        for dataset in (task.train_set, task.val_set, task.test_set):
            if dataset:
                for ex in dataset:
                    self._examples[ex.id] = ex

        self._pool = ThreadPoolExecutor(max_workers=max_concurrency)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── Direct Python API (used by in-process backends like GEPA) ───────

    def evaluate(self, candidate: str, example: Example | None = None) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate with budget enforcement.

        Raises:
            BudgetExhausted: When the budget has been used up.
        """
        self._eval_semaphore.acquire()
        try:
            self.budget.check()

            if example is not None:
                score, info = self.eval_fn(candidate, example)
            else:
                score, info = self.eval_fn(candidate)

            self.budget.record(score)
            self._track(candidate, score, info)

            info = dict(info) if info else {}
            info["_budget"] = self.budget.status()
            return score, info
        finally:
            self._eval_semaphore.release()

    def evaluate_examples(
        self,
        candidate: str,
        example_ids: list[str] | None = None,
        split: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate on specific examples or a whole split, in parallel.

        Provide either ``example_ids`` or ``split`` (``"train"``/``"val"``/
        ``"test"``/``"all"``); ``example_ids`` takes precedence. For
        single-task tasks, delegates to :meth:`evaluate`.

        Returns:
            (average_score, info_dict) with per-example scores and metadata.
        """
        if not self.task.has_dataset:
            score, info = self.evaluate(candidate)
            return score, {
                "scores": {"_single": score},
                "infos": {"_single": info},
                "num_evaluated": 1,
                "num_total": 1,
                "partial": False,
                "_budget": self.budget.status(),
            }

        examples: list[Example] = []
        if example_ids is not None:
            for eid in example_ids:
                if eid in self._examples:
                    examples.append(self._examples[eid])
        elif split is not None:
            if split in ("train", "all") and self.task.train_set:
                examples.extend(self.task.train_set)
            if split in ("val", "all") and self.task.val_set:
                examples.extend(self.task.val_set)
            if split in ("test", "all") and self.task.test_set:
                examples.extend(self.task.test_set)
        elif self.task.train_set:
            examples.extend(self.task.train_set)

        if not examples:
            score, info = self.evaluate(candidate)
            return score, {
                "scores": {"_single": score},
                "infos": {"_single": info},
                "num_evaluated": 1,
                "num_total": 1,
                "partial": False,
                "_budget": self.budget.status(),
            }

        if self.budget.remaining is not None and self.budget.remaining < len(examples):
            raise BudgetExhausted(
                f"Not enough budget to evaluate all examples: {self.budget.remaining} remaining, {len(examples)} needed"
            )

        scores: dict[str, float] = {}
        infos: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}

        def _eval_one(ex: Example) -> tuple[str, float, dict[str, Any] | None, str | None]:
            try:
                score, info = self.evaluate(candidate, ex)
                return (ex.id, score, info, None)
            except Exception as e:
                return (ex.id, 0.0, None, str(e))

        futures = {self._pool.submit(_eval_one, ex): ex for ex in examples}
        for future in as_completed(futures):
            eid, score, ex_info, err = future.result()
            if err is not None:
                errors[eid] = err
            else:
                scores[eid] = score
                if ex_info is not None:
                    infos[eid] = ex_info

        all_scores = {**scores, **dict.fromkeys(errors, 0.0)}
        avg = sum(all_scores.values()) / len(all_scores)
        info: dict[str, Any] = {
            "scores": all_scores,
            "infos": infos,
            "num_evaluated": len(examples),
            "_budget": self.budget.status(),
        }
        if errors:
            info["errors"] = errors
        return avg, info

    def validate(self, candidate: str) -> dict[str, Any]:
        """Evaluate ``candidate`` on the hidden ``val_set`` and log progress."""
        if not self.task.val_set:
            raise ValueError("validate() requires a task with val_set")
        val_ids = [ex.id for ex in self.task.val_set]
        avg_score, _ = self.evaluate_examples(candidate, example_ids=val_ids)
        return self.log_progress(avg_score, candidate=candidate)

    def log_progress(
        self, val_score: float, candidate: str | None = None, reflection_cost: float = 0.0
    ) -> dict[str, Any]:
        """Record a progress checkpoint."""
        candidate_id: int | None = None
        if candidate is not None:
            candidate_id = self._register_candidate(candidate)

        with self._lock:
            if val_score > self._best_val_score:
                self._best_val_score = val_score
            entry: dict[str, Any] = {
                "val_score": val_score,
                "best_val_score": self._best_val_score,
                "total_evals": self.budget.used,
                "wall_time": time.time() - self._start_time,
                "total_cost": self.total_cost,
                "reflection_cost": reflection_cost,
            }
            if candidate_id is not None:
                entry["candidate_id"] = candidate_id
            self._progress_log.append(entry)

        if self.output_dir is not None:
            with self._io_lock:
                with open(self.output_dir / "progress_log.jsonl", "a") as f:
                    f.write(json.dumps(entry) + "\n")

        return {"val_score": val_score, "best_val_score": self._best_val_score}

    def _register_candidate(self, candidate: str) -> int:
        if candidate in self._candidate_registry:
            return self._candidate_registry[candidate]
        cid = self._next_candidate_id
        self._next_candidate_id += 1
        self._candidate_registry[candidate] = cid
        if self.output_dir is not None:
            with self._io_lock:
                with open(self.output_dir / "candidates.jsonl", "a") as f:
                    f.write(json.dumps({"candidate_id": cid, "candidate": candidate}) + "\n")
        return cid

    @property
    def progress_log(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._progress_log)

    # ── HTTP server (used by external backends) ─────────────────────────

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("Server not started")
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def start(self, port: int = 0) -> int:
        """Start the HTTP server. Returns the bound port."""
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/evaluate":
                    server_ref._handle_evaluate(self)
                elif self.path == "/evaluate_examples":
                    server_ref._handle_evaluate_examples(self)
                elif self.path == "/validate":
                    server_ref._handle_validate(self)
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
                pass

        # threading.HTTPServer would be better for high concurrency, but the
        # underlying eval is already serialized through ``_eval_semaphore``;
        # we keep the simple single-threaded HTTPServer to avoid surprising
        # ordering, and rely on the thread pool inside evaluate_examples to
        # parallelize batches.
        from http.server import ThreadingHTTPServer

        self._server = ThreadingHTTPServer(("localhost", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        self._pool.shutdown(wait=False)

    # ── Internal ────────────────────────────────────────────────────────

    def _track(self, candidate: str, score: float, info: dict[str, Any] | None = None) -> None:
        cost = float(info.get("cost", 0.0)) if info else 0.0
        with self._lock:
            self.total_cost += cost
            idx = len(self.eval_log)
            entry: dict[str, Any] = {
                "eval": self.budget.used,
                "score": score,
                "candidate_len": len(candidate),
                "wall_time": time.time() - self._start_time,
                "cumulative_cost": self.total_cost,
            }
            if cost:
                entry["cost"] = cost
            self.eval_log.append(entry)
            if score > self.best_score:
                self.best_score = score
                self.best_candidate = candidate
            if self.tracker is not None:
                try:
                    self.tracker.log_eval(self.budget.used, score, self.best_score, cost)
                except Exception:
                    pass
            snapshot = self._snapshot()

        if self.output_dir is not None:
            with self._io_lock:
                self._write_summary(snapshot)
                self._write_eval_record(idx, entry, candidate, info)

    def _snapshot(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "total_evals": self.budget.used,
            "total_cost": self.total_cost,
            "budget": self.budget.status(),
        }

    def _write_summary(self, snapshot: dict[str, Any]) -> None:
        assert self.output_dir is not None
        summary_path = self.output_dir / "summary.json"
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, indent=2, default=str))
        tmp.replace(summary_path)

    def _write_eval_record(self, idx: int, entry: dict[str, Any], candidate: str, info: dict[str, Any] | None) -> None:
        assert self._evals_dir is not None
        record = {**entry, "timestamp": time.time(), "candidate": candidate, "info": info}
        try:
            (self._evals_dir / f"{idx}.json").write_text(json.dumps(record, indent=2, default=str))
        except Exception:
            pass

    def _handle_evaluate(self, handler: BaseHTTPRequestHandler) -> None:
        body = self._read_body(handler)
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

    def _handle_evaluate_examples(self, handler: BaseHTTPRequestHandler) -> None:
        body = self._read_body(handler)
        candidate = body.get("candidate", "")
        example_ids = body.get("example_ids")
        split = body.get("split", "train")

        if self.budget.exhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)
            return

        try:
            avg_score, info = self.evaluate_examples(
                candidate,
                example_ids=example_ids,
                split=split if example_ids is None else None,
            )
            self._send_json(
                handler,
                {
                    "average_score": avg_score,
                    "scores": info.get("scores", {}),
                    "infos": info.get("infos", {}),
                    "num_evaluated": info.get("num_evaluated", 1),
                    "errors": info.get("errors", {}),
                    "budget": self.budget.status(),
                },
            )
        except BudgetExhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)
        except Exception as e:
            self._send_json(handler, {"error": str(e), "budget": self.budget.status()}, status=500)

    def _handle_validate(self, handler: BaseHTTPRequestHandler) -> None:
        body = self._read_body(handler)
        candidate = body.get("candidate", "")
        if not self.task.val_set:
            self._send_json(handler, {"error": "Task has no validation set"}, status=400)
            return
        if self.budget.exhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)
            return
        try:
            result = self.validate(candidate)
            self._send_json(handler, {**result, "budget": self.budget.status()})
        except BudgetExhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)
        except Exception as e:
            self._send_json(handler, {"error": str(e), "budget": self.budget.status()}, status=500)

    def _handle_status(self, handler: BaseHTTPRequestHandler) -> None:
        self._send_json(
            handler,
            {
                "budget": self.budget.status(),
                "task": self.task.name,
                "best_score": self.best_score,
                "total_evals": self.budget.used,
                "total_cost": self.total_cost,
            },
        )

    def _handle_task_info(self, handler: BaseHTTPRequestHandler) -> None:
        self._send_json(
            handler,
            {
                "name": self.task.name,
                "objective": self.task.objective,
                "background": self.task.background,
                "initial_candidate": self.task.initial_candidate,
                "has_dataset": self.task.has_dataset,
                "train_size": len(self.task.train_set) if self.task.train_set else 0,
                "val_size": len(self.task.val_set) if self.task.val_set else 0,
                "test_size": len(self.task.test_set) if self.task.test_set else 0,
                "metadata": {k: v for k, v in self.task.metadata.items() if isinstance(v, str | int | float | bool)},
            },
        )

    @staticmethod
    def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
        content_length = int(handler.headers.get("Content-Length", 0))
        if not content_length:
            return {}
        return json.loads(handler.rfile.read(content_length))

    @staticmethod
    def _send_json(handler: BaseHTTPRequestHandler, data: dict, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
