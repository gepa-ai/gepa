"""Eval server: the single choke point for evaluation, budget, and tracking.

Every engine — in-process or external — goes through the :class:`EvalServer`.
It exposes two interfaces to the same budget-controlled eval:

1. ``server.evaluate(candidate)`` — direct Python call (no HTTP overhead).
2. ``POST server.url/evaluate`` — HTTP endpoint for external/black-box engines.

The api always creates the EvalServer; engines never construct their own.

HTTP Protocol
-------------
POST /evaluate
    Body: {"candidate": "<text>", "example_id": "<optional>"}
    Response: {"score": 1.23, "info": {...}, "budget": {"used": 5, ...}}
    Status 429 when budget exhausted.

POST /evaluate_examples
    Body: {"candidate": "<text>", "example_ids": ["a","b","c"]}
       or {"candidate": "<text>"}                # default: full agent-visible pool
    Evaluates the candidate on the requested examples in parallel,
    respecting the concurrency limit. Each example = 1 budget tick.

    The agent-visible pool is ``train_set + val_set`` (val merged in so
    engines/agents see one combined set).

GET /status   → {"budget": {...}, "task": "...", "best_score": ..., ...}
GET /task     → task metadata (objective/background/seed_candidate/...)
"""

from __future__ import annotations

import functools
import inspect
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from gepa.oa.budget import BudgetExhausted, BudgetTracker
from gepa.oa.task import Task

DEFAULT_MAX_CONCURRENCY = 8


def _resolve_id(item: Any, fallback: str) -> str:
    """Return ``item``'s stable id if it has one, else ``fallback``.

    Recognizes ``item["id"]`` (mapping key) and ``item.id`` (attribute).
    """
    if isinstance(item, Mapping):
        if "id" in item:
            return str(item["id"])
    elif hasattr(item, "id"):
        return str(item.id)
    return fallback


def _normalize_result(result: Any) -> tuple[float, dict[str, Any]]:
    """Normalize one evaluation result to ``(score, info)``.

    Accepts the legacy launcher shapes — a bare ``score``, ``(score, info)``,
    or the 3-tuple ``(score, output, info)`` batch form.
    """
    if isinstance(result, tuple | list):
        items = list(result)
        if not items:
            raise ValueError("evaluator returned an empty tuple; expected score or (score, info)")
        info = items[2] if len(items) > 2 else items[1] if len(items) == 2 else None
        return float(items[0]), dict(info) if info else {}
    return float(result), {}


def _normalize_eval_fn(fn: Callable[..., Any]) -> Callable[..., tuple[float, dict[str, Any]]]:
    """Wrap ``fn`` so every return normalizes to ``(score, info)``.

    Extra kwargs (e.g. ``opt_state``) are forwarded only when ``fn``'s
    signature accepts them (a matching parameter name or ``**kwargs``).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        accepted: set[str] | None = set()  # unintrospectable — drop all extra kwargs
    else:
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        accepted = None if has_var_keyword else set(sig.parameters)

    @functools.wraps(fn)
    def evaluate(*args: Any, **kwargs: Any) -> tuple[float, dict[str, Any]]:
        if accepted is not None:
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return _normalize_result(fn(*args, **kwargs))

    return evaluate


def _normalize_batch_fn(fn: Callable[..., Any]) -> Callable[..., list[tuple[float, dict[str, Any]]]]:
    """Wrap a user ``batch_evaluator`` so one grouped call returns per-pair ``(score, info)``.

    Launcher contract: the function receives ALL ``(candidate, example)`` pairs
    in one call and, when its signature accepts ``opt_states``, the aligned
    per-pair states.
    """
    sig = inspect.signature(fn)
    accepts_opt_states = "opt_states" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    @functools.wraps(fn)
    def evaluate_batch(
        pairs: list[tuple[Any, Any]], opt_states: list[Any] | None = None
    ) -> list[tuple[float, dict[str, Any]]]:
        kwargs: dict[str, Any] = {"opt_states": opt_states} if accepts_opt_states and opt_states is not None else {}
        results = list(fn(list(pairs), **kwargs))
        if len(results) != len(pairs):
            raise ValueError(f"batch_evaluator returned {len(results)} results but expected {len(pairs)}")
        return [_normalize_result(r) for r in results]

    return evaluate_batch


class EvalServer:
    """Eval server with budget enforcement — the single source of truth.

    In-process engines call :meth:`evaluate` directly. External engines POST
    to :attr:`url`. Both go through the same budget counter.

    Args:
        task: Task definition (name, datasets, objective/background).
        evaluate: Per-pair scoring function. Optional when ``batch_evaluate``
            is provided (at least one is required); single-pair evaluations
            then route through it as singleton batches.
        budget: Budget tracker for limiting evaluations.
        batch_evaluate: Optional grouped scoring function taking a list of
            ``(candidate, example)`` pairs and returning one result per pair.
            Multi-pair evaluations prefer it over per-pair fan-out.
        max_concurrency: Max parallel evaluations. Defaults to 8.
        output_dir: If set, each eval is persisted as ``<dir>/evals/<i>.json``
            and a rolling ``summary.json`` is written.
    """

    def __init__(
        self,
        task: Task,
        evaluate: Callable[..., Any] | None,
        budget: BudgetTracker,
        *,
        batch_evaluate: Callable[..., Any] | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        output_dir: str | Path | None = None,
    ) -> None:
        self.task = task
        self.batch_fn = _normalize_batch_fn(batch_evaluate) if batch_evaluate is not None else None
        if evaluate is None:
            if self.batch_fn is None:
                raise ValueError("Provide evaluator=, batch_evaluator=, or both.")
            batch_fn = self.batch_fn

            def _singleton(
                candidate: Any, example: Any | None = None, opt_state: Any | None = None
            ) -> tuple[float, dict[str, Any]]:
                # Launcher parity: with only a batch_evaluator, single-pair
                # evaluations route through it as singleton batches.
                return batch_fn([(candidate, example)], opt_states=[opt_state] if opt_state is not None else None)[0]

            evaluate = _singleton

        self.eval_fn: Callable[..., tuple[float, dict[str, Any]]] = _normalize_eval_fn(evaluate)
        self.budget = budget
        self.max_concurrency = max_concurrency
        self.best_score: float = float("-inf")
        # A dict seed is a legal multi-component candidate (see ``_track``,
        # which already reads/stores dict candidates), so the tracked best is
        # ``str | dict``, not ``str``.
        self.best_candidate: str | dict[str, str] = task.seed_candidate or ""
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

        # The eval server holds only the agent-visible pool (train + val). The
        # held-out test_set is never registered here: the optimize_anything
        # runner scores it directly via ``eval_fn``, so test data is unreachable
        # from any engine, agent, or HTTP endpoint by construction.
        self._examples: dict[str, Any] = {}
        self._split_ids: dict[str, list[str]] = {"train": [], "val": []}
        for split_name, dataset in (
            ("train", task.train_set),
            ("val", task.val_set),
        ):
            if not dataset:
                continue
            for i, item in enumerate(dataset):
                eid = _resolve_id(item, f"{split_name}_{i}")
                if eid in self._examples and self._examples[eid] is not item:
                    eid = f"{split_name}_{i}"
                self._examples[eid] = item
                self._split_ids[split_name].append(eid)

        self._pool = ThreadPoolExecutor(max_workers=max_concurrency)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── Direct Python API (used by in-process engines like GEPA) ───────

    def iter_split(self, split: str) -> Iterator[tuple[str, Any]]:
        """Yield ``(id, item)`` pairs for the given split (``train``/``val``).

        The server holds no ``test`` split, so ``iter_split("test")`` yields
        nothing; test is scored by the optimize_anything runner directly.
        """
        for eid in self._split_ids.get(split, []):
            yield eid, self._examples[eid]

    def evaluate(self, candidate: str, example: Any | None = None, **kwargs: Any) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate with budget enforcement.

        Failed attempts consume budget (launcher parity). Extra kwargs (e.g.
        ``opt_state``) are forwarded to the evaluator only if its signature
        accepts them.

        Raises:
            BudgetExhausted: When the budget has been used up.
        """
        self._eval_semaphore.acquire()
        try:
            self.budget.check()

            try:
                if example is not None:
                    score, info = self.eval_fn(candidate, example, **kwargs)
                else:
                    score, info = self.eval_fn(candidate, **kwargs)
            except Exception:
                # check() passed above, so this record cannot itself raise.
                self.budget.record(0.0)
                raise

            self.budget.record(score)
            self._track(candidate, score, info)

            info = dict(info) if info else {}
            info["_budget"] = self.budget.status()
            return score, info
        finally:
            self._eval_semaphore.release()

    def evaluate_batch(
        self,
        pairs: list[tuple[str, Any]],
        opt_states: list[Any] | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Evaluate ``pairs`` in ONE call to the user's ``batch_evaluate`` function.

        The grouped analogue of :meth:`evaluate` (e.g. to submit a provider
        batch job): the user function receives all pairs at once; each pair is
        then recorded against the budget and tracked individually. The budget
        is checked once up front, so a batch may overshoot the cap by at most
        ``len(pairs) - 1`` — mirroring the launcher's iteration-boundary stops.
        Failed attempts consume budget (launcher parity): a failed grouped
        call records one tick per pair.

        Raises:
            BudgetExhausted: When the budget is already used up.
            RuntimeError: When the server was constructed without ``batch_evaluate``.
        """
        if self.batch_fn is None:
            raise RuntimeError("evaluate_batch requires the server to be constructed with batch_evaluate")
        self._eval_semaphore.acquire()
        try:
            self.budget.check()
            try:
                results = self.batch_fn(pairs, opt_states=opt_states)
            except Exception:
                # Guard each record so crossing the cap while recording
                # failures never masks the user's original exception.
                for _ in pairs:
                    try:
                        self.budget.record(0.0)
                    except BudgetExhausted:
                        break
                raise
        finally:
            self._eval_semaphore.release()
        out: list[tuple[float, dict[str, Any]]] = []
        for (candidate, _example), (score, info) in zip(pairs, results, strict=True):
            self.budget.record(score)
            self._track(candidate, score, info)
            info = dict(info)
            info["_budget"] = self.budget.status()
            out.append((score, info))
        return out

    def evaluate_examples(
        self,
        candidate: str,
        example_ids: list[str] | None = None,
        split: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate on specific examples or a whole split, in parallel.

        Provide either ``example_ids`` or ``split`` (``"train"``/``"val"``/
        ``"all"``); ``example_ids`` takes precedence. The server holds no
        ``test`` split, so ``"all"`` means train+val. For single-task tasks,
        delegates to :meth:`evaluate`.

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

        examples: list[tuple[str, Any]] = []
        if example_ids is not None:
            for eid in example_ids:
                if eid in self._examples:
                    examples.append((eid, self._examples[eid]))
        elif split is not None:
            splits = ("train", "val") if split == "all" else (split,)
            for s in splits:
                examples.extend(self.iter_split(s))
        elif self.task.train_set:
            examples.extend(self.iter_split("train"))

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

        grouped_done = False
        if self.batch_fn is not None:
            # Multi-pair evaluations prefer the grouped path (launcher parity):
            # the user's batch function sees all pairs in one call.
            try:
                results = self.evaluate_batch([(candidate, ex) for _eid, ex in examples])
            except BudgetExhausted:
                raise
            except Exception:
                # A failed grouped call must not zero the whole stage: fall
                # through to per-pair evaluation so one bad pair costs one 0.0,
                # matching the per-pair path's isolation. Both attempts consume
                # budget — each was a real eval call.
                pass
            else:
                grouped_done = True
                for (eid, _ex), (score, ex_info) in zip(examples, results, strict=True):
                    scores[eid] = score
                    infos[eid] = ex_info
        if not grouped_done:

            def _eval_one(eid: str, ex: Any) -> tuple[str, float, dict[str, Any] | None, str | None]:
                try:
                    score, info = self.evaluate(candidate, ex)
                    return (eid, score, info, None)
                except Exception as e:
                    return (eid, 0.0, None, str(e))

            futures = {self._pool.submit(_eval_one, eid, ex): eid for eid, ex in examples}
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
        avg_score, _ = self.evaluate_examples(candidate, example_ids=list(self._split_ids["val"]))
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
        # Tolerate non-str candidates (e.g. a legacy dict) — key by a stable dump.
        key = candidate if isinstance(candidate, str) else json.dumps(candidate, sort_keys=True)
        if key in self._candidate_registry:
            return self._candidate_registry[key]
        cid = self._next_candidate_id
        self._next_candidate_id += 1
        self._candidate_registry[key] = cid
        if self.output_dir is not None:
            with self._io_lock:
                with open(self.output_dir / "candidates.jsonl", "a") as f:
                    f.write(json.dumps({"candidate_id": cid, "candidate": candidate}) + "\n")
        return cid

    @property
    def progress_log(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._progress_log)

    # ── HTTP server (used by external engines) ─────────────────────────

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
                "candidate_len": len(candidate) if isinstance(candidate, str) else sum(map(len, candidate.values())),
                "wall_time": time.time() - self._start_time,
                "cumulative_cost": self.total_cost,
            }
            if cost:
                entry["cost"] = cost
            self.eval_log.append(entry)
            if score > self.best_score:
                self.best_score = score
                self.best_candidate = candidate
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
        tmp = summary_path.with_name(f".summary.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(snapshot, indent=2, default=str))
            tmp.replace(summary_path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _write_eval_record(self, idx: int, entry: dict[str, Any], candidate: str, info: dict[str, Any] | None) -> None:
        assert self._evals_dir is not None
        record = {**entry, "timestamp": time.time(), "candidate": candidate, "info": info}
        try:
            (self._evals_dir / f"{idx}.json").write_text(json.dumps(record, indent=2, default=str))
        except Exception:
            pass

    def _agent_visible_ids(self) -> list[str]:
        """Train + val ids combined. Agents see one merged set; test is held outside the server."""
        return list(self._split_ids["train"]) + list(self._split_ids["val"])

    def _handle_evaluate(self, handler: BaseHTTPRequestHandler) -> None:
        body = self._read_body(handler)
        candidate = body.get("candidate", "")
        example_id = body.get("example_id")

        if self.budget.exhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)
            return

        # Reject any example_id outside the agent-visible pool (train+val).
        # Single-task tasks have no examples, so example_id is ignored there.
        if example_id is not None and self.task.has_dataset:
            if example_id not in set(self._agent_visible_ids()):
                self._send_json(
                    handler,
                    {"error": f"Unknown or sealed example_id: {example_id!r}"},
                    status=400,
                )
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

        if self.budget.exhausted:
            self._send_json(handler, {"error": "Budget exhausted", "budget": self.budget.status()}, status=429)
            return

        # The HTTP endpoint speaks only the agent-visible pool (train+val).
        # ``split`` is no longer a parameter — agents either default to the
        # whole visible pool or pass explicit ``example_ids`` from it. Any
        # example_id outside the visible pool is rejected; the test set is
        # not even registered in the server, so it cannot be probed at all.
        visible = set(self._agent_visible_ids())
        if example_ids is not None:
            invalid = [eid for eid in example_ids if eid not in visible]
            if invalid:
                self._send_json(
                    handler,
                    {"error": f"Unknown or sealed example_ids: {invalid[:5]}"},
                    status=400,
                )
                return
            target_ids: list[str] = list(example_ids)
        else:
            target_ids = self._agent_visible_ids()

        try:
            avg_score, info = self.evaluate_examples(candidate, example_ids=target_ids)
            if set(target_ids) == visible:
                self.log_progress(avg_score, candidate=candidate)
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
        # Report only the agent-visible pool. train_size = train+val combined;
        # val_size/test_size are zeroed so an HTTP caller cannot infer that
        # any held-out split exists.
        visible_size = len(self._agent_visible_ids()) if self.task.has_dataset else 0
        self._send_json(
            handler,
            {
                "objective": self.task.objective,
                "background": self.task.background,
                "seed_candidate": self.task.seed_candidate,
                "has_dataset": self.task.has_dataset,
                "train_size": visible_size,
                "val_size": 0,
                "test_size": 0,
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
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except BrokenPipeError:
            # Clients may disconnect after their own timeout or after budget
            # exhaustion while the server is finishing in-flight evals. Treat
            # that as a closed response, not as an eval-server failure.
            return
