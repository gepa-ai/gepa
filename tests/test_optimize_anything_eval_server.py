from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from gepa.oa.budget import BudgetExhausted, BudgetTracker
from gepa.oa.eval_server import EvalServer
from gepa.oa.task import Task


class OptimizeAnythingEvalServerTests(unittest.TestCase):
    def test_shared_output_dir_summary_writes_use_independent_temp_files(self) -> None:
        """Composition engines may create independent servers in one run dir."""

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            task = Task(name="task", seed_candidate="seed")
            server_a = EvalServer(
                task,
                lambda candidate: (1.0, {}),
                BudgetTracker(max_evals=1),
                output_dir=output_dir,
            )
            server_b = EvalServer(
                task,
                lambda candidate: (0.0, {}),
                BudgetTracker(max_evals=1),
                output_dir=output_dir,
            )
            barrier = threading.Barrier(2)
            original_replace = Path.replace

            def delayed_replace(self: Path, target: Path) -> Path:
                if self.name.startswith(".summary.") and target.name == "summary.json":
                    barrier.wait(timeout=5)
                return original_replace(self, target)

            try:
                with patch.object(Path, "replace", delayed_replace):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [
                            pool.submit(server_a._write_summary, {"best_score": 1.0}),
                            pool.submit(server_b._write_summary, {"best_score": 0.0}),
                        ]
                        for future in futures:
                            future.result(timeout=5)
            finally:
                server_a.stop()
                server_b.stop()

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertFalse(list(output_dir.glob(".summary.*.tmp")))

    def test_evaluate_batch_is_refused_before_running_when_it_does_not_fit(self) -> None:
        """A grouped stage is reserved in bulk: either the whole batch fits or
        the user's function is never called and nothing is spent (issue #448)."""
        task = Task(name="task", seed_candidate="seed", train_set=[1, 2, 3, 4])
        seen: list[list[int]] = []

        def batch_fn(pairs, opt_states=None):
            seen.append([ex for _c, ex in pairs])
            return [(float(ex), {"ex": ex}) for _c, ex in pairs]

        budget = BudgetTracker(max_evals=3)
        server = EvalServer(task, lambda candidate, example: (0.0, {}), budget, batch_evaluate=batch_fn)
        try:
            with self.assertRaises(BudgetExhausted):
                server.evaluate_batch([("seed", ex) for ex in (1, 2, 3, 4)])
            self.assertEqual(seen, [])
            self.assertEqual((budget.used, budget.reserved), (0, 0))

            out = server.evaluate_batch([("seed", ex) for ex in (1, 2, 3)])
            self.assertEqual([score for score, _info in out], [1.0, 2.0, 3.0])
            self.assertEqual(seen, [[1, 2, 3]])
            self.assertEqual((budget.used, budget.reserved), (3, 0))
            self.assertTrue(budget.exhausted)
            with self.assertRaises(BudgetExhausted):
                server.evaluate_batch([("seed", 1)])
        finally:
            server.stop()

    def test_evaluate_batch_without_batch_fn_runs_pairs_on_the_pool_in_order(self) -> None:
        """With only a per-pair evaluator the bulk reservation still applies:
        the stage is reserved once, then fanned out under max_concurrency."""
        task = Task(name="task", seed_candidate="seed", train_set=[1, 2, 3])
        budget = BudgetTracker(max_evals=3)
        server = EvalServer(
            task, lambda candidate, example: (float(example), {"ex": example}), budget, max_concurrency=2
        )
        try:
            out = server.evaluate_batch([("seed", 3), ("seed", 1), ("seed", 2)])
            self.assertEqual(out, [(3.0, {"ex": 3}), (1.0, {"ex": 1}), (2.0, {"ex": 2})])
            self.assertEqual((budget.used, budget.reserved), (3, 0))
            with self.assertRaises(BudgetExhausted):
                server.evaluate_batch([("seed", 1)])
        finally:
            server.stop()

    def test_evaluate_examples_reserves_the_whole_group(self) -> None:
        task = Task(name="task", seed_candidate="seed", train_set=[1, 2, 3, 4])
        calls: list[int] = []

        def evaluator(candidate, example):
            calls.append(example)
            return 1.0, {}

        budget = BudgetTracker(max_evals=3)
        server = EvalServer(task, evaluator, budget, max_concurrency=4)
        try:
            with self.assertRaises(BudgetExhausted):
                server.evaluate_examples("seed", split="train")
            self.assertEqual(calls, [])
            self.assertEqual((budget.used, budget.reserved), (0, 0))
        finally:
            server.stop()

    def test_concurrent_callers_cannot_overshoot_the_cap(self) -> None:
        """Racing single evals for the last slot: exactly one runs, the others
        are refused before the evaluator is invoked."""
        n = 6
        calls: list[int] = []
        calls_lock = threading.Lock()

        def evaluator(candidate, example):
            with calls_lock:
                calls.append(example)
            time.sleep(0.05)
            return 1.0, {}

        task = Task(name="task", seed_candidate="seed", train_set=list(range(n)))
        budget = BudgetTracker(max_evals=3)
        server = EvalServer(task, evaluator, budget, max_concurrency=n)
        try:
            server.evaluate("seed", 0)
            server.evaluate("seed", 0)  # used = 2, one left
            calls.clear()
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(server.evaluate, "seed", i) for i in range(n)]
                outcomes = []
                for f in futures:
                    try:
                        f.result(timeout=5)
                        outcomes.append("ok")
                    except BudgetExhausted:
                        outcomes.append("refused")
        finally:
            server.stop()

        self.assertEqual(outcomes.count("ok"), 1)
        self.assertEqual(outcomes.count("refused"), n - 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual((budget.used, budget.reserved), (3, 0))

    def test_per_example_info_carries_no_server_bookkeeping(self) -> None:
        """``_budget`` used to be injected into every per-example ``info`` and
        reached the reflection LM through the adapter's reflective dataset.
        Budget status belongs in the ``evaluate_examples`` envelope and
        ``/status`` only."""
        task = Task(name="task", seed_candidate="seed", train_set=[1, 2])
        server = EvalServer(
            task,
            lambda candidate, example: (1.0, {"note": example}),
            BudgetTracker(max_evals=10),
            batch_evaluate=lambda pairs, opt_states=None: [(1.0, {"note": ex}) for _c, ex in pairs],
        )
        try:
            _score, info = server.evaluate("seed", 1)
            self.assertEqual(info, {"note": 1})
            batch = server.evaluate_batch([("seed", 1), ("seed", 2)])
            self.assertEqual([i for _s, i in batch], [{"note": 1}, {"note": 2}])
            _avg, envelope = server.evaluate_examples("seed", split="train")
            self.assertIn("_budget", envelope)
            self.assertEqual(envelope["infos"]["train_0"], {"note": 1})
        finally:
            server.stop()

    def test_http_evaluate_examples_logs_aggregate_progress(self) -> None:
        import urllib.request

        task = Task(
            name="task",
            seed_candidate="seed",
            train_set=["a", "b"],
        )
        server = EvalServer(
            task,
            lambda candidate, example: (1.0 if candidate == "good" and example == "a" else 0.0, {}),
            BudgetTracker(max_evals=2),
            max_concurrency=1,
        )
        server.start()
        try:
            req = urllib.request.Request(
                f"{server.url}/evaluate_examples",
                data=json.dumps({"candidate": "good"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode())
        finally:
            server.stop()

        self.assertEqual(payload["average_score"], 0.5)
        self.assertEqual(len(server.progress_log), 1)
        self.assertEqual(server.progress_log[0]["val_score"], 0.5)
        self.assertIn("candidate_id", server.progress_log[0])

    def test_http_evaluate_examples_does_not_log_partial_progress(self) -> None:
        import urllib.request

        task = Task(
            name="task",
            seed_candidate="seed",
            train_set=["a", "b"],
        )
        server = EvalServer(
            task,
            lambda candidate, example: (1.0, {}),
            BudgetTracker(max_evals=1),
            max_concurrency=1,
        )
        server.start()
        try:
            first_id = server._agent_visible_ids()[0]
            req = urllib.request.Request(
                f"{server.url}/evaluate_examples",
                data=json.dumps({"candidate": "partial", "example_ids": [first_id]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode())
        finally:
            server.stop()

        self.assertEqual(payload["average_score"], 1.0)
        self.assertEqual(server.progress_log, [])


if __name__ == "__main__":
    unittest.main()
