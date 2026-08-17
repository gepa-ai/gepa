from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from gepa.oa.budget import BudgetTracker
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

    def test_wait_idle_blocks_until_in_flight_evaluate_finishes(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def evaluate(_candidate: str) -> tuple[float, dict[str, object]]:
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return 0.9, {}

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, evaluate, BudgetTracker(max_evals=2))
        idle_done = threading.Event()

        def run_eval() -> None:
            server.evaluate("winner")

        def wait() -> None:
            server.wait_idle()
            idle_done.set()

        eval_thread = threading.Thread(target=run_eval)
        wait_thread = threading.Thread(target=wait)
        eval_thread.start()
        self.assertTrue(started.wait(timeout=2))
        wait_thread.start()
        self.assertFalse(idle_done.wait(timeout=0.1))
        release.set()
        eval_thread.join(timeout=2)
        wait_thread.join(timeout=2)
        self.assertTrue(idle_done.is_set())
        self.assertEqual(server.best_candidate, "winner")
        self.assertEqual(server.best_score, 0.9)

    def test_wait_idle_waits_for_queued_evaluate_examples(self) -> None:
        started = threading.Event()
        release = threading.Event()
        seen: list[str] = []

        def evaluate(_candidate: str, example: object) -> tuple[float, dict[str, object]]:
            seen.append(str(example))
            if str(example) == "a":
                started.set()
                self.assertTrue(release.wait(timeout=2))
            return 1.0, {}

        task = Task(name="task", seed_candidate="seed", train_set=["a", "b", "c"])
        server = EvalServer(task, evaluate, BudgetTracker(max_evals=10), max_concurrency=1)
        idle_done = threading.Event()
        outcome: list[tuple[float, dict[str, object]]] = []

        def run_eval() -> None:
            outcome.append(server.evaluate_examples("good"))

        def wait() -> None:
            server.wait_idle()
            idle_done.set()

        eval_thread = threading.Thread(target=run_eval)
        wait_thread = threading.Thread(target=wait)
        eval_thread.start()
        self.assertTrue(started.wait(timeout=2))
        wait_thread.start()
        self.assertFalse(idle_done.wait(timeout=0.1))
        release.set()
        eval_thread.join(timeout=2)
        wait_thread.join(timeout=2)
        self.assertTrue(idle_done.is_set())
        self.assertEqual(len(outcome), 1)
        self.assertEqual(outcome[0][0], 1.0)
        self.assertEqual(sorted(seen), ["a", "b", "c"])

    def test_pause_http_rejects_new_requests_and_keeps_python_evaluate(self) -> None:
        import urllib.error
        import urllib.request

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=4))
        server.start()
        try:
            server.pause_http()
            req = urllib.request.Request(
                f"{server.url}/evaluate",
                data=json.dumps({"candidate": "late"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(raised.exception.code, 409)
            score, _ = server.evaluate("direct")
            self.assertEqual(score, 1.0)
            server.resume_http()
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
