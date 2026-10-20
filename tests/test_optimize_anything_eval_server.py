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
            attempted = threading.local()

            def delayed_replace(self: Path, target: Path) -> Path:
                if (
                    self.name.startswith(".summary.")
                    and target.name == "summary.json"
                    and not getattr(attempted, "seen", False)
                ):
                    attempted.seen = True
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

    def test_summary_retries_only_windows_access_conflicts(self) -> None:
        for winerror in (5, 32, 33, None):
            with self.subTest(winerror=winerror), tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)
                server = EvalServer(
                    Task(name="task", seed_candidate="seed"),
                    lambda candidate: (1.0, {}),
                    BudgetTracker(max_evals=1),
                    output_dir=output_dir,
                )
                error = PermissionError("access denied")
                if winerror is not None:
                    error.winerror = winerror
                original_replace = Path.replace
                calls = []

                def replace_once(source, target):
                    calls.append(source)
                    if len(calls) == 1:
                        raise error
                    return original_replace(source, target)

                try:
                    with patch.object(Path, "replace", replace_once), patch("gepa.oa.eval_server.time.sleep") as sleep:
                        if winerror is None:
                            with self.assertRaises(PermissionError):
                                server._write_summary({"best_score": 1.0})
                            sleep.assert_not_called()
                            self.assertEqual(len(calls), 1)
                        else:
                            server._write_summary({"best_score": 1.0})
                            self.assertEqual(len(calls), 2)
                            self.assertEqual(json.loads((output_dir / "summary.json").read_text()), {"best_score": 1.0})
                            sleep.assert_called_once_with(0.01)
                    self.assertFalse(list(output_dir.glob(".summary.*.tmp")))
                finally:
                    server.stop()

    def test_summary_persistent_access_error_is_bounded_and_preserves_previous_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summary = output_dir / "summary.json"
            summary.write_text('{"best_score": 0.5}')
            server = EvalServer(
                Task(name="task", seed_candidate="seed"),
                lambda candidate: (1.0, {}),
                BudgetTracker(max_evals=1),
                output_dir=output_dir,
            )
            error = PermissionError("access denied")
            error.winerror = 5
            try:
                with (
                    patch.object(Path, "replace", side_effect=error) as replace,
                    patch("gepa.oa.eval_server.time.sleep") as sleep,
                ):
                    with self.assertRaises(PermissionError):
                        server._write_summary({"best_score": 1.0})
                    self.assertEqual(replace.call_count, 5)
                    self.assertEqual(sleep.call_count, 4)
                self.assertEqual(json.loads(summary.read_text()), {"best_score": 0.5})
                self.assertFalse(list(output_dir.glob(".summary.*.tmp")))
            finally:
                server.stop()

    def test_summary_write_failure_warns_without_failing_the_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            server = EvalServer(
                Task(name="task", seed_candidate="seed"),
                lambda candidate: (1.0, {}),
                BudgetTracker(max_evals=1),
                output_dir=output_dir,
            )
            try:
                with (
                    patch.object(Path, "replace", side_effect=PermissionError("access denied")),
                    self.assertWarnsRegex(RuntimeWarning, "Could not update summary.json"),
                ):
                    score, _ = server.evaluate("candidate")
                self.assertEqual(score, 1.0)
                self.assertEqual(server.best_score, 1.0)
                self.assertEqual(server.budget.used, 1)
                # The per-eval record is still written after the summary write fails.
                self.assertTrue((output_dir / "evals" / "0.json").exists())
                self.assertFalse(list(output_dir.glob(".summary.*.tmp")))
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
