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

    def test_http_rejects_a_closed_evaluation_session(self) -> None:
        import urllib.error
        import urllib.request

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=1))
        session_id = server.open_evaluation_session("seed")
        server.close_evaluation_session(session_id, timeout=0.1)
        server.start()
        try:
            req = urllib.request.Request(
                f"{server.url}/evaluate",
                data=json.dumps({"candidate": "late", "evaluation_session_id": session_id}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(req, timeout=5)
        finally:
            server.stop()

        self.assertEqual(raised.exception.code, 409)
        self.assertEqual(server.best_candidate, "seed")

    def test_http_requires_a_token_while_an_external_engine_is_active(self) -> None:
        import urllib.error
        import urllib.request

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=2))
        session_id = server.open_evaluation_session("seed")
        server.start()
        try:
            req = urllib.request.Request(
                f"{server.url}/evaluate",
                data=json.dumps({"candidate": "untracked"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(req, timeout=5)
            score, _ = server.evaluate("direct")
        finally:
            server.close_evaluation_session(session_id, timeout=0.1)
            server.stop()

        self.assertEqual(raised.exception.code, 409)
        self.assertEqual(score, 1.0)

    def test_close_waits_for_http_aggregate_progress(self) -> None:
        import urllib.request

        task = Task(name="task", seed_candidate="seed", train_set=["example"])
        server = EvalServer(task, lambda candidate, example: (1.0, {}), BudgetTracker(max_evals=1))
        session_id = server.open_evaluation_session("seed")
        admitted = threading.Event()
        release = threading.Event()
        close_done = threading.Event()
        closed: list[tuple[str, float]] = []
        original_register = server._register_candidate

        def block_register(candidate: str) -> int:
            admitted.set()
            self.assertTrue(release.wait(timeout=2))
            return original_register(candidate)

        server._register_candidate = block_register  # type: ignore[method-assign]
        server.start()
        try:
            request = urllib.request.Request(
                f"{server.url}/evaluate_examples",
                data=json.dumps({"candidate": "good", "evaluation_session_id": session_id}).encode(),
                headers={"Content-Type": "application/json"},
            )
            response: list[object] = []
            request_thread = threading.Thread(
                target=lambda: response.append(urllib.request.urlopen(request, timeout=5)), daemon=True
            )
            request_thread.start()
            self.assertTrue(admitted.wait(timeout=2))

            def close_session() -> None:
                closed.append(server.close_evaluation_session(session_id, timeout=2))
                close_done.set()

            close_thread = threading.Thread(target=close_session)
            close_thread.start()
            self.assertFalse(close_done.wait(timeout=0.1))
            release.set()
            request_thread.join(timeout=2)
            close_thread.join(timeout=2)
        finally:
            release.set()
            server.stop()

        self.assertTrue(close_done.is_set())
        self.assertEqual(closed, [("good", 1.0)])
        self.assertEqual(len(server.progress_log), 1)

    def test_close_waits_between_http_evaluation_and_progress(self) -> None:
        import urllib.request

        task = Task(name="task", seed_candidate="seed", train_set=["example"])
        server = EvalServer(task, lambda candidate, example: (1.0, {}), BudgetTracker(max_evals=1))
        session_id = server.open_evaluation_session("seed")
        evaluation_returned = threading.Event()
        release = threading.Event()
        close_done = threading.Event()
        closed: list[tuple[str, float]] = []
        original_evaluate_examples = server.evaluate_examples

        def pause_before_progress(*args: object, **kwargs: object) -> tuple[float, dict[str, object]]:
            result = original_evaluate_examples(*args, **kwargs)
            evaluation_returned.set()
            self.assertTrue(release.wait(timeout=2))
            return result

        server.evaluate_examples = pause_before_progress  # type: ignore[method-assign]
        server.start()
        try:
            request = urllib.request.Request(
                f"{server.url}/evaluate_examples",
                data=json.dumps({"candidate": "good", "evaluation_session_id": session_id}).encode(),
                headers={"Content-Type": "application/json"},
            )
            response: list[object] = []
            request_thread = threading.Thread(
                target=lambda: response.append(urllib.request.urlopen(request, timeout=5)), daemon=True
            )
            request_thread.start()
            self.assertTrue(evaluation_returned.wait(timeout=2))

            def close_session() -> None:
                closed.append(server.close_evaluation_session(session_id, timeout=2))
                close_done.set()

            close_thread = threading.Thread(target=close_session)
            close_thread.start()
            self.assertFalse(close_done.wait(timeout=0.1))
            release.set()
            request_thread.join(timeout=2)
            close_thread.join(timeout=2)
        finally:
            release.set()
            server.stop()

        self.assertTrue(close_done.is_set())
        self.assertEqual(closed, [("good", 1.0)])
        self.assertEqual(len(server.progress_log), 1)

    def test_close_waits_between_http_validation_and_progress(self) -> None:
        import urllib.request

        task = Task(name="task", seed_candidate="seed", val_set=["example"])
        server = EvalServer(task, lambda candidate, example: (1.0, {}), BudgetTracker(max_evals=1))
        session_id = server.open_evaluation_session("seed")
        evaluation_returned = threading.Event()
        release = threading.Event()
        close_done = threading.Event()
        closed: list[tuple[str, float]] = []
        original_evaluate_examples = server.evaluate_examples

        def pause_before_progress(*args: object, **kwargs: object) -> tuple[float, dict[str, object]]:
            result = original_evaluate_examples(*args, **kwargs)
            evaluation_returned.set()
            self.assertTrue(release.wait(timeout=2))
            return result

        server.evaluate_examples = pause_before_progress  # type: ignore[method-assign]
        server.start()
        try:
            request = urllib.request.Request(
                f"{server.url}/validate",
                data=json.dumps({"candidate": "good", "evaluation_session_id": session_id}).encode(),
                headers={"Content-Type": "application/json"},
            )
            response: list[object] = []
            request_thread = threading.Thread(
                target=lambda: response.append(urllib.request.urlopen(request, timeout=5)), daemon=True
            )
            request_thread.start()
            self.assertTrue(evaluation_returned.wait(timeout=2))

            def close_session() -> None:
                closed.append(server.close_evaluation_session(session_id, timeout=2))
                close_done.set()

            close_thread = threading.Thread(target=close_session)
            close_thread.start()
            self.assertFalse(close_done.wait(timeout=0.1))
            release.set()
            request_thread.join(timeout=2)
            close_thread.join(timeout=2)
        finally:
            release.set()
            server.stop()

        self.assertTrue(close_done.is_set())
        self.assertEqual(closed, [("good", 1.0)])
        self.assertEqual(len(server.progress_log), 1)


if __name__ == "__main__":
    unittest.main()
