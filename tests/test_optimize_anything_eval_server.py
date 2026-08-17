from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from gepa.oa.budget import BudgetTracker
from gepa.oa.eval_server import EvalServer, EvaluationSessionClosedError, EvaluationSessionResult
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
        closed: list[EvaluationSessionResult] = []
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
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].best_candidate, "good")
        self.assertEqual(closed[0].best_score, 1.0)
        self.assertEqual(closed[0].aggregate_candidate, "good")
        self.assertEqual(closed[0].aggregate_score, 1.0)
        self.assertEqual(len(server.progress_log), 1)

    def test_close_waits_between_http_evaluation_and_progress(self) -> None:
        import urllib.request

        task = Task(name="task", seed_candidate="seed", train_set=["example"])
        server = EvalServer(task, lambda candidate, example: (1.0, {}), BudgetTracker(max_evals=1))
        session_id = server.open_evaluation_session("seed")
        evaluation_returned = threading.Event()
        release = threading.Event()
        close_done = threading.Event()
        closed: list[EvaluationSessionResult] = []
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
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].best_candidate, "good")
        self.assertEqual(closed[0].best_score, 1.0)
        self.assertEqual(closed[0].aggregate_candidate, "good")
        self.assertEqual(closed[0].aggregate_score, 1.0)
        self.assertEqual(len(server.progress_log), 1)

    def test_close_waits_between_http_validation_and_progress(self) -> None:
        import urllib.request

        task = Task(name="task", seed_candidate="seed", val_set=["example"])
        server = EvalServer(task, lambda candidate, example: (1.0, {}), BudgetTracker(max_evals=1))
        session_id = server.open_evaluation_session("seed")
        evaluation_returned = threading.Event()
        release = threading.Event()
        close_done = threading.Event()
        closed: list[EvaluationSessionResult] = []
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
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].best_candidate, "good")
        self.assertEqual(closed[0].best_score, 1.0)
        self.assertEqual(closed[0].aggregate_candidate, "good")
        self.assertEqual(closed[0].aggregate_score, 1.0)
        self.assertEqual(len(server.progress_log), 1)

    def test_close_does_not_zero_queued_nested_evaluate_examples(self) -> None:
        started = threading.Event()
        release = threading.Event()
        close_done = threading.Event()
        closed: list[EvaluationSessionResult] = []
        seen: list[str] = []

        def evaluate(_candidate: str, example: object) -> tuple[float, dict[str, object]]:
            seen.append(str(example))
            if str(example) == "a":
                started.set()
                self.assertTrue(release.wait(timeout=2))
            return 1.0, {}

        task = Task(name="task", seed_candidate="seed", train_set=["a", "b", "c"])
        server = EvalServer(task, evaluate, BudgetTracker(max_evals=10), max_concurrency=1)
        session_id = server.open_evaluation_session("seed")
        outcome: list[tuple[float, dict[str, object]]] = []

        def run_eval() -> None:
            outcome.append(server.evaluate_examples("good", evaluation_session_id=session_id))

        try:
            eval_thread = threading.Thread(target=run_eval)
            eval_thread.start()
            self.assertTrue(started.wait(timeout=2))

            def close_session() -> None:
                closed.append(server.close_evaluation_session(session_id, timeout=2))
                close_done.set()

            close_thread = threading.Thread(target=close_session)
            close_thread.start()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with server._idle:
                    session = server._evaluation_sessions.get(session_id)
                    if session is not None and session.closed:
                        break
                time.sleep(0.01)
            else:
                self.fail("session did not close while the first example was blocked")
            release.set()
            eval_thread.join(timeout=2)
            close_thread.join(timeout=2)
        finally:
            release.set()

        self.assertTrue(close_done.is_set())
        self.assertEqual(len(outcome), 1)
        avg, info = outcome[0]
        self.assertEqual(avg, 1.0)
        scores = info["scores"]
        assert isinstance(scores, dict)
        self.assertEqual(sorted(scores.values()), [1.0, 1.0, 1.0])
        self.assertNotIn("errors", info)
        self.assertEqual(sorted(seen), ["a", "b", "c"])
        self.assertEqual(closed[0].best_candidate, "good")
        self.assertEqual(closed[0].best_score, 1.0)
        self.assertEqual(closed[0].aggregate_candidate, "good")
        self.assertEqual(closed[0].aggregate_score, 1.0)

    def test_http_close_does_not_zero_queued_evaluate_examples(self) -> None:
        import urllib.request

        started = threading.Event()
        release = threading.Event()
        close_done = threading.Event()
        closed: list[EvaluationSessionResult] = []

        def evaluate(_candidate: str, example: object) -> tuple[float, dict[str, object]]:
            if str(example) == "a":
                started.set()
                self.assertTrue(release.wait(timeout=2))
            return 1.0, {}

        task = Task(name="task", seed_candidate="seed", train_set=["a", "b", "c"])
        server = EvalServer(task, evaluate, BudgetTracker(max_evals=10), max_concurrency=1)
        session_id = server.open_evaluation_session("seed")
        server.start()
        try:
            request = urllib.request.Request(
                f"{server.url}/evaluate_examples",
                data=json.dumps({"candidate": "good", "evaluation_session_id": session_id}).encode(),
                headers={"Content-Type": "application/json"},
            )
            payload: list[dict[str, object]] = []
            request_thread = threading.Thread(
                target=lambda: payload.append(json.loads(urllib.request.urlopen(request, timeout=5).read().decode())),
                daemon=True,
            )
            request_thread.start()
            self.assertTrue(started.wait(timeout=2))

            def close_session() -> None:
                closed.append(server.close_evaluation_session(session_id, timeout=2))
                close_done.set()

            close_thread = threading.Thread(target=close_session)
            close_thread.start()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with server._idle:
                    session = server._evaluation_sessions.get(session_id)
                    if session is not None and session.closed:
                        break
                time.sleep(0.01)
            else:
                self.fail("session did not close while the first example was blocked")
            release.set()
            request_thread.join(timeout=2)
            close_thread.join(timeout=2)
        finally:
            release.set()
            server.stop()

        self.assertTrue(close_done.is_set())
        self.assertEqual(payload[0]["average_score"], 1.0)
        self.assertEqual(payload[0]["errors"], {})
        self.assertEqual(closed[0].best_candidate, "good")
        self.assertEqual(closed[0].best_score, 1.0)
        self.assertEqual(closed[0].aggregate_candidate, "good")
        self.assertEqual(closed[0].aggregate_score, 1.0)
        self.assertEqual(server.evaluation_session_aggregate(session_id), ("good", 1.0))

    def test_http_rejects_an_unknown_token_while_an_external_engine_is_active(self) -> None:
        import urllib.error
        import urllib.request

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=1))
        session_id = server.open_evaluation_session("seed")
        server.start()
        try:
            req = urllib.request.Request(
                f"{server.url}/evaluate",
                data=json.dumps({"candidate": "forged", "evaluation_session_id": "not-a-session"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(req, timeout=5)
        finally:
            server.close_evaluation_session(session_id, timeout=0.1)
            server.stop()

        self.assertEqual(raised.exception.code, 409)
        self.assertEqual(server.best_candidate, "seed")

    def test_in_process_evaluate_rejects_a_completed_session(self) -> None:
        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=1))
        session_id = server.open_evaluation_session("seed")
        server.close_evaluation_session(session_id, timeout=0.1)
        with self.assertRaises(EvaluationSessionClosedError):
            server.evaluate("late", evaluation_session_id=session_id)

    def test_http_requires_a_token_while_a_session_is_draining(self) -> None:
        import urllib.error
        import urllib.request

        started = threading.Event()
        release = threading.Event()

        def evaluate(_candidate: str) -> tuple[float, dict[str, object]]:
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return 1.0, {}

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, evaluate, BudgetTracker(max_evals=4), max_concurrency=1)
        session_id = server.open_evaluation_session("seed")
        server.start()
        tokenless_status: list[int] = []

        def run_eval() -> None:
            server.evaluate("good", evaluation_session_id=session_id)

        try:
            eval_thread = threading.Thread(target=run_eval)
            eval_thread.start()
            self.assertTrue(started.wait(timeout=2))

            close_thread = threading.Thread(target=lambda: server.close_evaluation_session(session_id, timeout=2))
            close_thread.start()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with server._idle:
                    session = server._evaluation_sessions.get(session_id)
                    if session is not None and session.closed:
                        break
                time.sleep(0.01)
            else:
                self.fail("session did not close while evaluation was blocked")

            req = urllib.request.Request(
                f"{server.url}/evaluate",
                data=json.dumps({"candidate": "tokenless"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                tokenless_status.append(200)
            except urllib.error.HTTPError as e:
                tokenless_status.append(e.code)

            release.set()
            eval_thread.join(timeout=2)
            close_thread.join(timeout=2)

            after = urllib.request.Request(
                f"{server.url}/evaluate",
                data=json.dumps({"candidate": "after-drain"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(after, timeout=5) as resp:
                tokenless_status.append(resp.status)
        finally:
            release.set()
            server.stop()

        self.assertEqual(tokenless_status, [409, 200])
        self.assertEqual(server.best_candidate, "good")

    def test_concurrent_close_of_the_same_session_is_safe(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def evaluate(_candidate: str) -> tuple[float, dict[str, object]]:
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return 0.5, {}

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, evaluate, BudgetTracker(max_evals=2), max_concurrency=1)
        session_id = server.open_evaluation_session("seed")
        results: list[EvaluationSessionResult] = []
        errors: list[BaseException] = []

        def run_eval() -> None:
            server.evaluate("winner", evaluation_session_id=session_id)

        def closer() -> None:
            try:
                results.append(server.close_evaluation_session(session_id, timeout=2))
            except BaseException as e:
                errors.append(e)

        eval_thread = threading.Thread(target=run_eval)
        eval_thread.start()
        self.assertTrue(started.wait(timeout=2))
        close_threads = [threading.Thread(target=closer) for _ in range(2)]
        for thread in close_threads:
            thread.start()
        release.set()
        eval_thread.join(timeout=2)
        for thread in close_threads:
            thread.join(timeout=2)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual({(item.best_candidate, item.best_score) for item in results}, {("winner", 0.5)})

    def test_completed_sessions_are_bounded(self) -> None:
        from gepa.oa.eval_server import _MAX_COMPLETED_EVALUATION_SESSIONS

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=100))
        for i in range(_MAX_COMPLETED_EVALUATION_SESSIONS + 5):
            session_id = server.open_evaluation_session("seed")
            server.evaluate(f"c{i}", evaluation_session_id=session_id)
            server.close_evaluation_session(session_id, timeout=0.1)
        self.assertLessEqual(len(server._completed_sessions), _MAX_COMPLETED_EVALUATION_SESSIONS)

    def test_python_evaluate_examples_records_full_pool_aggregate(self) -> None:
        task = Task(name="task", seed_candidate="seed", train_set=["a", "b"])
        server = EvalServer(task, lambda _candidate, _example: (0.5, {}), BudgetTracker(max_evals=4))
        session_id = server.open_evaluation_session("seed")
        avg, _info = server.evaluate_examples("steady", evaluation_session_id=session_id)
        result = server.close_evaluation_session(session_id, timeout=0.1)
        self.assertEqual(avg, 0.5)
        self.assertEqual(result.aggregate_candidate, "steady")
        self.assertEqual(result.aggregate_score, 0.5)

    def test_python_subset_evaluate_examples_does_not_record_aggregate(self) -> None:
        task = Task(name="task", seed_candidate="seed", train_set=["a", "b"])
        server = EvalServer(task, lambda _candidate, _example: (1.0, {}), BudgetTracker(max_evals=4))
        session_id = server.open_evaluation_session("seed")
        first_id = server._agent_visible_ids()[0]
        server.evaluate_examples("partial", example_ids=[first_id], evaluation_session_id=session_id)
        result = server.close_evaluation_session(session_id, timeout=0.1)
        self.assertIsNone(result.aggregate_candidate)
        self.assertEqual(result.best_candidate, "partial")

    def test_validate_does_not_record_session_aggregate_when_train_exists(self) -> None:
        task = Task(name="task", seed_candidate="seed", train_set=["a"], val_set=["b"])
        server = EvalServer(task, lambda _candidate, _example: (1.0, {}), BudgetTracker(max_evals=4))
        session_id = server.open_evaluation_session("seed")
        server.validate("val-only", evaluation_session_id=session_id)
        result = server.close_evaluation_session(session_id, timeout=0.1)
        self.assertIsNone(result.aggregate_candidate)
        self.assertEqual(result.best_candidate, "val-only")

    def test_drain_timeout_retires_session_and_allows_tokenless_http(self) -> None:
        import urllib.error
        import urllib.request

        started = threading.Event()
        release = threading.Event()

        def evaluate(candidate: str) -> tuple[float, dict[str, object]]:
            if candidate == "hung":
                started.set()
                self.assertTrue(release.wait(timeout=2))
            return 1.0, {}

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, evaluate, BudgetTracker(max_evals=4), max_concurrency=2)
        session_id = server.open_evaluation_session("seed")
        server.start()
        eval_thread = threading.Thread(target=lambda: server.evaluate("hung", evaluation_session_id=session_id))
        try:
            eval_thread.start()
            self.assertTrue(started.wait(timeout=2))
            with self.assertRaises(TimeoutError):
                server.close_evaluation_session(session_id, timeout=0.05)
            self.assertNotIn(session_id, server._evaluation_sessions)

            tokenless = urllib.request.Request(
                f"{server.url}/evaluate",
                data=json.dumps({"candidate": "tokenless"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(tokenless, timeout=5) as resp:
                self.assertEqual(resp.status, 200)

            stale = urllib.request.Request(
                f"{server.url}/evaluate",
                data=json.dumps({"candidate": "stale", "evaluation_session_id": session_id}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(stale, timeout=5)
            self.assertEqual(raised.exception.code, 409)
        finally:
            release.set()
            eval_thread.join(timeout=2)
            server.stop()

    def test_stop_retires_live_sessions(self) -> None:
        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=1))
        session_id = server.open_evaluation_session("seed")
        server.stop()
        self.assertNotIn(session_id, server._evaluation_sessions)
        with self.assertRaises(EvaluationSessionClosedError):
            server.evaluate("late", evaluation_session_id=session_id)


if __name__ == "__main__":
    unittest.main()
