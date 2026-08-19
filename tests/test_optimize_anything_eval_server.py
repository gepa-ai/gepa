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

    def test_wait_idle_timeout_returns_false_while_eval_blocked(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def evaluate(_candidate: str) -> tuple[float, dict[str, object]]:
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return 0.9, {}

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, evaluate, BudgetTracker(max_evals=2))
        eval_thread = threading.Thread(target=lambda: server.evaluate("slow"))
        eval_thread.start()
        self.assertTrue(started.wait(timeout=2))
        try:
            self.assertFalse(server.wait_idle(timeout=0.15))
        finally:
            release.set()
            eval_thread.join(timeout=2)
        self.assertTrue(server.wait_idle(timeout=2))

    def test_drain_http_times_out_and_pauses(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def evaluate(_candidate: str) -> tuple[float, dict[str, object]]:
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return 0.9, {}

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, evaluate, BudgetTracker(max_evals=2))
        eval_thread = threading.Thread(target=lambda: server.evaluate("slow"))
        eval_thread.start()
        self.assertTrue(started.wait(timeout=2))
        try:
            started_at = time.monotonic()
            ok = server.drain_http(timeout=0.2, quiet=0.0)
            self.assertFalse(ok)
            self.assertLess(time.monotonic() - started_at, 1.0)
            self.assertFalse(server._http_accepting)
        finally:
            release.set()
            eval_thread.join(timeout=2)
        server.resume_http()

    def test_drain_http_waits_for_request_blocked_in_body_read(self) -> None:
        import urllib.request

        started = threading.Event()
        release = threading.Event()
        original_read = EvalServer._read_body

        def blocked_read(handler: object) -> dict[str, object]:
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return original_read(handler)

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(
            task,
            lambda candidate: (0.9 if candidate == "in-transit" else 0.0, {}),
            BudgetTracker(max_evals=4),
        )
        server.start()
        outcome: dict[str, object] = {}
        try:
            with patch.object(EvalServer, "_read_body", staticmethod(blocked_read)):

                def do_http() -> None:
                    req = urllib.request.Request(
                        f"{server.url}/evaluate",
                        data=json.dumps({"candidate": "in-transit"}).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        outcome["status"] = resp.status
                        outcome["body"] = json.loads(resp.read().decode())

                http_thread = threading.Thread(target=do_http)
                http_thread.start()
                self.assertTrue(started.wait(timeout=2))
                self.assertGreater(server._inflight, 0)

                drain_ok: dict[str, bool] = {}
                drain_done = threading.Event()

                def do_drain() -> None:
                    # quiet=0 is enough: admit happens before the body is read.
                    drain_ok["ok"] = server.drain_http(timeout=2.0, quiet=0.0)
                    drain_done.set()

                drain_thread = threading.Thread(target=do_drain)
                drain_thread.start()
                self.assertFalse(drain_done.wait(timeout=0.1))
                release.set()
                http_thread.join(timeout=2)
                drain_thread.join(timeout=2)
                self.assertTrue(drain_ok.get("ok"))
                self.assertEqual(outcome.get("status"), 200)
                self.assertEqual(server.best_candidate, "in-transit")
                self.assertFalse(server._http_accepting)
        finally:
            release.set()
            server.stop()

    def test_validate_progress_is_not_selectable(self) -> None:
        task = Task(name="task", seed_candidate="seed", train_set=["a"], val_set=["v"])
        server = EvalServer(task, lambda _candidate, _example: (1.0, {}), BudgetTracker(max_evals=10))
        result = server.validate("val-only")
        self.assertEqual(result["val_score"], 1.0)
        self.assertEqual(len(server.progress_log), 1)
        self.assertFalse(server.progress_log[0]["selectable"])

    def test_drain_http_rejects_new_requests_after_quiet_period(self) -> None:
        import urllib.error
        import urllib.request

        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=4))
        server.start()
        try:
            self.assertTrue(server.drain_http(timeout=2.0, quiet=0.0))
            self.assertFalse(server._http_accepting)
            req = urllib.request.Request(
                f"{server.url}/evaluate",
                data=json.dumps({"candidate": "late"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(raised.exception.code, 409)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
