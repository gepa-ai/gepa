from __future__ import annotations

import http.server
import json
import tempfile
import threading
import unittest
import urllib.request
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

    def test_in_process_gepa_run_never_binds_http_listener(self) -> None:
        """A pure in-process ``optimize_anything`` run never opens the localhost
        HTTP listener (#448, item 3a): any bind attempt goes through
        ``ThreadingHTTPServer``, so patching it to raise turns a regression
        back into a loud failure while the run itself must still succeed."""

        class _FakeLM:
            def __call__(self, *args, **kwargs):
                return "IMPROVED CANDIDATE: a longer and better answer than before"

        def _refuse_to_bind(*args, **kwargs):
            raise AssertionError("HTTP listener must not be bound for in-process runs")

        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

        with patch.object(http.server, "ThreadingHTTPServer", _refuse_to_bind):
            result = optimize_anything(
                seed_candidate="short",
                evaluator=lambda candidate, example=None: (float(len(candidate)) / 100.0, {}),
                objective="maximize length",
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=4),
                    reflection=ReflectionConfig(reflection_lm=_FakeLM()),
                ),
            )

        self.assertTrue(result.best_candidate)
        self.assertGreaterEqual(result.total_evals, 1)

    def test_url_access_binds_listener_lazily_and_endpoint_answers(self) -> None:
        """Reading ``.url`` starts the listener on demand; the surfaced URL is
        live, so external engines can rely on it without an explicit start."""
        task = Task(name="task", seed_candidate="seed", train_set=["a", "b"])
        server = EvalServer(
            task,
            lambda candidate, example: (1.0, {}),
            BudgetTracker(max_evals=4),
            max_concurrency=1,
        )
        try:
            self.assertIsNone(server._server)
            url = server.url
            self.assertIsNotNone(server._server)
            self.assertEqual(url, f"http://localhost:{server.port}")
            with urllib.request.urlopen(f"{url}/status", timeout=5) as resp:
                payload = json.loads(resp.read().decode())
        finally:
            server.stop()

        self.assertIn("budget", payload)

    def test_explicit_start_is_idempotent_and_stop_is_final(self) -> None:
        """Callers owning the server (ensemble compositions) keep the explicit
        contract: ``start()`` is idempotent and rebinds nothing; after
        ``stop()`` the listener cannot silently come back."""
        task = Task(name="task", seed_candidate="seed")
        server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=2))
        try:
            self.assertIsNone(server._server)
            port = server.start()
            self.assertEqual(server.start(), port)
            self.assertEqual(server.port, port)
        finally:
            server.stop()

        self.assertIsNone(server._server)
        with self.assertRaises(RuntimeError):
            server.port  # noqa: B018


if __name__ == "__main__":
    unittest.main()
