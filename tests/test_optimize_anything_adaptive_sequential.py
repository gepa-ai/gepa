from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gepa.oa import BudgetTracker, EvalServer, OptimizeAnythingConfig, Task
from gepa.oa.engines.autoresearch import _best_aggregate_candidate
from gepa.oa.engines.gepa import GepaEngine
from gepa.oa.ensemble import optimize_adaptive_sequential_with_server


def _evaluate(candidate: str) -> tuple[float, dict]:
    del candidate
    return 1.0, {}


class OptimizeAnythingAdaptiveSequentialTests(unittest.TestCase):
    def test_switches_after_plateau_and_preserves_global_best(self) -> None:
        server = EvalServer(
            Task(name="task", initial_candidate="seed"),
            _evaluate,
            BudgetTracker(max_evals=3),
            max_concurrency=1,
        )
        configs = [OptimizeAnythingConfig(engine="gepa"), OptimizeAnythingConfig(engine="autoresearch")]
        calls: list[tuple[str, str]] = []

        def fake_optimize(active_server, cfg):
            calls.append((str(cfg.engine), active_server.task.initial_candidate))
            active_server.evaluate(str(cfg.engine))
            if len(calls) <= 2:
                return SimpleNamespace(best_candidate="gepa-best", best_score=0.8, metadata={})
            return SimpleNamespace(best_candidate="regressed", best_score=0.2, metadata={})

        with patch("gepa.oa.ensemble.optimize_anything_with_server", side_effect=fake_optimize):
            result = optimize_adaptive_sequential_with_server(
                server,
                configs,
                plateau_evals=1,
                patience=1,
                min_evals_per_stage=1,
                cycle=False,
            )

        self.assertEqual(calls, [("gepa", "seed"), ("gepa", "gepa-best"), ("autoresearch", "gepa-best")])
        self.assertEqual(result.best_candidate, "gepa-best")
        self.assertEqual(result.best_score, 0.8)
        self.assertEqual(result.metadata["adaptive_switches"], 1)
        self.assertEqual(server.budget.max_evals, 3)

    def test_uses_engine_aggregate_not_per_eval_server_best(self) -> None:
        server = EvalServer(
            Task(name="task", initial_candidate="seed"),
            lambda candidate: (1.0 if candidate == "spiky" else 0.0, {}),
            BudgetTracker(max_evals=2),
            max_concurrency=1,
        )
        configs = [OptimizeAnythingConfig(engine="gepa"), OptimizeAnythingConfig(engine="autoresearch")]

        def fake_optimize(active_server, cfg):
            if cfg.engine == "gepa":
                active_server.evaluate("spiky")
                return SimpleNamespace(best_candidate="aggregate-best", best_score=0.5, metadata={})
            self.assertEqual(active_server.task.initial_candidate, "aggregate-best")
            active_server.evaluate("next")
            return SimpleNamespace(best_candidate="next", best_score=0.1, metadata={})

        with patch("gepa.oa.ensemble.optimize_anything_with_server", side_effect=fake_optimize):
            result = optimize_adaptive_sequential_with_server(
                server,
                configs,
                plateau_evals=1,
                patience=1,
                min_evals_per_stage=1,
                cycle=False,
            )

        self.assertEqual(server.best_candidate, "spiky")
        self.assertEqual(server.best_score, 1.0)
        self.assertEqual(result.best_candidate, "aggregate-best")
        self.assertEqual(result.best_score, 0.5)

    def test_does_not_start_new_slice_below_minimum_remaining_budget(self) -> None:
        server = EvalServer(
            Task(name="task", initial_candidate="seed"),
            _evaluate,
            BudgetTracker(max_evals=3),
            max_concurrency=1,
        )
        configs = [OptimizeAnythingConfig(engine="gepa"), OptimizeAnythingConfig(engine="autoresearch")]
        calls: list[str] = []

        def fake_optimize(active_server, cfg):
            calls.append(str(cfg.engine))
            active_server.evaluate("candidate-a")
            active_server.evaluate("candidate-b")
            return SimpleNamespace(best_candidate="best", best_score=0.1, metadata={})

        with patch("gepa.oa.ensemble.optimize_anything_with_server", side_effect=fake_optimize):
            result = optimize_adaptive_sequential_with_server(
                server,
                configs,
                plateau_evals=2,
                patience=1,
                min_evals_per_stage=2,
                cycle=True,
            )

        self.assertEqual(calls, ["gepa"])
        self.assertEqual(server.budget.used, 2)
        self.assertEqual(result.metadata["adaptive_stop_reason"], "scheduler_stopped")

    def test_autoresearch_aggregate_candidate_helper_ignores_per_eval_best(self) -> None:
        server = EvalServer(
            Task(name="task", initial_candidate="seed"),
            _evaluate,
            BudgetTracker(max_evals=10),
            max_concurrency=1,
        )
        server.best_candidate = "spiky"
        server.best_score = 1.0
        server.log_progress(0.4, candidate="aggregate-a")
        server.log_progress(0.7, candidate="aggregate-b")

        self.assertEqual(_best_aggregate_candidate(server), ("aggregate-b", 0.7))

    def test_autoresearch_dataset_without_aggregate_score_returns_negative_infinity(self) -> None:
        from gepa.oa.engines.autoresearch import AutoResearchEngine

        engine = AutoResearchEngine(
            OptimizeAnythingConfig(engine="autoresearch", engine_config={"model": "sonnet", "ralph": False})
        )
        task = Task(name="task", initial_candidate="seed", train_set=["a"])
        server = EvalServer(task, lambda candidate, example: (1.0, {}), BudgetTracker(max_evals=1), max_concurrency=1)
        server.start()

        try:
            with patch.object(engine, "_run_claude") as fake_run:
                fake_run.return_value = SimpleNamespace(returncode=0, stdout='{"total_cost_usd": 0.0}', stderr="")
                result = engine.run(task, server)
        finally:
            server.stop()

        self.assertEqual(result.best_score, float("-inf"))

    def test_autoresearch_stops_ralph_when_eval_script_reports_budget_exhausted(self) -> None:
        from gepa.oa.engines.autoresearch import AutoResearchEngine

        engine = AutoResearchEngine(
            OptimizeAnythingConfig(engine="autoresearch", engine_config={"model": "sonnet", "ralph": True})
        )
        task = Task(name="task", initial_candidate="seed")
        server = EvalServer(task, lambda candidate: (1.0, {}), BudgetTracker(max_evals=10), max_concurrency=1)
        server.start()

        try:
            with patch.object(engine, "_run_claude") as fake_run:
                fake_run.return_value = SimpleNamespace(
                    returncode=0,
                    stdout='{"total_cost_usd": 0.01, "result": "BUDGET_EXHAUSTED"}',
                    stderr="",
                )
                engine.run(task, server)
        finally:
            server.stop()

        self.assertEqual(fake_run.call_count, 1)

    def test_autoresearch_kills_process_after_local_budget_exhaustion_grace(self) -> None:
        from pathlib import Path

        from gepa.oa.engines.autoresearch import AutoResearchEngine

        class FakeProcess:
            returncode = None

            def __init__(self) -> None:
                self.terminated = False

            def poll(self):
                return None if not self.terminated else -15

            def terminate(self) -> None:
                self.terminated = True
                self.returncode = -15

            def communicate(self, timeout=None):
                del timeout
                return "", ""

        proc = FakeProcess()
        engine = AutoResearchEngine(OptimizeAnythingConfig(engine="autoresearch", engine_config={"model": "sonnet"}))
        budget = BudgetTracker(max_evals=0)

        with (
            patch("gepa.oa.engines.autoresearch.subprocess.Popen", return_value=proc),
            patch("gepa.oa.engines.autoresearch.time.sleep", return_value=None),
            patch("gepa.oa.engines.autoresearch._BUDGET_EXHAUSTION_GRACE_SECONDS", 0.0),
        ):
            result = engine._run_claude(
                work_dir=Path("."),
                session_id="sid",
                prompt="prompt",
                budget=budget,
                adapter_cost=0.0,
                resume=False,
                env={},
            )

        self.assertIn("BUDGET_EXHAUSTED", result.stderr)

    def test_gepa_engine_reports_delta_cost_for_reused_reflection_lm(self) -> None:
        class FakeLM:
            total_cost = 2.0

        fake_lm = FakeLM()
        seen: dict[str, float] = {}

        def fake_optimize(**kwargs):
            seen["max_reflection_cost"] = kwargs["config"].engine.max_reflection_cost
            fake_lm.total_cost = 2.75
            return SimpleNamespace(best_candidate="best", best_idx=0, val_aggregate_scores=[0.9])

        server = EvalServer(
            Task(name="task", initial_candidate="seed"),
            _evaluate,
            BudgetTracker(max_evals=10),
            max_concurrency=1,
        )
        engine = GepaEngine(
            OptimizeAnythingConfig(
                engine="gepa",
                max_token_cost=0.5,
                engine_config={"reflection": {"reflection_lm": fake_lm}},
            )
        )

        with patch("gepa.legacy_optimize_anything.optimize_anything", side_effect=fake_optimize):
            result = engine.run(server.task, server)

        self.assertEqual(seen["max_reflection_cost"], 2.5)
        self.assertEqual(result.metadata["adapter_cost"], 0.75)
        self.assertEqual(result.metadata["reflection_cost_initial"], 2.0)
        self.assertEqual(result.metadata["reflection_cost_final"], 2.75)


if __name__ == "__main__":
    unittest.main()
