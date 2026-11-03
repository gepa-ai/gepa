from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gepa.oa import BudgetTracker, EvalServer, OptimizeAnythingConfig, Task
from gepa.oa.engines.autoresearch import _best_aggregate_candidate
from gepa.oa.engines.gepa import GepaEngine
from gepa.oa.ensemble import optimize_adaptive_sequential, optimize_adaptive_sequential_with_server


@pytest.fixture(autouse=True)
def _skip_claude_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests mock the claude subprocess; skip the CLI/bwrap preflight."""
    monkeypatch.setattr("gepa.oa.engines.autoresearch.preflight_claude_engine", lambda *a, **k: None)


def _evaluate(candidate: str) -> tuple[float, dict]:
    del candidate
    return 1.0, {}


class OptimizeAnythingAdaptiveSequentialTests(unittest.TestCase):
    def test_switches_after_plateau_and_preserves_global_best(self) -> None:
        server = EvalServer(
            Task(name="task", seed_candidate="seed"),
            _evaluate,
            BudgetTracker(max_evals=3),
            max_concurrency=1,
        )
        configs = [OptimizeAnythingConfig(engine="gepa"), OptimizeAnythingConfig(engine="autoresearch")]
        calls: list[tuple[str, str]] = []

        def fake_optimize(active_server, cfg):
            calls.append((str(cfg.engine), active_server.task.seed_candidate))
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
            Task(name="task", seed_candidate="seed"),
            lambda candidate: (1.0 if candidate == "spiky" else 0.0, {}),
            BudgetTracker(max_evals=2),
            max_concurrency=1,
        )
        configs = [OptimizeAnythingConfig(engine="gepa"), OptimizeAnythingConfig(engine="autoresearch")]

        def fake_optimize(active_server, cfg):
            if cfg.engine == "gepa":
                active_server.evaluate("spiky")
                return SimpleNamespace(best_candidate="aggregate-best", best_score=0.5, metadata={})
            self.assertEqual(active_server.task.seed_candidate, "aggregate-best")
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
            Task(name="task", seed_candidate="seed"),
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

    def test_convenience_wrapper_shares_one_server_and_scores_test_outside_budget(self) -> None:
        servers: list[EvalServer] = []
        calls: list[tuple[str, str | None]] = []

        def fake_optimize(active_server, cfg):
            servers.append(active_server)
            calls.append((str(cfg.engine), active_server.task.seed_candidate))
            active_server.evaluate(str(cfg.engine))
            if len(calls) == 1:
                return SimpleNamespace(best_candidate="improved", best_score=0.8, metadata={})
            return SimpleNamespace(best_candidate="regressed", best_score=0.1, metadata={})

        test_scored: list[str] = []

        def evaluate(candidate: str, example=None) -> tuple[float, dict]:
            if example is not None:
                test_scored.append(candidate)
            return 1.0, {}

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("gepa.oa.ensemble.optimize_anything_with_server", side_effect=fake_optimize),
        ):
            result = optimize_adaptive_sequential(
                "seed",
                evaluator=evaluate,
                configs=[
                    OptimizeAnythingConfig(engine="gepa"),
                    OptimizeAnythingConfig(engine="autoresearch"),
                ],
                plateau_evals=1,
                test_set=[{"id": "t0"}],
                max_evals=3,
                patience=1,
                min_evals_per_stage=1,
                cycle=False,
                output_dir=tmp,
            )

        # One shared server across every slice; slices never see the test_set.
        self.assertEqual(len({id(s) for s in servers}), 1)
        self.assertIsNone(servers[0].task.test_set)
        self.assertEqual(calls, [("gepa", "seed"), ("gepa", "improved"), ("autoresearch", "improved")])
        self.assertEqual(result.best_candidate, "improved")
        self.assertEqual(result.best_score, 0.8)
        # Test pass runs once for the seed and once for the best, outside the budget.
        self.assertEqual(test_scored, ["seed", "improved"])
        self.assertEqual(result.metadata["baseline_test_score"], 1.0)
        self.assertEqual(result.metadata["test_score"], 1.0)
        self.assertEqual(
            result.metadata["budget"], {"exhausted": True, "max_evals": 3, "used": 3, "remaining_evals": 0}
        )

    def test_convenience_wrapper_warns_when_fully_unbounded(self) -> None:
        def fake_optimize(active_server, cfg):
            del active_server, cfg
            return SimpleNamespace(best_candidate="seed", best_score=0.0, metadata={})

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("gepa.oa.ensemble.optimize_anything_with_server", side_effect=fake_optimize),
            self.assertWarns(UserWarning),
        ):
            optimize_adaptive_sequential(
                "seed",
                evaluator=_evaluate,
                configs=[OptimizeAnythingConfig(engine="gepa")],
                plateau_evals=1,
                output_dir=tmp,
            )

    def test_autoresearch_aggregate_candidate_helper_ignores_per_eval_best(self) -> None:
        server = EvalServer(
            Task(name="task", seed_candidate="seed"),
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
        task = Task(name="task", seed_candidate="seed", train_set=["a"])
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
        task = Task(name="task", seed_candidate="seed")
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
            Task(name="task", seed_candidate="seed"),
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

        with patch("gepa.gepa_launcher.optimize_anything", side_effect=fake_optimize):
            result = engine.run(server.task, server)

        # The cost source is treated as fresh: the cap is max_token_cost itself,
        # and adapter_cost is the full total_cost the run reported.
        self.assertEqual(seen["max_reflection_cost"], 0.5)
        self.assertEqual(result.metadata["adapter_cost"], 2.75)


# ── Real gepa slices on a shared server ────────────────────────────────────


def _real_gepa_config() -> OptimizeAnythingConfig:
    from gepa.gepa_launcher import EngineConfig, ReflectionConfig

    counter = iter(range(1, 10_000))
    return OptimizeAnythingConfig(
        engine="gepa",
        engine_config={
            "reflection": ReflectionConfig(
                reflection_lm=lambda prompt: "```\n" + "improved " * next(counter) + "\n```",
                reflection_minibatch_size=2,
            ),
            "engine": EngineConfig(parallel=False, seed=0, raise_on_exception=True),
        },
    )


class _HardCapEngine:
    """Stands in for an engine that relies on the server's hard cap to stop
    (autoresearch, meta_harness): it evaluates until ``BudgetExhausted``."""

    name = "hardcap"

    def __init__(self, config):
        del config

    def run(self, task, server):
        from gepa.oa.budget import BudgetExhausted
        from gepa.oa.engine import Result

        try:
            for n in range(500):  # a missing cap shows up as 500 calls, not a hang
                server.evaluate(f"hardcap {n}", task.train_set[n % len(task.train_set)])
        except BudgetExhausted:
            pass
        return Result(best_candidate=str(server.best_candidate), best_score=0.0, total_evals=server.budget.used)

    def process_result(self, result, output_dir):
        del result, output_dir


DATA = [f"ex{i}" for i in range(4)]


def test_gepa_slice_restores_the_hard_cap_for_the_next_engine(tmp_path, monkeypatch):
    """The gepa engine turns the server's hard cap off for its own run (it stops
    between iterations instead). It must turn it back on: the next slice's
    engine shares the server and relies on it."""
    from gepa.oa.registry import _ENGINES

    monkeypatch.setitem(_ENGINES, "hardcap", _HardCapEngine)
    calls = []

    def evaluator(candidate, example):
        calls.append(candidate)
        return 0.5, {}  # flat score: the gepa stage plateaus and the scheduler switches

    result = optimize_adaptive_sequential(
        "seed",
        evaluator=evaluator,
        configs=[_real_gepa_config(), OptimizeAnythingConfig(engine="hardcap")],
        plateau_evals=10,
        dataset=DATA,
        valset=DATA,
        max_evals=30,
        patience=1,
        cycle=False,
        output_dir=tmp_path,
    )

    engines = [stage["engine"] for stage in result.metadata["adaptive_schedule"]]
    assert engines[0] == "gepa" and engines[-1] == "hardcap"
    assert len(calls) == 30


def test_gepa_slices_on_a_shared_server_each_get_their_own_window(tmp_path):
    """Each gepa slice is capped at what is left of its window, counted from
    zero, and starts from a fresh state directory instead of resuming the
    previous slice (which had already used more than the new window)."""
    calls = []

    def evaluator(candidate, example):
        calls.append(candidate)
        return min(1.0, len(candidate) / 200.0), {}

    result = optimize_adaptive_sequential(
        "seed",
        evaluator=evaluator,
        configs=[_real_gepa_config(), _real_gepa_config()],
        plateau_evals=10,
        dataset=DATA,
        valset=DATA,
        max_evals=40,
        patience=1,
        output_dir=tmp_path,
    )

    deltas = [stage["eval_delta"] for stage in result.metadata["adaptive_schedule"]]
    assert len(deltas) > 1 and all(d > 0 for d in deltas)
    # Soft cap: a slice finishes the iteration in flight (seed valset pass of 4
    # plus one iteration of 8 here), so the total stays within one slice's
    # overshoot of the cap.
    assert 40 <= len(calls) <= 40 + 12
    assert result.total_evals == len(calls)
    assert len(list(tmp_path.rglob("gepa_state*"))) >= len(deltas)


def test_gepa_engine_counts_from_what_is_left_and_restores_enforcement():
    from gepa.oa.ensemble import optimize_anything_with_server

    calls = []

    def evaluator(candidate, example):
        calls.append(candidate)
        return min(1.0, len(candidate) / 200.0), {}

    server = EvalServer(
        Task(name="t", seed_candidate="seed", train_set=DATA, val_set=DATA),
        evaluator,
        BudgetTracker(max_evals=22),
        max_concurrency=1,
    )
    for _ in range(12):  # an earlier slice already used 12 of the 22
        server.budget.record(0.0)

    result = optimize_anything_with_server(server, _real_gepa_config())

    gepa_result = result.metadata["gepa_result"]
    assert 10 <= len(calls) <= 10 + 8  # 10 left, plus at most the iteration in flight
    assert gepa_result.total_metric_calls == len(calls)
    assert server.budget.enforce is True


def test_gepa_engine_on_a_used_up_budget_evaluates_nothing():
    from gepa.oa.ensemble import optimize_anything_with_server

    calls = []
    server = EvalServer(
        Task(name="t", seed_candidate="seed", train_set=DATA, val_set=DATA),
        lambda candidate, example: (calls.append(candidate), (1.0, {}))[1],
        BudgetTracker(max_evals=5),
        max_concurrency=1,
    )
    for _ in range(5):
        server.budget.record(0.0)

    optimize_anything_with_server(server, _real_gepa_config())

    assert calls == []
    assert server.budget.enforce is True


if __name__ == "__main__":
    unittest.main()
