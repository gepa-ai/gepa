# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for wandb_attach_existing and mlflow_attach_existing flags."""

from unittest.mock import MagicMock, patch, call

import pytest

from gepa.logging.experiment_tracker import ExperimentTracker, create_experiment_tracker
from gepa.optimize_anything import GEPAConfig, TrackingConfig


# ---------------------------------------------------------------------------
# ExperimentTracker — wandb_attach_existing
# ---------------------------------------------------------------------------

class TestWandbAttachExisting:
    def test_attach_existing_skips_init(self):
        """wandb.init() is not called when attach_existing=True."""
        tracker = ExperimentTracker(use_wandb=True, wandb_attach_existing=True)
        with patch("wandb.init") as mock_init, \
             patch("wandb.login"):
            tracker.initialize()
            tracker.start_run()
        mock_init.assert_not_called()

    def test_attach_existing_skips_finish(self):
        """wandb.finish() is not called when attach_existing=True."""
        tracker = ExperimentTracker(use_wandb=True, wandb_attach_existing=True)
        mock_run = MagicMock()
        with patch("wandb.run", mock_run), \
             patch("wandb.finish") as mock_finish:
            tracker.end_run()
        mock_finish.assert_not_called()

    def test_attach_existing_still_logs(self):
        """Metrics are logged via wandb.log() even in attach mode."""
        tracker = ExperimentTracker(use_wandb=True, wandb_attach_existing=True)
        with patch("wandb.log") as mock_log:
            tracker.log_metrics({"score": 0.8}, step=1)
        mock_log.assert_called_once_with({"score": 0.8}, step=1)

    def test_normal_mode_calls_init_and_finish(self):
        """Without attach_existing, wandb.init() and wandb.finish() are called."""
        tracker = ExperimentTracker(use_wandb=True, wandb_attach_existing=False)
        mock_run = MagicMock()
        with patch("wandb.login"), \
             patch("wandb.init") as mock_init, \
             patch("wandb.run", mock_run), \
             patch("wandb.finish") as mock_finish:
            tracker.initialize()
            tracker.start_run()
            tracker.end_run()
        mock_init.assert_called_once()
        mock_finish.assert_called_once()

    def test_context_manager_attach_existing(self):
        """Context manager entry/exit respects attach_existing."""
        tracker = ExperimentTracker(use_wandb=True, wandb_attach_existing=True)
        mock_run = MagicMock()
        with patch("wandb.login"), \
             patch("wandb.init") as mock_init, \
             patch("wandb.run", mock_run), \
             patch("wandb.finish") as mock_finish:
            with tracker:
                pass
        mock_init.assert_not_called()
        mock_finish.assert_not_called()


# ---------------------------------------------------------------------------
# ExperimentTracker — mlflow_attach_existing
# ---------------------------------------------------------------------------

class TestMlflowAttachExisting:
    def test_attach_existing_skips_start_run(self):
        """mlflow.start_run() is not called when attach_existing=True."""
        tracker = ExperimentTracker(use_mlflow=True, mlflow_attach_existing=True)
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.active_run", return_value=MagicMock()), \
             patch("mlflow.start_run") as mock_start:
            tracker.initialize()
            tracker.start_run()
        mock_start.assert_not_called()
        assert tracker._created_mlflow_run is False

    def test_attach_existing_skips_end_run(self):
        """mlflow.end_run() is not called when attach_existing=True."""
        tracker = ExperimentTracker(use_mlflow=True, mlflow_attach_existing=True)
        tracker._created_mlflow_run = False  # never created
        with patch("mlflow.end_run") as mock_end, \
             patch("mlflow.active_run", return_value=MagicMock()):
            tracker.end_run()
        mock_end.assert_not_called()

    def test_attach_existing_still_logs_metrics(self):
        """Metrics are logged via mlflow.log_metrics() in attach mode."""
        tracker = ExperimentTracker(use_mlflow=True, mlflow_attach_existing=True)
        with patch("mlflow.log_metrics") as mock_log:
            tracker.log_metrics({"val_score": 0.9}, step=2)
        mock_log.assert_called_once_with({"val_score": 0.9}, step=2)

    def test_normal_mode_creates_and_ends_run(self):
        """Without attach_existing, mlflow.start_run() and end_run() are called."""
        tracker = ExperimentTracker(use_mlflow=True, mlflow_attach_existing=False)
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.active_run", return_value=None), \
             patch("mlflow.start_run") as mock_start, \
             patch("mlflow.end_run") as mock_end:
            # active_run returns a mock after start_run
            mock_start.return_value = MagicMock()
            tracker.initialize()
            tracker.start_run()
            assert tracker._created_mlflow_run is True
            with patch("mlflow.active_run", return_value=MagicMock()):
                tracker.end_run()
        mock_start.assert_called_once()
        mock_end.assert_called_once()

    def test_context_manager_attach_existing(self):
        """Context manager entry/exit respects mlflow attach_existing."""
        tracker = ExperimentTracker(use_mlflow=True, mlflow_attach_existing=True)
        with patch("mlflow.active_run", return_value=MagicMock()), \
             patch("mlflow.start_run") as mock_start, \
             patch("mlflow.end_run") as mock_end:
            with tracker:
                pass
        mock_start.assert_not_called()
        mock_end.assert_not_called()


# ---------------------------------------------------------------------------
# create_experiment_tracker — new flags threaded through
# ---------------------------------------------------------------------------

class TestCreateExperimentTracker:
    def test_wandb_attach_existing_passed_through(self):
        tracker = create_experiment_tracker(
            use_wandb=True, wandb_attach_existing=True
        )
        assert tracker.wandb_attach_existing is True

    def test_mlflow_attach_existing_passed_through(self):
        tracker = create_experiment_tracker(
            use_mlflow=True, mlflow_attach_existing=True
        )
        assert tracker.mlflow_attach_existing is True

    def test_defaults_are_false(self):
        tracker = create_experiment_tracker()
        assert tracker.wandb_attach_existing is False
        assert tracker.mlflow_attach_existing is False


# ---------------------------------------------------------------------------
# TrackingConfig — fields present and wired to experiment tracker
# ---------------------------------------------------------------------------

class TestTrackingConfig:
    def test_fields_exist_with_defaults(self):
        cfg = TrackingConfig()
        assert cfg.wandb_attach_existing is False
        assert cfg.mlflow_attach_existing is False

    def test_fields_settable(self):
        cfg = TrackingConfig(
            use_wandb=True,
            wandb_attach_existing=True,
            use_mlflow=True,
            mlflow_attach_existing=True,
        )
        assert cfg.wandb_attach_existing is True
        assert cfg.mlflow_attach_existing is True

    def test_config_wired_to_tracker_via_optimize_anything(self):
        """TrackingConfig.wandb_attach_existing reaches the ExperimentTracker."""
        from gepa.optimize_anything import optimize_anything, EngineConfig, ReflectionConfig

        created_trackers: list[ExperimentTracker] = []
        original = create_experiment_tracker

        def spy(*args, **kwargs):
            t = original(*args, **kwargs)
            created_trackers.append(t)
            return t

        with patch("gepa.optimize_anything.create_experiment_tracker", side_effect=spy), \
             patch("gepa.optimize_anything.GEPAEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_state = MagicMock()
            mock_state.program_candidates = [{"current_candidate": "v0"}]
            mock_state.parent_program_for_candidate = [[None]]
            mock_state.prog_candidate_val_subscores = [{}]
            mock_state.program_at_pareto_front_valset = {}
            mock_state.program_full_scores_val_set = [0.5]
            mock_state.num_metric_calls_by_discovery = [0]
            mock_state.total_num_evals = 1
            mock_state.num_full_ds_evals = 1
            mock_state.best_outputs_valset = None
            mock_state.objective_pareto_front = {}
            mock_state.program_at_pareto_front_objectives = {}
            mock_engine.run.return_value = mock_state
            mock_cls.return_value = mock_engine

            optimize_anything(
                seed_candidate="x",
                evaluator=lambda c: 0.5,
                config=GEPAConfig(
                    engine=EngineConfig(max_metric_calls=3),
                    reflection=ReflectionConfig(
                        reflection_lm=MagicMock(return_value="```\nx\n```")
                    ),
                    tracking=TrackingConfig(
                        use_wandb=True,
                        wandb_attach_existing=True,
                        use_mlflow=True,
                        mlflow_attach_existing=True,
                    ),
                ),
            )

        assert len(created_trackers) == 1
        assert created_trackers[0].wandb_attach_existing is True
        assert created_trackers[0].mlflow_attach_existing is True


# ---------------------------------------------------------------------------
# gepa.optimize — attach flags in the flat API
# ---------------------------------------------------------------------------

class TestOptimizeApiAttachExisting:
    def test_wandb_attach_existing_in_optimize(self):
        """gepa.optimize passes wandb_attach_existing to the tracker."""
        import gepa
        from gepa.core.adapter import EvaluationBatch

        created: list[ExperimentTracker] = []

        def spy(**kwargs):
            t = ExperimentTracker(**kwargs)
            created.append(t)
            return t

        class DummyAdapter:
            propose_new_texts = None

            def evaluate(self, batch, candidate, capture_traces=False):
                return EvaluationBatch(outputs=[0.5]*len(batch), scores=[0.5]*len(batch))

            def make_reflective_dataset(self, candidate, eval_batch, components):
                return {c: [] for c in components}

        with patch("gepa.api.create_experiment_tracker", side_effect=spy), \
             patch("gepa.api.GEPAEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_state = MagicMock()
            mock_state.program_candidates = [{"system_prompt": "v0"}]
            mock_state.parent_program_for_candidate = [[None]]
            mock_state.prog_candidate_val_subscores = [{}]
            mock_state.program_at_pareto_front_valset = {}
            mock_state.program_full_scores_val_set = [0.5]
            mock_state.num_metric_calls_by_discovery = [0]
            mock_state.total_num_evals = 1
            mock_state.num_full_ds_evals = 1
            mock_state.best_outputs_valset = None
            mock_state.objective_pareto_front = {}
            mock_state.program_at_pareto_front_objectives = {}
            mock_engine.run.return_value = mock_state
            mock_cls.return_value = mock_engine

            gepa.optimize(
                seed_candidate={"system_prompt": "hello"},
                trainset=[{"q": "x"}],
                adapter=DummyAdapter(),
                reflection_lm=MagicMock(return_value="```\nhello\n```"),
                max_metric_calls=3,
                use_wandb=True,
                wandb_attach_existing=True,
            )

        assert len(created) == 1
        assert created[0].wandb_attach_existing is True
