# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from unittest.mock import MagicMock, patch

from gepa.logging.experiment_tracker import ExperimentTracker


class TestLogTable:
    """Test ExperimentTracker.log_table() for wandb and mlflow backends."""

    def test_log_table_wandb(self):
        tracker = ExperimentTracker(use_wandb=True)
        columns = ["name", "score"]
        data = [["alice", 0.9], ["bob", 0.8]]

        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker.log_table("test_table", columns=columns, data=data)

        mock_wandb.Table.assert_called_once_with(columns=columns, data=data)
        mock_wandb.log.assert_called_once()
        logged = mock_wandb.log.call_args[0][0]
        assert "test_table" in logged
        assert mock_wandb.log.call_args[1]["commit"] is False

    def test_log_table_mlflow(self):
        tracker = ExperimentTracker(use_mlflow=True)
        columns = ["name", "score"]
        data = [["alice", 0.9], ["bob", 0.8]]

        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker.log_table("test_table", columns=columns, data=data)

        mock_mlflow.log_table.assert_called_once()
        call_kwargs = mock_mlflow.log_table.call_args[1]
        assert call_kwargs["artifact_file"] == "test_table.json"
        table_dict = call_kwargs["data"]
        assert table_dict == {"name": ["alice", "bob"], "score": [0.9, 0.8]}

    def test_log_table_no_backends(self):
        tracker = ExperimentTracker(use_wandb=False, use_mlflow=False)
        # Should not raise
        tracker.log_table("test", columns=["a"], data=[[1]])

    def test_log_table_wandb_error_handled(self):
        tracker = ExperimentTracker(use_wandb=True)

        mock_wandb = MagicMock()
        mock_wandb.Table.side_effect = RuntimeError("wandb error")
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            # Should not raise
            tracker.log_table("test", columns=["a"], data=[[1]])

    def test_log_table_mlflow_error_handled(self):
        tracker = ExperimentTracker(use_mlflow=True)

        mock_mlflow = MagicMock()
        mock_mlflow.log_table.side_effect = RuntimeError("mlflow error")
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            # Should not raise
            tracker.log_table("test", columns=["a"], data=[[1]])


class TestLogMetricsNumericFilter:
    """Test that log_metrics filters non-numeric values for both backends."""

    def test_wandb_filters_strings(self):
        tracker = ExperimentTracker(use_wandb=True)
        metrics = {"score": 0.9, "name": "test", "count": 5}

        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker.log_metrics(metrics, step=1)

        logged = mock_wandb.log.call_args[0][0]
        assert "score" in logged
        assert "count" in logged
        assert "name" not in logged

    def test_mlflow_filters_strings(self):
        tracker = ExperimentTracker(use_mlflow=True)
        metrics = {"score": 0.9, "label": "test", "count": 5}

        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker.log_metrics(metrics, step=1)

        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert "score" in logged
        assert "count" in logged
        assert "label" not in logged


class TestLogConfig:
    """Test ExperimentTracker.log_config() for wandb and mlflow backends."""

    def test_wandb_config_update(self):
        tracker = ExperimentTracker(use_wandb=True)
        config = {"seed": 42, "lr": 0.01, "name": "test"}

        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker.log_config(config)

        mock_wandb.config.update.assert_called_once()
        logged = mock_wandb.config.update.call_args[0][0]
        assert logged == {"seed": 42, "lr": 0.01, "name": "test"}
        assert mock_wandb.config.update.call_args[1]["allow_val_change"] is True

    def test_non_serializable_values_stringified(self):
        tracker = ExperimentTracker(use_wandb=True)
        config = {"seed": 42, "components": ["a", "b"], "obj": object()}

        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker.log_config(config)

        logged = mock_wandb.config.update.call_args[0][0]
        assert logged["seed"] == 42
        assert isinstance(logged["components"], str)
        assert isinstance(logged["obj"], str)

    def test_mlflow_params_as_strings(self):
        tracker = ExperimentTracker(use_mlflow=True)
        config = {"seed": 42, "lr": 0.01, "name": "test"}

        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker.log_config(config)

        mock_mlflow.log_params.assert_called_once()
        logged = mock_mlflow.log_params.call_args[0][0]
        assert all(isinstance(v, str) for v in logged.values())
        assert logged["seed"] == "42"

    def test_no_backends_no_error(self):
        tracker = ExperimentTracker(use_wandb=False, use_mlflow=False)
        tracker.log_config({"key": "value"})

    def test_wandb_error_handled(self):
        tracker = ExperimentTracker(use_wandb=True)

        mock_wandb = MagicMock()
        mock_wandb.config.update.side_effect = RuntimeError("wandb error")
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker.log_config({"key": "value"})


class TestLogSummary:
    """Test ExperimentTracker.log_summary() for wandb and mlflow backends."""

    def test_wandb_summary_set(self):
        tracker = ExperimentTracker(use_wandb=True)
        summary = {"best_score": 0.95, "best_idx": 3, "best_prompt": "Do X"}

        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker.log_summary(summary)

        mock_wandb.run.summary.__setitem__.assert_any_call("best_score", 0.95)
        mock_wandb.run.summary.__setitem__.assert_any_call("best_idx", 3)
        mock_wandb.run.summary.__setitem__.assert_any_call("best_prompt", "Do X")

    def test_mlflow_splits_numeric_and_text(self):
        tracker = ExperimentTracker(use_mlflow=True)
        summary = {"best_score": 0.95, "best_prompt": "Do X", "count": 10}

        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker.log_summary(summary)

        # Numeric values logged as metrics
        metrics_call = mock_mlflow.log_metrics.call_args[0][0]
        assert "best_score" in metrics_call
        assert "count" in metrics_call
        assert "best_prompt" not in metrics_call

        # String values logged as params with summary/ prefix
        params_call = mock_mlflow.log_params.call_args[0][0]
        assert "summary/best_prompt" in params_call
        assert params_call["summary/best_prompt"] == "Do X"

    def test_no_backends_no_error(self):
        tracker = ExperimentTracker(use_wandb=False, use_mlflow=False)
        tracker.log_summary({"key": "value"})

    def test_wandb_error_handled(self):
        tracker = ExperimentTracker(use_wandb=True)

        mock_wandb = MagicMock()
        mock_wandb.run.summary.__setitem__.side_effect = RuntimeError("wandb error")
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker.log_summary({"key": "value"})
