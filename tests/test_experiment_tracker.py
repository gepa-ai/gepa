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
