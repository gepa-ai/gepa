import os
import shutil
import tempfile
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from gepa.logging.experiment_tracker import ExperimentTracker, create_experiment_tracker


def has_wandb():
    """Check if wandb is available."""
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def has_mlflow():
    """Check if mlflow is available."""
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


class TestCreateExperimentTracker:
    """Test cases for create_experiment_tracker function."""

    def test_create_wandb_only(self):
        """Test creating tracker with wandb only."""
        tracker = create_experiment_tracker(
            use_wandb=True,
            wandb_api_key="test_key",
            use_mlflow=False,
        )

        assert isinstance(tracker, ExperimentTracker)
        assert tracker.use_wandb is True
        assert tracker.use_mlflow is False
        assert tracker.wandb_api_key == "test_key"

    def test_create_mlflow_only(self):
        """Test creating tracker with mlflow only."""
        tracker = create_experiment_tracker(
            use_wandb=False,
            use_mlflow=True,
            mlflow_tracking_uri="file:///tmp/mlflow",
        )

        assert isinstance(tracker, ExperimentTracker)
        assert tracker.use_wandb is False
        assert tracker.use_mlflow is True
        assert tracker.mlflow_tracking_uri == "file:///tmp/mlflow"

    def test_create_both_backends(self):
        """Test creating tracker with both backends."""
        tracker = create_experiment_tracker(
            use_wandb=True,
            wandb_api_key="test_key",
            use_mlflow=True,
            mlflow_tracking_uri="file:///tmp/mlflow",
        )

        assert isinstance(tracker, ExperimentTracker)
        assert tracker.use_wandb is True
        assert tracker.use_mlflow is True

    def test_create_no_backends(self):
        """Test creating tracker with no backends."""
        tracker = create_experiment_tracker(
            use_wandb=False,
            use_mlflow=False,
        )

        assert isinstance(tracker, ExperimentTracker)
        assert tracker.use_wandb is False
        assert tracker.use_mlflow is False

    def test_create_experiment_tracker_factory(self):
        """Test the create_experiment_tracker factory function."""
        # Test with no backends
        tracker1 = create_experiment_tracker(use_wandb=False, use_mlflow=False)
        assert isinstance(tracker1, ExperimentTracker)
        assert tracker1.use_wandb is False
        assert tracker1.use_mlflow is False

        # Test with wandb only (if available)
        if has_wandb():
            tracker2 = create_experiment_tracker(
                use_wandb=True,
                wandb_api_key="test_key",
                use_mlflow=False,
            )
            assert isinstance(tracker2, ExperimentTracker)
            assert tracker2.use_wandb is True
            assert tracker2.use_mlflow is False
            assert tracker2.wandb_api_key == "test_key"

        # Test with mlflow only (if available)
        if has_mlflow():
            tracker3 = create_experiment_tracker(
                use_wandb=False,
                use_mlflow=True,
                mlflow_tracking_uri="file:///tmp/mlflow",
            )
            assert isinstance(tracker3, ExperimentTracker)
            assert tracker3.use_wandb is False
            assert tracker3.use_mlflow is True
            assert tracker3.mlflow_tracking_uri == "file:///tmp/mlflow"


class TestExperimentTrackerIntegration:
    """Integration tests using real libraries with offline mode."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_wandb(self):
        """Mock wandb."""
        if not has_wandb():
            pytest.skip("wandb not available")
        if 'wandb' in sys.modules:
            del sys.modules['wandb']
        wandb = MagicMock()
        def finish():
            wandb.run = None
        wandb.finish.side_effect = finish
        with patch.dict("sys.modules", {"wandb": wandb}):
            yield wandb

    def test_no_backends_works(self):
        """Test that no backends configuration works."""
        tracker = create_experiment_tracker(
            use_wandb=False,
            use_mlflow=False,
        )

        assert isinstance(tracker, ExperimentTracker)
        assert tracker.use_wandb is False
        assert tracker.use_mlflow is False

        # Should work with context manager
        with tracker:
            tracker.log_metrics({"test": 1.0}, step=1)

        assert not tracker.is_active()

    @pytest.mark.skipif(not has_wandb(), reason="wandb not available")
    def test_wandb_offline_initialization(self, mock_wandb, temp_dir):
        """Test wandb initialization in offline mode."""
        tracker = ExperimentTracker(
            use_wandb=True,
            wandb_init_kwargs={
                "project": "test-project",
                "dir": temp_dir,
            },
            use_mlflow=False,
        )

        # Should initialize without errors
        tracker.initialize()
        tracker.start_run()

        # Should be able to log metrics
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
        tracker.log_metrics({"loss": 0.4, "accuracy": 0.95}, step=2)
        
        # Verify wandb.log was called correctly
        assert mock_wandb.log.call_count == 2
        assert mock_wandb.log.call_args_list[0][0][0] == {"loss": 0.5, "accuracy": 0.9}
        assert mock_wandb.log.call_args_list[0][1]["step"] == 1
        assert mock_wandb.log.call_args_list[1][0][0] == {"loss": 0.4, "accuracy": 0.95}
        assert mock_wandb.log.call_args_list[1][1]["step"] == 2

        # Should be active
        assert tracker.is_active()

        # End run to finalize logging
        tracker.end_run()

        # Should not be active after ending
        assert not tracker.is_active()

    @pytest.mark.skipif(not has_mlflow(), reason="mlflow not available")
    def test_mlflow_initialization(self, temp_dir):
        """Test mlflow initialization with local tracking."""
        tracker = ExperimentTracker(
            use_wandb=False,
            use_mlflow=True,
            mlflow_tracking_uri=f"file://{temp_dir}/mlflow",
            mlflow_experiment_name="test-experiment",
        )

        # Should initialize without errors
        tracker.initialize()
        tracker.start_run()

        # Should be able to log metrics
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
        tracker.log_metrics({"loss": 0.4, "accuracy": 0.95}, step=2)

        # Should be active
        assert tracker.is_active()

        # Verify metrics were logged by checking mlflow run data
        import mlflow
        run = mlflow.active_run()
        assert run is not None
        assert run.info.run_id is not None

        # End run to finalize logging
        tracker.end_run()

        # Should not be active after ending
        assert not tracker.is_active()

        # Verify mlflow tracking directory was created
        mlflow_dir = Path(temp_dir) / "mlflow"
        assert mlflow_dir.exists()

        # Check that metrics were stored in the mlflow tracking store
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=f"file://{temp_dir}/mlflow")

        # Get the experiment
        experiment = client.get_experiment_by_name("test-experiment")
        assert experiment is not None

        # Get runs for this experiment
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0

        # Get the latest run
        run = runs[0]
        assert run.data.metrics is not None

        # Verify our metrics are in the run data
        metrics = run.data.metrics
        assert "loss" in metrics
        assert "accuracy" in metrics
        # Note: mlflow stores the latest value for each metric
        assert metrics["loss"] == 0.4  # Last logged value
        assert metrics["accuracy"] == 0.95  # Last logged value

    @pytest.mark.skipif(not has_wandb() or not has_mlflow(), reason="wandb or mlflow not available")
    def test_both_backends_offline(self, mock_wandb, temp_dir):
        """Test using both wandb and mlflow simultaneously in offline mode."""
        tracker = ExperimentTracker(
            use_wandb=True,
            wandb_init_kwargs={
                "project": "test-project",
                "dir": temp_dir,
            },
            use_mlflow=True,
            mlflow_tracking_uri=f"file://{temp_dir}/mlflow",
            mlflow_experiment_name="test-experiment",
        )

        # Should initialize both backends
        tracker.initialize()
        tracker.start_run()

        # Should be able to log metrics to both backends
        with patch("wandb.log") as mock_wandb_log:
            tracker.log_metrics({"loss": 0.4, "accuracy": 0.95})
            
            # Verify wandb was called
            mock_wandb_log.assert_called_once_with({"loss": 0.4, "accuracy": 0.95}, step=None)

        # Should be active
        assert tracker.is_active()

        # Check wandb
        import wandb
        wandb_run = wandb.run
        assert wandb_run is not None

        # Check mlflow
        import mlflow
        mlflow_run = mlflow.active_run()
        assert mlflow_run is not None
        assert mlflow_run.info.run_id is not None

        # End both runs cleanly
        tracker.end_run()

        # Should not be active after ending
        assert not tracker.is_active()

        # Verify mlflow metrics were stored
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=f"file://{temp_dir}/mlflow")
        experiment = client.get_experiment_by_name("test-experiment")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0

        # Check mlflow metrics
        mlflow_metrics = runs[0].data.metrics
        assert "loss" in mlflow_metrics
        assert "accuracy" in mlflow_metrics
        assert mlflow_metrics["loss"] == 0.4
        assert mlflow_metrics["accuracy"] == 0.95

    @pytest.mark.skipif(not has_wandb(), reason="wandb not available")
    def test_context_manager_wandb(self, mock_wandb, temp_dir):
        """Test context manager with wandb in offline mode."""
        tracker = ExperimentTracker(
            use_wandb=True,
            wandb_init_kwargs={
                "project": "test-project",
                "dir": temp_dir,
            },
            use_mlflow=False,
        )

        # Test context manager workflow
        with tracker:
            assert tracker.is_active()
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.log_metrics({"accuracy": 0.9}, step=2)
            
            # Verify wandb.log was called correctly
            assert mock_wandb.log.call_count == 2
            assert mock_wandb.log.call_args_list[0][0][0] == {"loss": 0.5}
            assert mock_wandb.log.call_args_list[0][1]["step"] == 1
            assert mock_wandb.log.call_args_list[1][0][0] == {"accuracy": 0.9}
            assert mock_wandb.log.call_args_list[1][1]["step"] == 2

        # Should not be active after context exit
        assert not tracker.is_active()

    @pytest.mark.skipif(not has_mlflow(), reason="mlflow not available")
    def test_context_manager_mlflow(self, temp_dir):
        """Test context manager with mlflow."""
        tracker = ExperimentTracker(
            use_wandb=False,
            use_mlflow=True,
            mlflow_tracking_uri=f"file://{temp_dir}/mlflow",
            mlflow_experiment_name="test-experiment",
        )

        # Test context manager workflow
        with tracker:
            assert tracker.is_active()
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.log_metrics({"accuracy": 0.9}, step=2)

        # Should not be active after context exit
        assert not tracker.is_active()

    @pytest.mark.skipif(not has_wandb() or not has_mlflow(), reason="wandb or mlflow not available")
    def test_context_manager_both_backends(self, mock_wandb, temp_dir):
        """Test context manager with both backends."""
        tracker = ExperimentTracker(
            use_wandb=True,
            wandb_init_kwargs={
                "project": "test-project",
                "dir": temp_dir,
            },
            use_mlflow=True,
            mlflow_tracking_uri=f"file://{temp_dir}/mlflow",
            mlflow_experiment_name="test-experiment",
        )

    
        with tracker:
            assert tracker.is_active()
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.log_metrics({"accuracy": 0.9}, step=2)
            
            # Verify wandb was called correctly
            assert mock_wandb.log.call_count == 2
            assert mock_wandb.log.call_args_list[0][0][0] == {"loss": 0.5}
            assert mock_wandb.log.call_args_list[0][1]["step"] == 1
            assert mock_wandb.log.call_args_list[1][0][0] == {"accuracy": 0.9}
            assert mock_wandb.log.call_args_list[1][1]["step"] == 2

        # Should not be active after context exit
        assert not tracker.is_active()

        # Verify mlflow metrics were stored
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=f"file://{temp_dir}/mlflow")
        experiment = client.get_experiment_by_name("test-experiment")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0

        # Check mlflow metrics
        mlflow_metrics = runs[0].data.metrics
        assert "loss" in mlflow_metrics
        assert "accuracy" in mlflow_metrics
        assert mlflow_metrics["loss"] == 0.5
        assert mlflow_metrics["accuracy"] == 0.9

    @pytest.mark.skipif(not has_wandb(), reason="wandb not available")
    def test_context_manager_with_exception_wandb(self, mock_wandb, temp_dir):
        """Test context manager with exception - should still clean up."""
        tracker = ExperimentTracker(
            use_wandb=True,
            wandb_init_kwargs={
                "project": "test-project",
                "dir": temp_dir,
            },
            use_mlflow=False,
        )

        with pytest.raises(ValueError):
            with tracker:
                tracker.log_metrics({"test": 1.0}, step=1)
                raise ValueError("test exception")
        
        # Verify wandb.log was called before the exception
        mock_wandb.log.assert_called_once_with({"test": 1.0}, step=1)

        # Should not be active after exception
        assert not tracker.is_active()

    @pytest.mark.skipif(not has_mlflow(), reason="mlflow not available")
    def test_context_manager_with_exception_mlflow(self, temp_dir):
        """Test context manager with exception - should still clean up."""
        tracker = ExperimentTracker(
            use_wandb=False,
            use_mlflow=True,
            mlflow_tracking_uri=f"file://{temp_dir}/mlflow",
            mlflow_experiment_name="test-experiment",
        )

        with pytest.raises(ValueError):
            with tracker:
                tracker.log_metrics({"test": 1.0}, step=1)
                raise ValueError("test exception")


    @pytest.mark.skipif(not has_mlflow(), reason="mlflow not available")
    def test_mlflow_experiment_creation(self, temp_dir):
        """Test that mlflow experiments are created correctly."""
        tracker = ExperimentTracker(
            use_wandb=False,
            use_mlflow=True,
            mlflow_tracking_uri=f"file://{temp_dir}/mlflow",
            mlflow_experiment_name="test-experiment",
        )

        tracker.initialize()

        # Check that mlflow tracking directory was created
        mlflow_dir = Path(temp_dir) / "mlflow"
        assert mlflow_dir.exists()

        with tracker:
            tracker.log_metrics({"test": 1.0}, step=1)

    @pytest.mark.skipif(not has_wandb(), reason="wandb not available")
    def test_metric_logging_variations_wandb(self, mock_wandb, temp_dir):
        """Test various metric logging scenarios with wandb."""
        tracker = ExperimentTracker(
            use_wandb=True,
            wandb_init_kwargs={
                "project": "test-project",
                "dir": temp_dir,
            },
            use_mlflow=False,
        )

        with tracker:
            # Test different metric types
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.log_metrics({"accuracy": 0.9, "f1": 0.85}, step=2)
            tracker.log_metrics({"learning_rate": 0.001}, step=3)

            # Test without step
            tracker.log_metrics({"final_loss": 0.1})

            # Test with None step
            tracker.log_metrics({"test_metric": 42}, step=None)

        # Verify all metrics were logged to wandb
        assert mock_wandb.log.call_count == 5
        
        # Check each call
        expected_calls = [
            ({"loss": 0.5}, {"step": 1}),
            ({"accuracy": 0.9, "f1": 0.85}, {"step": 2}),
            ({"learning_rate": 0.001}, {"step": 3}),
            ({"final_loss": 0.1}, {"step": None}),
            ({"test_metric": 42}, {"step": None}),
        ]
        
        for i, (expected_metrics, expected_kwargs) in enumerate(expected_calls):
            call_args, call_kwargs = mock_wandb.log.call_args_list[i]
            assert call_args[0] == expected_metrics
            assert call_kwargs == expected_kwargs

    @pytest.mark.skipif(not has_mlflow(), reason="mlflow not available")
    def test_metric_logging_variations_mlflow(self, temp_dir):
        """Test various metric logging scenarios with mlflow."""
        tracker = ExperimentTracker(
            use_wandb=False,
            use_mlflow=True,
            mlflow_tracking_uri=f"file://{temp_dir}/mlflow",
            mlflow_experiment_name="test-experiment",
        )

        with tracker:
            # Test different metric types
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.log_metrics({"accuracy": 0.9, "f1": 0.85}, step=2)
            tracker.log_metrics({"learning_rate": 0.001}, step=3)

            # Test without step
            tracker.log_metrics({"final_loss": 0.1})

            # Test with None step
            tracker.log_metrics({"test_metric": 42}, step=None)

        # Verify all metrics were logged to mlflow
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=f"file://{temp_dir}/mlflow")
        experiment = client.get_experiment_by_name("test-experiment")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0

        # Get the latest run and check metrics
        run = runs[0]
        metrics = run.data.metrics
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "learning_rate" in metrics
        assert "final_loss" in metrics
        assert "test_metric" in metrics

        # Verify metric values (mlflow stores the latest value for each metric)
        assert metrics["loss"] == 0.5
        assert metrics["accuracy"] == 0.9
        assert metrics["f1"] == 0.85
        assert metrics["learning_rate"] == 0.001
        assert metrics["final_loss"] == 0.1
        assert metrics["test_metric"] == 42
