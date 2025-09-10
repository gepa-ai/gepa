# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any


class ExperimentTracker:
    """
    Unified experiment tracking that supports both wandb and mlflow.
    """

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - always end the run."""
        self.end_run()
        return False  # Don't suppress exceptions

    def __init__(
        self,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        use_mlflow: bool = False,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
    ):
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow

        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs or {}
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name

    def initialize(self):
        """Initialize the logging backends."""
        if self.use_wandb:
            self._initialize_wandb()
        if self.use_mlflow:
            self._initialize_mlflow()

    def _initialize_wandb(self):
        """Initialize wandb."""
        try:
            import wandb  # type: ignore
            if self.wandb_api_key:
                wandb.login(key=self.wandb_api_key, verify=True)
            else:
                wandb.login()
        except ImportError:
            raise ImportError("wandb is not installed. Please install it or set backend='mlflow' or 'none'.")
        except Exception as e:
            raise RuntimeError(f"Error logging into wandb: {e}")

    def _initialize_mlflow(self):
        """Initialize mlflow."""
        try:
            import mlflow  # type: ignore
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            if self.mlflow_experiment_name:
                mlflow.set_experiment(self.mlflow_experiment_name)
        except ImportError:
            raise ImportError("mlflow is not installed. Please install it or set backend='wandb' or 'none'.")
        except Exception as e:
            raise RuntimeError(f"Error setting up mlflow: {e}")

    def start_run(self):
        """Start a new run."""
        if self.use_wandb:
            import wandb  # type: ignore
            wandb.init(**self.wandb_init_kwargs)
        if self.use_mlflow:
            import mlflow  # type: ignore
            mlflow.start_run(nested=True)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """Log metrics to the active backends."""
        if self.use_wandb:
            try:
                import wandb  # type: ignore
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log to mlflow: {e}")

    def end_run(self):
        """End the current run."""
        if self.use_wandb:
            try:
                import wandb  # type: ignore
                if wandb.run is not None:
                    wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to end wandb run: {e}")

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore
                if mlflow.active_run() is not None:
                    mlflow.end_run()
            except Exception as e:
                print(f"Warning: Failed to end mlflow run: {e}")

    def is_active(self) -> bool:
        """Check if any backend has an active run."""
        if self.use_wandb:
            try:
                import wandb  # type: ignore
                if wandb.run is not None:
                    return True
            except Exception:
                pass

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore
                if mlflow.active_run() is not None:
                    return True
            except Exception:
                pass

        return False


def create_experiment_tracker(
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
) -> ExperimentTracker:
    """
    Create an experiment tracker based on the specified backends.

    Args:
        use_wandb: Whether to use wandb
        use_mlflow: Whether to use mlflow
        wandb_api_key: API key for wandb
        wandb_init_kwargs: Additional kwargs for wandb.init()
        mlflow_tracking_uri: Tracking URI for mlflow
        mlflow_experiment_name: Experiment name for mlflow

    Returns:
        ExperimentTracker instance

    Note:
        Both wandb and mlflow can be used simultaneously if desired.
    """
    return ExperimentTracker(
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
