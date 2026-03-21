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
        wandb_attach_existing: bool = False,
        wandb_step_metric: str | None = None,
        use_mlflow: bool = False,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
        mlflow_attach_existing: bool = False,
        key_prefix: str = "",
    ):
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow

        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs or {}
        self.wandb_attach_existing = wandb_attach_existing
        self.wandb_step_metric = wandb_step_metric
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_attach_existing = mlflow_attach_existing
        self.key_prefix = key_prefix

        self._created_mlflow_run = False
        self._wandb_step_metric_defined = False

        # Accumulate table rows so each wandb.log() sends the full growing
        # table, not just the latest row.  Without this, commit=False causes
        # the pending dict to overwrite earlier single-row tables with the
        # newest one, and only the last row per commit cycle survives.
        self._wandb_table_rows: dict[str, tuple[list[str], list[list]]] = {}

    def _p(self, key: str) -> str:
        """Prepend key_prefix to a key/name, if one is set."""
        return f"{self.key_prefix}{key}" if self.key_prefix else key

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

    def _define_wandb_step_metric(self) -> None:
        """Declare a custom x-axis for all GEPA metrics in wandb.

        Called once on the first ``log_metrics`` call (not in ``start_run``)
        so that it works regardless of whether GEPA owns the ``wandb.init``
        call or is attaching to an existing run.

        When ``wandb_step_metric`` is set, all GEPA metrics are plotted
        against this custom step metric instead of wandb's global monotonic
        step counter.  This avoids conflicts when GEPA is embedded inside
        a host training loop that uses its own step counter.
        """
        if self._wandb_step_metric_defined or not self.wandb_step_metric:
            return
        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                wandb.define_metric(self.wandb_step_metric, hidden=False)
                # Scope the custom x-axis to GEPA's prefixed metrics only.
                # Using "*" would override the x-axis for ALL metrics in the
                # run, including the host's metrics (e.g. train/loss), causing
                # wandb to drop the host's data after GEPA runs.
                if self.key_prefix:
                    glob = f"{self.key_prefix}*"
                else:
                    glob = "*"
                wandb.define_metric(glob, step_metric=self.wandb_step_metric)
                self._wandb_step_metric_defined = True
        except Exception as e:
            print(f"Warning: Failed to define wandb step metric: {e}")

    def start_run(self):
        """Start a new run.

        When ``wandb_attach_existing=True`` the tracker skips ``wandb.init()``
        and logs into whatever run is already active in the process.
        When ``mlflow_attach_existing=True`` the tracker skips
        ``mlflow.start_run()`` and logs into the already-active MLflow run.
        In both cases ``end_run()`` will not terminate the run.
        """
        if self.use_wandb:
            if self.wandb_attach_existing:
                # Attach to the active run — no init, no finish later.
                pass
            else:
                import wandb  # type: ignore

                wandb.init(**self.wandb_init_kwargs)
        if self.use_mlflow:
            if self.mlflow_attach_existing:
                # Attach to the active run — no start, no end later.
                self._created_mlflow_run = False
            else:
                import mlflow  # type: ignore

                # Only start a new run if there's no active run
                if mlflow.active_run() is None:
                    mlflow.start_run()
                    self._created_mlflow_run = True
                else:
                    self._created_mlflow_run = False

    def log_config(self, config: dict[str, Any]) -> None:
        """Log run configuration/hyperparameters to the active backends.

        Args:
            config: Flat dict of config key-value pairs. Non-serializable values
                    are converted to strings.
        """
        safe_config = {}
        for k, v in config.items():
            if isinstance(v, bool | int | float | str | type(None)):
                safe_config[k] = v
            else:
                safe_config[k] = str(v)

        if self.use_wandb:
            try:
                import wandb  # type: ignore

                prefixed = {self._p(k): v for k, v in safe_config.items()}
                wandb.config.update(prefixed, allow_val_change=True)
            except Exception as e:
                print(f"Warning: Failed to log config to wandb: {e}")

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore

                # mlflow params must be strings
                str_params = {self._p(k): str(v) for k, v in safe_config.items()}
                mlflow.log_params(str_params)
            except Exception as e:
                print(f"Warning: Failed to log config to mlflow: {e}")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """Log metrics to the active backends."""
        if self.use_wandb:
            try:
                import wandb  # type: ignore

                # Lazily define the custom step metric on the first log call.
                self._define_wandb_step_metric()

                # Filter to numeric values only — non-numeric data (dicts, strings)
                # is logged via log_table() instead to avoid noisy flat charts
                numeric_metrics = {self._p(k): v for k, v in metrics.items() if isinstance(v, int | float)}
                if numeric_metrics:
                    if self.wandb_step_metric and step is not None:
                        # Use custom x-axis: inject the step as a metric value
                        # and omit the step= arg to avoid conflicting with the
                        # host run's global step counter.
                        numeric_metrics[self.wandb_step_metric] = step
                        wandb.log(numeric_metrics)
                    else:
                        wandb.log(numeric_metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore

                # MLflow only accepts numeric metrics, filter out non-numeric values
                numeric_metrics = {self._p(k): float(v) for k, v in metrics.items() if isinstance(v, int | float)}
                if numeric_metrics:
                    mlflow.log_metrics(numeric_metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log to mlflow: {e}")

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log run summary data (visible on the run overview page).

        Args:
            summary: Key-value pairs for the run summary. Supports strings,
                     numbers, and other serializable values.
        """
        if self.use_wandb:
            try:
                import wandb  # type: ignore

                for k, v in summary.items():
                    wandb.run.summary[self._p(k)] = v  # type: ignore[union-attr]
            except Exception as e:
                print(f"Warning: Failed to log summary to wandb: {e}")

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore

                # mlflow: numeric values as metrics, strings as params
                numeric = {self._p(k): float(v) for k, v in summary.items() if isinstance(v, int | float)}
                text = {self._p(k): str(v) for k, v in summary.items() if isinstance(v, str)}
                if numeric:
                    mlflow.log_metrics(numeric)
                if text:
                    mlflow.log_params({f"summary/{k}": v for k, v in text.items()})
            except Exception as e:
                print(f"Warning: Failed to log summary to mlflow: {e}")

    def log_table(self, table_name: str, columns: list[str], data: list[list[Any]]) -> None:
        """Log a table to the active backends.

        Args:
            table_name: Name/key for the table.
            columns: Column headers.
            data: Rows of data (each row is a list matching columns).
        """
        if self.use_wandb:
            try:
                import wandb  # type: ignore

                # Accumulate rows: each call appends to the stored rows for
                # this table, then logs the full growing table.  This ensures
                # all rows survive even when multiple log_table calls share
                # the same commit cycle (commit=False overwrites the pending
                # dict, so a single-row table would replace the previous one).
                key = self._p(table_name)
                if key not in self._wandb_table_rows:
                    self._wandb_table_rows[key] = (columns, list(data))
                else:
                    self._wandb_table_rows[key][1].extend(data)
                all_columns, all_rows = self._wandb_table_rows[key]
                table = wandb.Table(columns=all_columns, data=all_rows)
                wandb.log({key: table}, commit=False)
            except Exception as e:
                print(f"Warning: Failed to log table to wandb: {e}")

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore

                # mlflow.log_table expects a dict of column -> list of values
                table_dict = {col: [row[i] for row in data] for i, col in enumerate(columns)}
                mlflow.log_table(data=table_dict, artifact_file=f"{self._p(table_name)}.json")
            except Exception as e:
                print(f"Warning: Failed to log table to mlflow: {e}")

    def log_html(self, html_content: str, key: str = "candidate_tree") -> None:
        """Log an HTML string as a rich media artifact.

        Args:
            html_content: Self-contained HTML string.
            key: Artifact key / name used in the dashboard.
        """
        if self.use_wandb:
            try:
                import wandb  # type: ignore

                html_obj = wandb.Html(html_content)
                pkey = self._p(key)
                wandb.log({pkey: html_obj}, commit=False)
                # Also write to run summary so the panel always shows the latest tree
                wandb.run.summary[pkey] = html_obj  # type: ignore[union-attr]
            except Exception as e:
                print(f"Warning: Failed to log HTML to wandb: {e}")

        if self.use_mlflow:
            try:
                import tempfile

                import mlflow  # type: ignore

                with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
                    f.write(html_content)
                    tmp_path = f.name
                mlflow.log_artifact(tmp_path, artifact_path=self._p(key))
            except Exception as e:
                print(f"Warning: Failed to log HTML to mlflow: {e}")

    def end_run(self):
        """End the current run.

        When ``wandb_attach_existing=True`` or ``mlflow_attach_existing=True``
        the respective run is left open — the caller owns its lifecycle.
        """
        if self.use_wandb and not self.wandb_attach_existing:
            try:
                import wandb  # type: ignore

                if wandb.run is not None:
                    wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to end wandb run: {e}")

        if self.use_mlflow:
            try:
                import mlflow  # type: ignore

                if self._created_mlflow_run and mlflow.active_run() is not None:
                    mlflow.end_run()
                    self._created_mlflow_run = False
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
    wandb_attach_existing: bool = False,
    wandb_step_metric: str | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_attach_existing: bool = False,
    key_prefix: str = "",
) -> ExperimentTracker:
    """
    Create an experiment tracker based on the specified backends.

    Args:
        use_wandb: Whether to use wandb
        use_mlflow: Whether to use mlflow
        wandb_api_key: API key for wandb
        wandb_init_kwargs: Additional kwargs for wandb.init()
        wandb_attach_existing: When True, skip wandb.init() and wandb.finish()
            and log into the already-active run.
        wandb_step_metric: Custom x-axis metric name for wandb.  When set,
            GEPA uses ``wandb.define_metric`` to log all metrics against this
            custom step instead of wandb's global monotonic step counter.
            Required when embedding GEPA inside a host training loop that
            manages its own wandb step counter.
        mlflow_tracking_uri: Tracking URI for mlflow
        mlflow_experiment_name: Experiment name for mlflow
        mlflow_attach_existing: When True, skip mlflow.start_run() and
            mlflow.end_run() and log into the already-active run.

    Returns:
        ExperimentTracker instance

    Note:
        Both wandb and mlflow can be used simultaneously if desired.
    """
    return ExperimentTracker(
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        wandb_attach_existing=wandb_attach_existing,
        wandb_step_metric=wandb_step_metric,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_attach_existing=mlflow_attach_existing,
        key_prefix=key_prefix,
    )
