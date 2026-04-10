"""Experiment tracking for Terrarium runs (wandb and/or mlflow)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gepa.core.callbacks import (
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    IterationEndEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    ValsetEvaluatedEvent,
)
from gepa.logging.experiment_tracker import ExperimentTracker


@dataclass
class TrackingConfig:
    """Configuration for terrarium experiment tracking."""

    use_wandb: bool = False
    wandb_project: str = "terrarium"
    wandb_entity: str | None = None
    wandb_api_key: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    use_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "terrarium"
    mlflow_run_name: str | None = None


class TerrariumTracker:
    """Thin wrapper around GEPA's ExperimentTracker for terrarium-level logging.

    Logs per-eval scores, best-score trajectory, and run summary.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        wandb_init_kwargs: dict[str, Any] = {"project": config.wandb_project}
        if config.wandb_entity:
            wandb_init_kwargs["entity"] = config.wandb_entity
        if config.wandb_tags:
            wandb_init_kwargs["tags"] = config.wandb_tags

        self._tracker = ExperimentTracker(
            use_wandb=config.use_wandb,
            wandb_api_key=config.wandb_api_key,
            wandb_init_kwargs=wandb_init_kwargs,
            use_mlflow=config.use_mlflow,
            mlflow_tracking_uri=config.mlflow_tracking_uri,
            mlflow_experiment_name=config.mlflow_experiment_name,
            mlflow_attach_existing=True,
        )
        self._owns_mlflow_run = False

    @property
    def active(self) -> bool:
        return self.config.use_wandb or self.config.use_mlflow

    def start(self, run_config: dict[str, Any]) -> None:
        """Initialize backends and log run config."""
        if not self.active:
            return
        self._tracker.initialize()
        # Start the mlflow run ourselves so we can set run_name,
        # then let the tracker attach to it (mlflow_attach_existing=True).
        if self.config.use_mlflow:
            import mlflow

            if mlflow.active_run() is None:
                mlflow.start_run(run_name=self.config.mlflow_run_name)
                self._owns_mlflow_run = True
        self._tracker.start_run()
        self._tracker.log_config(run_config)

    def log_eval(self, eval_num: int, score: float, best_score: float) -> None:
        """Log a single evaluation step."""
        if not self.active:
            return
        self._tracker.log_metrics(
            {"eval/score": score, "eval/best_score": best_score},
            step=eval_num,
        )

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log final run summary."""
        if not self.active:
            return
        self._tracker.log_summary(summary)

    def end(self) -> None:
        """End the tracking run."""
        if not self.active:
            return
        self._tracker.end_run()
        if self._owns_mlflow_run:
            import mlflow

            if mlflow.active_run() is not None:
                mlflow.end_run()
            self._owns_mlflow_run = False

    def create_callback(self) -> TrackingCallback:
        """Create a GEPA callback that logs iteration-level metrics."""
        return TrackingCallback(self._tracker)


class TrackingCallback:
    """GEPA callback that logs iteration and valset metrics to wandb/mlflow.

    Plugs into GEPA's callback system via GEPAConfig.callbacks to capture
    per-iteration data: valset scores, candidate acceptance, and best scores.
    """

    def __init__(self, tracker: ExperimentTracker) -> None:
        self._tracker = tracker

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        self._tracker.log_config({
            "gepa/trainset_size": event["trainset_size"],
            "gepa/valset_size": event["valset_size"],
        })

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        iteration = event["iteration"]
        self._tracker.log_metrics(
            {
                "iteration/valset_score": event["average_score"],
                "iteration/candidate_idx": event["candidate_idx"],
                "iteration/num_val_examples": event["num_examples_evaluated"],
                "iteration/is_best": int(event["is_best_program"]),
            },
            step=iteration,
        )

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        self._tracker.log_metrics(
            {
                "iteration/accepted_score": event["new_score"],
                "iteration/accepted": 1,
            },
            step=event["iteration"],
        )

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        self._tracker.log_metrics(
            {
                "iteration/rejected_new_score": event["new_score"],
                "iteration/rejected_old_score": event["old_score"],
                "iteration/accepted": 0,
            },
            step=event["iteration"],
        )

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        state = event["state"]
        self._tracker.log_metrics(
            {
                "iteration/best_val_score": state.val_aggregate_scores[state.best_idx]
                if state.val_aggregate_scores
                else 0.0,
                "iteration/num_candidates": len(state.candidates),
                "iteration/metric_calls_used": state.metric_calls_used,
            },
            step=event["iteration"],
        )

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        final = event["final_state"]
        self._tracker.log_summary({
            "gepa/best_candidate_idx": event["best_candidate_idx"],
            "gepa/total_iterations": event["total_iterations"],
            "gepa/total_metric_calls": event["total_metric_calls"],
            "gepa/final_best_val_score": final.val_aggregate_scores[final.best_idx]
            if final.val_aggregate_scores
            else 0.0,
            "gepa/num_candidates": len(final.candidates),
        })
