"""Adapter for optimizing prompts in single-model iterations."""

from __future__ import annotations

from typing import Any

from glean_gepa.adapter_types import (
    SingleModelALDataInst,
    SingleModelALRolloutOutput,
    SingleModelALTrajectory,
)
from glean_gepa.al_adapter import ALRunner, GleanAdapterBase, Thresholds


class SingleModelAdapter(GleanAdapterBase):
    """Optimize prompts for a single student model using shell-tool error evidence."""

    def __init__(
        self,
        runner: ALRunner,
        thresholds: Thresholds,
        student_model: str,
        *,
        bigquery_client: Any | None = None,
        shell_error_lookback_days: int = 7,
        cache_file: str | None = None,
    ):
        super().__init__(
            runner=runner,
            teacher_model="",
            thresholds=thresholds,
            student_model=student_model,
            judging_mode="single_model",
            bigquery_client=bigquery_client,
            shell_error_lookback_days=shell_error_lookback_days,
            cache_file=cache_file,
        )


__all__ = [
    "SingleModelALDataInst",
    "SingleModelALRolloutOutput",
    "SingleModelALTrajectory",
    "SingleModelAdapter",
]
