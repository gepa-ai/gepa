"""GEPA adapter: runs optimize_anything as the evolution backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from terrarium.adapter import Result
from terrarium.budget import BudgetExhausted
from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer


class GEPAAdapter:
    """Adapter that uses GEPA's optimize_anything as the evolution engine.

    All evaluations go through the terrarium EvalServer (via its direct
    Python API — no HTTP overhead). Budget is enforced by the server.

    Usage::

        from terrarium import run
        from terrarium.adapters.gepa import GEPAAdapter

        adapter = GEPAAdapter(reflection_lm="openai/gpt-5")
        result = run("circle_packing", adapter, max_evals=150)
    """

    def __init__(
        self,
        reflection_lm: str = "openai/gpt-5",
        run_dir: str = "outputs/terrarium",
        parallel: bool = True,
        max_workers: int = 16,
        callbacks: list[Any] | None = None,
        **engine_kwargs: Any,
    ) -> None:
        self.reflection_lm = reflection_lm
        self.run_dir = run_dir
        self.parallel = parallel
        self.max_workers = max_workers
        self.callbacks = callbacks or []
        self.engine_kwargs = engine_kwargs

    def evolve(self, task: Task, server: EvalServer, max_evals: int) -> Result:
        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

        config = GEPAConfig(
            engine=EngineConfig(
                run_dir=f"{self.run_dir}/{task.name}",
                max_metric_calls=max_evals,
                cache_evaluation=True,
                track_best_outputs=True,
                parallel=self.parallel,
                max_workers=self.max_workers,
                **self.engine_kwargs,
            ),
            reflection=ReflectionConfig(reflection_lm=self.reflection_lm),
            callbacks=self.callbacks if self.callbacks else None,
        )

        # Bridge: optimize_anything calls this evaluator, which goes through
        # the terrarium server (same budget counter as HTTP endpoint).
        def evaluator(candidate, example=None, **kwargs):
            return server.evaluate(candidate, example)

        oa_kwargs: dict[str, Any] = {
            "seed_candidate": task.initial_candidate,
            "evaluator": evaluator,
            "config": config,
        }

        if task.has_dataset:
            if task.train_set:
                oa_kwargs["dataset"] = task.train_set
            if task.test_set:
                oa_kwargs["valset"] = task.test_set
            if "val_set" in task.metadata:
                oa_kwargs.setdefault("valset", task.metadata["val_set"])

        if "objective" in task.metadata:
            oa_kwargs["objective"] = task.metadata["objective"]
        if "background" in task.metadata:
            oa_kwargs["background"] = task.metadata["background"]

        try:
            gepa_result = optimize_anything(**oa_kwargs)
        except BudgetExhausted:
            gepa_result = None

        if gepa_result is not None:
            return Result(
                best_candidate=gepa_result.best_candidate,
                best_score=gepa_result.val_aggregate_scores[gepa_result.best_idx],
                total_evals=server.budget.used,
                eval_log=server.eval_log,
                metadata={"gepa_result": gepa_result},
            )

        return Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
        )


def create_adapter(**kwargs: Any) -> GEPAAdapter:
    """Factory for CLI usage: ``python -m terrarium <task> adapters/gepa.py``."""
    return GEPAAdapter(**kwargs)
