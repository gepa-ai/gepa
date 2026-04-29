"""GEPA backend: runs ``gepa.optimize_anything`` against the omni eval server.

In-process — calls ``server.evaluate(candidate, example)`` directly. Budget is
enforced by the server. ``max_token_cost`` is enforced via the GEPA engine's
``max_reflection_cost`` stopper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gepa.omni.backend import Result
from gepa.omni.budget import BudgetExhausted

if TYPE_CHECKING:
    from gepa.omni.eval_server import EvalServer
    from gepa.omni.task import Task


class GepaBackend:
    """Runs GEPA's ``optimize_anything`` against an omni task.

    Config layout mirrors :class:`gepa.optimize_anything.GEPAConfig`:

    - ``engine``: kwargs for ``EngineConfig``
    - ``reflection``: kwargs for ``ReflectionConfig``
    - ``merge``: kwargs for ``MergeConfig`` (or ``None`` to disable)
    - ``refiner``: kwargs for ``RefinerConfig`` (or ``None`` to disable)

    Plus top-level forwards (``stop_at_score``, ``max_thinking_tokens``,
    ``reflection_lm_kwargs``, ``objective``, ``background``). The api injects
    ``run_dir`` and ``max_metric_calls`` (= ``max_evals``).
    """

    name = "gepa"

    def __init__(
        self,
        *,
        run_dir: str | None = None,
        engine: dict[str, Any] | None = None,
        reflection: dict[str, Any] | None = None,
        merge: dict[str, Any] | None = None,
        refiner: dict[str, Any] | None = None,
        objective: str | None = None,
        background: str | None = None,
        callbacks: list[Any] | None = None,
        reflection_lm_kwargs: dict[str, Any] | None = None,
        stop_at_score: float | None = None,
        max_thinking_tokens: int | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.engine = dict(engine) if engine else {}
        self.reflection = dict(reflection) if reflection else {}
        self.merge = dict(merge) if merge else None
        self.refiner = dict(refiner) if refiner else None
        self.objective = objective
        self.background = background
        self.callbacks = callbacks or []
        self.reflection_lm_kwargs = dict(reflection_lm_kwargs) if reflection_lm_kwargs else {}
        self.stop_at_score = stop_at_score
        self.max_thinking_tokens = max_thinking_tokens

    def run(self, task: Task, server: EvalServer) -> Result:
        from gepa.lm import LM
        from gepa.optimize_anything import GEPAConfig, optimize_anything

        budget = server.budget

        reflection_kwargs = dict(self.reflection)
        reflection_lm: Any | None = None
        if "reflection_lm" in reflection_kwargs:
            lm_name = reflection_kwargs["reflection_lm"]
            if isinstance(lm_name, str):
                lm_kwargs = dict(self.reflection_lm_kwargs)
                if self.max_thinking_tokens is not None:
                    lm_kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.max_thinking_tokens}
                    lm_kwargs.pop("reasoning_effort", None)
                reflection_lm = LM(lm_name, **lm_kwargs)
                reflection_kwargs["reflection_lm"] = reflection_lm

        engine_kwargs: dict[str, Any] = {
            **self.engine,
            "run_dir": self.run_dir,
            "max_metric_calls": budget.max_evals,
        }
        if budget.max_token_cost is not None and reflection_lm is None:
            engine_kwargs["max_reflection_cost"] = budget.max_token_cost

        callbacks = list(self.callbacks)
        cost_callback: _ReflectionCostCallback | None = None
        if reflection_lm is not None:
            cost_callback = _ReflectionCostCallback(reflection_lm, server.tracker, output_dir=self.run_dir)
            callbacks.append(cost_callback)
        if task.val_set:
            callbacks.append(_ProgressCallback(server, reflection_lm=reflection_lm))

        stop_callbacks: list[Any] = []
        if self.stop_at_score is not None:
            from gepa.utils.stop_condition import ScoreThresholdStopper

            stop_callbacks.append(ScoreThresholdStopper(self.stop_at_score))
        if budget.max_token_cost is not None and reflection_lm is not None:
            from gepa.utils.stop_condition import MaxReflectionCostStopper

            stop_callbacks.append(MaxReflectionCostStopper(budget.max_token_cost, reflection_lm=reflection_lm))

        config = GEPAConfig(
            engine=engine_kwargs,
            reflection=reflection_kwargs,
            merge=self.merge,
            refiner=self.refiner,
            callbacks=callbacks or None,
            stop_callbacks=stop_callbacks or None,
        )

        # The omni eval server (server.evaluate) is our single budget choke
        # point — gepa.optimize_anything's evaluator goes straight through it.
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

        objective = self.objective or task.objective
        background = self.background or task.background
        if objective:
            oa_kwargs["objective"] = objective
        if background:
            oa_kwargs["background"] = background

        try:
            gepa_result = optimize_anything(**oa_kwargs)
        except BudgetExhausted:
            gepa_result = None

        adapter_cost = 0.0
        reflection_meta: dict[str, Any] = {}
        if cost_callback is not None:
            reflection_meta = {"reflection_cost_log": cost_callback.cost_log}
            if cost_callback.cost_log:
                adapter_cost = cost_callback.cost_log[-1]["reflection_cost"]

        if gepa_result is not None:
            return Result(
                best_candidate=gepa_result.best_candidate,
                best_score=gepa_result.val_aggregate_scores[gepa_result.best_idx],
                total_evals=server.budget.used,
                eval_log=server.eval_log,
                metadata={"gepa_result": gepa_result, "adapter_cost": adapter_cost, **reflection_meta},
            )
        return Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={"adapter_cost": adapter_cost, **reflection_meta},
        )

    def process_result(self, result: Result, output_dir: Path) -> None:
        return


class _ReflectionCostCallback:
    def __init__(self, lm: Any, tracker: Any | None, output_dir: str | Path | None = None) -> None:
        self._lm = lm
        self._tracker = tracker
        self._cost_log: list[dict[str, Any]] = []
        self._log_path: Path | None = None
        if output_dir is not None:
            self._log_path = Path(output_dir) / "reflection_cost_log.jsonl"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def cost_log(self) -> list[dict[str, Any]]:
        return list(self._cost_log)

    def on_iteration_end(self, event: dict[str, Any]) -> None:
        entry = {
            "iteration": event["iteration"],
            "reflection_cost": self._lm.total_cost,
            "reflection_tokens_in": self._lm.total_tokens_in,
            "reflection_tokens_out": self._lm.total_tokens_out,
        }
        self._cost_log.append(entry)
        if self._log_path is not None:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        if self._tracker is not None:
            try:
                self._tracker.log_metrics(
                    {
                        "reflection/cost": entry["reflection_cost"],
                        "reflection/tokens_in": entry["reflection_tokens_in"],
                        "reflection/tokens_out": entry["reflection_tokens_out"],
                    },
                    step=entry["iteration"],
                )
            except Exception:
                pass


class _ProgressCallback:
    def __init__(self, server: Any, reflection_lm: Any = None) -> None:
        self._server = server
        self._reflection_lm = reflection_lm

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        candidate_dict = event.get("candidate", {})
        candidate_text = next(iter(candidate_dict.values()), None) if candidate_dict else None
        reflection_cost = self._reflection_lm.total_cost if self._reflection_lm else 0.0
        self._server.log_progress(event["average_score"], candidate=candidate_text, reflection_cost=reflection_cost)
