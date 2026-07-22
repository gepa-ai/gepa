"""GEPA engine: runs the archived GEPA optimizer against the optimize_anything eval server.

In-process — calls ``server.evaluate(candidate, example)`` directly. Budget is
enforced by the server. ``max_token_cost`` is enforced via the GEPA engine's
``max_reflection_cost`` stopper.

``OptimizeAnythingConfig.engine_config`` maps directly to a
:class:`~gepa.gepa_launcher.GEPAConfig`; the OA layer only overlays the eval
budget, ``run_dir``, and its own callbacks/stoppers on top.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gepa.oa.budget import BudgetExhausted
from gepa.oa.engine import Result

if TYPE_CHECKING:
    from gepa.oa.config import OptimizeAnythingConfig
    from gepa.oa.eval_server import EvalServer
    from gepa.oa.task import Task


class GepaEngine:
    """Runs GEPA's ``optimize_anything`` against an optimize_anything task.

    ``OptimizeAnythingConfig.engine_config`` maps directly to a
    :class:`~gepa.gepa_launcher.GEPAConfig` (see its fields for the options).
    The OA layer only overlays the eval budget, ``run_dir``, and its own
    callbacks/stoppers on top.
    """

    name = "gepa"

    def __init__(self, config: OptimizeAnythingConfig) -> None:
        from gepa.gepa_launcher import GEPAConfig

        # Cross-cutting (read directly off OptimizeAnythingConfig)
        self.run_dir = config.run_dir
        self.stop_at_score = config.stop_at_score
        # Proposer-cost cap: USD this engine may spend on its reflection LM /
        # agent. Enforced via GEPA's max_reflection_cost stopper; the eval
        # server never sees reflection spend.
        self.max_token_cost = config.max_token_cost
        # engine_config is a GEPAConfig-shaped dict, forwarded verbatim.
        # GEPAConfig(**...) validates it now (TypeError on an unknown key, so a
        # typo fails fast) and coerces nested dicts to their dataclasses. OA
        # overlays (budget, run_dir) are applied in run().
        self.gepa_config = GEPAConfig(**config.engine_config)

    def run(self, task: Task, server: EvalServer) -> Result:
        from gepa.gepa_launcher import optimize_anything

        budget = server.budget
        objective = task.objective
        background = task.background

        gepa_config = self.gepa_config

        # OA owns the eval budget, workspace, and the top-level convenience
        # limits — set them as EngineConfig fields; GEPA core installs the
        # matching stoppers. The eval-call cap must win over any user value.
        gepa_config.engine.max_metric_calls = budget.max_evals
        if self.run_dir is not None:
            gepa_config.engine.run_dir = self.run_dir
        if gepa_config.engine.run_dir is None:
            # Always persist GEPA state (gepa_state.bin, saved by core at each
            # iteration boundary): on BudgetExhausted the val-aggregate/Pareto
            # best is reloaded from it instead of falling back to the server's
            # per-example argmax.
            if server.output_dir is not None:
                gepa_config.engine.run_dir = str(server.output_dir / "gepa_state")
            else:
                gepa_config.engine.run_dir = tempfile.mkdtemp(prefix="gepa-state-")
        run_dir = gepa_config.engine.run_dir
        if self.stop_at_score is not None:
            gepa_config.engine.stop_at_score = self.stop_at_score

        # Resolve the reflection LM cost source: GEPA core builds the LM from a
        # string itself, but we need a live handle now to report reflection cost
        # on the Result and to set the proposer-cost cap. Build it here and place
        # it back so core reuses the same object. A custom proposer
        # (e.g. ClaudeCodeAgentProposer) is the cost source when present. It
        # starts fresh (zero cost) — we don't support reusing a pre-spent one.
        cost_source = self._resolve_cost_source(gepa_config)

        # Proposer-cost cap → EngineConfig.max_reflection_cost; core turns it
        # into a MaxReflectionCostStopper bound to the effective cost source.
        if self.max_token_cost is not None and gepa_config.engine.max_reflection_cost is None:
            gepa_config.engine.max_reflection_cost = self.max_token_cost

        # The optimize_anything eval server (server.evaluate) is our single
        # budget choke point — gepa.gepa_launcher's evaluator goes straight
        # through it.
        def evaluator(candidate, example=None, **kwargs):
            return server.evaluate(candidate, example, **kwargs)

        oa_kwargs: dict[str, Any] = {
            "seed_candidate": task.seed_candidate,
            "evaluator": evaluator,
            "config": gepa_config,
        }
        if server.batch_fn is not None:
            # Grouped stages (minibatch, valset, merge evals) flow through the
            # server's batch path in ONE user call per stage; opt_states pass
            # through when the user's batch function accepts them.
            def batch_evaluator(pairs, opt_states=None):
                return server.evaluate_batch(pairs, opt_states=opt_states)

            oa_kwargs["batch_evaluator"] = batch_evaluator
        if task.has_dataset:
            if task.train_set:
                oa_kwargs["dataset"] = task.train_set
            # val_set only — test_set is a held-out split reserved for
            # post-run eval and must never leak into the optimization loop.
            if task.val_set:
                oa_kwargs["valset"] = task.val_set
        if objective:
            oa_kwargs["objective"] = objective
        if background:
            oa_kwargs["background"] = background

        try:
            gepa_result = optimize_anything(**oa_kwargs)
        except BudgetExhausted:
            gepa_result = self._load_result_from_state(
                run_dir=run_dir,
                seed=gepa_config.engine.seed,
                str_candidate_mode=not isinstance(task.seed_candidate, dict),
            )

        # Reflection/proposer spend for this run, read straight off the cost
        # source's cumulative total_cost (it started fresh).
        adapter_cost = float(getattr(cost_source, "total_cost", 0.0) or 0.0) if cost_source is not None else 0.0

        if gepa_result is not None:
            best = gepa_result.best_candidate
            if isinstance(best, dict) and not isinstance(task.seed_candidate, dict):
                # str/seedless mode: unwrap GEPA's single-key internal form. A
                # legacy dict seed keeps its full multi-component candidate.
                best = next(iter(best.values()), "")
            return Result(
                best_candidate=cast(str, best),
                best_score=gepa_result.val_aggregate_scores[gepa_result.best_idx],
                total_evals=server.budget.used,
                eval_log=server.eval_log,
                metadata={"gepa_result": gepa_result, "adapter_cost": adapter_cost},
            )
        return Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={"adapter_cost": adapter_cost},
        )

    def process_result(self, result: Result, output_dir: Path | None) -> None:
        return

    def _resolve_cost_source(self, gepa_config: Any) -> Any | None:
        """Return the live LM/proposer whose ``total_cost`` we track and cap.

        A custom candidate proposer (e.g. ``ClaudeCodeAgentProposer``) is the
        cost source when set. Otherwise, if ``reflection.reflection_lm`` is a
        model-name string, build the ``LM`` now and place it back on the config
        so GEPA core reuses the same object (rather than building its own,
        which we couldn't then read cost from).
        """
        proposer = gepa_config.reflection.custom_candidate_proposer
        if proposer is not None:
            return proposer
        reflection_lm = gepa_config.reflection.reflection_lm
        if isinstance(reflection_lm, str):
            from gepa.lm import LM

            lm = LM(reflection_lm, **(gepa_config.reflection.reflection_lm_kwargs or {}))
            gepa_config.reflection.reflection_lm = lm
            return lm
        # Already a callable/None — core wraps callables in TrackingLM (cost
        # always 0.0), so there's nothing meaningful to track here.
        return reflection_lm if hasattr(reflection_lm, "total_cost") else None

    def _load_result_from_state(
        self,
        *,
        run_dir: str | Path | None,
        seed: int | None,
        str_candidate_mode: bool,
    ) -> Any | None:
        if run_dir is None:
            return None
        try:
            from gepa.core.result import GEPAResult
            from gepa.core.state import GEPAState

            state = GEPAState.load(str(run_dir))
            return GEPAResult.from_state(
                state,
                run_dir=str(run_dir),
                seed=seed,
                str_candidate_key="current_candidate" if str_candidate_mode else None,
            )
        except Exception:
            return None
