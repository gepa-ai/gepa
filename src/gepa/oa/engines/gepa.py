"""GEPA engine: runs the archived GEPA optimizer against the optimize_anything eval server.

In-process — evaluates through the server, which enforces the eval budget by
reservation. Validations are reserved per candidate, in selection order, so a
step whose selected children do not all fit validates as many as do and rejects
the rest before any of their validation runs; minibatch and merge screens are
reserved in bulk; core stops when the remainder cannot carry one more proposal
to a commit.
``max_token_cost`` is enforced via the GEPA engine's ``max_reflection_cost``
stopper.

``OptimizeAnythingConfig.engine_config`` maps directly to a
:class:`~gepa.gepa_launcher.GEPAConfig`; the OA layer only overlays the eval
budget, ``run_dir``, and its own callbacks/stoppers on top.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gepa.oa.budget import BudgetExhausted, BudgetTracker
from gepa.oa.engine import Result

if TYPE_CHECKING:
    from gepa.oa.config import OptimizeAnythingConfig
    from gepa.oa.eval_server import EvalServer
    from gepa.oa.task import Task


class _ServerEvalBudget:
    """Reservation view of the eval server's budget for GEPA core (``GEPAAdapter.eval_budget``).

    Core reserves each candidate's validation here before sending it. The
    reservation is held on the server's tracker, and the batch evaluator
    below runs the next grouped call against it (``reserved=True``) instead of
    reserving again. Stages core does not pre-reserve (minibatch screens, merge
    screens) still reserve in bulk on the server as before.
    """

    def __init__(self, budget: BudgetTracker, *, isolate_errors: bool = False, sequential: bool = False) -> None:
        self.budget = budget
        self.isolate_errors = isolate_errors
        self.sequential = sequential
        self.outstanding = 0  # evals core reserved for the next grouped call
        self.bulk_refusals = 0  # server refusals of calls core did not pre-reserve

    def reserve(self, n: int) -> bool:
        try:
            self.budget.reserve(n)
        except BudgetExhausted:
            return False
        self.outstanding += n
        return True

    def release(self, n: int) -> None:
        self.budget.release(n)
        self.outstanding = max(0, self.outstanding - n)

    def settle(self) -> None:
        """Core's grouped call is over: whatever it reserved but never sent goes back
        (the adapter's cache may have served every pair, so run_batch never ran)."""
        if self.outstanding:
            self.budget.release(self.outstanding)
            self.outstanding = 0

    @property
    def remaining(self) -> int | None:
        return self.budget.remaining

    def run_batch(self, server: EvalServer, pairs: list[Any], opt_states: Any) -> Any:
        """Run one grouped call against core's outstanding reservation, or in bulk."""
        pre, self.outstanding = self.outstanding, 0
        if pre == 0:
            try:
                return server.evaluate_batch(
                    pairs, opt_states=opt_states, isolate_errors=self.isolate_errors, sequential=self.sequential
                )
            except BudgetExhausted:
                self.bulk_refusals += 1
                raise
        if pre < len(pairs):
            # Core reserved less than it sends (an adapter cache turned off between
            # calls, say): top the reservation up so the strict ceiling still holds.
            try:
                self.budget.reserve(len(pairs) - pre)
            except BudgetExhausted:
                self.budget.release(pre)
                self.bulk_refusals += 1
                raise
        elif pre > len(pairs):
            # Core reserved more than the adapter sends (its own cache served some
            # pairs): give the difference back before running.
            self.budget.release(pre - len(pairs))
        return server.evaluate_batch(
            pairs, opt_states=opt_states, reserved=True, isolate_errors=self.isolate_errors, sequential=self.sequential
        )


class _StopOnRefusedReservation:
    """GEPA stop callback: end the run at the next iteration boundary once the
    eval server has refused a stage core did not pre-reserve (a minibatch or
    merge screen). Per-candidate validation refusals are core's own decision
    and do not stop the run; its affordability check does."""

    def __init__(self, eval_budget: _ServerEvalBudget) -> None:
        self.eval_budget = eval_budget

    def __call__(self, gepa_state: Any) -> bool:
        return self.eval_budget.bulk_refusals > 0


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
        # Core reserves each candidate's validation through this view before
        # sending it (per-proposal reservation) and stops when the remainder
        # cannot carry one more proposal to a commit. A refused bulk stage
        # (minibatch or merge screen) is the backstop stop signal: with
        # raise_on_exception=True (default) the BudgetExhausted propagates and
        # run() answers from saved state; with raise_on_exception=False the
        # adapter converts it into zero scores instead, so also end the loop at
        # the next boundary.
        # The server's own fan-out (no user batch function) honours the engine
        # config as the per-pair path did: a failing example is isolated when
        # raise_on_exception is off, and pairs run one at a time when parallel
        # is off (an evaluator that is not thread-safe).
        eval_budget = _ServerEvalBudget(
            budget,
            isolate_errors=not gepa_config.engine.raise_on_exception,
            sequential=not gepa_config.engine.parallel,
        )
        stoppers = gepa_config.stop_callbacks
        stoppers = [] if stoppers is None else list(stoppers) if isinstance(stoppers, Sequence) else [stoppers]
        gepa_config.stop_callbacks = [s for s in stoppers if not isinstance(s, _StopOnRefusedReservation)] + [
            _StopOnRefusedReservation(eval_budget)
        ]
        if self.run_dir is not None:
            gepa_config.engine.run_dir = self.run_dir
        if gepa_config.engine.run_dir is None:
            # Always persist GEPA state (gepa_state.bin, saved at each iteration
            # boundary) so a BudgetExhausted is answered from the val-aggregate/
            # Pareto best rather than the server's per-example argmax.
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

        # The eval server is the single budget choke point. Grouped stages
        # (seed/valset passes, minibatches, merge evals) take the batch path so
        # the whole stage is reserved at once and refused before any spend
        # when it does not fit; the per-pair evaluator serves singleton
        # resolutions (refiner steps).
        def evaluator(candidate, example=None, **kwargs):
            return server.evaluate(candidate, example, **kwargs)

        def batch_evaluator(pairs, opt_states=None):
            return eval_budget.run_batch(server, pairs, opt_states)

        oa_kwargs: dict[str, Any] = {
            "seed_candidate": task.seed_candidate,
            "evaluator": evaluator,
            "config": gepa_config,
        }
        if not gepa_config.engine.capture_stdio:
            # capture_stdio scopes stdout/stderr per evaluation, which only the
            # per-pair path can do; those runs reserve pair by pair instead and
            # core does not pre-reserve for them.
            oa_kwargs["batch_evaluator"] = batch_evaluator
            if gepa_config.refiner is None:
                # A refiner evaluates per example through the per-pair path (its
                # retries make a candidate's cost unknowable up front), so core's
                # per-candidate reservation would never be consumed; those runs
                # keep per-pair reservation on the server.
                oa_kwargs["eval_budget"] = eval_budget
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

        from gepa.core.engine import EvalBudgetRefusedError

        try:
            gepa_result = optimize_anything(**oa_kwargs)
        except EvalBudgetRefusedError as refused:
            # A validation the loop cannot skip did not fit: the seed's on a
            # fresh run (nothing useful to report), or a new seed's on a resumed
            # run (the saved pool still holds a valid best).
            gepa_result = self._load_result_from_state(
                run_dir=run_dir,
                seed=gepa_config.engine.seed,
                str_candidate_mode=not isinstance(task.seed_candidate, dict),
            )
            if gepa_result is None:
                raise BudgetExhausted(str(refused)) from refused
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
            best_candidate=cast(str, server.best_candidate),
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={"adapter_cost": adapter_cost},
        )

    def process_result(self, result: Result, output_dir: Path | None) -> None:
        return

    def _resolve_cost_source(self, gepa_config: Any) -> Any | None:
        """Return the live LM/proposer whose ``total_cost`` we track and cap.

        A reflection strategy (e.g. ``ComBEEReflectionLM``) accumulates its own
        spend and is the cost source when set; a custom candidate proposer
        (e.g. ``ClaudeCodeAgentProposer``) likewise. The two are mutually
        exclusive (the proposer constructor raises if both are given).
        Otherwise, if ``reflection.reflection_lm`` is a model-name string,
        build the ``LM`` now and place it back on the config so GEPA core
        reuses the same object (rather than building its own, which we
        couldn't then read cost from). Mirrors the launcher's precedence:
        strategy -> custom proposer -> reflection_lm.
        """
        strategy = gepa_config.reflection.reflection_strategy
        if strategy is not None and hasattr(strategy, "total_cost"):
            return strategy
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
