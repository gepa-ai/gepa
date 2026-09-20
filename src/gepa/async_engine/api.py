# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Front door for the stage-decoupled asynchronous engine.

:func:`optimize` takes the same core arguments as :func:`gepa.optimize` and
returns the same :class:`~gepa.core.result.GEPAResult`. Run statistics
(per-stage timing, staleness histograms, overlap) are attached under
``result.metadata["async"]`` and written to ``run_dir`` when one is given.
"""

from __future__ import annotations

import dataclasses
import os
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from gepa.async_engine.config import AsyncEngineConfig
from gepa.async_engine.engine import AsyncStageEngine
from gepa.core.adapter import DataInst, GEPAAdapter, ProposalFn, RolloutOutput, Trajectory
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.result import GEPAResult
from gepa.core.state import EvaluationCache, FrontierType
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import Logger, LoggerProtocol, StdOutLogger
from gepa.proposer.reflective_mutation.base import CandidateSelector, LanguageModel, ReflectionComponentSelector
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionLM, StatelessReflectionLM
from gepa.strategies.acceptance import AcceptanceCriterion, ImprovementOrEqualAcceptance, StrictImprovementAcceptance
from gepa.strategies.batch_sampler import BatchSampler, EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import (
    CurrentBestCandidateSelector,
    EpsilonGreedyCandidateSelector,
    ParetoCandidateSelector,
    TopKParetoCandidateSelector,
)
from gepa.strategies.component_selector import AllReflectionComponentSelector, RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.utils import CompositeStopper, FileStopper, MaxMetricCallsStopper, MaxReflectionCostStopper, StopperProtocol

if TYPE_CHECKING:
    from gepa.adapters.default_adapter.default_adapter import ChatCompletionCallable, Evaluator
    from gepa.core.callbacks import GEPACallback


def optimize(
    seed_candidate: dict[str, str],
    trainset: list[DataInst] | DataLoader[DataId, DataInst],
    valset: list[DataInst] | DataLoader[DataId, DataInst] | None = None,
    adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput] | None = None,
    task_lm: str | ChatCompletionCallable | None = None,
    evaluator: Evaluator | None = None,
    # Reflection
    reflection_lm: LanguageModel | str | None = None,
    reflection_lm_kwargs: dict[str, Any] | None = None,
    reflection_strategy: ReflectionLM | None = None,
    reflection_prompt_template: str | dict[str, str] | None = None,
    custom_candidate_proposer: ProposalFn | None = None,
    # Search strategies
    candidate_selection_strategy: CandidateSelector
    | Literal["pareto", "current_best", "epsilon_greedy", "top_k_pareto"] = "pareto",
    frontier_type: FrontierType = "instance",
    module_selector: ReflectionComponentSelector | Literal["round_robin", "all"] = "round_robin",
    batch_sampler: BatchSampler | Literal["epoch_shuffled"] = "epoch_shuffled",
    reflection_minibatch_size: int | None = None,
    perfect_score: float = 1.0,
    skip_perfect_score: bool = True,
    acceptance_criterion: AcceptanceCriterion
    | Literal["strict_improvement", "improvement_or_equal"] = "strict_improvement",
    val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | Literal["full_eval"] | None = None,
    # Budget and stopping
    max_metric_calls: int | None = None,
    max_reflection_cost: float | None = None,
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None,
    # Asynchronous execution
    async_config: AsyncEngineConfig | None = None,
    # Logging and callbacks
    logger: LoggerProtocol | None = None,
    run_dir: str | None = None,
    callbacks: list[GEPACallback] | None = None,
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    tracking_key_prefix: str = "",
    track_best_outputs: bool = True,
    use_cloudpickle: bool = False,
    cache_evaluation: bool = False,
    seed: int = 0,
    raise_on_exception: bool = True,
) -> GEPAResult[RolloutOutput, DataId]:
    """Optimize ``seed_candidate`` with the stage-decoupled asynchronous engine.

    The search is the same reflective evolution as :func:`gepa.optimize`:
    Pareto-based parent selection, reflection on minibatch traces, a
    parent-relative acceptance gate, and full validation of accepted children.
    What differs is execution. Parent rollouts, reflection, child screening and
    validation run as separate stages with their own workers and buffers
    (see :class:`~gepa.async_engine.config.AsyncEngineConfig`), so several work
    items are in different stages at the same time.

    Requirements and differences from the synchronous engine:

    - ``adapter.evaluate`` (or ``batch_evaluate``) and the reflection LM are
      called from several threads at once and must be thread-safe.
    - Proposals are accepted one by one as they finish. Batch selection
      strategies (best-of-batch, top-k) and the merge proposer do not apply.
    - A child may be built from a parent chosen several commits ago. The
      staleness policy in ``async_config`` bounds or repairs that.
    - ``AsyncEngineConfig.sequential()`` keeps one item in the pipeline and
      reproduces the synchronous engine's order of operations.
    """
    if adapter is None:
        assert task_lm is not None, "Provide an adapter, or a task_lm for the default adapter."
        from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

        active_adapter = cast(
            "GEPAAdapter[DataInst, Trajectory, RolloutOutput]", DefaultAdapter(model=task_lm, evaluator=evaluator)
        )
    else:
        assert task_lm is None and evaluator is None, "task_lm and evaluator are only used without an adapter."
        active_adapter = adapter

    train_loader = ensure_loader(trainset)
    val_loader = train_loader if valset is None or valset is trainset else ensure_loader(valset)

    adapter_has_propose = getattr(active_adapter, "propose_new_texts", None) is not None
    if adapter_has_propose and custom_candidate_proposer is not None:
        raise ValueError("Cannot provide both adapter.propose_new_texts and custom_candidate_proposer.")
    if reflection_strategy is not None and (adapter_has_propose or custom_candidate_proposer is not None):
        raise ValueError("reflection_strategy would be ignored: another component owns proposal generation.")
    if not adapter_has_propose and custom_candidate_proposer is None:
        assert reflection_lm is not None or reflection_strategy is not None, (
            "reflection_lm (or reflection_strategy) is required when neither the adapter nor a custom proposer "
            "generates proposals."
        )
    if reflection_prompt_template is not None:
        assert not adapter_has_propose, "reflection_prompt_template is ignored when the adapter proposes texts."
        templates = (
            reflection_prompt_template.values()
            if isinstance(reflection_prompt_template, dict)
            else [reflection_prompt_template]
        )
        for template in templates:
            InstructionProposalSignature.validate_prompt_template(template)

    lm_callable: LanguageModel | None = None
    if isinstance(reflection_lm, str):
        from gepa.lm import LM

        lm_callable = LM(reflection_lm, **(reflection_lm_kwargs or {}))
    elif reflection_lm is not None:
        from gepa.lm import TrackingLM

        lm_callable = reflection_lm if hasattr(reflection_lm, "total_cost") else TrackingLM(reflection_lm)

    stoppers: list[StopperProtocol] = []
    if stop_callbacks is not None:
        stoppers.extend(stop_callbacks if isinstance(stop_callbacks, Sequence) else [stop_callbacks])
    if run_dir is not None:
        stoppers.append(FileStopper(os.path.join(run_dir, "gepa.stop")))
    if max_metric_calls is not None:
        stoppers.append(MaxMetricCallsStopper(max_metric_calls))
    if max_reflection_cost is not None:
        cost_source = reflection_strategy if reflection_strategy is not None else lm_callable
        if cost_source is None or not hasattr(cost_source, "total_cost"):
            raise ValueError("max_reflection_cost needs a reflection LM or strategy that exposes total_cost.")
        stoppers.append(MaxReflectionCostStopper(max_reflection_cost, reflection_lm=cost_source))
    if not stoppers:
        raise ValueError("Provide at least one of stop_callbacks, max_metric_calls, or max_reflection_cost.")
    stop_callback: StopperProtocol = stoppers[0] if len(stoppers) == 1 else CompositeStopper(*stoppers)

    if logger is None:
        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            logger = Logger(os.path.join(run_dir, "run_log.txt"))
        else:
            logger = StdOutLogger()

    rng = random.Random(seed)
    if isinstance(candidate_selection_strategy, str):
        selectors = {
            "pareto": lambda: ParetoCandidateSelector(rng=rng),
            "current_best": lambda: CurrentBestCandidateSelector(),
            "epsilon_greedy": lambda: EpsilonGreedyCandidateSelector(epsilon=0.1, rng=rng),
            "top_k_pareto": lambda: TopKParetoCandidateSelector(k=5, rng=rng),
        }
        if candidate_selection_strategy not in selectors:
            raise ValueError(f"Unknown candidate_selection_strategy: {candidate_selection_strategy}")
        candidate_selector: CandidateSelector = selectors[candidate_selection_strategy]()
    else:
        candidate_selector = candidate_selection_strategy

    if isinstance(module_selector, str):
        module_selectors = {"round_robin": RoundRobinReflectionComponentSelector, "all": AllReflectionComponentSelector}
        if module_selector not in module_selectors:
            raise ValueError(f"Unknown module_selector: {module_selector}")
        module_selector_instance: ReflectionComponentSelector = module_selectors[module_selector]()
    else:
        module_selector_instance = module_selector

    if batch_sampler == "epoch_shuffled":
        batch_sampler = EpochShuffledBatchSampler(minibatch_size=reflection_minibatch_size or 3, rng=rng)
    else:
        assert reflection_minibatch_size is None, "reflection_minibatch_size only applies to 'epoch_shuffled'."

    if isinstance(acceptance_criterion, str):
        criteria = {
            "strict_improvement": StrictImprovementAcceptance,
            "improvement_or_equal": ImprovementOrEqualAcceptance,
        }
        if acceptance_criterion not in criteria:
            raise ValueError(f"Unknown acceptance_criterion: {acceptance_criterion}")
        acceptance: AcceptanceCriterion = criteria[acceptance_criterion]()
    else:
        acceptance = acceptance_criterion

    if val_evaluation_policy is None or val_evaluation_policy == "full_eval":
        policy: EvaluationPolicy = FullEvaluationPolicy()
    else:
        policy = val_evaluation_policy

    if reflection_strategy is not None:
        for hook, arg in (("bind_rng", rng), ("bind_logger", logger), ("bind_lm_kwargs", reflection_lm_kwargs)):
            bind = getattr(reflection_strategy, hook, None)
            if callable(bind):
                bind(arg)
        bind_template = getattr(reflection_strategy, "bind_reflection_prompt_template", None)
        if callable(bind_template):
            bind_template(reflection_prompt_template)
    reflector_lm: ReflectionLM | None = reflection_strategy or (
        StatelessReflectionLM(lm_callable, reflection_prompt_template, logger) if lm_callable is not None else None
    )

    experiment_tracker = create_experiment_tracker(
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        key_prefix=tracking_key_prefix,
    )

    engine = AsyncStageEngine(
        adapter=active_adapter,
        run_dir=run_dir,
        trainset=train_loader,
        valset=val_loader,
        seed_candidate=seed_candidate,
        candidate_selector=candidate_selector,
        module_selector=module_selector_instance,
        batch_sampler=batch_sampler,
        reflection_lm=reflector_lm,
        raw_reflection_lm=lm_callable,
        custom_candidate_proposer=custom_candidate_proposer,
        perfect_score=perfect_score,
        skip_perfect_score=skip_perfect_score,
        seed=seed,
        frontier_type=frontier_type,
        logger=logger,
        experiment_tracker=experiment_tracker,
        stop_callback=stop_callback,
        callbacks=callbacks,
        track_best_outputs=track_best_outputs,
        raise_on_exception=raise_on_exception,
        use_cloudpickle=use_cloudpickle,
        val_evaluation_policy=policy,
        acceptance_criterion=acceptance,
        evaluation_cache=EvaluationCache() if cache_evaluation else None,
        config=async_config,
    )

    with experiment_tracker:
        if isinstance(logger, Logger):
            with logger:
                state = engine.run()
        else:
            state = engine.run()

    result = GEPAResult.from_state(state, run_dir=run_dir, seed=seed)
    return dataclasses.replace(result, metadata={**(result.metadata or {}), "async": engine.stats})
