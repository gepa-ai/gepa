# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import os
import random
from typing import Any, Callable

from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import LoggerProtocol, StdOutLogger
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.base import LanguageModel, ReflectionComponentSelector
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import CurrentBestCandidateSelector, ParetoCandidateSelector
from gepa.strategies.component_selector import (
    AllReflectionComponentSelector,
    RoundRobinReflectionComponentSelector,
)
from gepa.utils import FileStopper, StopperProtocol


def optimize(
    seed_candidate: dict[str, str],
    trainset: list[DataInst],
    valset: list[DataInst] | None = None,
    adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput] | None = None,
    task_lm: str | Callable | None = None,
    # Reflection-based configuration
    reflection_lm: LanguageModel | str | None = None,
    candidate_selection_strategy: str = "pareto",
    skip_perfect_score=True,
    reflection_minibatch_size=3,
    perfect_score=1,
    # Component selection configuration
    module_selector: "ReflectionComponentSelector | str" = "round_robin",
    # Merge-based configuration
    use_merge=False,
    max_merge_invocations=5,
    # Budget and Stop Condition
    max_metric_calls=None,
    stop_callbacks: "StopperProtocol | list[StopperProtocol] | None" = None,
    # Logging
    logger: LoggerProtocol | None = None,
    run_dir: str | None = None,
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    track_best_outputs: bool = False,
    display_progress_bar: bool = False,
    # Reproducibility
    seed: int = 0,
    raise_on_exception: bool = True,
):
    """
    GEPA is an evolutionary optimizer that evolves (multiple) text components of a complex system to optimize them towards a given metric.
    GEPA can also leverage rich textual feedback obtained from the system's execution environment, evaluation,
    and the system's own execution traces to iteratively improve the system's performance.

    Concepts:
    - System: A harness that uses text components to perform a task. Each text component of the system to be optimized is a named component of the system.
    - Candidate: A mapping from component names to component text. A concrete instantiation of the system is realized by setting the text of each system component
      to the text provided by the candidate mapping.
    - `DataInst`: An (uninterpreted) data type over which the system operates.
    - `RolloutOutput`: The output of the system on a `DataInst`.

    Each execution of the system produces a `RolloutOutput`, which can be evaluated to produce a score. The execution of the system also produces a trajectory,
    which consists of the operations performed by different components of the system, including the text of the components that were executed.

    GEPA can be applied to optimize any system that uses text components (e.g., prompts in a AI system, code snippets/code files/functions/classes in a codebase, etc.).
    In order for GEPA to plug into your system's environment, GEPA requires an adapter, `GEPAAdapter` to be implemented. The adapter is responsible for:
    1. Evaluating a proposed candidate on a batch of inputs.
       - The adapter receives a candidate proposed by GEPA, along with a batch of inputs selected from the training/validation set.
       - The adapter instantiates the system with the texts proposed in the candidate.
       - The adapter then evaluates the candidate on the batch of inputs, and returns the scores.
       - The adapter should also capture relevant information from the execution of the candidate, like system and evaluation traces.
    2. Identifying textual information relevant to a component of the candidate
       - Given the trajectories captured during the execution of the candidate, GEPA selects a component of the candidate to update.
       - The adapter receives the candidate, the batch of inputs, and the trajectories captured during the execution of the candidate.
       - The adapter is responsible for identifying the textual information relevant to the component to update.
       - This information is used by GEPA to reflect on the performnace of the component, and propose new component texts.

    At each iteration, GEPA proposes a new candidate using one of the following strategies:
    1. Reflective mutation: GEPA proposes a new candidate by mutating the current candidate, leveraging rich textual feedback.
    2. Merge: GEPA proposes a new candidate by merging 2 candidates that are on the Pareto frontier.

    GEPA also tracks the Pareto frontier of performance achieved by different candidates on the validation set. This way, it can leverage candidates that
    work well on a subset of inputs to improve the system's performance on the entire validation set, by evolving from the Pareto frontier.

    Parameters:
    - seed_candidate: The initial candidate to start with.
    - trainset: The training set to use for reflective updates.
    - valset: The validation set to use for tracking Pareto scores. If not provided, GEPA will use the trainset for both.
    - adapter: A `GEPAAdapter` instance that implements the adapter interface. This allows GEPA to plug into your system's environment. If not provided, GEPA will use a default adapter: `gepa.adapters.default_adapter.default_adapter.DefaultAdapter`, with model defined by `task_lm`.
    - task_lm: Optional. The model to use for the task. This is only used if `adapter` is not provided, and is used to initialize the default adapter.

    # Reflection-based configuration
    - reflection_lm: A `LanguageModel` instance that is used to reflect on the performance of the candidate program.
    - candidate_selection_strategy: The strategy to use for selecting the candidate to update.
    - skip_perfect_score: Whether to skip updating the candidate if it achieves a perfect score on the minibatch.
    - reflection_minibatch_size: The number of examples to use for reflection in each proposal step.
    - perfect_score: The perfect score to achieve.

    # Component selection configuration
    - module_selector: Component selection strategy. Can be a ReflectionComponentSelector instance or a string ('round_robin', 'all'). Defaults to 'round_robin'. The 'round_robin' strategy cycles through components in order. The 'all' strategy selects all components for modification in every GEPA iteration.

    # Merge-based configuration
    - use_merge: Whether to use the merge strategy.
    - max_merge_invocations: The maximum number of merge invocations to perform.

    # Budget and Stop Condition
    - max_metric_calls: Optional maximum number of metric calls to perform. If not provided, stop_callbacks must be provided.
    - stop_callbacks: Optional stopper(s) that return True when optimization should stop. Can be a single StopperProtocol or a list of StopperProtocol instances. Examples: FileStopper, TimeoutStopCondition, SignalStopper, NoImprovementStopper, or custom stopping logic. If not provided, max_metric_calls must be provided.

    # Logging
    - logger: A `LoggerProtocol` instance that is used to log the progress of the optimization.
    - run_dir: The directory to save the results to. Optimization state and results will be saved to this directory. If the directory already exists, GEPA will read the state from this directory and resume the optimization from the last saved state. If provided, a FileStopper is automatically created which checks for the presence of "gepa.stop" in this directory, allowing graceful stopping of the optimization process upon its presence.
    - use_wandb: Whether to use Weights and Biases to log the progress of the optimization.
    - wandb_api_key: The API key to use for Weights and Biases.
    - wandb_init_kwargs: Additional keyword arguments to pass to the Weights and Biases initialization.
    - use_mlflow: Whether to use MLflow to log the progress of the optimization.
      Both wandb and mlflow can be used simultaneously if desired.
    - mlflow_tracking_uri: The tracking URI to use for MLflow.
    - mlflow_experiment_name: The experiment name to use for MLflow.
    - track_best_outputs: Whether to track the best outputs on the validation set. If True, GEPAResult will contain the best outputs obtained for each task in the validation set.

    # Reproducibility
    - seed: The seed to use for the random number generator.
    """
    if adapter is None:
        assert task_lm is not None, (
            "Since no adapter is provided, GEPA requires a task LM to be provided. Please set the `task_lm` parameter."
        )
        adapter = DefaultAdapter(model=task_lm)
    else:
        assert task_lm is None, (
            "Since an adapter is provided, GEPA does not require a task LM to be provided. Please set the `task_lm` parameter to None."
        )

    # Comprehensive stop_callback logic
    # Convert stop_callbacks to a list if it's not already
    stop_callbacks_list = []
    if stop_callbacks is not None:
        if isinstance(stop_callbacks, list):
            stop_callbacks_list.extend(stop_callbacks)
        else:
            stop_callbacks_list.append(stop_callbacks)

    # Add file stopper if run_dir is provided
    if run_dir is not None:
        stop_file_path = os.path.join(run_dir, "gepa.stop")
        file_stopper = FileStopper(stop_file_path)
        stop_callbacks_list.append(file_stopper)

    # Add max_metric_calls stopper if provided
    if max_metric_calls is not None:
        from gepa.utils import MaxMetricCallsStopper

        max_calls_stopper = MaxMetricCallsStopper(max_metric_calls)
        stop_callbacks_list.append(max_calls_stopper)

    # Assert that at least one stopping condition is provided
    if len(stop_callbacks_list) == 0:
        raise ValueError(
            "The user must provide at least one of stop_callbacks or max_metric_calls to specify a stopping condition."
        )

    # Create composite stopper if multiple stoppers, or use single stopper
    if len(stop_callbacks_list) == 1:
        stop_callback = stop_callbacks_list[0]
    else:
        from gepa.utils import CompositeStopper

        stop_callback = CompositeStopper(*stop_callbacks_list)

    if not hasattr(adapter, "propose_new_texts"):
        assert reflection_lm is not None, (
            f"reflection_lm was not provided. The adapter used '{adapter!s}' does not provide a propose_new_texts method, "
            + "and hence, GEPA will use the default proposer, which requires a reflection_lm to be specified."
        )

    if isinstance(reflection_lm, str):
        import litellm

        reflection_lm_name = reflection_lm
        reflection_lm = (
            lambda prompt: litellm.completion(model=reflection_lm_name, messages=[{"role": "user", "content": prompt}])
            .choices[0]
            .message.content
        )

    if logger is None:
        logger = StdOutLogger()

    if valset is None:
        valset = trainset

    rng = random.Random(seed)
    candidate_selector = (
        ParetoCandidateSelector(rng=rng) if candidate_selection_strategy == "pareto" else CurrentBestCandidateSelector()
    )

    if isinstance(module_selector, str):
        module_selector_cls = {
            "round_robin": RoundRobinReflectionComponentSelector,
            "all": AllReflectionComponentSelector,
        }.get(module_selector)

        assert module_selector_cls is not None, (
            f"Unknown module_selector strategy: {module_selector}. Supported strategies: 'round_robin', 'all'"
        )

        module_selector = module_selector_cls()

    batch_sampler = EpochShuffledBatchSampler(minibatch_size=reflection_minibatch_size, rng=rng)

    experiment_tracker = create_experiment_tracker(
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    reflective_proposer = ReflectiveMutationProposer(
        logger=logger,
        trainset=trainset,
        adapter=adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector,
        batch_sampler=batch_sampler,
        perfect_score=perfect_score,
        skip_perfect_score=skip_perfect_score,
        experiment_tracker=experiment_tracker,
        reflection_lm=reflection_lm,
    )

    def evaluator(inputs, prog):
        eval_out = adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores

    merge_proposer = None
    if use_merge:
        merge_proposer = MergeProposer(
            logger=logger,
            valset=valset,
            evaluator=evaluator,
            use_merge=use_merge,
            max_merge_invocations=max_merge_invocations,
            rng=rng,
        )

    engine = GEPAEngine(
        run_dir=run_dir,
        evaluator=evaluator,
        valset=valset,
        seed_candidate=seed_candidate,
        perfect_score=perfect_score,
        seed=seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=merge_proposer,
        logger=logger,
        experiment_tracker=experiment_tracker,
        track_best_outputs=track_best_outputs,
        display_progress_bar=display_progress_bar,
        raise_on_exception=raise_on_exception,
        stop_callback=stop_callback,
    )

    with experiment_tracker:
        state = engine.run()

    result = GEPAResult.from_state(state)
    return result
