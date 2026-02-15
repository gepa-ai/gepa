# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import os
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from gepa.core.callbacks import GEPACallback

from gepa.adapters.default_adapter.default_adapter import (
    ChatCompletionCallable,
    DefaultAdapter,
    Evaluator,
)
from gepa.core.adapter import DataInst, GEPAAdapter, ProposalFn, RolloutOutput, Trajectory
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.core.state import EvaluationCache, FrontierType
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import LoggerProtocol, StdOutLogger
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.base import CandidateSelector, LanguageModel, ReflectionComponentSelector
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import BatchSampler, EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import (
    CurrentBestCandidateSelector,
    EpsilonGreedyCandidateSelector,
    ParetoCandidateSelector,
)
from gepa.strategies.component_selector import (
    AllReflectionComponentSelector,
    RoundRobinReflectionComponentSelector,
)
from gepa.strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy
from gepa.utils import FileStopper, StopperProtocol


def optimize(
    seed_candidate: dict[str, str],
    trainset: list[DataInst] | DataLoader[DataId, DataInst],
    valset: list[DataInst] | DataLoader[DataId, DataInst] | None = None,
    adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput] | None = None,
    task_lm: str | ChatCompletionCallable | None = None,
    evaluator: Evaluator | None = None,
    # Reflection-based configuration
    reflection_lm: LanguageModel | str | None = None,
    candidate_selection_strategy: CandidateSelector | Literal["pareto", "current_best", "epsilon_greedy"] = "pareto",
    frontier_type: FrontierType = "instance",
    skip_perfect_score: bool = True,
    batch_sampler: BatchSampler | Literal["epoch_shuffled"] = "epoch_shuffled",
    reflection_minibatch_size: int | None = None,
    perfect_score: float = 1.0,
    reflection_prompt_template: str | dict[str, str] | None = None,
    custom_candidate_proposer: ProposalFn | None = None,
    # Component selection configuration
    module_selector: ReflectionComponentSelector | str = "round_robin",
    # Merge-based configuration
    use_merge: bool = False,
    max_merge_invocations: int = 5,
    merge_val_overlap_floor: int = 5,
    # Budget and Stop Condition
    max_metric_calls: int | None = None,
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None,
    # Logging and Callbacks
    logger: LoggerProtocol | None = None,
    run_dir: str | None = None,
    callbacks: "list[GEPACallback] | None" = None,
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    track_best_outputs: bool = False,
    display_progress_bar: bool = False,
    use_cloudpickle: bool = False,
    # Evaluation caching
    cache_evaluation: bool = False,
    # Reproducibility
    seed: int = 0,
    raise_on_exception: bool = True,
    val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | Literal["full_eval"] | None = None,
) -> GEPAResult[RolloutOutput, DataId]:
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
    - trainset: Training data supplied as an in-memory sequence or a `DataLoader` yielding batches for reflective updates.
    - valset: Validation data source (sequence or `DataLoader`) used for tracking Pareto scores. If not provided, GEPA reuses the trainset.
    - adapter: A `GEPAAdapter` instance that implements the adapter interface. This allows GEPA to plug into your system's environment. If not provided, GEPA will use a default adapter: `gepa.adapters.default_adapter.default_adapter.DefaultAdapter`, with model defined by `task_lm`.
    - task_lm: Optional. The model to use for the task. This is only used if `adapter` is not provided, and is used to initialize the default adapter.
    - evaluator: Optional. A custom evaluator to use for evaluating the candidate program. If not provided, GEPA will use the default evaluator: `gepa.adapters.default_adapter.default_adapter.ContainsAnswerEvaluator`. Only used if `adapter` is not provided.

    # Reflection-based configuration
    - reflection_lm: A `LanguageModel` instance that is used to reflect on the performance of the candidate program.
    - candidate_selection_strategy: The strategy to use for selecting the candidate to update. Supported strategies: 'pareto', 'current_best', 'epsilon_greedy'. Defaults to 'pareto'.
    - frontier_type: Strategy for tracking Pareto frontiers. 'instance' tracks per validation example, 'objective' tracks per objective metric, 'hybrid' combines both, 'cartesian' tracks per (example, objective) pair. Defaults to 'instance'.
    - skip_perfect_score: Whether to skip updating the candidate if it achieves a perfect score on the minibatch.
    - batch_sampler: Strategy for selecting training examples. Can be a [BatchSampler](src/gepa/strategies/batch_sampler.py) instance or a string for a predefined strategy from ['epoch_shuffled']. Defaults to 'epoch_shuffled', which creates an [EpochShuffledBatchSampler](src/gepa/strategies/batch_sampler.py).
    - reflection_minibatch_size: The number of examples to use for reflection in each proposal step. Defaults to 3. Only valid when batch_sampler='epoch_shuffled' (default), and is ignored otherwise.
    - perfect_score: The perfect score to achieve.
    - reflection_prompt_template: The prompt template to use for reflection. Can be either a string (applied to all components) or a dict mapping component names to their specific templates. If not provided, GEPA will use the default prompt template (see [InstructionProposalSignature](src/gepa/strategies/instruction_proposal.py)). Each prompt template must contain the following placeholders, which will be replaced with actual values: `<curr_param>` (will be replaced by the instructions/component to evolve) and `<side_info>` (replaced with the inputs, outputs, and feedback generated with current instruction). When using a dict, components without a specified template will use the default template. This will be ignored if the adapter provides its own `propose_new_texts` method.
    - custom_candidate_proposer: Optional custom function for proposing new candidates. If provided, this will be used instead of the default LLM-based reflection approach. Cannot be used if adapter provides `propose_new_texts`. Signature: `(candidate, reflective_dataset, components_to_update) -> dict[str, str]`.

    # Component selection configuration
    - module_selector: Component selection strategy. Can be a ReflectionComponentSelector instance or a string ('round_robin', 'all'). Defaults to 'round_robin'. The 'round_robin' strategy cycles through components in order. The 'all' strategy selects all components for modification in every GEPA iteration.

    # Merge-based configuration
    - use_merge: Whether to use the merge strategy.
    - max_merge_invocations: The maximum number of merge invocations to perform.
    - merge_val_overlap_floor: Minimum number of shared validation ids required between parents before attempting a merge subsample. Only relevant when using `val_evaluation_policy` other than `full_eval`.

    # Budget and Stop Condition
    - max_metric_calls: Optional maximum number of metric calls to perform. If not provided, stop_callbacks must be provided.
    - stop_callbacks: Optional stopper(s) that return True when optimization should stop. Can be a single StopperProtocol or a list or tuple of StopperProtocol instances. Examples: FileStopper, TimeoutStopCondition, SignalStopper, NoImprovementStopper, or custom stopping logic. If not provided, max_metric_calls must be provided.

    # Logging and Callbacks
    - logger: A `LoggerProtocol` instance that is used to log the progress of the optimization.
    - callbacks: Optional list of callback objects for observing optimization progress. Callbacks receive events like on_optimization_start, on_iteration_start, on_candidate_accepted, etc. See `gepa.core.callbacks.GEPACallback` for the full protocol.
    - run_dir: The directory to save the results to. Optimization state and results will be saved to this directory. If the directory already exists, GEPA will read the state from this directory and resume the optimization from the last saved state. If provided, a FileStopper is automatically created which checks for the presence of "gepa.stop" in this directory, allowing graceful stopping of the optimization process upon its presence.
    - use_wandb: Whether to use Weights and Biases to log the progress of the optimization.
    - wandb_api_key: The API key to use for Weights and Biases.
    - wandb_init_kwargs: Additional keyword arguments to pass to the Weights and Biases initialization.
    - use_mlflow: Whether to use MLflow to log the progress of the optimization.
      Both wandb and mlflow can be used simultaneously if desired.
    - mlflow_tracking_uri: The tracking URI to use for MLflow.
    - mlflow_experiment_name: The experiment name to use for MLflow.
    - track_best_outputs: Whether to track the best outputs on the validation set. If True, GEPAResult will contain the best outputs obtained for each task in the validation set.
    - display_progress_bar: Show a tqdm progress bar over metric calls when enabled.
    - use_cloudpickle: Use cloudpickle instead of pickle. This can be helpful when the serialized state contains dynamically generated DSPy signatures.

    # Evaluation caching
    - cache_evaluation: Whether to cache the (score, output, objective_scores) of (candidate, example) pairs. If True and a cache entry exists, GEPA will skip the fitness evaluation and use the cached results. This helps avoid redundant evaluations and saves metric calls. Defaults to False.

    # Reproducibility
    - seed: The seed to use for the random number generator.
    - val_evaluation_policy: Strategy controlling which validation ids to score each iteration and which candidate is currently best. Supported strings: "full_eval" (evaluate every id each time) Passing None defaults to "full_eval".
    - raise_on_exception: Whether to propagate proposer/evaluator exceptions instead of stopping gracefully.
    """
    # Validate seed_candidate is not None or empty
    if seed_candidate is None or not seed_candidate:
        raise ValueError("seed_candidate must contain at least one component text.")

    active_adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput] | None = None
    if adapter is None:
        assert task_lm is not None, (
            "Since no adapter is provided, GEPA requires a task LM to be provided. Please set the `task_lm` parameter."
        )
        active_adapter = cast(
            GEPAAdapter[DataInst, Trajectory, RolloutOutput], DefaultAdapter(model=task_lm, evaluator=evaluator)
        )
    else:
        assert task_lm is None, (
            "Since an adapter is provided, GEPA does not require a task LM to be provided. Please set the `task_lm` parameter to None."
        )
        assert evaluator is None, (
            "Since an adapter is provided, GEPA does not require an evaluator to be provided. Please set the `evaluator` parameter to None."
        )
        active_adapter = adapter

    # Normalize datasets to DataLoader instances
    train_loader = ensure_loader(trainset)
    val_loader = ensure_loader(valset) if valset is not None else train_loader

    # Comprehensive stop_callback logic
    # Convert stop_callbacks to a list if it's not already
    stop_callbacks_list: list[StopperProtocol] = []
    if stop_callbacks is not None:
        if isinstance(stop_callbacks, Sequence):
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
    if not stop_callbacks_list:
        raise ValueError(
            "The user must provide at least one of stop_callbacks or max_metric_calls to specify a stopping condition."
        )

    # Create composite stopper if multiple stoppers, or use single stopper
    stop_callback: StopperProtocol
    if len(stop_callbacks_list) == 1:
        stop_callback = stop_callbacks_list[0]
    else:
        from gepa.utils import CompositeStopper

        stop_callback = CompositeStopper(*stop_callbacks_list)

    # Validate that only one custom proposal method is provided
    adapter_has_propose = hasattr(active_adapter, "propose_new_texts") and active_adapter.propose_new_texts is not None
    if adapter_has_propose and custom_candidate_proposer is not None:
        raise ValueError(
            "Cannot provide both adapter.propose_new_texts and custom_candidate_proposer. "
            "Please use only one custom proposal method."
        )

    if not adapter_has_propose and custom_candidate_proposer is None:
        assert reflection_lm is not None, (
            f"reflection_lm was not provided. The adapter used '{active_adapter!s}' does not provide a propose_new_texts method, "
            + "and custom_candidate_proposer was not provided. "
            + "GEPA will use the default proposer, which requires a reflection_lm to be specified."
        )


    reflection_lm_callable: LanguageModel | None = None
    if isinstance(reflection_lm, str):
        import litellm

        reflection_lm_name = reflection_lm

        def _reflection_lm(prompt: str | list[dict[str, str]]) -> str:
            if isinstance(prompt, str):
                completion = litellm.completion(
                    model=reflection_lm_name, messages=[{"role": "user", "content": prompt}]
                )
            else:
                completion = litellm.completion(model=reflection_lm_name, messages=prompt)
            return completion.choices[0].message.content  # type: ignore

        reflection_lm_callable = _reflection_lm
    else:
        reflection_lm_callable = reflection_lm

    if logger is None:
        logger = StdOutLogger()

    rng = random.Random(seed)

    candidate_selector: CandidateSelector
    if isinstance(candidate_selection_strategy, str):
        factories = {
            "pareto": lambda: ParetoCandidateSelector(rng=rng),
            "current_best": lambda: CurrentBestCandidateSelector(),
            "epsilon_greedy": lambda: EpsilonGreedyCandidateSelector(epsilon=0.1, rng=rng),
        }

        try:
            candidate_selector = factories[candidate_selection_strategy]()
        except KeyError as exc:
            raise ValueError(
                f"Unknown candidate_selector strategy: {candidate_selection_strategy}. "
                "Supported strategies: 'pareto', 'current_best', 'epsilon_greedy'"
            ) from exc
    elif isinstance(candidate_selection_strategy, CandidateSelector):
        candidate_selector = candidate_selection_strategy
    else:
        raise TypeError(
            "candidate_selection_strategy must be a supported string strategy or an instance of CandidateSelector."
        )

    if val_evaluation_policy is None or val_evaluation_policy == "full_eval":
        val_evaluation_policy = FullEvaluationPolicy()
    elif not isinstance(val_evaluation_policy, EvaluationPolicy):
        raise ValueError(
            f"val_evaluation_policy should be one of 'full_eval' or an instance of EvaluationPolicy, but got {type(val_evaluation_policy)}"
        )

    if isinstance(module_selector, str):
        module_selector_cls = {
            "round_robin": RoundRobinReflectionComponentSelector,
            "all": AllReflectionComponentSelector,
        }.get(module_selector)

        assert module_selector_cls is not None, (
            f"Unknown module_selector strategy: {module_selector}. Supported strategies: 'round_robin', 'all'"
        )

        module_selector_instance: ReflectionComponentSelector = module_selector_cls()
    else:
        module_selector_instance = module_selector

    if batch_sampler == "epoch_shuffled":
        batch_sampler = EpochShuffledBatchSampler(minibatch_size=reflection_minibatch_size or 3, rng=rng)
    else:
        assert reflection_minibatch_size is None, (
            "reflection_minibatch_size only accepted if batch_sampler is 'epoch_shuffled'"
        )

    experiment_tracker = create_experiment_tracker(
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    if reflection_prompt_template is not None:
        assert not (adapter is not None and getattr(adapter, "propose_new_texts", None) is not None), (
            f"Adapter {adapter!s} provides its own propose_new_texts method; reflection_prompt_template will be ignored. "
            "Set reflection_prompt_template to None."
        )

    # Create evaluation cache if enabled
    evaluation_cache: EvaluationCache[RolloutOutput, DataId] | None = None
    if cache_evaluation:
        evaluation_cache = EvaluationCache[RolloutOutput, DataId]()

    reflective_proposer = ReflectiveMutationProposer(
        logger=logger,
        trainset=train_loader,
        adapter=active_adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector_instance,
        batch_sampler=batch_sampler,
        perfect_score=perfect_score,
        skip_perfect_score=skip_perfect_score,
        experiment_tracker=experiment_tracker,
        reflection_lm=reflection_lm_callable,
        reflection_prompt_template=reflection_prompt_template,
        custom_candidate_proposer=custom_candidate_proposer,
        callbacks=callbacks,
    )

    def evaluator_fn(
        inputs: list[DataInst], prog: dict[str, str]
    ) -> tuple[list[RolloutOutput], list[float], Sequence[dict[str, float]] | None]:
        eval_out = active_adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores, eval_out.objective_scores

    merge_proposer: MergeProposer | None = None
    if use_merge:
        merge_proposer = MergeProposer(
            logger=logger,
            valset=val_loader,
            evaluator=evaluator_fn,
            use_merge=use_merge,
            max_merge_invocations=max_merge_invocations,
            rng=rng,
            val_overlap_floor=merge_val_overlap_floor,
            callbacks=callbacks,
        )

    engine = GEPAEngine(
        adapter=active_adapter,
        run_dir=run_dir,
        valset=val_loader,
        seed_candidate=seed_candidate,
        perfect_score=perfect_score,
        seed=seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=merge_proposer,
        frontier_type=frontier_type,
        logger=logger,
        experiment_tracker=experiment_tracker,
        callbacks=callbacks,
        track_best_outputs=track_best_outputs,
        display_progress_bar=display_progress_bar,
        raise_on_exception=raise_on_exception,
        stop_callback=stop_callback,
        val_evaluation_policy=val_evaluation_policy,
        use_cloudpickle=use_cloudpickle,
        evaluation_cache=evaluation_cache,
    )

    with experiment_tracker:
        state = engine.run()

    return GEPAResult.from_state(state, run_dir=run_dir, seed=seed)
