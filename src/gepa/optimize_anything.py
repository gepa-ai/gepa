import os
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Generic, Literal, Protocol, Sequence, Type, TypeVar, cast

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.result import GEPAResult
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.batch_sampler import BatchSampler

from gepa.adapters.default_adapter.default_adapter import ChatCompletionCallable, DefaultAdapter
from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
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

OptimizableParam = str
class FitnessFn(Protocol):
    def __call__(
        self, 
        candidate: dict[str, OptimizableParam],
        batch: list[DataInst]
    ) -> list[tuple[float, dict[str, Any]]]:
        """
        Evaluate the fitness of a candidate on a batch of data.

        Parameters:
        - candidate: The candidate to evaluate.
        - batch: The batch of data to evaluate the candidate on.

        Returns:
        - A list of size len(batch) containing tuple(score, side_info) for each example in the batch.
          side_info is dict[str, Any], which can contain additional information about the evaluation
          of the candidate on the example. side_info can contain a mix of text and numeric information
          which will be used by the optimize to guide the search. For example, side_info can be:
          {
            "Input": ...,
            "Output": ...,
            "Feedback": ...,
          } 
        """
        ...

@dataclass
class ReflectionConfig:
    reflection_lm: LanguageModel | str | None = None
    skip_perfect_score: bool = False
    perfect_score: float | None = None
    batch_sampler: BatchSampler | Literal["epoch_shuffled"] = "epoch_shuffled"
    reflection_minibatch_size: int = 3
    reflection_prompt_template: str | None = None
    module_selector: ReflectionComponentSelector | Literal["round_robin", "all"] = "round_robin"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ReflectionConfig":
        return ReflectionConfig(**d)

@dataclass
class GEPAConfig:
    objective: Literal["hill_climb", "generalize"] = "hill_climb"
    candidate_selection_strategy: CandidateSelector | Literal["pareto", "current_best", "epsilon_greedy"] = "pareto"
    val_evaluation_policy: EvaluationPolicy | Literal["full_eval"] = "full_eval"
    # Reflection configuration
    reflection_config: ReflectionConfig = field(default_factory=ReflectionConfig)
    # Merge-based configuration
    use_merge: bool = False
    max_merge_invocations: int = 5
    merge_val_overlap_floor: int = 5
    # Budget and Stop Condition
    max_metric_calls: int | None = None
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None
    # Logging
    logger: LoggerProtocol | None = None
    run_dir: str | None = None
    use_wandb: bool = False
    wandb_api_key: str | None = None
    wandb_init_kwargs: dict[str, Any] | None = None
    use_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    track_best_outputs: bool = False
    display_progress_bar: bool = False
    use_cloudpickle: bool = False
    # Reproducibility
    seed: int = 0
    raise_on_exception: bool = True

    def __post_init__(self):
        # Handle reflection_config if passed as dict
        if isinstance(self.reflection_config, dict):
            self.reflection_config = ReflectionConfig.from_dict(self.reflection_config)
    
    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['reflection_config'] = self.reflection_config.to_dict()
        return result
    
    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GEPAConfig":
        return GEPAConfig(**d)

def optimize_anything(
    seed_candidate: dict[str, OptimizableParam] | list[dict[str, OptimizableParam]],
    fitness_fn: FitnessFn,
    dataset: list[DataInst],
    valset: list[DataInst] | None = None,
    config: GEPAConfig | None = None
) -> GEPAResult:
    # Use default config if not provided
    if config is None:
        config = GEPAConfig()
    
    active_adapter: GEPAAdapter = OptimizeAnythingAdapter(fitness_fn=fitness_fn)

    # Normalize datasets to DataLoader instances
    train_loader = ensure_loader(dataset)
    val_loader = ensure_loader(valset) if valset is not None else train_loader

    # Comprehensive stop_callback logic
    # Convert stop_callbacks to a list if it's not already
    stop_callbacks_list: list[StopperProtocol] = []
    if config.stop_callbacks is not None:
        if isinstance(config.stop_callbacks, Sequence):
            stop_callbacks_list.extend(config.stop_callbacks)
        else:
            stop_callbacks_list.append(config.stop_callbacks)

    # Add file stopper if run_dir is provided
    if config.run_dir is not None:
        stop_file_path = os.path.join(config.run_dir, "gepa.stop")
        file_stopper = FileStopper(stop_file_path)
        stop_callbacks_list.append(file_stopper)

    # Add max_metric_calls stopper if provided
    if config.max_metric_calls is not None:
        from gepa.utils import MaxMetricCallsStopper

        max_calls_stopper = MaxMetricCallsStopper(config.max_metric_calls)
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

    if not hasattr(active_adapter, "propose_new_texts"):
        assert config.reflection_config.reflection_lm is not None, (
            f"reflection_lm was not provided. The adapter used '{active_adapter!s}' does not provide a propose_new_texts method, "
            + "and hence, GEPA will use the default proposer, which requires a reflection_lm to be specified."
        )

    if isinstance(config.reflection_config.reflection_lm, str):
        import litellm

        reflection_lm_name = config.reflection_config.reflection_lm

        def _reflection_lm(prompt: str) -> str:
            completion = litellm.completion(model=reflection_lm_name, messages=[{"role": "user", "content": prompt}])
            return completion.choices[0].message.content  # type: ignore

        config.reflection_config.reflection_lm = _reflection_lm

    if config.logger is None:
        config.logger = StdOutLogger()

    rng = random.Random(config.seed)

    candidate_selector: CandidateSelector
    if isinstance(config.candidate_selection_strategy, str):
        factories = {
            "pareto": lambda: ParetoCandidateSelector(rng=rng),
            "current_best": lambda: CurrentBestCandidateSelector(),
            "epsilon_greedy": lambda: EpsilonGreedyCandidateSelector(epsilon=0.1, rng=rng),
        }

        try:
            candidate_selector = factories[config.candidate_selection_strategy]()
        except KeyError as exc:
            raise ValueError(
                f"Unknown candidate_selector strategy: {config.candidate_selection_strategy}. "
                "Supported strategies: 'pareto', 'current_best', 'epsilon_greedy'"
            ) from exc
    elif isinstance(config.candidate_selection_strategy, CandidateSelector):
        candidate_selector = config.candidate_selection_strategy
    else:
        raise TypeError(
            "candidate_selection_strategy must be a supported string strategy or an instance of CandidateSelector."
        )

    if config.val_evaluation_policy is None or config.val_evaluation_policy == "full_eval":
        config.val_evaluation_policy = FullEvaluationPolicy()
    elif not isinstance(config.val_evaluation_policy, EvaluationPolicy):
        raise ValueError(
            f"val_evaluation_policy should be one of 'full_eval' or an instance of EvaluationPolicy, but got {type(config.val_evaluation_policy)}"
        )

    if isinstance(config.reflection_config.module_selector, str):
        module_selector_cls = {
            "round_robin": RoundRobinReflectionComponentSelector,
            "all": AllReflectionComponentSelector,
        }.get(config.reflection_config.module_selector)

        assert module_selector_cls is not None, (
            f"Unknown module_selector strategy: {config.reflection_config.module_selector}. Supported strategies: 'round_robin', 'all'"
        )

        module_selector_instance: ReflectionComponentSelector = module_selector_cls()
    else:
        module_selector_instance = config.reflection_config.module_selector

    if config.reflection_config.batch_sampler == "epoch_shuffled":
        config.reflection_config.batch_sampler = EpochShuffledBatchSampler(minibatch_size=config.reflection_config.reflection_minibatch_size or 3, rng=rng)
    else:
        assert config.reflection_config.reflection_minibatch_size is None, (
            "reflection_minibatch_size only accepted if batch_sampler is 'epoch_shuffled'"
        )

    experiment_tracker = create_experiment_tracker(
        use_wandb=config.use_wandb,
        wandb_api_key=config.wandb_api_key,
        wandb_init_kwargs=config.wandb_init_kwargs,
        use_mlflow=config.use_mlflow,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        mlflow_experiment_name=config.mlflow_experiment_name,
    )

    if config.reflection_config.reflection_prompt_template is not None:
        assert not (active_adapter is not None and getattr(active_adapter, "propose_new_texts", None) is not None), (
            f"Adapter {active_adapter!s} provides its own propose_new_texts method; reflection_prompt_template will be ignored. "
            "Set reflection_prompt_template to None."
        )

    reflective_proposer = ReflectiveMutationProposer(
        logger=config.logger,
        trainset=train_loader,
        adapter=active_adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector_instance,
        batch_sampler=config.reflection_config.batch_sampler,
        perfect_score=config.reflection_config.perfect_score,
        skip_perfect_score=config.reflection_config.skip_perfect_score,
        experiment_tracker=experiment_tracker,
        reflection_lm=config.reflection_config.reflection_lm,
        reflection_prompt_template=config.reflection_config.reflection_prompt_template,
    )

    def evaluator(inputs: list[DataInst], prog: dict[str, str]) -> tuple[list[RolloutOutput], list[float]]:
        eval_out = active_adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores

    merge_proposer: MergeProposer | None = None
    if config.use_merge:
        merge_proposer = MergeProposer(
            logger=config.logger,
            valset=val_loader,
            evaluator=evaluator,
            use_merge=config.use_merge,
            max_merge_invocations=config.max_merge_invocations,
            rng=rng,
            val_overlap_floor=config.merge_val_overlap_floor,
        )

    engine = GEPAEngine(
        run_dir=config.run_dir,
        evaluator=evaluator,
        valset=val_loader,
        seed_candidate=[seed_candidate] if not isinstance(seed_candidate, list) else seed_candidate,
        perfect_score=config.reflection_config.perfect_score,
        seed=config.seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=merge_proposer,
        logger=config.logger,
        experiment_tracker=experiment_tracker,
        track_best_outputs=config.track_best_outputs,
        display_progress_bar=config.display_progress_bar,
        raise_on_exception=config.raise_on_exception,
        stop_callback=stop_callback,
        val_evaluation_policy=config.val_evaluation_policy,
        use_cloudpickle=config.use_cloudpickle,
    )

    with experiment_tracker:
        state = engine.run()

    return GEPAResult.from_state(state, run_dir=config.run_dir, seed=config.seed)
