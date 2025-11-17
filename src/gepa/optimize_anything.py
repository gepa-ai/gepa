"""
optimize_anything API - A simplified interface for GEPA optimization.

This module provides a clean, configuration-based API for optimizing any
parameterized system using evolutionary algorithms with LLM-based reflection.
"""

import os
import random
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Literal,
    Protocol,
    Sequence,
    TypeAlias,
)

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
from gepa.core.adapter import DataInst, GEPAAdapter, ProposalFn, RolloutOutput
from gepa.core.data_loader import ensure_loader
from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.core.state import FrontierType
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
Candidate = dict[str, OptimizableParam]

SideInfo: TypeAlias = dict[str, Any]
"""
Auxiliary evaluation information that powers LLM-based reflection in GEPA optimization.

SideInfo is a dictionary returned alongside each evaluation score from your fitness function.
It contains rich diagnostic information that GEPA's reflective mutation proposer uses to
understand what went wrong and generate targeted improvements to candidates. The more
informative your SideInfo, the more effective the LLM-guided optimization becomes.

**Role in the Optimization Loop:**

1. Your fitness_fn evaluates a candidate and returns (score, rollout_output, side_info)
2. GEPA collects side_info across multiple evaluation instances
3. The reflective proposer presents this information to an LLM with the candidate parameters
4. The LLM analyzes failures/successes and proposes parameter improvements
5. New candidates are generated and the cycle repeats

**Structure:**

SideInfo has two primary components:

1. **Multi-dimensional Scores** (optional, via "scores" field):
    Additional numeric metrics beyond the main fitness score, used for multi-objective
    optimization and providing quantitative feedback to the reflection LLM.
    
    - Must be specified as: {"scores": {"metric1": value1, "metric2": value2, ...}}
    - All scores must follow "higher is better" convention (invert if needed)
    - Enables Pareto-optimal candidate selection across multiple objectives
    - Example: {"scores": {"accuracy": 0.85, "speed": 12.5, "robustness": 0.72}}

2. **Contextual Information** (all other fields):
    Qualitative and quantitative information that helps the LLM understand the evaluation
    context and guide parameter improvements. Can include text, numbers, or structured data.
    
    Example fields:
    - "Input" / "Inputs": The input(s) provided to the candidate
    - "Output" / "Outputs": What the candidate actually produced
    - "Expected" / "ExpectedOutput": Ground truth or desired output, if available
    - "Feedback": Qualitative assessment of candidate performance. Can be human-written or machine-generated.
    - "Error" / "ErrorMessage": Parsing / Compilation / Execution / Other errors or failure descriptions
    - "Reasoning": Step-by-step reasoning trace or intermediate computations
    - "ProfilerData": Performance profiling information
    - "ValidationErrors": Schema validation or constraint violations
    
    Design principle: Include anything that would help a human understand why the
    candidate succeeded or failed on this instance. Be descriptive in field names and values.

**Parameter-Specific Information** (optional):

For multi-parameter optimization, you can provide targeted diagnostic information for
individual parameters using "<param_name>_specific_info" fields. This is especially
useful when different parameters have different failure modes.

- Format: {"<param_name>_specific_info": <nested_SideInfo_dict>}
- The nested dict can contain its own "scores" and contextual fields
- During reflection on parameter X, GEPA combines:
    * All top-level SideInfo fields (shared context)
    * The "<X>_specific_info" fields (parameter-specific context)
- Use cases: parsing errors, parameter-specific constraints, component-level metrics

**Complete Example:**

```python
{
    # Multi-objective scores
    "scores": {
        "accuracy": 0.73,
        "response_time_ms": 850,  # Already inverted (higher=better)
        "user_satisfaction": 4.2
    },
    
    # Shared evaluation context
    "Input": "Translate 'Hello world' to French",
    "Output": "Salut monde",
    "Expected": "Bonjour le monde",
    "Feedback": "Translation is too informal for the context",
    "ExecutionTime": 0.85, # Numeric value, still useful for context, but not "higher is better".
    
    # Parameter-specific diagnostics
    "system_prompt_specific_info": {
        "scores": {
            "tone_appropriateness": 0.3  # Low score explains the issue
        },
        "ParsedTone": "casual",
        "ExpectedTone": "formal",
        "Analysis": "System prompt led to overly casual translation"
    },
    
    "temperature_specific_info": {
        "Feedback": "Temperature 0.9 caused inconsistent outputs"
    }
}
```

**Best Practices:**

- **Be generous with information**: More context = better LLM reflection
- **Include failures prominently**: Error messages and failure reasons are crucial
- **Use consistent field names**: Helps the LLM recognize patterns across evaluations
- **Normalize scores**: Ensure "higher is better" for all metrics in "scores"
- **Add context for metrics**: Don't just say accuracy=0.7, explain what was wrong
- **Use parameter-specific info judiciously**: Only when you have parameter-targeted insights

**Integration with Fitness Function:**

Your fitness_fn should construct SideInfo for each evaluation instance. See the
FitnessFn protocol documentation for the complete evaluation interface.
"""


class FitnessFn(Protocol):
    def __call__(
        self, candidate: Candidate, batch: Sequence[DataInst], **kwargs: Any
    ) -> Sequence[tuple[float, RolloutOutput, SideInfo]]:
        """
        Core evaluation interface for GEPA optimization.

        Evaluates a candidate (parameterized system) on a batch of data instances,
        returning scores and diagnostic information that guide the optimization process.

        **Parameters:**

        - **candidate**: Dictionary mapping parameter names to values representing one
          system configuration.

          Examples:
          - Prompt optimization: `{"system_prompt": "You are...", "user_prefix": "Answer:"}`
          - Hyperparameter tuning: `{"learning_rate": "0.001", "batch_size": "32"}`
          - Multi-component system: `{"component_1_config": "...", "component_2_config": "..."}`

        - **batch**: List of data instances to evaluate on. Each instance is one evaluation
          example (test case, input-output pair, task instance). Batch size controlled by
          GEPA's BatchSampler strategy.

        - **kwargs**: Additional keyword arguments passed during evaluation.

        **Returns:**

        List of evaluation results, one per batch instance. Each result is a 3-tuple:

        1. **score** (float): Fitness score for this instance. Higher is better. Must be finite.
           Primary optimization signal.

        2. **rollout_output** (RolloutOutput): Actual output produced by the candidate.
           Opaque to GEPA—only tracked for returning best outputs per data instance in
           GEPAResult. Can be any type (str, dict, custom object).

        3. **side_info** (SideInfo): Auxiliary evaluation information powering LLM-based
           reflection. See SideInfo documentation for detailed guidance on structure and
           best practices.

        **Requirements:**

        - Return list length must equal `len(batch)`
        - Include rich diagnostic information in side_info for effective reflection
        - All scores follow "higher is better" convention
        """
        ...


# --- Component 1: Engine & Stopping Configuration ---
# Controls the main GEPAEngine run loop
@dataclass
class EngineConfig:
    """Configuration for the GEPA engine run loop."""

    run_dir: str | None = None
    seed: int = 0
    display_progress_bar: bool = False
    raise_on_exception: bool = True
    use_cloudpickle: bool = False
    track_best_outputs: bool = False

    # Simple stopping conditions
    max_metric_calls: int | None = None

    # Strategy selection for the engine
    val_evaluation_policy: EvaluationPolicy | Literal["full_eval"] = "full_eval"
    candidate_selection_strategy: CandidateSelector | Literal["pareto", "current_best", "epsilon_greedy"] = "pareto"
    frontier_type: FrontierType = "instance"


optimize_anything_reflection_prompt_template: str = """I am optimizing a parameter in my system. The current parameter value is:
```
<curr_param>
```

Below is evaluation data showing how this parameter value performed across multiple test cases. The data contains performance metrics, diagnostic information, and other relevant details from the evaluation:
```
<side_info>
```

Your task is to propose a new, improved parameter value that can be used as a drop-in replacement for the current one.

Carefully analyze all the evaluation data provided above. Look for patterns that indicate what works and what doesn't. Pay special attention to:
- Performance metrics and how they correlate with parameter behavior
- Recurring issues, errors, or failure patterns across multiple test cases
- Successful patterns or behaviors that should be preserved or enhanced
- Any domain-specific requirements, constraints, or factual information revealed in the evaluation data
- Specific technical details that are crucial for understanding the parameter's role

Based on your analysis, propose a new parameter value that addresses the identified issues while maintaining or improving upon what works well. Your proposal should be directly informed by the patterns and insights from the evaluation data.

Provide the new parameter value within ``` blocks."""


# --- Component 2: Proposer Configurations ---
# Config for the main reflective proposer
@dataclass
class ReflectionConfig:
    """Configuration for the reflective mutation proposer."""

    skip_perfect_score: bool = False
    perfect_score: float | None = None
    batch_sampler: BatchSampler | Literal["epoch_shuffled"] = "epoch_shuffled"
    reflection_minibatch_size: int = 3
    module_selector: ReflectionComponentSelector | Literal["round_robin", "all"] = "round_robin"
    reflection_lm: LanguageModel | str | None = None
    reflection_prompt_template: str | dict[str, str] | None = optimize_anything_reflection_prompt_template
    custom_candidate_proposer: ProposalFn | None = None


# Config for the optional merge proposer
# The existence of this config signals to use merge functionality
@dataclass
class MergeConfig:
    """Configuration for the merge proposer."""

    max_merge_invocations: int = 5
    merge_val_overlap_floor: int = 5


# --- Component 3: Experiment Tracking Configuration ---
# Groups all logging and tracking settings
@dataclass
class TrackingConfig:
    """Configuration for experiment tracking and logging."""

    logger: LoggerProtocol | None = None
    use_wandb: bool = False
    wandb_api_key: str | None = None
    wandb_init_kwargs: dict[str, Any] | None = None
    use_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None


# --- The NEW Top-Level Configuration Class ---
# This class is now a clean container for component-specific configs
@dataclass
class GEPAConfig:
    """
    Top-level configuration for GEPA optimization.

    This config follows a nested structure inspired by Hugging Face Trainer,
    Lightning, and Hydra, where each component has its own configuration class.
    """

    # Component configurations
    engine: EngineConfig = field(default_factory=EngineConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # Use 'None' as the default to disable this component
    merge: MergeConfig | None = None

    # Complex callbacks that aren't serializable
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None

    def __post_init__(self):
        """Handle dicts passed in (e.g., from a JSON/YAML file)."""
        if isinstance(self.engine, dict):
            self.engine = EngineConfig(**self.engine)
        if isinstance(self.reflection, dict):
            self.reflection = ReflectionConfig(**self.reflection)
        if isinstance(self.tracking, dict):
            self.tracking = TrackingConfig(**self.tracking)
        if isinstance(self.merge, dict):
            self.merge = MergeConfig(**self.merge)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary representation."""
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GEPAConfig":
        """Create config from dictionary representation."""
        return GEPAConfig(**d)


def optimize_anything(
    seed_candidate: Candidate | list[Candidate],
    fitness_fn: FitnessFn,
    dataset: list[DataInst],
    valset: list[DataInst] | None = None,
    config: GEPAConfig | None = None,
) -> GEPAResult:
    # Use default config if not provided
    if config is None:
        config = GEPAConfig()

    active_adapter: GEPAAdapter = OptimizeAnythingAdapter(fitness_fn=fitness_fn)

    # Normalize datasets to DataLoader instances
    train_loader = ensure_loader(dataset)
    val_loader = ensure_loader(valset) if valset is not None else train_loader

    # --- 1. Build stoppers from the EngineConfig and root config ---
    stop_callbacks_list: list[StopperProtocol] = []

    # Add custom stop callbacks if provided
    if config.stop_callbacks is not None:
        if isinstance(config.stop_callbacks, Sequence):
            stop_callbacks_list.extend(config.stop_callbacks)
        else:
            stop_callbacks_list.append(config.stop_callbacks)

    # Add file stopper if run_dir is provided
    if config.engine.run_dir is not None:
        stop_file_path = os.path.join(config.engine.run_dir, "gepa.stop")
        file_stopper = FileStopper(stop_file_path)
        stop_callbacks_list.append(file_stopper)

    # Add max_metric_calls stopper if provided
    if config.engine.max_metric_calls is not None:
        from gepa.utils import MaxMetricCallsStopper

        max_calls_stopper = MaxMetricCallsStopper(config.engine.max_metric_calls)
        stop_callbacks_list.append(max_calls_stopper)

    # Assert that at least one stopping condition is provided
    if not stop_callbacks_list:
        raise ValueError(
            "At least one stopping condition must be provided via config.engine.max_metric_calls or config.stop_callbacks."
        )

    # Create composite stopper if multiple stoppers, or use single stopper
    stop_callback: StopperProtocol
    if len(stop_callbacks_list) == 1:
        stop_callback = stop_callbacks_list[0]
    else:
        from gepa.utils import CompositeStopper

        stop_callback = CompositeStopper(*stop_callbacks_list)

    # --- 2. Validate and setup reflection LM ---
    if not hasattr(active_adapter, "propose_new_texts"):
        assert config.reflection.reflection_lm is not None, (
            f"reflection_lm was not provided. The adapter '{active_adapter!s}' does not provide a propose_new_texts method, "
            + "and hence, GEPA will use the default proposer, which requires a reflection_lm to be specified."
        )

    # Convert string LM to callable
    if isinstance(config.reflection.reflection_lm, str):
        import litellm

        reflection_lm_name = config.reflection.reflection_lm

        def _reflection_lm(prompt: str | list[dict[str, str]]) -> str:
            if isinstance(prompt, str):
                completion = litellm.completion(
                    model=reflection_lm_name, messages=[{"role": "user", "content": prompt}]
                )
            else:
                completion = litellm.completion(model=reflection_lm_name, messages=prompt)
            return completion.choices[0].message.content  # type: ignore

        config.reflection.reflection_lm = _reflection_lm

    # Setup default logger if not provided
    if config.tracking.logger is None:
        config.tracking.logger = StdOutLogger()

    # --- 3. Setup random number generator ---
    rng = random.Random(config.engine.seed)

    # --- 4. Build candidate selector from EngineConfig ---
    candidate_selector: CandidateSelector
    if isinstance(config.engine.candidate_selection_strategy, str):
        factories = {
            "pareto": lambda: ParetoCandidateSelector(rng=rng),
            "current_best": lambda: CurrentBestCandidateSelector(),
            "epsilon_greedy": lambda: EpsilonGreedyCandidateSelector(epsilon=0.1, rng=rng),
        }

        try:
            candidate_selector = factories[config.engine.candidate_selection_strategy]()
        except KeyError as exc:
            raise ValueError(
                f"Unknown candidate_selector strategy: {config.engine.candidate_selection_strategy}. "
                "Supported strategies: 'pareto', 'current_best', 'epsilon_greedy'"
            ) from exc
    elif isinstance(config.engine.candidate_selection_strategy, CandidateSelector):
        candidate_selector = config.engine.candidate_selection_strategy
    else:
        raise TypeError(
            "candidate_selection_strategy must be a supported string strategy or an instance of CandidateSelector."
        )

    # --- 5. Build evaluation policy from EngineConfig ---
    if config.engine.val_evaluation_policy is None or config.engine.val_evaluation_policy == "full_eval":
        config.engine.val_evaluation_policy = FullEvaluationPolicy()
    elif not isinstance(config.engine.val_evaluation_policy, EvaluationPolicy):
        raise ValueError(
            f"val_evaluation_policy should be 'full_eval' or an EvaluationPolicy instance, but got {type(config.engine.val_evaluation_policy)}"
        )

    # --- 6. Build module selector from ReflectionConfig ---
    if isinstance(config.reflection.module_selector, str):
        module_selector_cls = {
            "round_robin": RoundRobinReflectionComponentSelector,
            "all": AllReflectionComponentSelector,
        }.get(config.reflection.module_selector)

        assert module_selector_cls is not None, (
            f"Unknown module_selector strategy: {config.reflection.module_selector}. "
            "Supported strategies: 'round_robin', 'all'"
        )

        module_selector_instance: ReflectionComponentSelector = module_selector_cls()
    else:
        module_selector_instance = config.reflection.module_selector

    # --- 7. Build batch sampler from ReflectionConfig ---
    if config.reflection.batch_sampler == "epoch_shuffled":
        config.reflection.batch_sampler = EpochShuffledBatchSampler(
            minibatch_size=config.reflection.reflection_minibatch_size or 3, rng=rng
        )
    else:
        assert config.reflection.reflection_minibatch_size is None, (
            "reflection_minibatch_size only accepted if batch_sampler is 'epoch_shuffled'"
        )

    # --- 8. Build experiment tracker from TrackingConfig ---
    experiment_tracker = create_experiment_tracker(
        use_wandb=config.tracking.use_wandb,
        wandb_api_key=config.tracking.wandb_api_key,
        wandb_init_kwargs=config.tracking.wandb_init_kwargs,
        use_mlflow=config.tracking.use_mlflow,
        mlflow_tracking_uri=config.tracking.mlflow_tracking_uri,
        mlflow_experiment_name=config.tracking.mlflow_experiment_name,
    )

    # --- 9. Validate reflection prompt template ---
    if config.reflection.reflection_prompt_template is not None:
        assert not (active_adapter is not None and getattr(active_adapter, "propose_new_texts", None) is not None), (
            f"Adapter {active_adapter!s} provides its own propose_new_texts method; "
            "reflection_prompt_template will be ignored. Set reflection_prompt_template to None."
        )

        # Validate template(s) - can be a single string or dict of templates
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        if isinstance(config.reflection.reflection_prompt_template, dict):
            for param_name, template in config.reflection.reflection_prompt_template.items():
                try:
                    InstructionProposalSignature.validate_prompt_template(template)
                except ValueError as e:
                    raise ValueError(f"Invalid reflection_prompt_template for parameter '{param_name}': {e}") from e
        else:
            InstructionProposalSignature.validate_prompt_template(config.reflection.reflection_prompt_template)

    # --- 10. Build reflective proposer from ReflectionConfig ---
    reflective_proposer = ReflectiveMutationProposer(
        logger=config.tracking.logger,
        trainset=train_loader,
        adapter=active_adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector_instance,
        batch_sampler=config.reflection.batch_sampler,
        perfect_score=config.reflection.perfect_score,
        skip_perfect_score=config.reflection.skip_perfect_score,
        experiment_tracker=experiment_tracker,
        reflection_lm=config.reflection.reflection_lm,
        reflection_prompt_template=config.reflection.reflection_prompt_template,
        custom_candidate_proposer=config.reflection.custom_candidate_proposer,
    )

    # Define evaluator for merge proposer
    def evaluator(inputs: list[DataInst], prog: Candidate) -> tuple[list[RolloutOutput], list[float]]:
        eval_out = active_adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores

    # --- 11. Build merge proposer from MergeConfig (if provided) ---
    merge_proposer: MergeProposer | None = None
    if config.merge is not None:
        merge_proposer = MergeProposer(
            logger=config.tracking.logger,
            valset=val_loader,
            evaluator=evaluator,
            use_merge=True,
            max_merge_invocations=config.merge.max_merge_invocations,
            rng=rng,
            val_overlap_floor=config.merge.merge_val_overlap_floor,
        )

    # --- 12. Build the main engine from EngineConfig ---
    engine = GEPAEngine(
        adapter=active_adapter,
        run_dir=config.engine.run_dir,
        valset=val_loader,
        seed_candidate=[seed_candidate] if not isinstance(seed_candidate, list) else seed_candidate,
        perfect_score=config.reflection.perfect_score,
        seed=config.engine.seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=merge_proposer,
        frontier_type=config.engine.frontier_type,
        logger=config.tracking.logger,
        experiment_tracker=experiment_tracker,
        track_best_outputs=config.engine.track_best_outputs,
        display_progress_bar=config.engine.display_progress_bar,
        raise_on_exception=config.engine.raise_on_exception,
        stop_callback=stop_callback,
        val_evaluation_policy=config.engine.val_evaluation_policy,
        use_cloudpickle=config.engine.use_cloudpickle,
    )

    # --- 13. Run optimization ---
    with experiment_tracker:
        state = engine.run()

    return GEPAResult.from_state(state, run_dir=config.engine.run_dir, seed=config.engine.seed)
