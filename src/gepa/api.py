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
from gepa.logging.logger import Logger, LoggerProtocol, StdOutLogger
from gepa.logging.logger import LoggerProtocol, StdOutLogger
from gepa.proposer.base import ProposeNewCandidate
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.base import CandidateSelector, LanguageModel, ReflectionComponentSelector
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.strategies.batch_sampler import BatchSampler, EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import (
    CurrentBestCandidateSelector,
    EpsilonGreedyCandidateSelector,
    ParetoCandidateSelector,
    TopKParetoCandidateSelector,
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
    logger: LoggerProtocol = StdOutLogger(),
    frontier_type: FrontierType = "instance",
    perfect_score: float = 1.0,
    run_dir: str | None = None,
    proposer: ProposeNewCandidate | None = None,
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

    active_adapter = adapter

    # Normalize datasets to DataLoader instances
    train_loader = ensure_loader(trainset)
    val_loader = ensure_loader(valset) if valset is not None else train_loader

    # Comprehensive stop_callback logic
    # Convert stop_callbacks to a list if it's not already
    # stop_callbacks_list: list[StopperProtocol] = []
    # if stop_callbacks is not None:
    #     if isinstance(stop_callbacks, Sequence):
    #         stop_callbacks_list.extend(stop_callbacks)
    #     else:
    #         stop_callbacks_list.append(stop_callbacks)

    # Add file stopper if run_dir is provided
    # if run_dir is not None:
    #     stop_file_path = os.path.join(run_dir, "gepa.stop")
    #     file_stopper = FileStopper(stop_file_path)
    #     stop_callbacks_list.append(file_stopper)
    #
    # # Add max_metric_calls stopper if provided
    # if max_metric_calls is not None:
    #     from gepa.utils import MaxMetricCallsStopper
    #
    #     max_calls_stopper = MaxMetricCallsStopper(max_metric_calls)
    #     stop_callbacks_list.append(max_calls_stopper)
    #
    # # Assert that at least one stopping condition is provided
    # if not stop_callbacks_list:
    #     raise ValueError(
    #         "The user must provide at least one of stop_callbacks or max_metric_calls to specify a stopping condition."
    #     )

    experiment_tracker = create_experiment_tracker(
        use_wandb=False,
        use_mlflow=False,
    )

    # Build proposer: use the custom one if provided, otherwise create ReflectiveMutationProposer
    active_proposer = proposer

    engine = GEPAEngine(
        adapter=active_adapter,
        run_dir=run_dir,
        valset=val_loader,
        seed_candidate=seed_candidate,
        perfect_score=perfect_score,
        seed=0,
        reflective_proposer=active_proposer,
        merge_proposer=None,
        frontier_type=frontier_type,
        logger=logger,
        experiment_tracker=experiment_tracker,
    )

    with experiment_tracker:
        state = engine.run()

    return GEPAResult.from_state(state, run_dir=run_dir, seed=0)


def main() -> None:
    """CLI entry point to run optimize with Glean AL adapter and evolutionary proposer."""
    import argparse
    import hashlib
    import json
    from pathlib import Path

    from gepa.adapters.glean_adapter import ALBatchSampler, ALDataInst
    from gepa.adapters.glean_adapter.al_adapter import (
        MODULES,
        ALRunner,
        Judge,
        AssistantALAdapter,
        Thresholds,
        ModuleSpec,
    )
    from gepa.proposer.evolutionary_proposer import EvolutionaryProposer

    parser = argparse.ArgumentParser(
        description="Run GEPA optimize with Glean AL adapter and evolutionary proposer."
    )
    parser.add_argument(
        "--seed_candidate",
        required=True,
        help='Path to a .json file',
    )
    parser.add_argument("--max_metric_calls", type=int, default=10, help="Maximum number of metric calls (default: 10)")
    parser.add_argument("--run_dir", type=Path, default=None, help="Directory for run artifacts and resume")
    parser.add_argument("--student_model", type=str, default='fast', help="Student model name (default: claude)")
    parser.add_argument("--teacher_model", type=str, default="gpt", help="Teacher model name (default: gpt)")
    parser.add_argument("--reflection_lm_model", type=str, default="gpt-5.1", help="Model for reflection LLM (default: gpt-5.1")
    parser.add_argument("--global_token_cap", type=int, default=4096, help="Global token cap for candidates (default: 4096)")
    parser.add_argument("--cookie", type=str, default=None, help="Cookie string for Glean API authentication")
    args = parser.parse_args()

    # Load seed_candidate (JSON string or path)
    raw = args.seed_candidate.strip()
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(f"seed_candidate file not found: {path}")
    seed_candidate_raw = json.loads(path.read_text())
    if not isinstance(seed_candidate_raw, dict) or not seed_candidate_raw:
        raise SystemExit("seed_candidate must be a non-empty JSON object")

    # Flatten seed candidate: TOOL_USAGE list becomes TOOL_USAGE_1-4
    seed_candidate_flat: dict[str, str] = {}
    for k, v in seed_candidate_raw.items():
        if k == "TOOL_USAGE" and isinstance(v, list):
            # Flatten TOOL_USAGE list into TOOL_USAGE_1, TOOL_USAGE_2, etc.
            for i, part in enumerate(v[:4], start=1):
                seed_candidate_flat[f"TOOL_USAGE_{i}"] = str(part)
        elif isinstance(v, str):
            seed_candidate_flat[str(k)] = v
        elif isinstance(v, list) and len(v) == 1:
            # Single-element list, unwrap it
            seed_candidate_flat[str(k)] = str(v[0])
        else:
            raise SystemExit(
                f"seed_candidate values must be strings or (for TOOL_USAGE) a list of strings. "
                f"Got key={k!r} type={type(v)}"
            )

    # Eval set names (hardcoded - always the same)
    # Important: We don't have separate "train" and "val" sets. We train ON eval runs.
    # Mini-batch sampling = running on a small eval set (Small)
    # Full evaluation = running on a larger eval set (Medium/Large)
    SCREEN_EVAL_SET_NAME = "AI Answers Small"  # For screening/mini-batch
    # TODO(Cathy): Update this to use the correct eval set name
    FULL_EVAL_SET_NAME = "AI Answers Small"   # For full evaluation

    # Create multiple ALDataInst objects with different eval_set_versions (dates)
    # Each represents a complete eval set snapshot from different dates
    # This provides diversity in the training data across different time periods

    # Screening/training eval sets - multiple dates for diverse mini-batch sampling
    eval_versions = [
        "20260403",  # March 16, 2026
        "20260406",
        "20260408",
        "20260410",
        "20260412",
    ]

    screen_evalset: list[ALDataInst] = [
        {
            "eval_set_name": SCREEN_EVAL_SET_NAME,
            "eval_set_version": version,
            "deployment_ids": ["scio-prod"],
            "status": "active",
        }
        for version in eval_versions
    ]


    full_evalset: list[ALDataInst] = [
        {
            "eval_set_name": FULL_EVAL_SET_NAME,
            "eval_set_version": version,
            "deployment_ids": ["scio-prod"],
            "status": "active",
        }
        for version in eval_versions
    ]

    al_adapter = AssistantALAdapter(
        runner=ALRunner(
            cookie=args.cookie,
        ),
        judge=Judge(
            cookie=args.cookie,
        ),
        teacher_model=args.teacher_model,
        thresholds=Thresholds(
            quality_min=0.7,
            tools_min=0.7,
            max_student_tokens=100000
        ),
        student_model=args.student_model,
        cache_file="~/eval_cache.json"
    )

    # Module specs with default token budgets
    module_specs = {mid: ModuleSpec(module_id=mid, kind="free_text", token_budget=1024) for mid in MODULES}

    baseline_prompt_hash = hashlib.md5(json.dumps(seed_candidate_flat, sort_keys=True).encode()).hexdigest()

    # Set up shared components for the proposer
    logger = StdOutLogger()
    experiment_tracker = create_experiment_tracker()

    # Reflection LLM callable
    reflection_lm_name = args.reflection_lm_model
    MAX_TOKENS = 4096

    def reflection_llm(prompt: str) -> str:
        import openai

        try:
            completion = openai.chat.completions.create(
                model=reflection_lm_name,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="none",
                max_completion_tokens=MAX_TOKENS,
            )
            content = completion.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            print(f'OpenAI API call failed: {e}')
            return ""

    # Create EvolutionaryProposer
    proposer = EvolutionaryProposer(
        logger=logger,
        trainset=screen_evalset,
        al_adapter=al_adapter,
        reflection_llm=reflection_llm,
        experiment_tracker=experiment_tracker,
        model=args.student_model,
        module_specs=module_specs,
        global_token_cap=args.global_token_cap,
        baseline_prompt_hash=baseline_prompt_hash,
    )

    optimize(
        seed_candidate=seed_candidate_flat,
        trainset=screen_evalset,
        valset=full_evalset,
        adapter=al_adapter,
        proposer=proposer,
        logger=logger,
        # max_metric_calls=args.max_metric_calls,
        run_dir=None,
        # run_dir=str(args.run_dir) if args.run_dir else None,
        frontier_type="objective",
    )


if __name__ == "__main__":
    main()
