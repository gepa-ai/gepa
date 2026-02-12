"""
optimize_anything API - A simplified interface for GEPA optimization.

This module provides a clean, configuration-based API for optimizing any
parameterized system using evolutionary algorithms with LLM-based reflection.
"""

import inspect
import io
import os
import random
import threading
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Literal,
    Protocol,
    Sequence,
    TypeAlias,
)

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
from gepa.core.adapter import DataInst, GEPAAdapter, ProposalFn
from gepa.core.data_loader import ensure_loader
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
from gepa.utils.stdio_capture import ThreadLocalStreamCapture, stream_manager

OptimizableParam = str
Candidate = dict[str, OptimizableParam]

# Cache storage modes for evaluator caching (when cache_evaluation=True)
CacheEvaluationStorage = Literal["memory", "disk", "auto"]


# Sentinel object for single-instance mode
class _SingleInstanceSentinel:
    """Sentinel object used to represent single-instance optimization mode."""

    def __repr__(self) -> str:
        return "<SingleInstanceSentinel>"


_SINGLE_INSTANCE_SENTINEL = _SingleInstanceSentinel()

# Internal key used to wrap a plain-str seed_candidate into a dict.
_STR_CANDIDATE_KEY = "current_candidate"

SideInfo: TypeAlias = dict[str, Any]
"""
Auxiliary evaluation information that powers LLM-based reflection in GEPA optimization.

SideInfo is a dictionary returned alongside each evaluation score from your evaluator.
It contains rich diagnostic information that GEPA's reflective mutation proposer uses to
understand what went wrong and generate targeted improvements to candidates. The more
informative your SideInfo, the more effective the LLM-guided optimization becomes.

**Role in the Optimization Loop:**

1. Your evaluator evaluates a candidate and returns (score, side_info) or just score
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

**Integration with Evaluator:**

Your evaluator should construct SideInfo for each evaluation instance. See the
Evaluator protocol documentation for the complete evaluation interface.
"""


@dataclass
class OptimizationState:
    """Optimization context passed to the evaluator function.

    Contains accumulated optimization state that the evaluator can use
    for warm-starting or informed evaluation decisions.
    """

    best_example_evals: list[dict]
    """Top-K best evaluations for the current example, sorted descending by score.
    Each entry is {"score": float, "side_info": dict}."""


# ---------------------------------------------------------------------------
# Evaluation log context — captures diagnostic output without polluting stdout
# ---------------------------------------------------------------------------


class _LogContext:
    """Thread-safe log buffer for a single evaluator invocation.

    All ``oa.log()`` calls within the same evaluator call write to the same
    buffer, even from child threads (when properly propagated via
    :func:`get_log_context` / :func:`set_log_context`).  Writes are
    serialized with a lock so concurrent threads never interleave.
    """

    def __init__(self) -> None:
        self._buffer = io.StringIO()
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        with self._lock:
            self._buffer.write(text)

    def reset(self) -> str:
        """Read and reset the buffer contents.  Returns accumulated text."""
        with self._lock:
            old = self._buffer
            text = old.getvalue()
            old.close()
            self._buffer = io.StringIO()
            return text


# Thread-local storage for the active _LogContext on each thread.
_log_tls = threading.local()


def _get_log_context() -> "_LogContext | None":
    """Return the active log context for the current thread, or None."""
    return getattr(_log_tls, "context", None)


def _set_log_context(ctx: "_LogContext | None") -> None:
    """Set (or clear) the active log context on the current thread."""
    _log_tls.context = ctx


def get_log_context() -> _LogContext:
    """Return the active log context for the current evaluator call.

    Use this to propagate ``oa.log()`` capture to child threads spawned
    inside your evaluator::

        import threading
        import gepa.optimize_anything as oa

        def my_evaluator(candidate):
            ctx = oa.get_log_context()

            def worker():
                oa.set_log_context(ctx)
                oa.log("from child thread")

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            oa.log("from main evaluator thread")
            return score

    Raises:
        RuntimeError: If called outside an evaluator invocation.
    """
    ctx = _get_log_context()
    if ctx is None:
        raise RuntimeError(
            "No active log context. get_log_context() must be called inside an evaluator passed to optimize_anything()."
        )
    return ctx


def set_log_context(ctx: _LogContext) -> None:
    """Set the log context on the current thread.

    Call this at the start of a child thread to route ``oa.log()`` output
    into the parent evaluator's log buffer.  See :func:`get_log_context`
    for a complete usage example.
    """
    _set_log_context(ctx)


def log(*args: Any, sep: str = " ", end: str = "\n") -> None:
    """Log diagnostic information during evaluation without printing to stdout.

    Has the same calling convention as ``print()``.  Output is captured
    per-evaluator-call (thread-safe) and automatically included in
    side_info under the ``"log"`` key.

    Must only be called inside an evaluator function passed to
    ``optimize_anything``.  Calling it outside that context will emit a
    warning and the output will be silently discarded.

    For child threads spawned by your evaluator, propagate the log context
    via :func:`get_log_context` / :func:`set_log_context`.

    Usage::

        import gepa.optimize_anything as oa
        oa.log("Landing distance:", distance, "meters")
    """
    ctx = _get_log_context()
    if ctx is None:
        warnings.warn(
            "oa.log() called outside of an evaluator function. "
            "Output will be discarded. Only call oa.log() inside your evaluator, "
            "or propagate the log context to child threads via "
            "oa.get_log_context() / oa.set_log_context().",
            stacklevel=2,
        )
        return
    text = sep.join(str(a) for a in args) + end
    ctx.write(text)


# Thread-safe stdout / stderr capture utilities are in gepa.utils.stdio_capture.
# The module-level ``stream_manager`` singleton and ``ThreadLocalStreamCapture``
# class are imported at the top of this file.


class Evaluator(Protocol):
    def __call__(
        self, candidate: str | Candidate, example: object | None = None, **kwargs: Any
    ) -> tuple[float, SideInfo] | float:
        """
        Core evaluation interface for GEPA optimization.

        Evaluates a candidate on a single example, returning a score and optional
        diagnostic information that guide the optimization process.

        **Parameters:**

        - **candidate**: The system configuration to evaluate.  Receives the same
          type passed as ``seed_candidate`` to ``optimize_anything``:
          ``str`` when ``seed_candidate`` is a plain string, or
          ``dict[str, str]`` when ``seed_candidate`` is a dict.

        - **example**: A single example from the dataset to evaluate on (test case,
          input-output pair, task instance). In single-instance mode (when both ``dataset=None``
          and ``valset=None`` are passed to ``optimize_anything``), this parameter will NOT be
          passed to the evaluator at all.

        - **opt_state** *(optional)*: An ``OptimizationState`` instance containing
          accumulated optimization context (e.g. ``best_example_evals``).  To receive it,
          simply declare ``opt_state`` as a parameter in your evaluator signature; it will
          be filtered out automatically if not declared.

          **Note:** ``opt_state`` is a **reserved parameter name**.  GEPA injects this
          keyword argument automatically during evaluation.  Do not use the
          name ``opt_state`` for an unrelated parameter in your evaluator
          signature — doing so will cause it to receive the
          ``OptimizationState`` object instead of whatever value you intended.

        **Returns:**

        The evaluator can return values in two formats:

        1. **Tuple format**: ``(score, side_info)``
           - **score** (float): Fitness score for this instance. Higher is better.
           - **side_info** (SideInfo): Auxiliary evaluation information powering LLM-based
             reflection. See SideInfo documentation for details.

        2. **Score-only format**: ``score`` (float)
           - Returns only the score as a float.
           - Diagnostic output can be provided via ``oa.log()`` (captured per-thread,
             included in side_info under ``"log"``).
           - If ``capture_stdio=True`` in ``EngineConfig``, stdout/stderr are also
             captured automatically and included in side_info.

        **Reserved side_info keys:**

        The keys ``"log"``, ``"stdout"``, and ``"stderr"`` are used by
        GEPA's automatic capture output.  If your evaluator returns a
        ``side_info`` dict containing any of these keys *and* the corresponding
        capture mechanism is active (``oa.log()`` for ``"log"``,
        ``capture_stdio=True`` for ``"stdout"``/``"stderr"``), a warning
        will be emitted and the captured output will be stored under a
        prefixed key (e.g. ``"_gepa_log"``) to avoid data loss.  Rename
        your keys to avoid the warning.

        **Single-Instance Mode (Holistic Optimization):**

        When both ``dataset=None`` and ``valset=None`` are passed to ``optimize_anything``, the
        evaluator is called WITHOUT the ``example`` parameter::

            # With explicit side_info (recommended)
            def evaluate(candidate):
                result = execute_my_code(candidate["code"])
                score = compute_score(result)
                return score, {"Output": str(result), "ExecutionTime": result.time}

            # With oa.log()
            import gepa.optimize_anything as oa
            def evaluate(candidate):
                result = execute_my_code(candidate["code"])
                score = compute_score(result)
                oa.log(f"Output: {result}")
                oa.log(f"Execution time: {result.time}")
                return score

        **Per-Instance Mode:**

        When a dataset is provided, the evaluator is called with each example::

            def evaluate(candidate, example):
                prediction = candidate_predict(candidate, example["input"])
                score = compute_score(prediction, example["expected"])
                return score, {"Input": example["input"], "Prediction": prediction}
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
    use_cloudpickle: bool = True
    track_best_outputs: bool = False

    # Simple stopping conditions
    max_metric_calls: int | None = None
    max_candidate_proposals: int | None = None

    # Strategy selection for the engine
    val_evaluation_policy: EvaluationPolicy | Literal["full_eval"] = "full_eval"
    candidate_selection_strategy: CandidateSelector | Literal["pareto", "current_best", "epsilon_greedy"] = "pareto"
    frontier_type: FrontierType = "instance"

    # Parallelization settings for evaluation
    parallel: bool = False
    max_workers: int | None = None

    # Evaluation caching
    cache_evaluation: bool = False
    cache_evaluation_storage: CacheEvaluationStorage = "auto"

    # Track top-K best evaluations per example, passed to evaluator via OptimizationState
    # Useful for warm-starting optimization from previous best solutions
    best_example_evals_k: int = 30

    # When True, automatically capture stdout/stderr during evaluation and
    # include it in side_info as {"stdout": "...", "stderr": "..."}.
    # Thread-safe via per-thread sys.stdout/stderr replacement.
    #
    # Captures all Python-level output: print(), sys.stdout.write(), and
    # third-party library output — anything that goes through sys.stdout.
    #
    # Does NOT capture output that bypasses Python's sys.stdout:
    # C extensions writing directly to fd 1/2, or subprocesses spawned
    # internally by libraries. For those, use oa.log() or capture subprocess
    # output manually and pass it to oa.log().
    capture_stdio: bool = False


def _build_reflection_prompt_template(objective: str | None = None, background: str | None = None) -> str:
    """
    Build a reflection prompt template dynamically based on provided objective and background.

    Only includes sections that have content, ensuring the prompt feels natural
    regardless of which optional parameters are provided.

    Args:
        objective: High-level goal describing what the optimized component should achieve.
        background: Domain knowledge, constraints, strategies, or implementation requirements.

    Returns:
        A reflection prompt template string with <curr_param> and <side_info> placeholders.
    """
    sections = []

    # System context - always present
    sections.append(
        "You are an expert optimization assistant. Your task is to analyze evaluation "
        "feedback and propose an improved version of a system component."
    )

    # Objective section
    if objective:
        sections.append(f"""
## Optimization Goal

{objective}""")

    # Background/context section
    if background:
        sections.append(f"""
## Domain Context & Constraints

{background}""")

    # Current component and evaluation data - always present
    sections.append("""
## Current Component

The component being optimized:

```
<curr_param>
```

## Evaluation Results

Performance data from evaluating the current component across test cases:

```
<side_info>
```""")

    # Analysis instructions - tailored based on what context is available
    analysis_points = []
    if objective:
        analysis_points.append(
            "- **Goal alignment**: How well does the current component achieve the stated optimization goal?"
        )
    analysis_points.extend(
        [
            "- **Failure patterns**: What specific errors, edge cases, or failure modes appear in the evaluation data?",
            "- **Success patterns**: What behaviors or approaches worked well and should be preserved?",
            "- **Root causes**: What underlying issues explain the observed failures?",
        ]
    )
    if background:
        analysis_points.append(
            "- **Constraint compliance**: Does the component satisfy all requirements from the domain context?"
        )

    analysis_section = "\n".join(analysis_points)
    constraint_line = "\n4. Adheres to all constraints and requirements from the domain context" if background else ""
    sections.append(f"""
## Your Task

Analyze the evaluation results systematically:

{analysis_section}

Based on your analysis, propose an improved version that:
1. Addresses the identified failure patterns and root causes
2. Preserves successful behaviors from the current version
3. Makes meaningful improvements rather than superficial changes{constraint_line}""")

    # Output format - always present
    sections.append("""
## Output Format

Provide ONLY the improved version within ``` blocks. The output must be a complete, 
drop-in replacement for the current component (whether it's a prompt, configuration, 
code, or any other parameter type).
Do not include explanations, commentary, or markdown outside the ``` blocks.""")

    return "\n".join(sections)


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
    reflection_minibatch_size: int | None = None  # Default: 1 for single-instance mode, 3 otherwise
    module_selector: ReflectionComponentSelector | Literal["round_robin", "all"] = "round_robin"
    reflection_lm: LanguageModel | str | None = "openai/gpt-5.1"
    reflection_prompt_template: str | dict[str, str] | None = optimize_anything_reflection_prompt_template
    custom_candidate_proposer: ProposalFn | None = None


# Config for the optional merge proposer
# The existence of this config signals to use merge functionality
@dataclass
class MergeConfig:
    """Configuration for the merge proposer."""

    max_merge_invocations: int = 5
    merge_val_overlap_floor: int = 5


# --- Refiner Configuration ---

DEFAULT_REFINER_PROMPT = """You are a refinement agent improving candidates in an optimization loop.

## What We're Optimizing For
The overall optimization objective is:
{objective}

This tells you what "better" means - use it to guide your improvements.

## Domain Knowledge
{background}

## Your Task
Given a candidate and its evaluation feedback:
1. Understand why it scored the way it did
2. Fix any errors (errors = zero score)
3. Make improvements that move toward the objective
4. Return the complete improved candidate
"""


@dataclass
class RefinerConfig:
    """Configuration for automatic candidate refinement (enabled by default).

    The refiner automatically improves candidates by calling an LLM after each
    evaluation. A `refiner_prompt` is auto-injected into seed candidates and
    co-evolved by GEPA. All non-refiner params are refined together as a JSON dict.

    Refinement runs at least once, then continues until no improvement or max_refinements.
    Set `config.refiner = None` to disable.
    """

    # Language model for refinement (defaults to reflection_lm if not specified)
    refiner_lm: LanguageModel | str | None = None

    # Maximum refinement iterations per evaluation
    max_refinements: int = 1


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

    # Use 'None' to disable these optional components
    merge: MergeConfig | None = None
    refiner: RefinerConfig | None = None

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
        if isinstance(self.refiner, dict):
            self.refiner = RefinerConfig(**self.refiner)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary representation."""
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GEPAConfig":
        """Create config from dictionary representation."""
        return GEPAConfig(**d)


def make_litellm_lm(model_name: str) -> LanguageModel:
    """Convert a LiteLLM model name string to a :class:`LanguageModel` callable.

    The returned callable conforms to the ``LanguageModel`` protocol and
    accepts both a plain ``str`` prompt and a ``list[dict[str, str]]``
    chat-messages list.
    """
    import litellm

    def _lm(prompt: str | list[dict[str, str]]) -> str:
        if isinstance(prompt, str):
            messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        completion = litellm.completion(model=model_name, messages=messages)
        return completion.choices[0].message.content  # type: ignore[union-attr]

    return _lm


class EvaluatorWrapper:
    """Wraps a user's evaluator with GEPA's normalization layer.

    Wraps the user's evaluator to:

    1. Handle single-instance mode (calls evaluator without ``example`` parameter)
    2. Unwrap candidate dict to str when ``str_candidate_mode`` is True
    3. Filter kwargs to only pass what the evaluator accepts (incl. ``opt_state``)
    4. Capture ``oa.log()`` output and optionally stdout/stderr
    5. Detect whether evaluator returns ``(score, side_info)`` tuple or just ``score``
    6. Normalize return value to always be ``(score, output, side_info)`` tuple

    The wrapper is callable — calling the instance invokes the wrapped evaluator::

        wrapper = EvaluatorWrapper(eval_fn, single_instance_mode=True)
        score, output, side_info = wrapper(candidate, example=example)
    """

    def __init__(
        self,
        evaluator_fn: Callable[..., Any],
        single_instance_mode: bool,
        capture_stdio: bool = False,
        str_candidate_mode: bool = False,
    ) -> None:
        self._capture_stdio = capture_stdio

        # Inspect the evaluator's signature once to determine which kwargs it accepts.
        sig = inspect.signature(evaluator_fn)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_keyword:
            accepted_params = None  # accept all
        else:
            accepted_params = set(sig.parameters.keys())

        def _filter_kwargs(kwargs: dict) -> dict:
            if accepted_params is None:
                return kwargs
            return {k: v for k, v in kwargs.items() if k in accepted_params}

        def wrapped_evaluator(
            candidate: Candidate, example: object | None = None, **kwargs: Any
        ) -> tuple[float, Any, SideInfo]:
            # Create a fresh, shared log context for this evaluator call.
            # The same _LogContext is accessible from child threads via
            # oa.get_log_context() / oa.set_log_context().
            log_ctx = _LogContext()
            _set_log_context(log_ctx)

            # Build full kwargs dict. In single-instance mode, don't forward
            # example to the evaluator at all.
            if single_instance_mode:
                all_kwargs = kwargs
            else:
                all_kwargs = {"example": example, **kwargs}

            filtered = _filter_kwargs(all_kwargs)

            # Unwrap candidate for str_candidate_mode
            eval_candidate: Candidate | str = candidate
            if str_candidate_mode:
                eval_candidate = candidate[_STR_CANDIDATE_KEY]

            # Acquire per-thread stream capture from the shared manager per-call.
            # This scopes the sys.stdout/stderr replacement to only the duration
            # of evaluator execution, restoring the originals between calls.
            stdout_capturer: ThreadLocalStreamCapture | None = None
            stderr_capturer: ThreadLocalStreamCapture | None = None
            if capture_stdio:
                stdout_capturer, stderr_capturer = stream_manager.acquire()
                stdout_capturer.start_capture()
                stderr_capturer.start_capture()

            exc: Exception | None = None
            result: tuple[float, SideInfo] | float | None = None
            try:
                result = evaluator_fn(eval_candidate, **filtered)
            except Exception as e:
                exc = e
            finally:
                captured_stdout = stdout_capturer.stop_capture() if stdout_capturer else ""
                captured_stderr = stderr_capturer.stop_capture() if stderr_capturer else ""
                if capture_stdio:
                    stream_manager.release()
                log_output = log_ctx.reset()
                _set_log_context(None)

            # If evaluator raised, preserve captured diagnostics before re-raising
            if exc is not None:
                fail_side_info: SideInfo = {"error": str(exc)}
                if log_output:
                    fail_side_info["log"] = log_output
                if captured_stdout:
                    fail_side_info["stdout"] = captured_stdout
                if captured_stderr:
                    fail_side_info["stderr"] = captured_stderr
                score_neg_inf: float = float("-inf")
                return score_neg_inf, None, fail_side_info

            # Detect return type and normalize to (score, output, side_info)
            if isinstance(result, tuple):
                score, side_info = result
                side_info = dict(side_info) if side_info is not None else {}

                # Inject captured output, renaming on collision with a warning
                injected: dict[str, str] = {}
                if log_output:
                    injected["log"] = log_output
                if captured_stdout:
                    injected["stdout"] = captured_stdout
                if captured_stderr:
                    injected["stderr"] = captured_stderr

                for key in list(injected):
                    if key in side_info:
                        prefixed = f"_gepa_{key}"
                        warnings.warn(
                            f"Your evaluator returned side_info with key '{key}' that conflicts "
                            f"with GEPA's captured output key. The captured output will be stored "
                            f"under '{prefixed}' instead.",
                            stacklevel=2,
                        )
                        injected[prefixed] = injected.pop(key)

                side_info.update(injected)
                return score, None, side_info
            else:
                score = result
                auto_side_info: SideInfo = {}
                if captured_stdout:
                    auto_side_info["stdout"] = captured_stdout
                if captured_stderr:
                    auto_side_info["stderr"] = captured_stderr
                if log_output:
                    auto_side_info["log"] = log_output
                return score, None, auto_side_info

        self._wrapped = wrapped_evaluator

    def __call__(
        self, candidate: Candidate, example: object | None = None, **kwargs: Any
    ) -> tuple[float, Any, SideInfo]:
        return self._wrapped(candidate, example=example, **kwargs)


def optimize_anything(
    seed_candidate: str | Candidate | list[Candidate],
    evaluator: Evaluator,
    dataset: list[DataInst] | None = None,
    valset: list[DataInst] | None = None,
    objective: str | None = None,
    background: str | None = None,
    config: GEPAConfig | None = None,
) -> GEPAResult:
    """
    Optimize any parameterized system using evolutionary algorithms with LLM-based reflection.

    This is the main entry point for GEPA optimization. It accepts a seed candidate (initial
    parameter configuration), an evaluator function, and optionally a dataset of test cases.

    **Two modes of operation:**

    1. **Per-instance mode** (when ``dataset`` is provided): The evaluator is called
       once per example in the dataset, and scores are aggregated.

    2. **Single-instance mode** (default, when ``dataset=None``): The evaluator is called
       without the ``example`` parameter.

    Args:
        seed_candidate: Initial candidate(s) to start optimization from.  Can be:
            - A plain ``str`` (wrapped internally; the evaluator receives the
              raw ``str``).
            - A ``dict[str, str]`` mapping parameter names to values.
            - A ``list`` of candidate dicts for population-based initialization.
        evaluator: Function that evaluates a candidate, returning ``score`` or
            ``(score, side_info)``.  See :class:`Evaluator` protocol for details.
        dataset: Examples to evaluate on. ``None`` → single-instance mode.
        valset: Optional separate validation set. Defaults to ``dataset``.
        objective: High-level goal guiding LLM reflection.
        background: Domain knowledge / constraints for reflection.
        config: Optional :class:`GEPAConfig`.

    Returns:
        :class:`GEPAResult` with the best candidate(s), history, and metrics.

    Example (per-instance mode)::

        result = optimize_anything(
            seed_candidate={"prompt": "Solve this math problem:"},
            evaluator=my_evaluator,
            dataset=test_cases,
            objective="Generate prompts that help solve math word problems.",
            config=GEPAConfig(engine=EngineConfig(max_metric_calls=100)),
        )

    Example (single-instance mode with str seed)::

        import gepa.optimize_anything as oa

        def evaluate(candidate: str) -> float:
            result = run_code(candidate)
            oa.log(f"Output: {result}")
            return compute_score(result)

        result = optimize_anything(
            seed_candidate="def solve(): return 0",
            evaluator=evaluate,
            objective="Evolve code to maximize the objective function.",
            config=GEPAConfig(engine=EngineConfig(max_metric_calls=500)),
        )
    """
    # Use default config if not provided
    if config is None:
        config = GEPAConfig()

    # Normalize seed_candidate: str -> {_STR_CANDIDATE_KEY: str}
    str_candidate_mode = isinstance(seed_candidate, str)
    if str_candidate_mode:
        seed_candidate = {_STR_CANDIDATE_KEY: seed_candidate}

    # Detect single-instance mode: when both dataset=None and valset=None
    single_instance_mode = dataset is None and valset is None

    # Set reflection_minibatch_size default based on mode (if not explicitly set)
    if config.reflection.reflection_minibatch_size is None:
        config.reflection.reflection_minibatch_size = 1 if single_instance_mode else 3

    # Handle single-instance mode: when both dataset=None and valset=None, create a
    # dataset with a single sentinel element. The evaluator will be called
    # without the example parameter.
    if single_instance_mode:
        effective_dataset: list[DataInst] = [_SINGLE_INSTANCE_SENTINEL]  # type: ignore[list-item]
    else:
        effective_dataset = dataset if dataset is not None else [None]  # type: ignore[list-item]

    # Wrap the evaluator to handle signature normalization, log/stdout capture, etc.
    wrapped_evaluator = EvaluatorWrapper(
        evaluator,
        single_instance_mode,
        capture_stdio=config.engine.capture_stdio,
        str_candidate_mode=str_candidate_mode,
    )

    # Resolve cache mode: cache_evaluation controls on/off, cache_evaluation_storage controls where
    if not config.engine.cache_evaluation:
        resolved_cache_mode = "off"
        if config.engine.cache_evaluation_storage != "auto":
            warnings.warn(
                f"cache_evaluation_storage={config.engine.cache_evaluation_storage!r} is set but "
                f"cache_evaluation=False, so caching is disabled. Set cache_evaluation=True to "
                f"enable caching with the specified storage mode.",
                stacklevel=2,
            )
    elif config.engine.cache_evaluation_storage == "auto":
        resolved_cache_mode = "disk" if config.engine.run_dir else "memory"
    else:
        resolved_cache_mode = config.engine.cache_evaluation_storage

    # Validate disk mode requires run_dir
    if resolved_cache_mode == "disk" and not config.engine.run_dir:
        raise ValueError("cache_evaluation_storage='disk' requires run_dir in EngineConfig")

    # Configure cloudpickle for code execution subprocess serialization
    from gepa.utils.code_execution import set_use_cloudpickle

    set_use_cloudpickle(config.engine.use_cloudpickle)

    active_adapter: GEPAAdapter = OptimizeAnythingAdapter(
        evaluator=wrapped_evaluator,
        parallel=config.engine.parallel,
        max_workers=config.engine.max_workers,
        refiner_config=config.refiner,
        best_example_evals_k=config.engine.best_example_evals_k,
        objective=objective,
        background=background,
        cache_mode=resolved_cache_mode,
        cache_dir=config.engine.run_dir,
    )

    # Normalize datasets to DataLoader instances
    train_loader = ensure_loader(effective_dataset)
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

    # Add max_candidate_proposals stopper if provided
    if config.engine.max_candidate_proposals is not None:
        # Define stopper inline to avoid modifying stop_condition.py
        # Note: state.i starts at -1, and is incremented at the START of each loop iteration.
        # The stopper is checked BEFORE the increment, so when state.i = N-1, we're about to
        # run proposal N. To allow exactly max_proposals proposals, we stop when
        # state.i >= max_proposals - 1 (i.e., we've completed max_proposals proposals).
        class MaxCandidateProposalsStopper:
            def __init__(self, max_proposals: int):
                self.max_proposals = max_proposals

            def __call__(self, gepa_state) -> bool:
                return gepa_state.i >= self.max_proposals - 1

        proposals_stopper = MaxCandidateProposalsStopper(config.engine.max_candidate_proposals)
        stop_callbacks_list.append(proposals_stopper)

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

    # Default refiner_lm to reflection_lm name BEFORE converting reflection_lm to callable
    if config.refiner is not None and config.refiner.refiner_lm is None:
        config.refiner.refiner_lm = config.reflection.reflection_lm

    # Convert reflection_lm string to callable
    if isinstance(config.reflection.reflection_lm, str):
        config.reflection.reflection_lm = make_litellm_lm(config.reflection.reflection_lm)

    # Convert refiner_lm string to LiteLLM callable (if refiner is enabled)
    if config.refiner is not None:
        if isinstance(config.refiner.refiner_lm, str):
            config.refiner.refiner_lm = make_litellm_lm(config.refiner.refiner_lm)

    # Auto-inject refiner_prompt into seed_candidate(s) if refiner is enabled
    if config.refiner is not None:
        formatted_refiner_prompt = DEFAULT_REFINER_PROMPT.format(
            objective=objective or "Maximize the score",
            background=background or "No additional background provided.",
        )
        candidates_to_check = seed_candidate if isinstance(seed_candidate, list) else [seed_candidate]
        for sc in candidates_to_check:
            if "refiner_prompt" not in sc:
                sc["refiner_prompt"] = formatted_refiner_prompt
            # If user provides their own refiner_prompt, use it (allows custom refiner prompts)

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
            minibatch_size=config.reflection.reflection_minibatch_size, rng=rng
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

    # --- 9. Build reflection prompt template from objective/background if provided ---
    # Check for conflicting configuration: user cannot provide both objective/background
    # AND a custom reflection_prompt_template (these are mutually exclusive approaches)
    user_provided_custom_template = (
        config.reflection.reflection_prompt_template is not None
        and config.reflection.reflection_prompt_template != optimize_anything_reflection_prompt_template
    )
    # Treat empty strings as "not provided" - only non-empty strings count
    user_provided_objective_or_background = bool(objective) or bool(background)

    if user_provided_custom_template and user_provided_objective_or_background:
        raise ValueError(
            "Cannot specify both 'objective'/'background' parameters and a custom "
            "'config.reflection.reflection_prompt_template'. These are mutually exclusive options. "
            "Either use objective/background to auto-generate a reflection prompt, or provide "
            "your own custom template via config.reflection.reflection_prompt_template."
        )

    # If objective or background are provided, build a custom reflection prompt template
    # with those values filled in, creating a template with <curr_param> and <side_info> placeholders
    if user_provided_objective_or_background:
        config.reflection.reflection_prompt_template = _build_reflection_prompt_template(
            objective=objective, background=background
        )

    # --- 10. Validate reflection prompt template ---
    if config.reflection.reflection_prompt_template is not None:
        assert not (
            active_adapter is not None and getattr(active_adapter, "propose_new_texts", None) is not None
        ), (
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

    # --- 11. Build reflective proposer from ReflectionConfig ---
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

    # Define evaluator function for merge proposer
    def merge_evaluator(
        inputs: list[DataInst], prog: Candidate
    ) -> tuple[list[object], list[float], list[dict[str, float]] | None]:
        eval_out = active_adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores, eval_out.objective_scores

    # --- 12. Build merge proposer from MergeConfig (if provided) ---
    merge_proposer: MergeProposer | None = None
    if config.merge is not None:
        merge_proposer = MergeProposer(
            logger=config.tracking.logger,
            valset=val_loader,
            evaluator=merge_evaluator,
            use_merge=True,
            max_merge_invocations=config.merge.max_merge_invocations,
            rng=rng,
            val_overlap_floor=config.merge.merge_val_overlap_floor,
        )

    # --- 13. Create evaluation cache if enabled ---
    evaluation_cache: EvaluationCache[Any, Any] | None = None
    if config.engine.cache_evaluation:
        evaluation_cache = EvaluationCache[Any, Any]()

    # --- 14. Build the main engine from EngineConfig ---
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
        evaluation_cache=evaluation_cache,
    )

    # --- 15. Run optimization ---
    with experiment_tracker:
        state = engine.run()

    return GEPAResult.from_state(state, run_dir=config.engine.run_dir, seed=config.engine.seed)
