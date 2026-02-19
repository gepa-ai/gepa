"""
``optimize_anything`` — a universal API for optimizing any text parameter.

This is the primary public API for GEPA.  You declare **what** to optimize
(a text-representable artifact — code, prompts, agent architectures,
configurations, policies, SVG graphics) and **how** to measure it (an
evaluator); ``optimize_anything`` handles the **how**: prompt construction,
LLM reflection, candidate selection, and Pareto-efficient search.

The key insight is that a wide range of problems can be formulated as
optimizing a text artifact: speeding up a CUDA kernel, tuning a scheduling
policy, refining a prompt template, or redesigning an agent architecture.
If it can be serialized to a string and its quality measured, an LLM can
reason about it and propose improvements.

Core workflow::

    seed_candidate  →  evaluate  →  reflect on ASI  →  propose  →  repeat
                          ↑                                  |
                          └──────────────────────────────────┘

Three optimization modes
------------------------
``optimize_anything`` unifies three optimization paradigms under one
declarative API, determined by whether you provide ``dataset`` and ``valset``:

1. **Single-Task Search** (``dataset=None, valset=None``):
   Solve one hard problem.  The candidate *is* the solution.
   Evaluator is called without ``example``.
   *E.g. circle packing, blackbox mathematical optimization.*

2. **Multi-Task Search** (``dataset=<list>, valset=None``):
   Solve a batch of related problems with cross-task transfer.  Insights
   from solving one help solve the others.  Evaluator is called per-example.
   *E.g. CUDA kernel generation for multiple PyTorch operations,
   multi-aspect SVG optimization.*

3. **Generalization** (``dataset=<list>, valset=<list>``):
   Build a skill that transfers to unseen problems.  Evaluator is called
   per-example; candidates must generalize to ``valset``.
   *E.g. prompt optimization for AIME math, agent architecture evolution
   for ARC-AGI, cloud scheduling policy discovery.*

Seedless mode
-------------
When you don't have a starting artifact, pass ``seed_candidate=None`` and
provide ``objective`` (and optionally ``background``).  The reflection LM
bootstraps the first candidate from the description, then iterates as usual.
Useful for creative or exploratory tasks where the solution space is large.

Quick example::

    import gepa.optimize_anything as oa
    from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

    def evaluate(candidate: str) -> float:
        score, diagnostic = run_candidate(candidate)
        oa.log("Diagnostic:", diagnostic)   # captured as ASI
        return score

    # Start from an existing artifact…
    result = optimize_anything(
        seed_candidate="<initial code>",
        evaluator=evaluate,
        objective="Maximize throughput.",
        config=GEPAConfig(engine=EngineConfig(max_metric_calls=300)),
    )

    # … or just describe what you need (seedless mode).
    result = optimize_anything(
        evaluator=evaluate,
        objective="Generate a Python function that reverses a string.",
    )

    print(result.best_candidate)

Key concepts:
    - **Candidate**: ``dict[str, str]`` or plain ``str`` of optimizable text
      parameters — prompts, code, configs, agent architectures, etc.
    - **Evaluator**: Your scoring function.  Returns ``(score, side_info)`` or
      just ``score``.  Higher scores are better.
    - **ASI / SideInfo**: Actionable Side Information — diagnostic feedback
      returned by the evaluator (or captured via ``oa.log()``).  ASI is the
      text-optimization analogue of the gradient: where gradients tell a
      numerical optimizer which direction to move, ASI tells the LLM proposer
      *why* a candidate failed and *how* to fix it.  Can include error
      messages, profiling traces, rendered images (via :class:`Image`), or
      any structured data that would help an expert diagnose failures.
    - **Pareto-efficient search**: Scores are tracked per-task and per-metric
      individually; any candidate that is the best at *something* survives on
      the frontier, enabling focused improvements that are preserved rather
      than averaged away.
    - **Objective / Background**: Natural-language strings that guide the
      reflection LLM (what to optimize for, domain constraints).

Public API:
    - :func:`optimize_anything` — main entry point
    - :data:`SideInfo` — type alias for evaluation diagnostics (ASI)
    - :class:`Evaluator` — evaluator protocol
    - :class:`OptimizationState` — context injected into evaluators
    - :class:`GEPAConfig`, :class:`EngineConfig`, :class:`ReflectionConfig`,
      :class:`RefinerConfig`, :class:`MergeConfig`, :class:`TrackingConfig`
    - :func:`log`, :func:`get_log_context`, :func:`set_log_context` — in-evaluator logging
    - :func:`make_litellm_lm` — convert a model name string to a callable
    - :class:`Image` — wrapper for including images in side_info (VLM reflection)
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
from gepa.image import Image  # noqa: F401 — re-exported for user convenience
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
"""Actionable Side Information (ASI) returned by the evaluator alongside each score.

ASI is the text-optimization analogue of the gradient.  Where gradients tell
a numerical optimizer which direction to move, ASI tells an LLM proposer
*why* a candidate failed and *how* to fix it.

Traditional optimizers know *that* a candidate failed but not *why*.  SideInfo
provides the *why* — error messages, expected vs actual output, profiling
traces, compiler diagnostics, rendered images — enabling the reflection LLM
to take targeted corrective action rather than random mutation.

**More informative SideInfo → better optimization.**

You can provide SideInfo in two ways:

1. Return ``(score, side_info_dict)`` from your evaluator.
2. Call ``oa.log(...)`` inside your evaluator (captured under ``"log"`` key).

Structure
---------
1. **``"scores"`` (optional)** — multi-objective metrics for Pareto tracking.
   All values must follow "higher is better" convention.

   ``{"scores": {"accuracy": 0.85, "latency_inv": 12.5}}``

2. **Contextual fields** — any other keys.  Common conventions:

   - ``"Input"`` / ``"Output"`` / ``"Expected"`` — what went in and came out
   - ``"Feedback"`` — qualitative assessment (human or machine)
   - ``"Error"`` — error messages, tracebacks, compilation failures
   - ``"Reasoning"`` — intermediate reasoning traces

3. **Parameter-specific info** — ``"<param_name>_specific_info"`` dicts with
   their own ``"scores"`` and contextual fields.  During reflection on
   parameter *X*, GEPA merges top-level fields with ``X_specific_info``.

4. **Images** — use :class:`~gepa.Image` for visual feedback (rendered
   SVGs, charts, screenshots).  Requires a VLM as ``reflection_lm``.

Example::

    {
        "scores": {"accuracy": 0.73, "user_satisfaction": 4.2},
        "Input": "Translate 'Hello world' to French",
        "Output": "Salut monde",
        "Expected": "Bonjour le monde",
        "Feedback": "Translation is too informal for the context",
        "system_prompt_specific_info": {
            "scores": {"tone": 0.3},
            "Analysis": "System prompt led to overly casual translation",
        },
    }

Best practices:
    - Include error messages and failure reasons prominently
    - Use consistent field names across evaluations
    - Add context beyond raw numbers (explain *what* went wrong)
    - Use ``"scores"`` only for "higher is better" metrics used in Pareto tracking
"""


@dataclass
class OptimizationState:
    """Accumulated optimization context injected into evaluators that declare an ``opt_state`` parameter.

    Provides historical evaluation results so your evaluator can warm-start
    from previous best solutions (e.g., pass the best-known circle packing to
    a new optimization attempt).

    To receive this, simply add ``opt_state: OptimizationState`` to your
    evaluator signature — GEPA injects it automatically.

    Example::

        def evaluator(candidate, example, opt_state: OptimizationState):
            prev_best = opt_state.best_example_evals[0]["side_info"] if opt_state.best_example_evals else None
            # ... use prev_best to warm-start ...
    """

    best_example_evals: list[dict]
    """Top-K best evaluations for the current example, sorted by score (descending).

    Each entry: ``{"score": float, "side_info": dict}``.  K is controlled by
    ``EngineConfig.best_example_evals_k`` (default 30)."""


# ---------------------------------------------------------------------------
# Evaluation log context — captures diagnostic output without polluting stdout
# ---------------------------------------------------------------------------


class LogContext:
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

    def drain(self) -> str:
        """Drain and return all accumulated text, leaving the buffer empty."""
        with self._lock:
            old = self._buffer
            text = old.getvalue()
            old.close()
            self._buffer = io.StringIO()
            return text


# Thread-local storage for the active LogContext on each thread.
_log_tls = threading.local()


def _get_log_context() -> "LogContext | None":
    """Return the active log context for the current thread, or None."""
    return getattr(_log_tls, "context", None)


def _set_log_context(ctx: "LogContext | None") -> None:
    """Set (or clear) the active log context on the current thread."""
    _log_tls.context = ctx


def get_log_context() -> LogContext:
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


def set_log_context(ctx: LogContext) -> None:
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
        """Score a candidate, returning a score and diagnostic side information (ASI).

        This is the function you write.  GEPA calls it repeatedly with
        mutated candidates and collects the returned scores and diagnostics
        to drive the optimization loop.

        Args:
            candidate: The text parameter(s) to evaluate.  Type matches
                ``seed_candidate``: plain ``str`` if you passed a string,
                or ``dict[str, str]`` if you passed a dict.
            example: One item from ``dataset``.  **Not passed** in
                single-task search mode (``dataset=None``).
            opt_state: *(optional)* Declare ``opt_state: OptimizationState``
                in your signature to receive historical best evaluations
                for warm-starting.  This is a **reserved parameter name**
                — GEPA injects it automatically.

        Returns:
            Either ``(score, side_info)`` or just ``score``:

            - **score** (float): Higher is better.
            - **side_info** (:data:`SideInfo`): Diagnostic dict (ASI)
              powering LLM reflection.

            If you return score-only, use ``oa.log()`` or
            ``capture_stdio=True`` to provide diagnostic context.

        Evaluator signature per mode:

        **Single-Task Search** (``dataset=None``) — called without ``example``::

            import gepa.optimize_anything as oa

            def evaluate(candidate):
                result = run_code(candidate["code"])
                oa.log(f"Output: {result}")      # ASI via oa.log()
                return compute_score(result)

        **Multi-Task Search / Generalization** (``dataset`` provided) — called per example::

            def evaluate(candidate, example):
                pred = run(candidate["prompt"], example["input"])
                score = 1.0 if pred == example["expected"] else 0.0
                return score, {"Input": example["input"], "Output": pred}

        Reserved side_info keys:
            ``"log"``, ``"stdout"``, ``"stderr"`` are used by GEPA's
            automatic capture.  If you use them, GEPA stores captured
            output under ``"_gepa_log"`` etc. to avoid collisions.
        """
        ...


# --- Component 1: Engine & Stopping Configuration ---
@dataclass
class EngineConfig:
    """Controls the optimization run loop: budget, parallelism, caching, and stopping.

    Most users only need to set ``max_metric_calls`` (evaluation budget) and
    optionally ``parallel``/``max_workers`` for concurrent evaluation.

    Set ``capture_stdio=True`` to automatically route any ``print()`` output
    inside your evaluator into ASI (under ``"stdout"``/``"stderr"`` keys),
    with no code changes needed.  Useful for quick prototyping or wrapping
    existing evaluation scripts that already have print statements.
    """

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
    frontier_type: FrontierType = "hybrid"

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


def _build_seed_generation_prompt(
    objective: str,
    background: str | None = None,
    dataset: list[DataInst] | None = None,
) -> str:
    """Build a prompt for the reflection LM to generate an initial seed candidate.

    Used when ``seed_candidate=None`` — the LLM bootstraps the first candidate
    from the objective, optional background, and optional dataset examples.
    """
    sections = []

    sections.append(
        "You are an expert assistant. Your task is to generate an initial candidate "
        "that will be iteratively refined by an optimization system."
    )

    sections.append(f"\n## Goal\n\n{objective}")

    if background:
        sections.append(f"\n## Domain Context & Constraints\n\n{background}")

    if dataset is not None:
        examples = dataset[:3]
        example_lines = [f"- Example {i}: {ex}" for i, ex in enumerate(examples, 1)]
        sections.append(
            "\n## Sample Inputs\n\n"
            "The candidate will be evaluated on inputs like these:\n\n"
            + "\n".join(example_lines)
        )

    sections.append(
        "\n## Output Format\n\n"
        "Generate a strong initial candidate based on the goal above.\n"
        "Provide ONLY the candidate within ``` blocks. "
        "Do not include explanations or commentary outside the ``` blocks."
    )

    return "\n".join(sections)


def _generate_seed_candidate(
    lm: LanguageModel,
    objective: str,
    background: str | None = None,
    dataset: list[DataInst] | None = None,
    logger: LoggerProtocol | None = None,
) -> Candidate:
    """Call the reflection LM to generate an initial seed candidate.

    Returns a single-key candidate dict ``{_STR_CANDIDATE_KEY: generated_text}``.
    """
    from gepa.strategies.instruction_proposal import InstructionProposalSignature

    prompt = _build_seed_generation_prompt(
        objective=objective,
        background=background,
        dataset=dataset,
    )

    if logger:
        logger.log("Generating initial seed candidate via LLM...")

    lm_output = lm(prompt)
    extracted = InstructionProposalSignature.output_extractor(lm_output)
    generated_text = extracted["new_instruction"]

    if logger:
        logger.log(f"Generated seed candidate ({len(generated_text)} chars)")

    return {_STR_CANDIDATE_KEY: generated_text}


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
@dataclass
class ReflectionConfig:
    """Controls how the LLM proposes improved candidates each iteration.

    The reflection LM sees evaluation feedback (side_info) for a minibatch of
    examples and proposes an improved candidate.  ``reflection_lm`` is the
    model used for this step (defaults to ``openai/gpt-5.1``).

    ``reflection_minibatch_size`` controls how many examples are shown per
    reflection step (default: 1 for single-task, 3 otherwise).  Showing a
    small minibatch rather than all examples at once produces focused,
    targeted improvements on that subset.  Over iterations, all examples get
    attention, and the Pareto frontier preserves specialized gains across
    iterations rather than averaging them away.
    """

    skip_perfect_score: bool = False
    perfect_score: float | None = None
    batch_sampler: BatchSampler | Literal["epoch_shuffled"] = "epoch_shuffled"
    reflection_minibatch_size: int | None = None  # Default: 1 for single-instance mode, 3 otherwise
    module_selector: ReflectionComponentSelector | Literal["round_robin", "all"] = "round_robin"
    reflection_lm: LanguageModel | str | None = "openai/gpt-5.1"
    reflection_prompt_template: str | dict[str, str] | None = optimize_anything_reflection_prompt_template
    custom_candidate_proposer: ProposalFn | None = None


@dataclass
class MergeConfig:
    """Enables cross-pollination between candidates on the Pareto frontier.

    When set, GEPA periodically attempts to merge strengths of two candidates
    that each excel on different subsets of the validation set.
    """

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
    """Automatic per-evaluation candidate refinement via LLM.

    When enabled, after each evaluation GEPA calls an LLM to propose a refined
    version of the candidate based on the evaluation feedback.  The refined
    candidate is re-evaluated, and the better of (original, refined) is kept.

    A ``refiner_prompt`` parameter is auto-injected into seed candidates and
    co-evolved alongside the other parameters.  All non-refiner params are
    refined together as a JSON dict.

    Set ``config.refiner = None`` to disable refinement.
    """

    # Language model for refinement (defaults to reflection_lm if not specified)
    refiner_lm: LanguageModel | str | None = None

    # Maximum refinement iterations per evaluation
    max_refinements: int = 1


# --- Component 3: Experiment Tracking Configuration ---
@dataclass
class TrackingConfig:
    """Experiment tracking and logging (W&B, MLflow, or custom logger)."""

    logger: LoggerProtocol | None = None
    use_wandb: bool = False
    wandb_api_key: str | None = None
    wandb_init_kwargs: dict[str, Any] | None = None
    use_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None


@dataclass
class GEPAConfig:
    """Top-level configuration for :func:`optimize_anything`.

    Groups all settings into nested component configs.  Sensible defaults are
    provided — most users only need to set ``engine.max_metric_calls`` and
    optionally ``reflection.reflection_lm``.

    Example::

        config = GEPAConfig(
            engine=EngineConfig(max_metric_calls=200, parallel=True, max_workers=16),
            reflection=ReflectionConfig(reflection_lm="openai/gpt-5.1"),
            refiner=RefinerConfig(max_refinements=2),
        )
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
    accepts a plain ``str`` prompt, a ``list[dict]`` chat-messages list, or
    a multimodal messages list (with content arrays containing images).
    """
    import litellm

    def _lm(prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        completion = litellm.completion(model=model_name, messages=messages)
        return completion.choices[0].message.content  # type: ignore[union-attr]

    return _lm


class EvaluatorWrapper:
    """Internal wrapper that adapts a user's evaluator to GEPA's internal interface.

    Handles: single-instance mode (omits ``example``), str-candidate unwrapping,
    kwarg filtering (incl. ``opt_state`` injection), ``oa.log()`` capture,
    optional stdout/stderr capture, and normalizing the return value to
    ``(score, output, side_info)`` regardless of what the user returns.
    """

    def __init__(
        self,
        evaluator_fn: Callable[..., Any],
        single_instance_mode: bool,
        capture_stdio: bool = False,
        str_candidate_mode: bool = False,
        raise_on_exception: bool = True,
    ) -> None:
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
            # The same LogContext is accessible from child threads via
            # oa.get_log_context() / oa.set_log_context().
            log_ctx = LogContext()
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
            # Both acquire/start_capture and the evaluator call are inside the
            # same try/finally so that stream_manager.release() is always called
            # even if start_capture() raises (e.g. assertion on double-capture).
            stdout_capturer: ThreadLocalStreamCapture | None = None
            stderr_capturer: ThreadLocalStreamCapture | None = None
            try:
                if capture_stdio:
                    stdout_capturer, stderr_capturer = stream_manager.acquire()
                    stdout_capturer.start_capture()
                    stderr_capturer.start_capture()

                result = evaluator_fn(eval_candidate, **filtered)
            except Exception as e:
                result = e  # Sentinel; handled below after cleanup
            finally:
                captured_stdout = stdout_capturer.stop_capture() if stdout_capturer else ""
                captured_stderr = stderr_capturer.stop_capture() if stderr_capturer else ""
                if capture_stdio and stdout_capturer is not None:
                    stream_manager.release()
                log_output = log_ctx.drain()
                _set_log_context(None)

            # If evaluator raised, preserve captured diagnostics
            if isinstance(result, Exception):
                if raise_on_exception:
                    raise result
                fail_side_info: SideInfo = {"error": str(result)}
                if log_output:
                    fail_side_info["log"] = log_output
                if captured_stdout:
                    fail_side_info["stdout"] = captured_stdout
                if captured_stderr:
                    fail_side_info["stderr"] = captured_stderr
                return 0.0, None, fail_side_info

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
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Callable[..., Any],
    dataset: list[DataInst] | None = None,
    valset: list[DataInst] | None = None,
    objective: str | None = None,
    background: str | None = None,
    config: GEPAConfig | None = None,
) -> GEPAResult:
    """Optimize any text artifact using LLM-guided search.

    This is the main entry point for GEPA.  You declare the **what** — your
    artifact, your evaluator, and any domain knowledge — and
    ``optimize_anything`` handles the **how**: prompt construction, reflection,
    candidate selection, and Pareto-efficient search.

    **Three optimization modes** (determined by ``dataset`` / ``valset``):

    1. **Single-Task Search** (``dataset=None, valset=None``):
       Solve one hard problem.  The candidate *is* the solution.
       Evaluator called without ``example``.
       *E.g. circle packing, blackbox mathematical optimization.*

    2. **Multi-Task Search** (``dataset=<list>, valset=None``):
       Solve a batch of related problems with cross-task transfer.
       Insights from solving one help solve the others.
       ``valset`` defaults to ``dataset``.
       *E.g. CUDA kernel generation, multi-aspect SVG optimization.*

    3. **Generalization** (``dataset=<list>, valset=<list>``):
       Build a skill that transfers to unseen problems.
       *E.g. prompt optimization for AIME math, agent architecture evolution
       for ARC-AGI, cloud scheduling policy discovery.*

    Args:
        seed_candidate: Starting point for optimization.

            - ``str`` — single text parameter (evaluator receives ``str``).
            - ``dict[str, str]`` — named parameters (evaluator receives the dict).
            - ``None`` — **seedless mode**: the reflection LLM generates the
              initial candidate from ``objective`` (and optionally ``background``
              / ``dataset``).  Requires ``objective``.  Useful for creative or
              exploratory tasks where you know *what good looks like* but not
              where to begin.

        evaluator: Scoring function.  Returns ``(score, side_info)`` or ``score``.
            See :class:`Evaluator`.  Diagnostic output via ``oa.log()`` is
            automatically captured as Actionable Side Information (ASI).
            For richer diagnostics, return a ``(score, dict)`` tuple with
            structured feedback, error messages, or even rendered images
            (via :class:`~gepa.Image`).
        dataset: Examples for multi-task or generalization modes.
            ``None`` = single-task search mode.
        valset: Held-out validation set for generalization mode.
            ``None`` = defaults to ``dataset`` (multi-task search).
        objective: Natural-language goal for the reflection LLM (e.g.
            ``"Generate prompts that solve competition math problems."``).
        background: Domain knowledge, constraints, or strategies for the
            reflection LLM.
        config: Full configuration.  See :class:`GEPAConfig`.

    Returns:
        :class:`~gepa.core.result.GEPAResult` — access ``result.best_candidate``
        for the optimized parameter(s) and the full optimization history.

    Examples:

        Single-task search (circle packing)::

            import gepa.optimize_anything as oa

            def evaluate(candidate: str) -> float:
                result = run_code(candidate)
                oa.log(f"Score: {result.score}, Overlaps: {result.overlaps}")
                return result.score

            result = optimize_anything(
                seed_candidate="def pack_circles(): ...",
                evaluator=evaluate,
                objective="Maximize the sum of radii for n circles in a unit square.",
                config=GEPAConfig(engine=EngineConfig(max_metric_calls=500)),
            )

        Multi-task search (CUDA kernels)::

            result = optimize_anything(
                seed_candidate={"prompt": "Write an optimized CUDA kernel."},
                evaluator=kernel_evaluator,
                dataset=kernel_problems,       # batch of related problems
                objective="Generate prompts that produce fast, correct CUDA kernels.",
                config=GEPAConfig(engine=EngineConfig(max_metric_calls=300)),
            )

        Generalization (prompt optimization for math)::

            result = optimize_anything(
                seed_candidate={"prompt": "Solve this math problem step by step:"},
                evaluator=math_evaluator,
                dataset=train_problems,        # train on these
                valset=val_problems,           # must generalize to these
                objective="Generate system prompts that improve math reasoning.",
                config=GEPAConfig(engine=EngineConfig(max_metric_calls=200)),
            )

        Seedless mode (no starting artifact)::

            result = optimize_anything(
                seed_candidate=None,           # LLM writes the first draft
                evaluator=evaluate_3d_render,
                dataset=visual_aspects,
                objective="Optimize a Python program to generate a 3D unicorn.",
                background="Use build123d for CSG geometry, export to STL, render with pyrender.",
            )
    """
    # Use default config if not provided
    if config is None:
        config = GEPAConfig()

    # Detect seed generation mode: when seed_candidate is None, the LLM
    # will generate the initial candidate from the objective.
    needs_seed_generation = False
    if seed_candidate is None:
        needs_seed_generation = True
        str_candidate_mode = True
        if not objective or not objective.strip():
            raise ValueError(
                "'objective' is required when seed_candidate is None. "
                "The reflection LLM needs the objective to generate an initial candidate."
            )
        seed_candidate = {_STR_CANDIDATE_KEY: ""}  # placeholder until LLM generates it
    else:
        # Normalize seed_candidate: str -> {_STR_CANDIDATE_KEY: str}
        str_candidate_mode = isinstance(seed_candidate, str)
        if isinstance(seed_candidate, str):
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
        raise_on_exception=config.engine.raise_on_exception,
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
        from gepa.utils import MaxCandidateProposalsStopper

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
    if needs_seed_generation and config.reflection.reflection_lm is None:
        raise ValueError(
            "reflection_lm is required when seed_candidate is None. "
            "Set config.reflection.reflection_lm to a model name or callable."
        )
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

    # Generate seed candidate via LLM if seed_candidate was None
    if needs_seed_generation:
        assert config.reflection.reflection_lm is not None and not isinstance(config.reflection.reflection_lm, str)
        assert objective is not None  # validated earlier in needs_seed_generation block
        seed_candidate = _generate_seed_candidate(
            lm=config.reflection.reflection_lm,
            objective=objective,
            background=background,
            dataset=dataset,
            logger=config.tracking.logger or StdOutLogger(),
        )

    # Auto-inject refiner_prompt into seed_candidate if refiner is enabled
    if config.refiner is not None:
        formatted_refiner_prompt = DEFAULT_REFINER_PROMPT.format(
            objective=objective or "Maximize the score",
            background=background or "No additional background provided.",
        )
        if "refiner_prompt" not in seed_candidate:
            seed_candidate["refiner_prompt"] = formatted_refiner_prompt
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
        seed_candidate=seed_candidate,
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

    return GEPAResult.from_state(
        state,
        run_dir=config.engine.run_dir,
        seed=config.engine.seed,
        str_candidate_key=_STR_CANDIDATE_KEY if str_candidate_mode else None,
    )
