"""
DSPy adapter for TurboGEPA.

This adapter allows TurboGEPA to optimize DSPy programs by evolving their
instruction text. It provides async evaluation with trace capture for reflection.

Example usage::

    from turbo_gepa.adapters import DSpyAdapter
    import dspy

    # Create your DSPy module
    class MyModule(dspy.Module):
        def __init__(self):
            self.predictor = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.predictor(question=question)

    # Define metric
    def metric(example, prediction, trace=None):
        return example.answer.lower() == prediction.answer.lower()

    # Create adapter
    adapter = DSpyAdapter(
        student_module=MyModule(),
        metric_fn=metric,
        trainset=trainset,
    )

    # Optimize
    result = await adapter.optimize_async(
        seed_instructions={"predictor": "Solve the problem step by step."},
        max_rounds=10,
    )

    best = result['best_program']
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import dspy
from dspy.primitives import Example, Prediction

from turbo_gepa.archive import Archive
from turbo_gepa.cache import DiskCache
from turbo_gepa.config import DEFAULT_CONFIG, Config
from turbo_gepa.scoring import ScoringFn, SCORE_KEY
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate
from turbo_gepa.mutator import MutationConfig, Mutator
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.sampler import InstanceSampler
from turbo_gepa.utils.litellm_client import configure_litellm_client


DSPyTrace = list[tuple[Any, dict[str, Any], Prediction]]


@dataclass
class ScoreWithFeedback:
    """Feedback from predictor evaluation."""

    score: float
    feedback: str


class DSpyAdapter:
    """
    Adapter for optimizing DSPy programs with TurboGEPA.

    This adapter evolves the instruction text of DSPy predictors using
    async evaluation and reflection-based mutations.
    """

    def __init__(
        self,
        student_module: dspy.Module,
        metric_fn: Callable,
        trainset: Sequence[Example],
        *,
        feedback_map: dict[str, Callable] | None = None,
        failure_score: float = 0.0,
        num_threads: int | None = None,
        config: Config = DEFAULT_CONFIG,
        cache_dir: str | None = None,
        log_dir: str | None = None,
        sampler_seed: int = 42,
        rng: random.Random | None = None,
        scoring_fn: ScoringFn | None = None,
    ) -> None:
        """
        Initialize DSPy adapter.

        Args:
            student_module: DSPy module to optimize
            metric_fn: Evaluation metric (example, prediction, trace) -> float
            trainset: Training examples
            feedback_map: Optional map of predictor names to feedback functions
            failure_score: Score for failed predictions
            num_threads: Number of evaluation threads
            config: TurboGEPA configuration
            cache_dir: Cache directory (default: config.cache_path)
            log_dir: Log directory (default: config.log_path)
            sampler_seed: Random seed for sampler
            rng: Random number generator
            scoring_fn: Optional ``ScoringContext`` callback returning the scalar
                objective to optimize (defaults to ``maximize_metric(\"quality\")``)
        """
        self.student = student_module
        self.metric_fn = metric_fn
        self.feedback_map = feedback_map or {}
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.config = config
        self.config.promote_objective = SCORE_KEY
        if scoring_fn is not None:
            self.config.scoring_fn = scoring_fn
        self.rng = rng or random.Random(0)

        # Cache predictor names
        self.named_predictors = list(self.student.named_predictors())

        # Create dataset mapping
        self.trainset = list(trainset)
        self.example_map = {f"example-{idx}": ex for idx, ex in enumerate(self.trainset)}

        # Initialize TurboGEPA components
        self.sampler = InstanceSampler(list(self.example_map.keys()), seed=sampler_seed)
        env_cache = os.getenv("TURBOGEPA_CACHE_PATH")
        env_logs = os.getenv("TURBOGEPA_LOG_PATH")
        env_control = os.getenv("TURBOGEPA_CONTROL_PATH")
        self.base_cache_dir = os.path.abspath(cache_dir or env_cache or config.cache_path)
        self.base_log_dir = os.path.abspath(log_dir or env_logs or config.log_path)
        Path(self.base_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.base_log_dir).mkdir(parents=True, exist_ok=True)
        control_path = config.control_dir if config.control_dir is not None else env_control
        self.control_dir = os.path.abspath(control_path) if control_path else None
        if self.control_dir:
            Path(self.control_dir).mkdir(parents=True, exist_ok=True)
        namespace = self.config.shared_cache_namespace or "dspy"
        self.cache = DiskCache(self.base_cache_dir, namespace=namespace)
        self.archive = Archive()
        self.log_dir = self.base_log_dir

        # Configure litellm's httpx client based on eval_concurrency
        # This ensures connection pooling matches the user's concurrency settings
        configure_litellm_client(self.config.eval_concurrency)

        # Reflection runner
        self.reflection_runner = self._create_reflection_runner()
        self._async_random = random.Random()

        async def batch_reflection_runner(
            parent_contexts: list[dict[str, object]],
            num_mutations: int,
            _task_examples: list[dict[str, object]] | None = None,
        ) -> list[str]:
            if num_mutations <= 0 or not parent_contexts:
                return []
            proposals: list[str] = []
            for ctx in parent_contexts:
                candidate = ctx.get("candidate")
                if candidate is None:
                    continue
                failures = ctx.get("failures", []) or []
                traces: list[dict] = []
                for _example_id, trace_list in failures:
                    traces.extend(trace_list)
                try:
                    mutated = await self.reflection_runner(traces, candidate.text)
                except RuntimeError:
                    continue
                if not mutated:
                    continue
                for text in mutated:
                    proposals.append(text)
                    if len(proposals) >= num_mutations:
                        return proposals
            return proposals[:num_mutations]

        # Mutator
        self.mutator = Mutator(
            MutationConfig(
                reflection_batch_size=config.reflection_batch_size,
                max_mutations=config.max_mutations_per_round,
                max_tokens=config.max_tokens,
                objective_key=config.promote_objective,
            ),
            batch_reflection_runner=batch_reflection_runner,
            spec_induction_runner=None,
        )

    def build_program(self, instructions: dict[str, str]) -> dspy.Module:
        """Build a DSPy program with updated instructions."""
        new_prog = self.student.deepcopy()
        for name, pred in new_prog.named_predictors():
            if name in instructions:
                pred.signature = pred.signature.with_instructions(instructions[name])
        return new_prog

    async def _evaluate_single(
        self,
        program: dspy.Module,
        example: Example,
        example_id: str,
        capture_traces: bool = False,
    ) -> dict[str, float]:
        """Evaluate program on a single example (async)."""
        # Run in thread pool since DSPy evaluation is sync
        loop = asyncio.get_event_loop()

        def _eval():
            try:
                if capture_traces:
                    # Use bootstrap_trace_data for trace capture
                    from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data

                    trajs = bootstrap_trace_data(
                        program=program,
                        dataset=[example],
                        metric=self.metric_fn,
                        num_threads=self.num_threads or 1,
                        raise_on_error=False,
                        capture_failed_parses=True,
                        failure_score=self.failure_score,
                        format_failure_score=self.failure_score,
                    )
                    traj = trajs[0] if trajs else None
                    if traj:
                        score = traj.get("score", self.failure_score)
                        if hasattr(score, "score"):
                            score = score["score"]
                        return {
                            "quality": float(score),
                            "trace": traj.get("trace", []),
                            "prediction": traj.get("prediction"),
                        }
                else:
                    # Standard evaluation
                    prediction = program(**example.inputs())
                    score = self.metric_fn(example, prediction, trace=None)
                    if hasattr(score, "score"):
                        score = score["score"]
                    return {"quality": float(score), "prediction": prediction}

            except Exception as e:
                logging.warning(f"Evaluation failed for {example_id}: {e}")
                return {"quality": self.failure_score}

            return {"quality": self.failure_score}

        return await loop.run_in_executor(None, _eval)

    async def _task_runner(self, candidate: Candidate, example_id: str) -> dict[str, float]:
        """
        Task runner for TurboGEPA evaluator.

        Builds a program with the candidate instructions and evaluates it.
        """
        # Parse candidate text as instruction dictionary
        # Expected format: JSON or newline-separated "predictor_name: instruction"
        import json

        try:
            instructions = json.loads(candidate.text)
        except json.JSONDecodeError:
            # Fallback: assume single predictor
            if len(self.named_predictors) == 1:
                instructions = {self.named_predictors[0][0]: candidate.text}
            else:
                # Parse multi-line format
                instructions = {}
                for line in candidate.text.split("\n"):
                    if ":" in line:
                        name, inst = line.split(":", 1)
                        instructions[name.strip()] = inst.strip()

        # Build program
        program = self.build_program(instructions)

        # Get example
        example = self.example_map[example_id]

        # Evaluate (always capture traces for reflection)
        result = await self._evaluate_single(program, example, example_id, capture_traces=True)

        # Add token cost (approximate from instruction length)
        tokens = sum(len(inst.split()) for inst in instructions.values())
        result["tokens"] = float(tokens)
        result["neg_cost"] = -float(tokens)
        result["example_id"] = example_id

        # Add example inputs for reflection
        result["input"] = str(example.inputs())
        result["expected_answer"] = str(example.labels()) if hasattr(example, "labels") else ""

        return result

    def _create_reflection_runner(self):
        """Create async reflection runner for DSPy feedback.

        Requires both a feedback_map and a reflection_lm to be provided;
        no rule-based fallback is implemented.
        """

        async def reflect(traces: list[dict], parent_prompt: str) -> list[str]:
            """
            Generate improved instructions based on feedback using LLM-based reflection.
            """
            if not traces:
                return []

            # Parse current instructions from JSON
            try:
                import json

                current_instructions = json.loads(parent_prompt)
            except json.JSONDecodeError:
                # Single predictor
                if len(self.named_predictors) == 1:
                    current_instructions = {self.named_predictors[0][0]: parent_prompt}
                else:
                    current_instructions = {"default": parent_prompt}

            # Require feedback_map and reflection_lm
            has_feedback_map = bool(self.feedback_map)
            has_reflection_lm = hasattr(self, "reflection_lm") and self.reflection_lm is not None
            if not (has_feedback_map and has_reflection_lm):
                raise RuntimeError(
                    "DSpyAdapter requires both feedback_map and reflection_lm to perform reflection. "
                    "Provide a feedback_map for predictors and an async reflection_lm callable."
                )

            # LLM-based reflection with feedback
            mutations = await self._llm_reflection(traces, current_instructions, parent_prompt)
            return mutations or []

        return reflect

    # Removed rule-based fallback: reflection requires an LLM

    async def _call_with_retries(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        label: str,
        max_attempts: int = 3,
        base_delay: float = 1.5,
    ):
        """Retry helper for LLM calls."""
        import asyncio

        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await coro_factory()
            except Exception as exc:  # pragma: no cover - defensive path
                last_exc = exc
                if attempt >= max_attempts:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                delay += self._async_random.uniform(0.0, 0.5)
                logging.warning(
                    "LLM call '%s' failed (attempt %s/%s): %s. Retrying in %.1fs",
                    label,
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        if last_exc:
            raise last_exc

    async def _llm_reflection(
        self,
        traces: list[dict],
        current_instructions: dict[str, str],
        parent_prompt: str,
    ) -> list[str]:
        """
        Generate mutations using LLM-based reflection.

        This uses the InstructionProposalPrompt to generate improved instructions
        based on feedback from failed predictions.
        """
        import json

        from turbo_gepa.dspy_utils import InstructionProposalPrompt

        # Build reflective dataset per predictor
        reflective_dataset = self._build_reflective_dataset(traces, current_instructions)

        if not reflective_dataset:
            return []

        # Generate improved instructions for each predictor
        mutations = []

        for pred_name, dataset in reflective_dataset.items():
            if not dataset:
                continue

            current_inst = current_instructions.get(pred_name, "")

            # Build LLM prompt
            prompt = InstructionProposalPrompt.build_prompt(current_inst, dataset)

            # Call reflection LLM with retries
            async def invoke():
                return await self.reflection_lm(prompt)

            response = await self._call_with_retries(
                invoke,
                label=f"dspy_reflection:{pred_name}",
            )

            # Extract new instruction
            new_instruction = InstructionProposalPrompt.extract_instruction(response)

            # Create mutated candidate
            new_instructions = current_instructions.copy()
            new_instructions[pred_name] = new_instruction

            mutations.append(json.dumps(new_instructions))

        return mutations

    def _build_reflective_dataset(
        self, traces: list[dict], current_instructions: dict[str, str]
    ) -> dict[str, list[dict]]:
        """
        Build reflective dataset from traces using feedback functions.

        Returns:
            Dict mapping predictor names to lists of feedback samples
        """
        dataset = {pred_name: [] for pred_name in current_instructions.keys()}

        for trace in traces:
            # Extract trace data
            dspy_trace = trace.get("trace", [])
            prediction = trace.get("prediction")
            example_id = trace.get("example_id", "unknown")
            trace.get("quality", 0.0)

            if not dspy_trace or not prediction:
                continue

            # Get example
            example = self.example_map.get(example_id)
            if not example:
                continue

            # Process each predictor
            for pred_name, pred in self.named_predictors:
                if pred_name not in self.feedback_map:
                    continue

                # Find trace instances for this predictor
                trace_instances = [
                    t for t in dspy_trace if hasattr(t[0], "signature") and t[0].signature.equals(pred.signature)
                ]

                if not trace_instances:
                    continue

                # Select a representative trace (prefer failed ones)
                selected = trace_instances[0]
                for t in trace_instances:
                    # Check if this is a failed prediction
                    if hasattr(t[2], "__class__") and "FailedPrediction" in str(type(t[2])):
                        selected = t
                        break

                predictor_inputs = selected[1]
                predictor_output = selected[2]

                # Call user's feedback function
                try:
                    feedback_fn = self.feedback_map[pred_name]
                    fb = feedback_fn(
                        predictor_output=predictor_output,
                        predictor_inputs=predictor_inputs,
                        module_inputs=example,
                        module_outputs=prediction,
                        captured_trace=dspy_trace,
                    )

                    # Build dataset entry
                    dataset[pred_name].append(
                        {
                            "Inputs": {k: str(v) for k, v in predictor_inputs.items()},
                            "Generated Outputs": {k: str(v) for k, v in predictor_output.items()}
                            if hasattr(predictor_output, "items")
                            else str(predictor_output),
                            "Feedback": fb.feedback if hasattr(fb, "feedback") else str(fb),
                        }
                    )
                except Exception as e:
                    logging.warning(f"Feedback function failed for {pred_name}: {e}")
                    continue

        return dataset

    def _build_orchestrator(self, max_rounds: int = 100) -> Orchestrator:
        """Build TurboGEPA orchestrator."""
        objective = self.config.promote_objective or "quality"

        def metrics_mapper(metrics: dict[str, float]) -> dict[str, float]:
            value = metrics.get(objective)
            if value is None:
                value = metrics.get("quality", 0.0)
            mapped: dict[str, float] = {objective: value}
            if objective != "quality" and "quality" in metrics:
                mapped["quality"] = metrics.get("quality", 0.0)
            for extra in ("tokens", "neg_cost"):
                if extra in metrics:
                    mapped[extra] = metrics[extra]
            return mapped

        evaluator = AsyncEvaluator(
            cache=self.cache,
            task_runner=self._task_runner,
            metrics_mapper=metrics_mapper,
            timeout_seconds=self.config.eval_timeout_seconds,
            min_improve=self.config.eps_improve,
            skip_final_straggler_cutoff=self.config.skip_final_straggler_cutoff,
            promote_objective=objective,
            cancel_stragglers_immediately=self.config.cancel_stragglers_immediately,
            replay_stragglers=self.config.replay_stragglers,
            min_samples_for_confidence=self.config.min_samples_for_confidence,
            target_quality=self.config.target_quality,
            confidence_z=self.config.confidence_z,
        )
        return Orchestrator(
            config=self.config,
            evaluator=evaluator,
            archive=self.archive,
            sampler=self.sampler,
            mutator=self.mutator,
            cache=self.cache,
            control_dir=self.control_dir,
        )

    async def optimize_async(
        self,
        seed_instructions: dict[str, str],
        *,
        max_rounds: int | None = None,
        max_evaluations: int | None = None,
        reflection_lm: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Optimize DSPy program instructions asynchronously.

        Args:
            seed_instructions: Initial instructions for each predictor
            max_rounds: Maximum optimization rounds
            max_evaluations: Maximum evaluations budget
            reflection_lm: Optional async LLM function for reflection
                          (prompt: str) -> str. If provided with feedback_map,
                          enables LLM-based instruction proposal.

        Returns:
            Dict with 'best_program', 'best_instructions', 'pareto', 'archive'
        """
        # Set reflection LLM if provided
        if reflection_lm is not None:
            self.reflection_lm = reflection_lm
        import json

        orchestrator = self._build_orchestrator(max_rounds=max_rounds or 100)

        # Create seed candidate (JSON-encoded instructions)
        seed_text = json.dumps(seed_instructions)
        seed_candidate = Candidate(text=seed_text, meta={"source": "seed"})

        # Run optimization
        try:
            await orchestrator.run(
                [seed_candidate],
                max_rounds=max_rounds,
                max_evaluations=max_evaluations,
            )
        finally:
            orchestrator.finalize_control()

        # Get best result
        pareto_entries = orchestrator.archive.pareto_entries()
        if not pareto_entries:
            raise RuntimeError("No candidates in Pareto frontier")

        best_entry = max(
            pareto_entries,
            key=lambda e: e.result.objectives.get("quality", 0.0),
        )

        # Parse best instructions
        best_instructions = json.loads(best_entry.candidate.text)

        # Build best program
        best_program = self.build_program(best_instructions)

        return {
            "best_program": best_program,
            "best_instructions": best_instructions,
            "best_quality": best_entry.result.objectives.get("quality", 0),
            "pareto": orchestrator.archive.pareto_candidates(),
            "pareto_entries": pareto_entries,
            "qd_elites": [],  # Deprecated
            "archive": orchestrator.archive,
            "orchestrator": orchestrator,
        }

    def optimize(
        self,
        seed_instructions: dict[str, str],
        *,
        max_rounds: int | None = None,
        max_evaluations: int | None = None,
        reflection_lm: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Optimize DSPy program instructions (sync wrapper).

        Args:
            seed_instructions: Initial instructions for each predictor
            max_rounds: Maximum optimization rounds
            max_evaluations: Maximum evaluations budget
            reflection_lm: Optional async LLM function for reflection

        Returns:
            Dict with 'best_program', 'best_instructions', 'pareto', 'archive'
        """
        return asyncio.run(
            self.optimize_async(
                seed_instructions,
                max_rounds=max_rounds,
                max_evaluations=max_evaluations,
                reflection_lm=reflection_lm,
            )
        )
