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
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import dspy
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Prediction

from turbo_gepa.archive import Archive
from turbo_gepa.cache import DiskCache
from turbo_gepa.config import Config, DEFAULT_CONFIG
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate
from turbo_gepa.logging_utils import EventLogger, build_logger
from turbo_gepa.mutator import MutationConfig, Mutator
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.sampler import InstanceSampler
from turbo_gepa.token_controller import TokenCostController


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
        feedback_map: Optional[Dict[str, Callable]] = None,
        failure_score: float = 0.0,
        num_threads: Optional[int] = None,
        config: Config = DEFAULT_CONFIG,
        cache_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        sampler_seed: int = 42,
        rng: Optional[random.Random] = None,
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
        """
        self.student = student_module
        self.metric_fn = metric_fn
        self.feedback_map = feedback_map or {}
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.config = config
        self.rng = rng or random.Random(0)

        # Cache predictor names
        self.named_predictors = list(self.student.named_predictors())

        # Create dataset mapping
        self.trainset = list(trainset)
        self.example_map = {f"example-{idx}": ex for idx, ex in enumerate(self.trainset)}

        # Initialize TurboGEPA components
        self.sampler = InstanceSampler(list(self.example_map.keys()), seed=sampler_seed)
        self.cache = DiskCache(cache_dir or config.cache_path)
        self.archive = Archive(
            bins_length=config.qd_bins_length,
            bins_bullets=config.qd_bins_bullets,
            flags=config.qd_flags,
        )
        self.token_controller = TokenCostController(config.max_tokens)
        self.log_dir = log_dir or config.log_path

        # Reflection runner
        self.reflection_runner = self._create_reflection_runner()

        # Mutator
        self.mutator = Mutator(
            MutationConfig(
                amortized_rate=config.amortized_rate,
                reflection_batch_size=config.reflection_batch_size,
                max_mutations=config.max_mutations_per_round,
                max_tokens=config.max_tokens,
            ),
            reflection_runner=self.reflection_runner,
        )

    def build_program(self, instructions: Dict[str, str]) -> dspy.Module:
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
    ) -> Dict[str, float]:
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
                        num_threads=1,
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

    async def _task_runner(self, candidate: Candidate, example_id: str) -> Dict[str, float]:
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

        # Evaluate
        result = await self._evaluate_single(program, example, example_id, capture_traces=False)

        # Add token cost (approximate from instruction length)
        tokens = sum(len(inst.split()) for inst in instructions.values())
        result["tokens"] = float(tokens)
        result["neg_cost"] = -float(tokens)
        result["example_id"] = example_id

        return result

    def _create_reflection_runner(self):
        """Create async reflection runner for DSPy feedback."""

        async def reflect(traces: List[Dict], parent_prompt: str) -> List[str]:
            """
            Generate improved instructions based on feedback.

            If feedback_map is provided and reflection_lm is set, uses LLM-based
            reflection. Otherwise falls back to rule-based mutations.
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

            # Check if we can use LLM-based reflection
            has_feedback_map = bool(self.feedback_map)
            has_reflection_lm = hasattr(self, "reflection_lm") and self.reflection_lm is not None

            if has_feedback_map and has_reflection_lm:
                # LLM-based reflection with feedback
                try:
                    mutations = await self._llm_reflection(
                        traces, current_instructions, parent_prompt
                    )
                    if mutations:
                        return mutations
                except Exception as e:
                    logging.warning(f"LLM reflection failed: {e}, falling back to rule-based")

            # Fallback: Rule-based mutations
            return self._rule_based_mutations(traces, current_instructions)

        return reflect

    def _rule_based_mutations(
        self, traces: List[Dict], current_instructions: Dict[str, str]
    ) -> List[str]:
        """Generate rule-based mutations (fallback)."""
        import json

        failures = [t for t in traces if t.get("quality", 0) < 0.5]
        if not failures:
            return []

        mutations = []
        for pred_name, inst in current_instructions.items():
            mutations.append(
                json.dumps({**current_instructions, pred_name: inst + " Be more precise."})
            )
            mutations.append(
                json.dumps(
                    {
                        **current_instructions,
                        pred_name: inst + " Show your reasoning step by step.",
                    }
                )
            )

        return mutations[:3]

    async def _llm_reflection(
        self,
        traces: List[Dict],
        current_instructions: Dict[str, str],
        parent_prompt: str,
    ) -> List[str]:
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

            # Call reflection LLM
            response = await self.reflection_lm(prompt)

            # Extract new instruction
            new_instruction = InstructionProposalPrompt.extract_instruction(response)

            # Create mutated candidate
            new_instructions = current_instructions.copy()
            new_instructions[pred_name] = new_instruction

            mutations.append(json.dumps(new_instructions))

        return mutations

    def _build_reflective_dataset(
        self, traces: List[Dict], current_instructions: Dict[str, str]
    ) -> Dict[str, List[Dict]]:
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
            quality = trace.get("quality", 0.0)

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
                    t for t in dspy_trace if hasattr(t[0], "signature") and
                    t[0].signature.equals(pred.signature)
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
                            "Inputs": {
                                k: str(v) for k, v in predictor_inputs.items()
                            },
                            "Generated Outputs": {
                                k: str(v) for k, v in predictor_output.items()
                            } if hasattr(predictor_output, "items") else str(predictor_output),
                            "Feedback": fb.feedback if hasattr(fb, "feedback") else str(fb),
                        }
                    )
                except Exception as e:
                    logging.warning(f"Feedback function failed for {pred_name}: {e}")
                    continue

        return dataset

    def _build_orchestrator(self, logger: Optional[EventLogger]) -> Orchestrator:
        """Build TurboGEPA orchestrator."""
        evaluator = AsyncEvaluator(
            cache=self.cache,
            task_runner=self._task_runner,
        )
        orchestrator_logger = logger or build_logger(self.log_dir, "dspy_opt")
        return Orchestrator(
            config=self.config,
            evaluator=evaluator,
            archive=self.archive,
            sampler=self.sampler,
            mutator=self.mutator,
            cache=self.cache,
            logger=orchestrator_logger,
            token_controller=self.token_controller,
        )

    async def optimize_async(
        self,
        seed_instructions: Dict[str, str],
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        logger: Optional[EventLogger] = None,
        reflection_lm: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Optimize DSPy program instructions asynchronously.

        Args:
            seed_instructions: Initial instructions for each predictor
            max_rounds: Maximum optimization rounds
            max_evaluations: Maximum evaluations budget
            logger: Optional event logger
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

        orchestrator = self._build_orchestrator(logger)

        # Create seed candidate (JSON-encoded instructions)
        seed_text = json.dumps(seed_instructions)
        seed_candidate = Candidate(text=seed_text, meta={"source": "seed"})

        # Run optimization
        await orchestrator.run(
            [seed_candidate],
            max_rounds=max_rounds,
            max_evaluations=max_evaluations,
        )

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
            "qd_elites": orchestrator.archive.sample_qd(limit=len(pareto_entries)),
            "archive": orchestrator.archive,
            "orchestrator": orchestrator,
        }

    def optimize(
        self,
        seed_instructions: Dict[str, str],
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        logger: Optional[EventLogger] = None,
        reflection_lm: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Optimize DSPy program instructions (sync wrapper).

        Args:
            seed_instructions: Initial instructions for each predictor
            max_rounds: Maximum optimization rounds
            max_evaluations: Maximum evaluations budget
            logger: Optional event logger
            reflection_lm: Optional async LLM function for reflection

        Returns:
            Dict with 'best_program', 'best_instructions', 'pareto', 'archive'
        """
        return asyncio.run(
            self.optimize_async(
                seed_instructions,
                max_rounds=max_rounds,
                max_evaluations=max_evaluations,
                logger=logger,
                reflection_lm=reflection_lm,
            )
        )
