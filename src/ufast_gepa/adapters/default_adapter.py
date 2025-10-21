"""
Drop-in default adapter for uFast-GEPA mirroring the classic GEPA prompt flow.

This adapter expects data instances with `input`, `additional_context`, and
`answer` fields (matching `gepa.adapters.default_adapter`). Users supply their
own `task_lm_call` / `reflect_lm_call` implementations via
``ufast_gepa.user_plugs_in``. The adapter wires those hooks into the
high-throughput orchestrator so existing GEPA workflows can migrate with minimal
changes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ufast_gepa.archive import Archive
from ufast_gepa.cache import DiskCache
from ufast_gepa.config import Config, DEFAULT_CONFIG
from ufast_gepa.evaluator import AsyncEvaluator
from ufast_gepa.interfaces import Candidate
from ufast_gepa.logging_utils import EventLogger, build_logger
from ufast_gepa.mutator import MutationConfig, Mutator
from ufast_gepa.orchestrator import Orchestrator
from ufast_gepa.sampler import InstanceSampler
from ufast_gepa.token_controller import TokenCostController
from ufast_gepa.user_plugs_in import reflect_lm_call, task_lm_call


@dataclass(slots=True)
class DefaultDataInst:
    """Minimal data instance for prompt-based tasks."""

    input: str
    answer: str
    additional_context: Dict[str, str] | None = None
    id: str | None = None
    difficulty: float | None = None

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "input": self.input,
            "answer": self.answer,
        }
        if self.difficulty is not None:
            payload["difficulty"] = self.difficulty
        if self.additional_context is not None:
            payload["additional_context"] = self.additional_context
        return payload


class DefaultAdapter:
    """
    Helper harness for running uFast-GEPA on single-component prompts.

    Example usage::

        adapter = DefaultAdapter(dataset=trainset, reflection_batch_size=6)
        result = adapter.optimize(
            seeds=["You are a helpful assistant."],
            max_rounds=4,
            config=DEFAULT_CONFIG,
        )
        pareto = result["pareto"]
    """

    def __init__(
        self,
        dataset: Sequence[DefaultDataInst],
        *,
        config: Config = DEFAULT_CONFIG,
        mutation_config: MutationConfig | None = None,
        cache_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        sampler_seed: int = 42,
        task_lm: Optional[str] = None,
        reflection_lm: Optional[str] = None,
        task_lm_temperature: Optional[float] = None,
        reflection_lm_temperature: Optional[float] = None,
    ) -> None:
        if not dataset:
            raise ValueError("dataset must contain at least one data instance")
        self.config = config
        self.dataset = list(dataset)
        self.task_lm = task_lm
        self.reflection_lm = reflection_lm
        # Use provided temperatures or fall back to config defaults
        self.task_lm_temperature = task_lm_temperature if task_lm_temperature is not None else config.task_lm_temperature
        self.reflection_lm_temperature = reflection_lm_temperature if reflection_lm_temperature is not None else config.reflection_lm_temperature
        self.example_map = {
            data.id if data.id is not None else f"example-{idx}": data for idx, data in enumerate(self.dataset)
        }
        self.sampler = InstanceSampler(list(self.example_map.keys()), seed=sampler_seed)
        self.cache = DiskCache(cache_dir or config.cache_path)
        self.archive = Archive(
            bins_length=config.qd_bins_length,
            bins_bullets=config.qd_bins_bullets,
            flags=config.qd_flags,
        )
        self.logger: EventLogger | None = None

        # Create reflection runner based on whether reflection_lm is provided
        if reflection_lm:
            reflection_runner = self._create_llm_reflection_runner(reflection_lm)
        else:
            reflection_runner = reflect_lm_call  # Use heuristic default

        self.mutator = Mutator(
            mutation_config
            or MutationConfig(
                amortized_rate=config.amortized_rate,
                reflection_batch_size=config.reflection_batch_size,
                max_mutations=config.max_mutations_per_round,
                max_tokens=config.max_tokens,
            ),
            reflection_runner=reflection_runner,
        )
        self.token_controller = TokenCostController(config.max_tokens)
        self.log_dir = log_dir or config.log_path

    def _create_llm_reflection_runner(self, reflection_lm: str):
        """Create a reflection runner that calls a real LLM."""
        async def llm_reflection_runner(traces: list, parent_prompt: str) -> list[str]:
            if not traces:
                return []

            try:
                import openai
                import os

                client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                # Build reflection prompt
                failure_summary = "\n".join([
                    f"- Example {i+1}: quality={t.get('quality', 0):.2f}"
                    for i, t in enumerate(traces[:3])
                ])

                reflection_prompt = f"""You are optimizing a prompt for better performance.

Current prompt:
"{parent_prompt}"

Recent performance on examples:
{failure_summary}

Suggest 2-3 improved versions of this prompt that would perform better. Focus on:
1. Adding helpful instructions or constraints
2. Improving clarity and structure
3. Addressing common failure patterns

Output format: Return each improved prompt on a new line, separated by "---"."""

                response = await client.chat.completions.create(
                    model=reflection_lm.replace("openai/", ""),
                    messages=[{"role": "user", "content": reflection_prompt}],
                    temperature=0.7,
                )

                content = response.choices[0].message.content
                # Split by --- and clean up
                mutations = [m.strip() for m in content.split("---") if m.strip()]
                return mutations[:3]  # Limit to 3 mutations

            except Exception as e:
                print(f"Warning: LLM reflection failed ({e}), using heuristic fallback")
                return await reflect_lm_call(traces, parent_prompt)

        return llm_reflection_runner

    async def _task_runner(self, candidate: Candidate, example_id: str) -> Dict[str, float]:
        example = self.example_map[example_id].to_payload()

        # Use real LLM if task_lm is provided, otherwise use heuristic
        if self.task_lm:
            try:
                import openai
                import os

                client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                response = await client.chat.completions.create(
                    model=self.task_lm.replace("openai/", ""),
                    messages=[
                        {"role": "system", "content": candidate.text},
                        {"role": "user", "content": example["input"]},
                    ],
                )

                model_output = response.choices[0].message.content
                tokens_used = response.usage.total_tokens

                # Check if answer is in output
                quality = 1.0 if example["answer"] in model_output else 0.0

                metrics = {
                    "quality": quality,
                    "neg_cost": -float(tokens_used),
                    "tokens": float(tokens_used),
                    "response": model_output,
                }
            except Exception as e:
                print(f"Warning: LLM call failed ({e}), using heuristic")
                metrics = await task_lm_call(candidate.text, example)
        else:
            # Use heuristic evaluation
            metrics = await task_lm_call(candidate.text, example)

        if "tokens" not in metrics:
            metrics["tokens"] = float(len(candidate.text.split()))
        metrics.setdefault("neg_cost", -float(metrics["tokens"]))
        metrics.setdefault("quality", 0.0)
        metrics["example_id"] = example_id
        metrics.setdefault("output", metrics.get("response"))
        return metrics

    def _build_orchestrator(self, logger: Optional[EventLogger]) -> Orchestrator:
        evaluator = AsyncEvaluator(
            cache=self.cache,
            task_runner=self._task_runner,
        )
        orchestrator_logger = logger or build_logger(self.log_dir, "ufast_default")
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
        seeds: Sequence[str],
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        logger: Optional[EventLogger] = None,
        task_lm: Optional[str] = None,  # Accept but ignore (uses heuristic)
        reflection_lm: Optional[str] = None,  # Accept but ignore (uses heuristic)
    ) -> Dict[str, Any]:
        orchestrator = self._build_orchestrator(logger)
        seed_candidates = [Candidate(text=seed, meta={"source": "seed"}) for seed in seeds]
        await orchestrator.run(seed_candidates, max_rounds=max_rounds, max_evaluations=max_evaluations)
        pareto = orchestrator.archive.pareto_candidates()
        qd = orchestrator.archive.sample_qd(limit=len(pareto))
        return {
            "pareto": pareto,
            "pareto_entries": orchestrator.archive.pareto_entries(),
            "qd_elites": qd,
        }

    def optimize(
        self,
        seeds: Sequence[str],
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        logger: Optional[EventLogger] = None,
        task_lm: Optional[str] = None,  # Accept for API compatibility (uses heuristic)
        reflection_lm: Optional[str] = None,  # Accept for API compatibility (uses heuristic)
    ) -> Dict[str, Any]:
        """
        Optimize prompts using uFast-GEPA with heuristic evaluation.

        Note: task_lm and reflection_lm are accepted for API compatibility with
        benchmark scripts but are not used by DefaultAdapter. This adapter uses
        deterministic heuristic evaluation instead of calling real LLMs.

        For actual LLM-based optimization, use ufast_gepa.optimize() instead.
        """
        return asyncio.run(
            self.optimize_async(
                seeds,
                max_rounds=max_rounds,
                max_evaluations=max_evaluations,
                logger=logger,
                task_lm=task_lm,
                reflection_lm=reflection_lm,
            )
        )
