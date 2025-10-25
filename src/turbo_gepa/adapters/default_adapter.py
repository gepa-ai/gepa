"""
Drop-in default adapter for TurboGEPA mirroring the classic GEPA prompt flow.

This adapter expects data instances with `input`, `additional_context`, and
`answer` fields (matching `gepa.adapters.default_adapter`). Users supply their
own `task_lm_call` / `reflect_lm_call` implementations via
``turbo_gepa.user_plugs_in``. The adapter wires those hooks into the
high-throughput orchestrator so existing GEPA workflows can migrate with minimal
changes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import litellm

from turbo_gepa.archive import Archive
from turbo_gepa.cache import DiskCache
from turbo_gepa.config import Config, DEFAULT_CONFIG, adaptive_config, adaptive_shards
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.islands import IslandContext, spawn_islands
from turbo_gepa.mutator import MutationConfig, Mutator
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.sampler import InstanceSampler
from turbo_gepa.user_plugs_in import batch_reflect_lm_call, spec_induction_lm_call, task_lm_call


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


@dataclass(slots=True)
class ModelConfig:
    """Configuration for an LLM call."""

    name: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 24000
    reasoning_effort: Optional[str] = None


class DefaultAdapter:
    """
    Helper harness for running TurboGEPA on single-component prompts.

    This adapter automatically optimizes all configuration based on dataset size,
    including shards, batch sizes, concurrency, and other parameters.

    Parameters:
        dataset: Sequence of training/validation examples
        config: Configuration object (will be auto-configured if using defaults)
        mutation_config: Optional mutation configuration
        cache_dir: Directory for disk cache (default: .turbo_gepa/cache)
        log_dir: Directory for logs (default: .turbo_gepa/logs)
        sampler_seed: Random seed for example sampling
        task_lm: LLM model for task execution (REQUIRED)
        reflection_lm: LLM model for reflection (REQUIRED)
        task_lm_temperature: Temperature for task LLM (None = use model default)
        reflection_lm_temperature: Temperature for reflection LLM (None = use model default)
        auto_config: Enable automatic configuration (default: True)
        shard_strategy: Strategy - "balanced", "conservative", or "aggressive"
        available_compute: "laptop", "workstation", or "server" (default: "laptop")

    Example usage::

        # Fully automatic configuration (recommended)
        adapter = DefaultAdapter(
            dataset=trainset,
            task_lm="openrouter/google/gemini-flash-1.5",
            reflection_lm="openrouter/google/gemini-flash-1.5"
        )
        result = adapter.optimize(seeds=["You are a helpful assistant."])

        # Conservative strategy for important tasks
        adapter = DefaultAdapter(
            dataset=trainset,
            task_lm="openrouter/google/gemini-flash-1.5",
            reflection_lm="openrouter/google/gemini-flash-1.5",
            shard_strategy="conservative"
        )

        # Server deployment with aggressive exploration
        adapter = DefaultAdapter(
            dataset=large_trainset,
            task_lm="openrouter/google/gemini-flash-1.5",
            reflection_lm="openrouter/google/gemini-flash-1.5",
            shard_strategy="aggressive",
            available_compute="server"
        )

        # Manual configuration (disables auto-config)
        config = Config(shards=(0.10, 0.30, 1.0), batch_size=16)
        adapter = DefaultAdapter(
            dataset=trainset,
            task_lm="openrouter/google/gemini-flash-1.5",
            reflection_lm="openrouter/google/gemini-flash-1.5",
            config=config,
            auto_config=False
        )
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
        auto_config: bool = True,
        shard_strategy: str = "balanced",
        available_compute: str = "laptop",
    ) -> None:
        if not dataset:
            raise ValueError("dataset must contain at least one data instance")

        # Require task_lm and reflection_lm - TurboGEPA requires real LLM integration
        if not task_lm:
            raise ValueError(
                "task_lm is required. TurboGEPA requires real LLM integration. "
                "Provide a model string like 'openrouter/google/gemini-flash-1.5'"
            )
        if not reflection_lm:
            raise ValueError(
                "reflection_lm is required. TurboGEPA requires real LLM integration. "
                "Provide a model string like 'openrouter/google/gemini-flash-1.5'"
            )

        # Apply adaptive configuration if enabled and using default config
        if auto_config and config == DEFAULT_CONFIG:
            config = adaptive_config(
                len(dataset),
                strategy=shard_strategy,
                available_compute=available_compute
            )
            print(f"🔧 Auto-configured for {len(dataset)} examples ({available_compute}, {shard_strategy}):")
            print(f"   Shards: {config.shards}")
            print(f"   Batch size: {config.batch_size}, Mutations/round: {config.max_mutations_per_round}")
            print(f"   Eval concurrency: {config.eval_concurrency}, Islands: {config.n_islands}")

        self.config = config
        self.dataset = list(dataset)

        # Normalise model configuration objects
        if isinstance(task_lm, ModelConfig):
            if task_lm_temperature is not None:
                print("⚠️  task_lm_temperature ignored because a ModelConfig was provided for task_lm.")
            self.task_model = task_lm
        else:
            assert isinstance(task_lm, str)
            self.task_model = ModelConfig(
                name=task_lm,
                temperature=task_lm_temperature if task_lm_temperature is not None else config.task_lm_temperature,
            )

        if isinstance(reflection_lm, ModelConfig):
            if reflection_lm_temperature is not None:
                print("⚠️  reflection_lm_temperature ignored because a ModelConfig was provided for reflection_lm.")
            self.reflection_model = reflection_lm
        else:
            assert isinstance(reflection_lm, str)
            self.reflection_model = ModelConfig(
                name=reflection_lm,
                temperature=reflection_lm_temperature if reflection_lm_temperature is not None else config.reflection_lm_temperature,
            )

        # Convenience string attributes (backwards-compatibility)
        self.task_lm = self.task_model.name
        self.reflection_lm = self.reflection_model.name
        self.example_map = {
            data.id if data.id is not None else f"example-{idx}": data for idx, data in enumerate(self.dataset)
        }
        self._example_ids = list(self.example_map.keys())
        self._sampler_seed = sampler_seed
        self.sampler = InstanceSampler(self._example_ids, seed=self._sampler_seed)
        self.base_cache_dir = cache_dir or config.cache_path
        self.base_log_dir = log_dir or config.log_path
        Path(self.base_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.base_log_dir).mkdir(parents=True, exist_ok=True)
        self.cache = DiskCache(self.base_cache_dir)
        self.archive = Archive(
            bins_length=config.qd_bins_length,
            bins_bullets=config.qd_bins_bullets,
            flags=config.qd_flags,
        )

        # Temperature support will be checked lazily during Phase 2 (temperature optimization)
        # No upfront LLM call needed - we optimize prompts first (Phase 1), then temperature (Phase 2)
        self.temperature_supported = True  # Assume supported, check later if needed

        # Create batched reflection runner and spec induction runner
        batch_reflection_runner = self._create_batched_llm_reflection_runner(reflection_lm)
        spec_induction_runner = self._create_spec_induction_runner(reflection_lm)

        # Pass temperature support flag to mutator
        self._mutation_config = mutation_config or MutationConfig(
            reflection_batch_size=config.reflection_batch_size,
            max_mutations=config.max_mutations_per_round,
            max_tokens=config.max_tokens,
        )
        self._batch_reflection_runner = batch_reflection_runner
        self._spec_induction_runner = spec_induction_runner
        self.mutator = Mutator(
            self._mutation_config,
            batch_reflection_runner=self._batch_reflection_runner,
            spec_induction_runner=self._spec_induction_runner,
            temperature_mutations_enabled=False,  # Disabled for Phase 1 - only optimize prompt quality
        )
        self.log_dir = log_dir or config.log_path

    def _check_temperature_support(self, model: str, test_temp: float) -> bool:
        """Quick test to see if model supports custom temperature.

        Uses litellm for provider-agnostic testing (OpenAI, Anthropic, etc.)
        """
        import time
        start = time.time()
        print(f"🔍 Checking temperature support for {model}...")

        try:
            import litellm

            # Quick test call with minimal tokens
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                temperature=test_temp,
                max_tokens=1,
            )
            elapsed = time.time() - start
            print(f"✓ Temperature check passed ({elapsed:.2f}s)")
            return True
        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e).lower()
            # Check if error is specifically about temperature
            if "temperature" in error_msg or "does not support" in error_msg or "not supported" in error_msg:
                print(f"✗ Temperature not supported ({elapsed:.2f}s)")
                return False
            # Other errors (auth, network) - assume temperature works
            print(f"⚠️  Temperature check had error but assuming support ({elapsed:.2f}s): {str(e)[:100]}")
            return True

    def _create_batched_llm_reflection_runner(self, reflection_lm: str):
        """Create a batched reflection runner that calls a real LLM with multiple parents."""
        async def batched_llm_reflection_runner(parent_contexts: list, num_mutations: int) -> list[str]:
            if not parent_contexts:
                return []

            import time
            start_time = time.time()

            try:
                from litellm import acompletion

                # Build rich reflection prompt showing multiple successful prompts
                parent_summaries = []
                for i, ctx in enumerate(parent_contexts[:5]):  # Limit to 5 parents for token efficiency
                    prompt_text = ctx.get("prompt", "")
                    meta = ctx.get("meta", {})
                    quality = meta.get("quality", 0.0)
                    traces = ctx.get("traces", [])

                    # Temperature context if available
                    temp_info = ""
                    if "temperature" in meta:
                        temp_info = f", temp={meta['temperature']:.1f}"

                    # Performance summary from traces
                    if traces:
                        avg_quality = sum(t.get("quality", 0) for t in traces[:3]) / min(len(traces), 3)
                        perf_summary = f"Recent avg: {avg_quality:.1%}"
                    else:
                        perf_summary = f"Quality: {quality:.1%}"

                    parent_summaries.append(f"""PROMPT {chr(65+i)} ({perf_summary}{temp_info}):
"{prompt_text}"
""")

                all_parents_text = "\n".join(parent_summaries)

                example_summaries = []
                if isinstance(getattr(self.mutator, "_reflection_examples", None), list) and self.mutator._reflection_examples:
                    for j, ex in enumerate(self.mutator._reflection_examples[:5]):
                        example_input = ex.get("input", "").strip()
                        example_answer = (ex.get("expected_answer") or ex.get("answer") or "").strip()
                        assistant_output = ex.get("assistant_output", "").strip()
                        feedback_text = ex.get("feedback", "").strip()
                        additional = ex.get("additional_context") or {}
                        solution = additional.get("solution") if isinstance(additional, dict) else None
                        example_block = [f"Example {j + 1} Input: {example_input}"]
                        if assistant_output:
                            example_block.append(f"Example {j + 1} Assistant Output: {assistant_output}")
                        if example_answer:
                            example_block.append(f"Example {j + 1} Correct Answer: {example_answer}")
                        if feedback_text:
                            example_block.append(f"Example {j + 1} Feedback: {feedback_text}")
                        if solution:
                            formatted_solution = "\n".join(str(solution).splitlines())
                            example_block.append(
                                f"Example {j + 1} Reference Solution:\n{formatted_solution}"
                            )
                        example_summaries.append("\n".join(example_block))

                examples_text = "\n\n".join(example_summaries)

                reflection_prompt = f"""I provided an assistant with the following instructions to perform a task:

Existing high-performing instructions and their recent quality:
{all_parents_text}

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:

{examples_text if example_summaries else '(no additional examples available)'}

Your task is to write {num_mutations} new instruction variants for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain-specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task; if so, include that in the instruction as well.

IMPORTANT guidance:
- Extract and include domain-specific factual knowledge, techniques, and patterns from the examples and solutions
- Include key mathematical principles, common solution approaches, and problem-solving strategies observed in the reference solutions
- Capture the types of problems, solution methods, and domain expertise needed to solve similar problems
- Address common pitfalls and edge cases specific to this problem domain
- Ensure each instruction emphasizes the required answer format

You CAN and SHOULD include domain-specific terminology, solution techniques, and factual knowledge from the examples and reference solutions. The goal is to teach the assistant to solve NEW problems in the SAME DOMAIN by providing it with the domain knowledge and strategies it needs.

Write {num_mutations} new instruction variants. Separate each instruction with a line containing only "---"."""

                completion_kwargs: Dict[str, Any] = {
                    "model": self.reflection_model.name,
                    "messages": [{"role": "user", "content": reflection_prompt}],
                }
                if self.reflection_model.max_tokens is not None:
                    completion_kwargs["max_tokens"] = self.reflection_model.max_tokens
                if self.reflection_model.temperature is not None:
                    completion_kwargs["temperature"] = self.reflection_model.temperature
                if self.reflection_model.reasoning_effort is not None:
                    completion_kwargs["reasoning_effort"] = self.reflection_model.reasoning_effort

                print(f"🔵 REFLECTION CALL START: model={completion_kwargs['model']}, mutations={num_mutations}", flush=True)
                response = await acompletion(**completion_kwargs)
                print(f"🟢 REFLECTION CALL SUCCESS: mutations={num_mutations}", flush=True)

                elapsed = time.time() - start_time
                content = response.choices[0].message.content
                # Split by --- and clean up
                mutations = [m.strip() for m in content.split("---") if m.strip()]

                # Log timing for diagnostics
                print(f"   ⚡ Batched reflection ({len(parent_contexts)} parents): {elapsed:.2f}s → {len(mutations)} mutations")

                return mutations[:num_mutations]

            except Exception as e:
                elapsed = time.time() - start_time
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"❌ REFLECTION CALL ERROR after {elapsed:.2f}s: {error_type}: {error_msg}", flush=True)
                raise RuntimeError(
                    f"Batched reflection LLM call failed after {elapsed:.2f}s ({error_type}: {error_msg}). "
                    "Check your API key, model name, and network connection."
                ) from e

        return batched_llm_reflection_runner

    def _create_spec_induction_runner(self, reflection_lm: str):
        """Create a spec induction runner that generates fresh prompts from task examples."""
        async def spec_induction_runner(task_examples: list, num_specs: int) -> list[str]:
            if not task_examples:
                return []

            import time
            start_time = time.time()

            try:
                from litellm import acompletion

                # Build rich examples summary with solutions (like incremental_reflection)
                example_summaries = []
                for i, ex in enumerate(task_examples[:3]):  # Limit to 3 examples
                    input_text = ex.get("input", "")
                    answer_text = ex.get("answer", "")
                    additional_context = ex.get("additional_context") or {}

                    example_block = [f"Example {i+1}:"]
                    example_block.append(f"Input: {input_text}")
                    example_block.append(f"Expected Output: {answer_text}")

                    # Show additional context (AIME solutions, hints, etc.)
                    if additional_context and isinstance(additional_context, dict):
                        for k, v in additional_context.items():
                            # Format solution nicely
                            formatted_value = "\n".join(str(v).splitlines())
                            example_block.append(f"{k.title()}: {formatted_value}")

                    example_summaries.append("\n".join(example_block))

                all_examples_text = "\n\n".join(example_summaries)

                # Spec induction prompt matching OG GEPA's philosophy
                spec_prompt = f"""Below are {len(task_examples)} examples of a task with full context including reference solutions:

{all_examples_text}

Your task is to generate {num_specs} different instruction variants that would teach an AI assistant to solve tasks like these.

Read the inputs carefully and identify the input format and infer detailed task description.

Read all the reference solutions and identify all niche and domain-specific factual information about the task and include it in the instructions, as a lot of it may not be available to the assistant in the future. Extract generalizable strategies used in the solutions and include those as well.

IMPORTANT guidance:
- Extract and include domain-specific factual knowledge, techniques, and patterns from the examples and solutions
- Include key mathematical principles, common solution approaches, and problem-solving strategies observed in the reference solutions
- Capture the types of problems, solution methods, and domain expertise needed to solve similar problems
- Address common pitfalls and edge cases specific to this problem domain
- Ensure each instruction emphasizes the required answer format

Each instruction should:
- Be self-contained and clear
- Teach the assistant with domain knowledge and strategies from the examples
- Be different from the others in approach or emphasis

Output format: Return each instruction separated by "---" (exactly {num_specs} instructions)."""

                # Build kwargs
                completion_kwargs: Dict[str, Any] = {
                    "model": self.reflection_model.name,
                    "messages": [{"role": "user", "content": spec_prompt}],
                }
                if self.reflection_model.max_tokens is not None:
                    completion_kwargs["max_tokens"] = self.reflection_model.max_tokens
                if self.reflection_model.temperature is not None:
                    completion_kwargs["temperature"] = self.reflection_model.temperature
                if self.reflection_model.reasoning_effort is not None:
                    completion_kwargs["reasoning_effort"] = self.reflection_model.reasoning_effort
                print(f"🔵 SPEC CALL START: model={completion_kwargs['model']}, specs={num_specs}", flush=True)

                response = await acompletion(**completion_kwargs)

                elapsed = time.time() - start_time
                content = response.choices[0].message.content
                # Split by --- and clean up
                specs = [s.strip() for s in content.split("---") if s.strip()]

                # Log timing for diagnostics
                print(f"🟢 SPEC CALL SUCCESS: specs={len(specs)} in {elapsed:.2f}s", flush=True)
                print(f"   ⚡ Spec induction ({len(task_examples)} examples): {elapsed:.2f}s → {len(specs)} specs")

                return specs[:num_specs]

            except Exception as e:
                elapsed = time.time() - start_time
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"❌ SPEC CALL ERROR after {elapsed:.2f}s: {error_type}: {error_msg}", flush=True)
                raise RuntimeError(
                    f"Spec induction LLM call failed after {elapsed:.2f}s ({error_type}: {error_msg}). "
                    "Check your API key, model name, and network connection."
                ) from e

        return spec_induction_runner

    def _sample_examples(self, num_examples: int) -> list[dict]:
        """Sample random examples for spec induction."""
        import random
        example_ids = list(self.example_map.keys())
        if len(example_ids) <= num_examples:
            sampled_ids = example_ids
        else:
            sampled_ids = random.sample(example_ids, num_examples)

        return [self.example_map[eid].to_payload() for eid in sampled_ids]

    def _normalize_seeds(self, seeds: Sequence[str | Candidate], *, source: str) -> List[Candidate]:
        normalized: List[Candidate] = []
        for seed in seeds:
            if isinstance(seed, Candidate):
                meta = dict(seed.meta)
                meta["source"] = source
                normalized.append(Candidate(text=seed.text, meta=meta))
            else:
                normalized.append(Candidate(text=seed, meta={"source": source}))
        return normalized

    def _make_mutator(self) -> Mutator:
        return Mutator(
            self._mutation_config,
            batch_reflection_runner=self._batch_reflection_runner,
            spec_induction_runner=self._spec_induction_runner,
            temperature_mutations_enabled=False,  # Disabled - Phase 2 can enable via set_temperature_mutations_enabled
        )

    def _make_archive(self) -> Archive:
        return Archive(
            bins_length=self.config.qd_bins_length,
            bins_bullets=self.config.qd_bins_bullets,
            flags=self.config.qd_flags,
        )

    def _make_sampler(self, *, seed_offset: int = 0) -> InstanceSampler:
        return InstanceSampler(self._example_ids, seed=self._sampler_seed + seed_offset)

    def _make_cache(self, island_id: int | None = None) -> DiskCache:
        if island_id is None:
            return DiskCache(self.base_cache_dir)
        path = Path(self.base_cache_dir)
        island_path = path / f"island_{island_id}"
        island_path.mkdir(parents=True, exist_ok=True)
        return DiskCache(str(island_path))

    def _make_log_dir(self, island_id: int | None = None) -> str:
        path = Path(self.base_log_dir)
        if island_id is None:
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        island_path = path / f"island_{island_id}"
        island_path.mkdir(parents=True, exist_ok=True)
        return str(island_path)

    async def _task_runner(self, candidate: Candidate, example_id: str) -> Dict[str, float]:
        """Execute task LLM on a single example."""
        example = self.example_map[example_id].to_payload()

        try:
            from litellm import acompletion

            completion_kwargs: Dict[str, Any] = {
                "model": self.task_model.name,
                "messages": [
                    {"role": "system", "content": candidate.text},
                    {"role": "user", "content": example["input"]},
                ],
            }
            if self.task_model.max_tokens is not None:
                completion_kwargs["max_tokens"] = self.task_model.max_tokens

            candidate_temperature = candidate.meta.get("temperature")
            if candidate_temperature is not None:
                completion_kwargs["temperature"] = candidate_temperature
            elif self.task_model.temperature is not None:
                completion_kwargs["temperature"] = self.task_model.temperature

            reasoning_effort = candidate.meta.get("reasoning_effort", self.task_model.reasoning_effort)
            if reasoning_effort is not None:
                completion_kwargs["reasoning_effort"] = reasoning_effort

            # Try with temperature, fall back without it if model doesn't support it
            import time as _time_module
            _start_llm = _time_module.time()
            print(f"🔵 LLM CALL START: model={completion_kwargs['model']}, example_id={example_id}")
            try:
                response = await acompletion(**completion_kwargs)
                _elapsed_llm = _time_module.time() - _start_llm
                print(f"🟢 LLM CALL SUCCESS: {_elapsed_llm:.2f}s, example_id={example_id}")
            except Exception as e:
                _elapsed_llm = _time_module.time() - _start_llm
                print(f"🔴 LLM CALL ERROR: {_elapsed_llm:.2f}s, error={type(e).__name__}: {str(e)[:100]}")
                # Some models don't support custom temperature (e.g., o1-preview)
                if "temperature" in str(e).lower() and completion_kwargs.get("temperature") is not None:
                    print(f"🟡 RETRYING without temperature parameter...")
                    completion_kwargs.pop("temperature", None)
                    response = await acompletion(**completion_kwargs)
                    print(f"🟢 RETRY SUCCESS")
                else:
                    raise  # Re-raise if it's a different error

            model_output = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            # Check if answer is in output
            quality = 1.0 if example["answer"] in model_output else 0.0

            metrics = {
                "quality": quality,
                "neg_cost": -float(tokens_used),
                "tokens": float(tokens_used),
                "response": model_output,
                "example_id": example_id,
                "output": model_output,
                "input": example.get("input", ""),
                "expected_answer": example.get("answer"),
                "additional_context": example.get("additional_context"),
            }
            return metrics

        except Exception as e:
            # No heuristic fallback - raise the error with clear message
            error_type = type(e).__name__
            error_msg = str(e)
            raise RuntimeError(
                f"Task LLM call failed ({error_type}: {error_msg}). "
                "Check your API key, model name, and network connection."
            ) from e

    def _build_orchestrator(
        self,
        *,
        enable_auto_stop: bool = False,
        display_progress: bool = True,
        temperature_mutations_enabled: bool | None = None,
        island_context: Optional[IslandContext] = None,
        cache: Optional[DiskCache] = None,
        archive: Optional[Archive] = None,
        sampler: Optional[InstanceSampler] = None,
        mutator: Optional[Mutator] = None,
        log_dir: Optional[str] = None,
        metrics_callback: Optional[Callable] = None,
    ) -> Orchestrator:
        target_mutator = mutator or self.mutator
        if temperature_mutations_enabled is not None:
            target_mutator.set_temperature_mutations_enabled(temperature_mutations_enabled)
        mutator = target_mutator

        # Only optimize for quality, ignore token cost
        def metrics_mapper(metrics: Dict[str, float]) -> Dict[str, float]:
            return {"quality": metrics.get("quality", 0.0)}

        evaluator = AsyncEvaluator(
            cache=cache or self.cache,
            task_runner=self._task_runner,
            metrics_mapper=metrics_mapper,
        )
        log_path = log_dir or self.base_log_dir
        # Create stop governor if auto-stop enabled
        # Use provided metrics_callback, or create dashboard if progress display is enabled
        dashboard_enabled = False
        if metrics_callback is None and display_progress:
            try:
                from turbo_gepa.dashboard import TerminalDashboard
                dashboard = TerminalDashboard()
                metrics_callback = dashboard.update
                dashboard_enabled = True
            except ImportError:
                # plotext not installed, fall back to simple progress
                pass

        return Orchestrator(
            config=self.config,
            evaluator=evaluator,
            archive=archive or self.archive,
            sampler=sampler or self.sampler,
            mutator=mutator,
            cache=cache or self.cache,
            enable_auto_stop=enable_auto_stop,
            show_progress=display_progress and not dashboard_enabled,  # Disable inline progress when dashboard is active
            example_sampler=self._sample_examples,
            island_context=island_context,
            metrics_callback=metrics_callback,
        )

    async def optimize_async(
        self,
        seeds: Sequence[str | Candidate],
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        task_lm: Optional[str] = None,  # Accept but ignore (uses heuristic)
        reflection_lm: Optional[str] = None,  # Accept but ignore (uses heuristic)
        enable_auto_stop: bool = False,  # Enable automatic stopping
        optimize_temperature_after_convergence: bool = False,  # Stage temperature optimization
        display_progress: bool = True,  # Show progress charts
        metrics_callback: Optional[Callable] = None,  # Callback for dashboard updates
    ) -> Dict[str, Any]:
        if self.config.n_islands > 1:
            if optimize_temperature_after_convergence:
                return await self._optimize_multi_island_staged(
                    seeds,
                    max_rounds=max_rounds,
                    max_evaluations=max_evaluations,
                    enable_auto_stop=enable_auto_stop,
                    display_progress=display_progress,
                )
            return await self._optimize_multi_island(
                seeds,
                max_rounds=max_rounds,
                max_evaluations=max_evaluations,
                enable_auto_stop=enable_auto_stop,
                display_progress=display_progress,
            )

        # Staged temperature optimization: two-phase approach
        if optimize_temperature_after_convergence:
            return await self._optimize_staged_temperature(
                seeds, max_rounds, max_evaluations, enable_auto_stop, display_progress
            )

        # Standard integrated optimization
        orchestrator = self._build_orchestrator(
            enable_auto_stop=enable_auto_stop,
            display_progress=display_progress,
            metrics_callback=metrics_callback,
        )
        # Accept either strings or Candidate objects
        seed_candidates = []
        for seed in seeds:
            if isinstance(seed, Candidate):
                # Preserve metadata (including temperature)
                meta = dict(seed.meta, source="seed")
                seed_candidates.append(Candidate(text=seed.text, meta=meta))
            else:
                # String seed
                seed_candidates.append(Candidate(text=seed, meta={"source": "seed"}))
        await orchestrator.run(seed_candidates, max_rounds=max_rounds, max_evaluations=max_evaluations)
        pareto = orchestrator.archive.pareto_candidates()
        qd = orchestrator.archive.sample_qd(limit=len(pareto))
        return {
            "pareto": pareto,
            "pareto_entries": orchestrator.archive.pareto_entries(),
            "qd_elites": qd,
        }

    async def _optimize_staged_temperature(
        self,
        seeds: Sequence[str | Candidate],
        max_rounds: Optional[int],
        max_evaluations: Optional[int],
        enable_auto_stop: bool,
        display_progress: bool = True,
    ) -> Dict[str, Any]:
        """Two-phase optimization: prompts first, then temperature.

        Phase 1: Optimize prompts WITHOUT temperature (seeds have no temperature)
        Phase 2: Take top-K prompts and create temperature seeds for final optimization

        This approach avoids variance issues by separating "what to say" from
        "how stochastic to be".
        """
        print("\n🔄 PHASE 1: Optimizing prompts (no temperature cycling)...")

        # Phase 1: Ensure seeds have NO temperature (so mutator won't create temp variants)
        phase1_seeds = []
        for seed in seeds:
            if isinstance(seed, Candidate):
                # Strip temperature if present
                meta = {k: v for k, v in seed.meta.items() if k != "temperature"}
                meta["source"] = "seed_phase1"
                phase1_seeds.append(Candidate(text=seed.text, meta=meta))
            else:
                phase1_seeds.append(Candidate(text=seed, meta={"source": "seed_phase1"}))

        # Run Phase 1 optimization (70% of budget)
        # Since seeds have no temperature, mutator won't create temp variants (opt-in design)
        phase1_budget = int(max_evaluations * 0.7) if max_evaluations else None
        phase1_rounds = max_rounds if max_rounds else 10  # Default to 10 if unlimited

        orchestrator1 = self._build_orchestrator(
            enable_auto_stop=enable_auto_stop,
            display_progress=display_progress,
            temperature_mutations_enabled=False,
        )
        await orchestrator1.run(phase1_seeds, max_rounds=phase1_rounds, max_evaluations=phase1_budget)

        phase1_pareto = orchestrator1.archive.pareto_entries()
        print(f"   ✓ Phase 1 complete: {len(phase1_pareto)} candidates on Pareto frontier")

        # Early exit if temperature not supported
        if not self.temperature_supported:
            print("   ⚠️  Skipping Phase 2 (temperature not supported by model)")
            return {
                "pareto": [e.candidate for e in phase1_pareto],
                "pareto_entries": phase1_pareto,
                "qd_elites": orchestrator1.archive.sample_qd(limit=len(phase1_pareto)),
                "phase1_pareto": [e.candidate for e in phase1_pareto],
            }

        # Early exit if no pareto frontier
        if not phase1_pareto:
            print("   ⚠️  Skipping Phase 2 (no candidates from Phase 1)")
            return {
                "pareto": [],
                "pareto_entries": [],
                "qd_elites": [],
                "phase1_pareto": [],
            }

        print("\n🌡️  PHASE 2: Temperature optimization for top prompts...")

        # Take top K prompts sorted by quality
        top_k = min(5, len(phase1_pareto))
        top_entries = sorted(
            phase1_pareto,
            key=lambda e: e.result.objectives.get(self.config.promote_objective, 0.0),
            reverse=True
        )[:top_k]

        # Create temperature-enabled seeds from top prompts
        # Now mutator WILL create temperature variants (because seeds have temperature)
        temp_seeds = []
        for entry in top_entries:
            # Start with mid-range temperature
            meta = dict(entry.candidate.meta, temperature=0.5, source="phase2_seed")
            temp_seeds.append(Candidate(text=entry.candidate.text, meta=meta))

        print(f"   Starting with {len(temp_seeds)} temperature-enabled seeds from top prompts")

        # Run Phase 2 optimization (30% of remaining budget, shorter rounds)
        phase2_budget = int(max_evaluations * 0.3) if max_evaluations else None
        phase2_rounds = min(5, max_rounds) if max_rounds else 5  # Shorter optimization

        orchestrator2 = self._build_orchestrator(
            enable_auto_stop=False,
            display_progress=display_progress,
            temperature_mutations_enabled=True,
        )  # Short phase, no auto-stop
        await orchestrator2.run(temp_seeds, max_rounds=phase2_rounds, max_evaluations=phase2_budget)

        phase2_pareto = orchestrator2.archive.pareto_entries()
        print(f"   ✓ Phase 2 complete: {len(phase2_pareto)} best (prompt, temperature) combinations")

        return {
            "pareto": [e.candidate for e in phase2_pareto],
            "pareto_entries": phase2_pareto,
            "qd_elites": orchestrator2.archive.sample_qd(limit=len(phase2_pareto)),
            "phase1_pareto": [e.candidate for e in phase1_pareto],  # Also return phase 1 results
        }

    async def _optimize_multi_island(
        self,
        seeds: Sequence[str | Candidate],
        *,
        max_rounds: Optional[int],
        max_evaluations: Optional[int],
        enable_auto_stop: bool,
        display_progress: bool,
        temperature_mutations_enabled: bool | None = None,
    ) -> Dict[str, Any]:
        n_islands = max(1, self.config.n_islands)
        if n_islands > 1:
            tuned_period = max(1, n_islands // 2)
            tuned_k = max(1, min(n_islands, self.config.migration_k or n_islands))
            if (
                self.config.migration_period != tuned_period
                or self.config.migration_k != tuned_k
            ):
                print(
                    f"🔁 Multi-island tuning: migration_period {self.config.migration_period}→{tuned_period}, "
                    f"migration_k {self.config.migration_k}→{tuned_k}"
                )
                self.config.migration_period = tuned_period
                self.config.migration_k = tuned_k
        normalized_seeds = self._normalize_seeds(seeds, source="seed")

        # Set up larger thread pool for concurrent file operations
        import concurrent.futures
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
        loop.set_default_executor(executor)

        # Create shared cache across all islands (better cache hits + controlled concurrency)
        shared_cache = self._make_cache(island_id=None)

        metrics_queue: asyncio.Queue | None = None
        aggregator_task: asyncio.Task | None = None
        aggregator: IslandProgressAggregator | None = None
        if display_progress:
            metrics_queue = asyncio.Queue()
            aggregator = IslandProgressAggregator(n_islands=n_islands, max_rounds=max_rounds or 100)

            

        async def run_islands() -> List[Orchestrator | None]:
            island_results: List[Orchestrator | None] = [None] * n_islands

            async def worker(context: IslandContext) -> None:
                # Use shared cache across islands
                cache = shared_cache
                archive = self._make_archive()
                sampler = self._make_sampler(seed_offset=context.island_id)
                mutator = self._make_mutator()
                if temperature_mutations_enabled is not None:
                    mutator.set_temperature_mutations_enabled(temperature_mutations_enabled)
                log_dir = self._make_log_dir(context.island_id)
                orchestrator = self._build_orchestrator(
                    enable_auto_stop=enable_auto_stop,
                    display_progress=False if metrics_queue else (display_progress and context.island_id == 0),
                    temperature_mutations_enabled=temperature_mutations_enabled,
                    island_context=context,
                    cache=cache,
                    archive=archive,
                    sampler=sampler,
                    mutator=mutator,
                    log_dir=log_dir
                )
                island_seeds = [
                    Candidate(text=seed.text, meta=dict(seed.meta, island=context.island_id))
                    for seed in normalized_seeds
                ]
                await orchestrator.run(
                    island_seeds,
                    max_rounds=max_rounds,
                    max_evaluations=max_evaluations,
                )
                island_results[context.island_id] = orchestrator

            tasks = await spawn_islands(n_islands, worker, metrics_queue=metrics_queue)
            await asyncio.gather(*tasks)
            return island_results

        orchestrators = await run_islands()

        if metrics_queue is not None:
            await metrics_queue.put(None)
            if aggregator_task:
                await aggregator_task

        combined_archive = self._make_archive()
        inserts: List[tuple[Candidate, EvalResult]] = []
        for orchestrator in orchestrators:
            if orchestrator is None:
                continue
            for entry in orchestrator.archive.pareto_entries():
                inserts.append((entry.candidate, entry.result))
        if inserts:
            await combined_archive.batch_insert(inserts)
        pareto_entries = combined_archive.pareto_entries()
        pareto_candidates = [entry.candidate for entry in pareto_entries]
        qd_elites = combined_archive.sample_qd(limit=len(pareto_entries)) if pareto_entries else []
        return {
            "pareto": pareto_candidates,
            "pareto_entries": pareto_entries,
            "qd_elites": qd_elites,
        }

    async def _optimize_multi_island_staged(
        self,
        seeds: Sequence[str | Candidate],
        *,
        max_rounds: Optional[int],
        max_evaluations: Optional[int],
        enable_auto_stop: bool,
        display_progress: bool,
    ) -> Dict[str, Any]:
        phase1_budget = int(max_evaluations * 0.7) if max_evaluations else None
        phase1_rounds = max_rounds if max_rounds else 10

        phase1_result = await self._optimize_multi_island(
            seeds,
            max_rounds=phase1_rounds,
            max_evaluations=phase1_budget,
            enable_auto_stop=enable_auto_stop,
            display_progress=display_progress,
            temperature_mutations_enabled=False,
        )
        phase1_entries = phase1_result.get("pareto_entries", [])
        if not phase1_entries:
            return phase1_result | {"phase1_pareto": []}

        top_k = min(5, len(phase1_entries))
        top_entries = sorted(
            phase1_entries,
            key=lambda e: e.result.objectives.get(self.config.promote_objective, 0.0),
            reverse=True,
        )[:top_k]

        temp_seeds = []
        for entry in top_entries:
            meta = dict(entry.candidate.meta, temperature=0.5, source="phase2_seed")
            temp_seeds.append(Candidate(text=entry.candidate.text, meta=meta))

        phase2_budget = int(max_evaluations * 0.3) if max_evaluations else None
        phase2_rounds = min(5, max_rounds) if max_rounds else 5

        phase2_result = await self._optimize_multi_island(
            temp_seeds,
            max_rounds=phase2_rounds,
            max_evaluations=phase2_budget,
            enable_auto_stop=False,
            display_progress=display_progress,
            temperature_mutations_enabled=True,
        )
        phase2_result["phase1_pareto"] = [entry.candidate for entry in phase1_entries]
        return phase2_result

    def optimize(
        self,
        seeds: Sequence[str | Candidate] | None = None,
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        task_lm: Optional[str] = None,  # Accept for API compatibility (uses heuristic)
        reflection_lm: Optional[str] = None,  # Accept for API compatibility (uses heuristic)
        enable_auto_stop: bool = False,  # Enable automatic stopping
        optimize_temperature_after_convergence: bool = False,  # Stage temperature optimization
        display_progress: bool = True,  # Show progress charts
        enable_seed_initialization: bool = False,  # Use PROMPT-MII-style seed generation
        num_generated_seeds: int = 3,  # How many seeds to generate if initializing
        metrics_callback: Optional[Callable] = None,  # Callback for dashboard updates
    ) -> Dict[str, Any]:
        """
        Optimize prompts using TurboGEPA with heuristic evaluation.

        Parameters:
            seeds: Initial prompt candidates
            max_rounds: Maximum optimization rounds (None = unlimited)
            max_evaluations: Maximum evaluations (None = unlimited)
            task_lm: Task LLM (for compatibility)
            reflection_lm: Reflection LLM (for compatibility)
            enable_auto_stop: Enable automatic convergence detection
            optimize_temperature_after_convergence: Stage temperature optimization after
                prompt optimization (Phase 1: optimize prompts, Phase 2: optimize temperature)

        Note: task_lm and reflection_lm are accepted for API compatibility with
        benchmark scripts but are not used by DefaultAdapter. This adapter uses
        deterministic heuristic evaluation instead of calling real LLMs.

        For actual LLM-based optimization, use turbo_gepa.optimize() instead.

        Parameters:
            enable_seed_initialization: If True, use PROMPT-MII-style spec induction to
                generate smart initial seeds from task examples instead of using generic prompts.
                Requires reflection_lm to be set. Can optimize user-provided seeds or generate from scratch.
            num_generated_seeds: Number of seeds to generate if enable_seed_initialization=True
        """
        import time
        optimize_start = time.time()
        print(f"\n⏱️  optimize() called at {optimize_start:.2f}")

        # Handle seed initialization if requested
        if enable_seed_initialization:
            print(f"🌱 Initializing seeds...")
            from ..seed_initializer import maybe_initialize_seeds

            # Convert user seeds to strings if provided
            user_seed_strings = None
            if seeds:
                user_seed_strings = []
                for seed in seeds:
                    if isinstance(seed, Candidate):
                        user_seed_strings.append(seed.text)
                    else:
                        user_seed_strings.append(seed)

            # Generate/optimize seeds
            seeds = asyncio.run(
                maybe_initialize_seeds(
                    dataset=self.dataset,
                    user_seeds=user_seed_strings,
                    enable_seed_initialization=True,
                    num_generated_seeds=num_generated_seeds,
                    reflection_lm=self.reflection_lm,
                    reflection_lm_temperature=self.reflection_model.temperature,
                )
            )
        elif seeds is None:
            # No seeds provided and initialization disabled - use default
            seeds = ["You are a helpful assistant. Follow the instructions carefully."]

        print(f"⏱️  Calling optimize_async() after {time.time() - optimize_start:.2f}s")
        return asyncio.run(
            self.optimize_async(
                seeds,
                max_rounds=max_rounds,
                max_evaluations=max_evaluations,
                task_lm=task_lm,
                reflection_lm=reflection_lm,
                enable_auto_stop=enable_auto_stop,
                optimize_temperature_after_convergence=optimize_temperature_after_convergence,
                display_progress=display_progress,
                metrics_callback=metrics_callback,
            )
        )
