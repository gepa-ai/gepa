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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from turbo_gepa.archive import Archive
from turbo_gepa.cache import DiskCache
from turbo_gepa.config import Config, DEFAULT_CONFIG, adaptive_config, adaptive_shards
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate
from turbo_gepa.logging_utils import EventLogger, build_logger
from turbo_gepa.mutator import MutationConfig, Mutator
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.sampler import InstanceSampler
from turbo_gepa.stop_governor import StopGovernor, StopGovernorConfig
from turbo_gepa.token_controller import TokenCostController
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

        # Check if model supports temperature (quick upfront test)
        self.temperature_supported = True  # Assume supported unless proven otherwise
        if task_lm and task_lm_temperature is not None:
            self.temperature_supported = self._check_temperature_support(task_lm, task_lm_temperature)
            if not self.temperature_supported:
                print(f"⚠️  Model {task_lm} doesn't support custom temperature - using default")
                self.task_lm_temperature = None

        # Create batched reflection runner and spec induction runner
        batch_reflection_runner = self._create_batched_llm_reflection_runner(reflection_lm)
        spec_induction_runner = self._create_spec_induction_runner(reflection_lm)

        # Pass temperature support flag to mutator
        self.mutator = Mutator(
            mutation_config
            or MutationConfig(
                reflection_batch_size=config.reflection_batch_size,
                max_mutations=config.max_mutations_per_round,
                max_tokens=config.max_tokens,
            ),
            batch_reflection_runner=batch_reflection_runner,
            spec_induction_runner=spec_induction_runner,
            temperature_mutations_enabled=self.temperature_supported,
        )
        self.token_controller = TokenCostController(config.max_tokens)
        self.log_dir = log_dir or config.log_path

    def _check_temperature_support(self, model: str, test_temp: float) -> bool:
        """Quick test to see if model supports custom temperature.

        Uses litellm for provider-agnostic testing (OpenAI, Anthropic, etc.)
        """
        try:
            import litellm

            # Quick test call with minimal tokens
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                temperature=test_temp,
                max_tokens=1,
            )
            return True
        except Exception as e:
            error_msg = str(e).lower()
            # Check if error is specifically about temperature
            if "temperature" in error_msg or "does not support" in error_msg or "not supported" in error_msg:
                return False
            # Other errors (auth, network) - assume temperature works
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

                reflection_prompt = f"""You are optimizing prompts for a challenging task. Below are {len(parent_contexts)} successful prompts with different approaches and trade-offs:

{all_parents_text}

Generate {num_mutations} NEW prompt variants that:
1. Synthesize the best ideas from multiple successful prompts above
2. Address any common weaknesses you observe
3. Explore different quality/efficiency trade-offs
4. Are substantially different from each other

Output format: Return each new prompt separated by "---" (exactly {num_mutations} prompts)."""

                # Build kwargs, only include temperature if not None
                completion_kwargs = {
                    "model": reflection_lm,
                    "messages": [{"role": "user", "content": reflection_prompt}],
                }
                if self.reflection_lm_temperature is not None:
                    completion_kwargs["temperature"] = self.reflection_lm_temperature

                response = await acompletion(**completion_kwargs)

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

                # Build examples summary
                example_summaries = []
                for i, ex in enumerate(task_examples[:3]):  # Limit to 3 examples
                    input_text = ex.get("input", "")
                    answer_text = ex.get("answer", "")
                    example_summaries.append(f"""Example {i+1}:
Input: {input_text[:200]}{'...' if len(input_text) > 200 else ''}
Expected Output: {answer_text[:100]}{'...' if len(answer_text) > 100 else ''}
""")

                all_examples_text = "\n".join(example_summaries)

                # Simple, freeform spec induction prompt
                spec_prompt = f"""You are designing prompts for an AI system. Below are {len(task_examples)} examples of the task:

{all_examples_text}

Generate {num_specs} different prompt variations that would teach an AI to solve tasks like these.

Each prompt should:
- Be self-contained and clear
- Address the key skills needed (reasoning, calculation, formatting, etc.)
- Be different from the others in approach or emphasis

Output format: Return each prompt separated by "---" (exactly {num_specs} prompts)."""

                # Build kwargs
                completion_kwargs = {
                    "model": reflection_lm,
                    "messages": [{"role": "user", "content": spec_prompt}],
                }
                if self.reflection_lm_temperature is not None:
                    completion_kwargs["temperature"] = self.reflection_lm_temperature

                response = await acompletion(**completion_kwargs)

                elapsed = time.time() - start_time
                content = response.choices[0].message.content
                # Split by --- and clean up
                specs = [s.strip() for s in content.split("---") if s.strip()]

                # Log timing for diagnostics
                print(f"   ⚡ Spec induction ({len(task_examples)} examples): {elapsed:.2f}s → {len(specs)} specs")

                return specs[:num_specs]

            except Exception as e:
                elapsed = time.time() - start_time
                error_type = type(e).__name__
                error_msg = str(e)
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

    async def _task_runner(self, candidate: Candidate, example_id: str) -> Dict[str, float]:
        """Execute task LLM on a single example."""
        example = self.example_map[example_id].to_payload()

        try:
            from litellm import acompletion

            # Determine temperature: candidate.meta > adapter config > None
            temperature = candidate.meta.get("temperature", self.task_lm_temperature)

            # Build kwargs, only include temperature if not None
            completion_kwargs = {
                "model": self.task_lm,
                "messages": [
                    {"role": "system", "content": candidate.text},
                    {"role": "user", "content": example["input"]},
                ],
            }
            if temperature is not None:
                completion_kwargs["temperature"] = temperature

            # Try with temperature, fall back without it if model doesn't support it
            try:
                response = await acompletion(**completion_kwargs)
            except Exception as e:
                # Some models don't support custom temperature (e.g., o1-preview)
                if "temperature" in str(e).lower() and temperature is not None:
                    # Retry without temperature
                    completion_kwargs.pop("temperature", None)
                    response = await acompletion(**completion_kwargs)
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

    def _build_orchestrator(self, logger: Optional[EventLogger], enable_auto_stop: bool = False, display_progress: bool = True) -> Orchestrator:
        evaluator = AsyncEvaluator(
            cache=self.cache,
            task_runner=self._task_runner,
        )
        orchestrator_logger = logger or build_logger(self.log_dir, "turbo_default", display_progress=display_progress)

        # Create stop governor if auto-stop enabled
        stop_governor = StopGovernor(StopGovernorConfig()) if enable_auto_stop else None

        return Orchestrator(
            config=self.config,
            evaluator=evaluator,
            archive=self.archive,
            sampler=self.sampler,
            mutator=self.mutator,
            cache=self.cache,
            logger=orchestrator_logger,
            token_controller=self.token_controller,
            stop_governor=stop_governor,
            enable_auto_stop=enable_auto_stop,
            show_progress=display_progress,
            example_sampler=self._sample_examples,
        )

    async def optimize_async(
        self,
        seeds: Sequence[str | Candidate],
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        logger: Optional[EventLogger] = None,
        task_lm: Optional[str] = None,  # Accept but ignore (uses heuristic)
        reflection_lm: Optional[str] = None,  # Accept but ignore (uses heuristic)
        enable_auto_stop: bool = False,  # Enable automatic stopping
        optimize_temperature_after_convergence: bool = False,  # Stage temperature optimization
        display_progress: bool = True,  # Show progress charts
    ) -> Dict[str, Any]:
        # Staged temperature optimization: two-phase approach
        if optimize_temperature_after_convergence:
            return await self._optimize_staged_temperature(
                seeds, max_rounds, max_evaluations, logger, enable_auto_stop, display_progress
            )

        # Standard integrated optimization
        orchestrator = self._build_orchestrator(logger, enable_auto_stop=enable_auto_stop, display_progress=display_progress)
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
        logger: Optional[EventLogger],
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

        orchestrator1 = self._build_orchestrator(logger, enable_auto_stop=enable_auto_stop, display_progress=display_progress)
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

        orchestrator2 = self._build_orchestrator(logger, enable_auto_stop=False, display_progress=display_progress)  # Short phase, no auto-stop
        await orchestrator2.run(temp_seeds, max_rounds=phase2_rounds, max_evaluations=phase2_budget)

        phase2_pareto = orchestrator2.archive.pareto_entries()
        print(f"   ✓ Phase 2 complete: {len(phase2_pareto)} best (prompt, temperature) combinations")

        return {
            "pareto": [e.candidate for e in phase2_pareto],
            "pareto_entries": phase2_pareto,
            "qd_elites": orchestrator2.archive.sample_qd(limit=len(phase2_pareto)),
            "phase1_pareto": [e.candidate for e in phase1_pareto],  # Also return phase 1 results
        }

    def optimize(
        self,
        seeds: Sequence[str | Candidate] | None = None,
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        logger: Optional[EventLogger] = None,
        task_lm: Optional[str] = None,  # Accept for API compatibility (uses heuristic)
        reflection_lm: Optional[str] = None,  # Accept for API compatibility (uses heuristic)
        enable_auto_stop: bool = False,  # Enable automatic stopping
        optimize_temperature_after_convergence: bool = False,  # Stage temperature optimization
        display_progress: bool = True,  # Show progress charts
        enable_seed_initialization: bool = False,  # Use PROMPT-MII-style seed generation
        num_generated_seeds: int = 3,  # How many seeds to generate if initializing
    ) -> Dict[str, Any]:
        """
        Optimize prompts using TurboGEPA with heuristic evaluation.

        Parameters:
            seeds: Initial prompt candidates
            max_rounds: Maximum optimization rounds (None = unlimited)
            max_evaluations: Maximum evaluations (None = unlimited)
            logger: Custom event logger
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
        # Handle seed initialization if requested
        if enable_seed_initialization:
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
                    reflection_lm_temperature=self.reflection_lm_temperature,
                )
            )
        elif seeds is None:
            # No seeds provided and initialization disabled - use default
            seeds = ["You are a helpful assistant. Follow the instructions carefully."]

        return asyncio.run(
            self.optimize_async(
                seeds,
                max_rounds=max_rounds,
                max_evaluations=max_evaluations,
                logger=logger,
                task_lm=task_lm,
                reflection_lm=reflection_lm,
                enable_auto_stop=enable_auto_stop,
                optimize_temperature_after_convergence=optimize_temperature_after_convergence,
                display_progress=display_progress,
            )
        )
