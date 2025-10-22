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
from turbo_gepa.config import Config, DEFAULT_CONFIG, adaptive_shards
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate
from turbo_gepa.logging_utils import EventLogger, build_logger
from turbo_gepa.mutator import MutationConfig, Mutator
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.sampler import InstanceSampler
from turbo_gepa.stop_governor import StopGovernor, StopGovernorConfig
from turbo_gepa.token_controller import TokenCostController
from turbo_gepa.user_plugs_in import reflect_lm_call, task_lm_call


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

    This adapter automatically selects optimal ASHA shard configuration based on
    your dataset size, following the "Rule of 15" for statistical reliability.

    Parameters:
        dataset: Sequence of training/validation examples
        config: Configuration object (shards will be auto-selected if using defaults)
        mutation_config: Optional mutation configuration
        cache_dir: Directory for disk cache (default: .turbo_gepa/cache)
        log_dir: Directory for logs (default: .turbo_gepa/logs)
        sampler_seed: Random seed for example sampling
        task_lm: LLM model for task execution (e.g., "openai/gpt-4o-mini")
        reflection_lm: LLM model for reflection (e.g., "openai/gpt-4o")
        task_lm_temperature: Temperature for task LLM (None = use model default)
        reflection_lm_temperature: Temperature for reflection LLM (None = use model default)
        auto_shards: Enable automatic shard selection (default: True)
        shard_strategy: Strategy for auto-sharding - "balanced", "conservative", or "aggressive"

    Example usage::

        # Basic usage with auto-sharding
        adapter = DefaultAdapter(dataset=trainset)
        result = adapter.optimize(
            seeds=["You are a helpful assistant."],
            max_rounds=4,
        )

        # Conservative strategy for important tasks
        adapter = DefaultAdapter(
            dataset=trainset,
            shard_strategy="conservative",
        )

        # Manual sharding (disables auto-sharding)
        config = Config(shards=(0.10, 0.30, 1.0))
        adapter = DefaultAdapter(dataset=trainset, config=config)

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
        auto_shards: bool = True,
        shard_strategy: str = "balanced",
    ) -> None:
        if not dataset:
            raise ValueError("dataset must contain at least one data instance")

        # Apply adaptive sharding if enabled and config uses default shards
        if auto_shards and config.shards == DEFAULT_CONFIG.shards:
            optimal_shards = adaptive_shards(len(dataset), strategy=shard_strategy)
            config = Config(
                **{k: getattr(config, k) for k in config.__dataclass_fields__}
            )
            config.shards = optimal_shards
            print(f"🔧 Auto-selected shards for {len(dataset)} examples: {optimal_shards} (strategy={shard_strategy})")

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

        # Create reflection runner based on whether reflection_lm is provided
        if reflection_lm:
            reflection_runner = self._create_llm_reflection_runner(reflection_lm)
        else:
            reflection_runner = reflect_lm_call  # Use heuristic default

        # Pass temperature support flag to mutator
        self.mutator = Mutator(
            mutation_config
            or MutationConfig(
                amortized_rate=config.amortized_rate,
                reflection_batch_size=config.reflection_batch_size,
                max_mutations=config.max_mutations_per_round,
                max_tokens=config.max_tokens,
            ),
            reflection_runner=reflection_runner,
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

    def _create_llm_reflection_runner(self, reflection_lm: str):
        """Create a reflection runner that calls a real LLM."""
        async def llm_reflection_runner(traces: list, parent_prompt: str, parent_meta: dict = None) -> list[str]:
            if not traces:
                return []

            try:
                from litellm import acompletion

                # Build reflection prompt
                failure_summary = "\n".join([
                    f"- Example {i+1}: quality={t.get('quality', 0):.2f}"
                    for i, t in enumerate(traces[:3])
                ])

                # Include temperature context if available
                temp_context = ""
                if parent_meta and "temperature" in parent_meta:
                    temp = parent_meta["temperature"]
                    temp_context = f" (using temperature={temp})"

                reflection_prompt = f"""You are optimizing a prompt for better performance.

Current prompt{temp_context}:
"{parent_prompt}"

Recent performance on examples:
{failure_summary}

Suggest 2-3 improved versions of this prompt that would perform better. Focus on:
1. Adding helpful instructions or constraints
2. Improving clarity and structure
3. Addressing common failure patterns

Note: Keep suggestions focused on the prompt text. Temperature will be explored separately.

Output format: Return each improved prompt on a new line, separated by "---"."""

                # Build kwargs, only include temperature if not None
                completion_kwargs = {
                    "model": reflection_lm,  # Use full model name
                    "messages": [{"role": "user", "content": reflection_prompt}],
                }
                if self.reflection_lm_temperature is not None:
                    completion_kwargs["temperature"] = self.reflection_lm_temperature

                response = await acompletion(**completion_kwargs)

                content = response.choices[0].message.content
                # Split by --- and clean up
                mutations = [m.strip() for m in content.split("---") if m.strip()]
                return mutations[:3]  # Limit to 3 mutations

            except Exception as e:
                print(f"Warning: LLM reflection failed ({e}), using heuristic fallback")
                return await reflect_lm_call(traces, parent_prompt, parent_meta)

        return llm_reflection_runner

    async def _task_runner(self, candidate: Candidate, example_id: str) -> Dict[str, float]:
        example = self.example_map[example_id].to_payload()

        # Use real LLM if task_lm is provided, otherwise use heuristic
        if self.task_lm:
            try:
                from litellm import acompletion

                # Determine temperature: candidate.meta > adapter config > None
                temperature = candidate.meta.get("temperature", self.task_lm_temperature)

                # Build kwargs, only include temperature if not None
                completion_kwargs = {
                    "model": self.task_lm,  # Use full model name (e.g., "openrouter/x-ai/grok-4-fast")
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

    def _build_orchestrator(self, logger: Optional[EventLogger], enable_auto_stop: bool = False) -> Orchestrator:
        evaluator = AsyncEvaluator(
            cache=self.cache,
            task_runner=self._task_runner,
        )
        orchestrator_logger = logger or build_logger(self.log_dir, "turbo_default")

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
    ) -> Dict[str, Any]:
        # Staged temperature optimization: two-phase approach
        if optimize_temperature_after_convergence:
            return await self._optimize_staged_temperature(
                seeds, max_rounds, max_evaluations, logger, enable_auto_stop
            )

        # Standard integrated optimization
        orchestrator = self._build_orchestrator(logger, enable_auto_stop=enable_auto_stop)
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

        orchestrator1 = self._build_orchestrator(logger, enable_auto_stop=enable_auto_stop)
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

        orchestrator2 = self._build_orchestrator(logger, enable_auto_stop=False)  # Short phase, no auto-stop
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
        seeds: Sequence[str | Candidate],
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        logger: Optional[EventLogger] = None,
        task_lm: Optional[str] = None,  # Accept for API compatibility (uses heuristic)
        reflection_lm: Optional[str] = None,  # Accept for API compatibility (uses heuristic)
        enable_auto_stop: bool = False,  # Enable automatic stopping
        optimize_temperature_after_convergence: bool = False,  # Stage temperature optimization
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
        """
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
            )
        )
