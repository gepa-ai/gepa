"""
Drop-in default adapter for TurboGEPA mirroring the classic GEPA prompt flow.

This adapter expects data instances with `input`, `additional_context`, and
`answer` fields (similar to `gepa.adapters.default_adapter`). It uses LiteLLM
for provider-agnostic LLM calls given model names (e.g., OpenAI, Anthropic,
OpenRouter providers). Pass `task_lm` and `reflection_lm` model IDs to the
constructor; the adapter performs real LLM calls for evaluation and reflection.

Projects that need custom LLM plumbing can fork the adapter or pass model
settings via ModelConfig; DefaultAdapter uses LiteLLM directly by default.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
import tempfile
from typing import Any, Awaitable, Sequence
import os
import uuid

import random
import re
import math

from turbo_gepa.archive import Archive, ArchiveEntry
from turbo_gepa.cache import DiskCache
from turbo_gepa.config import (
    DEFAULT_CONFIG,
    Config,
    adaptive_config,
    recommended_executor_workers,
)
from turbo_gepa.scoring import ScoringFn, SCORE_KEY
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.islands import IslandContext, spawn_islands
from turbo_gepa.logging import ProgressReporter
from turbo_gepa.logging.logger import LogLevel, StdOutLogger
from turbo_gepa.migrations import MigrationBackend
from turbo_gepa.mutator import MutationConfig, Mutator
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.sampler import InstanceSampler
from turbo_gepa.strategies import ReflectionStrategy
from turbo_gepa.utils.litellm_client import configure_litellm_client


def _detect_fd_guard() -> int | None:
    try:
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        return max(32, int(soft * 0.5))
    except Exception:  # pragma: no cover - platform specific
        return None

_AIME_HASHTAG_RE = re.compile(r"###\s*([\-+]?\d+)", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_INTEGER_RE = re.compile(r"(-?\d+)")


def _normalize_numeric_token(token: str | None) -> str | None:
    if not token:
        return None
    stripped = token.strip()
    if not stripped:
        return None
    try:
        return str(int(stripped))
    except ValueError:
        return None


def _extract_numeric_answer(text: str | None, *, preferred_length: int | None = None) -> str | None:
    if not isinstance(text, str):
        return None
    for pattern in (_AIME_HASHTAG_RE, _BOXED_RE):
        matches = list(pattern.finditer(text))
        for matched in reversed(matches):
            normalised = _normalize_numeric_token(matched.group(1))
            if normalised is None:
                continue
            if preferred_length is not None and preferred_length > 0 and len(normalised) != preferred_length:
                continue
            return normalised
    matches = list(_INTEGER_RE.finditer(text))
    if preferred_length is not None and preferred_length > 0:
        for match in reversed(matches):
            normalised = _normalize_numeric_token(match.group(1))
            if normalised is not None and len(normalised) == preferred_length:
                return normalised
    for match in reversed(matches):
        normalised = _normalize_numeric_token(match.group(1))
        if normalised is not None:
            return normalised
    return None


@dataclass(slots=True)
class DefaultDataInst:
    """Minimal data instance for prompt-based tasks."""

    input: str
    answer: str
    additional_context: dict[str, str] | None = None
    id: str | None = None
    difficulty: float | None = None

    def to_payload(self) -> dict[str, Any]:
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
    temperature: float | None = None
    max_tokens: int | None = 24000
    reasoning_effort: str | None = None


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
        auto_config: Enable automatic configuration with principled sharding (default: True)
        scoring_fn: Optional callback accepting a ``ScoringContext`` that returns the
            scalar score to optimize (defaults to ``maximize_metric(\"quality\")``)

    Example usage::

        # Fully automatic configuration (recommended)
        adapter = DefaultAdapter(
            dataset=trainset,
            task_lm="openrouter/google/gemini-flash-1.5",
            reflection_lm="openrouter/google/gemini-flash-1.5"
        )
        result = adapter.optimize(seeds=["You are a helpful assistant."])

        # Manual configuration (disables auto-config)
        config = Config(shards=(0.10, 0.30, 1.0), batch_size=16)
        adapter = DefaultAdapter(
            dataset=trainset,
            task_lm="openrouter/google/gemini-flash-1.5",
            reflection_lm="openrouter/google/gemini-flash-1.5",
            config=config,
            auto_config=False
        )

        # After auto-config, tweak reflection strategies explicitly
        adapter = DefaultAdapter(
            dataset=trainset,
            task_lm="openrouter/google/gemini-flash-1.5",
            reflection_lm="openrouter/google/gemini-flash-1.5",
        )
        adapter.config.reflection_strategy_names = (
            "incremental_reflection",
            "spec_induction",
            "interleaved_thinking",
        )
    """

    def __init__(
        self,
        dataset: Sequence[DefaultDataInst],
        *,
        config: Config = DEFAULT_CONFIG,
        mutation_config: MutationConfig | None = None,
        cache_dir: str | None = None,
        log_dir: str | None = None,
        sampler_seed: int = 42,
        task_lm: str | None = None,
        reflection_lm: str | None = None,
        task_lm_temperature: float | None = None,
        reflection_lm_temperature: float | None = None,
        auto_config: bool = True,
        scoring_fn: ScoringFn | None = None,
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
            config = adaptive_config(len(dataset))

        config = replace(config)
        config.promote_objective = SCORE_KEY
        if scoring_fn is not None:
            config.scoring_fn = scoring_fn
        self.config = config
        self.dataset = list(dataset)

        env_cache = os.getenv("TURBOGEPA_CACHE_PATH")
        env_logs = os.getenv("TURBOGEPA_LOG_PATH")
        env_control = os.getenv("TURBOGEPA_CONTROL_PATH")
        base_cache = cache_dir or env_cache or config.cache_path
        base_logs = log_dir or env_logs or config.log_path
        self.base_cache_dir = os.path.abspath(base_cache)
        self.base_log_dir = os.path.abspath(base_logs)
        Path(self.base_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.base_log_dir).mkdir(parents=True, exist_ok=True)
        control_path = (
            config.control_dir
            if config.control_dir is not None
            else env_control
        )
        self.control_dir = os.path.abspath(control_path) if control_path else None
        if self.control_dir:
            Path(self.control_dir).mkdir(parents=True, exist_ok=True)
        self.cache_namespace = self.config.shared_cache_namespace or "main"
        self.cache = DiskCache(self.base_cache_dir, namespace=self.cache_namespace)
        self.archive = Archive()
        self.log_dir = self.base_log_dir
        self._current_run_token: str | None = None
        self._forced_run_token: str | None = None

        # Configure litellm's httpx client based on eval_concurrency
        configure_litellm_client(self.config.eval_concurrency)
        self._fd_guard_limit = _detect_fd_guard()

        min_level = self._resolve_log_level(self.config.log_level)
        if self.config.enable_debug_log:
            min_level = LogLevel.DEBUG
        self.logger = StdOutLogger(min_level=min_level)
        self._debug_enabled = min_level <= LogLevel.DEBUG
        import asyncio as _asyncio_adapter
        base_limit = int(self.config.llm_connection_limit or self.config.eval_concurrency)
        if self._fd_guard_limit is not None:
            base_limit = min(base_limit, max(4, self._fd_guard_limit // 2))
        # Use separate semaphores for task and reflection calls so evaluation
        # cannot starve mutation LLM calls (and vice versa).
        task_limit = max(1, base_limit)
        reflection_limit = max(1, base_limit // 2)
        self._task_llm_semaphore = _asyncio_adapter.Semaphore(task_limit)
        self._reflection_llm_semaphore = _asyncio_adapter.Semaphore(reflection_limit)

        # Normalise model configuration objects
        if isinstance(task_lm, ModelConfig):
            self.task_model = task_lm
        else:
            assert isinstance(task_lm, str)
            self.task_model = ModelConfig(
                name=task_lm,
                temperature=task_lm_temperature if task_lm_temperature is not None else config.task_lm_temperature,
            )

        if isinstance(reflection_lm, ModelConfig):
            self.reflection_model = reflection_lm
        else:
            assert isinstance(reflection_lm, str)
            self.reflection_model = ModelConfig(
                name=reflection_lm,
                temperature=reflection_lm_temperature
                if reflection_lm_temperature is not None
                else config.reflection_lm_temperature,
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

        # Temperature support will be checked lazily during Phase 2 (temperature optimization)
        # No upfront LLM call needed - we optimize prompts first (Phase 1), then temperature (Phase 2)
        self.temperature_supported = True  # Assume supported, check later if needed
        self._temperature_warned = False

        # Configure reflection/spec strategies
        self._reflection_strategies: tuple[ReflectionStrategy, ...] = tuple(self.config.reflection_strategies or ())
        reflection_strategies = [strategy for strategy in self._reflection_strategies if not strategy.requires_examples]
        spec_strategies = [strategy for strategy in self._reflection_strategies if strategy.requires_examples]

        batch_reflection_runner = self._create_strategy_runner(reflection_strategies)
        spec_induction_runner = self._create_strategy_runner(spec_strategies) if spec_strategies else None

        # Pass temperature support flag to mutator
        self._mutation_config = mutation_config or MutationConfig(
            reflection_batch_size=config.reflection_batch_size,
            max_mutations=config.max_mutations_per_round,
            max_tokens=config.max_tokens,
            objective_key=config.promote_objective,
        )
        self._batch_reflection_runner = batch_reflection_runner
        self._spec_induction_runner = spec_induction_runner
        self.mutator = Mutator(
            self._mutation_config,
            batch_reflection_runner=self._batch_reflection_runner,
            spec_induction_runner=self._spec_induction_runner,
            temperature_mutations_enabled=False,  # Disabled for Phase 1 - only optimize prompt quality
            logger=self.logger,
        )
        self.log_dir = self.base_log_dir

        # Shared HTTP client for all LLM calls (caps sockets, reuses TLS, HTTP/2)
        try:
            import httpx

            max_conn = max(1, int(config.eval_concurrency))
            if self._fd_guard_limit is not None:
                max_conn = min(max_conn, max(4, self._fd_guard_limit // 2))
            limits = httpx.Limits(
                max_connections=max_conn,
                max_keepalive_connections=max_conn,
                keepalive_expiry=15.0,
            )
            timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)
            self._httpx_client = httpx.AsyncClient(http2=True, limits=limits, timeout=timeout)
        except Exception:
            self._httpx_client = None

    @staticmethod
    def _resolve_log_level(level: str) -> LogLevel:
        """Map string-based config log level to LogLevel enum."""
        if isinstance(level, LogLevel):
            return level
        lookup = {
            "debug": LogLevel.DEBUG,
            "info": LogLevel.INFO,
            "warning": LogLevel.WARNING,
            "error": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL,
        }
        return lookup.get(str(level).lower(), LogLevel.WARNING)

    def _log_debug(self, message: str) -> None:
        if self._debug_enabled:
            self.logger.log(message, LogLevel.DEBUG)

    async def _acompletion_with_client(
        self,
        acompletion,
        completion_kwargs: dict,
        timeout: float,
        semaphore=None,
    ):
        """Call litellm.acompletion with our shared httpx client when supported.

        Tries 'httpx_client' then 'client' keyword; falls back to no client if unsupported.
        """
        import asyncio

        async def _invoke(kwargs: dict) -> Any:
            if semaphore is None:
                return await asyncio.wait_for(acompletion(**kwargs), timeout=timeout)
            async with semaphore:
                return await asyncio.wait_for(acompletion(**kwargs), timeout=timeout)

        return await _invoke(completion_kwargs)

    def _disable_temperature_support(self, context: str | None = None) -> None:
        """Disable temperature tuning after a model rejects the parameter."""
        if not self.temperature_supported:
            return
        self.temperature_supported = False
        self.task_model.temperature = None
        self.reflection_model.temperature = None
        self.mutator.set_temperature_mutations_enabled(False)
        if not self._temperature_warned:
            reason = context if context else "model rejected temperature parameter"
            self.logger.log(f"⚠️  Disabling temperature optimization: {reason}", LogLevel.WARNING)
            self._temperature_warned = True

    def _check_temperature_support(self, model: str, test_temp: float) -> bool:
        """Quick test to see if model supports custom temperature.

        Uses litellm for provider-agnostic testing (OpenAI, Anthropic, etc.)
        """
        try:
            import litellm

            # Quick test call with minimal tokens
            litellm.completion(
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

    def _make_metrics_mapper(self) -> Callable[[dict[str, float]], dict[str, float]]:
        objective = self.config.promote_objective or "quality"

        def mapper(metrics: dict[str, float]) -> dict[str, float]:
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

        return mapper


    def _create_strategy_runner(
        self,
        strategies: Sequence[ReflectionStrategy],
    ):
        """Return an async runner that iterates through configured strategies."""

        if not strategies:
            return None

        async def run_strategies(
            parent_contexts: list[dict[str, object]],
            num_mutations: int,
            task_examples: list[dict[str, object]] | None = None,
        ) -> list[str]:
            if num_mutations <= 0:
                return []

            import time

            collected: list[str] = []

            for strategy in strategies:
                if len(collected) >= num_mutations:
                    break
                if strategy.requires_examples and not task_examples:
                    continue

                strategy_label = getattr(strategy, "name", strategy.__class__.__name__)

                remaining = num_mutations - len(collected)
                reflection_examples = (
                    getattr(self.mutator, "_reflection_examples", []) if hasattr(self, "mutator") else []
                )
                user_prompt = strategy.user_prompt_builder(
                    parent_contexts,
                    reflection_examples,
                    task_examples or [],
                    remaining,
                )
                messages = [
                    {"role": "system", "content": strategy.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                completion_kwargs: dict[str, Any] = {
                    "model": self.reflection_model.name,
                    "messages": messages,
                }
                if self.reflection_model.max_tokens is not None:
                    completion_kwargs["max_tokens"] = self.reflection_model.max_tokens
                if self.temperature_supported and self.reflection_model.temperature is not None:
                    completion_kwargs["temperature"] = self.reflection_model.temperature
                if self.reflection_model.reasoning_effort is not None:
                    completion_kwargs["reasoning_effort"] = self.reflection_model.reasoning_effort

                start_time = time.time()
                try:
                    from litellm import acompletion

                    import asyncio

                    async def invoke_reflection():
                        return await self._acompletion_with_client(
                            acompletion,
                            completion_kwargs,
                            180.0,
                            semaphore=self._reflection_llm_semaphore,
                        )

                    try:
                        response = await self._call_with_retries(
                            invoke_reflection,
                            label=f"reflection:{strategy_label}",
                        )
                    except asyncio.TimeoutError:
                        raise
                    except Exception as err:
                        if "temperature" in str(err).lower() and completion_kwargs.pop("temperature", None) is not None:
                            self._disable_temperature_support(
                                f"{self.reflection_model.name} rejected temperature parameter"
                            )

                            async def invoke_no_temp():
                                return await self._acompletion_with_client(
                                    acompletion,
                                    completion_kwargs,
                                    180.0,
                                    semaphore=self._reflection_llm_semaphore,
                                )

                            response = await self._call_with_retries(
                                invoke_no_temp,
                                label=f"reflection:{strategy_label}",
                                max_attempts=2,
                            )
                        else:
                            raise
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    self.logger.log(
                        f"❌ Strategy '{strategy_label}' LLM call TIMEOUT after {elapsed:.1f}s "
                        f"(model={self.reflection_model.name})",
                        LogLevel.ERROR,
                    )
                    raise RuntimeError(
                        f"Strategy '{strategy_label}' LLM call timed out after {elapsed:.1f}s."
                    ) from None
                except Exception as exc:
                    elapsed = time.time() - start_time
                    self.logger.log(
                        f"❌ Strategy '{strategy_label}' LLM call FAILED after {elapsed:.1f}s "
                        f"(model={self.reflection_model.name}): {type(exc).__name__}: {exc}",
                        LogLevel.ERROR,
                    )
                    raise RuntimeError(
                        f"Strategy '{strategy_label}' LLM call failed after {elapsed:.1f}s "
                        f"({type(exc).__name__}: {exc})."
                    ) from exc

                content = response.choices[0].message.content
                # Record per-strategy call in metrics (if available)
                try:
                    if hasattr(self, "_metrics") and self._metrics is not None:
                        elapsed = time.time() - start_time
                        self._metrics.record_strategy_call(strategy_label, elapsed)
                except Exception:
                    pass
                parsed = strategy.response_parser(content or "")
                cleaned = self._filter_strategy_mutations(parsed, strategy_name=strategy_label)
                collected.extend(cleaned)

            return collected[:num_mutations]

        return run_strategies

    def _filter_strategy_mutations(self, raw_texts: Sequence[str], *, strategy_name: str) -> list[str]:
        """Validate and clean raw prompt outputs from a strategy."""

        cleaned_mutations: list[str] = []
        for idx, raw in enumerate(raw_texts, 1):
            cleaned = raw.strip()
            if not cleaned:
                continue
            if len(cleaned) < 50:
                self._log_debug(
                    f"   ⚠️ Skipping {strategy_name} mutation {idx}: Too short ({len(cleaned)} chars)"
                )
                continue
            if cleaned.startswith("###"):
                self._log_debug(
                    f"   ⚠️ Skipping {strategy_name} mutation {idx}: Looks like an answer, not a prompt"
                )
                continue
            stripped = cleaned.replace("#", "").replace(" ", "")
            if len(cleaned) < 100 and stripped.isdigit():
                self._log_debug(
                    f"   ⚠️ Skipping {strategy_name} mutation {idx}: Appears to be numeric output"
                )
                continue
            cleaned_mutations.append(cleaned)

        if not cleaned_mutations and raw_texts:
            self.logger.log(
                f"⚠️  Strategy '{strategy_name}' produced no usable prompts. Check reflection output.",
                LogLevel.WARNING,
            )

        return cleaned_mutations

    def _sample_examples(self, num_examples: int) -> list[dict]:
        """Sample random examples for spec induction."""
        import random

        example_ids = list(self.example_map.keys())
        if len(example_ids) <= num_examples:
            sampled_ids = example_ids
        else:
            sampled_ids = random.sample(example_ids, num_examples)

        return [self.example_map[eid].to_payload() for eid in sampled_ids]

    def _normalize_seeds(self, seeds: Sequence[str | Candidate], *, source: str) -> list[Candidate]:
        normalized: list[Candidate] = []
        for seed in seeds:
            if isinstance(seed, Candidate):
                meta = dict(seed.meta)
                if not self.temperature_supported:
                    meta.pop("temperature", None)
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
        return Archive()

    def _make_sampler(self, *, seed_offset: int = 0) -> InstanceSampler:
        return InstanceSampler(self._example_ids, seed=self._sampler_seed + seed_offset)

    def _assigned_worker_islands(self) -> list[int] | None:
        cfg = self.config
        if cfg.worker_id is None or not cfg.worker_count or cfg.worker_count <= 0:
            return None
        total_islands = max(1, cfg.n_islands)
        per_worker = cfg.islands_per_worker or math.ceil(total_islands / max(1, cfg.worker_count))
        if per_worker <= 0:
            per_worker = 1
        start = max(0, int(cfg.worker_id) * per_worker)
        if start >= total_islands:
            return []
        end = min(total_islands, start + per_worker)
        return list(range(start, end))

    def _make_cache(self, island_id: int | None = None) -> DiskCache:
        if island_id is None:
            return DiskCache(self.base_cache_dir, namespace=self.cache_namespace)
        path = Path(self.base_cache_dir)
        island_path = path / f"island_{island_id}"
        island_path.mkdir(parents=True, exist_ok=True)
        ns = f"{self.cache_namespace}_island_{island_id}"
        return DiskCache(str(island_path), namespace=ns)

    def _make_log_dir(self, island_id: int | None = None) -> str:
        path = Path(self.base_log_dir)
        if island_id is None:
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        island_path = path / f"island_{island_id}"
        island_path.mkdir(parents=True, exist_ok=True)
        return str(island_path)

    def _combine_evolution_snapshots(self, snapshots: Sequence[dict[str, Any]]) -> dict[str, Any]:
        combined = {
            "mutations_requested": 0,
            "mutations_generated": 0,
            "mutations_enqueued": 0,
            "mutations_promoted": 0,
            "unique_parents": 0,
            "unique_children": 0,
            "evolution_edges": 0,
            "total_evaluations": 0,
            "islands": [],
        }
        parent_children: defaultdict[str, set[str]] = defaultdict(set)
        promoted_children: set[str] = set()
        all_children: set[str] = set()
        promoted_total = 0

        for snapshot in snapshots:
            if not snapshot:
                continue
            combined["mutations_requested"] += snapshot.get("mutations_requested", 0)
            combined["mutations_generated"] += snapshot.get("mutations_generated", 0)
            if "strategy_stats" in snapshot:
                combined.setdefault("strategy_stats", {}).setdefault("islands", []).append(snapshot["strategy_stats"])
            combined["mutations_enqueued"] += snapshot.get("mutations_enqueued", 0)
            combined["total_evaluations"] += snapshot.get("total_evaluations", 0)
            promoted_total += snapshot.get("mutations_promoted", 0)

            detail_sources = snapshot.get("islands")
            if detail_sources:
                combined["islands"].extend(detail_sources)
            else:
                combined["islands"].append(snapshot)

            for detail in detail_sources or [snapshot]:
                parent_map = detail.get("parent_children") or {}
                for parent, children in parent_map.items():
                    parent_children[parent].update(children)
                promoted_children.update(detail.get("promoted_children") or [])
                all_children.update(detail.get("children") or [])

        if parent_children:
            combined["unique_parents"] = len(parent_children)
            combined["evolution_edges"] = sum(len(children) for children in parent_children.values())
        if all_children:
            combined["unique_children"] = len(all_children)
        if promoted_children:
            combined["mutations_promoted"] = len(promoted_children)
        else:
            combined["mutations_promoted"] = promoted_total

        return combined

    def _aggregate_evolution_stats(self, orchestrators: Sequence[Orchestrator | None]) -> dict[str, Any]:
        snapshots: list[dict[str, Any]] = []
        for orchestrator in orchestrators:
            if orchestrator is None:
                continue
            snapshots.append(orchestrator.evolution_snapshot(include_edges=True))
        return self._combine_evolution_snapshots(snapshots)

    async def _summarize_island_runs(
        self, orchestrators: Sequence[Orchestrator | None]
    ) -> dict[str, Any]:
        combined_archive = self._make_archive()
        inserts: list[tuple[Candidate, EvalResult]] = []
        for orchestrator in orchestrators:
            if orchestrator is None:
                continue
            for entry in orchestrator.archive.pareto_entries():
                inserts.append((entry.candidate, entry.result))
        if inserts:
            await combined_archive.batch_insert(inserts)
        pareto_entries = combined_archive.pareto_entries()
        pareto_candidates = [entry.candidate for entry in pareto_entries]
        evolution_stats = self._aggregate_evolution_stats(orchestrators)
        total_candidates = len(combined_archive.pareto)
        island_metadata = [
            self._build_run_metadata(orchestrator, orchestrator.archive.pareto_entries())
            for orchestrator in orchestrators
            if orchestrator is not None
        ]
        metrics_per_island = [
            meta.get("metrics") if isinstance(meta, dict) else None for meta in island_metadata
        ]
        best_meta: dict[str, Any] | None = None
        best_quality = float("-inf")
        for meta in island_metadata:
            if not isinstance(meta, dict):
                continue
            quality = meta.get("best_quality")
            if isinstance(quality, (int, float)) and quality > best_quality:
                best_quality = quality
                best_meta = meta
        if best_meta is None and island_metadata:
            best_meta = next((m for m in island_metadata if isinstance(m, dict)), None)

        result: dict[str, Any] = {
            "pareto": pareto_candidates,
            "pareto_entries": pareto_entries,
            "qd_elites": [],
            "evolution_stats": evolution_stats,
            "total_candidates": total_candidates,
            "run_metadata_per_island": island_metadata,
            "metrics_per_island": metrics_per_island,
            "metrics": {"per_island": metrics_per_island},
        }
        if best_meta:
            result["run_metadata"] = best_meta
            best_metrics = best_meta.get("metrics") if isinstance(best_meta, dict) else None
            if isinstance(best_metrics, dict):
                result["metrics"]["best"] = best_metrics
        else:
            result["run_metadata"] = {}
        return result

    def _build_run_metadata(
        self,
        orchestrator: Orchestrator,
        pareto_entries: Sequence[ArchiveEntry],
    ) -> dict[str, Any]:
        promote_obj = orchestrator.config.promote_objective
        best_quality = 0.0
        best_shard = None
        best_prompt = None

        def _pack_candidate(candidate: Candidate | None, result: EvalResult | None) -> dict[str, Any] | None:
            if candidate is None or result is None:
                return None
            shard = result.shard_fraction or 0.0
            quality = result.objectives.get(promote_obj, 0.0)
            meta = candidate.meta if isinstance(candidate.meta, dict) else {}
            is_seed = meta.get("source") == "seed"
            return {
                "candidate": candidate,
                "quality": quality,
                "shard": shard,
                "is_seed": is_seed,
            }

        def _pick_best(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
            if not entries:
                return None
            non_seed = [entry for entry in entries if not entry.get("is_seed")]
            pool = non_seed or entries
            return max(pool, key=lambda entry: (entry.get("shard", 0.0), entry.get("quality", 0.0)))

        # Prefer the exact candidate that achieved the target on the final rung,
        # captured at the moment of attainment (it may no longer be on Pareto).
        try:
            star_prompt = getattr(orchestrator, "_north_star_prompt", None)
            star_quality = getattr(orchestrator, "_north_star_quality", None)
            star_shard = getattr(orchestrator, "_north_star_shard", None)
            if isinstance(star_prompt, str) and isinstance(star_quality, (int, float)) and isinstance(star_shard, (int, float)):
                best_prompt = star_prompt
                best_quality = float(star_quality)
                best_shard = float(star_shard)
        except Exception:
            pass

        # Fallback #1: scan orchestrator.latest_results for any full-shard results
        # and recover the corresponding prompt via orchestrator's candidate map.
        if best_prompt is None:
            try:
                entries: list[dict[str, Any]] = []
                cmap = getattr(orchestrator, "_candidates_by_fp", {})
                for fp, res in getattr(orchestrator, "latest_results", {}).items():
                    entry = _pack_candidate(cmap.get(fp), res)
                    if entry:
                        entries.append(entry)
                best_entry = _pick_best(entries)
                if best_entry:
                    best_prompt = best_entry["candidate"].text
                    best_quality = best_entry["quality"]
                    best_shard = best_entry["shard"]
            except Exception:
                pass

        # Fallback #2: current Pareto entries, preferring deepest shard even if partial
        if best_prompt is None:
            entries: list[dict[str, Any]] = []
            for entry in pareto_entries:
                packed = _pack_candidate(entry.candidate, entry.result)
                if packed:
                    entries.append(packed)
            best_entry = _pick_best(entries)
            if best_entry:
                best_prompt = best_entry["candidate"].text
                best_quality = best_entry["quality"]
                best_shard = best_entry["shard"]
        metrics_snapshot = orchestrator.metrics_snapshot()
        metadata: dict[str, Any] = {
            "run_id": orchestrator.run_id,
            "stop_reason": orchestrator.stop_reason,
            "evaluations": orchestrator.evaluations_run,
            "best_quality": best_quality,
            "best_quality_shard": best_shard,
            "best_prompt": best_prompt,
        }
        if metrics_snapshot:
            metadata["metrics"] = metrics_snapshot
            if metrics_snapshot.get("time_to_target_seconds") is not None:
                metadata["time_to_target_seconds"] = metrics_snapshot["time_to_target_seconds"]
            if metrics_snapshot.get("turbo_score") is not None:
                metadata["turbo_score"] = metrics_snapshot["turbo_score"]
            if metrics_snapshot.get("promotions_by_rung"):
                metadata["promotions_by_rung"] = metrics_snapshot["promotions_by_rung"]
        return metadata

    async def evaluate_prompt_async(
        self,
        prompt: str | Candidate,
        *,
        example_ids: Sequence[str] | None = None,
        concurrency: int | None = None,
        bypass_cache: bool = True,
        show_progress: bool = True,
        label: str = "verification",
        cache_dir: str | None = None,
    ) -> EvalResult:
        """
        Evaluate a single prompt on the adapter dataset without running optimization.

        Returns the aggregated EvalResult covering all requested examples.
        """
        target_ids = list(example_ids) if example_ids is not None else list(self._example_ids)
        if not target_ids:
            raise ValueError("No example IDs supplied for verification.")

        if isinstance(prompt, Candidate):
            candidate = prompt
            if "source" not in candidate.meta and label:
                candidate = candidate.with_meta(source=label)
        else:
            candidate = Candidate(text=prompt, meta={"source": label})

        eff_concurrency = max(1, concurrency or self.config.eval_concurrency)

        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        cache = self.cache
        if bypass_cache:
            if cache_dir:
                verify_cache = Path(cache_dir)
                shutil.rmtree(verify_cache, ignore_errors=True)
                verify_cache.mkdir(parents=True, exist_ok=True)
            else:
                base_cache = Path(self.base_cache_dir)
                base_cache.mkdir(parents=True, exist_ok=True)
                temp_dir = tempfile.TemporaryDirectory(prefix="verify_", dir=str(base_cache))
                verify_cache = Path(temp_dir.name)
            cache = DiskCache(str(verify_cache))

        mapper = self._make_metrics_mapper()
        evaluator = AsyncEvaluator(
            cache=cache,
            task_runner=self._task_runner,
            metrics_mapper=mapper,
            timeout_seconds=self.config.eval_timeout_seconds,
            skip_final_straggler_cutoff=False,
            logger=self.logger,
            promote_objective=self.config.promote_objective,
            cancel_stragglers_immediately=self.config.cancel_stragglers_immediately,
            replay_stragglers=self.config.replay_stragglers,
            min_samples_for_confidence=self.config.min_samples_for_confidence,
            target_quality=self.config.target_quality,
            confidence_z=self.config.confidence_z,
        )

        try:
            return await evaluator.eval_on_shard(
                candidate,
                target_ids,
                eff_concurrency,
                shard_fraction=1.0,
                show_progress=show_progress,
                is_final_shard=True,
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def evaluate_prompt(
        self,
        prompt: str | Candidate,
        *,
        example_ids: Sequence[str] | None = None,
        concurrency: int | None = None,
        bypass_cache: bool = True,
        show_progress: bool = True,
        label: str = "verification",
        cache_dir: str | None = None,
    ) -> EvalResult:
        """Sync wrapper around evaluate_prompt_async for convenience."""
        return asyncio.run(
            self.evaluate_prompt_async(
                prompt,
                example_ids=example_ids,
                concurrency=concurrency,
                bypass_cache=bypass_cache,
                show_progress=show_progress,
                label=label,
                cache_dir=cache_dir,
            )
        )

    async def _call_with_retries(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        label: str,
        max_attempts: int = 3,
        base_delay: float = 1.5,
    ):
        """Execute an async factory with exponential backoff."""
        import asyncio

        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await coro_factory()
            except Exception as exc:
                last_exc = exc
                transient = isinstance(exc, RuntimeError) or "OpenrouterException" in str(exc)
                critical = any(keyword in str(exc).lower() for keyword in ("too many open files", "ssl", "timeout"))
                if not transient or critical:
                    raise
                if attempt >= max_attempts:
                    raise
                delay = base_delay * (1.5 ** (attempt - 1))
                delay = min(3.0, delay)
                delay += random.uniform(0.0, 0.3)
                self.logger.log(
                    f"⚠️  LLM call '{label}' failed (attempt {attempt}/{max_attempts}): {exc}. "
                    f"Retrying in {delay:.1f}s",
                    LogLevel.WARNING,
                )
                await asyncio.sleep(delay)
        if last_exc:
            raise last_exc

    async def _task_runner(self, candidate: Candidate, example_id: str) -> dict[str, float]:
        """Execute task LLM on a single example."""
        example = self.example_map[example_id].to_payload()

        try:
            from litellm import acompletion

            system_prompt = (candidate.text or "").rstrip()
            completion_kwargs: dict[str, Any] = {
                "model": self.task_model.name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["input"]},
                ],
            }
            # Let the model decide output length by default (no forced max_tokens).
            # Respect a user-provided cap only when explicitly configured.
            if self.task_model.max_tokens is not None:
                try:
                    completion_kwargs["max_tokens"] = int(self.task_model.max_tokens)
                except Exception:
                    pass

            if self.temperature_supported:
                candidate_temperature = candidate.meta.get("temperature")
                if candidate_temperature is not None:
                    completion_kwargs["temperature"] = candidate_temperature
                elif self.task_model.temperature is not None:
                    completion_kwargs["temperature"] = self.task_model.temperature

            reasoning_effort = candidate.meta.get("reasoning_effort", self.task_model.reasoning_effort)
            if reasoning_effort is not None and not self.task_model.name.endswith("gpt-oss-120b:nitro"):
                completion_kwargs["reasoning_effort"] = reasoning_effort

            import time as _time_module
            import asyncio

            _start_llm = _time_module.time()

            async def invoke():
                return await self._acompletion_with_client(
                    acompletion,
                    completion_kwargs,
                    120.0,
                    semaphore=self._task_llm_semaphore,
                )

            try:
                response = await self._call_with_retries(invoke, label=f"task:{example_id}")
            except asyncio.TimeoutError:
                _elapsed_llm = _time_module.time() - _start_llm
                if hasattr(self, "_metrics") and self._metrics is not None:
                    self._metrics.llm_timeouts += 1
                self.logger.log(
                    f"❌ Task LLM call TIMEOUT after {_elapsed_llm:.1f}s "
                    f"(example={example_id}, model={self.task_model.name}). "
                    "This may indicate API rate limits or a very long response.",
                    LogLevel.ERROR,
                )
                raise RuntimeError(
                    f"Task LLM call timed out after {_elapsed_llm:.1f}s for example {example_id}. "
                    "This may indicate API rate limits or a very long response. Consider using a faster model."
                )
            except Exception as e:
                if "temperature" in str(e).lower() and completion_kwargs.get("temperature") is not None:
                    self._disable_temperature_support(f"{self.task_model.name} rejected temperature parameter")
                    completion_kwargs.pop("temperature", None)
                    if isinstance(candidate.meta, dict):
                        candidate.meta.pop("temperature", None)

                    async def invoke_no_temp():
                        return await self._acompletion_with_client(
                            acompletion,
                            completion_kwargs,
                            120.0,
                            semaphore=self._task_llm_semaphore,
                        )

                    response = await self._call_with_retries(
                        invoke_no_temp,
                        label=f"task:{example_id}",
                        max_attempts=2,
                    )
                else:
                    raise

            _elapsed_llm = _time_module.time() - _start_llm

            if hasattr(self, "_metrics") and self._metrics is not None:
                self._metrics.record_llm_call("task", _elapsed_llm)

            if _elapsed_llm > 60.0:
                self.logger.log(
                    f"⚠️  Slow task LLM call: {_elapsed_llm:.1f}s (example={example_id}, model={self.task_model.name})",
                    LogLevel.WARNING,
                )

            model_output = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            answer_field = example.get("answer")
            expected_token = _extract_numeric_answer(answer_field)
            preferred_len = len(expected_token) if expected_token is not None else None
            actual_token = _extract_numeric_answer(model_output, preferred_length=preferred_len)

            if expected_token is not None:
                if actual_token is not None:
                    quality = 1.0 if expected_token == actual_token else 0.0
                else:
                    digits_only = "".join(ch for ch in (model_output or "") if ch.isdigit())
                    quality = 1.0 if expected_token and expected_token in digits_only else 0.0
            else:
                haystack = (model_output or "").lower()
                needle = str(answer_field or "").lower()
                quality = 1.0 if needle and needle in haystack else 0.0

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
            error_type = type(e).__name__
            error_msg = str(e)
            self.logger.log(
                f"❌ Task LLM call FAILED (example={example_id}, model={self.task_model.name}): "
                f"{error_type}: {error_msg}",
                LogLevel.ERROR,
            )
            raise RuntimeError(
                f"Task LLM call failed ({error_type}: {error_msg}). "
                "Check your API key, model name, and network connection."
            ) from e

    def _build_orchestrator(
        self,
        *,
        display_progress: bool = True,
        temperature_mutations_enabled: bool | None = None,
        island_context: IslandContext | None = None,
        cache: DiskCache | None = None,
        archive: Archive | None = None,
        sampler: InstanceSampler | None = None,
        mutator: Mutator | None = None,
        log_dir: str | None = None,
        metrics_callback: Callable | None = None,
        island_id: int | None = None,
        migration_backend: MigrationBackend | None = None,
        control_dir: str | None = None,
    ) -> Orchestrator:
        target_mutator = mutator or self.mutator
        if temperature_mutations_enabled and not self.temperature_supported:
            if not self._temperature_warned:
                self.logger.log(
                    "⚠️  Temperature mutations requested but disabled due to unsupported model.",
                    LogLevel.WARNING,
                )
            temperature_mutations_enabled = False
        if temperature_mutations_enabled is not None:
            target_mutator.set_temperature_mutations_enabled(temperature_mutations_enabled)
        mutator = target_mutator

        # Only optimize for quality, ignore token cost
        metrics_mapper = self._make_metrics_mapper()

        progress_callback = metrics_callback
        if display_progress and progress_callback is None:
            progress_callback = ProgressReporter(self.logger)

        evaluator = AsyncEvaluator(
            cache=cache or self.cache,
            task_runner=self._task_runner,
            metrics_mapper=metrics_mapper,
            timeout_seconds=self.config.eval_timeout_seconds,
            min_improve=0.0,  # Disabled: variance-aware promotion handles this
            skip_final_straggler_cutoff=False,
            logger=self.logger,
            promote_objective=self.config.promote_objective,
            cancel_stragglers_immediately=self.config.cancel_stragglers_immediately,
            replay_stragglers=self.config.replay_stragglers,
            min_samples_for_confidence=self.config.min_samples_for_confidence,
            target_quality=self.config.target_quality,
        )

        base_run_id = getattr(self, "_current_run_token", None) or uuid.uuid4().hex[:8]
        suffix = None
        if island_context is not None:
            suffix = f"island{island_context.island_id}"
        elif island_id is not None:
            suffix = f"island{island_id}"
        if suffix is None:
            suffix = "solo"
        run_id = f"{base_run_id}-{suffix}"

        return Orchestrator(
            config=self.config,
            evaluator=evaluator,
            archive=archive or self.archive,
            sampler=sampler or self.sampler,
            mutator=mutator,
            cache=cache or self.cache,
            show_progress=display_progress,
            example_sampler=self._sample_examples,
            island_context=island_context,
            island_id=island_id,
            migration_backend=migration_backend,
            control_dir=control_dir or self.control_dir,
            metrics_callback=progress_callback,
            logger=self.logger,
            run_id=run_id,
        )

    async def optimize_async(
        self,
        seeds: Sequence[str | Candidate],
        *,
        max_rounds: int | None = None,
        max_evaluations: int | None = None,
        task_lm: str | None = None,  # Kept for API compatibility; models come from adapter init
        reflection_lm: str | None = None,  # Kept for API compatibility; models come from adapter init
        optimize_temperature_after_convergence: bool = False,  # Stage temperature optimization
        display_progress: bool = True,  # Show progress charts
        enable_auto_stop: bool = True,
        metrics_callback: Callable | None = None,  # Callback for dashboard updates
    ) -> dict[str, Any]:
        run_token = self._forced_run_token or os.getenv("TURBOGEPA_RUN_ID") or uuid.uuid4().hex[:8]
        self._current_run_token = run_token
        if self.config.n_islands > 1 and not self.control_dir:
            base_cache = Path(self.base_cache_dir).resolve()
            auto_control = base_cache.parent / "control" / run_token
            auto_control.mkdir(parents=True, exist_ok=True)
            self.control_dir = str(auto_control)
            self.config.control_dir = str(auto_control)
        worker_islands = self._assigned_worker_islands()
        if worker_islands is not None:
            if not getattr(self.config, "migration_backend", None):
                self.config.migration_backend = "volume"
            if not getattr(self.config, "migration_path", None):
                self.config.migration_path = os.path.join(self.base_cache_dir, "migrations")
            return await self._optimize_distributed_islands(
                seeds,
                island_ids=worker_islands,
                max_rounds=max_rounds,
                max_evaluations=max_evaluations,
                display_progress=display_progress,
            )
        if self.config.n_islands > 1:
            if optimize_temperature_after_convergence:
                return await self._optimize_multi_island_staged(
                    seeds,
                    max_rounds=max_rounds,
                    max_evaluations=max_evaluations,
                    display_progress=display_progress,
                )
            return await self._optimize_multi_island(
                seeds,
                max_rounds=max_rounds,
                max_evaluations=max_evaluations,
                display_progress=display_progress,
            )

        # Staged temperature optimization: two-phase approach
        if optimize_temperature_after_convergence and self.temperature_supported:
            return await self._optimize_staged_temperature(
                seeds, max_rounds, max_evaluations, enable_auto_stop, display_progress
            )

        # Standard integrated optimization
        orchestrator = self._build_orchestrator(
            display_progress=display_progress,
            metrics_callback=metrics_callback,
            control_dir=self.control_dir,
        )
        # Store metrics reference in adapter for LLM call tracking
        self._metrics = orchestrator.metrics
        # Also pass metrics to mutator for tracking mutation LLM calls
        self.mutator._metrics = orchestrator.metrics

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

        # Apply global timeout - but don't use asyncio.wait_for() because that cancels tasks
        # Instead, let the orchestrator's own timeout logic handle it gracefully
        try:
            await orchestrator.run(seed_candidates, max_rounds=max_rounds, max_evaluations=max_evaluations)
        finally:
            orchestrator.finalize_control()

        pareto_entries = orchestrator.archive.pareto_entries()
        pareto = [entry.candidate for entry in pareto_entries]
        return {
            "pareto": pareto,
            "pareto_entries": pareto_entries,
            "qd_elites": [],  # Deprecated field kept for compatibility; always empty
            "evolution_stats": orchestrator.evolution_snapshot(include_edges=True),
            "lineage": orchestrator.get_candidate_lineage_data(),
            "run_metadata": self._build_run_metadata(orchestrator, pareto_entries),
            "metrics": orchestrator.metrics_snapshot(),
        }

    async def _optimize_staged_temperature(
        self,
        seeds: Sequence[str | Candidate],
        max_rounds: int | None,
        max_evaluations: int | None,
        enable_auto_stop: bool = True,
        display_progress: bool = True,
    ) -> dict[str, Any]:
        """Two-phase optimization: prompts first, then temperature.

        Phase 1: Optimize prompts WITHOUT temperature (seeds have no temperature)
        Phase 2: Take top-K prompts and create temperature seeds for final optimization

        This approach avoids variance issues by separating "what to say" from
        "how stochastic to be".
        """

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
            display_progress=display_progress,
            temperature_mutations_enabled=False,
            control_dir=self.control_dir,
        )
        # Pass metrics to mutator for tracking mutation LLM calls
        self.mutator._metrics = orchestrator1.metrics
        try:
            await orchestrator1.run(phase1_seeds, max_rounds=phase1_rounds, max_evaluations=phase1_budget)
        finally:
            orchestrator1.finalize_control()

        phase1_pareto = orchestrator1.archive.pareto_entries()
        phase1_stats = orchestrator1.evolution_snapshot(include_edges=True)
        phase1_metadata = self._build_run_metadata(orchestrator1, phase1_pareto)

        # Early exit if temperature not supported
        if not self.temperature_supported:
            return {
                "pareto": [e.candidate for e in phase1_pareto],
                "pareto_entries": phase1_pareto,
                "qd_elites": [],  # Deprecated
                "phase1_pareto": [e.candidate for e in phase1_pareto],
                "phase1_evolution_stats": phase1_stats,
                "evolution_stats": self._combine_evolution_snapshots([phase1_stats]),
                "run_metadata": phase1_metadata,
                "metrics": phase1_metadata.get("metrics"),
            }

        # Early exit if no pareto frontier
        if not phase1_pareto:
            return {
                "pareto": [],
                "pareto_entries": [],
                "qd_elites": [],  # Deprecated
                "phase1_pareto": [],
                "phase1_evolution_stats": phase1_stats,
                "evolution_stats": self._combine_evolution_snapshots([phase1_stats]),
                "run_metadata": phase1_metadata,
                "metrics": phase1_metadata.get("metrics"),
            }

        # Take top K prompts sorted by quality
        top_k = min(5, len(phase1_pareto))
        top_entries = sorted(
            phase1_pareto, key=lambda e: e.result.objectives.get(self.config.promote_objective, 0.0), reverse=True
        )[:top_k]

        # Create temperature-enabled seeds from top prompts
        # Now mutator WILL create temperature variants (because seeds have temperature)
        temp_seeds = []
        for entry in top_entries:
            # Start with mid-range temperature
            meta = dict(entry.candidate.meta, temperature=0.5, source="phase2_seed")
            temp_seeds.append(Candidate(text=entry.candidate.text, meta=meta))

        # Run Phase 2 optimization (30% of remaining budget, single round)
        # Safety: Always enforce round limit to prevent infinite loops
        max_phase2_rounds = 1  # Single round of temperature exploration
        phase2_budget = int(max_evaluations * 0.3) if max_evaluations else None
        phase2_rounds = min(max_phase2_rounds, max_rounds) if max_rounds else max_phase2_rounds

        orchestrator2 = self._build_orchestrator(
            display_progress=display_progress,
            temperature_mutations_enabled=True,
            control_dir=self.control_dir,
        )
        # Pass metrics to mutator for tracking mutation LLM calls
        self.mutator._metrics = orchestrator2.metrics
        try:
            await orchestrator2.run(temp_seeds, max_rounds=phase2_rounds, max_evaluations=phase2_budget)
        finally:
            orchestrator2.finalize_control()

        phase2_pareto = orchestrator2.archive.pareto_entries()
        phase2_stats = orchestrator2.evolution_snapshot(include_edges=True)
        phase2_metadata = self._build_run_metadata(orchestrator2, phase2_pareto)

        combined_stats = self._combine_evolution_snapshots([phase1_stats, phase2_stats])

        return {
            "pareto": [e.candidate for e in phase2_pareto],
            "pareto_entries": phase2_pareto,
            "qd_elites": [],  # Deprecated
            "phase1_pareto": [e.candidate for e in phase1_pareto],  # Also return phase 1 results
            "phase1_evolution_stats": phase1_stats,
            "phase2_evolution_stats": phase2_stats,
            "evolution_stats": combined_stats,
            "run_metadata": phase2_metadata,
            "phase1_run_metadata": phase1_metadata,
            "metrics": phase2_metadata.get("metrics"),
            "phase1_metrics": phase1_metadata.get("metrics"),
            "phase2_metrics": phase2_metadata.get("metrics"),
        }

    async def _optimize_multi_island(
        self,
        seeds: Sequence[str | Candidate],
        *,
        max_rounds: int | None,
        max_evaluations: int | None,
        display_progress: bool,
        temperature_mutations_enabled: bool | None = None,
    ) -> dict[str, Any]:
        n_islands = max(1, self.config.n_islands)
        if n_islands > 1:
            tuned_period = max(1, n_islands // 2)
            tuned_k = max(1, min(n_islands, self.config.migration_k or n_islands))
            if self.config.migration_period != tuned_period or self.config.migration_k != tuned_k:
                self.config.migration_period = tuned_period
                self.config.migration_k = tuned_k
        normalized_seeds = self._normalize_seeds(seeds, source="seed")

        # Set up thread pool for concurrent file operations (bounded to avoid oversubscription)
        import concurrent.futures

        loop = asyncio.get_running_loop()
        previous_executor = getattr(loop, "_default_executor", None)
        max_workers = recommended_executor_workers(self.config.eval_concurrency)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        loop.set_default_executor(executor)

        # Create shared cache across all islands (better cache hits + controlled concurrency)
        shared_cache = self._make_cache(island_id=None)

        async def run_islands() -> list[Orchestrator | None]:
            island_results: list[Orchestrator | None] = [None] * n_islands

            async def worker(context: IslandContext) -> None:
                # Use shared cache across islands
                cache = shared_cache
                archive = self._make_archive()
                sampler = self._make_sampler(seed_offset=context.island_id)
                mutator = self._make_mutator()
                if temperature_mutations_enabled is not None:
                    mutator.set_temperature_mutations_enabled(temperature_mutations_enabled)
                log_dir = self._make_log_dir(context.island_id)
                display_local = display_progress and context.island_id == 0
                orchestrator = self._build_orchestrator(
                    display_progress=display_local,
                    temperature_mutations_enabled=temperature_mutations_enabled,
                    island_context=context,
                    island_id=context.island_id,
                    cache=cache,
                    archive=archive,
                    sampler=sampler,
                    mutator=mutator,
                    log_dir=log_dir,
                    control_dir=self.control_dir,
                )
                # Pass metrics to mutator for tracking mutation LLM calls
                mutator._metrics = orchestrator.metrics
                island_seeds = [
                    Candidate(text=seed.text, meta=dict(seed.meta, island=context.island_id))
                    for seed in normalized_seeds
                ]
                try:
                    await orchestrator.run(
                        island_seeds,
                        max_rounds=max_rounds,
                        max_evaluations=max_evaluations,
                    )
                finally:
                    orchestrator.finalize_control()
                island_results[context.island_id] = orchestrator

            tasks = await spawn_islands(n_islands, worker, metrics_queue=None)
            await asyncio.gather(*tasks)
            return island_results

        try:
            orchestrators = await run_islands()
        finally:
            if previous_executor is not None:
                loop.set_default_executor(previous_executor)
            executor.shutdown(wait=True)

        return await self._summarize_island_runs(orchestrators)

    async def _optimize_distributed_islands(
        self,
        seeds: Sequence[str | Candidate],
        *,
        island_ids: Sequence[int],
        max_rounds: int | None,
        max_evaluations: int | None,
        display_progress: bool,
        temperature_mutations_enabled: bool | None = None,
    ) -> dict[str, Any]:
        if not island_ids:
            return await self._summarize_island_runs([])
        normalized_seeds = self._normalize_seeds(seeds, source="seed")
        orchestrators: list[Orchestrator | None] = []
        for idx, island_id in enumerate(island_ids):
            cache = DiskCache(self.base_cache_dir, namespace=f"{self.cache_namespace}_island_{island_id}")
            archive = self._make_archive()
            sampler = self._make_sampler(seed_offset=island_id)
            mutator = self._make_mutator()
            if temperature_mutations_enabled is not None:
                mutator.set_temperature_mutations_enabled(temperature_mutations_enabled)
            log_dir = self._make_log_dir(island_id)
            display_local = display_progress and idx == 0
            orchestrator = self._build_orchestrator(
                display_progress=display_local,
                temperature_mutations_enabled=temperature_mutations_enabled,
                island_context=None,
                cache=cache,
                archive=archive,
                sampler=sampler,
                mutator=mutator,
                log_dir=log_dir,
                island_id=island_id,
                control_dir=self.control_dir,
            )
            mutator._metrics = orchestrator.metrics
            island_seeds = [
                Candidate(text=seed.text, meta=dict(seed.meta, island=island_id))
                for seed in normalized_seeds
            ]
            try:
                await orchestrator.run(
                    island_seeds,
                    max_rounds=max_rounds,
                    max_evaluations=max_evaluations,
                )
            finally:
                orchestrator.finalize_control()
            orchestrators.append(orchestrator)
        return await self._summarize_island_runs(orchestrators)

    async def _optimize_multi_island_staged(
        self,
        seeds: Sequence[str | Candidate],
        *,
        max_rounds: int | None,
        max_evaluations: int | None,
        display_progress: bool,
    ) -> dict[str, Any]:
        phase1_budget = int(max_evaluations * 0.7) if max_evaluations else None
        phase1_rounds = max_rounds if max_rounds else 10

        phase1_result = await self._optimize_multi_island(
            seeds,
            max_rounds=phase1_rounds,
            max_evaluations=phase1_budget,
            display_progress=display_progress,
            temperature_mutations_enabled=False,
        )
        phase1_entries = phase1_result.get("pareto_entries", [])
        phase1_stats = phase1_result.get("evolution_stats", {})
        phase1_metadata_per_island = phase1_result.get("run_metadata_per_island", [])
        if not phase1_entries:
            return {
                **phase1_result,
                "phase1_pareto": [],
                "phase1_evolution_stats": phase1_stats,
                "evolution_stats": phase1_stats,
                "phase1_run_metadata_per_island": phase1_metadata_per_island,
            }

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
            display_progress=display_progress,
            temperature_mutations_enabled=True,
        )
        phase2_stats = phase2_result.get("evolution_stats", {})
        combined_stats = self._combine_evolution_snapshots(
            (phase1_stats.get("islands", []) if phase1_stats else [])
            + (phase2_stats.get("islands", []) if phase2_stats else [])
        )
        phase2_result["phase1_pareto"] = [entry.candidate for entry in phase1_entries]
        phase2_result["phase1_evolution_stats"] = phase1_stats
        phase2_result["phase2_evolution_stats"] = phase2_stats
        phase2_result["evolution_stats"] = combined_stats
        phase2_result["phase1_run_metadata_per_island"] = phase1_metadata_per_island
        phase2_result["phase1_metrics_per_island"] = phase1_result.get("metrics_per_island")
        phase2_result.setdefault("run_metadata_per_island", [])
        if "metrics_per_island" not in phase2_result:
            per_island = None
            metrics_entry = phase2_result.get("metrics")
            if isinstance(metrics_entry, dict):
                per_island = metrics_entry.get("per_island")
            phase2_result["metrics_per_island"] = per_island
        return phase2_result

    def optimize(
        self,
        seeds: Sequence[str | Candidate] | None = None,
        *,
        max_rounds: int | None = None,
        max_evaluations: int | None = None,
        task_lm: str | None = None,  # Kept for API compatibility; models come from adapter init
        reflection_lm: str | None = None,  # Kept for API compatibility; models come from adapter init
        optimize_temperature_after_convergence: bool = False,  # Stage temperature optimization
        display_progress: bool = True,  # Show progress charts
        enable_auto_stop: bool = True,
        enable_seed_initialization: bool = False,  # Use PROMPT-MII-style seed generation
        num_generated_seeds: int = 3,  # How many seeds to generate if initializing
        metrics_callback: Callable | None = None,  # Callback for dashboard updates
    ) -> dict[str, Any]:
        """
        Optimize prompts using TurboGEPA with real LLM evaluation via LiteLLM.

        Parameters:
            seeds: Initial prompt candidates
            max_rounds: Maximum optimization rounds (None = unlimited)
            max_evaluations: Maximum evaluations (None = unlimited)
            task_lm: Kept for compatibility; adapter uses model set at construction
            reflection_lm: Kept for compatibility; adapter uses model set at construction
            optimize_temperature_after_convergence: Stage temperature optimization after
                prompt optimization (Phase 1: optimize prompts, Phase 2: optimize temperature)
            enable_auto_stop: Disable to keep the stop governor from short-circuiting
                long sweeps (handy for regression tests/benchmarks)
            enable_seed_initialization: If True, use PROMPT-MII-style spec induction to
                generate smart initial seeds from task examples instead of using generic prompts.
                Requires a reflection model to be set. Can optimize user-provided seeds or generate from scratch.
            num_generated_seeds: Number of seeds to generate if enable_seed_initialization=True
        """
        import time

        time.time()

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
                    reflection_lm_temperature=self.reflection_model.temperature,
                )
            )
        elif seeds is None:
            # No seeds provided and initialization disabled - use default
            seeds = ["You are a helpful assistant. Follow the instructions carefully."]

        # Just run normally - timeout is handled inside orchestrator
        return asyncio.run(
            self.optimize_async(
                seeds,
                max_rounds=max_rounds,
                max_evaluations=max_evaluations,
                task_lm=task_lm,
                reflection_lm=reflection_lm,
                optimize_temperature_after_convergence=optimize_temperature_after_convergence,
                display_progress=display_progress,
                enable_auto_stop=enable_auto_stop,
                metrics_callback=metrics_callback,
            )
        )

    async def aclose(self) -> None:
        """Close shared HTTP client (if created)."""
        client = getattr(self, "_httpx_client", None)
        if client is not None:
            try:
                await client.aclose()  # type: ignore[attr-defined]
            except Exception:
                pass
