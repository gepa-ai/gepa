"""
High-level orchestrator coordinating selection, evaluation, and mutation.

This module stitches together archive, scheduler, evaluator, and mutator
components. The orchestrator operates within a single island; multi-process
execution is handled by ``islands.spawn_islands`` which can launch several
instances of the loop in parallel.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Awaitable, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from .archive import Archive, ArchiveEntry
from .cache import DiskCache
from .config import Config
from .evaluator import AsyncEvaluator
from .interfaces import Candidate, EvalResult
from .islands import IslandContext, integrate_in, migrate_out
from .logging_utils import EventLogger
# Merge functionality removed - reflection LLM handles combining prompt ideas
from .multi_island_dashboard import IslandDashboard
from .mutator import Mutator
from .sampler import InstanceSampler
from .scheduler import BudgetedScheduler, SchedulerConfig
from .stop_governor import StopGovernor, StopGovernorConfig, EpochMetrics, compute_hypervolume_2d
from .token_controller import TokenCostController


class _NoopLogger:
    async def log(self, _event_type: str, _payload: Dict[str, object]) -> None:  # pragma: no cover - noop logger
        return


class Orchestrator:
    """Single-island orchestrator loop."""

    def __init__(
        self,
        config: Config,
        evaluator: AsyncEvaluator,
        archive: Archive,
        sampler: InstanceSampler,
        mutator: Mutator,
        cache: DiskCache,
        *,
        logger: Optional[EventLogger] = None,
        token_controller: Optional[TokenCostController] = None,
        island_context: Optional[IslandContext] = None,
        show_progress: bool = True,
        stop_governor: Optional[StopGovernor] = None,
        enable_auto_stop: bool = False,
        example_sampler: Optional[Callable[[int], List[Dict[str, object]]]] = None,
        max_rounds: int = 100,  # For progress calculation
        dashboard: Optional[IslandDashboard] = None,
        reflection_eval_fn: Optional[Callable[[Candidate, Sequence[str]], Awaitable[List[Dict[str, object]]]]] = None,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.archive = archive
        self.sampler = sampler
        self.mutator = mutator
        self.cache = cache
        self.island_context = island_context
        self.example_sampler = example_sampler
        self.reflection_eval_fn = reflection_eval_fn
        self.scheduler = BudgetedScheduler(
            SchedulerConfig(
                shards=config.shards,
                eps_improve=config.eps_improve,
                quantile=config.cohort_quantile,
            )
        )
        # Merge manager removed - reflection LLM handles combining ideas
        self.token_controller = token_controller or TokenCostController(config.max_tokens)
        self.logger: EventLogger | _NoopLogger = logger or _NoopLogger()
        self.queue: Deque[Candidate] = deque(maxlen=config.queue_limit)
        self.latest_results: Dict[str, EvalResult] = {}
        self.evaluations_run: int = 0
        self.round_index: int = 0
        # Merge tracking removed
        self.show_progress = show_progress
        # Use IslandDashboard for all cases (single or multi-island)
        island_id = island_context.island_id if island_context else 0
        if dashboard is not None:
            self.dashboard = dashboard
        elif show_progress:
            self.dashboard = IslandDashboard(n_islands=config.n_islands, max_rounds=max_rounds)
        else:
            self.dashboard = None
        self.island_id = island_id
        self.stop_governor = stop_governor if enable_auto_stop else None
        self.enable_auto_stop = enable_auto_stop
        self.total_tokens_spent: int = 0
        self.qd_cells_seen: set[tuple] = set()
        self._shard_cache: Dict[tuple[int, float], List[str]] = {}

        # Streaming mode: buffer for mutations generated in background
        self._mutation_buffer: Deque[Candidate] = deque()
        self._mutation_task: Optional[asyncio.Task] = None
        self.eval_batches_completed: int = 0  # For migration timing

    def enqueue(self, candidates: Iterable[Candidate]) -> None:
        for candidate in candidates:
            if not candidate.text.strip():
                continue
            if len(self.queue) == self.queue.maxlen:
                self.queue.popleft()
            self.queue.append(candidate)

    async def run(
        self,
        seeds: Sequence[Candidate],
        *,
        max_rounds: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        resume: bool = True,  # Enable automatic resume from cache
    ) -> None:
        """Execute the orchestration loop until budgets are exhausted.

        Args:
            seeds: Initial candidates to start optimization
            max_rounds: Maximum number of rounds (None = unlimited)
            max_evaluations: Maximum evaluations (None = unlimited)
            resume: If True, automatically resume from saved state if available
        """
        import time

        if not seeds:
            return

        # Attempt to resume from saved state
        resumed = False
        if resume and self.cache.has_state():
            state = self.cache.load_state()
            if state:
                await self._restore_state(state)
                resumed = True
                print(f"🔄 Resumed from round {self.round_index + 1} ({self.evaluations_run} evaluations)")

        if not resumed:
            await self._seed_archive(seeds)

        # STREAMING MODE: Start first mutation task immediately after seed evaluation
        # Mutations will be generated in background and streamed into the buffer
        self._mutation_task = asyncio.create_task(
            self._spawn_mutations(callback=self._mutation_buffer.append)
        )

        # Main streaming loop - runs until convergence
        while True:
            batch_start_time = time.time()

            # Stop conditions
            if max_rounds is not None and self.eval_batches_completed >= max_rounds:
                break
            if max_evaluations is not None and self.evaluations_run >= max_evaluations:
                break

            # 1. Drain mutation buffer into queue
            while self._mutation_buffer:
                candidate = self._mutation_buffer.popleft()
                self.enqueue([candidate])

            # 2. Try to select a batch
            batch = self._select_batch()

            # 3. If queue is empty, wait for mutations to refill it
            if not batch:
                if self._mutation_task and not self._mutation_task.done():
                    # Wait for current mutation task to complete
                    await self._mutation_task
                    # Drain buffer again after mutations complete
                    while self._mutation_buffer:
                        candidate = self._mutation_buffer.popleft()
                        self.enqueue([candidate])
                    # Try selecting again
                    batch = self._select_batch()

                # If still no batch, we're done (queue empty + no mutations running)
                if not batch:
                    break

            # 4. Evaluate batch
            select_time = time.time() - batch_start_time
            race_start = time.time()
            await self._race_batch(batch)
            race_time = time.time() - race_start
            self.eval_batches_completed += 1

            # 5. Start next mutation task (if previous one finished)
            if self._mutation_task is None or self._mutation_task.done():
                self._mutation_task = asyncio.create_task(
                    self._spawn_mutations(callback=self._mutation_buffer.append)
                )

            # 6. Migration (based on eval batches instead of rounds)
            migrate_start = time.time()
            await self._maybe_migrate()
            migrate_time = time.time() - migrate_start

            # 7. Logging
            log_start = time.time()
            await self._maybe_log_summary()
            log_time = time.time() - log_start

            self._shard_cache.clear()

            # Log timing breakdown
            batch_total_time = time.time() - batch_start_time
            await self._log("timing_breakdown", {
                "batch": self.eval_batches_completed,
                "total_time": batch_total_time,
                "select_batch_time": select_time,
                "race_batch_time": race_time,
                "migrate_time": migrate_time,
                "log_summary_time": log_time,
            })

            # Print timing summary
            if self.show_progress:
                print(f"\n{'='*80}")
                print(f"⏱️  BATCH {self.eval_batches_completed} TIMING BREAKDOWN")
                print(f"{'='*80}")
                print(f"  Select batch:          {select_time:>8.2f}s")
                print(f"  Race batch:           {race_time:>8.2f}s")
                print(f"  Migration:            {migrate_time:>8.2f}s")
                print(f"  Logging:              {log_time:>8.2f}s")
                print(f"  {'-'*80}")
                print(f"  TOTAL BATCH TIME:     {batch_total_time:>8.2f}s")
                print(f"  💡 Mutations generating in background...")
                print(f"{'='*80}\n")

            # 8. Check target quality threshold
            if self.config.target_quality is not None:
                pareto = self.archive.pareto_entries()
                if pareto:
                    full_shard = self.config.shards[-1]
                    full_eval_entries = [
                        entry for entry in pareto
                        if entry.result.shard_fraction == full_shard
                    ]

                    if full_eval_entries:
                        best_quality = max(
                            entry.result.objectives.get(self.config.promote_objective, 0.0)
                            for entry in full_eval_entries
                        )
                        if best_quality >= self.config.target_quality:
                            await self._log("target_quality_reached", {
                                "target": self.config.target_quality,
                                "achieved": best_quality,
                                "batch": self.eval_batches_completed,
                            })
                            if self.show_progress:
                                print(f"\n🎯 Target quality {self.config.target_quality:.2%} reached! Achieved: {best_quality:.2%}")
                                print(f"   (Verified on full dataset: {full_shard:.0%} of examples)")
                            break

            # 9. Check stop governor
            if self.enable_auto_stop and self.stop_governor is not None:
                should_stop, debug_info = self._check_stop_governor()
                if should_stop:
                    await self._log("auto_stop", debug_info)
                    if self.show_progress:
                        print(f"\n🛑 Auto-stopping: {debug_info['reason']}")
                    break

            # Keep round_index in sync for compatibility with existing code
            self.round_index = self.eval_batches_completed

        # Clean up any pending mutation task before exiting
        if self._mutation_task:
            try:
                await self._mutation_task
            except Exception:
                pass  # Ignore errors on cleanup

            # Save state after each round for resumability
            await self._save_state()

        await self.finalize()

        # Clear state only if we reached completion (not if stopped early)
        completed = (
            (max_rounds is not None and self.round_index >= max_rounds) or
            (max_evaluations is not None and self.evaluations_run >= max_evaluations)
        )
        if completed:
            self.cache.clear_state()

    async def finalize(self, delta: Optional[float] = None) -> None:
        """Run token-cost pruning on the Pareto frontier."""

        # Wait for any pending state save to complete before finalizing
        if hasattr(self, '_save_task') and not self._save_task.done():
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass  # Task was cancelled, that's okay

        entries = self.archive.pareto_entries()
        if not entries:
            return

        delta = self.config.prune_delta if delta is None else delta
        # Use separate compression concurrency for better finalization throughput
        limit = max(
            1,
            min(
                int(self.config.compression_concurrency or 16),
                len(entries),
            ),
        )
        compression_sem = asyncio.Semaphore(limit)

        async def evaluate(candidate: Candidate) -> float:
            shard_fraction = self.config.compression_shard_fraction
            shard_size = self._shard_size(shard_fraction)
            shard = self.sampler.sample_shard(self.round_index, shard_size)
            async with compression_sem:
                result = await self.evaluator.eval_on_shard(
                    candidate,
                    shard,
                    concurrency=self.config.eval_concurrency,
                    shard_fraction=shard_fraction,
                )
            self.latest_results[candidate.fingerprint] = result
            return result.objectives.get(self.config.compression_objective, 0.0)

        async def process(entry: ArchiveEntry) -> None:
            compressed = await self.token_controller.compress(entry.candidate, delta, evaluate)
            if compressed.text == entry.candidate.text:
                return
            result = self.latest_results.get(compressed.fingerprint)
            if result is None:
                shard_fraction = self.config.shards[-1]
                shard_size = self._shard_size(shard_fraction)
                shard = self.sampler.sample_shard(self.round_index, shard_size)
                async with compression_sem:
                    result = await self.evaluator.eval_on_shard(
                        compressed,
                        shard,
                        concurrency=self.config.eval_concurrency,
                        shard_fraction=shard_fraction,
                    )
                self.latest_results[compressed.fingerprint] = result
            await self._log(
                "compression_applied",
                {
                    "candidate": compressed.fingerprint,
                    "objective": result.objectives,
                },
            )
            await self.archive.insert(compressed, result)

        await asyncio.gather(*(process(entry) for entry in entries))

    async def _seed_archive(self, seeds: Sequence[Candidate]) -> None:
        shard_fraction = self.config.shards[0]
        shard_size = self._shard_size(shard_fraction)
        shard = self.sampler.sample_shard(self.round_index, shard_size)

        if self.show_progress:
            print(f"\n🌱 Evaluating {len(seeds)} seed prompt(s) on {shard_size} examples ({shard_fraction:.0%} of dataset)...")
            print(f"   This establishes the baseline for optimization.")

        results = await asyncio.gather(
            *[
                self._evaluate_candidate(seed, shard, shard_fraction, show_progress=self.show_progress)
                for seed in seeds
            ]
        )

        if self.show_progress:
            print(f"   ✓ Seed evaluation complete!\n")

        for pair in results:
            await self._handle_result(pair)
        promotions = self.scheduler.promote_ready()
        self.enqueue(promotions)
        self.enqueue(seeds)

    def _select_batch(self) -> List[Candidate]:
        batch: List[Candidate] = []
        seen: set[str] = set()
        while self.queue and len(batch) < self.config.batch_size:
            candidate = self.queue.popleft()
            if candidate.fingerprint in seen:
                continue
            batch.append(candidate)
            seen.add(candidate.fingerprint)

        if len(batch) < self.config.batch_size:
            needed = self.config.batch_size - len(batch)
            exploit = (needed + 1) // 2
            explore = needed // 2
            archive_candidates = self.archive.select_for_generation(
                exploit,
                explore,
                objective=self.config.promote_objective,
            )
            for candidate in archive_candidates:
                if len(batch) >= self.config.batch_size:
                    break
                if candidate.fingerprint in seen:
                    continue
                batch.append(candidate)
                seen.add(candidate.fingerprint)
        return batch

    async def _race_batch(self, batch: Sequence[Candidate]) -> None:
        # Show progress for Round 1 (seed full evaluation)
        if self.show_progress and self.round_index == 0 and len(batch) == 1:
            print(f"🔄 Evaluating seed on full dataset (100%)...")

        tasks: set[asyncio.Task] = set()
        for candidate in batch:
            idx = self.scheduler.current_shard_index(candidate)
            shard_fraction = self.scheduler.shard_fraction_for_index(idx)
            shard_size = self._shard_size(shard_fraction)
            cache_key = (self.round_index, shard_fraction)
            shard = self._shard_cache.get(cache_key)
            if shard is None:
                shard = self.sampler.sample_shard(self.round_index, shard_size)
                self._shard_cache[cache_key] = shard

            # Show progress for seed baseline evaluation (Round 1)
            show_eval_progress = self.show_progress and self.round_index == 0 and len(batch) == 1
            task = asyncio.create_task(self._evaluate_candidate(candidate, shard, shard_fraction, show_progress=show_eval_progress))
            tasks.add(task)

        # Track progress for dashboard updates
        evals_total = len(tasks)
        evals_done = 0

        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    pair = await task
                except Exception as exc:  # pragma: no cover - defensive logging
                    await self._log(
                        "evaluation_error",
                        {
                            "error": str(exc),
                        },
                    )
                    continue
                await self._handle_result(pair)
                promotions = self.scheduler.promote_ready()
                if promotions:
                    await self._log("promote", {"count": len(promotions)})
                    self.enqueue(promotions)

                # Update dashboard with eval progress and current scores
                evals_done += 1
                self._update_dashboard_progress(evals_done, evals_total)

        # Show completion for Round 1 seed evaluation
        if self.show_progress and self.round_index == 0 and len(batch) == 1:
            print(f"   ✓ Seed evaluation on full dataset complete!\n")

        # Flush any remaining promotions
        promotions = self.scheduler.promote_ready()
        if promotions:
            await self._log("promote", {"count": len(promotions)})
            if self.show_progress:
                print(f"\n   🚀 Promoted {len(promotions)} candidate(s) to next shard for evaluation")
            self.enqueue(promotions)

        # Show batch quality summary during early optimization (first 10 batches)
        if self.show_progress and self.eval_batches_completed < 10 and len(batch) > 1:
            self._log_batch_quality_summary(batch)

        if self.show_progress:
            self._show_inline_progress()

    def _get_best_quality_from_full_shard(self) -> float:
        """Get best quality from candidates evaluated on the longest/full shard only.

        This ensures displayed quality represents performance on the full dataset,
        not partial shards which may give inflated scores due to overfitting.

        Falls back to best quality from ANY shard if no full-shard evaluations
        exist yet (e.g., during seed baseline evaluation).
        """
        pareto = self.archive.pareto_entries()
        if not pareto:
            return 0.0

        full_shard = self.config.shards[-1]  # Longest shard (e.g., 1.0 = 100%)

        # Try to get quality from full-shard evaluations first
        full_shard_quality = 0.0
        has_full_shard = False
        for entry in pareto:
            if entry.result.shard_fraction == full_shard:
                has_full_shard = True
                quality = entry.result.objectives.get(self.config.promote_objective, 0.0)
                full_shard_quality = max(full_shard_quality, quality)

        if has_full_shard:
            return full_shard_quality

        # Fallback: Return best quality from ANY shard
        # This ensures we always show something meaningful during early optimization
        best_quality = max(
            entry.result.objectives.get(self.config.promote_objective, 0.0)
            for entry in pareto
        )
        return best_quality

    def _show_inline_progress(self) -> None:
        import sys

        pareto = self.archive.pareto_entries()
        # Show best quality from full shard only (not partial shards)
        best_quality = self._get_best_quality_from_full_shard()
        avg_quality = (
            sum(e.result.objectives.get(self.config.promote_objective, 0.0) for e in pareto) / len(pareto)
            if pareto else 0.0
        )
        sys.stdout.write(
            f"\r   🔄 Batch {self.eval_batches_completed + 1} | Evals: {self.evaluations_run} | Best: {best_quality:.2%} | Avg: {avg_quality:.2%}  "
        )
        sys.stdout.flush()

    async def _handle_result(self, pair: Tuple[Candidate, EvalResult, str]) -> None:
        candidate, result, decision = pair
        quality = result.objectives.get(self.config.promote_objective, 0.0)

        # Only update stored quality when this evaluation covers an equal or larger shard
        shard_fraction = result.shard_fraction or 0.0
        meta = dict(candidate.meta)
        prev_fraction = meta.get("quality_shard_fraction", 0.0)
        prev_quality = meta.get("quality", float("-inf"))

        if shard_fraction > prev_fraction or (
            shard_fraction == prev_fraction and quality >= prev_quality
        ):
            meta["quality"] = quality
            meta["quality_shard_fraction"] = shard_fraction

        meta["parent_objectives"] = result.objectives
        candidate = Candidate(text=candidate.text, meta=meta)
        self.latest_results[candidate.fingerprint] = result
        await self._log(
            "archive_update",
            {
                "candidate": candidate.fingerprint,
                "objectives": result.objectives,
                "n_examples": result.n_examples,
            },
        )

        # Check if it's in Pareto before insertion
        was_in_pareto = any(e.candidate.fingerprint == candidate.fingerprint for e in self.archive.pareto_entries())

        await self.archive.insert(candidate, result)

        # Check if it made it to Pareto after insertion
        is_in_pareto = any(e.candidate.fingerprint == candidate.fingerprint for e in self.archive.pareto_entries())

        # Log if a new candidate joined the Pareto frontier
        if is_in_pareto and not was_in_pareto and self.show_progress:
            display_quality = candidate.meta.get("quality", quality)
            print("\n" + "🌟" * 40)
            print(f"   ⭐ NEW PARETO-OPTIMAL CANDIDATE! (Quality: {display_quality:.2%})")
            print("   " + "-" * 78)
            for line in candidate.text.strip().splitlines():
                print(f"   {line}")
            print("   " + "-" * 78)
            print("🌟" * 40 + "\n")

        self._register_failures(result)
        if decision == "pruned":
            await self._log(
                "prune",
                {
                    "candidate": candidate.fingerprint,
                    "shard_fraction": result.shard_fraction,
                    "objective": result.objectives.get(self.config.promote_objective),
                },
            )
        if decision == "promoted":
            await self._log(
                "promotion_recorded",
                {
                    "candidate": candidate.fingerprint,
                    "next_shard_fraction": self.scheduler.current_shard_fraction(candidate),
                },
            )

    async def _spawn_mutations(self, callback: Optional[Callable[[Candidate], None]] = None) -> None:
        """Generate mutations using batched reflection for efficiency.

        Args:
            callback: Optional function to call for each mutation as it's generated (for streaming mode)
        """
        import time
        spawn_start = time.time()

        entries = self.archive.pareto_entries()
        if not entries:
            return

        num_mutations = self.config.max_mutations_per_round
        if num_mutations <= 0:
            return

        gen_start = time.time()
        mutations = await self._generate_mutations_batched(entries, num_mutations)
        gen_time = time.time() - gen_start

        if mutations:
            # Log prompts to file in background (fire-and-forget) to avoid blocking
            # This saves 20-100ms per round by not waiting for disk I/O
            asyncio.create_task(self._log_prompts_to_file(mutations))

            if self.show_progress:
                # Compute mutation statistics for debugging
                gen_methods = {}
                for m in mutations:
                    method = m.meta.get("generation_method", "unknown")
                    gen_methods[method] = gen_methods.get(method, 0) + 1

                # Find parent qualities from latest_results
                parent_qualities = []
                for m in mutations:
                    parent_fp = m.meta.get("parent")
                    if parent_fp and parent_fp in self.latest_results:
                        parent_result = self.latest_results[parent_fp]
                        parent_quality = parent_result.objectives.get(self.config.promote_objective, 0.0)
                        parent_qualities.append(parent_quality)

                print("\n" + "=" * 80)
                print(f"   ✨ NEW PROMPTS GENERATED ({len(mutations)} total)")
                print(f"   Generation methods: {dict(gen_methods)}")
                if parent_qualities:
                    avg_parent_quality = sum(parent_qualities) / len(parent_qualities)
                    print(f"   Parent qualities: avg={avg_parent_quality:.2%}, min={min(parent_qualities):.2%}, max={max(parent_qualities):.2%}")
                print("=" * 80)

                for idx, candidate in enumerate(mutations, start=1):
                    # Show generation method and parent info
                    gen_method = candidate.meta.get("generation_method", "unknown")
                    parent_quality = candidate.meta.get("parent_objectives", {}).get("quality", "N/A")

                    print(f"\n   Prompt #{idx} [{gen_method}]")
                    if parent_quality != "N/A":
                        print(f"   Parent quality: {parent_quality:.2%}")
                    print("   " + "-" * 76)

                    # Show full prompt text, nicely formatted
                    for line in candidate.text.strip().splitlines():
                        print(f"   {line}")

                    print("   " + "-" * 76)

                print("=" * 80 + "\n")

            await self._log("mutation_proposed", {"count": len(mutations)})

            # Stream mutations via callback (for streaming mode) or enqueue directly (for round mode)
            if callback:
                for candidate in mutations:
                    callback(candidate)
            else:
                self.enqueue(mutations)

            await self._log(
                "mutation_accepted",
                {
                    "candidates": [candidate.fingerprint for candidate in mutations],
                },
            )

        spawn_total = time.time() - spawn_start
        await self._log("spawn_mutations_timing", {
            "total_time": spawn_total,
            "generation_time": gen_time,
            "num_mutations": len(mutations) if mutations else 0,
        })

    async def _generate_mutations_batched(
        self,
        entries: List[ArchiveEntry],
        num_mutations: int,
    ) -> List[Candidate]:
        """Generate mutations using batched reflection."""
        import time
        batch_start = time.time()

        # Smart parent selection: mix of top quality + QD diversity
        # Select 3-5 parents to give reflection LLM rich context
        parent_select_start = time.time()
        num_parents = min(5, max(3, len(entries)))

        # Get top performers by quality
        sorted_entries = sorted(
            entries,
            key=lambda e: e.result.objectives.get(self.config.promote_objective, float("-inf")),
            reverse=True,
        )
        top_quality = sorted_entries[: num_parents // 2 + 1]  # At least half are top quality

        # Add diversity from QD grid
        qd_diverse = self.archive.sample_qd(num_parents - len(top_quality))

        # Combine (deduplicate by fingerprint)
        selected_fingerprints = {e.candidate.fingerprint for e in top_quality}
        selected_entries = list(top_quality)

        for candidate in qd_diverse:
            if candidate.fingerprint not in selected_fingerprints:
                # Find the entry for this candidate
                entry = next((e for e in entries if e.candidate.fingerprint == candidate.fingerprint), None)
                if entry:
                    selected_entries.append(entry)
                    selected_fingerprints.add(candidate.fingerprint)
                if len(selected_entries) >= num_parents:
                    break

        parent_select_time = time.time() - parent_select_start

        # Use cached evaluation results instead of fresh eval (saves 3-5s per round!)
        # We already have traces from the ASHA evaluation that just ran
        reflection_minibatch_size = 5
        reflection_examples: List[Dict[str, object]] = []

        # Sample task examples for spec induction
        task_examples = self._sample_task_examples_for_spec_induction(num_examples=reflection_minibatch_size)

        cache_start = time.time()
        if selected_entries:
            # Get cached traces from best parent's most recent evaluation
            best_parent_fingerprint = selected_entries[0].candidate.fingerprint
            cached_result = self.latest_results.get(best_parent_fingerprint)

            if cached_result and cached_result.traces:
                # Use cached traces from recent ASHA evaluation
                # OG GEPA passes ALL examples (successes + failures), not just failures
                # This shows reflection LLM both what works and what needs fixing
                all_traces = list(cached_result.traces)

                # Prioritize failures (more informative), but include some successes too
                failure_traces = [t for t in all_traces if t.get("quality", 0.0) < 1.0]
                success_traces = [t for t in all_traces if t.get("quality", 0.0) >= 1.0]

                # Take mostly failures (4) + some successes (1) if available
                selected_traces = failure_traces[:4] + success_traces[:1]
                selected_traces = selected_traces[:reflection_minibatch_size]

                for trace in selected_traces:
                    example_input = trace.get("input", "")
                    assistant_output = trace.get("response", "") or trace.get("output", "")
                    expected_answer = trace.get("expected_answer", "")
                    additional_context = trace.get("additional_context")
                    quality = trace.get("quality", 0.0)

                    if quality >= 1.0:
                        feedback = f"The generated response is correct. The response includes the correct answer '{expected_answer}'"
                    else:
                        feedback_parts = [f"The generated response is incorrect. The correct answer is '{expected_answer}'."]
                        feedback_parts.append("Ensure that the correct answer is included in the response exactly as it is.")
                        if additional_context and isinstance(additional_context, dict):
                            context_lines = [f"{k}: {v}" for k, v in additional_context.items()]
                            if context_lines:
                                feedback_parts.append(f"Here is some additional context that might be helpful:\n{chr(10).join(context_lines)}")
                        feedback = " ".join(feedback_parts)

                    reflection_examples.append({
                        "input": example_input,
                        "assistant_output": assistant_output,
                        "expected_answer": expected_answer,
                        "feedback": feedback,
                        "additional_context": additional_context,
                    })

        cache_time = time.time() - cache_start

        # Build parent contexts for mutation (all selected parents)
        # IMPORTANT: Use latest_results instead of entry.result to get the most recent
        # (and typically full-dataset) evaluation, not the archived partial-shard result
        parent_contexts: List[Dict[str, object]] = []
        for entry in selected_entries:
            # Prefer latest_results (most recent eval, usually full dataset)
            # Fall back to entry.result if not found in latest_results
            latest_result = self.latest_results.get(entry.candidate.fingerprint)
            parent_objectives = latest_result.objectives if latest_result else entry.result.objectives

            candidate_with_parent = entry.candidate.with_meta(
                parent_objectives=parent_objectives,
            )
            parent_contexts.append({
                "candidate": candidate_with_parent,
                "failures": [],  # Not used anymore, we have fresh reflection_examples
            })

        if not parent_contexts:
            return []

        # Set reflection examples for the mutator
        if hasattr(self.mutator, "set_reflection_examples"):
            if reflection_examples:
                self.mutator.set_reflection_examples(reflection_examples)
            elif task_examples:
                # If fresh eval failed, use task examples as fallback
                self.mutator.set_reflection_examples(task_examples)

        mutator_start = time.time()
        mutations = await self.mutator.propose(parent_contexts, num_mutations, task_examples=task_examples)
        mutator_time = time.time() - mutator_start

        batch_total_time = time.time() - batch_start

        # Log detailed timing breakdown
        if self.show_progress:
            print(f"\n⏱️  Mutation generation timing:")
            print(f"   Parent selection:    {parent_select_time:.2f}s")
            print(f"   Cache lookup:        {cache_time:.2f}s (reusing {len(reflection_examples)} failures from ASHA)")
            print(f"   Mutator.propose():   {mutator_time:.2f}s (LLM calls)")
            print(f"   Total:               {batch_total_time:.2f}s\n")

        await self._log(
            "batch_reflection",
            {
                "num_parents": len(parent_contexts),
                "num_mutations_requested": num_mutations,
                "num_mutations_generated": len(mutations),
            },
        )

        await self._log("mutation_generation_timing", {
            "total_time": batch_total_time,
            "parent_selection_time": parent_select_time,
            "cache_time": cache_time,
            "mutator_propose_time": mutator_time,
        })

        return mutations

    def _sample_task_examples_for_spec_induction(self, num_examples: int = 3) -> List[Dict[str, object]]:
        """Sample a few task examples for spec induction (PROMPT-MII style)."""
        if self.example_sampler:
            # Use adapter-provided sampler (has access to actual example data)
            return self.example_sampler(num_examples)
        else:
            # Fallback: return empty list (spec induction won't run)
            return []

    # _maybe_merge method removed - reflection LLM handles combining ideas from multiple prompts

    async def _maybe_migrate(self) -> None:
        if not self.island_context:
            return
        if self.eval_batches_completed == 0:
            return
        # Trigger migration based on evaluation batches completed (streaming mode)
        if self.eval_batches_completed % self.config.migration_period != 0:
            return
        elites = self.archive.select_for_generation(
            self.config.migration_k,
            0,
            objective=self.config.promote_objective,
        )
        if elites:
            migrate_out(self.island_context, elites)
            await self._log("migrate_out", {"count": len(elites)})
        incoming = integrate_in(self.island_context)
        if incoming:
            await self._log("migrate_in", {"count": len(incoming)})
            self.enqueue(incoming)

    async def _evaluate_candidate(
        self,
        candidate: Candidate,
        shard: Sequence[str],
        shard_fraction: float,
        show_progress: bool = False,
    ) -> Tuple[Candidate, EvalResult, str]:
        await self._log(
            "eval_start",
            {
                "candidate": candidate.fingerprint,
                "shard_fraction": shard_fraction,
                "shard_size": len(shard),
            },
        )
        result = await self.evaluator.eval_on_shard(
            candidate,
            shard,
            concurrency=self.config.eval_concurrency,
            shard_fraction=shard_fraction,
            show_progress=show_progress,
        )
        self.evaluations_run += result.n_examples
        decision = self.scheduler.record(candidate, result, objective_key=self.config.promote_objective)

        # Debug: Log promotion decisions and mutation quality during early batches
        if self.show_progress and self.eval_batches_completed < 10:
            quality = result.objectives.get(self.config.promote_objective, 0.0)
            current_shard = result.shard_fraction or 0.0

            # Extract metadata for detailed logging
            generation_method = candidate.meta.get("generation_method", "unknown")
            parent_fp = candidate.meta.get("parent", "none")

            # Check if this is a promoted evaluation (candidate was on smaller shard before)
            prev_result = self.latest_results.get(candidate.fingerprint)
            if prev_result and prev_result.shard_fraction:
                prev_shard = prev_result.shard_fraction
                if current_shard > prev_shard:
                    import sys
                    prev_quality = prev_result.objectives.get(self.config.promote_objective, 0.0)
                    sys.stderr.write(f"   🔼 [Batch {self.eval_batches_completed}] PROMOTION EVAL: {prev_shard:.0%}→{current_shard:.0%}, quality: {prev_quality:.2%}→{quality:.2%}, decision: {decision}\n")
                    sys.stderr.flush()
                    return candidate, result, decision

            # Log mutation quality with generation method
            parent_quality = "N/A"
            if parent_fp != "none":
                parent_result = self.latest_results.get(parent_fp)
                if parent_result:
                    parent_quality = f"{parent_result.objectives.get(self.config.promote_objective, 0.0):.2%}"

            import sys
            sys.stderr.write(f"   [Batch {self.eval_batches_completed}] {generation_method}: quality={quality:.2%}, parent_quality={parent_quality}, shard={current_shard:.0%}, decision={decision}\n")
            sys.stderr.flush()

        await self._log(
            "eval_done",
            {
                "candidate": candidate.fingerprint,
                "objectives": result.objectives,
                "n_examples": result.n_examples,
                "shard_fraction": shard_fraction,
            },
        )
        return candidate, result, decision

    def _register_failures(self, result: EvalResult) -> None:
        hard_examples: List[str] = []
        for trace in result.traces:
            example_id = trace.get("example_id") if isinstance(trace, dict) else None
            quality = trace.get("quality") if isinstance(trace, dict) else None
            if example_id is None or quality is None:
                continue
            if quality < result.objectives.get("quality", quality):
                hard_examples.append(example_id)
        if hard_examples:
            self.sampler.register_hard_examples(hard_examples)

    def _log_batch_quality_summary(self, batch: List[Candidate]) -> None:
        """Log quality distribution summary for a batch of evaluated candidates."""
        import sys

        # Collect qualities by generation method
        qualities_by_method: Dict[str, List[float]] = {}
        for candidate in batch:
            result = self.latest_results.get(candidate.fingerprint)
            if not result:
                continue
            quality = result.objectives.get(self.config.promote_objective, 0.0)
            method = candidate.meta.get("generation_method", "unknown")
            if method not in qualities_by_method:
                qualities_by_method[method] = []
            qualities_by_method[method].append(quality)

        # Print summary
        sys.stderr.write(f"\n   📊 [Batch {self.eval_batches_completed}] Quality Summary:\n")
        for method, qualities in sorted(qualities_by_method.items()):
            if not qualities:
                continue
            avg = sum(qualities) / len(qualities)
            min_q = min(qualities)
            max_q = max(qualities)
            zero_count = sum(1 for q in qualities if q == 0.0)
            sys.stderr.write(f"      {method:25s}: avg={avg:5.2%}, min={min_q:5.2%}, max={max_q:5.2%}, zeros={zero_count}/{len(qualities)}\n")
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _shard_size(self, shard_fraction: float) -> int:
        total = max(len(self.sampler.example_ids), 1)
        size = max(1, int(total * shard_fraction))
        return min(size, total)

    async def _log(self, event_type: str, payload: Dict[str, object]) -> None:
        await self.logger.log(event_type, payload)

    async def _log_prompts_to_file(self, mutations: List[Candidate]) -> None:
        """Write all prompts to a dedicated file for easy inspection.

        Uses retries and proper file handling to avoid "too many open files" errors.
        """
        from pathlib import Path
        import time

        # Create prompts log directory
        # Handle both EventLogger (has path) and _NoopLogger (no path)
        if isinstance(self.logger, _NoopLogger):
            # Use default location for noop logger
            log_base = Path(".turbo_gepa/logs")
        else:
            # Get parent directory of the log file
            log_base = Path(self.logger.path).parent

        prompts_dir = log_base / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        # Create a file for this round
        island_suffix = f"_island{self.island_id + 1}" if hasattr(self, 'island_id') else ""
        filename = f"round_{self.round_index + 1:03d}{island_suffix}.txt"
        filepath = prompts_dir / filename

        lines = []
        lines.append("=" * 80)
        lines.append(f"PROMPTS GENERATED - Round {self.round_index + 1}")
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Count: {len(mutations)}")
        lines.append("=" * 80)
        lines.append("")

        for idx, candidate in enumerate(mutations, start=1):
            gen_method = candidate.meta.get("generation_method", "unknown")
            parent_quality = candidate.meta.get("parent_objectives", {}).get("quality", "N/A")

            lines.append(f"Prompt #{idx} [{gen_method}]")
            if parent_quality != "N/A":
                lines.append(f"Parent quality: {parent_quality:.2%}")
            lines.append("-" * 80)
            lines.append(candidate.text.strip())
            lines.append("-" * 80)
            lines.append("")

        content = "\n".join(lines)

        # Write with retries and proper cleanup
        def write_with_retry(path: Path, text: str) -> None:
            """Write to file with exponential backoff retry on OSError."""
            import time as time_module
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Use context manager to ensure file is closed
                    with path.open("w", encoding="utf-8") as f:
                        f.write(text)
                    return  # Success
                except OSError as e:
                    if attempt < max_attempts - 1:
                        # Exponential backoff: 0.1s, 0.2s, 0.4s
                        wait = 0.1 * (2 ** attempt)
                        time_module.sleep(wait)
                    else:
                        # Last attempt failed, log error but don't crash
                        import sys
                        print(f"Warning: Failed to write {path} after {max_attempts} attempts: {e}", file=sys.stderr)

        def append_with_retry(path: Path, text: str) -> None:
            """Append to file with exponential backoff retry on OSError."""
            import time as time_module
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Use context manager to ensure file is closed
                    with path.open("a", encoding="utf-8") as f:
                        f.write(text)
                        f.write("\n\n")
                    return  # Success
                except OSError as e:
                    if attempt < max_attempts - 1:
                        wait = 0.1 * (2 ** attempt)
                        time_module.sleep(wait)
                    else:
                        import sys
                        print(f"Warning: Failed to append to {path} after {max_attempts} attempts: {e}", file=sys.stderr)

        # Write files in thread pool to avoid blocking
        try:
            await asyncio.to_thread(write_with_retry, filepath, content)
            combined_log = prompts_dir / f"all_prompts{island_suffix}.txt"
            await asyncio.to_thread(append_with_retry, combined_log, content)
        except Exception as e:
            # Catch any remaining errors to prevent task exceptions
            import sys
            print(f"Warning: Error in _log_prompts_to_file: {e}", file=sys.stderr)

    def _update_dashboard_progress(self, evals_done: int, evals_total: int) -> None:
        """Update dashboard with real-time eval progress during a round."""
        if self.dashboard is None:
            return

        # Get current metrics from archive
        pareto = self.archive.pareto_entries()
        if not pareto:
            return

        # Show best quality from full shard only (not partial shards)
        best_quality = self._get_best_quality_from_full_shard()
        current_quality = sum(
            entry.result.objectives.get(self.config.promote_objective, 0.0)
            for entry in pareto
        ) / len(pareto)

        # Update dashboard with eval progress
        self.dashboard.update_island(
            island_id=self.island_id,
            round_num=self.round_index,
            best_quality=best_quality,
            avg_quality=current_quality,
            pareto_size=len(pareto),
            evals_done=evals_done,
            evals_total=evals_total,
        )

        # Display if we have meaningful data
        has_meaningful_data = best_quality > 0.001 or self.round_index > 0
        if has_meaningful_data:
            self.dashboard.display()

    async def _maybe_log_summary(self) -> None:
        interval = max(self.config.log_summary_interval, 0)
        if interval == 0:
            return

        # Adaptive display frequency: More frequent early, less frequent later
        # Batch 0: Always show (baseline)
        # Batches 1-5: Every batch (rapid iteration feedback)
        # Batches 6-20: Every 3 batches (still learning fast)
        # Batches 21+: Every interval batches (config default: 10)
        should_display = False
        if self.eval_batches_completed == 0:
            should_display = True  # Always show seed baseline
        elif self.eval_batches_completed <= 5:
            should_display = True  # Show every batch early on
        elif self.eval_batches_completed <= 20:
            should_display = (self.eval_batches_completed % 3 == 0)  # Every 3 batches
        else:
            should_display = ((self.eval_batches_completed + 1) % interval == 0)  # Normal interval

        if not should_display:
            return

        hardness_size = self.sampler.hardness_size() if hasattr(self.sampler, "hardness_size") else 0

        # Calculate metrics (needed for both chart display and island dashboard)
        pareto = self.archive.pareto_entries()
        if pareto:
            # Get best quality from full shard only (not partial shards)
            best_quality = self._get_best_quality_from_full_shard()
            # Get current average quality
            current_quality = sum(
                entry.result.objectives.get(self.config.promote_objective, 0.0)
                for entry in pareto
            ) / len(pareto)

            # Update and display IslandDashboard if enabled
            metrics_queue = None
            if self.island_context is not None:
                metrics_queue = self.island_context.metrics_queue

            if metrics_queue is not None:
                try:
                    metrics_queue.put_nowait({
                        "island_id": self.island_id,
                        "round_num": self.eval_batches_completed,  # Use eval_batches for streaming mode
                        "best_quality": best_quality,
                        "avg_quality": current_quality,
                        "pareto_size": len(pareto),
                        "evals_done": self.evaluations_run,
                        "evals_total": 0,
                    })
                except asyncio.QueueFull:
                    pass

            should_show_local = (
                self.dashboard is not None
                and (metrics_queue is None)
            )

            if should_show_local:
                self.dashboard.update_island(
                    island_id=self.island_id,
                    round_num=self.round_index,
                    best_quality=best_quality,
                    avg_quality=current_quality,
                    pareto_size=len(pareto),
                )

                # Skip display if we have no meaningful data yet (all zeros)
                # This happens when task returns 0 quality for all examples
                has_meaningful_data = best_quality > 0.001 or self.round_index > 0

                if has_meaningful_data:
                    # Clear inline progress and show dashboard
                    print()  # Newline to clear inline progress
                    self.dashboard.display()
                elif self.round_index == 0:
                    # First round with zeros - inform user
                    print()
                    print(f"\n   ℹ️  Baseline: {best_quality:.2%} quality on seeds (round 1)")
                    print(f"   Starting optimization to improve performance...\n")

        await self._log(
            "summary",
            {
                "round": self.round_index,
                "queue_depth": len(self.queue),
                "pareto_size": len(self.archive.pareto_candidates()),
                "evaluations": self.evaluations_run,
                "hardness_size": hardness_size,
            },
        )

    def _check_stop_governor(self) -> tuple[bool, Dict]:
        """Update stop governor with current metrics and check if we should stop."""
        if self.stop_governor is None:
            return False, {}

        # Collect current epoch metrics
        pareto_entries = self.archive.pareto_entries()

        if not pareto_entries:
            return False, {"reason": "no_pareto_candidates"}

        # Compute hypervolume (use reference point below all possible values)
        points = [
            (
                entry.result.objectives.get(self.config.promote_objective, 0.0),
                entry.result.objectives.get("neg_cost", -entry.result.objectives.get("tokens", 1000)),
            )
            for entry in pareto_entries
        ]
        # Reference point: (0 quality, -max_tokens) to ensure all points dominate it
        hypervolume = compute_hypervolume_2d(points, reference=(0.0, -self.config.max_tokens))

        # Find best candidate
        best_entry = max(
            pareto_entries,
            key=lambda e: (
                e.result.objectives.get(self.config.promote_objective, 0.0),
                e.result.objectives.get("neg_cost", float('-inf')),
            ),
        )

        # Track QD novelty
        qd_entries = list(self.archive.qd_grid.values())
        current_cells = set()
        for entry in qd_entries:
            # Create a hashable cell identifier from QD features
            cell_id = (
                int(entry.result.objectives.get("quality", 0) * 100),  # Discretize
                len(entry.candidate.text) // 100,  # Length bins
            )
            current_cells.add(cell_id)

        new_cells = current_cells - self.qd_cells_seen
        qd_novelty_rate = len(new_cells) / max(len(current_cells), 1)
        self.qd_cells_seen.update(new_cells)

        # Update token spending
        tokens_this_round = sum(
            self.latest_results.get(e.candidate.fingerprint, e.result).objectives.get("tokens", 0)
            for e in pareto_entries
        )
        self.total_tokens_spent += int(tokens_this_round)

        # Create epoch metrics
        metrics = EpochMetrics(
            round_num=self.round_index,
            hypervolume=hypervolume,
            new_evaluations=self.evaluations_run,
            best_quality=best_entry.result.objectives.get(self.config.promote_objective, 0.0),
            best_cost=best_entry.result.objectives.get("neg_cost", float('-inf')),
            frontier_ids={e.candidate.fingerprint for e in pareto_entries},
            qd_filled_cells=len(current_cells),
            qd_total_cells=self.config.qd_bins_length * self.config.qd_bins_bullets * len(self.config.qd_flags),
            qd_novelty_rate=qd_novelty_rate,
            total_tokens_spent=self.total_tokens_spent,
            shadow_significant=True,  # TODO: implement shadow shard testing
        )

        self.stop_governor.update(metrics)
        return self.stop_governor.should_stop()

    # State persistence for resumable optimization

    async def _save_state(self) -> None:
        """Save current orchestrator state to cache for resumability (non-blocking)."""
        # Cancel any previous save that's still running to avoid queue buildup
        if hasattr(self, '_save_task') and not self._save_task.done():
            self._save_task.cancel()

        pareto_candidates = self.archive.pareto_candidates()
        qd_candidates = self.archive.sample_qd(limit=len(self.archive.qd_grid))

        # Run save in background - don't block the optimization loop
        self._save_task = asyncio.create_task(
            asyncio.to_thread(
                self.cache.save_state,
                round_num=self.round_index,
                evaluations=self.evaluations_run,
                pareto_candidates=pareto_candidates,
                qd_candidates=qd_candidates,
                queue=list(self.queue),
            )
        )

    async def _restore_state(self, state: Dict) -> None:
        """Restore orchestrator state from saved checkpoint."""
        self.round_index = state["round"]
        self.evaluations_run = state["evaluations"]

        # Restore archive by re-inserting candidates with concurrency limit
        # to avoid "too many open files" errors when reading from cache
        # Note: We don't have the full EvalResult objects, so we'll need to
        # re-evaluate them (but cache will make this instant)

        # Limit concurrent restorations to avoid file descriptor exhaustion
        # Use a conservative limit since each restoration may open multiple files
        restore_semaphore = asyncio.Semaphore(5)

        async def restore_candidate(candidate: Candidate) -> None:
            async with restore_semaphore:
                # Re-evaluate to get full result (will hit cache)
                shard = self.sampler.sample_shard(self.round_index, len(self.sampler.example_ids))
                result = await self.evaluator.eval_on_shard(
                    candidate,
                    shard,
                    concurrency=self.config.eval_concurrency,
                    shard_fraction=1.0,
                )
                await self.archive.insert(candidate, result)

        # Restore all candidates with controlled parallelism
        all_candidates = state["pareto"] + state["qd"]
        await asyncio.gather(*(restore_candidate(c) for c in all_candidates))

        # Restore queue
        self.queue = deque(state["queue"], maxlen=self.config.queue_limit)
