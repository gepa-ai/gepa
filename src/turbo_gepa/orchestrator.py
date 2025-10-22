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
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from .archive import Archive, ArchiveEntry
from .cache import DiskCache
from .config import Config
from .evaluator import AsyncEvaluator
from .interfaces import Candidate, EvalResult
from .islands import IslandContext, integrate_in, migrate_out
from .logging_utils import EventLogger
# Merge functionality removed - reflection LLM handles combining prompt ideas
from .mutator import Mutator
from .progress_chart import ProgressChart
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
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.archive = archive
        self.sampler = sampler
        self.mutator = mutator
        self.cache = cache
        self.island_context = island_context
        self.example_sampler = example_sampler
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
        self.progress_chart = ProgressChart() if show_progress else None
        self.stop_governor = stop_governor if enable_auto_stop else None
        self.enable_auto_stop = enable_auto_stop
        self.total_tokens_spent: int = 0
        self.qd_cells_seen: set[tuple] = set()

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

        if not seeds:
            return

        # Attempt to resume from saved state
        resumed = False
        if resume and self.cache.has_state():
            state = self.cache.load_state()
            if state:
                await self._restore_state(state)
                resumed = True
                print(f"🔄 Resumed from round {self.round_index} ({self.evaluations_run} evaluations)")

        if not resumed:
            await self._seed_archive(seeds)

        while True:
            if max_rounds is not None and self.round_index >= max_rounds:
                break
            if max_evaluations is not None and self.evaluations_run >= max_evaluations:
                break

            batch = self._select_batch()
            if not batch:
                break

            results = await self._race_batch(batch)
            await self._handle_results(results)
            await self._spawn_mutations()
            # Merge removed - reflection LLM handles combining ideas from multiple prompts
            await self._maybe_migrate()
            await self._maybe_log_summary()

            # Check stop governor
            if self.enable_auto_stop and self.stop_governor is not None:
                should_stop, debug_info = self._check_stop_governor()
                if should_stop:
                    await self._log("auto_stop", debug_info)
                    if self.show_progress:
                        print(f"\n🛑 Auto-stopping: {debug_info['reason']}")
                    break

            self.round_index += 1

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
        results = await asyncio.gather(
            *[
                self._evaluate_candidate(seed, shard, shard_fraction)
                for seed in seeds
            ]
        )
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

    async def _race_batch(self, batch: Sequence[Candidate]) -> List[Tuple[Candidate, EvalResult, str]]:
        jobs = []
        for candidate in batch:
            idx = self.scheduler.current_shard_index(candidate)
            shard_fraction = self.scheduler.shard_fraction_for_index(idx)
            shard_size = self._shard_size(shard_fraction)
            shard = self.sampler.sample_shard(self.round_index, shard_size)
            jobs.append(self._evaluate_candidate(candidate, shard, shard_fraction))
        results = await asyncio.gather(*jobs)
        promotions = self.scheduler.promote_ready()
        if promotions:
            await self._log("promote", {"count": len(promotions)})
            self.enqueue(promotions)
        return results

    async def _handle_results(self, results: Sequence[Tuple[Candidate, EvalResult, str]]) -> None:
        # Parallelize result handling for 2-3x speedup (archive now has proper locking)
        await asyncio.gather(*(self._handle_result(pair) for pair in results))

        # Show progress after processing batch
        if self.show_progress:
            import sys
            pareto = self.archive.pareto_entries()
            # Fix: Read from result.objectives, not candidate.meta
            best_quality = max(e.result.objectives.get(self.config.promote_objective, 0.0) for e in pareto) if pareto else 0.0
            avg_quality = sum(e.result.objectives.get(self.config.promote_objective, 0.0) for e in pareto) / len(pareto) if pareto else 0.0
            sys.stdout.write(f"\r   🔄 Round {self.round_index} | Evals: {self.evaluations_run} | Best: {best_quality:.2%} | Avg: {avg_quality:.2%}  ")
            sys.stdout.flush()

    async def _handle_result(self, pair: Tuple[Candidate, EvalResult, str]) -> None:
        candidate, result, decision = pair
        self.latest_results[candidate.fingerprint] = result
        await self._log(
            "archive_update",
            {
                "candidate": candidate.fingerprint,
                "objectives": result.objectives,
                "n_examples": result.n_examples,
            },
        )
        await self.archive.insert(candidate, result)
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

    async def _spawn_mutations(self) -> None:
        """Generate mutations using batched reflection for efficiency."""
        entries = self.archive.pareto_entries()
        if not entries:
            return

        num_mutations = self.config.max_mutations_per_round
        if num_mutations <= 0:
            return

        mutations = await self._generate_mutations_batched(entries, num_mutations)

        if mutations:
            await self._log("mutation_proposed", {"count": len(mutations)})
            self.enqueue(mutations)
            await self._log(
                "mutation_accepted",
                {
                    "candidates": [candidate.fingerprint for candidate in mutations],
                },
            )

    async def _generate_mutations_batched(
        self,
        entries: List[ArchiveEntry],
        num_mutations: int,
    ) -> List[Candidate]:
        """Generate mutations using batched reflection."""
        # Smart parent selection: mix of top quality + QD diversity
        # Select 3-5 parents to give reflection LLM rich context
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

        # Collect failure traces for selected parents (in parallel)
        progressed_contexts: List[Dict[str, object]] = []
        fallback_contexts: List[Dict[str, object]] = []
        for entry in selected_entries:
            # Allow reflection in early rounds even if candidates haven't progressed past first shard
            # This prevents chicken-and-egg problem: need reflection to improve seeds, but ASHA kills them first
            # After round 1, require progression to avoid wasting compute on repeatedly bad candidates
            shard_progressed = (
                len(self.config.shards) <= 1  # Single shard = always allow
                or self.round_index <= 1  # First 2 rounds = always allow (give reflection a chance)
                or self.scheduler.current_shard_index(entry.candidate) > 0  # Later rounds = must progress
            )
            if shard_progressed:
                failures = await self.cache.sample_failures(entry.candidate)
                progressed_contexts.append({
                    "candidate": entry.candidate,
                    "failures": failures,
                })
            else:
                fallback_contexts.append({
                    "candidate": entry.candidate,
                    "failures": [],
                })

        parent_contexts = progressed_contexts or fallback_contexts
        if not parent_contexts:
            return []

        # Sample task examples for spec induction (3 examples)
        task_examples = self._sample_task_examples_for_spec_induction(num_examples=3)

        # Hybrid reflection: runs incremental + spec induction concurrently
        mutations = await self.mutator.propose(parent_contexts, num_mutations, task_examples=task_examples)

        await self._log(
            "batch_reflection",
            {
                "num_parents": len(parent_contexts),
                "num_mutations_requested": num_mutations,
                "num_mutations_generated": len(mutations),
            },
        )

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
        if self.round_index == 0:
            return
        if self.round_index % self.config.migration_period != 0:
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
        )
        self.evaluations_run += result.n_examples
        decision = self.scheduler.record(candidate, result, objective_key=self.config.promote_objective)
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

    def _shard_size(self, shard_fraction: float) -> int:
        total = max(len(self.sampler.example_ids), 1)
        size = max(1, int(total * shard_fraction))
        return min(size, total)

    async def _log(self, event_type: str, payload: Dict[str, object]) -> None:
        await self.logger.log(event_type, payload)

    async def _maybe_log_summary(self) -> None:
        interval = max(self.config.log_summary_interval, 0)
        if interval == 0:
            return

        # Adaptive display frequency: More frequent early, less frequent later
        # Round 0: Always show (baseline)
        # Rounds 1-5: Every round (rapid iteration feedback)
        # Rounds 6-20: Every 3 rounds (still learning fast)
        # Rounds 21+: Every interval rounds (config default: 10)
        should_display = False
        if self.round_index == 0:
            should_display = True  # Always show seed baseline
        elif self.round_index <= 5:
            should_display = True  # Show every round early on
        elif self.round_index <= 20:
            should_display = (self.round_index % 3 == 0)  # Every 3 rounds
        else:
            should_display = ((self.round_index + 1) % interval == 0)  # Normal interval

        if not should_display:
            return

        hardness_size = self.sampler.hardness_size() if hasattr(self.sampler, "hardness_size") else 0

        # Calculate metrics (needed for both chart display and island dashboard)
        pareto = self.archive.pareto_entries()
        if pareto:
            # Get best quality from Pareto
            best_quality = max(
                entry.result.objectives.get(self.config.promote_objective, 0.0)
                for entry in pareto
            )
            # Get current average quality
            current_quality = sum(
                entry.result.objectives.get(self.config.promote_objective, 0.0)
                for entry in pareto
            ) / len(pareto)

            # Send metrics to island dashboard if available (even if not showing local chart)
            if self.island_context and self.island_context.metrics_queue:
                try:
                    self.island_context.metrics_queue.put_nowait({
                        "island_id": self.island_context.island_id,
                        "round": self.round_index,
                        "best_quality": best_quality,
                        "avg_quality": current_quality,
                        "pareto_size": len(pareto),
                    })
                except asyncio.QueueFull:
                    pass  # Dashboard fell behind, skip this update

            # Update and display progress chart if enabled
            if self.progress_chart is not None:
                self.progress_chart.update(
                    round_num=self.round_index,
                    current_quality=current_quality,
                    best_quality=best_quality,
                )

                # Skip display if we have no meaningful data yet (all zeros)
                # This happens when task returns 0 quality for all examples
                has_meaningful_data = best_quality > 0.001 or len(self.progress_chart.quality_history) > 1

                if has_meaningful_data:
                    # Clear inline progress and show full chart
                    print()  # Newline to clear inline progress
                    self.progress_chart.display()
                elif self.round_index == 0:
                    # First round with zeros - inform user
                    print()
                    print(f"\n   ℹ️  Baseline: {best_quality:.2%} quality on seeds (round 0)")
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

        # Restore archive by re-inserting candidates (parallelized)
        # Note: We don't have the full EvalResult objects, so we'll need to
        # re-evaluate them (but cache will make this instant)
        async def restore_candidate(candidate: Candidate) -> None:
            # Re-evaluate to get full result (will hit cache)
            shard = self.sampler.sample_shard(self.round_index, len(self.sampler.example_ids))
            result = await self.evaluator.eval_on_shard(
                candidate,
                shard,
                concurrency=self.config.eval_concurrency,
                shard_fraction=1.0,
            )
            await self.archive.insert(candidate, result)

        # Restore all candidates in parallel
        all_candidates = state["pareto"] + state["qd"]
        await asyncio.gather(*(restore_candidate(c) for c in all_candidates))

        # Restore queue
        self.queue = deque(state["queue"], maxlen=self.config.queue_limit)
