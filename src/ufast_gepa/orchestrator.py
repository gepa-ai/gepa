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
from .merge import MergeManager
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
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.archive = archive
        self.sampler = sampler
        self.mutator = mutator
        self.cache = cache
        self.island_context = island_context
        self.scheduler = BudgetedScheduler(
            SchedulerConfig(
                shards=config.shards,
                eps_improve=config.eps_improve,
                quantile=config.cohort_quantile,
            )
        )
        self.merge_manager = MergeManager()
        self.token_controller = token_controller or TokenCostController(config.max_tokens)
        self.logger: EventLogger | _NoopLogger = logger or _NoopLogger()
        self.queue: Deque[Candidate] = deque(maxlen=config.queue_limit)
        self.latest_results: Dict[str, EvalResult] = {}
        self.evaluations_run: int = 0
        self.round_index: int = 0
        self._last_merge_round: int = -1
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
    ) -> None:
        """Execute the orchestration loop until budgets are exhausted."""

        if not seeds:
            return

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
            await self._maybe_merge()
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

        await self.finalize()

    async def finalize(self, delta: Optional[float] = None) -> None:
        """Run token-cost pruning on the Pareto frontier."""

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
        entries = self.archive.pareto_entries()
        if not entries:
            return
        remaining = self.config.max_mutations_per_round
        if remaining <= 0:
            return
        proposals = await asyncio.gather(*(self._prepare_mutations(entry) for entry in entries))
        for enriched in proposals:
            if remaining <= 0:
                break
            if not enriched:
                continue
            accepted = enriched[:remaining]
            await self._log("mutation_proposed", {"count": len(accepted)})
            self.enqueue(accepted)
            await self._log(
                "mutation_accepted",
                {
                    "candidates": [candidate.fingerprint for candidate in accepted],
                },
            )
            remaining -= len(accepted)

    async def _maybe_merge(self) -> None:
        if self.round_index - self._last_merge_round < self.config.merge_period:
            return
        entries = self.archive.pareto_entries()
        if len(entries) < 2:
            return
        merged_candidates = self.merge_manager.propose([entry.candidate for entry in entries])
        if not merged_candidates:
            return

        best_entry = max(
            entries,
            key=lambda entry: entry.result.objectives.get(self.config.promote_objective, float("-inf")),
        )
        parent_score = best_entry.result.objectives.get(self.config.promote_objective, float("-inf"))
        shard_fraction = self.config.shards[0]
        shard_size = self._shard_size(shard_fraction)
        shard = self.sampler.sample_shard(self.round_index, shard_size)

        accepted: List[Candidate] = []
        evaluations = await asyncio.gather(
            *(
                self.evaluator.eval_on_shard(
                    candidate,
                    shard,
                    concurrency=self.config.eval_concurrency,
                    shard_fraction=shard_fraction,
                )
                for candidate in merged_candidates
            )
        )
        for candidate, result in zip(merged_candidates, evaluations):
            uplift = result.objectives.get(self.config.promote_objective, float("-inf")) - parent_score
            if uplift >= self.config.merge_uplift_min:
                meta = dict(candidate.meta)
                meta["parent_objectives"] = best_entry.result.objectives
                merged_candidate = Candidate(text=candidate.text, meta=meta)
                accepted.append(merged_candidate)
                await self._log(
                    "merge_accepted",
                    {"candidate": candidate.fingerprint, "uplift": uplift},
                )
                # _handle_result now awaits archive.insert internally
                await self._handle_result((merged_candidate, result, "merged"))
            else:
                await self._log(
                    "merge_rejected",
                    {"candidate": candidate.fingerprint, "uplift": uplift},
                )
        if accepted:
            self.enqueue(accepted)
            await self._log("merge_proposed", {"count": len(accepted)})
        self._last_merge_round = self.round_index

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
        if (self.round_index + 1) % interval != 0:
            return
        hardness_size = self.sampler.hardness_size() if hasattr(self.sampler, "hardness_size") else 0

        # Update progress chart
        if self.progress_chart is not None:
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

                self.progress_chart.update(
                    round_num=self.round_index,
                    current_quality=current_quality,
                    best_quality=best_quality,
                )
                self.progress_chart.display()

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

    async def _prepare_mutations(self, entry: ArchiveEntry) -> List[Candidate]:
        failures = await self.cache.sample_failures(entry.candidate)
        mutations = await self.mutator.propose(entry.candidate, failures)
        enriched: List[Candidate] = []
        for mutation in mutations:
            meta = dict(mutation.meta)
            meta.setdefault("parent", entry.candidate.fingerprint)
            meta.setdefault("parent_objectives", entry.result.objectives)
            enriched.append(Candidate(text=mutation.text, meta=meta))
        return enriched

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
