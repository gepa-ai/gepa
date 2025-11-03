"""
Comprehensive metrics tracking for TurboGEPA optimization runs.

Provides detailed instrumentation of LLM calls, cache performance,
scheduler decisions, and mutation effectiveness.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Metrics:
    """
    Comprehensive metrics for a TurboGEPA optimization run.

    Tracks performance across all major subsystems:
    - LLM calls and latency
    - Cache hit/miss rates
    - Scheduler promotions/pruning
    - Mutation generation
    - Early stopping effectiveness
    """

    # LLM Performance
    llm_calls_total: int = 0
    llm_calls_task: int = 0
    llm_calls_reflection: int = 0
    llm_calls_spec_induction: int = 0
    llm_latency_sum: float = 0.0
    llm_latency_samples: list[float] = field(default_factory=list)
    llm_timeouts: int = 0
    llm_errors: int = 0

    # Cache Performance
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0

    # Evaluation Throughput
    evaluations_total: int = 0
    evaluations_by_shard: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    concurrent_evals_peak: int = 0
    eval_time_sum: float = 0.0

    # Scheduler Decisions
    candidates_promoted: int = 0
    candidates_pruned: int = 0
    candidates_completed: int = 0
    promotions_by_rung: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # Mutation Performance
    mutations_generated: int = 0
    mutations_by_operator: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    mutation_latency_sum: float = 0.0
    mutation_batches: int = 0

    # Early Stopping
    early_stops_parent_target: int = 0
    early_stops_stragglers: int = 0
    candidates_early_stopped: int = 0

    # Operator Success Tracking
    operator_delta_quality: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    # Archive Stats
    pareto_size_max: int = 0
    qd_grid_size_max: int = 0

    # Timing Breakdown
    time_eval_total: float = 0.0
    time_mutation_total: float = 0.0
    time_scheduler_total: float = 0.0
    time_archive_total: float = 0.0

    # Round-level tracking
    round_start_times: list[float] = field(default_factory=list)
    round_durations: list[float] = field(default_factory=list)

    def record_llm_call(self, call_type: str, latency: float) -> None:
        """Record an LLM API call with timing."""
        self.llm_calls_total += 1
        self.llm_latency_sum += latency
        self.llm_latency_samples.append(latency)

        if call_type == "task":
            self.llm_calls_task += 1
        elif call_type == "reflection":
            self.llm_calls_reflection += 1
        elif call_type == "spec_induction":
            self.llm_calls_spec_induction += 1

    def record_cache_lookup(self, hit: bool) -> None:
        """Record a cache lookup result."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_cache_write(self) -> None:
        """Record a cache write."""
        self.cache_writes += 1

    def record_evaluation(self, shard_fraction: float, duration: float) -> None:
        """Record an evaluation completion."""
        self.evaluations_total += 1
        self.evaluations_by_shard[shard_fraction] += 1
        self.eval_time_sum += duration

    def record_promotion(self, from_rung: int) -> None:
        """Record a candidate promotion."""
        self.candidates_promoted += 1
        self.promotions_by_rung[from_rung] += 1

    def record_pruning(self) -> None:
        """Record a candidate being pruned."""
        self.candidates_pruned += 1

    def record_completion(self) -> None:
        """Record a candidate completing all rungs."""
        self.candidates_completed += 1

    def record_mutation(self, operator: str, latency: float) -> None:
        """Record a mutation generation."""
        self.mutations_generated += 1
        self.mutations_by_operator[operator] += 1
        self.mutation_latency_sum += latency

    def record_mutation_batch(self, count: int, latency: float) -> None:
        """Record a batch mutation generation."""
        self.mutation_batches += 1
        self.mutations_generated += count
        self.mutation_latency_sum += latency

    def record_operator_outcome(self, operator: str, delta_quality: float) -> None:
        """Record the quality improvement from a mutation operator."""
        self.operator_delta_quality[operator].append(delta_quality)

    def record_early_stop(self, reason: str) -> None:
        """Record an early stopping event."""
        if reason == "parent_target":
            self.early_stops_parent_target += 1
        elif reason == "stragglers":
            self.early_stops_stragglers += 1
        self.candidates_early_stopped += 1

    def update_concurrent_evals(self, current: int) -> None:
        """Update peak concurrent evaluations."""
        if current > self.concurrent_evals_peak:
            self.concurrent_evals_peak = current

    def update_archive_sizes(self, pareto_size: int, qd_size: int) -> None:
        """Update archive size tracking."""
        if pareto_size > self.pareto_size_max:
            self.pareto_size_max = pareto_size
        if qd_size > self.qd_grid_size_max:
            self.qd_grid_size_max = qd_size

    def start_round(self) -> None:
        """Mark the start of a new optimization round."""
        self.round_start_times.append(time.time())

    def end_round(self) -> None:
        """Mark the end of the current round."""
        if self.round_start_times:
            duration = time.time() - self.round_start_times[-1]
            self.round_durations.append(duration)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def promotion_rate(self) -> float:
        """Calculate promotion rate (promotions / total decisions)."""
        total = self.candidates_promoted + self.candidates_pruned
        return self.candidates_promoted / total if total > 0 else 0.0

    @property
    def llm_latency_mean(self) -> float:
        """Calculate mean LLM latency."""
        return self.llm_latency_sum / self.llm_calls_total if self.llm_calls_total > 0 else 0.0

    @property
    def llm_latency_p50(self) -> float:
        """Calculate 50th percentile LLM latency."""
        if not self.llm_latency_samples:
            return 0.0
        sorted_samples = sorted(self.llm_latency_samples)
        return sorted_samples[len(sorted_samples) // 2]

    @property
    def llm_latency_p95(self) -> float:
        """Calculate 95th percentile LLM latency."""
        if not self.llm_latency_samples:
            return 0.0
        sorted_samples = sorted(self.llm_latency_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def evals_per_second(self) -> float:
        """Calculate evaluation throughput."""
        return self.evaluations_total / self.eval_time_sum if self.eval_time_sum > 0 else 0.0

    @property
    def mutation_latency_mean(self) -> float:
        """Calculate mean mutation generation latency."""
        return self.mutation_latency_sum / self.mutation_batches if self.mutation_batches > 0 else 0.0

    def operator_success_rate(self, operator: str) -> float:
        """Calculate success rate for a mutation operator (% with positive delta)."""
        deltas = self.operator_delta_quality.get(operator, [])
        if not deltas:
            return 0.0
        positive = sum(1 for d in deltas if d > 0)
        return positive / len(deltas)

    def operator_mean_improvement(self, operator: str) -> float:
        """Calculate mean quality improvement for an operator."""
        deltas = self.operator_delta_quality.get(operator, [])
        return sum(deltas) / len(deltas) if deltas else 0.0

    def format_summary(self) -> str:
        """Generate a human-readable metrics summary."""
        lines = [
            "=" * 80,
            "TurboGEPA Optimization Metrics Summary",
            "=" * 80,
            "",
            "🔥 LLM Performance:",
            f"  Total calls: {self.llm_calls_total}",
            f"    - Task evaluations: {self.llm_calls_task}",
            f"    - Reflections: {self.llm_calls_reflection}",
            f"    - Spec induction: {self.llm_calls_spec_induction}",
            f"  Latency: mean={self.llm_latency_mean:.2f}s, p50={self.llm_latency_p50:.2f}s, p95={self.llm_latency_p95:.2f}s",
            f"  Timeouts: {self.llm_timeouts}, Errors: {self.llm_errors}",
            "",
            "💾 Cache Performance:",
            f"  Hit rate: {self.cache_hit_rate:.1%} ({self.cache_hits}/{self.cache_hits + self.cache_misses})",
            f"  Writes: {self.cache_writes}",
            "",
            "⚡ Evaluation Throughput:",
            f"  Total evaluations: {self.evaluations_total}",
            f"  Throughput: {self.evals_per_second:.2f} evals/sec",
            f"  Peak concurrency: {self.concurrent_evals_peak}",
            f"  By shard: {dict(self.evaluations_by_shard)}",
            "",
            "📊 Scheduler Decisions:",
            f"  Promoted: {self.candidates_promoted}",
            f"  Pruned: {self.candidates_pruned}",
            f"  Completed: {self.candidates_completed}",
            f"  Promotion rate: {self.promotion_rate:.1%}",
            f"  Promotions by rung: {dict(self.promotions_by_rung)}",
            "",
            "🔬 Mutation Generation:",
            f"  Total mutations: {self.mutations_generated}",
            f"  Batches: {self.mutation_batches}",
            f"  Mean latency: {self.mutation_latency_mean:.2f}s",
            f"  By operator: {dict(self.mutations_by_operator)}",
            "",
            "⏱️  Timing Breakdown:",
            f"  Evaluation: {self.time_eval_total:.1f}s",
            f"  Mutation: {self.time_mutation_total:.1f}s",
            f"  Scheduler: {self.time_scheduler_total:.1f}s",
            f"  Archive: {self.time_archive_total:.1f}s",
            "",
            "🎯 Operator Performance:",
        ]

        for operator in sorted(self.operator_delta_quality.keys()):
            success_rate = self.operator_success_rate(operator)
            mean_improvement = self.operator_mean_improvement(operator)
            count = len(self.operator_delta_quality[operator])
            lines.append(f"  {operator}: {success_rate:.1%} success, {mean_improvement:+.3f} mean Δ ({count} samples)")

        lines.extend([
            "",
            "🚫 Early Stopping:",
            f"  Parent target: {self.early_stops_parent_target}",
            f"  Stragglers: {self.early_stops_stragglers}",
            f"  Total candidates early-stopped: {self.candidates_early_stopped}",
            "",
            "📦 Archive:",
            f"  Max Pareto size: {self.pareto_size_max}",
            f"  Max QD grid size: {self.qd_grid_size_max}",
            "",
            "=" * 80,
        ])

        return "\n".join(lines)


# Dashboard metrics (for backward compatibility)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .orchestrator import Orchestrator


@dataclass
class DashboardMetrics:
    """Snapshot of orchestrator metrics for dashboard display."""

    timestamp: float
    round: int
    evaluations: int
    best_quality: float
    best_quality_shard: float  # Shard fraction where best quality was achieved
    avg_quality: float
    avg_quality_shard: float  # Average shard fraction across Pareto frontier
    pareto_size: int
    qd_size: int
    total_candidates: int
    rung_activity: dict[str, int]  # rung_key -> inflight count
    max_rounds: int | None = None
    max_evaluations: int | None = None
    mutations_requested: int = 0
    mutations_generated: int = 0
    mutations_enqueued: int = 0
    mutations_promoted: int = 0
    unique_parents: int = 0
    unique_children: int = 0
    evolution_edges: int = 0
    lineage_data: list[dict[str, Any]] = field(default_factory=list)


def extract_metrics(orchestrator: Orchestrator) -> DashboardMetrics:
    """
    Extract current metrics from orchestrator state for dashboard.

    This is a pure read operation - no orchestrator state is modified.
    """
    # Get archive statistics
    pareto = orchestrator.archive.pareto_entries()
    qd_elites = orchestrator.archive.sample_qd(limit=1000)  # Get all QD elites
    # Total unique candidates = pareto + QD grid
    total = len(orchestrator.archive.pareto) + len(orchestrator.archive.qd_grid)

    # Calculate best and average quality from pareto frontier
    # Track which shard the best quality came from
    promote_objective = orchestrator.config.promote_objective
    if pareto:
        best_entry = max(pareto, key=lambda e: e.result.objectives.get(promote_objective, 0.0))
        best_quality = best_entry.result.objectives.get(promote_objective, 0.0)
        best_quality_shard = best_entry.result.shard_fraction or 0.0
        avg_quality = sum(e.result.objectives.get(promote_objective, 0.0) for e in pareto) / len(pareto)
        # Calculate average shard across all pareto entries
        avg_quality_shard = sum(e.result.shard_fraction or 0.0 for e in pareto) / len(pareto)
    else:
        best_quality = 0.0
        best_quality_shard = 0.0
        avg_quality = 0.0
        avg_quality_shard = 0.0

    # Get rung activity (inflight counts per rung)
    # With priority queue, we compute this from the queue itself
    rung_activity = {}
    for _, rung_idx, _, candidate in orchestrator._priority_queue:
        rung_fraction = orchestrator.scheduler.shard_fraction_for_index(rung_idx)
        rung_key = f"{rung_fraction:.2f}"
        rung_activity[rung_key] = rung_activity.get(rung_key, 0) + 1

    evo = orchestrator.evolution_snapshot()

    # Get lineage data for visualization
    lineage_data = orchestrator.get_candidate_lineage_data()

    return DashboardMetrics(
        timestamp=time.time(),
        round=orchestrator.rounds_completed,
        evaluations=orchestrator.evaluations_run,
        best_quality=best_quality,
        best_quality_shard=best_quality_shard,
        avg_quality=avg_quality,
        avg_quality_shard=avg_quality_shard,
        pareto_size=len(pareto),
        qd_size=len(qd_elites),
        total_candidates=total,
        rung_activity=rung_activity,
        max_rounds=orchestrator.max_rounds,
        max_evaluations=orchestrator.max_evaluations,
        mutations_requested=evo["mutations_requested"],
        mutations_generated=evo["mutations_generated"],
        mutations_enqueued=evo["mutations_enqueued"],
        mutations_promoted=evo["mutations_promoted"],
        unique_parents=evo["unique_parents"],
        unique_children=evo["unique_children"],
        evolution_edges=evo["evolution_edges"],
        lineage_data=lineage_data,
    )
