"""
Comprehensive metrics tracking for TurboGEPA optimization runs.

Provides detailed instrumentation of LLM calls, cache performance,
scheduler decisions, and mutation effectiveness.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


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
    concurrent_budget_clamps: int = 0
    eval_time_sum: float = 0.0
    shard_outcomes: dict[float, dict[str, float]] = field(default_factory=dict)

    # Scheduler Decisions
    candidates_promoted: int = 0
    candidates_pruned: int = 0
    candidates_completed: int = 0
    promotions_by_rung: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    promotion_attempts_by_rung: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    promotion_pruned_by_rung: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # Mutation Performance
    mutations_generated: int = 0
    mutations_by_operator: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    mutation_latency_sum: float = 0.0
    mutation_batches: int = 0
    # Strategy call tracking (reflection variants)
    strategy_call_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    strategy_latency_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # Early Stopping
    early_stops_parent_target: int = 0
    early_stops_stragglers: int = 0
    early_stops_target_quality: int = 0
    final_rung_early_stops_target_quality: int = 0
    candidates_early_stopped: int = 0

    # Final-rung evaluation diagnostics
    final_rung_launches: int = 0
    final_rung_completions: int = 0
    final_rung_inflight_peak: int = 0
    final_rung_inflight_sum: int = 0
    final_rung_inflight_samples: int = 0

    # Operator Success Tracking
    operator_delta_quality: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    # Archive Stats
    pareto_size_max: int = 0

    # Timing Breakdown
    time_eval_total: float = 0.0
    time_mutation_total: float = 0.0
    time_scheduler_total: float = 0.0
    time_archive_total: float = 0.0

    # Round-level tracking
    round_start_times: list[float] = field(default_factory=list)
    round_durations: list[float] = field(default_factory=list)
    baseline_quality: float = 0.0
    baseline_recorded: bool = False
    target_quality: float | None = None
    target_shard_fraction: float | None = None
    time_to_target_seconds: float | None = None
    best_quality: float = 0.0
    best_shard_fraction: float = 0.0
    highest_rung_fraction: float = 0.0
    best_rung_quality: float = 0.0
    time_to_best_rung: float | None = None
    rung_baselines: dict[float, float] = field(default_factory=dict)

    def record_rung_sample(self, shard_fraction: float, quality: float, elapsed: float | None) -> None:
        shard_fraction = round(max(0.0, shard_fraction), 4)
        if shard_fraction not in self.rung_baselines:
            self.rung_baselines[shard_fraction] = quality
        target_shard = self.target_shard_fraction or 1.0
        if (
            not self.baseline_recorded
            and shard_fraction + 1e-6 >= target_shard
        ):
            self.baseline_quality = quality if self.baseline_quality == 0.0 else self.baseline_quality
            self.baseline_recorded = True
        eps = 1e-6
        if (
            shard_fraction > self.highest_rung_fraction + eps
            or (
                abs(shard_fraction - self.highest_rung_fraction) <= eps
                and quality > self.best_rung_quality
            )
        ):
            self.highest_rung_fraction = shard_fraction
            self.best_rung_quality = quality
            if elapsed is not None:
                self.time_to_best_rung = max(elapsed, 0.0)
    baseline_quality: float = 0.0
    target_quality: float | None = None
    time_to_target_seconds: float | None = None

    def turbo_score(self) -> float | None:
        if (
            self.target_quality is None
            or self.time_to_target_seconds is None
            or self.time_to_target_seconds <= 0
            or not self.baseline_recorded
        ):
            return None
        gain = self.target_quality - self.baseline_quality
        if gain <= 0:
            return 0.0
        return gain / self.time_to_target_seconds

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

    def record_shard_outcome(
        self,
        shard_fraction: float | None,
        coverage: float,
        stragglers_cancelled: int,
        duration: float,
    ) -> None:
        """Record aggregate stats for a shard evaluation batch."""
        key = shard_fraction if shard_fraction is not None else -1.0
        stats = self.shard_outcomes.setdefault(
            key,
            {
                "count": 0,
                "coverage_sum": 0.0,
                "stragglers": 0,
                "duration_sum": 0.0,
            },
        )
        stats["count"] += 1
        stats["coverage_sum"] += coverage
        stats["stragglers"] += stragglers_cancelled
        stats["duration_sum"] += duration

    def record_promotion_attempt(self, rung: int) -> None:
        """Record that the scheduler made a decision on this rung."""
        self.promotion_attempts_by_rung[rung] += 1

    def record_promotion(self, from_rung: int) -> None:
        """Record a candidate promotion."""
        self.candidates_promoted += 1
        self.promotions_by_rung[from_rung] += 1

    def record_pruning(self, rung: int | None = None) -> None:
        """Record a candidate being pruned."""
        self.candidates_pruned += 1
        if rung is not None:
            self.promotion_pruned_by_rung[rung] += 1

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

    def record_early_stop(self, reason: str, *, is_final_rung: bool = False) -> None:
        """Record an early stopping event."""
        if reason == "parent_target":
            self.early_stops_parent_target += 1
        elif reason == "stragglers":
            self.early_stops_stragglers += 1
        elif reason == "target_quality":
            self.early_stops_target_quality += 1
            if is_final_rung:
                self.final_rung_early_stops_target_quality += 1
        self.candidates_early_stopped += 1

    def record_final_rung_launch(self) -> None:
        """Record launch of a final-rung evaluation."""
        self.final_rung_launches += 1

    def record_final_rung_completion(self) -> None:
        """Record completion of a final-rung evaluation."""
        self.final_rung_completions += 1

    def record_final_rung_inflight(self, current: int) -> None:
        """Track inflight concurrency for final rung."""
        if current > self.final_rung_inflight_peak:
            self.final_rung_inflight_peak = current
        self.final_rung_inflight_samples += 1
        self.final_rung_inflight_sum += max(0, int(current))

    def update_concurrent_evals(self, current: int) -> None:
        """Update peak concurrent evaluations."""
        if current > self.concurrent_evals_peak:
            self.concurrent_evals_peak = current

    def record_budget_clamp(self) -> None:
        self.concurrent_budget_clamps += 1

    def update_archive_sizes(self, pareto_size: int) -> None:
        """Update archive size tracking."""
        if pareto_size > self.pareto_size_max:
            self.pareto_size_max = pareto_size

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

    @property
    def final_rung_inflight_mean(self) -> float:
        """Average inflight final-rung concurrency."""
        if self.final_rung_inflight_samples <= 0:
            return 0.0
        return self.final_rung_inflight_sum / self.final_rung_inflight_samples

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

    def record_strategy_call(self, name: str, latency: float) -> None:
        self.strategy_call_counts[name] += 1
        self.strategy_latency_sum[name] += max(0.0, float(latency))

    def strategy_latency_mean(self, name: str) -> float:
        n = self.strategy_call_counts.get(name, 0)
        if n <= 0:
            return 0.0
        return self.strategy_latency_sum.get(name, 0.0) / n

    def turbo_score(self) -> float | None:
        if (
            self.target_quality is None
            or self.time_to_target_seconds is None
            or self.time_to_target_seconds <= 0
        ):
            return None
        gain = self.target_quality - self.baseline_quality
        if gain <= 0:
            return 0.0
        return gain / self.time_to_target_seconds

    def snapshot(self) -> dict[str, Any]:
        """
        Return a JSON-serializable summary of key KPI metrics so dashboards/CI can
        assert time-to-target and promotion health without re-computing anything.
        """
        return {
            "time_to_target_seconds": self.time_to_target_seconds,
            "turbo_score": self.turbo_score(),
            "target_quality": self.target_quality,
            "target_shard_fraction": self.target_shard_fraction,
            "baseline_quality": self.baseline_quality,
            "best_quality": self.best_quality,
            "best_shard_fraction": self.best_shard_fraction,
            "time_to_best_rung": self.time_to_best_rung,
            "evaluations_total": self.evaluations_total,
            "evaluation_throughput": self.evals_per_second,
            "candidates_promoted": self.candidates_promoted,
            "candidates_pruned": self.candidates_pruned,
            "candidates_completed": self.candidates_completed,
            "promotions_by_rung": dict(self.promotions_by_rung),
            "promotion_attempts_by_rung": dict(self.promotion_attempts_by_rung),
            "promotion_pruned_by_rung": dict(self.promotion_pruned_by_rung),
            "llm_calls_total": self.llm_calls_total,
            "llm_latency_mean": self.llm_latency_mean,
            "llm_latency_p50": self.llm_latency_p50,
            "llm_latency_p95": self.llm_latency_p95,
            "cache_hit_rate": self.cache_hit_rate,
            "concurrent_evals_peak": self.concurrent_evals_peak,
            "concurrent_budget_clamps": self.concurrent_budget_clamps,
            "mutations_generated": self.mutations_generated,
            "mutation_batches": self.mutation_batches,
            "mutation_latency_mean": self.mutation_latency_mean,
            "time_eval_total": self.time_eval_total,
            "time_mutation_total": self.time_mutation_total,
            "time_scheduler_total": self.time_scheduler_total,
            "time_archive_total": self.time_archive_total,
            "final_rung_launches": self.final_rung_launches,
            "final_rung_completions": self.final_rung_completions,
            "final_rung_inflight_peak": self.final_rung_inflight_peak,
            "final_rung_inflight_mean": self.final_rung_inflight_mean,
            "early_stops_target_quality": self.early_stops_target_quality,
            "final_rung_early_stops_target_quality": self.final_rung_early_stops_target_quality,
        }

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
            "🧩 Final-Rung Evaluation:",
            f"  Launches: {self.final_rung_launches}, completions: {self.final_rung_completions}",
            f"  Inflight peak: {self.final_rung_inflight_peak}, mean inflight: {self.final_rung_inflight_mean:.1f}",
            "",
            "📏 Shard Outcomes:",
        ]

        if not self.shard_outcomes:
            lines.append("  (no shard outcome data yet)")
        else:
            for shard, stats in sorted(self.shard_outcomes.items(), key=lambda kv: kv[0]):
                count = stats["count"]
                avg_cov = stats["coverage_sum"] / count if count else 0.0
                avg_duration = stats["duration_sum"] / count if count else 0.0
                lines.append(
                    f"  shard={shard:.2f}: coverage={avg_cov:.1%}, "
                    f"stragglers={int(stats['stragglers'])}, batches={count}, "
                    f"mean_duration={avg_duration:.1f}s"
                )

        lines.extend([
            "",
            "📊 Scheduler Decisions:",
            f"  Promoted: {self.candidates_promoted}",
            f"  Pruned: {self.candidates_pruned}",
            f"  Completed: {self.candidates_completed}",
            f"  Promotion rate: {self.promotion_rate:.1%}",
            f"  Promotions by rung: {dict(self.promotions_by_rung)}",
            "",
        ])

        if self.promotion_attempts_by_rung:
            lines.append("  Rung promotion stats:")
            for rung in sorted(self.promotion_attempts_by_rung):
                attempts = self.promotion_attempts_by_rung[rung]
                promoted = self.promotions_by_rung.get(rung, 0)
                pruned = self.promotion_pruned_by_rung.get(rung, 0)
                rate = promoted / attempts if attempts else 0.0
                lines.append(
                    f"    rung {rung}: attempts={attempts}, promoted={promoted}, "
                    f"pruned={pruned}, rate={rate:.1%}"
                )
        lines.extend([
            "",
            "🔬 Mutation Generation:",
            f"  Total mutations: {self.mutations_generated}",
            f"  Batches: {self.mutation_batches}",
            f"  Mean latency: {self.mutation_latency_mean:.2f}s",
            f"  By operator: {dict(self.mutations_by_operator)}",
            "",
            "🧠 Reflection Strategies:",
            f"  Calls: {dict(self.strategy_call_counts)}",
            f"  Mean latency by strategy: {{" + ", ".join(f"{k}: {self.strategy_latency_mean(k):.2f}s" for k in self.strategy_call_counts) + "}}",
            "",
            "⏱️  Timing Breakdown:",
            f"  Evaluation: {self.time_eval_total:.1f}s",
            f"  Mutation: {self.time_mutation_total:.1f}s",
            f"  Scheduler: {self.time_scheduler_total:.1f}s",
            f"  Archive: {self.time_archive_total:.1f}s",
            "",
            "🎯 Operator Performance:",
        ])

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
            f"  Target quality (all rungs): {self.early_stops_target_quality}",
            f"  Target quality (final rung): {self.final_rung_early_stops_target_quality}",
            f"  Total candidates early-stopped: {self.candidates_early_stopped}",
            "",
            "📦 Archive:",
            f"  Max Pareto size: {self.pareto_size_max}",
            "",
        ])

        if self.target_quality is not None:
            target_shard = self.target_shard_fraction if self.target_shard_fraction is not None else 1.0
            baseline_str = f"{self.baseline_quality:.3f}" if self.baseline_recorded else "n/a"
            if self.time_to_target_seconds is None:
                lines.extend([
                    "🚀 Turbo Metric:",
                    f"  Target shard: {target_shard:.2f}",
                    f"  Baseline={baseline_str} → Target={self.target_quality:.3f}",
                    "  Time to target: not reached",
                    "  Turbo score: n/a",
                    "",
                ])
            else:
                score = self.turbo_score()
                score_str = f"{score:.4f}" if score is not None else "n/a"
                lines.extend([
                    "🚀 Turbo Metric:",
                    f"  Target shard: {target_shard:.2f}",
                    f"  Baseline={baseline_str} → Target={self.target_quality:.3f}",
                    f"  Time to target: {self.time_to_target_seconds:.1f}s",
                    f"  Turbo score: {score_str}",
                    "",
                ])

        if self.time_to_target_seconds is None and self.time_to_best_rung is not None:
            rung = self.highest_rung_fraction
            baseline_rung = self.rung_baselines.get(round(rung, 4))
            rung_score = None
            if baseline_rung is not None and self.time_to_best_rung > 0:
                gain = self.best_rung_quality - baseline_rung
                rung_score = gain / self.time_to_best_rung if gain > 0 else 0.0
            lines.extend([
                "🚀 Rung Metric:",
                f"  Rung={rung:.2f} quality={self.best_rung_quality:.3f}",
                f"  Time to rung: {self.time_to_best_rung:.1f}s",
                f"  Rung score: {rung_score:.4f}" if rung_score is not None else "  Rung score: n/a",
                "",
            ])

        lines.extend([
            "Best observed:",
            f"  Quality={self.best_quality:.3f} @ shard {self.best_shard_fraction:.2f}",
            "",
        ])

        lines.append("=" * 80)

        return "\n".join(lines)
