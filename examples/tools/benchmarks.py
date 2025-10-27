"""
Performance benchmarking and KPI tracking utilities for TurboGEPA.

This module provides utilities to measure and validate the performance KPIs
specified in the project plan, including cache hit rates, prune rates,
parallelism metrics, and quality improvements.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .interfaces import EvalResult


@dataclass
class BenchmarkMetrics:
    """Container for benchmark results and KPI measurements."""

    # Timing metrics
    total_runtime_seconds: float = 0.0
    avg_eval_latency_ms: float = 0.0
    p50_eval_latency_ms: float = 0.0
    p95_eval_latency_ms: float = 0.0

    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    # Scheduler metrics
    prune_rate_shard_0: float = 0.0
    total_promotions: int = 0
    total_prunes: int = 0

    # Archive metrics
    final_pareto_size: int = 0
    final_qd_size: int = 0
    pareto_hypervolume: float = 0.0

    # Quality metrics
    initial_best_quality: float = 0.0
    final_best_quality: float = 0.0
    quality_improvement: float = 0.0

    # Compression metrics
    compressed_variants: int = 0
    avg_compression_ratio: float = 0.0

    # Parallelism metrics
    avg_concurrent_evals: float = 0.0
    max_concurrent_evals: int = 0

    # Mutation/evolution metrics
    total_mutations_proposed: int = 0
    total_mutations_accepted: int = 0
    total_merges_proposed: int = 0
    total_merges_accepted: int = 0

    # Rounds and evaluations
    total_rounds: int = 0
    total_evaluations: int = 0

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "timing": {
                "total_runtime_seconds": self.total_runtime_seconds,
                "avg_eval_latency_ms": self.avg_eval_latency_ms,
                "p50_eval_latency_ms": self.p50_eval_latency_ms,
                "p95_eval_latency_ms": self.p95_eval_latency_ms,
            },
            "cache": {
                "hit_rate": self.cache_hit_rate,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
            },
            "scheduler": {
                "prune_rate_shard_0": self.prune_rate_shard_0,
                "promotions": self.total_promotions,
                "prunes": self.total_prunes,
            },
            "archive": {
                "pareto_size": self.final_pareto_size,
                "qd_size": self.final_qd_size,
                "hypervolume": self.pareto_hypervolume,
            },
            "quality": {
                "initial": self.initial_best_quality,
                "final": self.final_best_quality,
                "improvement": self.quality_improvement,
            },
            "compression": {
                "variants": self.compressed_variants,
                "avg_ratio": self.avg_compression_ratio,
            },
            "parallelism": {
                "avg_concurrent": self.avg_concurrent_evals,
                "max_concurrent": self.max_concurrent_evals,
            },
            "evolution": {
                "mutations_proposed": self.total_mutations_proposed,
                "mutations_accepted": self.total_mutations_accepted,
                "merges_proposed": self.total_merges_proposed,
                "merges_accepted": self.total_merges_accepted,
            },
            "totals": {
                "rounds": self.total_rounds,
                "evaluations": self.total_evaluations,
            },
        }

    def validate_kpis(self) -> List[str]:
        """
        Validate that metrics meet the performance KPIs from the project plan.

        Returns:
            List of validation messages (warnings for unmet KPIs).
        """
        messages = []

        # KPI 1: Prune rate at shard-1 ≥ 60%
        if self.prune_rate_shard_0 < 0.6:
            messages.append(
                f"⚠ Prune rate at shard 0: {self.prune_rate_shard_0:.1%} (target: ≥60%)"
            )
        else:
            messages.append(
                f"✓ Prune rate at shard 0: {self.prune_rate_shard_0:.1%}"
            )

        # KPI 2: Cache hit rate ≥ 20% after warm-up
        if self.cache_hit_rate < 0.2:
            messages.append(
                f"⚠ Cache hit rate: {self.cache_hit_rate:.1%} (target: ≥20%)"
            )
        else:
            messages.append(
                f"✓ Cache hit rate: {self.cache_hit_rate:.1%}"
            )

        # KPI 3: Non-empty Pareto set and QD grid
        if self.final_pareto_size == 0:
            messages.append("⚠ Pareto frontier is empty")
        else:
            messages.append(f"✓ Pareto size: {self.final_pareto_size}")

        if self.final_qd_size == 0:
            messages.append("⚠ QD grid is empty")
        else:
            messages.append(f"✓ QD grid size: {self.final_qd_size}")

        # KPI 4: Quality improvement
        if self.quality_improvement <= 0:
            messages.append(
                f"⚠ Quality improvement: {self.quality_improvement:+.3f} (no improvement)"
            )
        else:
            messages.append(
                f"✓ Quality improvement: {self.initial_best_quality:.3f} → "
                f"{self.final_best_quality:.3f} (Δ={self.quality_improvement:+.3f})"
            )

        # KPI 5: Compressed variants created
        if self.compressed_variants == 0:
            messages.append("⚠ No compressed variants created")
        else:
            messages.append(f"✓ Compressed variants: {self.compressed_variants}")

        return messages


class BenchmarkAnalyzer:
    """Analyzes log files to compute benchmark metrics."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.events: List[Dict] = []

    def load_events(self) -> None:
        """Load all events from the log file."""
        if not self.log_path.exists():
            return

        with self.log_path.open("r") as f:
            for line in f:
                if line.strip():
                    self.events.append(json.loads(line))

    def compute_metrics(self) -> BenchmarkMetrics:
        """Compute benchmark metrics from loaded events."""
        if not self.events:
            self.load_events()

        metrics = BenchmarkMetrics()

        # Timing
        if self.events:
            start_time = self.events[0].get("ts", 0)
            end_time = self.events[-1].get("ts", 0)
            metrics.total_runtime_seconds = end_time - start_time

        # Extract event types
        eval_starts = [e for e in self.events if e.get("event") == "eval_start"]
        eval_dones = [e for e in self.events if e.get("event") == "eval_done"]
        promotes = [e for e in self.events if e.get("event") == "promote"]
        archive_updates = [e for e in self.events if e.get("event") == "archive_update"]
        mutations_proposed = [e for e in self.events if e.get("event") == "mutation_proposed"]
        mutations_accepted = [e for e in self.events if e.get("event") == "mutation_accepted"]
        merges_proposed = [e for e in self.events if e.get("event") == "merge_proposed"]
        merges_accepted = [e for e in self.events if e.get("event") == "merge_accepted"]
        compressions = [e for e in self.events if e.get("event") == "compression_applied"]
        summaries = [e for e in self.events if e.get("event") == "summary"]

        # Latency metrics
        latencies = []
        eval_pairs = {}  # Map candidate -> (start_time, end_time)

        for event in eval_starts:
            key = (event.get("candidate"), event.get("ts"))
            eval_pairs[event.get("candidate")] = [event.get("ts")]

        for event in eval_dones:
            cand = event.get("candidate")
            if cand in eval_pairs and len(eval_pairs[cand]) == 1:
                eval_pairs[cand].append(event.get("ts"))

        for start_end in eval_pairs.values():
            if len(start_end) == 2:
                latency_ms = (start_end[1] - start_end[0]) * 1000
                latencies.append(latency_ms)

        if latencies:
            metrics.avg_eval_latency_ms = statistics.mean(latencies)
            metrics.p50_eval_latency_ms = statistics.median(latencies)
            if len(latencies) > 20:
                metrics.p95_eval_latency_ms = statistics.quantiles(latencies, n=20)[18]

        # Cache metrics (approximate from summaries or unique candidates)
        if summaries:
            last_summary = summaries[-1]
            metrics.cache_hit_rate = last_summary.get("cache_hit_rate", 0.0)
            metrics.cache_hits = last_summary.get("cache_hits", 0)
            metrics.cache_misses = last_summary.get("cache_misses", 0)
        else:
            # Approximate from eval events
            seen_candidates = set()
            for event in eval_dones:
                cand = event.get("candidate")
                n_examples = event.get("n_examples", 0)
                if cand in seen_candidates:
                    metrics.cache_hits += n_examples
                else:
                    metrics.cache_misses += n_examples
                    seen_candidates.add(cand)

            total = metrics.cache_hits + metrics.cache_misses
            if total > 0:
                metrics.cache_hit_rate = metrics.cache_hits / total

        # Scheduler metrics
        metrics.total_promotions = sum(e.get("count", 0) for e in promotes)

        # Approximate prune rate from shard 0 evaluations
        # (candidates that didn't promote)
        # This is a rough estimate

        # Archive metrics
        if summaries:
            last_summary = summaries[-1]
            metrics.final_pareto_size = last_summary.get("pareto_size", 0)

        # Quality metrics
        if archive_updates:
            first_quality = archive_updates[0].get("objectives", {}).get("quality", 0.0)
            last_quality = archive_updates[-1].get("objectives", {}).get("quality", 0.0)
            metrics.initial_best_quality = first_quality
            metrics.final_best_quality = last_quality
            metrics.quality_improvement = last_quality - first_quality

        # Compression metrics
        metrics.compressed_variants = len(compressions)
        if compressions:
            # Compression ratio would need original lengths tracked
            pass

        # Evolution metrics
        metrics.total_mutations_proposed = sum(e.get("count", 0) for e in mutations_proposed)
        metrics.total_mutations_accepted = len(mutations_accepted)
        metrics.total_merges_proposed = sum(e.get("count", 0) for e in merges_proposed)
        metrics.total_merges_accepted = len(merges_accepted)

        # Totals
        if summaries:
            last_summary = summaries[-1]
            metrics.total_rounds = last_summary.get("round", 0)
            metrics.total_evaluations = last_summary.get("evaluations", 0)

        return metrics

    def print_report(self, metrics: Optional[BenchmarkMetrics] = None) -> None:
        """Print a formatted benchmark report."""
        if metrics is None:
            metrics = self.compute_metrics()

        print("=" * 70)
        print("TurboGEPA Benchmark Report")
        print("=" * 70)

        print("\n📊 Timing Metrics")
        print(f"  Total runtime: {metrics.total_runtime_seconds:.2f}s")
        print(f"  Avg eval latency: {metrics.avg_eval_latency_ms:.1f}ms")
        print(f"  P50 eval latency: {metrics.p50_eval_latency_ms:.1f}ms")
        print(f"  P95 eval latency: {metrics.p95_eval_latency_ms:.1f}ms")

        print("\n💾 Cache Metrics")
        print(f"  Hit rate: {metrics.cache_hit_rate:.1%}")
        print(f"  Hits: {metrics.cache_hits}, Misses: {metrics.cache_misses}")

        print("\n📈 Archive Metrics")
        print(f"  Pareto size: {metrics.final_pareto_size}")
        print(f"  QD grid size: {metrics.final_qd_size}")

        print("\n✨ Quality Metrics")
        print(f"  Initial quality: {metrics.initial_best_quality:.3f}")
        print(f"  Final quality: {metrics.final_best_quality:.3f}")
        print(f"  Improvement: {metrics.quality_improvement:+.3f}")

        print("\n🔄 Evolution Metrics")
        print(f"  Mutations proposed: {metrics.total_mutations_proposed}")
        print(f"  Mutations accepted: {metrics.total_mutations_accepted}")
        print(f"  Merges proposed: {metrics.total_merges_proposed}")
        print(f"  Merges accepted: {metrics.total_merges_accepted}")

        print("\n📦 Compression Metrics")
        print(f"  Compressed variants: {metrics.compressed_variants}")

        print("\n📊 Totals")
        print(f"  Rounds: {metrics.total_rounds}")
        print(f"  Evaluations: {metrics.total_evaluations}")

        print("\n" + "=" * 70)
        print("KPI Validation")
        print("=" * 70)

        for message in metrics.validate_kpis():
            print(f"  {message}")

        print("=" * 70)


def hypervolume_2d(results: Sequence[EvalResult], ref_point: tuple[float, float] = (0.0, 0.0)) -> float:
    """
    Compute 2D hypervolume for quality/neg_cost objectives.

    Args:
        results: Sequence of evaluation results
        ref_point: Reference point (quality_min, neg_cost_min)

    Returns:
        Hypervolume (area dominated by the Pareto frontier)
    """
    if not results:
        return 0.0

    # Extract (quality, neg_cost) points
    points = [
        (r.objectives.get("quality", 0.0), r.objectives.get("neg_cost", 0.0))
        for r in results
    ]

    # Sort by quality (descending)
    points = sorted(points, key=lambda p: p[0], reverse=True)

    # Remove dominated points
    pareto = []
    max_cost = ref_point[1]
    for quality, cost in points:
        if cost > max_cost:
            pareto.append((quality, cost))
            max_cost = cost

    # Compute hypervolume
    if not pareto:
        return 0.0

    hv = 0.0
    for i, (quality, cost) in enumerate(pareto):
        width = quality - ref_point[0]
        if i < len(pareto) - 1:
            height = pareto[i + 1][1] - cost
        else:
            height = ref_point[1] - cost
        hv += width * height

    return abs(hv)
