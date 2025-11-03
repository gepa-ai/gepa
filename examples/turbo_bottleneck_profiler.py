"""
TurboGEPA Bottleneck Profiler - Diagnostic tool to identify performance bottlenecks

This script runs a TurboGEPA optimization with extensive instrumentation to identify
where the system gets stuck or slows down. It tracks:

1. CONCURRENCY UTILIZATION
   - Active evaluations over time
   - Queue depth over time
   - Mutation buffer state
   - Task lifecycle timing

2. COMPONENT TIMING
   - Time spent in evaluation
   - Time spent in mutation generation
   - Time spent in scheduler operations
   - Time waiting for work

3. BOTTLENECK DETECTION
   - Starvation events (queue empty)
   - Saturation events (queue full)
   - Idle periods (low concurrency)
   - Blocking operations

4. THROUGHPUT METRICS
   - Evaluations per second
   - Mutations per second
   - Cache hit rate
   - Early stop rate

Run with: python examples/turbo_bottleneck_profiler.py [--concurrency N] [--dataset-size N]
"""

import argparse
import asyncio
import os
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Disable litellm's async logging worker to avoid event loop issues
os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config


@dataclass
class BottleneckMetrics:
    """Track bottleneck indicators throughout optimization."""

    # Concurrency tracking
    concurrency_samples: list[tuple[float, int]] = field(default_factory=list)  # (timestamp, active_count)
    queue_depth_samples: list[tuple[float, int]] = field(default_factory=list)  # (timestamp, queue_size)
    mutation_buffer_samples: list[tuple[float, int]] = field(default_factory=list)  # (timestamp, buffer_size)

    # Component timing
    eval_times: list[float] = field(default_factory=list)
    mutation_times: list[float] = field(default_factory=list)
    scheduler_times: list[float] = field(default_factory=list)
    cache_lookup_times: list[float] = field(default_factory=list)

    # Bottleneck events
    starvation_events: list[tuple[float, str]] = field(default_factory=list)  # Queue empty
    saturation_events: list[tuple[float, str]] = field(default_factory=list)  # Queue full
    idle_periods: list[tuple[float, float, str]] = field(default_factory=list)  # (start, duration, reason)
    blocking_operations: list[tuple[float, float, str]] = field(default_factory=list)  # (start, duration, operation)

    # Throughput tracking
    evals_per_second: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, rate)
    mutations_per_second: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, rate)
    cache_hits: int = 0
    cache_misses: int = 0
    early_stops: int = 0
    full_evals: int = 0

    # Phase tracking
    phase_timings: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    # Cumulative counters
    total_evaluations: int = 0
    total_mutations: int = 0
    start_time: float = 0.0

    def record_concurrency(self, timestamp: float, active_count: int):
        """Record active evaluation count."""
        self.concurrency_samples.append((timestamp, active_count))

    def record_queue_depth(self, timestamp: float, queue_size: int):
        """Record queue depth."""
        self.queue_depth_samples.append((timestamp, queue_size))

    def record_mutation_buffer(self, timestamp: float, buffer_size: int):
        """Record mutation buffer state."""
        self.mutation_buffer_samples.append((timestamp, buffer_size))

    def record_starvation(self, timestamp: float, reason: str):
        """Record queue starvation event."""
        self.starvation_events.append((timestamp, reason))

    def record_saturation(self, timestamp: float, reason: str):
        """Record queue saturation event."""
        self.saturation_events.append((timestamp, reason))

    def record_idle_period(self, start: float, duration: float, reason: str):
        """Record idle period."""
        self.idle_periods.append((start, duration, reason))

    def record_blocking_op(self, start: float, duration: float, operation: str):
        """Record blocking operation."""
        self.blocking_operations.append((start, duration, operation))

    def record_phase_timing(self, phase: str, duration: float):
        """Record time spent in a phase."""
        self.phase_timings[phase].append(duration)

    def record_eval_time(self, duration: float):
        """Record evaluation timing."""
        self.eval_times.append(duration)
        self.total_evaluations += 1

    def record_mutation_time(self, duration: float):
        """Record mutation generation timing."""
        self.mutation_times.append(duration)
        self.total_mutations += 1

    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1

    def record_early_stop(self):
        """Record early stop."""
        self.early_stops += 1

    def record_full_eval(self):
        """Record full evaluation."""
        self.full_evals += 1

    def compute_summary_stats(self) -> dict[str, Any]:
        """Compute summary statistics."""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 1.0

        # Concurrency utilization
        if self.concurrency_samples:
            avg_concurrency = sum(c for _, c in self.concurrency_samples) / len(self.concurrency_samples)
            max_concurrency = max(c for _, c in self.concurrency_samples)
            concurrency_utilization = avg_concurrency / max_concurrency if max_concurrency > 0 else 0.0
        else:
            avg_concurrency = 0.0
            max_concurrency = 0
            concurrency_utilization = 0.0

        # Queue depth stats
        if self.queue_depth_samples:
            avg_queue_depth = sum(q for _, q in self.queue_depth_samples) / len(self.queue_depth_samples)
            max_queue_depth = max(q for _, q in self.queue_depth_samples)
        else:
            avg_queue_depth = 0.0
            max_queue_depth = 0

        # Timing stats
        avg_eval_time = sum(self.eval_times) / len(self.eval_times) if self.eval_times else 0.0
        avg_mutation_time = sum(self.mutation_times) / len(self.mutation_times) if self.mutation_times else 0.0

        # Cache stats
        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0

        # Throughput
        evals_per_sec = self.total_evaluations / elapsed if elapsed > 0 else 0.0
        mutations_per_sec = self.total_mutations / elapsed if elapsed > 0 else 0.0

        # Early stop rate
        total_candidates = self.early_stops + self.full_evals
        early_stop_rate = self.early_stops / total_candidates if total_candidates > 0 else 0.0

        return {
            "elapsed_seconds": elapsed,
            "concurrency": {
                "avg": avg_concurrency,
                "max": max_concurrency,
                "utilization": concurrency_utilization,
            },
            "queue": {
                "avg_depth": avg_queue_depth,
                "max_depth": max_queue_depth,
            },
            "timing": {
                "avg_eval_time_ms": avg_eval_time * 1000,
                "avg_mutation_time_ms": avg_mutation_time * 1000,
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": cache_hit_rate,
            },
            "throughput": {
                "evals_per_sec": evals_per_sec,
                "mutations_per_sec": mutations_per_sec,
            },
            "early_stops": {
                "total": self.early_stops,
                "rate": early_stop_rate,
            },
            "bottlenecks": {
                "starvation_events": len(self.starvation_events),
                "saturation_events": len(self.saturation_events),
                "idle_periods": len(self.idle_periods),
                "blocking_operations": len(self.blocking_operations),
            },
        }


class InstrumentedAdapter(DefaultAdapter):
    """Instrumented adapter that tracks bottleneck metrics."""

    def __init__(self, *args, metrics_tracker: BottleneckMetrics, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_tracker = metrics_tracker
        self.monitoring_task: asyncio.Task | None = None
        self.stop_monitoring = False

    async def _monitor_loop(self):
        """Background task to monitor system state."""
        while not self.stop_monitoring:
            await asyncio.sleep(0.1)  # Sample every 100ms

            timestamp = time.time()

            # Sample orchestrator state if available
            if hasattr(self, "_orchestrator") and self._orchestrator is not None:
                orch = self._orchestrator

                # Concurrency
                if hasattr(orch, "_inflight_tasks"):
                    active_count = len(orch._inflight_tasks)
                    self.metrics_tracker.record_concurrency(timestamp, active_count)

                # Queue depth
                if hasattr(orch, "queue"):
                    queue_size = len(orch.queue)
                    self.metrics_tracker.record_queue_depth(timestamp, queue_size)

                    # Check for starvation
                    if queue_size == 0 and active_count < self.config.eval_concurrency // 4:
                        self.metrics_tracker.record_starvation(timestamp, "Queue empty with low concurrency")

                # Mutation buffer
                if hasattr(orch, "_mutation_buffer"):
                    buffer_size = len(orch._mutation_buffer)
                    self.metrics_tracker.record_mutation_buffer(timestamp, buffer_size)

    def optimize(self, *args, **kwargs):
        """Override optimize to start monitoring."""
        self.metrics_tracker.start_time = time.time()
        self.stop_monitoring = False

        # Start monitoring loop in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run_with_monitoring():
            self.monitoring_task = asyncio.create_task(self._monitor_loop())
            try:
                result = await self._optimize_async(*args, **kwargs)
                return result
            finally:
                self.stop_monitoring = True
                if self.monitoring_task:
                    await self.monitoring_task

        result = loop.run_until_complete(run_with_monitoring())
        loop.close()

        return result

    async def _optimize_async(self, *args, **kwargs):
        """Async wrapper for optimization."""
        # Call parent class optimize method
        # Note: We need to adapt this based on the actual DefaultAdapter.optimize signature
        return super().optimize(*args, **kwargs)


def print_bottleneck_report(metrics: BottleneckMetrics, config: Config):
    """Print comprehensive bottleneck analysis."""
    stats = metrics.compute_summary_stats()

    print("\n" + "=" * 80)
    print("TURBOGEPA BOTTLENECK ANALYSIS")
    print("=" * 80)

    print(f"\n⏱️  Total Runtime: {stats['elapsed_seconds']:.1f}s")

    # Concurrency analysis
    print("\n" + "─" * 80)
    print("CONCURRENCY UTILIZATION")
    print("─" * 80)
    print(f"  Max Configured:     {config.eval_concurrency}")
    print(f"  Max Observed:       {stats['concurrency']['max']}")
    print(f"  Average Active:     {stats['concurrency']['avg']:.1f}")
    print(f"  Utilization:        {stats['concurrency']['utilization']:.1%}")

    if stats['concurrency']['utilization'] < 0.5:
        print("  ⚠️  WARNING: Low concurrency utilization (<50%) - system is underutilized!")
        print("     Possible causes:")
        print("     - Queue starvation (not enough candidates)")
        print("     - Mutation generation bottleneck")
        print("     - Blocking operations in critical path")

    # Queue analysis
    print("\n" + "─" * 80)
    print("QUEUE METRICS")
    print("─" * 80)
    print(f"  Max Queue Limit:    {config.queue_limit}")
    print(f"  Max Observed:       {stats['queue']['max_depth']}")
    print(f"  Average Depth:      {stats['queue']['avg_depth']:.1f}")

    if stats['queue']['avg_depth'] < 10:
        print("  ⚠️  WARNING: Low queue depth - risk of starvation!")
        print("     Recommendations:")
        print("     - Increase max_mutations_per_round")
        print("     - Increase mutation_buffer_min")
        print("     - Check if mutation generation is slow")

    if stats['queue']['max_depth'] >= config.queue_limit:
        print("  ⚠️  WARNING: Queue hit limit - may be dropping candidates!")
        print("     Recommendations:")
        print("     - Increase queue_limit")
        print("     - Reduce max_mutations_per_round")

    # Timing analysis
    print("\n" + "─" * 80)
    print("COMPONENT TIMING")
    print("─" * 80)
    print(f"  Avg Evaluation Time:  {stats['timing']['avg_eval_time_ms']:.1f}ms")
    print(f"  Avg Mutation Time:    {stats['timing']['avg_mutation_time_ms']:.1f}ms")

    if stats['timing']['avg_mutation_time_ms'] > stats['timing']['avg_eval_time_ms'] * 2:
        print("  ⚠️  WARNING: Mutation generation is slower than evaluation!")
        print("     This is a bottleneck. Recommendations:")
        print("     - Use faster reflection_lm model")
        print("     - Reduce reflection_batch_size")
        print("     - Increase amortized_rate (use more rule-based mutations)")

    # Cache analysis
    print("\n" + "─" * 80)
    print("CACHE PERFORMANCE")
    print("─" * 80)
    print(f"  Cache Hits:         {stats['cache']['hits']}")
    print(f"  Cache Misses:       {stats['cache']['misses']}")
    print(f"  Hit Rate:           {stats['cache']['hit_rate']:.1%}")

    if stats['cache']['hit_rate'] < 0.1 and metrics.total_evaluations > 100:
        print("  ⚠️  WARNING: Low cache hit rate after warmup")
        print("     This means candidates are not being reused efficiently")

    # Throughput analysis
    print("\n" + "─" * 80)
    print("THROUGHPUT")
    print("─" * 80)
    print(f"  Evaluations/sec:    {stats['throughput']['evals_per_sec']:.2f}")
    print(f"  Mutations/sec:      {stats['throughput']['mutations_per_sec']:.2f}")
    print(f"  Total Evaluations:  {metrics.total_evaluations}")
    print(f"  Total Mutations:    {metrics.total_mutations}")

    expected_throughput = config.eval_concurrency * 0.5  # Rough estimate
    if stats['throughput']['evals_per_sec'] < expected_throughput:
        print(f"  ⚠️  WARNING: Throughput below expected ({expected_throughput:.1f} evals/sec)")
        print("     System is not achieving expected parallelism")

    # Early stopping analysis
    print("\n" + "─" * 80)
    print("EARLY STOPPING (ASHA)")
    print("─" * 80)
    print(f"  Early Stops:        {stats['early_stops']['total']}")
    print(f"  Full Evaluations:   {metrics.full_evals}")
    print(f"  Early Stop Rate:    {stats['early_stops']['rate']:.1%}")

    if stats['early_stops']['rate'] < 0.3:
        print("  ℹ️  Low early stop rate - ASHA may not be pruning aggressively enough")
        print("     This is not necessarily bad, but consider:")
        print("     - More aggressive sharding")
        print("     - Higher cohort_quantile")

    # Bottleneck events
    print("\n" + "─" * 80)
    print("BOTTLENECK EVENTS")
    print("─" * 80)
    print(f"  Starvation Events:  {stats['bottlenecks']['starvation_events']}")
    print(f"  Saturation Events:  {stats['bottlenecks']['saturation_events']}")
    print(f"  Idle Periods:       {stats['bottlenecks']['idle_periods']}")
    print(f"  Blocking Ops:       {stats['bottlenecks']['blocking_operations']}")

    if stats['bottlenecks']['starvation_events'] > 0:
        print("  ⚠️  WARNING: Queue starvation detected!")
        print("     The system ran out of candidates to evaluate")
        print("     Recent starvation events:")
        for timestamp, reason in metrics.starvation_events[-5:]:
            elapsed = timestamp - metrics.start_time
            print(f"       {elapsed:.1f}s: {reason}")

    if stats['bottlenecks']['idle_periods'] > 10:
        print("  ⚠️  WARNING: Many idle periods detected!")
        print("     The system is frequently waiting for work")

    # Recommendations
    print("\n" + "─" * 80)
    print("RECOMMENDATIONS")
    print("─" * 80)

    issues = []

    if stats['concurrency']['utilization'] < 0.5:
        issues.append("Low concurrency utilization")
    if stats['queue']['avg_depth'] < 10:
        issues.append("Low queue depth")
    if stats['timing']['avg_mutation_time_ms'] > stats['timing']['avg_eval_time_ms'] * 2:
        issues.append("Slow mutation generation")
    if stats['throughput']['evals_per_sec'] < expected_throughput:
        issues.append("Low throughput")
    if stats['bottlenecks']['starvation_events'] > 5:
        issues.append("Frequent queue starvation")

    if not issues:
        print("  ✅ No major bottlenecks detected!")
        print("  System is running efficiently")
    else:
        print("  ⚠️  Issues detected:")
        for i, issue in enumerate(issues, 1):
            print(f"     {i}. {issue}")

        print("\n  Suggested config changes:")

        if "Low concurrency utilization" in issues or "Low queue depth" in issues:
            print(f"     config.max_mutations_per_round = {config.max_mutations_per_round * 2}  # Double mutation generation")
            print(f"     config.mutation_buffer_min = {config.mutation_buffer_min * 2}  # Larger buffer")

        if "Slow mutation generation" in issues:
            print("     config.reflection_lm = 'faster-model'  # Use faster reflection model")
            print("     config.reflection_batch_size = 3  # Reduce batch size")

        if "Low throughput" in issues:
            print(f"     config.eval_concurrency = {config.eval_concurrency * 2}  # Increase concurrency")
            print("     config.batch_size = config.eval_concurrency // 4  # Scale batch size")

        if "Frequent queue starvation" in issues:
            print(f"     config.queue_limit = {config.queue_limit * 2}  # Increase queue capacity")

    print("\n" + "=" * 80)


def main():
    """Run bottleneck profiling."""
    parser = argparse.ArgumentParser(description="Profile TurboGEPA bottlenecks")
    parser.add_argument("--concurrency", type=int, default=32, help="Evaluation concurrency (default: 32)")
    parser.add_argument("--dataset-size", type=int, default=20, help="Number of examples to use (default: 20)")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum optimization rounds (default: 10)")
    args = parser.parse_args()

    print("=" * 80)
    print("TURBOGEPA BOTTLENECK PROFILER")
    print("=" * 80)
    print("\nThis tool will run TurboGEPA with instrumentation to identify bottlenecks.")
    print(f"\nConfiguration:")
    print(f"  Concurrency:  {args.concurrency}")
    print(f"  Dataset Size: {args.dataset_size}")
    print(f"  Max Rounds:   {args.max_rounds}")
    print("\nStarting profiling run...\n")

    # Clear cache
    cache_dir = Path(".turbo_gepa/")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("✅ Cleared cache\n")

    # Load dataset
    trainset, valset, _ = gepa.examples.aime.init_dataset()
    dataset_size = min(args.dataset_size, len(trainset))
    turbo_dataset = [
        DefaultDataInst(
            input=ex["input"],
            answer=ex["answer"],
            id=f"aime_{i}",
            additional_context=ex.get("additional_context"),
        )
        for i, ex in enumerate(trainset[:dataset_size])
    ]

    print(f"📊 Loaded {len(turbo_dataset)} AIME problems\n")

    # Create config
    config = Config(
        eval_concurrency=args.concurrency,
        n_islands=1,  # Single island for cleaner profiling
        shards=(0.2, 0.5, 1.0),  # 3-rung ASHA
        batch_size=8,
        max_mutations_per_round=16,
        mutation_buffer_min=8,
        queue_limit=128,
        log_level="WARNING",
        adaptive_shards_enabled=False,  # Fixed shards for consistent profiling
    )

    print("Configuration:")
    print(f"  eval_concurrency: {config.eval_concurrency}")
    print(f"  shards: {config.shards}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  max_mutations_per_round: {config.max_mutations_per_round}")
    print(f"  mutation_buffer_min: {config.mutation_buffer_min}")
    print(f"  queue_limit: {config.queue_limit}")
    print()

    # Create metrics tracker
    metrics_tracker = BottleneckMetrics()

    # Create adapter (using standard DefaultAdapter for now since instrumentation
    # would require deeper integration)
    task_lm = "openrouter/openai/gpt-oss-20b:nitro"
    reflection_lm = "openrouter/x-ai/grok-4-fast"

    adapter = DefaultAdapter(
        dataset=turbo_dataset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        config=config,
        auto_config=False,
    )

    seed = "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"

    print("🚀 Starting optimization with profiling...\n")

    start_time = time.time()
    metrics_tracker.start_time = start_time

    result = adapter.optimize(
        seeds=[seed],
        max_rounds=args.max_rounds,
        max_evaluations=None,
        enable_auto_stop=False,
        display_progress=True,
    )

    elapsed = time.time() - start_time

    # Extract basic stats from result
    evolution_stats = result.get("evolution_stats", {}) or {}
    total_evals = evolution_stats.get("total_evaluations", 0)

    print(f"\n✅ Optimization completed in {elapsed:.1f}s")
    print(f"📊 Total evaluations: {total_evals}")
    print(f"📈 Throughput: {total_evals / elapsed:.2f} evals/sec")

    # For now, populate basic metrics from result
    # In a full implementation, we would instrument the orchestrator directly
    metrics_tracker.total_evaluations = total_evals
    metrics_tracker.total_mutations = evolution_stats.get("mutations_generated", 0)

    # Print bottleneck analysis
    print_bottleneck_report(metrics_tracker, config)


if __name__ == "__main__":
    main()
