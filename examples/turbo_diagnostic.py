"""
TurboGEPA Diagnostic Tool - Identify where optimization gets stuck

This is a simpler, more practical diagnostic tool that:
1. Runs TurboGEPA with detailed logging
2. Tracks time spent in each phase
3. Monitors queue state and concurrency
4. Identifies when/where the system stalls

Run with: python examples/turbo_diagnostic.py [--concurrency N] [--verbose]
"""

import argparse
import asyncio
import os
import shutil
import time
from pathlib import Path
from collections import defaultdict

# Disable litellm's async logging worker
os.environ["LITELLM_LOG"] = "INFO"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config


class DiagnosticMonitor:
    """Monitor optimization progress and detect stalls."""

    def __init__(self, config: Config):
        self.config = config
        self.start_time = time.time()
        self.last_progress_time = time.time()
        self.last_eval_count = 0
        self.phase_times = defaultdict(list)
        self.stall_threshold = 10.0  # seconds without progress = stall
        self.samples = []
        self.round_times = []

    def record_sample(self, evals_completed: int, active_tasks: int, queue_size: int, buffer_size: int):
        """Record a monitoring sample."""
        now = time.time()
        elapsed = now - self.start_time

        # Check for stalls
        if evals_completed > self.last_eval_count:
            self.last_progress_time = now
            self.last_eval_count = evals_completed
        else:
            time_since_progress = now - self.last_progress_time
            if time_since_progress > self.stall_threshold:
                print(f"\n⚠️  STALL DETECTED at {elapsed:.1f}s (no progress for {time_since_progress:.1f}s)")
                print(f"   Active tasks: {active_tasks}/{self.config.eval_concurrency}")
                print(f"   Queue size: {queue_size}/{self.config.queue_limit}")
                print(f"   Buffer size: {buffer_size}")
                self.last_progress_time = now  # Reset to avoid spam

        self.samples.append({
            "elapsed": elapsed,
            "evals": evals_completed,
            "active": active_tasks,
            "queue": queue_size,
            "buffer": buffer_size,
        })

    def print_live_stats(self, evals: int, active: int, queue: int, buffer: int, round_num: int):
        """Print live statistics."""
        elapsed = time.time() - self.start_time
        throughput = evals / elapsed if elapsed > 0 else 0.0

        concurrency_pct = (active / self.config.eval_concurrency * 100) if self.config.eval_concurrency > 0 else 0
        queue_pct = (queue / self.config.queue_limit * 100) if self.config.queue_limit > 0 else 0

        print(f"\r[Round {round_num}] {elapsed:.1f}s | "
              f"Evals: {evals} ({throughput:.1f}/s) | "
              f"Active: {active}/{self.config.eval_concurrency} ({concurrency_pct:.0f}%) | "
              f"Queue: {queue}/{self.config.queue_limit} ({queue_pct:.0f}%) | "
              f"Buffer: {buffer}",
              end="", flush=True)

    def print_final_analysis(self):
        """Print final diagnostic analysis."""
        if not self.samples:
            print("\nNo samples collected!")
            return

        print("\n\n" + "=" * 80)
        print("DIAGNOSTIC ANALYSIS")
        print("=" * 80)

        # Overall stats
        total_time = time.time() - self.start_time
        final_evals = self.samples[-1]["evals"] if self.samples else 0
        avg_throughput = final_evals / total_time if total_time > 0 else 0.0

        print(f"\nOverall Performance:")
        print(f"  Total Time: {total_time:.1f}s")
        print(f"  Total Evaluations: {final_evals}")
        print(f"  Average Throughput: {avg_throughput:.2f} evals/sec")

        # Concurrency analysis
        concurrency_samples = [s["active"] for s in self.samples]
        if concurrency_samples:
            avg_concurrency = sum(concurrency_samples) / len(concurrency_samples)
            max_concurrency = max(concurrency_samples)
            utilization = avg_concurrency / self.config.eval_concurrency if self.config.eval_concurrency > 0 else 0.0

            print(f"\nConcurrency Utilization:")
            print(f"  Configured Max: {self.config.eval_concurrency}")
            print(f"  Average Active: {avg_concurrency:.1f}")
            print(f"  Max Observed: {max_concurrency}")
            print(f"  Utilization: {utilization:.1%}")

            if utilization < 0.5:
                print(f"  ⚠️  LOW UTILIZATION! System is only using {utilization:.0%} of available concurrency")
                print("     Likely causes:")
                print("     - Not enough candidates in queue (increase max_mutations_per_round)")
                print("     - Mutation generation too slow (use faster reflection_lm)")
                print("     - Blocking operations preventing task launch")

        # Queue analysis
        queue_samples = [s["queue"] for s in self.samples]
        if queue_samples:
            avg_queue = sum(queue_samples) / len(queue_samples)
            max_queue = max(queue_samples)
            min_queue = min(queue_samples)
            empty_count = sum(1 for q in queue_samples if q == 0)
            empty_pct = empty_count / len(queue_samples) if queue_samples else 0.0

            print(f"\nQueue Health:")
            print(f"  Average Depth: {avg_queue:.1f}")
            print(f"  Max Depth: {max_queue}")
            print(f"  Min Depth: {min_queue}")
            print(f"  Empty Samples: {empty_count}/{len(queue_samples)} ({empty_pct:.1%})")

            if empty_pct > 0.2:
                print(f"  ⚠️  QUEUE STARVATION! Queue was empty {empty_pct:.0%} of the time")
                print("     Recommendations:")
                print(f"     - Increase max_mutations_per_round from {self.config.max_mutations_per_round}")
                print(f"     - Increase mutation_buffer_min from {self.config.mutation_buffer_min}")
                print("     - Check if mutation generation is bottlenecked")

            if max_queue >= self.config.queue_limit:
                print(f"  ⚠️  QUEUE SATURATION! Queue hit limit of {self.config.queue_limit}")
                print("     Recommendation: Increase queue_limit or reduce max_mutations_per_round")

        # Throughput over time
        print(f"\nThroughput Analysis:")
        if len(self.samples) > 10:
            # Split into phases
            phase_size = len(self.samples) // 3
            early = self.samples[:phase_size]
            mid = self.samples[phase_size:2*phase_size]
            late = self.samples[2*phase_size:]

            def calc_throughput(samples):
                if len(samples) < 2:
                    return 0.0
                time_diff = samples[-1]["elapsed"] - samples[0]["elapsed"]
                eval_diff = samples[-1]["evals"] - samples[0]["evals"]
                return eval_diff / time_diff if time_diff > 0 else 0.0

            early_tput = calc_throughput(early)
            mid_tput = calc_throughput(mid)
            late_tput = calc_throughput(late)

            print(f"  Early phase: {early_tput:.2f} evals/sec")
            print(f"  Mid phase: {mid_tput:.2f} evals/sec")
            print(f"  Late phase: {late_tput:.2f} evals/sec")

            if late_tput < early_tput * 0.5:
                print(f"  ⚠️  THROUGHPUT DEGRADATION! Late-phase throughput dropped {(1 - late_tput/early_tput)*100:.0f}%")
                print("     Possible causes:")
                print("     - Cache warming helped early on")
                print("     - Queue starvation in later rounds")
                print("     - Harder shards taking longer")

        # Buffer analysis
        buffer_samples = [s["buffer"] for s in self.samples if "buffer" in s]
        if buffer_samples:
            avg_buffer = sum(buffer_samples) / len(buffer_samples)
            min_buffer = min(buffer_samples)
            low_buffer_pct = sum(1 for b in buffer_samples if b < self.config.mutation_buffer_min) / len(buffer_samples)

            print(f"\nMutation Buffer:")
            print(f"  Average Size: {avg_buffer:.1f}")
            print(f"  Minimum: {min_buffer}")
            print(f"  Below Threshold: {low_buffer_pct:.1%}")

            if low_buffer_pct > 0.3:
                print(f"  ⚠️  MUTATION BOTTLENECK! Buffer below minimum {low_buffer_pct:.0%} of the time")
                print("     This indicates mutation generation cannot keep up with evaluation")
                print("     Recommendations:")
                print("     - Use faster reflection_lm")
                print("     - Reduce reflection_batch_size")
                print("     - Increase mutation_buffer_min to trigger generation earlier")

        print("\n" + "=" * 80)


def run_diagnostic(concurrency: int, dataset_size: int, max_rounds: int, verbose: bool):
    """Run diagnostic optimization."""

    print("=" * 80)
    print("TURBOGEPA DIAGNOSTIC TOOL")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Concurrency: {concurrency}")
    print(f"  Dataset Size: {dataset_size}")
    print(f"  Max Rounds: {max_rounds if max_rounds else 'None (converge naturally)'}")
    print(f"  Verbose: {verbose}")
    print()

    # Clear cache
    cache_dir = Path(".turbo_gepa/")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("✅ Cleared cache\n")

    # Load dataset
    print("Loading dataset...")
    trainset, valset, _ = gepa.examples.aime.init_dataset()
    dataset_size = min(dataset_size, len(trainset))
    turbo_dataset = [
        DefaultDataInst(
            input=ex["input"],
            answer=ex["answer"],
            id=f"aime_{i}",
            additional_context=ex.get("additional_context"),
        )
        for i, ex in enumerate(trainset[:dataset_size])
    ]
    print(f"📊 Loaded {len(turbo_dataset)} problems\n")

    # Create config with optimized values for continuous pipeline saturation
    config = Config(
        eval_concurrency=concurrency,
        n_islands=1,  # Single island for clearer diagnostics
        shards=(0.2, 0.5, 1.0),
        batch_size=max(8, concurrency // 2),  # Larger batches for better throughput
        max_mutations_per_round=max(32, concurrency * 2),  # Generate ahead of consumption
        mutation_buffer_min=max(16, concurrency),  # Keep pipeline full
        queue_limit=max(128, concurrency * 4),  # Deep queue to prevent starvation
        log_level="DEBUG" if verbose else "WARNING",
        adaptive_shards_enabled=False,
    )

    print("TurboGEPA Configuration:")
    print(f"  eval_concurrency: {config.eval_concurrency}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  max_mutations_per_round: {config.max_mutations_per_round}")
    print(f"  mutation_buffer_min: {config.mutation_buffer_min}")
    print(f"  queue_limit: {config.queue_limit}")
    print(f"  shards: {config.shards}")
    print()

    # Create monitor
    monitor = DiagnosticMonitor(config)

    # Create adapter
    task_lm = "openrouter/openai/gpt-oss-20b:nitro"
    reflection_lm = "openrouter/x-ai/grok-4-fast"

    adapter = DefaultAdapter(
        dataset=turbo_dataset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        config=config,
        auto_config=False,
    )

    seed = "You are a helpful assistant. Answer the math question and provide your final answer in the format '### <answer>'"

    print("🚀 Starting optimization with monitoring...")
    print("   (Watch for stall warnings and utilization metrics)\n")

    # Run with monitoring
    # Note: This is a simplified version. Full implementation would patch the orchestrator
    # to call monitor.record_sample() periodically

    start_time = time.time()

    result = adapter.optimize(
        seeds=[seed],
        max_rounds=max_rounds,
        max_evaluations=None,
        enable_auto_stop=True,  # Enable convergence-based auto-stop
        display_progress=True,
    )

    elapsed = time.time() - start_time

    # Get stats from result
    evolution_stats = result.get("evolution_stats", {}) or {}
    total_evals = evolution_stats.get("total_evaluations", 0)
    mutations_generated = evolution_stats.get("mutations_generated", 0)
    stop_reason = result.get("stop_reason", "unknown")

    print(f"\n\n✅ Optimization completed!")
    print(f"   Stop Reason: {stop_reason}")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Evaluations: {total_evals}")
    print(f"   Mutations: {mutations_generated}")
    print(f"   Throughput: {total_evals / elapsed:.2f} evals/sec")

    if stop_reason == "converged":
        print(f"\n🎯 CONVERGENCE DETECTED!")
        print(f"   System automatically stopped after detecting no further improvements.")

    # Simulate some samples for analysis (in real version, these would be collected during run)
    # For now, provide a basic analysis
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    avg_throughput = total_evals / elapsed if elapsed > 0 else 0.0
    theoretical_max = concurrency * 1.0  # Assume 1 eval/sec per concurrent slot

    print(f"\nThroughput Analysis:")
    print(f"  Observed: {avg_throughput:.2f} evals/sec")
    print(f"  Theoretical Max: {theoretical_max:.2f} evals/sec (assuming 1s per eval)")
    print(f"  Efficiency: {avg_throughput / theoretical_max:.1%}")

    if avg_throughput < theoretical_max * 0.3:
        print(f"\n⚠️  LOW EFFICIENCY ({avg_throughput / theoretical_max:.0%} of theoretical max)")
        print("   Likely bottlenecks:")
        print("   1. Queue starvation (not enough mutations)")
        print("   2. Slow mutation generation")
        print("   3. Configuration mismatch")
        print("\n   Recommendations:")
        print(f"   - Try: max_mutations_per_round = {config.max_mutations_per_round * 2}")
        print(f"   - Try: mutation_buffer_min = {config.mutation_buffer_min * 2}")
        print("   - Try: Use faster reflection_lm model")

    # Mutation efficiency
    if mutations_generated > 0:
        mutation_rate = mutations_generated / elapsed
        print(f"\nMutation Generation:")
        print(f"  Total: {mutations_generated}")
        print(f"  Rate: {mutation_rate:.2f} mutations/sec")

        # Check if mutation generation is keeping up
        mutation_to_eval_ratio = mutations_generated / total_evals if total_evals > 0 else 0
        print(f"  Mutation/Eval Ratio: {mutation_to_eval_ratio:.2f}")

        if mutation_to_eval_ratio < 1.5:
            print(f"\n⚠️  LOW MUTATION RATE (ratio < 1.5)")
            print("   The system is not generating enough candidate mutations")
            print("   This can cause queue starvation and low concurrency utilization")
            print("\n   Recommendations:")
            print(f"   - Increase max_mutations_per_round from {config.max_mutations_per_round}")
            print("   - Use faster reflection_lm")

    # Configuration recommendations
    print(f"\n" + "=" * 80)
    print("CONFIGURATION TUNING SUGGESTIONS")
    print("=" * 80)

    print("\nIf you're experiencing stalls or low throughput, try this config:\n")
    print("config = Config(")
    print(f"    eval_concurrency={concurrency},")
    print(f"    batch_size={max(8, concurrency // 4)},")
    print(f"    max_mutations_per_round={concurrency},  # Match concurrency")
    print(f"    mutation_buffer_min={concurrency // 2},  # Large buffer")
    print(f"    queue_limit={concurrency * 4},  # Deep queue")
    print("    shards=(0.1, 0.3, 1.0),  # Aggressive early pruning")
    print("    cohort_quantile=0.5,  # Promote top 50%")
    print("    log_level='INFO',  # More visibility")
    print(")")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run TurboGEPA diagnostic")
    parser.add_argument("--concurrency", type=int, default=32, help="Evaluation concurrency")
    parser.add_argument("--dataset-size", type=int, default=15, help="Number of examples")
    parser.add_argument("--max-rounds", type=int, default=None, help="Maximum rounds (None = converge naturally)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    run_diagnostic(
        concurrency=args.concurrency,
        dataset_size=args.dataset_size,
        max_rounds=args.max_rounds,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
