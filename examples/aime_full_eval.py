#!/usr/bin/env python3
"""
Full AIME Evaluation with TurboGEPA + Auto-Convergence Detection

This script runs a complete optimization on the AIME dataset using:
- OpenRouter's Grok 4 Fast model
- Automatic convergence detection (stops when improvement plateaus)
- Staged temperature optimization (optional)
- Full progress tracking and visualization

Usage:
    export OPENROUTER_API_KEY=your_key_here
    python examples/aime_full_eval.py

Configuration:
    Edit the variables in the CONFIG section below to customize the run.
"""

import json
import os
import sys
import time
from pathlib import Path
from types import MethodType

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.islands import spawn_islands
from turbo_gepa.multi_island_dashboard import IslandProgressAggregator


# ============================================================================
# CONFIG - Edit these variables to customize your evaluation
# ============================================================================

# Dataset
DATASET_LIMIT = None  # Limit dataset size for testing (None = full dataset)
# Try 10-20 for quick tests, None for full evaluation

# Optimization
ENABLE_AUTO_STOP = True  # Auto-stop when convergence detected
MAX_ROUNDS = None  # Maximum rounds (None = unlimited with auto-stop)
# Set to a number like 10-20 if you want a hard limit

# Temperature optimization
WITH_TEMPERATURE = False  # Enable staged temperature optimization
# True = two-phase: prompts first, then temperature

# Islands
USE_MULTI_ISLANDS = True  # Enable multi-island optimization with async tasks
NUM_ISLANDS = 4  # Number of parallel islands to launch

# Performance
EVAL_CONCURRENCY = (
    32  # Concurrent evaluations per island (32=balanced, 16=safer, 64=aggressive)
)
# With 4 islands: 32×4 = 128 total concurrent operations (stays under OS file limit)
MAX_MUTATIONS_PER_ROUND = (
    8  # Mutations per round per island (8=more exploration, needed for AIME)
)
REFLECTION_BATCH_SIZE = (
    5  # Examples shown to reflection LLM (5=good context for hard problems)
)

# Cache
CLEAR_CACHE = True  # Clear cache before starting (True = fresh run, False = resume)
# Set to True for clean runs, False to resume from previous state

# Model
TASK_MODEL = "openrouter/openai/gpt-oss-120b:nitro"
REFLECTION_MODEL = "openrouter/x-ai/grok-4-fast"
# Set REFLECTION_MODEL = None to use fast heuristic reflection (no LLM calls)

# Seed prompt used by all islands
SEED_PROMPT = (
    "You are an expert mathematics problem solver specializing in AIME-level competition problems. "
    "For each problem:\n"
    "1. Read the problem carefully and identify what is being asked\n"
    "2. Break down the problem into clear steps\n"
    "3. Show your mathematical reasoning and calculations\n"
    "4. Verify your answer makes sense in context\n"
    "5. Provide your final answer in the exact format: ### <answer>\n\n"
    "Be precise, systematic, and double-check your work."
)

# ============================================================================
# END CONFIG
# ============================================================================


def build_island_config(dataset_size: int, *, island_tag: str | None = None) -> Config:
    """Create a Config object with optional island-specific cache/log paths."""
    config = Config(
        eval_concurrency=EVAL_CONCURRENCY,
        max_mutations_per_round=MAX_MUTATIONS_PER_ROUND,
        reflection_batch_size=REFLECTION_BATCH_SIZE,
        queue_limit=64,
        migration_period=10,
        log_summary_interval=1,
        cache_path=".turbo_gepa/aime_cache",
        log_path=".turbo_gepa/aime_logs",
    )

    if island_tag:
        cache_path = Path(config.cache_path) / island_tag
        log_path = Path(config.log_path) / island_tag
        cache_path.mkdir(parents=True, exist_ok=True)
        log_path.mkdir(parents=True, exist_ok=True)
        config.cache_path = cache_path.as_posix()
        config.log_path = log_path.as_posix()

    return config


def serialize_archive_entry(entry) -> dict:
    """Convert an ArchiveEntry to a JSON-serializable dict."""
    objectives = entry.result.objectives
    quality_val = objectives.get("quality", 0.0)
    tokens_val = objectives.get("tokens", 0.0)
    neg_cost_val = objectives.get("neg_cost")
    return {
        "text": entry.candidate.text,
        "quality": float(quality_val) if quality_val is not None else 0.0,
        "tokens": float(tokens_val) if tokens_val is not None else 0.0,
        "neg_cost": float(neg_cost_val) if neg_cost_val is not None else None,
        "meta": entry.candidate.meta,
    }


async def multi_island_worker(context) -> None:
    """Worker entrypoint for each island async task."""
    island_pid = os.getpid()
    island_id = context.island_id
    island_tag = f"island_{island_id}"
    results_dir = Path(".turbo_gepa/aime_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"{island_tag}.json"

    try:
        dataset, *_ = load_aime_dataset(limit=DATASET_LIMIT)
        config = build_island_config(len(dataset), island_tag=island_tag)

        adapter = DefaultAdapter(
            dataset=dataset,
            config=config,
            sampler_seed=42,
            cache_dir=config.cache_path,
            log_dir=config.log_path,
            task_lm=TASK_MODEL,
            reflection_lm=REFLECTION_MODEL,
            task_lm_temperature=None,
            reflection_lm_temperature=None,
        )

        original_builder = adapter._build_orchestrator

        def build_with_context(
            self, logger, enable_auto_stop=False, display_progress=False
        ):
            orchestrator = original_builder(logger, enable_auto_stop, display_progress)
            orchestrator.island_context = context
            return orchestrator

        adapter._build_orchestrator = MethodType(build_with_context, adapter)

        # Disable individual island charts - we show aggregated dashboard instead
        result = await adapter.optimize_async(
            seeds=[SEED_PROMPT],
            max_rounds=MAX_ROUNDS,
            max_evaluations=None,
            enable_auto_stop=ENABLE_AUTO_STOP,
            optimize_temperature_after_convergence=WITH_TEMPERATURE,
            display_progress=False,
        )

        pareto_entries = result.get("pareto_entries") or []
        serialized_pareto = [serialize_archive_entry(entry) for entry in pareto_entries]
        serialized_pareto.sort(key=lambda item: item.get("quality", 0.0), reverse=True)
        best_candidate = serialized_pareto[0] if serialized_pareto else None

        island_summary = {
            "pid": island_pid,
            "island_id": island_id,
            "tag": island_tag,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "best_candidate": best_candidate,
            "pareto": serialized_pareto[:10],
        }

        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(island_summary, handle, indent=2)

        print(f"[Island {island_id}] Completed optimization.")

    except Exception as exc:  # pragma: no cover - best effort logging in worker
        island_summary = {
            "pid": island_pid,
            "island_id": island_id,
            "tag": island_tag,
            "error": str(exc),
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(island_summary, handle, indent=2)
        print(f"[Island {island_id}] Failed with error: {exc}")


def load_aime_dataset(limit=None):
    """Load AIME dataset and convert to TurboGEPA format."""
    print("📥 Loading AIME dataset...")
    trainset, valset, testset = gepa.examples.aime.init_dataset()

    if limit:
        print(f"   Limiting to {limit} training examples (for testing)")
        trainset = trainset[:limit]
        valset = valset[: min(limit // 2, len(valset))]

    # Convert to TurboGEPA format
    turbo_trainset = [
        DefaultDataInst(
            input=x["input"],
            answer=x["answer"],
            additional_context=x.get("additional_context"),
            id=f"train-{i}",
            difficulty=0.5,  # AIME problems are uniformly hard
        )
        for i, x in enumerate(trainset)
    ]

    print(
        f"✓ Loaded {len(trainset)} training examples, {len(valset)} validation examples"
    )
    return turbo_trainset, trainset, valset, testset


def run_single_island_optimization(dataset):
    """Run TurboGEPA optimization on a single island (original behaviour)."""

    print("\n" + "=" * 80)
    print("⚡ TurboGEPA - AIME Full Evaluation (Single Island)")
    print("=" * 80)

    config = build_island_config(len(dataset))

    print(f"\n📊 Configuration:")
    print(f"   Dataset size: {len(dataset)} examples")
    print(f"   Task model: {TASK_MODEL}")
    print(f"   Reflection model: {REFLECTION_MODEL}")
    print(f"   Eval concurrency: {EVAL_CONCURRENCY}")
    print(f"   Auto-stop enabled: {ENABLE_AUTO_STOP}")
    print(f"   Max rounds: {MAX_ROUNDS if MAX_ROUNDS else 'unlimited (auto-stop)'}")
    print(
        f"   Temperature optimization: {'Yes (staged)' if WITH_TEMPERATURE else 'No'}"
    )
    print(f"   Cache: {config.cache_path}")

    adapter = DefaultAdapter(
        dataset=dataset,
        config=config,
        sampler_seed=42,
        cache_dir=config.cache_path,
        log_dir=config.log_path,
        task_lm=TASK_MODEL,
        reflection_lm=REFLECTION_MODEL,
        task_lm_temperature=None,
        reflection_lm_temperature=None,
    )

    print(f"\n🌱 Seed prompt ({len(SEED_PROMPT)} chars):")
    print(f'   "{SEED_PROMPT[:100]}..."')

    print(f"\n⏱️  Starting optimization...")
    print(f"\n💡 Auto-stop will detect convergence and stop automatically")
    print(f"   Convergence detected when improvement plateaus for multiple rounds\n")

    start_time = time.time()

    try:
        result = adapter.optimize(
            seeds=[SEED_PROMPT],
            max_rounds=MAX_ROUNDS,
            max_evaluations=None,
            enable_auto_stop=ENABLE_AUTO_STOP,
            optimize_temperature_after_convergence=WITH_TEMPERATURE,
        )

        elapsed = time.time() - start_time

        pareto_entries = result.get("pareto_entries") or []
        serialized_pareto = [serialize_archive_entry(entry) for entry in pareto_entries]
        serialized_pareto.sort(key=lambda item: item.get("quality", 0.0), reverse=True)
        phase1_pareto = result.get("phase1_pareto", [])

        print("\n" + "=" * 80)
        print("✅ OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"\n⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"📈 Pareto frontier: {len(serialized_pareto)} candidates")

        if WITH_TEMPERATURE and phase1_pareto:
            print(f"📈 Phase 1 (prompts only): {len(phase1_pareto)} candidates")

        best_candidate = serialized_pareto[0] if serialized_pareto else None

        if best_candidate:
            print(f"\n🏆 Best Candidate:")
            print(f"   Quality: {best_candidate['quality']:.2%}")
            print(f"   Tokens: {best_candidate['tokens']:.0f}")
            temp = (
                best_candidate["meta"].get("temperature")
                if best_candidate["meta"]
                else None
            )
            if temp is not None:
                print(f"   Temperature: {temp}")
            print(f"\n📝 Optimized Prompt:")
            print("-" * 80)
            print(best_candidate["text"])
            print("-" * 80)

        output_dir = Path(".turbo_gepa/aime_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"aime_eval_{timestamp}.json"

        results_data = {
            "timestamp": timestamp,
            "config": {
                "dataset_size": len(dataset),
                "task_model": TASK_MODEL,
                "reflection_model": REFLECTION_MODEL,
                "eval_concurrency": EVAL_CONCURRENCY,
                "auto_stop": ENABLE_AUTO_STOP,
                "max_rounds": MAX_ROUNDS,
                "with_temperature": WITH_TEMPERATURE,
                "num_islands": 1,
            },
            "results": {
                "elapsed_time": elapsed,
                "pareto_size": len(serialized_pareto),
                "phase1_size": len(phase1_pareto) if WITH_TEMPERATURE else None,
            },
            "best_candidate": best_candidate,
            "pareto_frontier": serialized_pareto[:10],
        }

        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(results_data, handle, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        return results_data

    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        elapsed = time.time() - start_time
        print(f"⏱️  Ran for {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return None
    except Exception as exc:
        print(f"\n❌ Error during optimization: {exc}")
        import traceback

        traceback.print_exc()
        return None


async def run_multi_island_optimization(dataset):
    """Run TurboGEPA optimization across multiple islands using async tasks."""
    import asyncio

    print("\n" + "=" * 80)
    print("⚡ TurboGEPA - AIME Full Evaluation (Multi-Island)")
    print("=" * 80)

    print(f"\n📊 Configuration:")
    print(f"   Dataset size: {len(dataset)} examples")
    print(f"   Task model: {TASK_MODEL}")
    print(f"   Reflection model: {REFLECTION_MODEL}")
    print(f"   Eval concurrency: {EVAL_CONCURRENCY}")
    print(f"   Auto-stop enabled: {ENABLE_AUTO_STOP}")
    print(f"   Max rounds: {MAX_ROUNDS if MAX_ROUNDS else 'unlimited (auto-stop)'}")
    print(
        f"   Temperature optimization: {'Yes (staged)' if WITH_TEMPERATURE else 'No'}"
    )
    print(f"   Islands: {NUM_ISLANDS}")
    print(f"   Cache root: .turbo_gepa/aime_cache")

    print(f"\n🌱 Seed prompt ({len(SEED_PROMPT)} chars) shared across islands.")
    print(f'   Preview: "{SEED_PROMPT[:100]}..."')

    print(f"\n⏱️  Launching {NUM_ISLANDS} islands (async tasks)...")

    results_dir = Path(".turbo_gepa/aime_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in results_dir.glob("island_*.json"):
        try:
            stale_file.unlink()
        except OSError:
            pass

    # Create metrics queue and dashboard aggregator
    metrics_queue = asyncio.Queue(maxsize=1000)
    dashboard = IslandProgressAggregator(n_islands=NUM_ISLANDS)

    # Dashboard monitoring task
    async def monitor_dashboard():
        """Continuously read from metrics queue and update dashboard."""
        try:
            while True:
                # Wait for metrics with timeout to allow periodic updates
                try:
                    metrics = await asyncio.wait_for(metrics_queue.get(), timeout=1.0)
                    dashboard.update(
                        island_id=metrics["island_id"],
                        round_num=metrics["round"],
                        best_quality=metrics["best_quality"],
                        avg_quality=metrics["avg_quality"],
                        pareto_size=metrics["pareto_size"],
                    )
                except asyncio.TimeoutError:
                    # No new metrics, just refresh display with current data
                    dashboard.display()
        except asyncio.CancelledError:
            # Final display before shutdown
            dashboard.display()
            raise

    start_time = time.time()

    # Start dashboard monitor
    dashboard_task = asyncio.create_task(monitor_dashboard())

    # Start island tasks
    island_tasks = await spawn_islands(
        NUM_ISLANDS, multi_island_worker, metrics_queue=metrics_queue
    )

    # Wait for islands to complete
    await asyncio.gather(*island_tasks)

    # Cancel dashboard monitor
    dashboard_task.cancel()
    try:
        await dashboard_task
    except asyncio.CancelledError:
        pass

    elapsed = time.time() - start_time
    print(f"\n✅ All islands completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    island_files = sorted(results_dir.glob("island_*.json"))

    if not island_files:
        print("❌ No island result files found - cannot aggregate results.")
        return None

    island_summaries = []
    for path in island_files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                island_summaries.append(json.load(handle))
        except json.JSONDecodeError:
            print(f"⚠️  Skipping unreadable island result file: {path}")

    if not island_summaries:
        print("❌ Failed to parse island results.")
        return None

    candidates = [
        summary["best_candidate"]
        for summary in island_summaries
        if summary.get("best_candidate")
    ]
    candidates.sort(key=lambda item: item.get("quality", 0.0), reverse=True)
    best_candidate = candidates[0] if candidates else None

    combined_pareto = []
    for summary in island_summaries:
        combined_pareto.extend(summary.get("pareto", []))
    combined_pareto.sort(key=lambda item: item.get("quality", 0.0), reverse=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    aggregated_file = results_dir / f"aime_eval_{timestamp}.json"

    results_data = {
        "timestamp": timestamp,
        "config": {
            "dataset_size": len(dataset),
            "task_model": TASK_MODEL,
            "reflection_model": REFLECTION_MODEL,
            "eval_concurrency": EVAL_CONCURRENCY,
            "auto_stop": ENABLE_AUTO_STOP,
            "max_rounds": MAX_ROUNDS,
            "with_temperature": WITH_TEMPERATURE,
            "num_islands": NUM_ISLANDS,
        },
        "results": {
            "elapsed_time": elapsed,
            "pareto_size": len(combined_pareto),
            "island_count": len(island_summaries),
        },
        "best_candidate": best_candidate,
        "pareto_frontier": combined_pareto[:10],
        "island_summaries": island_summaries,
    }

    with aggregated_file.open("w", encoding="utf-8") as handle:
        json.dump(results_data, handle, indent=2)

    print(f"💾 Aggregated multi-island results saved to: {aggregated_file}")

    if best_candidate:
        print(f"\n🏆 Global Best Candidate Quality: {best_candidate['quality']:.2%}")

    return results_data


async def run_optimization(dataset):
    """Dispatch to single- or multi-island optimization based on configuration."""
    if USE_MULTI_ISLANDS:
        return await run_multi_island_optimization(dataset)
    return run_single_island_optimization(dataset)


async def evaluate_on_valset(prompt_text, valset, temperature=None):
    """Evaluate the optimized prompt on validation set."""
    import asyncio
    from litellm import acompletion

    print("\n" + "=" * 80)
    print("📊 Validation Set Evaluation")
    print("=" * 80)
    print(f"   Evaluating best prompt on {len(valset)} validation examples...")
    if temperature is not None:
        print(f"   Temperature: {temperature}")

    correct = 0
    total = len(valset)

    # Run evaluations concurrently for speed
    async def eval_one(i, example):
        try:
            completion_kwargs = {
                "model": TASK_MODEL,
                "messages": [
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": example["input"]},
                ],
            }
            if temperature is not None:
                completion_kwargs["temperature"] = temperature

            response = await acompletion(**completion_kwargs)
            output = response.choices[0].message.content

            is_correct = example["answer"] in output
            if is_correct:
                print(f"   ✓ Example {i+1}/{total}")
            else:
                print(f"   ✗ Example {i+1}/{total}")
            return is_correct

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            # Truncate error message for readability
            error_display = error_msg[:80] + "..." if len(error_msg) > 80 else error_msg
            print(
                f"   ⚠️  Failed on example {i+1}/{total} ({error_type}: {error_display})"
            )
            # Count as failure
            return False

    # Evaluate all examples concurrently
    results = await asyncio.gather(
        *(eval_one(i, example) for i, example in enumerate(valset))
    )
    correct = sum(results)

    val_accuracy = correct / total if total > 0 else 0.0
    print(f"\n   📈 Validation Accuracy: {val_accuracy:.2%} ({correct}/{total})")

    return val_accuracy


async def main():
    """Main entry point."""

    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ ERROR: OPENROUTER_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("  export OPENROUTER_API_KEY=your_key_here")
        return 1

    print("🚀 TurboGEPA - AIME Full Evaluation")
    print("=" * 80)

    # Clear cache if requested
    if CLEAR_CACHE:
        import shutil

        cache_dir = Path(".turbo_gepa/aime_cache")
        if cache_dir.exists():
            print(f"\n🗑️  Clearing cache: {cache_dir}")
            shutil.rmtree(cache_dir)
            print("   ✓ Cache cleared")

    # Load dataset
    dataset, trainset, valset, testset = load_aime_dataset(limit=DATASET_LIMIT)

    # Run optimization
    results = await run_optimization(dataset=dataset)

    if results and results.get("best_candidate"):
        print("\n✨ Optimization complete!")
        print(f"   Training quality: {results['best_candidate']['quality']:.2%}")
        print(f"   Total time: {results['results']['elapsed_time']/60:.1f} minutes")

        # Evaluate on validation set
        best_prompt = results["best_candidate"]["text"]
        best_temperature = results["best_candidate"].get("temperature")

        val_accuracy = await evaluate_on_valset(best_prompt, valset, best_temperature)

        # Update results with validation accuracy
        results["validation_accuracy"] = val_accuracy

        # Re-save results with validation accuracy
        output_dir = Path(".turbo_gepa/aime_results")
        timestamp = results["timestamp"]
        output_file = output_dir / f"aime_eval_{timestamp}.json"

        with output_file.open("w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Final results (with validation) saved to: {output_file}")
        print(f"\n🎉 Final Results:")
        print(f"   Training Quality: {results['best_candidate']['quality']:.2%}")
        print(f"   Validation Accuracy: {val_accuracy:.2%}")

    return 0


if __name__ == "__main__":
    import asyncio

    exit(asyncio.run(main()))
