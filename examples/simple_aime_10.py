#!/usr/bin/env python3
"""
Simple AIME benchmark - 10 problems with OSS model.
The simplest possible script.
"""

import os
import sys
import shutil
from pathlib import Path

# Clear cache for fresh results
cache_dir = Path(__file__).parent.parent / ".turbo_gepa" / "cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("✓ Cleared cache\n")

import warnings

os.environ["LITELLM_LOG"] = "ERROR"
warnings.filterwarnings("ignore")

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

# Check API key
if not os.environ.get("OPENROUTER_API_KEY"):
    print("❌ Error: OPENROUTER_API_KEY not set")
    sys.exit(1)

print("📊 Loading 10 AIME problems...")
trainset, _, _ = gepa.examples.aime.init_dataset()
dataset = [
    DefaultDataInst(
        input=ex["input"],
        answer=ex["answer"],
        id=f"aime_{i}",
        additional_context=ex.get("additional_context"),
    )
    for i, ex in enumerate(trainset[:50])
]
print(f"✓ Loaded {len(dataset)} problems\n")

# Simple config knobs
MAX_ROUNDS = 4
BATCH_SIZE = 10
MAX_MUTATIONS_PER_ROUND = 5
MAX_EXAMPLES_INFLIGHT = 64

config = Config(
    shards=(
        0.3,
        1.0,
    ),
    batch_size=BATCH_SIZE,
    max_mutations_per_round=MAX_MUTATIONS_PER_ROUND,
    eval_concurrency=MAX_EXAMPLES_INFLIGHT,
    max_total_inflight=MAX_EXAMPLES_INFLIGHT,
    n_islands=2,
)

print(
    f"🔧 Config: {len(dataset)} problems | batch_size={BATCH_SIZE} | "
    f"mutations/round={MAX_MUTATIONS_PER_ROUND} | max_examples_inflight={MAX_EXAMPLES_INFLIGHT}\n"
)

# Create adapter
adapter = DefaultAdapter(
    dataset=dataset,
    task_lm="openrouter/openai/gpt-oss-20b:nitro",
    reflection_lm="openrouter/x-ai/grok-4-fast",
    auto_config=False,
)
adapter.config = config

print("🚀 Starting optimization...\n")

# Try to enable dashboard with plotext
try:
    from turbo_gepa.dashboard import TerminalDashboard

    dashboard = TerminalDashboard()
    metrics_callback = dashboard.update
    print("📈 Dashboard enabled\n")
except ImportError:
    metrics_callback = None
    print("📊 Dashboard not available (plotext not installed)\n")

import time

start = time.time()
SEED_PROMPT = (
    "You are a helpful assistant. You are given a question and you need to answer it. "
    "The answer should be given at the end of your response in exactly the format '### <final answer>'"
)
try:
    result = adapter.optimize(
        seeds=[SEED_PROMPT],
        max_rounds=MAX_ROUNDS,
        enable_auto_stop=False,
        display_progress=True,
        optimize_temperature_after_convergence=False,
        metrics_callback=metrics_callback,
    )
except KeyboardInterrupt:
    print("\n\n⚠️  Optimization interrupted by user.")
    sys.exit(1)

elapsed = time.time() - start
evolution_stats = result.get("evolution_stats", {}) or {}
mutations_requested = evolution_stats.get("mutations_requested", 0)
mutations_generated = evolution_stats.get("mutations_generated", 0)
mutations_enqueued = evolution_stats.get("mutations_enqueued", 0)
mutations_promoted = evolution_stats.get("mutations_promoted", 0)
unique_parents = evolution_stats.get("unique_parents", 0)
unique_children = evolution_stats.get("unique_children", 0)
evolution_edges = evolution_stats.get("evolution_edges", 0)
promotion_rate = (
    mutations_promoted / mutations_generated if mutations_generated else 0.0
)
avg_branching = evolution_edges / unique_parents if unique_parents else 0.0

# Analyse pareto results
pareto_entries = result.get("pareto_entries", []) or []
if pareto_entries:
    promote_objective = adapter.config.promote_objective
    best_entry = max(
        pareto_entries,
        key=lambda e: e.result.objectives.get(promote_objective, 0.0),
    )
    best = best_entry.candidate
    best_quality = best_entry.result.objectives.get(
        promote_objective, best.meta.get("quality", 0.0)
    )
else:
    best_entry = None
    best = None
    best_quality = 0.0

print(f"\n{'='*70}")
print(f"  RESULTS")
print(f"{'='*70}")
print(f"\n⏱️  Time: {elapsed:.1f} seconds")
print(f"📊 Best quality: {best_quality:.0%}")
print(f"🎯 Target was: 30%\n")
print(f"🧪 Mutations requested: {mutations_requested}")
print(f"🧬 Mutations generated: {mutations_generated}")
print(f"📦 Mutations enqueued: {mutations_enqueued}")
print(f"🏆 Mutations promoted to Pareto: {mutations_promoted}")
if mutations_generated:
    print(f"📈 Promotion rate: {promotion_rate:.0%}")
print(f"🌱 Unique parents mutated: {unique_parents}")
print(f"🌳 Unique children discovered: {unique_children}")
print(f"🔁 Evolution edges explored: {evolution_edges}")
if unique_parents:
    print(f"🔀 Avg children per parent: {avg_branching:.2f}")
island_stats = evolution_stats.get("islands") or []
if island_stats:
    print("\n🏝️  Island Breakdown:")
    for idx, snapshot in enumerate(island_stats, start=1):
        island_requested = snapshot.get("mutations_requested", 0)
        island_generated = snapshot.get("mutations_generated", 0)
        island_promoted = snapshot.get("mutations_promoted", 0)
        island_rate = (
            island_promoted / island_generated if island_generated else 0.0
        )
        island_parents = snapshot.get("unique_parents", 0)
        island_children = snapshot.get("unique_children", 0)
        print(
            f"   • Island {idx}: requested={island_requested}, generated={island_generated}, "
            f"promoted={island_promoted} ({island_rate:.0%}), parents={island_parents}, "
            f"children={island_children}"
        )
if best is not None:
    print("\n" + "=" * 70)
    print("BEST PROMPT (FULL TEXT):")
    print("=" * 70)
    print(best.text)
    print("=" * 70)
    print(f"\nPrompt length: {len(best.text)} characters")

if best_quality > 0:
    print(f"✅ SUCCESS! Achieved {best_quality:.0%} quality")
else:
    print(f"⚠️  No improvement from baseline")

print()
