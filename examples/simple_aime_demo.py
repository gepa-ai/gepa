#!/usr/bin/env python3
"""
Simple AIME Demo - Quick verification test for TurboGEPA

This is a minimal example to verify that TurboGEPA is working correctly
with the AIME dataset. It uses a small subset of the data and runs a
quick optimization to ensure everything is configured properly.

Usage:
    python examples/simple_aime_demo.py

Expected output:
    - Loads 5 AIME math problems
    - Optimizes a prompt using TurboGEPA
    - Shows the best prompt and quality score
    - Should complete in < 5 minutes
"""

import time
from pathlib import Path

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.litellm_cleanup import cleanup as cleanup_litellm

# ============================================================================
# Configuration
# ============================================================================

# LLM models to use
task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

# Dataset size (keep small for quick testing)
DATASET_SIZE = 5

# How many rounds to run
MAX_ROUNDS = 10

print("=" * 80)
print("SIMPLE AIME DEMO - TurboGEPA Verification")
print("=" * 80)
print(f"\n📋 Configuration:")
print(f"   Task LM: {task_lm}")
print(f"   Reflection LM: {reflection_lm}")
print(f"   Dataset size: {DATASET_SIZE} problems")
print(f"   Max rounds: {MAX_ROUNDS}")
print()

# ============================================================================
# Load Dataset
# ============================================================================

print("📊 Loading AIME dataset...")
trainset, valset, _ = gepa.examples.aime.init_dataset()

# Use very small subset for quick testing
trainset_small = trainset[:DATASET_SIZE]
print(f"   ✅ Loaded {len(trainset_small)} training problems")
print()

# Convert to TurboGEPA format
turbo_dataset = [
    DefaultDataInst(
        input=ex["input"],
        answer=ex["answer"],
        id=f"aime_{i}",
        additional_context=ex.get("additional_context"),
    )
    for i, ex in enumerate(trainset_small)
]

# ============================================================================
# Configure TurboGEPA
# ============================================================================

print("⚙️  Configuring TurboGEPA...")

# Simple single-shard configuration for testing
config = Config(
    shards=(1.0,),  # Single shard - evaluate on full dataset
    eval_concurrency=16,  # Moderate concurrency for quick testing
    max_mutations_per_round=8,  # Generate 8 mutations per round
    mutation_buffer_min=4,  # Keep buffer filled
    queue_limit=128,
    batch_size=5,  # Batch size = dataset size
    n_islands=1,  # Single island - no multi-processing overhead
)

# Quality settings
config.cohort_quantile = 0.5  # Keep top 50%
config.eps_improve = 0.0  # Accept equal quality

print(f"   Concurrency: {config.eval_concurrency}")
print(f"   Mutations per round: {config.max_mutations_per_round}")
print(f"   Batch size: {config.batch_size}")
print(f"   Shards: {config.shards}")
print()

# ============================================================================
# Create Adapter and Run Optimization
# ============================================================================

print("🚀 Creating adapter...")
adapter = DefaultAdapter(
    dataset=turbo_dataset,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    config=config,
    auto_config=False,
)

# Seed prompt
seed_prompt = (
    "You are a helpful assistant. You are given a question and you need to answer it. "
    "The answer should be given at the end of your response in exactly the format '### <final answer>'"
)

print("🎯 Starting optimization...")
print(f"   Seed prompt: {seed_prompt[:80]}...")
print()
print("=" * 80)
print()
print("📺 Live progress dashboard will appear below...")
print("   (This may take 3-5 minutes)")
print()

start_time = time.time()
result = adapter.optimize(
    seeds=[seed_prompt],
    enable_auto_stop=False,
    max_rounds=MAX_ROUNDS,
    max_evaluations=None,
    display_progress=True,
    optimize_temperature_after_convergence=False,
)
elapsed = time.time() - start_time
cleanup_litellm()

# ============================================================================
# Results
# ============================================================================

print("=" * 80)
print("\n✅ Optimization complete!")
print(f"⏱️  Time: {elapsed:.1f}s")
print()

# Extract best result
pareto_entries = result.get("pareto_entries", []) or []
evolution_stats = result.get("evolution_stats", {}) or {}

if pareto_entries:
    # Get best quality from Pareto frontier
    best_entry = max(
        pareto_entries,
        key=lambda e: e.result.objectives.get("quality", 0.0),
    )
    quality = best_entry.result.objectives.get("quality", 0.0)
    best_prompt = best_entry.candidate.text
    shard = best_entry.result.shard_fraction or 1.0

    print("📊 Results:")
    print(f"   Best quality: {quality:.1%} (evaluated on {shard:.1%} of data)")
    print(f"   Pareto frontier size: {len(pareto_entries)}")
    print(f"   Total evaluations: {evolution_stats.get('total_evaluations', 0)}")
    print(f"   Mutations generated: {evolution_stats.get('mutations_generated', 0)}")
    print(f"   Mutations promoted: {evolution_stats.get('mutations_promoted', 0)}")
    print()
    print("📝 Best prompt:")
    print("-" * 80)
    print(best_prompt)
    print("-" * 80)
else:
    print("❌ No results generated")
    print("   This might indicate a configuration or connectivity issue")

print()
print("=" * 80)
print("🎉 Demo complete!")
print()

# ============================================================================
# Verification Tips
# ============================================================================

print("💡 What to check:")
print("   ✓ Quality should be > 0% (ideally 20-60% for this small dataset)")
print("   ✓ Pareto frontier should have 2+ candidates")
print("   ✓ Mutations should be generated (5+ is good)")
print("   ✓ Best prompt should be different from seed")
print()
print("🔍 If quality is 0% or no mutations generated:")
print("   - Check OPENROUTER_API_KEY environment variable is set")
print("   - Verify the models are available on OpenRouter")
print("   - Check .turbo_gepa/logs/ for error messages")
print()
print("=" * 80)
