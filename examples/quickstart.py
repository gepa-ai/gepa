"""
TurboGEPA Quickstart Example

This example demonstrates basic usage of TurboGEPA to optimize a prompt
for solving math problems using the AIME dataset.

Usage:
    python examples/quickstart.py
"""

import os
from pathlib import Path
import shutil

# Clean cache before running
if Path(".turbo_gepa/").exists():
    shutil.rmtree(".turbo_gepa/")

# Suppress LiteLLM logging noise
os.environ["LITELLM_LOG"] = "ERROR"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

# Load AIME dataset (challenging high school math problems)
print("Loading AIME dataset...")
trainset, _, _ = gepa.examples.aime.init_dataset()

# Use first 5 problems for this demo
dataset = [
    DefaultDataInst(input=ex["input"], answer=ex["answer"], id=f"aime_{i}")
    for i, ex in enumerate(trainset[:5])
]

print(f"Dataset: {len(dataset)} AIME problems")
print(f"Example problem: {dataset[0].input[:100]}...")
print()

# Configure TurboGEPA
config = Config(
    eval_concurrency=4,  # Run 4 evaluations in parallel
    n_islands=1,  # Single island (simpler for demo)
    shards=(0.6, 1.0),  # Two-rung ASHA: 60% → 100% of dataset
    batch_size=4,  # Batch size for mutations
    max_mutations_per_round=4,  # Generate 4 mutations per round
    log_level="INFO",  # Show progress
    max_optimization_time_seconds=120,  # 2 minute timeout
)

# NEVER EVER MODIFY THESE!!!
task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

# Create adapter
print("Creating TurboGEPA adapter...")
adapter = DefaultAdapter(
    dataset=dataset,
    task_lm=task_lm,  # Fast, cheap model for task
    reflection_lm=reflection_lm,  # Smarter model for reflection
    config=config,
    auto_config=False,
)

# Seed prompt
seed_prompt = "You are a helpful math assistant. Solve the problem step by step and put your final answer after ###."

print("Starting optimization...")
print("=" * 80)

# Run optimization
result = adapter.optimize(
    seeds=[seed_prompt],
    max_rounds=3,  # Run for 3 rounds
    display_progress=True,  # Show live progress
)

# Display results
print()
print("=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)

pareto = result.get("pareto_entries", [])
print(f"\n📊 Pareto Frontier: {len(pareto)} candidates")

# Show best candidate by quality
if pareto:
    best = max(pareto, key=lambda e: e.result.objectives.get("quality", 0))
    quality = best.result.objectives.get("quality", 0)
    cost = -best.result.objectives.get("neg_cost", 0)

    print(f"\n🏆 Best Candidate:")
    print(f"   Quality: {quality:.1%}")
    print(f"   Cost: {cost:.0f} tokens")
    print(f"   Prompt: {best.candidate.text[:200]}...")

    # Show improvement over seed
    seed_entry = next(
        (e for e in pareto if e.candidate.meta.get("source") == "seed"), None
    )
    if seed_entry:
        seed_quality = seed_entry.result.objectives.get("quality", 0)
        improvement = quality - seed_quality
        print(f"\n📈 Improvement over seed: {improvement:+.1%}")

# Show evolution stats
evolution_stats = result.get("evolution_stats", {}) or {}
print(f"\n📈 Evolution Stats:")
print(f"   Total evaluations: {evolution_stats.get('total_evaluations', 0)}")
print(f"   Mutations generated: {evolution_stats.get('mutations_generated', 0)}")
print(f"   Cache hit rate: {evolution_stats.get('cache_hit_rate', 0):.1%}")

print("\n✅ Done! Cache saved to .turbo_gepa/")
