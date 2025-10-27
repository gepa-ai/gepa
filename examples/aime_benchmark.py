import gepa
import time

import time
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

# ============================================================================
# GEPA (Original) Benchmark
# ============================================================================

print("=" * 80)
print("GEPA (ORIGINAL) OPTIMIZATION")
print("=" * 80 + "\n")

# Load AIME dataset
trainset, valset, _ = gepa.examples.aime.init_dataset()

print(f"📊 Loaded {len(trainset)} training problems")
print(f"📊 Loaded {len(valset)} validation problems\n")

seed_prompt = {
    "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
}

print("🚀 Starting GEPA optimization...\n")

# Time the GEPA optimization
gepa_start = time.time()
gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm="openrouter/openai/gpt-oss-120b:nitro",  # Student model (fast, cheap)
    reflection_lm="openrouter/x-ai/grok-4-fast",
    max_metric_calls=150,  # <-- Set a budget
    display_progress_bar=True,
)
gepa_elapsed = time.time() - gepa_start

# Extract quality from GEPA result
gepa_quality = gepa_result.best_score if hasattr(gepa_result, "best_score") else 0.0

print(f"\n✅ GEPA completed in {gepa_elapsed:.1f}s")
print(f"📊 Best quality: {gepa_quality:.1%}")
print("GEPA Optimized Prompt:", gepa_result.best_candidate["system_prompt"])


print("\n" + "=" * 80)
print("TURBOGEPA OPTIMIZATION")
print("=" * 80 + "\n")

# Convert GEPA dataset to TurboGEPA format (use same data as GEPA)
turbo_dataset = [
    DefaultDataInst(
        input=ex["input"],
        answer=ex["answer"],
        id=f"aime_{i}",
        additional_context=ex.get("additional_context"),
    )
    for i, ex in enumerate(trainset)  # Use full trainset to match GEPA
]

print(f"📊 Loaded {len(turbo_dataset)} AIME problems (matching GEPA trainset)")

# Create config matching GEPA's evaluation budget
# GEPA: 150 metric calls total
# TurboGEPA: We'll use similar evaluation count with ASHA
config = Config(
    shards=(0.2, 0.5, 1.0),  # 3-tier ASHA
    eval_concurrency=64,  # High parallelism (batch_size auto-scales to 16)
    n_islands=2,  # Multi-island
)

# Create adapter
adapter = DefaultAdapter(
    dataset=turbo_dataset,
    task_lm="openrouter/openai/gpt-oss-120b:nitro",
    reflection_lm="openrouter/x-ai/grok-4-fast",
    auto_config=False,
)
adapter.config = config

seed_turbo = "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"

print("🚀 Starting TurboGEPA optimization...\n")

start_time = time.time()
turbo_result = adapter.optimize(
    seeds=[seed_turbo],
    max_rounds=4,  # Match number of rounds
    enable_auto_stop=False,
    display_progress=True,
    optimize_temperature_after_convergence=False,
)
turbo_elapsed = time.time() - start_time

# Extract best result
pareto_entries = turbo_result.get("pareto_entries", []) or []
if pareto_entries:
    best_entry = max(
        pareto_entries,
        key=lambda e: e.result.objectives.get("quality", 0.0),
    )
    turbo_quality = best_entry.result.objectives.get("quality", 0.0)
    turbo_prompt = best_entry.candidate.text
else:
    turbo_quality = 0.0
    turbo_prompt = seed_turbo

# Get evolution stats
evolution_stats = turbo_result.get("evolution_stats", {}) or {}
mutations_generated = evolution_stats.get("mutations_generated", 0)
mutations_promoted = evolution_stats.get("mutations_promoted", 0)

print("\n" + "=" * 80)
print("BENCHMARK RESULTS")
print("=" * 80)

print("\n📊 GEPA (Original):")
print(f"   Time: {gepa_elapsed:.1f}s")
print(f"   Quality: {gepa_quality:.1%}")
print(f"   Time per evolution: {gepa_elapsed / 150:.2f}s")

print("\n⚡ TurboGEPA:")
print(f"   Time: {turbo_elapsed:.1f}s")
print(f"   Quality: {turbo_quality:.1%}")
print(f"   Mutations generated: {mutations_generated}")
print(f"   Mutations promoted: {mutations_promoted}")
print(
    f"   Time per mutation: {turbo_elapsed / mutations_generated if mutations_generated else 0:.2f}s"
)

speedup = gepa_elapsed / turbo_elapsed if turbo_elapsed > 0 else 0
print(f"\n🏆 Speedup: {speedup:.1f}x faster")
print(
    f"🎯 Quality retention: {(turbo_quality / gepa_quality * 100) if gepa_quality > 0 else 0:.1f}%"
)

print("\n" + "=" * 80)
print("BEST PROMPTS")
print("=" * 80)

print("\n📝 GEPA Best Prompt:")
print(gepa_result.best_candidate["system_prompt"])

print("\n⚡ TurboGEPA Best Prompt:")
print(turbo_prompt)
print("\n" + "=" * 80)
