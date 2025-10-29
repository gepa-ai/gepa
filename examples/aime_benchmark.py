import argparse
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

"""
Speed Benchmarking Notes:
Gepa: 640.3s for 3 evolutions

================================================================================
BENCHMARK RESULTS
================================================================================

⚡ TurboGEPA:
   Time: 310.7s
   Quality: 83.3% (evaluated on 100.0% of dataset)
   Total evaluations: 0
   Time per evaluation: 0.00s

   Evolution:
   └─ Seeds → 3 parents → 116 children (116 edges)
   └─ Generated 96 mutations, 5 promoted to Pareto
   └─ Final Pareto size: 2, Total candidates: 2

================================================================================
BEST PROMPTS
================================================================================

⚡ TurboGEPA Best Prompt:
You are an expert in solving American Invitational Mathematics Examination (AIME) problems, which are advanced high school competition math problems typically requiring answers as integers from 0 to 999, presented in three digits with leading zeros if necessary. Your goal is to solve the given problem step-by-step using rigorous mathematical reasoning and provide the final numerical answer in the exact format '### <answer>', where <answer> is the three-digit integer (padded with leading zeros, e.g., 073 for 73, 033 for 33, 000 for 0) without any additional text, explanation, or symbols after the triple hashes. Always verify the answer is within 000-999 and format it precisely to avoid errors.

Key strategies and techniques for AIME problems:
- For probability and counting with constraints (e.g., even number of blocks between pairs of identical items like colors in arrangements), use parity of positions (odd/even slots) to separate pairs into two independent permutations: for 12 positions with 6 pairs, even arrangements often yield 6! × 6! favorable cases out of total multinomial 12! / (2!)^6. Alternatively, employ constructive counting by placing pairs sequentially, halving choices for even separations (e.g., 12×6 for first pair, 10×5 for second, etc.), simplifying to fractions like 16/231 where m+n=247. Watch for overcounting and ensure even parity condition holds for all pairs simultaneously.
- In inclusion-exclusion or set problems with mandatory items (e.g., all own candy hearts, plus subsets owning rings/clubs/spades with given exactly k overlaps), define variables for exactly 1,2,3,4 items total (including mandatory): total people equation w + x + y + z = n (e.g., 900), given x=437 (exactly 2), y=234 (exactly 3), so w + z = 229. Total items equation (sum of individual counts, e.g., 195+367+562+900=2024) gives w + 2x + 3y + 4z = total. Solve system for z (e.g., 073). Adjust for mandatory by counting "effective" items excluding it, but include in totals; common pitfall: misclassifying exactly k including/excluding mandatory.
- For systems of logarithmic equations (e.g., log2(x/(yz))=1/2, etc., solve for |log2(x^4 y^3 z^2)| as m/n with m+n), assign a=log2 x, b=log2 y, c=log2 z, form linear system a-b-c=1/2, -a+b-c=1/3, -a-b+c=1/4. Add all for - (a+b+c) = 13/12, subtract pairwise to isolate (e.g., 2a = -7/12 so a=-7/24). Compute 4a+3b+2c (e.g., -25/8), take absolute value 25/8, m+n=033. Alternative: add pairs to get -2 log x = sum, etc. Pitfall: ensure positive reals imply logs can be negative; verify by substitution.
- In number theory problems on repeating decimals (e.g., 0.abcd-bar as fraction in lowest terms, count distinct numerators modulo 1000), express as k/9999 where k=abcd (1≤k≤9998, nonzero digits). Factor 9999=3²×11×101; after reduction gcd(d)=g, numerator x=k/g, denominator y=9999/g. Use inclusion-exclusion over prime factors: coprime case φ(9999)=6000 (multiple of 1000, ≡0 mod 1000). Cases: divisible by 3 but not 11/101 (x multiples of 3 up to 1111/3=370.333, subtract subcases: 370-33-3=334); by 11 not 3/101 (82-27-0=55); by 33 not 101 (3-0=3); by 101 (0). Total 6392 ≡392 mod 1000. Pitfall: handle 3² carefully (cancel 9 for one 3 left in denom), ensure x≥1 and distinct across denominators; no overcounting as numerators unique per reduced fraction.
- For proportion/ratio problems with arrivals (e.g., adults 5/12 initially, become 11/25 after +50 people, min final adults), let initial total x (multiple of 12), adults (5/12)x integer. Final total x+50 multiple of 25, so x ≡ -50 ≡0 mod 25 (since 50=2×25). Solve Chinese Remainder: x ≡0 mod 12 and mod 25, lcm(12,25)=300, minimal x=300. Final adults= (300+50)×11/25=154. General: ensure integers via moduli, minimize by smallest positive solution; pitfall: forget integrality of adults or non-negative bus adults.
- Broader AIME tips: For systems with roots/constraints, try trig subs (x=2sin²θ). Floor/sums: modular residues, sum formulas. Geometry/polygons: symmetry, parallel chords, binomial for counts. Circles/3D: power of point, similar triangles, projections to trapezoids. Verify positives/uniqueness, check mod cycles, avoid extraneous roots from squaring.

Domain-specific facts:
- AIME answers: always 000-999, three digits, no units.
- Repeating decimals period 4: denom 9999=3²×11×101, divisors limited.
- Parity in linear arrangements: even separations force same-parity positions for pairs.
- Mandatory items in Venn: shift "exactly k" to include it, total items include all.
- Log systems: pairwise sums isolate variables efficiently.
- Proportions: simultaneous congruences for minimal integers.

Always compute carefully (squares/roots/mods), ensure fraction simplest (gcd=1), and output only reasoning + exact '### XXX' format.

================================================================================
(.venv) gmiller@Greg-Millers-M1-MackBook-Pro-3 gepa % 


"""


def _ensure_fd_limit(min_soft: int = 4096) -> Tuple[bool, Optional[int], Optional[int]]:
    """Raise soft RLIMIT_NOFILE if possible and register restoration."""

    try:
        import resource
    except ImportError:  # pragma: no cover - non-Unix systems
        return False, None, None

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return False, None, None

    desired = min_soft
    if hard != resource.RLIM_INFINITY:
        desired = min(desired, hard)

    if desired <= soft:
        return False, soft, soft

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    except (OSError, ValueError):
        return False, soft, soft

    import atexit

    def _restore_limit() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
        except (OSError, ValueError):
            pass

    atexit.register(_restore_limit)
    return True, desired, soft


# Parse command line arguments
parser = argparse.ArgumentParser(description="Benchmark GEPA vs TurboGEPA")
parser.add_argument(
    "--run",
    choices=["gepa", "turbo", "both"],
    default="turbo",
    help="Which benchmark to run: gepa, turbo, or both (default: turbo)",
)
args = parser.parse_args()

RUN_GEPA = args.run in ("gepa", "both")
RUN_TURBO = args.run in ("turbo", "both")

limit_changed, new_limit, previous_limit = _ensure_fd_limit()
if limit_changed and new_limit is not None and previous_limit is not None:
    print(f"🔧 Raised open file limit from {previous_limit} to {new_limit}\n")

# ============================================================================
# Load Dataset (shared by both benchmarks)
# ============================================================================

trainset, valset, _ = gepa.examples.aime.init_dataset()

# # Use smaller subset for faster benchmark
BENCHMARK_SIZE = 45  # Use small subset for quick debugging
trainset = trainset[:BENCHMARK_SIZE]
valset = valset[: min(BENCHMARK_SIZE, len(valset))]

print(f"📊 Loaded {len(trainset)} training problems (subset for benchmarking)")
print(f"📊 Loaded {len(valset)} validation problems\n")

# ============================================================================
# GEPA (Original) Benchmark
# ============================================================================

gepa_quality = 0.0
gepa_evaluations = 0
gepa_elapsed = 0.0
gepa_prompt = ""
task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"

if RUN_GEPA:
    print("=" * 80)
    print("GEPA (ORIGINAL) OPTIMIZATION")
    print("=" * 80 + "\n")

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
        task_lm=task_lm,  # Student model (fast, cheap)
        reflection_lm=reflection_lm,
        max_metric_calls=150,  # Reduced for faster benchmark
        display_progress_bar=True,
        raise_on_exception=False,
    )
    gepa_elapsed = time.time() - gepa_start

    # Extract quality and metrics from GEPA result
    if hasattr(gepa_result, "best_candidate") and gepa_result.best_candidate:
        # GEPA evaluates on valset, so we need to check the validation score
        if hasattr(gepa_result, "best_score"):
            gepa_quality = gepa_result.best_score
        elif hasattr(gepa_result, "candidates") and gepa_result.candidates:
            # Try to get the best score from candidates
            scores = [
                c.get("score", 0.0)
                for c in gepa_result.candidates
                if isinstance(c, dict)
            ]
            gepa_quality = max(scores) if scores else 0.0

    gepa_evaluations = 150  # max_metric_calls budget
    gepa_prompt = gepa_result.best_candidate["system_prompt"]

    print(f"\n✅ GEPA completed in {gepa_elapsed:.1f}s")
    print(f"📊 Best quality: {gepa_quality:.1%}")
    print(f"📊 Total evaluations: {gepa_evaluations}")
    print(f"📝 GEPA Optimized Prompt: {gepa_prompt}")


# ============================================================================
# TurboGEPA Benchmark
# ============================================================================

turbo_quality = 0.0
turbo_evaluations = 0
turbo_elapsed = 0.0
turbo_prompt = ""
mutations_generated = 0
mutations_promoted = 0

if RUN_TURBO:
    # Wipe the cache using shutil for safer cross-platform removal
    cache_dir = Path(".turbo_gepa/")
    print(f"🧹 Cache directory check: {cache_dir.resolve()}")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"   ✅ Cleared existing cache")
    else:
        print(f"   ℹ️  No existing cache to clear")
    print("\n" + "=" * 80)
    print("TURBOGEPA OPTIMIZATION")
    print("=" * 80 + "\n")

    # Convert GEPA dataset to TurboGEPA format (use same data as GEPA)
    quick_limit = min(len(trainset), 64)
    turbo_dataset = [
        DefaultDataInst(
            input=ex["input"],
            answer=ex["answer"],
            id=f"aime_{i}",
            additional_context=ex.get("additional_context"),
        )
        for i, ex in enumerate(trainset[:quick_limit])
    ]

    print(f"📊 Loaded {len(turbo_dataset)} AIME problems (quick benchmark subset)")

    # Create config optimized for DEBUGGING (fast iterations, verbose output)
    config = Config(
        shards=(0.1, 0.4, 1.0),
        eval_concurrency=32,
        max_total_inflight=32,
        n_islands=1,
        queue_limit=96,
        mutation_buffer_min=2,
        max_mutations_per_round=16,
        reflection_batch_size=4,
        target_quality=0.80,
        log_level="INFO",
    )

    # Create adapter
    adapter = DefaultAdapter(
        dataset=turbo_dataset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        auto_config=False,
    )
    adapter.config = config

    seed_turbo = "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"

    print("🚀 Starting TurboGEPA optimization...\n")
    print("⏱️  Quick benchmark mode: 60-evaluation budget, ~1-2 minute runtime.\n")

    start_time = time.time()
    turbo_result = adapter.optimize(
        seeds=[seed_turbo],
        enable_auto_stop=True,
        max_rounds=6,
        max_evaluations=60,
        display_progress=True,
        optimize_temperature_after_convergence=False,
    )
    turbo_elapsed = time.time() - start_time

    # Extract best result - prefer highest shard, then best quality within that shard
    pareto_entries = turbo_result.get("pareto_entries", []) or []
    full_shard = config.shards[-1]  # Last shard = 1.0 (100% of data)

    if pareto_entries:
        # Group candidates by shard fraction
        by_shard = {}
        for entry in pareto_entries:
            shard = entry.result.shard_fraction or 0.0
            if shard not in by_shard:
                by_shard[shard] = []
            by_shard[shard].append(entry)

        # Find highest shard with evaluations
        highest_shard = max(by_shard.keys())
        highest_shard_entries = by_shard[highest_shard]

        # Get best quality from highest shard
        best_entry = max(
            highest_shard_entries,
            key=lambda e: e.result.objectives.get("quality", 0.0),
        )
        turbo_quality = best_entry.result.objectives.get("quality", 0.0)
        turbo_prompt = best_entry.candidate.text
        turbo_shard = highest_shard

        # Warn if not evaluated on full dataset
        if turbo_shard < full_shard:
            print(
                f"⚠️  Warning: Best quality {turbo_quality:.1%} is from {turbo_shard:.1%} shard (not full {full_shard:.0%} dataset)"
            )
    else:
        turbo_quality = 0.0
        turbo_prompt = seed_turbo
        turbo_shard = 0.0

    # Get evolution stats
    evolution_stats = turbo_result.get("evolution_stats", {}) or {}
    mutations_generated = evolution_stats.get("mutations_generated", 0)
    mutations_promoted = evolution_stats.get("mutations_promoted", 0)
    mutations_requested = evolution_stats.get("mutations_requested", 0)
    mutations_enqueued = evolution_stats.get("mutations_enqueued", 0)
    unique_parents = evolution_stats.get("unique_parents", 0)
    unique_children = evolution_stats.get("unique_children", 0)
    evolution_edges = evolution_stats.get("evolution_edges", 0)
    turbo_evaluations = evolution_stats.get("total_evaluations", 0)

    # Get archive stats
    pareto_size = len(pareto_entries)
    total_candidates = turbo_result.get("total_candidates", pareto_size)

    print(f"\n✅ TurboGEPA completed in {turbo_elapsed:.1f}s")
    print(f"📊 Best quality: {turbo_quality:.1%}")
    print(f"📊 Total evaluations: {turbo_evaluations}")
    print(f"\n📈 Evolution Statistics:")
    print(f"   Seeds: 1")
    print(f"   Unique parents used: {unique_parents}")
    print(f"   Unique children generated: {unique_children}")
    print(f"   Total evolution edges: {evolution_edges}")
    print(f"   Mutations requested: {mutations_requested}")
    print(f"   Mutations generated: {mutations_generated}")
    print(f"   Mutations enqueued: {mutations_enqueued}")
    print(f"   Mutations promoted to archive: {mutations_promoted}")
    print(f"   Pareto frontier size: {pareto_size}")
    print(f"   Total unique candidates: {total_candidates}")
    print(f"\n📝 TurboGEPA Optimized Prompt: {turbo_prompt}")

# ============================================================================
# Benchmark Results Summary
# ============================================================================

if RUN_GEPA or RUN_TURBO:
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

if RUN_GEPA:
    print("\n📊 GEPA (Original):")
    print(f"   Time: {gepa_elapsed:.1f}s")
    print(f"   Quality: {gepa_quality:.1%}")
    print(f"   Total evaluations: {gepa_evaluations}")
    print(f"   Time per evaluation: {gepa_elapsed / gepa_evaluations:.2f}s")

if RUN_TURBO:
    print("\n⚡ TurboGEPA:")
    print(f"   Time: {turbo_elapsed:.1f}s")
    print(
        f"   Quality: {turbo_quality:.1%} (evaluated on {turbo_shard:.1%} of dataset)"
    )
    print(f"   Total evaluations: {turbo_evaluations}")
    print(
        f"   Time per evaluation: {turbo_elapsed / turbo_evaluations if turbo_evaluations else 0:.2f}s"
    )
    print(f"\n   Evolution:")
    print(
        f"   └─ Seeds → {unique_parents} parents → {unique_children} children ({evolution_edges} edges)"
    )
    print(
        f"   └─ Generated {mutations_generated} mutations, {mutations_promoted} promoted to Pareto"
    )
    print(
        f"   └─ Final Pareto size: {pareto_size}, Total candidates: {total_candidates}"
    )

# Comparison (only if both were run)
if RUN_GEPA and RUN_TURBO:
    speedup = gepa_elapsed / turbo_elapsed if turbo_elapsed > 0 else 0
    efficiency_gain = (
        (gepa_elapsed / gepa_evaluations) / (turbo_elapsed / turbo_evaluations)
        if turbo_evaluations > 0
        else 0
    )

    print(f"\n🏆 Wall-clock speedup: {speedup:.1f}x faster")
    print(f"⚡ Per-evaluation efficiency: {efficiency_gain:.1f}x faster per evaluation")

    # Quality comparison
    if gepa_quality > 0 and turbo_quality > 0:
        quality_diff = turbo_quality - gepa_quality
        print(
            f"🎯 Quality: TurboGEPA {turbo_quality:.1%} vs GEPA {gepa_quality:.1%} (Δ {quality_diff:+.1%})"
        )
    elif turbo_quality > 0:
        print(f"🎯 Quality: TurboGEPA achieved {turbo_quality:.1%}")
    elif gepa_quality > 0:
        print(f"🎯 Quality: GEPA achieved {gepa_quality:.1%}")

if RUN_GEPA or RUN_TURBO:
    print("\n" + "=" * 80)
    print("BEST PROMPTS")
    print("=" * 80)

if RUN_GEPA:
    print("\n📝 GEPA Best Prompt:")
    print(gepa_prompt)

if RUN_TURBO:
    print("\n⚡ TurboGEPA Best Prompt:")
    print(turbo_prompt)

if RUN_GEPA or RUN_TURBO:
    print("\n" + "=" * 80)
