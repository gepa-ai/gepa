# TurboGEPA Quickstart

This fork adds a high-throughput, island-aware optimizer inspired by GEPA. The `src/turbo_gepa` package is self-contained and keeps dependencies to the Python standard library. Use it when you want GEPA-quality prompt evolution with aggressive async orchestration and caching, without extra infrastructure.

## Quick Start

### Single-Island Demo

The simplest way to get started:

```bash
python examples/run_local_demo.py
```

This runs a single-island optimization with a synthetic dataset and logs results to `.turbo_gepa/logs/demo_run.jsonl`.

### Multi-Island Demo

For distributed optimization with migrations:

```bash
python examples/run_multi_island_demo.py
```

This launches 4 islands in parallel with ring topology migrations every 2 rounds.

### Benchmark Analysis

After running either demo, analyze performance KPIs:

```bash
python examples/benchmark_run.py --log-path .turbo_gepa/logs/demo_run.jsonl
```

Or for JSON output:

```bash
python examples/benchmark_run.py --json
```

### Adapter Usage

Want to mirror the classic GEPA default pipeline? Use the built-in adapter:

```python
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst

dataset = [DefaultDataInst(input=row["input"], answer=row["answer"], id=row["id"]) for row in rows]
adapter = DefaultAdapter(dataset)
result = adapter.optimize(
    seeds=["You are a helpful assistant."],
    max_rounds=3,
    max_evaluations=200,
)
```

`result["pareto"]` gives you the high-quality prompts, and the adapter takes care of the sampler, cache, mutator, and orchestrator wiring.

## How It Works (Simple Explanation)

TurboGEPA finds the best prompts by testing many variations and keeping the winners. Here's how the key features work:

### 🎯 **Sharding (Progressive Evaluation)**

Think of sharding like a tournament bracket:
- **Round 1**: Test all candidates on a small sample (5-20% of data)
- **Round 2**: Only the best advance, tested on more data (20-50%)
- **Final**: Top performers get full evaluation (100% of data)

**Why?** This saves 60-70% of evaluations by eliminating bad candidates early.

**Auto-Sharding**: The system automatically picks shard sizes based on your dataset:
- Small dataset (< 50 examples): `(0.30, 1.0)` → 2 stages with reliable signal
- Medium dataset (100-500): `(0.10, 0.30, 1.0)` → 3 stages for balanced filtering
- Large dataset (1000+): `(0.05, 0.20, 1.0)` → 3 stages for maximum exploration

You can override with `shard_strategy="conservative"` (safer, slower) or `"aggressive"` (faster, riskier).

### 🏝️ **Island Hopping (Parallel Optimization)**

Imagine 4 research teams working independently, sharing their best ideas:
- Each **island** runs its own optimization with different starting points
- Every few rounds, islands **migrate** their top candidates to neighbors
- This creates diversity: one island might find prompts the others missed

**Why?** 4 islands = 4× throughput, and sharing prevents getting stuck in local optima.

### 💾 **Caching (Smart Reuse)**

Every time we test a prompt on an example, we save the result to disk:
- Same prompt + same example = instant lookup (~20-40% of evaluations)
- Cache persists across runs, so restarting is fast
- Works even if you stop and restart optimization

**Why?** Evaluation is expensive (LLM calls), so never repeat work.

### ⚡ **Async Evaluation (Massive Parallelism)**

Instead of testing prompts one-by-one sequentially:
- Test **64 prompts in parallel** (configurable via `eval_concurrency`)
- Evaluate **8 candidates at once** per round (`batch_size`)
- All examples for one candidate run concurrently

**Why?** Turn minutes into seconds by maxing out your API rate limits.

### 🧬 **Mutation (Creating Better Prompts)**

Two ways to generate new candidates:
1. **Rule-based edits** (80%): Add examples, fix grammar, clarify instructions
2. **LLM reflection** (20%): Analyze failures and suggest improvements

**Why?** Mix cheap edits with smart reflection for efficient exploration.

### 📊 **Pareto Archive (Multi-Objective)**

Keep candidates that are best at different tradeoffs:
- **Candidate A**: 95% quality, 1000 tokens
- **Candidate B**: 90% quality, 500 tokens ← shorter but still good!
- **Candidate C**: 85% quality, 200 tokens ← very concise

**Why?** You might prefer shorter prompts for cost/speed even if slightly lower quality.

### 🎨 **Quality-Diversity (QD) Grid**

Beyond Pareto, also track diverse strategies:
- Short vs long prompts
- With vs without examples
- Different reasoning patterns (chain-of-thought, step-by-step, etc.)

**Why?** Sometimes a different *approach* works better for specific cases.

### 🛑 **Automatic Stopping (Convergence Detection)**

The stop governor monitors multiple signals to detect when optimization has plateaued:
- **Hypervolume rate**: Growth of Pareto frontier area
- **Quality improvement**: Best candidate getting better
- **Cost efficiency**: Token savings rate (ROI)
- **QD novelty**: New cells filled in diversity grid
- **Frontier stability**: How much Pareto front changes

When **all signals** indicate no progress for multiple rounds, optimization stops automatically.

**Why?** Saves you from wasting budget on rounds that won't improve results.

**How to use:**
```python
adapter.optimize(
    seeds=seeds,
    enable_auto_stop=True,  # Enable automatic stopping
    max_rounds=None,         # No hard limit, will stop when converged
)
```

**Signals tracked:**
- Uses EWMA smoothing to avoid noise
- Requires 3 consecutive low-score epochs (hysteresis)
- Stops if no quality improvement for 6 epochs (hard cap)

### 🌡️ **Temperature Cycling (LLM Hyperparameter Optimization)**

**Optimize both prompt text AND temperature together** to find the best combination for your task. Temperature becomes a first-class optimization dimension alongside prompt engineering.

#### Quick Start

```python
from turbo_gepa.interfaces import Candidate

# Seed with different temperatures - ASHA will find the best ones!
seeds = [
    Candidate(text="You are a helpful assistant.", meta={"temperature": 0.0}),  # Deterministic
    Candidate(text="You are a helpful assistant.", meta={"temperature": 0.5}),  # Balanced
    Candidate(text="You are a helpful assistant.", meta={"temperature": 1.0}),  # Creative
]

result = adapter.optimize(seeds=seeds, max_rounds=5)

# Pareto frontier contains best (prompt, temperature) combinations
for candidate in result["pareto"]:
    temp = candidate.meta.get("temperature", "default")
    print(f"Quality: {candidate.quality:.2f} | Temp: {temp}")
```

#### How It Works

**1. Temperature as Metadata**
- Temperature stored in `candidate.meta["temperature"]`
- Same prompt + different temp = different candidates (different cache fingerprints)
- Archive tracks Pareto frontier across **(quality, tokens, temperature)**

**2. Automatic Exploration**
- **Initial diversity**: Seed with 3-4 different temperatures
- **ASHA testing**: All temps tested cheaply on shard 0 (5% of data)
- **Smart pruning**: Worst 60% eliminated, only winners advance
- **Focused mutations**: 20% of mutations vary temperature around successful values
- **Efficient search**: Explores ±0.3 and anchor values (0.0, 0.5, 1.0)
- **Valid range**: All temperatures clamped to [0.0, 1.0]

**3. Example Optimization Flow**
```
Round 0: Seed 3 temps [0.0, 0.5, 1.0]
         → ASHA tests on 5% data (cheap!)
         → Prunes to best 2 temps [0.5, 1.0]

Round 1: Mutate prompt text (keep successful temps)
         → Generate temp variants: [0.2, 0.7]
         → Test on 20% data
         → Best combo: "Be concise." @ temp=0.2

Round 2: Explore around winners
         → Archive has multiple (prompt, temp) pairs on Pareto frontier
         → User picks best tradeoff of quality/cost/temperature
```

#### Smart Compatibility Detection

**Upfront check (one cheap call at init):**
- Tests if model supports custom temperature with `max_tokens=1` call
- Uses litellm (works with OpenAI, Anthropic, Cohere, etc.)
- If incompatible (e.g., `o1-preview`, `gpt-5-nano`), disables temp mutations entirely

```python
adapter = DefaultAdapter(dataset, task_lm="openai/o1-preview", task_lm_temperature=0.7)
# → Prints: "⚠️  Model doesn't support custom temperature - using default"
# → Temperature mutations: DISABLED
# → Only text mutations generated
```

#### Opt-In Design

**Temperature cycling only happens if:**
1. ✅ Seed candidates include temperature metadata
2. ✅ Model supports custom temperature (auto-detected)

**Backwards compatible:**
```python
# String seeds = no temperature cycling (works as before)
adapter.optimize(seeds=["You are helpful."], max_rounds=5)

# Candidate seeds WITHOUT temp = no temperature cycling
adapter.optimize(seeds=[Candidate(text="You are helpful.")], max_rounds=5)

# Candidate seeds WITH temp = temperature cycling enabled
adapter.optimize(seeds=[Candidate(text="You are helpful.", meta={"temperature": 0.7})], max_rounds=5)
```

#### When to Use Temperature Cycling

**✅ Good use cases:**
- Math/code tasks where determinism matters (try temp=0.0)
- Creative tasks needing diversity (try temp=1.5)
- Unknown optimal temperature (let optimizer find it)
- Multi-objective: want both quality AND cost-efficiency

**❌ Skip if:**
- Model doesn't support custom temperature
- Task is insensitive to temperature
- Want to minimize evaluation budget (each temp = separate eval)

#### Temperature Ranges by Task Type

- **0.0-0.3 (Low)**: Math, code generation, factual Q&A, classification
- **0.4-0.6 (Medium)**: General chatbots, summarization, instruction following
- **0.7-0.9 (High)**: Creative writing, brainstorming, open-ended generation
- **1.0 (Maximum)**: Maximum diversity within standard range

**Pro tip**: Start with `[0.0, 0.5, 1.0]` and let ASHA + mutations find the sweet spot!

#### Staged Temperature Optimization (Advanced)

For tasks where temperature variance might confuse ASHA's early pruning, consider **staged optimization**:

```python
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst

dataset = [DefaultDataInst(input=q, answer=a, id=id) for ...]
adapter = DefaultAdapter(dataset=dataset)

result = adapter.optimize(
    seeds=["You are a helpful assistant."],
    max_rounds=10,
    max_evaluations=500,
    optimize_temperature_after_convergence=True,  # Enable staged optimization
)

# Result contains both phase outputs
phase1_prompts = result["phase1_pareto"]  # Best prompts (no temperature)
final_combinations = result["pareto"]     # Best (prompt, temperature) pairs
```

**How it works:**

1. **Phase 1 (70% budget)**: Optimize prompts WITHOUT temperature cycling
   - Seeds have no temperature → mutator focuses on text improvements
   - Reduces variance, allows ASHA to make confident pruning decisions
   - Converges to top prompt candidates

2. **Phase 2 (30% budget)**: Temperature optimization on top-K prompts
   - Takes top 5 prompts from Phase 1
   - Creates temperature-enabled seeds (starting at temp=0.5)
   - Mutator explores temperature variants (±0.3 steps, anchors at 0.0/0.5/1.0)
   - Returns final Pareto frontier with (prompt, temperature) combinations

**When to use staged optimization:**

- ✅ Tasks where temperature significantly affects variance (creative, open-ended)
- ✅ Large evaluation budgets (≥300 evals) where you can afford two phases
- ✅ When you want to decouple "what to say" from "how stochastic to be"
- ❌ Skip if budget is tight or temperature doesn't affect variance much

**Note**: If your model doesn't support custom temperature, Phase 2 is automatically skipped and you get Phase 1 results only.

## Simple Mental Model

```
Start with seed prompts
    ↓
┌─────────────────────────┐
│ EACH ROUND:             │
│  1. Pick 8 candidates   │ ← From Pareto + QD archives
│  2. Test on shard       │ ← 5%, 20%, or 100% of data
│  3. Keep winners        │ ← Top 40% advance to next shard
│  4. Generate mutations  │ ← Create new variants from winners
│  5. Share with islands │ ← Migrate top-3 every few rounds
└─────────────────────────┘
    ↓ repeat until budget
Final result: Pareto frontier of optimal prompts
```

### Example Run

```
Dataset: 100 examples
Budget: 500 evaluations
Shards: (0.05, 0.20, 1.0) → [5 examples, 20 examples, 100 examples]

Round 1: Test 20 candidates × 5 examples = 100 evals
         → 12 advance (top 60%)

Round 2: Test 12 candidates × 20 examples = 240 evals
         → 5 advance (top 40%)

Round 3: Test 5 candidates × 100 examples = 500 evals
         → Keep best 3 on Pareto frontier

Total: 840 potential evals, actually used 500 (ASHA saved 40%)
       + cache hit 30% → effectively 350 fresh evals
```

**Without sharding**: Would only test 5 candidates thoroughly with 500 evals.
**With sharding**: Explored 20+ candidates, found better optima!

## Automatic Resume (Cancel & Continue)

TurboGEPA automatically saves state after each round, so you can **cancel anytime** (Ctrl+C) and **resume right where you left off**:

```python
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst

adapter = DefaultAdapter(dataset=dataset, cache_dir=".turbo_gepa/cache")

# Run 1: Start optimization
result = adapter.optimize(
    seeds=["You are helpful."],
    max_rounds=50,
    max_evaluations=1000,
)
# ... cancel with Ctrl+C after round 10 ...

# Run 2: Resume automatically (same cache_dir)
adapter2 = DefaultAdapter(dataset=dataset, cache_dir=".turbo_gepa/cache")
result = adapter2.optimize(
    seeds=["You are helpful."],  # Seeds ignored when resuming
    max_rounds=50,  # Will continue from round 10 to 50
    max_evaluations=1000,
)
# Output: "🔄 Resumed from round 10 (237 evaluations)"
```

**How it works:**
- **After each round**: Archive, queue, and counters saved to `{cache_dir}/orchestrator_state.json`
- **On restart**: Automatically detects saved state and resumes
- **Evaluation cache**: Already-evaluated candidates served instantly from disk cache
- **On completion**: State file automatically deleted

**Benefits:**
- ✅ No lost work - cancel anytime and resume
- ✅ Fast resume - cache hits mean instant re-evaluation
- ✅ Transparent - just use the same `cache_dir`
- ✅ Atomic writes - safe to Ctrl+C during state save

## Example Scripts

### `compare_gepa_vs_turbo.py` - GEPA vs TurboGEPA Comparison

Compare original GEPA with TurboGEPA on AIME dataset:

```bash
# Run all three methods
python examples/compare_gepa_vs_turbo.py --mode all --limit 20

# Run only TurboGEPA with temperature
python examples/compare_gepa_vs_turbo.py --mode turbo-temp --limit 20 --max-rounds 10

# Run original GEPA only
python examples/compare_gepa_vs_turbo.py --mode gepa --limit 10 --max-calls 100
```

**Options:**
- `--mode`: `gepa`, `turbo`, `turbo-temp`, or `all` (default: `all`)
- `--limit`: Dataset size limit (default: 20)
- `--max-calls`: Max evaluations (default: 150)
- `--max-rounds`: Max rounds for TurboGEPA (default: 10)
- `--output`: JSON output file (default: `comparison_results.json`)

**Note**: Original GEPA mode requires OpenAI API key. TurboGEPA mode uses heuristic evaluation.

## Integration Steps

1. **Provide LLM hooks** – implement `task_lm_call` and `reflect_lm_call` in `src/turbo_gepa/user_plugs_in.py`. The file ships with deterministic heuristics for local runs; replace them with calls into your production LLM stack. `task_lm_call` should accept a candidate prompt and example payload, returning metrics like `{"quality": float, "neg_cost": float, "tokens": float}`. `reflect_lm_call` receives failure traces plus the parent prompt and must return mutated prompt strings.

2. **Pick an adapter** – for single-component prompt optimization, use `turbo_gepa.adapters.DefaultAdapter`, which mirrors GEPA’s default adapter and wires the sampler, cache, mutator, and orchestrator automatically. You can still assemble components manually for custom setups.

3. **Create a dataset iterator** – feed `InstanceSampler` the IDs for your evaluation corpus (JSONL works well). Tag difficult cases so the hardness sampler can rebalance future shards. The default adapter accepts payloads with `input`, `answer`, and optional `additional_context` fields.

4. **Instantiate the orchestrator** – wire up `DiskCache`, `AsyncEvaluator`, `Archive`, `InstanceSampler`, and a `Mutator` built with `MutationConfig`, or let the adapter handle this boilerplate. The demo at `examples/run_local_demo.py` shows the minimal wiring and includes a synthetic dataset.

5. **Launch islands** – run one orchestrator per process using `multiprocessing`. Use `islands.spawn_islands` with your orchestrator worker to enable non-blocking migrations in a ring topology.

6. **Wire migrations** – pass the `IslandContext` from your worker into `Orchestrator` so `migration_period` and `migration_k` control how elites circulate.

## Multi-Island Example

Here's a complete multi-island setup:

```python
from turbo_gepa.islands import spawn_islands, IslandContext
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.config import Config

def island_worker(island_id: int, context: IslandContext, config: Config):
    # Create island-specific components
    orchestrator = Orchestrator(
        config=config,
        evaluator=...,
        archive=...,
        sampler=...,
        mutator=...,
        cache=...,
        island_context=context,  # Enable migrations
    )

    # Run with different seeds per island for diversity
    seeds = [Candidate(text=f"Island {island_id}: ...")]
    asyncio.run(orchestrator.run(seeds, max_rounds=10))

# Launch 4 islands with ring topology
config = Config(n_islands=4, migration_period=2, migration_k=3)
worker = lambda ctx: island_worker(get_island_id(), ctx, config)
processes = spawn_islands(config.n_islands, worker)

for proc in processes:
    proc.join()
```

The orchestrator automatically:
- Sends top-K elites to the next island every `migration_period` rounds
- Receives elites from the previous island
- Deduplicates incoming candidates by fingerprint
- Inserts valid migrants into the local archive

## Config Knobs (`turbo_gepa/config.py`)

### Concurrency & Parallelism
- `eval_concurrency` (default: 64) – Max concurrent evaluations per island
- `n_islands` (default: 4) – Number of parallel optimization processes

### Successive Halving
- `shards` (default: [0.05, 0.2, 1.0]) – Shard size rungs as fractions of dataset
- `eps_improve` (default: 0.01) – Minimum improvement to promote
- `cohort_quantile` (default: 0.6) – Top 40% promote to next shard

### Quality-Diversity Archive
- `qd_bins_length` (default: 8) – Bins for prompt length dimension
- `qd_bins_bullets` (default: 6) – Bins for bullet count dimension
- `qd_flags` (default: ["cot", "format", "fewshot"]) – Feature flags for QD grid

### Mutation & Evolution
- `amortized_rate` (default: 0.8) – Probability of rule-based edits vs reflection
- `reflection_batch_size` (default: 6) – Max traces per reflection call
- `max_mutations_per_round` (default: 16) – Cap on mutations generated per round

### Merging & Compression
- `merge_period` (default: 3) – Merge candidates every N rounds
- `merge_uplift_min` (default: 0.01) – Minimum improvement to accept merge
- `max_tokens` (default: 2048) – Token budget for candidates
- `prune_delta` (default: 0.005) – Quality tolerance for compression
- `compression_shard_fraction` (default: 0.2) – Shard size for compression validation

### Island Migration
- `migration_period` (default: 2) – Migrate elites every N rounds
- `migration_k` (default: 3) – Number of elites to send per migration

### Logging & Monitoring
- `cache_path` (default: ".turbo_gepa/cache") – Disk cache directory
- `log_path` (default: ".turbo_gepa/logs") – JSONL log directory
- `log_summary_interval` (default: 10) – Emit summary every N rounds

### Other
- `batch_size` (default: 8) – Candidates evaluated per round
- `queue_limit` (default: 128) – Max queue depth for pending candidates
- `promote_objective` (default: "quality") – Objective for promotion decisions
- `compression_objective` (default: "quality") – Objective to preserve during compression

Defaults live in code so you can override quickly via keyword args or by cloning `Config`.

## Logging & Outputs

### Event Logs

Structured JSONL logs land in `.turbo_gepa/logs/` (see `logging_utils.py`). Key events include:

- `eval_start` / `eval_done` – Evaluation lifecycle
- `promote` – Candidates promoted to next shard
- `archive_update` – Archive insertions with objectives
- `mutation_proposed` / `mutation_accepted` – Mutation lifecycle
- `merge_proposed` / `merge_accepted` / `merge_rejected` – Merge lifecycle
- `compression_applied` – Token compression results
- `migrate_out` / `migrate_in` – Island migrations
- `summary` – Periodic aggregated metrics

### Summary Metrics

Every `log_summary_interval` rounds, the orchestrator emits a summary with:
- **Queue depth** – Number of pending candidates
- **Pareto size** – Current Pareto frontier size
- **Evaluations** – Total evaluations run
- **Hardness size** – Number of hard examples tracked

With `SummaryLogger` (optional), you also get:
- **Cache hit rate** – Percentage of cache hits
- **Prune rate** – Percentage pruned at shard 0
- **Latency percentiles** – P50, P95 eval times
- **Objectives histogram** – Min/max/mean/median per objective

### Outputs

After a successful run you can inspect:

- `archive.pareto_candidates()` – Quality/token Pareto frontier
- `archive.sample_qd(k)` – Diversity-focused elites from QD grid
- `.turbo_gepa/cache/` – Cached evaluation results (persistent across runs)
- `.turbo_gepa/logs/` – JSONL event logs for analysis

## Performance KPIs

The implementation targets these KPIs from the project plan:

| KPI | Target | How to Measure |
|-----|--------|----------------|
| **Prune rate at shard-1** | ≥60% | Analyze `promote` events vs shard-0 evaluations |
| **Cache hit rate** | ≥20% after warm-up | Check `summary` events or final metrics |
| **Pareto hypervolume** | Increasing | Track Pareto frontier quality over rounds |
| **Compressed variants** | ≥1 | Count `compression_applied` events |
| **Speedup vs serial** | >10× | Compare wall-clock with baseline (requires real LLM) |

Use the benchmark analyzer to validate:

```bash
python examples/benchmark_run.py
```

Example output:
```
======================================================================
TurboGEPA Benchmark Report
======================================================================

📊 Timing Metrics
  Total runtime: 12.3s
  Avg eval latency: 45.2ms
  P50 eval latency: 38.1ms
  P95 eval latency: 89.3ms

💾 Cache Metrics
  Hit rate: 28.4%
  Hits: 142, Misses: 358

📈 Archive Metrics
  Pareto size: 5
  QD grid size: 12

✨ Quality Metrics
  Initial quality: 0.452
  Final quality: 0.687
  Improvement: +0.235

======================================================================
KPI Validation
======================================================================
  ✓ Prune rate at shard 0: 64.2%
  ✓ Cache hit rate: 28.4%
  ✓ Pareto size: 5
  ✓ QD grid size: 12
  ✓ Quality improvement: 0.452 → 0.687 (Δ=+0.235)
  ✓ Compressed variants: 3
======================================================================
```

## Testing

Unit coverage for the new stack sits in `tests/turbo_gepa`. The suite only requires `pytest`:

```bash
pip install pytest
pytest tests/turbo_gepa -v
```

End-to-end test with KPI validation:

```bash
pytest tests/turbo_gepa/test_end_to_end.py -v
```

The tests avoid third-party async helpers so they execute cleanly in constrained environments.

### Test Coverage

- `test_interfaces.py` – Data contracts
- `test_cache.py` – Disk cache idempotency
- `test_scheduler.py` – ASHA promotion/pruning
- `test_archive.py` – Pareto dominance + QD grid
- `test_mutator.py` – Edit filters and deduplication
- `test_token_controller.py` – Compression logic
- `test_orchestrator.py` – Integration test
- `test_end_to_end.py` – **Comprehensive KPI validation**

## Definition of Done

The definition of done (Section 13 of the project plan) is encoded in the orchestrator loop and validated by tests:

- ✅ Local run completes on example dataset
- ✅ Non-empty Pareto set and QD grid produced
- ✅ At least one candidate reaches full shard
- ✅ Compressed variant retained on Pareto
- ✅ Logs show pruning, migrations, cache hits
- ✅ Engineer can swap LLM calls without changing internals

Swap in your task/reflection calls, run `examples/run_local_demo.py`, then scale out to multiple islands for production workloads.

## Architecture Overview

```
Multi-Island Topology (Ring)
┌─────────────┐     ┌─────────────┐
│  Island 0   │────▶│  Island 1   │
└─────────────┘     └─────────────┘
       ▲                   │
       │                   ▼
┌─────────────┐     ┌─────────────┐
│  Island 3   │◀────│  Island 2   │
└─────────────┘     └─────────────┘

Per-Island Components
┌────────────────────────────────────────┐
│           Orchestrator                 │
│  ┌──────────────────────────────────┐  │
│  │  Selection → Race → Update       │  │
│  │  Mutation → Merge → Migration    │  │
│  └──────────────────────────────────┘  │
├────────────────────────────────────────┤
│  Archive (Pareto + QD Grid)            │
│  Scheduler (ASHA Successive Halving)   │
│  Mutator (Rule + Reflection)           │
│  Evaluator (Async + Cache)             │
│  Sampler (Coreset + Hardness)          │
│  Token Controller (Compression)        │
└────────────────────────────────────────┘
```

## Next Steps

1. **Plug in real LLMs** – Replace `fake_task_runner` with actual API calls (OpenAI, Anthropic, etc.)
2. **Add your dataset** – Load your evaluation examples instead of synthetic data
3. **Tune config** – Adjust concurrency, shards, and mutation rates for your workload
4. **Scale to production** – Launch multi-island runs for large-scale optimization
5. **Monitor KPIs** – Use benchmark analyzer to track performance over time

For advanced usage and integration patterns, see `examples/run_multi_island_demo.py` and the test suite.
