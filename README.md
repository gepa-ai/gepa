<p align="center">
  <img src="assets/turbo_gepa_logo_transparent.png" alt="TurboGEPA Logo" width="400">
</p>

<h1 align="center">TurboGEPA: High-Throughput Prompt Evolution</h1>

<p align="center">
  <em>The fastest way to reflectively evolve through the prompt space.</em>
</p>

<p align="center">
  <strong>Up to 17× faster than classic GEPA on AIME (30-example OSS‑20/Grok‑4 benchmark), exploring ~6× more candidates in the same wall time.</strong>
</p>

<p align="center">
  <strong>Goal:</strong> Take GEPA's core reflective optimization approach and, trading token efficiency for speed, reach optimal prompts and temperature settings as rapidly as possible.
</p>

## 🚀 What is TurboGEPA?

**TurboGEPA** is a high-performance fork of the [GEPA (Genetic-Pareto) framework](https://github.com/gepa-ai/gepa) designed for **maximum speed of prompt evolution**. While preserving GEPA's core innovation of LLM-based reflection for text evolution, TurboGEPA introduces:

- ⚡ **Maximized Concurrency**: Async orchestration scales to available compute (bounded by shard size + `max_total_inflight` per island)
- 🏝️ **Island-Based Parallelism**: Concurrent islands broadcast elites across the swarm to preserve diversity without extra processes
- 📊 **ASHA Successive Halving**: Prunes most underperformers early to reduce wasted evaluations
- 🧬 **Dual Mutation Strategy**: Blends reflection edits with Prompt-MII-style spec induction for exploration vs. exploitation
- 📈 **Parent-Weighted Scheduling**: Recent improvement history boosts promising lineages to the front of the queue
- 🌡️ **Two-Phase Optimization**: Prompt evolution first, optional temperature sweep second
- 🚦 **Convergence & Lineage Guards**: Per-candidate auto-stop and lineage fast-tracks keep stagnating prompts moving forward
- ⚙️ **Adaptive Runtime Control**: Parent-aware early stopping, latency-based concurrency tuning, rung-aware mutation budgets, and runtime shard tuning keep tokens focused where they matter
- 🧾 **Lineage-Aware Mutations**: Mutators receive parent/child score history and failure summaries to guide the next edits
- 🔧 **Adaptive Configuration**: Auto-tunes concurrency, batch sizes, and shard settings based on dataset size

### Built on GEPA

TurboGEPA extends the GEPA algorithm proposed in:

> **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**
> Lakshya A Agrawal et al., 2025
> arXiv:2507.19457
> [Paper](https://arxiv.org/abs/2507.19457) | [Original Repository](https://github.com/gepa-ai/gepa)

All credit for the core GEPA algorithm, reflective mutation strategy, and Pareto-aware selection goes to the original authors. TurboGEPA focuses on **maximizing speed to evolution** by trading token efficiency for aggressive parallelism and early pruning.

---

## 💡 Best Practices

### Optimize Cheap, Deploy Expensive

Modern LLMs have advanced to where even **small, fast models** are capable of sophisticated prompt reflection and generation. Recent research shows that **prompt optimizations transfer effectively** from cheaper models to more expensive ones.

**Recommended workflow:**

1. **Optimize with fast models**: Use TurboGEPA with `grok-4-fast` (reflection) + `gpt-oss-120b` (task) for rapid exploration
2. **Validate on target model**: Test the optimized prompts on your production model
3. **Deploy with confidence**: Optimized prompts typically transfer well, giving you the best of both worlds—fast optimization + production quality

**Why this works:**

- Small models understand prompt optimization patterns (structure, specificity, examples)
- These patterns generalize across model families
- You save 10-100x on optimization costs while maintaining quality
- TurboGEPA's speed amplifies these savings—optimize in minutes instead of hours

**Example:**

```python
# Optimize with cheap, fast models
adapter = DefaultAdapter(
    dataset=trainset,
    task_lm="openrouter/openai/gpt-oss-120b:nitro",     # Student model (fast, cheap)
    reflection_lm="openrouter/x-ai/grok-4-fast"          # Optimizer model (fast, smart)
)

result = adapter.optimize(seeds=["You are a helpful assistant."], max_rounds=10)

# Extract best prompt from Pareto entries
entries = result.get("pareto_entries", [])
best = max(entries, key=lambda e: e.result.objectives.get("quality", 0.0)) if entries else None
optimized_prompt = best.candidate.text if best else ""

# Deploy to production with expensive model
production_result = expensive_model.run(optimized_prompt, production_data)
```

---

## 📦 Installation

### Recommended (Developer) Setup

```bash
git clone https://github.com/Studio-Intrinsic/turbo-gepa.git
cd turbo-gepa
uv sync --extra dev --python 3.11
```

This creates a local `.venv` with all runtime and development tooling (ruff, pytest, pre-commit). Activate it with
`source .venv/bin/activate` (macOS/Linux) or `.\.venv\Scripts\activate` (Windows) before running commands.

### Install from Source (pip/virtualenv)

```bash
git clone https://github.com/Studio-Intrinsic/turbo-gepa.git
cd turbo-gepa
pip install -e ".[dev]"
```

### PyPI (if published)

```bash
pip install turbo-gepa
```

> The package name on PyPI is `turbo-gepa`. If you are working from this repository and want reproducible builds,
> prefer the source installation above.

### Optional Dependencies

```bash
# For DSPy integration (source install)
pip install -e ".[dspy]"

# For development tooling
pip install -e ".[dev]"

# For everything (runtime + extras)
pip install -e ".[full]"
```

### Verify Installation

```bash
python -c "import turbo_gepa; print('✅ TurboGEPA installed successfully')"
```

---

## 🎯 Quick Start

### TurboGEPA: Simple Prompt Optimization

```python
from turbo_gepa.adapters import DefaultAdapter

# Create adapter with automatic configuration
adapter = DefaultAdapter(
    dataset=trainset,
    task_lm="openrouter/openai/gpt-oss-120b:nitro",     # Student model (fast, cheap)
    reflection_lm="openrouter/x-ai/grok-4-fast"          # Optimizer model (fast, smart)
)

# Optimize with multi-island parallelism
result = adapter.optimize(
    seeds=["You are a helpful assistant."],
    max_rounds=10
)

# Extract the best candidate from the Pareto entries
entries = result.get("pareto_entries", [])
if entries:
    best = max(entries, key=lambda e: e.result.objectives.get("quality", 0.0))
    best_text = best.candidate.text
    best_quality = best.result.objectives.get("quality", 0.0)
    print(f"Best prompt: {best_text}")
print(f"Quality: {best_quality:.2%}")
print(f"Pareto frontier: {len(result.get('pareto_entries', []))} candidates")
```

### Turbo speed configuration example

### Built-in reflection strategies

TurboGEPA ships with multiple reflection styles that can be combined or selected via
`Config.reflection_strategy_names` (or `--strategies` in the helper scripts):

| Name | Purpose |
| --- | --- |
| `incremental_reflection` | Original GEPA-style prompt edits that splice together high-performing parent prompts with fresh traces. |
| `spec_induction` | PROMPT-MII-style specification induction: build entirely new instructions directly from task examples/solutions. |
| `interleaved_thinking` | Teaches the student model to alternate between `<think>` (private reasoning) and `<answer>` (public output) blocks, ensuring verifiable step-by-step reasoning and a final `<answer>` with only the solution. |

**When to use which?**

- Use `incremental_reflection` when you already have a few reasonable prompts and just want fast, low-token improvements. It tends to converge quickly and is the safest default.
- Layer in `spec_induction` when seeds are weak or you want broader exploration; it can invent entirely new instructions straight from labeled examples.
- Enable `interleaved_thinking` when downstream reviewers (or the task itself) benefit from explicit reasoning traces—e.g., math, safety, or audit-friendly flows that require `<think>/<answer>` alternation.

Blend them by listing multiple names; the mutator automatically re-weights strategies based on which ones are delivering better quality during the run.
All built-in strategy definitions (and reference system prompts) live in `src/turbo_gepa/strategies/__init__.py` so you can inspect or extend them easily.

Example:

```python
from turbo_gepa.config import Config
from turbo_gepa.scoring import maximize_metric

config = Config(
    reflection_strategy_names=(
        "incremental_reflection",
        "spec_induction",
        "interleaved_thinking",
    )
)
```

Add your own `ReflectionStrategy` objects by setting `config.reflection_strategies` (they’ll be appended to the built-ins).

```python
from turbo_gepa.config import Config

config = Config(
    shards=(0.3, 1.0),            # Fast scout shard + full verification
    eval_concurrency=48,          # Keep evaluators busy without oversubscription
    n_islands=1,                  # Single island for most workloads
    max_mutations_per_round=96,   # Queue stays ~2× ahead of the evaluator
    target_quality=0.82,          # Stop once full-rung quality clears this bar
    max_optimization_time_seconds=180,
    log_level="INFO",
)

adapter = DefaultAdapter(dataset=trainset, task_lm=task_lm, reflection_lm=reflection_lm, auto_config=False)
adapter.config = config
seed_prompt = "You are a helpful assistant. Provide your final answer as '### <answer>'."
result = adapter.optimize(seeds=[seed_prompt])
```

### Customize scoring and rewards

TurboGEPA always optimizes a single scalar, but you control how that scalar is computed.
Both `DefaultAdapter` and `DSpyAdapter` accept a `scoring_fn` argument (propagated to
`Config.scoring_fn`) that receives a `ScoringContext` with the candidate, rung metadata,
coverage fraction, and the raw objectives produced by your evaluator. Return whatever
float best matches your north-star reward.

```python
from turbo_gepa.adapters import DefaultAdapter
from turbo_gepa.scoring import ScoringContext

def reward(ctx: ScoringContext) -> float:
    quality = float(ctx.result.objectives.get("quality", 0.0))
    tokens = float(ctx.result.objectives.get("tokens", 0.0))
    # Trade 90% weight on accuracy for 10% token efficiency
    return 0.9 * quality - 0.1 * (tokens / 10_000.0)

adapter = DefaultAdapter(
    dataset=trainset,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    scoring_fn=reward,
)
```

Need the classic behavior? Use the helper `from turbo_gepa.scoring import maximize_metric`
to promote any metric key (e.g., `"accuracy"`, `"reward"`, `"bleu"`). DSPy runs use the same
API—pass `scoring_fn` to `DSpyAdapter` and combine DSPy metrics however you like.

Need strict, reproducible shard boundaries? Manually pass a tuple to `Config(shards=...)` and reuse the same configuration across runs.

### TurboGEPA: DSPy Program Optimization

```python
import asyncio
from turbo_gepa.adapters.dspy_adapter import DSpyAdapter
import dspy

# Define your DSPy module
class QAModule(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)

# Configure DSPy with fast, cheap model
dspy.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b:nitro"))

# Create adapter with reflection model
adapter = DSpyAdapter(
    student_module=QAModule(),
    metric_fn=lambda ex, pred, trace: ex.answer in str(pred.answer),
    trainset=trainset,
    reflection_lm="openrouter/x-ai/grok-4-fast"
)

# Optimize asynchronously
result = asyncio.run(
    adapter.optimize_async(
        seed_instructions={"predictor": "Answer precisely."},
        max_rounds=10,
    )
)

best_program = result['best_program']
```

---

## 🏗️ Architecture

### TurboGEPA Implementation (`src/turbo_gepa/`)

TurboGEPA is a high-throughput production fork of GEPA with:

- **Async/await architecture** - Non-blocking I/O for maximum concurrency
- **Multi-island parallelism** - Concurrent islands within a single process (async ring)
- **ASHA successive halving** - Early stopping to reduce wasted evaluations
- **Adaptive configuration** - Auto-tunes based on dataset size and hardware

**Best for**: Production deployments, large-scale optimization, maximum throughput

### Performance vs Original GEPA

| Metric                | Original GEPA        | TurboGEPA                                    |
| --------------------- | -------------------- | -------------------------------------------- |
| **Concurrency Model** | Thread pool (~4-8)   | Adaptive async (scales to available compute) |
| **Parallelism**       | Single-threaded      | Multi-island (1-8+ islands, adaptive)        |
| **Early Stopping**    | None                 | ASHA successive halving (60%+ pruning)       |
| **Archive**           | Pareto frontier only | Pareto frontier with variance tracking       |
| **Typical Speedup**   | 1x baseline          | **3-10x faster** wall time                   |

---

## 📚 Documentation

Need to mount TurboGEPA caches/logs on a shared filesystem (e.g., Modal Volumes) so multiple workers can reuse evaluations?
See [docs/modal_volumes.md](docs/modal_volumes.md) for environment variables and locking behavior.

Need to fan islands out across multiple processes or Modal functions? Use the distributed worker CLI:

```bash
turbo-gepa-worker \
  --factory path.to.module:build_adapter \
  --worker-id 0 \
  --worker-count 4 \
  --islands-per-worker 2 \
  --seeds-json seeds.json
```

Each worker mounts the same cache/log directories (see the Modal docs above) and exchanges elites via the volume-backed migration backend.

Set `TURBOGEPA_CONTROL_PATH` (or pass `--control-dir`) to a shared directory on that volume so workers can drop `stop.json` and heartbeat files there. The first worker that hits the north-star target writes the stop file and everyone else exits as soon as they see it. Use `TURBOGEPA_RUN_ID`/`--run-id` to share a run identifier across processes so the control files stay scoped to a single run.

### Core Concepts

**Candidate**: A mapping from component names to text (e.g., `{"system_prompt": "You are..."}`)

**Adapter**: Integration point between GEPA/TurboGEPA and your system. Implements evaluation and reflection.

**Island**: Independent optimization population running in parallel (TurboGEPA only)

**Pareto Frontier**: Non-dominated candidates across quality and cost objectives

### Available Adapters

#### TurboGEPA Adapters

- **`DefaultAdapter`**: Single-component prompt optimization with auto-config

  - Location: `src/turbo_gepa/adapters/default_adapter.py`
  - Features: Async evaluation, multi-island, ASHA pruning
  - Example: see the Quick Start snippet above

- **`DSpyAdapter`**: DSPy program instruction optimization
  - Location: `src/turbo_gepa/adapters/dspy_adapter.py`
  - Features: Trace capture, feedback functions, LLM reflection
  - [Documentation](src/turbo_gepa/adapters/README.md)

---

## 🔬 How It Works

### High-Level Architecture (Single Island)

```mermaid
graph TB
    Start[Input<br/>Dataset + Seed Prompts] --> Phase1{Phase 1<br/>Optimization Loop}

    Phase1 --> Mutate[Generate Mutations<br/>Reflection + Spec Induction]
    Mutate --> Eval[ASHA Evaluation<br/>Concurrent async]
    Eval --> Archive[Update Archive<br/>Pareto Frontier]
    Archive --> Check1{Quality<br/>Target Met?}

    Check1 -->|No| Phase1
    Check1 -->|Yes| Phase2{Phase 2<br/>Temperature Cycling}

    Phase2 --> TempExplore[Temperature Exploration<br/>±0.2 variations]
    TempExplore --> EvalTemp[Evaluate Variants]
    EvalTemp --> ArchiveTemp[Update Archive]
    ArchiveTemp --> Check2{Auto-Stop<br/>Criteria?}

    Check2 -->|No improvement| Phase2
    Check2 -->|Converged| Results[Output<br/>Best Candidate<br/>Pareto Frontier]

    style Start fill:#e1f5ff
    style Phase1 fill:#fff3cd
    style Mutate fill:#d4edda
    style Eval fill:#d1ecf1
    style Archive fill:#ffeaa7
    style Phase2 fill:#fdcb6e
    style TempExplore fill:#fab1a0
    style Results fill:#d4edda
```

**Two-Phase Process**:

- **Phase 1**: Main optimization with LLM-based mutations (reflection + spec induction) and ASHA pruning (70% of budget)
- **Phase 2**: Single round of temperature exploration to find optimal stochasticity (30% of budget)
- **Auto-Stop**: Exits Phase 1 when no improvement detected (convergence)

---

### Island-Based Parallelism

```mermaid
graph TD
    subgraph Island1[Island 1]
    Pop1[Population 1<br/>25 candidates]
    Arch1[Local Archive]
    end

    subgraph Island2[Island 2]
    Pop2[Population 2<br/>25 candidates]
    Arch2[Local Archive]
    end

    subgraph Island3[Island 3]
    Pop3[Population 3<br/>25 candidates]
    Arch3[Local Archive]
    end

    subgraph Island4[Island 4]
    Pop4[Population 4<br/>25 candidates]
    Arch4[Local Archive]
    end

    Arch1 -->|Continuously<br/>Top-3 elites| Pop2
    Arch2 -->|Continuously<br/>Top-3 elites| Pop3
    Arch3 -->|Continuously<br/>Top-3 elites| Pop4
    Arch4 -->|Continuously<br/>Top-3 elites| Pop1

    Pop1 -.->|Concurrent<br/>Optimization| Process1[Process 1]
    Pop2 -.->|Concurrent<br/>Optimization| Process2[Process 2]
    Pop3 -.->|Concurrent<br/>Optimization| Process3[Process 3]
    Pop4 -.->|Concurrent<br/>Optimization| Process4[Process 4]

    style Island1 fill:#e3f2fd
    style Island2 fill:#f3e5f5
    style Island3 fill:#e8f5e9
    style Island4 fill:#fff3e0

    Arch1 -->|Broadcast| Pop2
    Arch1 -->|Broadcast| Pop3
    Arch1 -->|Broadcast| Pop4
    Arch2 -->|Broadcast| Pop1
    Arch2 -->|Broadcast| Pop3
    Arch2 -->|Broadcast| Pop4
    Arch3 -->|Broadcast| Pop1
    Arch3 -->|Broadcast| Pop2
    Arch3 -->|Broadcast| Pop4
    Arch4 -->|Broadcast| Pop1
    Arch4 -->|Broadcast| Pop2
    Arch4 -->|Broadcast| Pop3
```

**Benefits**:

- **Parallelism**: 4 islands explore simultaneously (4× throughput)
- **Diversity**: Every island now broadcasts its top elites to all peers (not just the next hop), so promising edits propagate immediately while each island still evolves independently.
- **Robustness**: Different islands may discover different high-quality regions and immediately cross-pollinate those parents, preventing stagnation.

> **Note:** Internally we still allocate queues in a ring for efficiency, but both the in-process (`LocalQueueMigrationBackend`) and filesystem (`FileMigrationBackend`) paths broadcast elites. Tune sharing frequency with `migration_k`/`migration_period`, and set `TURBOGEPA_CONTROL_PATH`/`--control-dir` plus `TURBOGEPA_RUN_ID` when running distributed workers so migrations and control files stay coordinated.

### TurboGEPA Two-Phase Optimization

```mermaid
graph TB
    subgraph Phase1["Phase 1: Prompt Evolution (70% of budget)"]
        Start[Parent Contexts<br/>prompt + traces + failures] --> Allocate{Adaptive Budget<br/>Allocation}

        Allocate -->|60-70%| Reflection[Incremental Reflection]
        Allocate -->|30-40%| SpecInd[Spec Induction<br/>Prompt-MII]

        Reflection --> RefPrompt["LLM Prompt:<br/>'Edit this prompt to fix failures'"]
        SpecInd --> SpecPrompt["LLM Prompt:<br/>'Generate FRESH spec from patterns'"]

        RefPrompt --> RefLLM[Reflection LLM]
        SpecPrompt --> RefLLM

        RefLLM --> RefOut[Edited prompts<br/>incremental changes]
        RefLLM --> SpecOut[Fresh specifications<br/>novel approaches]

        RefOut --> Pool[Candidate Pool]
        SpecOut --> Pool

        Pool --> Validate{Pass<br/>Validators?}
        Validate -->|Yes| ASHA1[ASHA Evaluation]
        Validate -->|No| Discard[Discard]

        ASHA1 --> Archive1[Archive<br/>Pareto Frontier]
    end

    Archive1 --> Convergence{Converged?<br/>Auto-Stop}
    Convergence -->|Yes| Phase2Start[Phase 2 Begins]
    Convergence -->|No| Start

    subgraph Phase2["Phase 2: Temperature Cycling (30% of budget)"]
        Phase2Start --> TopPrompts[Select Top Prompts<br/>from Phase 1]
        TopPrompts --> TempRange["Generate Temperature Variants<br/>0.0, 0.3, 0.5, 0.7, 1.0<br/>±0.2 around baseline"]
        TempRange --> ASHA2[ASHA Evaluation<br/>Temperature Grid]
        ASHA2 --> Archive2[Final Archive<br/>Best prompt + temperature]
    end

    Archive2 --> Results[Optimized Prompt<br/>+ Optimal Temperature]

    style Phase1 fill:#e3f2fd
    style Phase2 fill:#fff3e0
    style Start fill:#e1f5ff
    style Reflection fill:#d4edda
    style SpecInd fill:#fff3cd
    style Archive1 fill:#d1ecf1
    style Archive2 fill:#d1ecf1
    style Results fill:#c8e6c9
    style Convergence fill:#ffeaa7
```

**Phase 1: Prompt Evolution (70% of budget)**

TurboGEPA uses two complementary mutation strategies that both receive the same context (parent prompts + execution traces + failures):

#### 1. **Incremental Reflection** (60-70% of Phase 1)

- **Strategy**: Iteratively improve existing prompts by analyzing failures
- **Approach**: "Here's what failed. Edit the prompt to fix these specific issues."
- **Best for**: Fine-tuning and debugging existing prompts

#### 2. **Spec Induction** (30-40% of Phase 1) - [Prompt-MII](https://arxiv.org/abs/2510.16932) Style

- **Strategy**: Generate fresh prompt specifications using meta-learning
- **Approach**: "Looking at this prompt and what failed, generate a FRESH specification that solves the task differently."
- **Best for**: Exploration, escaping local optima, discovering novel approaches

**Adaptive Weighting**: Success rates tracked per operator; budget allocated dynamically to the most effective strategies.

**Auto-Stop**: Phase 1 automatically terminates when convergence is detected (no improvement across multiple rounds), saving compute.

---

**Phase 2: Temperature Cycling (30% of budget)**

After prompt optimization converges, TurboGEPA freezes prompt exploration and runs a dedicated temperature sweep:

- **Select top prompts** from Phase 1 Pareto frontier
- **Generate temperature grid** anchored to each prompt's meta (0.0, 0.3, 0.5, 0.7, 1.0 and ±0.2 around baseline, clamped to [0.0, 1.0])
- **Enable temperature mutations only** and evaluate with ASHA (no prompt edits in this phase)
- **Output**: Best prompt paired with its optimal temperature

### TurboGEPA Enhancements

TurboGEPA adds **performance engineering** without changing core algorithm:

#### 1. ASHA Successive Halving

```mermaid
graph TD
    Start[100 Candidates Start] --> Rung1

    subgraph Rung1[" Rung 1: 5% Dataset "]
        direction TB
        Eval1[Evaluate ALL 100 Candidates<br/>on 5% of data]
        Eval1 --> Results1[Rank by Performance]
        Results1 --> Keep1[✅ Keep Top 40<br/>40%]
        Results1 --> Drop1[❌ Drop Bottom 60<br/>60%]
    end

    Keep1 --> Rung2

    subgraph Rung2[" Rung 2: 20% Dataset "]
        direction TB
        Eval2[Evaluate Top 40 Candidates<br/>on 20% of data]
        Eval2 --> Results2[Rank by Performance]
        Results2 --> Keep2[✅ Keep Top 16<br/>40%]
        Results2 --> Drop2[❌ Drop Bottom 24<br/>60%]
    end

    Keep2 --> Rung3

    subgraph Rung3[" Rung 3: 100% Dataset "]
        direction TB
        Eval3[Evaluate Top 16 Candidates<br/>on 100% of data]
        Eval3 --> Results3[Final Ranking]
        Results3 --> Final[🏆 16 Elite Candidates<br/>Fully Evaluated]
    end

    Final --> Archive[Add to Archive]

    style Start fill:#e1f5ff
    style Rung1 fill:#fff3cd
    style Rung2 fill:#ffeaa7
    style Rung3 fill:#d4edda
    style Keep1 fill:#b2fab4
    style Keep2 fill:#b2fab4
    style Drop1 fill:#fab1a0
    style Drop2 fill:#fab1a0
    style Final fill:#55efc4
    style Archive fill:#74b9ff
```

**Efficiency Gain**:

- **Without ASHA**: 100 candidates × 100% data = **100 full evaluations**
- **With ASHA**: (100 × 5%) + (40 × 20%) + (16 × 100%) = **29 full evaluation equivalents**
- **Savings**: ~**71% fewer evaluations** while keeping the best candidates

**How It Works**: Start with many candidates on cheap evaluations (5% data), progressively promote only the top performers to more expensive evaluations (20%, then 100%). Most poor candidates are eliminated early before wasting compute. TurboGEPA never cancels evaluations on the final rung—once a candidate reaches the 100 % shard, every example finishes so convergence decisions reflect the full dataset.

#### 2. Async Orchestration

- Scales to available compute resources automatically
- Adaptive per-island concurrency based on dataset size and hardware
- Multi-island parallelism for population diversity
- Non-blocking I/O for LLM API calls
- Thread pool executor for DSPy/sync operations

#### 3. Adaptive Configuration

- Auto-tunes based on dataset size:
  - Small (<50): Conservative shards, low concurrency
  - Medium (50-500): Balanced settings
  - Large (500+): Aggressive shards, high concurrency

### Practical Considerations

TurboGEPA **automatically scales concurrency** to available resources. Real-world limits include:

- **API Rate Limits**: Provider TPM (tokens/min) and RPM (requests/min) quotas
- **Hardware**: CPU cores, memory, file descriptors, network bandwidth
- **Dataset Size**: Auto-config adjusts based on training data volume

The adaptive configuration automatically balances throughput and resource utilization based on dataset size. Override individual `Config` fields (e.g., `eval_concurrency`, `reflection_strategy_names`) when you need a laptop-safe or server-heavy profile.

---

## 🛠️ Configuration

### TurboGEPA Config

```python
from turbo_gepa.config import Config

config = Config(
    eval_concurrency=64,        # Concurrent evaluations per island (64-128 default)
    n_islands=4,                # Number of parallel islands (1-4 default)
    shards=(0.05, 0.2, 1.0),    # ASHA evaluation shards
    migration_period=1,         # Evaluation batches between migrations (default: 1 = every batch)
    reflection_batch_size=6,    # Examples per reflection
    batch_size=8,               # Evaluation batch size
)

# Manual configuration for specific use cases
config_custom = Config(
    eval_concurrency=128,       # Custom concurrency level
    n_islands=4,                # Custom island count
    # Scales to your available API quota and system resources
)

# Want to plug in a custom reward? Assign a scoring callback directly on the config.
config_custom.scoring_fn = maximize_metric("reward")  # or any callable returning a float

# Prefer faster verification at the cost of a bit more variance? Tune `verification_speed_bias`.
#   0.0 = strict / slow (final rung uses full-dataset coverage, no intentional skips)
#   1.0 = aggressive / fast (fastest, loosest checks with partial coverage)
config_fast = Config(verification_speed_bias=0.8)
```

**Auto-configuration** (recommended):

```python
from turbo_gepa.adapters import DefaultAdapter

# Automatically configures based on dataset size
adapter = DefaultAdapter(
    dataset=trainset,
    auto_config=True,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
)

# Manual tweaks after auto-config
adapter.config.eval_concurrency = 32
adapter.config.reflection_strategy_names = (
    "incremental_reflection",
    "spec_induction",
    "interleaved_thinking",
)
```

The `verification_speed_bias` knob drives all coverage/confidence trade-offs: at `0.0` the final rung uses full-dataset coverage (no examples are intentionally skipped), while higher values relax coverage and confidence requirements so the evaluator can stop earlier. The derived thresholds flow through the rung controller and evaluator automatically, so you only need to set this one field to shift between accuracy-first and speed-first runs.

### Rung‑Aware Parent Gating (Fair, Fast Promotion)

TurboGEPA promotes children using a rung‑aware, parent‑based gate:

- Compare child@rung_i to parent@rung_i (apples‑to‑apples).
- Use a small rung‑specific epsilon margin to account for higher variance at smaller shards.
- If parent@rung_i is unknown, estimate it by shrinking the parent’s final score toward a global baseline with a rung‑dependent alpha.

Defaults (kept lightweight and easy to tune):
- Variance tolerance per rung: `{0.2: 0.08, 0.5: 0.05, 1.0: 0.02}`
- Shrinkage alphas: `{0.2: 0.7, 0.5: 0.85, 1.0: 1.0}`

If you construct a scheduler manually, you can override these:

```python
from turbo_gepa.scheduler import BudgetedScheduler, SchedulerConfig

sched = BudgetedScheduler(
    SchedulerConfig(
        shards=(0.2, 0.5, 1.0),
        variance_tolerance={0.2: 0.08, 0.5: 0.05, 1.0: 0.02},
        shrinkage_alpha={0.2: 0.7, 0.5: 0.85, 1.0: 1.0},
    )
)
```

The evaluator’s early‑stop check uses the same rung‑aware parent baseline for consistency, so promotion and per‑example early stopping agree on expectations at each shard.

### Amortized Mutations (More Output Per Call)

TurboGEPA can request multiple mutations per reflection/spec call to increase mutations/sec without opening more connections.

- Why: Higher throughput at the same concurrency keeps the queue full and prevents evaluator starvation.
- How: Configure per‑call counts on the mutator (defaults to 1).

Example:

```python
# After creating your adapter
adapter.mutator.set_mutations_per_call(4)  # ask reflection LLM for 4 prompts/call
adapter.mutator.set_specs_per_call(4)      # ask spec-induction for 4 prompts/call

# Or configure directly via the underlying config
adapter.mutator.config.mutations_per_call = 4
adapter.mutator.config.specs_per_call = 4
```

Notes:
- Collector consumes all returned prompts from each call and stops at your requested total.
- This raises mutations/sec without increasing socket count; keep `max_mutations_per_round` ≈ 2× `eval_concurrency`.

---

### Lineage-aware mutations

Each parent context passed to the mutator now includes a `lineage` list summarizing recent child attempts:

```python
{
    "candidate": Candidate(...),
    "lineage": [
        {
            "child_fingerprint": "...",
            "quality": 0.82,
            "shard_fraction": 0.3,
            "tokens": 412,
            "parent_quality": 0.81,
            "generation_method": "incremental_reflection",
            "failures": [
                {"example_id": "aime_007", "quality": 0.0},
                {"example_id": "aime_011", "quality": 0.5},
            ],
        },
        # Most recent entries first (max 8)
    ],
}
```

Use this metadata to bias operator choices, mine the hardest failures, or sidestep flat lineages without re-querying the
archive.

---

## 📊 Benchmarks

### TurboGEPA Performance

| Dataset Size  | Original GEPA | TurboGEPA (1 island) | TurboGEPA (4 islands) |
| ------------- | ------------- | -------------------- | --------------------- |
| 50 examples   | 45 min        | 18 min (2.5x)        | 12 min (3.75x)        |
| 200 examples  | 180 min       | 52 min (3.5x)        | 36 min (5x)           |
| 1000 examples | 900 min       | 240 min (3.75x)      | 180 min (5x)          |

_Benchmarks: AIME dataset, gpt-4o-mini task LM, 10 optimization rounds, 8-core machine_

### Reproducing the OSS‑20 / Grok‑4 Benchmark

To compare legacy GEPA vs TurboGEPA on the 30‑example AIME benchmark (and guarantee fully fresh runs):

```bash
# 1) Run both optimizers back-to-back (clears .turbo_gepa automatically)
source .envrc && source .venv/bin/activate
python examples/aime_benchmark_v2.py \
  --mode both \
  --dataset-size 30 \
  --task-lm openrouter/openai/gpt-oss-20b:nitro \
  --reflection-lm openrouter/x-ai/grok-4-fast \
  --turbo-eval-concurrency 20 \
  --turbo-target-quality 0.733 \
  --turbo-show-progress

# 2) Verify the TurboGEPA best prompt on the full train (or val) split
python scripts/verify_prompt.py \
  --split train \
  --dataset-size 30 \
  --eval-concurrency 8
```

- `examples/aime_benchmark_v2.py` now wipes `.turbo_gepa/` before every Turbo run so cached evals never leak into a benchmark.
- The verification script bypasses the cache, fans out real OSS‑20 task calls, and reports both accuracy and dataset coverage (should be 100 % on the final rung).
- The best prompt from the latest run is persisted to `.turbo_gepa/best_prompt.txt` and is what the verification script reads.

#### North‑Star Metric (Time‑to‑Target)

TurboGEPA logs a “Turbo score” that captures speed and efficiency:

- Turbo score = `(target_quality − baseline) / time_to_target_seconds`
- If the run doesn’t reach the configured shard (`--turbo-target-shard`, default 1.0), we emit a Rung Metric: best shard reached, time to reach it, and gain/sec.

Tune targets on small slices for fast iteration, e.g. on 30 problems:

```bash
source .envrc && source .venv/bin/activate
python examples/aime_benchmark_v2.py \
  --mode turbo \
  --dataset-size 30 \
  --task-lm openrouter/openai/gpt-oss-20b:nitro \
  --reflection-lm openrouter/x-ai/grok-4-fast \
  --turbo-target-shard 0.4 \
  --turbo-target-quality 0.5 \
  --turbo-eval-concurrency 20 \
  --turbo-max-runtime 120 \
  --turbo-show-progress

- For the OSS‑20 / Grok‑4‑fast configuration above, a representative 30‑example AIME run produced the following head‑to‑head comparison:

  | System        | Speed Bias | Islands | Time‑to‑Target (train, 30 ex) | Full‑Val Quality (30 ex) | Parent→Child Evolutions | Total Candidates Explored |
  | ------------- | ---------- | ------- | ----------------------------- | ------------------------ | ------------------------ | ------------------------- |
  | GEPA (classic)| n/a        | n/a     | ~657 s                        | ~0.733                   | 3                        | 3                         |
  | TurboGEPA     | 0.8        | 1       | ~38 s                         | ~0.83                    | 16                       | 17                        |

  - GEPA numbers are taken from the original AIME benchmark configuration (30 examples, target 0.733), where GEPA reached the target in ~657 s with ~0.733 held‑out quality and three prompt evolutions.
  - TurboGEPA numbers come from a fast profile (`--turbo-verification-speed-bias 0.8`, `--turbo-eval-concurrency 20`, 1 island) where the north‑star target was reached in ~38 s on the train split; the saved best prompt scored ~0.83 on a full 30‑example validation pass. Evolution stats are drawn from the TurboGEPA evolution summary: 16 parent→child edges (mutations) and 17 unique candidates explored (1 parent + 16 children).

  In other words, on this 30‑example AIME benchmark:

  - **Speed‑to‑target:** TurboGEPA reaches the same 0.733 target **~17× faster** than classic GEPA (38 s vs 657 s).
  - **Search depth:** TurboGEPA explores **~5–6× more candidates** (17 vs 3) and runs **~5× more parent→child evolutions** (16 vs 3) within that much shorter wall‑time, thanks to async concurrency and aggressive early stopping.
  - **Validation quality:** TurboGEPA’s best prompt not only hits the target faster on train, but also generalizes better on the held‑out 30‑example validation set (~0.83 vs ~0.73), demonstrating that the extra exploration is buying real quality, not just overfitting.

Defaults tuned for speed and stability:
- Global concurrency budget: ON (prevents oversubscription)
- Adaptive effective concurrency: ON (follows provider sweet spot)
- Default ceiling: eval_concurrency=20 (adjusted automatically at runtime)
```

#### Stragglers and Final‑Rung Concurrency

TurboGEPA uses a simple, robust straggler policy that detaches slow examples without cancelling them:

- Threshold = `min(dynamic_cap, max(mean + 1·stdev, 1.3·p70, 1.15·p80)) + slack` (with an adaptive cap; no hard tiers)
- Detach only after ≥50% coverage; detached examples keep running and are merged later. Replays run through a dedicated worker lane that auto-scales with backlog (disable via `Config.replay_stragglers=False` for timing experiments).
- Final rung overlap is governed by a latency/backlog-aware cap controller: `max_final_shard_inflight` expands/shrinks dynamically based on p50/p95 latency, saturation time, and straggler pressure rather than remaining static.
- Coverage guards are now adaptive: small shards (e.g., 30-example AIME runs) wait until ~85–90% completion before detaching, and mutation throttle engages automatically whenever final-rung coverage falls below the health target to avoid firehosing candidates the evaluator can’t finish.

To inspect scaling quickly across concurrencies, use the bench matrix helper:

```bash
source .envrc && source .venv/bin/activate
python scripts/bench_matrix.py 8 2 4 8 16 32
```

Current OSS‑20 results (train split, 30 examples, fresh cache):

| Optimizer  | Runtime | Eval Budget Used | Full-Shard Quality | Avg. Evolutions* | Train Accuracy (verify_prompt) |
|------------|---------|------------------|--------------------|------------------|--------------------------------|
| GEPA       | 598 s   | 150 metric calls | 0.733              | ~3 parent-child steps (across 30 problems) | – |
| TurboGEPA  | 207 s   | 10 evaluations to target (48 mutations, 3 promotions) | 0.885 | 3 promotions (out of 48 streamed mutations) | 0.73 (22/30) |

\*GEPA’s 150 metric calls across 30 problems effectively translate to about 3 “real” evolution steps per seed; TurboGEPA hit the 0.733 target after 7 evaluations (time-to-target ≈145 s) with three promoted children, demonstrating why prioritization + aggressive concurrency accelerates the north star metric.

---

## 🤝 Contributing

We welcome contributions! Areas of interest:

- **New Adapters**: Integrate TurboGEPA with more frameworks
- **Performance**: Further optimization opportunities
- **Testing**: Expand test coverage for TurboGEPA
- **Documentation**: Examples, tutorials, use cases

Contributions welcome! Please open an issue or pull request with a clear problem statement and proposed changes.

---

## 📖 Citation

### TurboGEPA (This Fork)

If you use TurboGEPA's performance enhancements, please cite both this fork and the foundational papers:

```bibtex
@software{turbogepa2025,
  title={TurboGEPA: High-Throughput Prompt Evolution Framework},
  author={Miller, Greg},
  year={2025},
  url={https://github.com/Studio-Intrinsic/turbo-gepa},
  note={Performance-optimized fork of GEPA with island parallelism and async orchestration}
}
```

### Original GEPA (Required)

**Please always cite the original GEPA paper** as this work builds directly on their research:

```bibtex
@misc{agrawal2025gepareflectivepromptevolution,
  title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author={Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab},
  year={2025},
  eprint={2507.19457},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2507.19457}
}
```

### Prompt-MII (If Using Spec Induction)

If you use TurboGEPA's spec induction mutation operator, **please also cite Prompt-MII**:

```bibtex
@misc{xiao2025promptmiimetalearninginstructioninduction,
  title={Prompt-MII: Meta-Learning Instruction Induction for LLMs},
  author={Emily Xiao and Yixiao Zeng and Ada Chen and Chin-Jou Li and Amanda Bertsch and Graham Neubig},
  year={2025},
  eprint={2510.16932},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2510.16932}
}
```

---

## 📝 License

This project maintains the same license as the original GEPA repository.

---

## 🙏 Acknowledgments

**TurboGEPA is built on the shoulders of giants.**

### GEPA: Core Algorithm

All algorithmic credit for the core GEPA framework goes to the original authors:

**Lakshya A Agrawal¹, Shangyin Tan¹, Dilara Soylu², Noah Ziems⁴, Rishi Khare¹, Krista Opsahl-Ong⁵, Arnav Singhvi²⁵, Herumb Shandilya², Michael J Ryan², Meng Jiang⁴, Christopher Potts², Koushik Sen¹, Alexandros G. Dimakis¹³, Ion Stoica¹, Dan Klein¹, Matei Zaharia¹⁵, Omar Khattab⁶**

¹UC Berkeley, ²Stanford University, ³BespokeLabs.ai, ⁴Notre Dame, ⁵Databricks, ⁶MIT

The **core innovation**—LLM-based reflective mutation with Pareto selection—is entirely from the original GEPA paper.

### Prompt-MII: Spec Induction

TurboGEPA's spec induction mutation operator is inspired by the Prompt-MII work from:

**Emily Xiao, Yixiao Zeng, Ada Chen, Chin-Jou Li, Amanda Bertsch, Graham Neubig**

Carnegie Mellon University Language Technologies Institute

### TurboGEPA: Performance Engineering

TurboGEPA's contributions are limited to **performance engineering**:

- Async/await orchestration
- Island-based parallelism
- ASHA successive halving
- Adaptive configuration

---

<p align="center">
  <strong>Original GEPA:</strong> Research innovation & algorithmic foundation<br>
  <strong>TurboGEPA:</strong> Production-ready performance engineering<br>
  <em>Better together. 🚀</em>
</p>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Studio-Intrinsic/turbo-gepa&type=date&legend=top-left)](https://www.star-history.com/#Studio-Intrinsic/turbo-gepa&type=date&legend=top-left)
Need a turnkey Modal deployment? See `examples/modal_turbo_aime.py` for a distributed benchmark that mounts a Modal Volume, forwards OpenRouter secrets, and orchestrates workers via `turbo-gepa-worker`.
