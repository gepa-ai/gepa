# Guides

Welcome to the GEPA guides! These guides will help you understand and use GEPA effectively.

## Getting Started

- [Quick Start](quickstart.md) - Get up and running with GEPA in minutes
- [Use Cases](use-cases.md) - Real-world applications of GEPA across industries
- [FAQ](faq.md) - Frequently asked questions about GEPA
- [Creating Adapters](adapters.md) - Learn how to integrate GEPA with your system
- [Using Callbacks](callbacks.md) - Monitor and instrument optimization runs
- [Contributing](contributing.md) - How to contribute to GEPA development

## Key Concepts

### What is GEPA?

GEPA (Genetic-Pareto) is a text evolution engine that optimizes any artifact representable as text — prompts, code, agent architectures, configurations, policies — using LLM-based reflection and Pareto-efficient search.

### Why Reflection Instead of Gradients?

Traditional optimization methods — reinforcement learning (GRPO), evolutionary strategies, Bayesian optimization — operate on purely numerical feedback. They know *that* a candidate failed, but not *why*. A compiler error that explains exactly which line crashed, an agent reasoning trace that reveals a flawed decomposition strategy, a constraint violation that pinpoints a geometric overlap — all of this diagnostic context is collapsed into a single scalar reward and discarded.

GEPA takes a fundamentally different approach: it uses LLMs to **reason in natural language** over full execution traces. Rather than making random perturbations and keeping whatever scores higher, GEPA analyzes *why* candidates succeeded or failed and proposes *targeted* improvements. This is the difference between blind search and directed search — and it is why GEPA can achieve 20% better performance than GRPO in 35x fewer rollouts on HotPotQA (3 hours vs. 24 hours, $20 vs. $300).

The mechanism that makes this work is **Actionable Side Information (ASI)** — diagnostic, domain-specific textual feedback that the LLM proposer reads during a dedicated reflection step. ASI can be anything that would help a human expert diagnose the problem: error messages, profiling traces, rendered images (via VLMs), constraint violations, agent reasoning logs. Where gradients tell a numerical optimizer which direction to move, ASI tells an LLM proposer *why* a candidate failed and *how* to fix it.

### The Three-Stage Pipeline

Each GEPA iteration runs a three-stage pipeline:

```
┌────────────┐      ┌────────────┐      ┌────────────┐
│  Executor  │ ───▶ │  Reflector │ ───▶ │   Curator  │
│            │      │            │      │            │
│ Run candi- │      │ Analyze    │      │ Generate   │
│ date on    │      │ traces to  │      │ improved   │
│ tasks,     │      │ diagnose   │      │ candidate  │
│ capture    │      │ failure    │      │ from diag- │
│ full       │      │ modes and  │      │ nostic     │
│ traces     │      │ causal     │      │ insights   │
│            │      │ patterns   │      │            │
└────────────┘      └────────────┘      └────────────┘
```

1. **Executor**: Runs the candidate on a minibatch of evaluation tasks using the task model, capturing complete execution traces — reasoning chains, intermediate outputs, error messages, performance metrics, and any ASI returned by the evaluator.

2. **Reflector**: A strong LLM (the `reflection_lm`) analyzes the collected traces to identify failure modes, logic breakdowns, and causal relationships. For example: "The agent used inductive reasoning on problem 3, but proof by contradiction would be more effective given the problem structure." The reflector sees both successes and failures to build a complete picture.

3. **Curator**: Based on the reflector's diagnosis, generates an improved candidate with concrete modifications to the text, code, agent logic, or system architecture. Each new candidate inherits accumulated lessons from all of its ancestors in the search tree.

### Dual-Strategy Candidate Generation

GEPA employs two complementary strategies, selected adaptively:

**Reflective Mutation**: Sample one candidate from the Pareto frontier, execute it on a minibatch, reflect on the traces, and propose an improved version. This enables focused, directed improvement of promising candidates.

**System-Aware Merge**: Sample two candidates from the frontier and strategically combine modules from each based on their evolution history. If a module was refined in candidate A but left unchanged in candidate B, the merge selects from A for that module. This creates hybrid candidates that combine complementary strengths — for example, merging a candidate that excels on algebraic problems with one that handles geometry well.

Strategy selection and candidate sampling use novelty-weighted approaches to balance exploration (trying diverse strategies) and exploitation (refining the best candidates).

### When to Use GEPA

GEPA is particularly effective in scenarios where traditional methods face constraints:

**Expensive or slow rollouts** — Scientific simulations requiring hours per evaluation, new hardware with slow compilation times (minutes per kernel), complex agents with long-running tool calls, multi-step verification processes. GEPA's sample efficiency (100–500 evaluations vs. 10,000+ for RL) matters most when each evaluation is costly.

**Data scarcity** — Brand-new hardware platforms with zero training data (e.g., AMD NPU kernels), novel task domains without existing examples, rapid prototyping with limited evaluation budget. GEPA has shown improvements with as few as 3 examples.

**API-only model access** — Frontier models (GPT-5, Claude, Gemini) that cannot be fine-tuned. GEPA requires only API access — no model weights, no custom infrastructure. This lets you optimize the most capable models available.

**Interpretability requirements** — Understanding *why* systems fail, not just *that* they failed. GEPA produces human-readable reflections and improvement rationales at every step. You can read the reflection log to understand exactly why a prompt changed, what failure mode it addresses, and what the expected improvement is.

**Complementary to RL/fine-tuning** — GEPA and gradient-based methods are not mutually exclusive. Organizations can use GEPA for rapid initial optimization (minutes to hours), then apply supervised fine-tuning or RL for additional gains. For scenarios with abundant supervised data and capacity for 100,000+ cheap rollouts, gradient-based methods remain highly effective.

### Core Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        GEPA Engine                         │
├────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐     │
│  │   Adapter   │  │  Proposer   │  │  Pareto Tracker │     │
│  │  (evaluate) │  │  (mutation) │  │   (selection)   │     │
│  └─────────────┘  └─────────────┘  └─────────────────┘     │
├────────────────────────────────────────────────────────────┤
│                        Your System                         │
│  ┌────────────┬────────────┬────────────┬─────────────┐    │
│  │Component 1 │Component 2 │Component N │    ...      │    │
│  │  (prompt)  │   (code)   │  (config)  │             │    │
│  └────────────┴────────────┴────────────┴─────────────┘    │
└────────────────────────────────────────────────────────────┘
```

- **Adapter**: Bridges GEPA to your system. Runs candidates against evaluation tasks, captures traces, and returns scores + ASI. Built-in adapters exist for DSPy, standalone LLM calls, and the `optimize_anything` API.
- **Proposer**: Implements the three-stage pipeline (Executor → Reflector → Curator) and the dual mutation/merge strategies.
- **Pareto Tracker**: Maintains a frontier of candidates that excel on different subsets of tasks or objectives. Candidates are not ranked by a single score — a candidate that solves problem A but not B is preserved alongside one that solves B but not A.

## Integration Options

### 1. DSPy Integration (Recommended)

The easiest way to use GEPA is through [DSPy](https://dspy.ai/):

```python
import dspy

# Configure your LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Use GEPA optimizer
optimizer = dspy.GEPA(...)
optimized_program = optimizer.compile(your_program, trainset=trainset)
```

### 2. Standalone GEPA

For custom systems, use the `gepa.optimize()` function with a custom adapter:

```python
import gepa

result = gepa.optimize(
    seed_candidate={"component": "initial text"},
    trainset=your_training_data,
    adapter=YourCustomAdapter(),
    reflection_lm="openai/gpt-4",
    max_metric_calls=100,
)
```

### 3. optimize_anything API

For the most flexible interface — optimizing code, agent architectures, policies, or any text artifact. `optimize_anything` is a universal, declarative API: you declare **what** to optimize and **how** to measure it; the system handles the search.

It supports three optimization modes under one interface:

- **Single-Task Search** — solve one hard problem (e.g., circle packing, blackbox optimization)
- **Multi-Task Search** — solve a batch of related problems with cross-task transfer (e.g., CUDA kernels)
- **Generalization** — build a skill that transfers to unseen problems (e.g., prompt optimization, agent architecture discovery)

```python
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

def evaluator(candidate, example):
    result = run_my_system(candidate, example)
    oa.log(f"Output: {result.output}")  # captured as Actionable Side Information (ASI)
    oa.log(f"Error: {result.error}")
    return result.score

# Start from an existing artifact…
result = oa.optimize_anything(
    seed_candidate="<your artifact>",
    evaluator=evaluator,
    dataset=my_data,
    objective="Optimize for accuracy and speed.",
    config=GEPAConfig(engine=EngineConfig(max_metric_calls=200)),
)

# … or just describe what you need (seedless mode).
result = oa.optimize_anything(
    evaluator=evaluator,
    objective="Generate a Python function that reverses a string.",
)
```

Key concepts:

- **Actionable Side Information (ASI)**: Diagnostic feedback returned by the evaluator — error messages, profiling traces, rendered images (via `gepa.Image`). ASI is the text-optimization analogue of the gradient: it tells the LLM proposer *why* a candidate failed and *how* to fix it.
- **Pareto-efficient search**: Scores are tracked per-task and per-metric individually. Any candidate that is the best at *something* survives on the frontier, enabling focused improvements that are preserved rather than averaged away.
- **`capture_stdio=True`**: Automatically route existing `print()` output inside your evaluator into ASI with no code changes needed.

See the [optimize_anything blog post](../blog/posts/2026-02-18-introducing-optimize-anything/index.md) for detailed examples across seven domains (circle packing, CUDA kernels, cloud scheduling, AIME math, ARC-AGI agent evolution, blackbox optimization, and coding agent skills).

## Learn More

- **[FAQ](faq.md)** - Common questions answered
- **[Use Cases](use-cases.md)** - See real-world GEPA applications
- **[Tutorials](../tutorials/index.md)** - Step-by-step learning resources

## Community

- **Discord**: [Join the GEPA community](https://discord.gg/WXFSeVGdbW)
- **Slack**: [GEPA Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)
- **Twitter/X**: Follow [@LakshyAAAgrawal](https://x.com/LakshyAAAgrawal) for updates
