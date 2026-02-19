# Frequently Asked Questions

Common questions about GEPA, answered by the community and the GEPA team.

---

## General Questions

### What exactly does GEPA output?

GEPA is a **text evolution engine** that can optimize any artifact representable as text — prompts, code, configs, agent architectures, policies, numeric parameters, etc.  The `optimize_anything` API is the primary interface: you write an evaluator that returns a score and **Actionable Side Information (ASI)**, and GEPA uses LLMs as intelligent proposers to iteratively refine the artifact.

`optimize_anything` supports three modes:

- **Single-Task Search**: Solve one hard problem (circle packing, math optimization)
- **Multi-Task Search**: Solve a batch of related problems with cross-transfer (CUDA kernels)
- **Generalization**: Build a skill that transfers to unseen data (prompt optimization, agent evolution)

This has been applied to:

- **Prompt optimization**: AIME 2025 math (+10%), enterprise agents (Databricks, 90x cheaper)
- **Code evolution**: circle packing (exceeding AlphaEvolve), CUDA kernels
- **Agent architecture discovery**: ARC-AGI agent programs, cloud scheduling policies
- **Parameter search**: rocket trajectory optimization, polynomial fitting
- **Visual optimization**: SVG generation with VLM feedback

### Does GEPA only work with DSPy?

No. The `optimize_anything` API works with **any system** — just write a Python function that scores candidates and provides diagnostic feedback (ASI).  No adapter or framework needed.

```python
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

def evaluator(candidate, example):
    result = my_system(candidate["prompt"], example["input"])
    score = compute_score(result, example["expected"])
    oa.log(f"Input: {example['input']}")
    oa.log(f"Output: {result}, Expected: {example['expected']}")
    return score

result = optimize_anything(
    seed_candidate={"prompt": "Initial prompt"},
    evaluator=evaluator,
    dataset=my_data,
    objective="Optimize prompts for accurate answers.",
    config=GEPAConfig(engine=EngineConfig(max_metric_calls=200)),
)
```

For DSPy programs, use `dspy.GEPA()` which wraps the same engine.  For advanced integration, see the [Adapters Guide](adapters.md).

### Does GEPA aim for brevity in prompts?

No, GEPA does not aim for brevity. It is a general optimization algorithm that optimizes text with any goal you specify. Typically the goal is to improve performance as much as possible, which often results in **detailed, context-rich prompts** rather than short ones.

If you compare GEPA's prompts to human-written prompts, they're often longer—and that's the point. No human should be manually fiddling with a 1000-2000 token prompt with "vibes" to optimize systems! Let a data-driven approach like GEPA handle that complexity.

That said, GEPA's prompts are still **up to 9x shorter** than those from leading few-shot optimizers, while being more effective.

!!! tip "Want shorter prompts?"
    If you need to optimize for both quality AND brevity, GEPA can do that! Provide a multi-objective metric that penalizes length, and GEPA will find prompts that balance both objectives.

### Can GEPA optimize for token efficiency (cost reduction)?

Yes! If you use GEPA for a multi-module DSPy program, you can improve token efficiency—achieving the same performance at fewer tokens, or better performance with cheaper models.

GEPA can take multiple metrics as input, so you can provide a multi-objective metric that balances quality and cost.

Research shows GEPA can help achieve **90x cheaper inference** while maintaining or improving performance (see Databricks case study).

---

## Configuration & Budget

### How do I control GEPA's runtime and budget?

Set `max_metric_calls` in `EngineConfig`, and/or pass additional stoppers via `GEPAConfig.stop_callbacks`.  GEPA stops when **any** stopper triggers.

```python
from gepa.optimize_anything import GEPAConfig, EngineConfig
from gepa.utils import TimeoutStopCondition, NoImprovementStopper

config = GEPAConfig(
    engine=EngineConfig(max_metric_calls=200),
    stop_callbacks=[
        TimeoutStopCondition(seconds=3600),
        NoImprovementStopper(patience=10),
    ],
)
```

Available stoppers: `MaxMetricCallsStopper`, `TimeoutStopCondition`, `NoImprovementStopper`, `ScoreThresholdStopper`, `SignalStopper`, `FileStopper`, `CompositeStopper`.

### What does a single GEPA iteration cost?

Each iteration involves:

1. **Minibatch evaluation**: Run the candidate on a minibatch of training examples (typically 2–5, set through `reflection_minibatch_size`) using the `task_lm`.
2. **Reflection**: 1 LLM call to the `reflection_lm` to analyze minibatch traces, diagnose failure modes and propose a new candidate.
3. **Minibatch validation**: Run the new candidate on same minibatch of training examples using the `task_lm`.
5. **Full validation** (if improved): If the new candidate improved on the minibatch, proceed to full validation.

We typically recommend calling GEPA with at least 15-30x of len(valset) to allow it to propose and evaluate upto 15 new candidates.

### What's the recommended train/validation split?

Use **80% train / 20% validation** when you have more than **200 total datapoints**. If you have fewer than 200 total datapoints, a **50/50 split** is usually better.

- **Validation set** should be small but truly representative of your task distribution
- **Training set** should contain as many examples as possible for GEPA to reflect on
- An improvement on the validation set should actually indicate improvement on your real task

```python
# Example split
trainset = examples[:80]  # 80% for training
valset = examples[80:]    # 20% for validation
```

### Can GEPA work with very few examples?

Yes! GEPA can show improvements with as few as **3 examples**. We've demonstrated +9% improvement on held-out data with just 3 examples in one GEPA iteration.

That said, more data generally leads to better optimization. Aim for **30-300 examples** for best results, using **80/20** when total examples exceed 200 and **50/50** when you have fewer than 200.

### How does GEPA differ from gradient-based RL (e.g., GRPO)?

GRPO and similar RL methods use random perturbations in LLM behavior over 25,000–100,000+ rollouts, collapsing all feedback — compiler errors, reasoning traces, constraint violations — into a single scalar reward. The optimizer never sees *why* a candidate failed, only *that* its score increased / decreased.

GEPA uses LLM self-reflection on full execution traces to generate directional updates in text space. The reflection LM reads the actual diagnostic output (ASI) and proposes targeted fixes: "The agent called `search_api()` which doesn't exist — rewrite the prompt to restrict tools to the provided schema."

Concrete comparison on HotPotQA: GEPA achieves **20% better performance** than GRPO in **35x fewer rollouts** (100–500 vs. 5,000+), **8x faster** wall-clock time (3 hours vs. 24 hours), at **15x lower cost** ($20 vs. $300).

**When to use which:** Use GEPA when rollouts are expensive or slow, data is scarce, or you need API-only optimization. For scenarios with abundant supervised data and capacity for 100,000+ cheap rollouts, gradient-based RL may outperform. The two are complementary — run GEPA first for rapid gains, then apply RL/SFT on top following the [BetterTogether](https://arxiv.org/abs/2407.10930) / [mmGRPO](https://arxiv.org/abs/2508.04660) recipe.

---

## Models & Performance

### What model should I use for `reflection_lm`?

We recommend using a **leading frontier model** for `reflection_lm`:

- **Preferred**: GPT-5.2, Gemini-3, Claude Opus 4.5
- **Minimum recommended tier**: post‑GPT‑5 or Gemini‑2.5‑Pro class models
- **Also works**: Models as small as Qwen3‑4B have been shown to work, but use the most capable model available for the reflection LM

!!! tip "Recommendation"
    Use a large `reflection_lm` for proposing improved prompts, but use the same LM for `task_lm` that you'll deploy in production.

### Do prompt optimizers work better for smaller models?

There's a common belief that prompt optimizers only help smaller models. **Counter evidence suggests otherwise**:

1. **OCR with Gemini 2.5 Pro**: 38% error rate reduction (already a large model)
2. **Databricks**: Open-source models optimized with GEPA outperform Claude Opus 4.1, Sonnet 4, and GPT-5

Prompt optimization helps models of **all sizes** achieve better cost-quality tradeoffs.

### Can GEPA work with multimodal/VLM tasks?

Yes! GEPA has been successfully used with multimodal LLMs for:

- OCR tasks (image → text)
- Image analysis pipelines
- Document understanding

For multimodal tasks, see the [Intrinsic Labs OCR report](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf).

---

## Multi-Component & Agent Optimization

### How do I optimize multi-module DSPy programs efficiently?

For programs with multiple signatures/modules, use `component_selector='all'`:

```python
optimizer = dspy.GEPA(
    metric=metric,
    component_selector='all'  # Update all signatures at once!
)
```

This provides a **large boost in rollout efficiency**. By default, GEPA updates just one signature per round; with `component_selector='all'`, it updates all signatures simultaneously.

### Can GEPA optimize entire agent architectures?

Yes! GEPA can evolve not just prompts but the **whole agent architecture**—including task decomposition, control flow, and module structure. 

Example: Starting from a simple `dspy.ChainOfThought('question -> answer')`, GEPA evolved a multi-step reasoning program, improving GPT-4.1 Nano's accuracy on MATH from **67% to 93%** in just 4 iterations.

See the [DSPy Full Program Evolution tutorial](../tutorials/dspy_full_program_evolution.ipynb).

### How do I optimize agents for fuzzy/creative tasks?

For tasks where evaluation is subjective (creative writing, persona generation, etc.), use the **Evaluator-Optimizer pattern**:

1. Create an LLM-based evaluator (even without ground truth labels)
2. Let the evaluator provide detailed feedback
3. Use GEPA to optimize against the evaluator's scores and feedback

You can also **tune the LLM-based evaluator itself with GEPA** using a small human‑annotated dataset to calibrate its judgments.

```python
def subjective_metric(example, pred, trace=None):
    # Use LLM-as-judge for evaluation
    evaluation = judge_lm(
        f"Rate this response: {pred.output}\nCriteria: {criteria}"
    )
    return dspy.Prediction(score=evaluation.score, feedback=evaluation.feedback)

optimizer = dspy.GEPA(metric=subjective_metric, ...)
```

See the [Papillon tutorial](https://dspy.ai/tutorials/gepa_papillon/) for a complete example.

---

## Debugging & Monitoring

### How can I see all the prompts GEPA proposes?

Several options:

1. **Console output**: GEPA prints all proposed prompts during optimization
2. **Experiment tracking**: Enable MLflow or Weights & Biases
   ```python
   result = gepa.optimize(..., use_wandb=True)
   ```
3. **Programmatic access**: After optimization, access `detailed_results`
   ```python
   optimized_program.detailed_results  # All proposed prompts with scores
   ```
4. **Enable detailed stats**: Pass `track_stats=True` to see all proposal details
5. **Callbacks**: Use the GEPA callback system to log, inspect, or persist proposals (see the [Callbacks Guide](callbacks.md))

### Can I continue optimization from a previous run?

Yes! Set `run_dir` in `EngineConfig` — GEPA saves state to disk and automatically resumes:

```python
config = GEPAConfig(engine=EngineConfig(
    run_dir="./runs/my_exp",      # Will resume if gepa_state.bin exists
    max_metric_calls=500,
))
result = optimize_anything(..., config=config)
```


### Does GEPA support async optimization?

GEPA's implementation serializes the agent trajectory to reflect on it, so async workflows should generally work. If you're running agentic systems with async operations, you'll want to ensure your trajectory data is properly captured before GEPA's reflection step.

If you encounter issues with async optimization, please share your experience with the community—we're actively iterating to improve support for complex async workflows.

### How should I serialize agent traces for GEPA?

GEPA assumes a simple interface for providing agent trajectories. You don't need to modify GEPA internals—just serialize your agent traces in a format GEPA can process. The key requirements:

1. **Capture rich trajectory information**: Include all relevant state changes, decisions, and outputs
2. **Provide enough context**: The reflection LM needs to understand what happened to propose improvements
3. **Include failure modes**: Errors and edge cases are especially valuable for optimization

For agentic systems with expensive rollouts (simulations, long runtime), this trajectory serialization is critical for GEPA's sample efficiency.

### Why are early GEPA prompts so long or contain training example content?

The initial rounds of GEPA tend to include a lot of information from the first examples it sees—sometimes even specific content from training examples. This is normal behavior. However, **as optimization progresses, GEPA creates generalized rules** and the prompts become more concise while remaining effective.

This is by design—GEPA first captures specific patterns, then abstracts them into general principles. If you want to prevent verbatim example inclusion, use custom instruction proposers with explicit constraints.

---

## Production Deployment

### What's the recommended deployment pattern for GEPA?

An emerging pattern for GEPA+DSPy deployment:

1. **Init & Deploy**
   - Use a small, high-quality initial dataset (e.g., labeled examples with explanations)
   - Run GEPA + DSPy to optimize the program/agent
   - Deploy the optimized system

2. **Monitor**
   - Collect user feedback from the deployed system
   - Track performance metrics in production

3. **Iterate**
   - Batch new feedback into training data
   - Re-run GEPA optimization
   - Deploy updated system

This creates a **continuous improvement loop** without requiring constant human annotation.

### Can GEPA help with model migration?

Yes! GEPA is very useful for migrating existing LLM-based workflows and agents to new models across model families. When you switch models:

1. Keep your DSPy program structure
2. Change only the LM initialization
3. Re-run GEPA optimization for the new model

This is much faster than manually re-tuning prompts for each new model.

### What about production costs?

GEPA vastly improves **token economics**:

- Databricks achieved **90x cost reduction** while maintaining or improving performance
- Open-source models optimized with GEPA can outperform expensive frontier models
- At 100,000 requests, serving costs represent 95%+ of AI expenditure—GEPA makes this sustainable

---

## Advanced Topics

### Can GEPA's meta-prompt itself be optimized?

Yes! GEPA uses a default reflection prompt that guides how the LLM proposes improvements. You can:

1. **Customize the reflection prompt** for domain-specific optimization:
   ```python
   dspy.GEPA(
       metric=metric,
       instruction_proposer=CustomProposer(...)  # Your custom logic
   )
   ```

2. **Add constraints** like "avoid including specific values from feedback" or "generated prompts should be no more than 5000 characters"

3. **Provide RAG-style retrieval** from domain-specific guides/textbooks

See the [Advanced GEPA documentation](https://dspy.ai/api/optimizers/GEPA/GEPA_Advanced/#custom-instruction-proposers) for details.

### What's the relationship between GEPA and finetuning?

GEPA and finetuning are complementary:

- **GEPA**: Optimizes prompts/instructions (no weight changes, cheaper, faster)
- **Finetuning**: Updates model weights (more permanent, requires more data)

Research shows **GEPA+Finetuning** together works great. For example:
- The BetterTogether paper combines RL weight updates + prompt optimization
- GEPA-optimized prompts can guide finetuning data generation

See: [BetterTogether paper](https://arxiv.org/abs/2508.04660) and [GEPA for AI Code Safety](https://www.lesswrong.com/posts/bALBxf3yGGx4bvvem/prompt-optimization-can-enable-ai-control-research)

---

## Understanding GEPA Output

### What do "valset pareto frontier" vs "valset score" mean?

- **Valset Pareto Frontier Score**: Performance obtained by selecting the best prompt for every task individually
- **Valset Score**: Performance achieved by a single best prompt across all validation tasks

A large gap between these values indicates that GEPA has found diverse strategies for different tasks but hasn't yet merged them into a single unified prompt. Running GEPA longer (with merges) typically closes this gap.

!!! tip "Inference-Time Search"
    For inference-time search applications, you might only care about the valset pareto frontier score—i.e., the best possible performance across all tasks.

---

## Working with Limited Data

### How does the seed prompt affect optimization?

Seeding with a good initial prompt leads to better search! However, be careful not to over-tune the seed:

**Good seed prompt**: Establishes all ground rules and constraints of the task—information necessary for a smart human to complete the task.

**Avoid in seed**: Process details that the model should figure out on its own.

Think of it as: What would you tell a smart human to get them started vs. what should they discover through practice?

### Can GEPA work for program synthesis / code evolution tasks?

Yes!  This is `optimize_anything`'s **single-task search** mode — the candidate *is* the code, and the evaluator executes it.  Examples include:

- **Circle packing**: Evolving algorithms that exceed AlphaEvolve results
- **CUDA kernels**: Optimizing GPU code via multi-task search mode
- **Agent architecture discovery**: Evolving entire agent code (ARC-AGI)
- **Rocket trajectory**: Optimizing launch parameters via numeric search
- **Cloud scheduling policies**: Evolving cost-minimizing code policies

Unlike AlphaEvolve/OpenEvolve/ShinkaEvolve (which only support single-task search), `optimize_anything` also supports multi-task search and generalization modes.

---

## Framework Integration

### What's the difference between gepa-ai/gepa and DSPy's GEPA?

They are the **same implementation**. DSPy uses `gepa-ai/gepa` as a dependency.  For DSPy programs, use `dspy.GEPA()`.  For everything else (custom evaluators, code evolution, agent optimization), use `optimize_anything` from `gepa-ai/gepa` directly.

### Does GEPA have any external dependencies?

GEPA has **zero hard dependencies**. LiteLLM is an optional dependency to use the default adapter. You can define an adapter for any other framework (Pydantic, LangChain, etc.) very easily using the `GEPAAdapter` interface.

### Can I use GEPA with Pydantic / other frameworks?

Yes! GEPA's `GEPAAdapter` interface allows integration with any framework without implementing from scratch:

- **DSPy**: Built-in adapter, recommended approach
- **Pydantic**: Custom adapter possible
- **OpenAI SDK**: Via DefaultAdapter
- **LangChain**: Via custom adapter
- **Opik (Comet)**: Official integration available
- **Google ADK**: Community tutorials available

See the [Adapters Guide](adapters.md) for implementation examples.

---

## Tips from the Community

### How much Actionable Side Information (ASI) should I include?

As much as possible.  ASI — the `side_info` dict returned by your evaluator, or output captured via `oa.log()` — is what the reflection LLM reads to diagnose failures and propose targeted improvements.  Traditional optimizers know *that* a candidate failed but not *why*; ASI provides the *why*.  Include:

- **What went wrong** — error messages, wrong outputs, constraint violations
- **Expected vs actual** — ground truth comparisons
- **Hints** — domain knowledge, reference solutions
- **Multi-dimensional scores** — via `{"scores": {"accuracy": 0.7, "speed": 3.2}}`

!!! warning "Score-Only Mode"
    Returning just a `float` score (without side_info) significantly limits optimization quality.
    The reflection LLM has no context to work with.  Always prefer returning `(score, side_info)`
    or use `oa.log()` to provide diagnostic output.

### Should I augment training examples with explanations?

Yes! Augmenting training examples with detailed explanations of why a particular label/answer is correct significantly helps GEPA:

```python
# Basic example
dspy.Example(question="Is this email urgent?", answer="Yes").with_inputs("question")

# Augmented example (recommended)
dspy.Example(
    question="Is this email urgent?", 
    answer="Yes",
    explanation="The email mentions a deadline of 'end of day today' and uses words like 'critical' and 'ASAP', indicating urgency."
).with_inputs("question")
```

This helps the reflection LLM understand the reasoning behind classifications.

---

## Common Gotchas & Tips

### Why is GEPA overfitting to my training data?

If you're seeing GEPA overfit, make sure you provide a **separate validation set**:

```python
optimizer = dspy.GEPA(metric=metric, ...)
optimized = optimizer.compile(program, trainset=train_data, valset=val_data)
```

Without a separate valset, GEPA will tend to overfit the training data. Follow the standard 80/20 train/val split.

### How do I use GEPA for agentic systems with expensive rollouts?

For tasks with costly rollouts (simulation, long runtime, complex agents), GEPA's sample efficiency is especially valuable:

1. **Batch feedback**: Collect production feedback and batch optimize periodically
2. **Sub-agent optimization**: If you have data to optimize sub-agents, that often performs better than optimizing the whole system
3. **Trajectory serialization**: Ensure you capture rich trajectory information for reflection

### Can GEPA co-evolve multiple components (adapter logic, tools, prompts)?

Yes! If your component is well-defined and the reward is well-defined, GEPA can optimize it. This includes:

- Tool definitions and schemas
- Agent routing logic
- Multi-component systems
- Entire agent architectures and control flow
- Adapter logic (how your system processes inputs, handles errors, or routes between sub-agents)

### How do I use multi-objective Pareto tracking?

Return a `"scores"` dict inside `side_info` — GEPA uses these for Pareto-optimal candidate selection across multiple objectives:

```python
def evaluator(candidate, example):
    pred = run_system(candidate["prompt"], example)
    accuracy = calculate_accuracy(pred, example)
    latency = calculate_latency(pred)

    score = accuracy  # Primary score (higher is better)
    side_info = {
        "scores": {
            "accuracy": accuracy,
            "speed": 1.0 / (1.0 + latency),  # Inverted: higher = better
        },
        "Input": example["input"],
        "Output": pred,
    }
    return score, side_info
```

All values in `"scores"` must follow **higher is better**.  GEPA maintains a Pareto frontier across these objectives, finding candidates that represent different trade-offs.

---

## Still have questions?

- **Discord**: [Join our community](https://discord.gg/WXFSeVGdbW)
- **Slack**: [GEPA Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)
- **GitHub Issues**: [gepa-ai/gepa](https://github.com/gepa-ai/gepa/issues)
- **Twitter/X**: Follow [@LakshyAAAgrawal](https://x.com/LakshyAAAgrawal) for updates and to ask questions
