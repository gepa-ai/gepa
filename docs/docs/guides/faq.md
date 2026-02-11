# Frequently Asked Questions

Common questions about GEPA, answered by the community and the GEPA team.

---

## General Questions

### What exactly does GEPA output?

GEPA is fundamentally a **text evolution engine**: given a target metric, GEPA can evolve any text component. Since prompts are just text, GEPA works great as a prompt optimizer (including multi-agent systems with multiple text components).

This broader capability has been applied to real systems, for example:

- **Enterprise agents**: Databricks reports 90x cheaper inference while maintaining or improving performance
- **OCR & document understanding**: Intrinsic Labs reduced OCR error rates across Gemini model classes
- **Agent frameworks**: Google ADK agents optimized with GEPA
- **Code synthesis & kernels**: OpenACC/CUDA-style code optimization for GPU parallelization

### Does GEPA only work with DSPy?

No. GEPA just requires visibility into the system it is executing—the same kind of information a human expert would need to improve the system. While DSPy is a recommended integration for prompt optimization, GEPA can work with **any framework** through its `GEPAAdapter` interface.

GEPA has already been integrated into frameworks including **verifiers**, **Comet-ML/Opik**, **Pydantic**, **LangStruct**, **Google ADK**, and DSPy, and it also ships adapters for MCP, RAG systems, and terminal agents.

```python
# Example: Using GEPA outside DSPy
import gepa

result = gepa.optimize(
    seed_candidate={"system_prompt": "Your initial prompt"},
    trainset=your_data,
    adapter=YourCustomAdapter(),  # Implement GEPAAdapter interface
    task_lm="openai/gpt-4o-mini",
    reflection_lm="openai/gpt-4o",
)
```

See the [Adapters Guide](adapters.md) for examples including DSPy, Opik, MCP, and Terminal agents.

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

Use the `StopperProtocol` and its implementations to define runtime and budget limits. You can pass one or more stoppers to `gepa.optimize()` via `stop_callbacks`, and GEPA will stop when any stopper triggers (or use a `CompositeStopper` for combined logic).

Available stoppers include:

- `MaxMetricCallsStopper`
- `TimeoutStopCondition`
- `NoImprovementStopper`
- `ScoreThresholdStopper`
- `SignalStopper`
- `FileStopper`
- `CompositeStopper`

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

Yes! GEPA supports continuing from previous runs:

```python
# Resume from saved state
result = gepa.optimize(
    ...,
    run_dir="./gepa_runs/my_exp",  # Will resume if state exists
)
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

### Can GEPA work for program synthesis tasks?

Yes! GEPA has been explored for:

- **Code kernel optimization**: CUDA and NPU kernels (Section 6 of the paper)
- **Agent architecture discovery**: Evolving entire agent code and structure
- **Material science applications**: Users have explored using GEPA for expensive simulation tasks

For any task with costly rollouts (simulation, long runtime), GEPA's sample efficiency makes it especially valuable.

---

## Framework Integration

### What's the difference between gepa-ai/gepa and DSPy's GEPA?

They are the **same implementation**. DSPy uses `gepa-ai/gepa` as a dependency. When possible, use GEPA through DSPy. For other frameworks or raw LLM calls, use `GEPAAdapter` from `gepa-ai/gepa` directly.

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

### How much detail should I include in feedback?

Provide as much detail as possible:

- What went wrong
- What can be improved
- What an ideal solution would have
- Score/grade breakdown across different dimensions
- Reference solutions (if available)

Don't hold anything back—the more context GEPA has, the better it can propose improvements.

!!! warning "Scores-Only vs Scores+Feedback"
    While GEPA accepts both score-only and score+feedback formats, **using score-only can significantly hurt optimization quality**. When you return just a numeric score without textual feedback, the reflection LM has very little information to work with.
    
    For some tasks, this means the optimization-necessary information never reaches the LLM, causing suboptimal results. Always prefer returning both a score AND detailed feedback text explaining why that score was given.

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

GEPA supports multi-objective optimization with Pareto tracking by reading `EvaluationBatch.objective_scores` from your `GEPAAdapter.evaluate()` implementation:

```python
class MyAdapter(GEPAAdapter):
    def evaluate(self, batch, candidate, capture_traces=False):
        outputs, scores, objective_scores, trajectories = [], [], [], []
        for example in batch:
            pred, trace = run_system(candidate, example)
            accuracy = calculate_accuracy(pred, example)
            efficiency = calculate_efficiency(trace)
            safety_score = calculate_safety(pred)
            outputs.append(pred)
            scores.append(accuracy)  # Primary score
            if capture_traces:
                trajectories.append(trace)
            objective_scores.append(
                {
                    "accuracy": accuracy,
                    "efficiency": efficiency,
                    "safety": safety_score,
                }
            )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=objective_scores,
        )
```

This allows GEPA to maintain a Pareto frontier across multiple objectives, finding prompts that represent different trade-offs between competing goals.

---

## Still have questions?

- **Discord**: [Join our community](https://discord.gg/A7dABbtmFw)
- **Slack**: [GEPA Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)
- **GitHub Issues**: [gepa-ai/gepa](https://github.com/gepa-ai/gepa/issues)
- **Twitter/X**: Follow [@LakshyAAAgrawal](https://x.com/LakshyAAAgrawal) for updates and to ask questions
