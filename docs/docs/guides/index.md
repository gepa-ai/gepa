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

GEPA (Genetic-Pareto) is a framework for optimizing text components of any system using:

1. **Evolutionary Search**: Iteratively proposes and evaluates candidate improvements
2. **LLM-based Reflection**: Uses language models to analyze failures and propose fixes
3. **Pareto Optimization**: Maintains a frontier of candidates that excel on different aspects

### When to Use GEPA

GEPA is ideal for:

- **Prompt Optimization**: Improve AI system prompts for specific tasks
- **Code Evolution**: Optimize code snippets or templates
- **Configuration Tuning**: Evolve textual configurations
- **Multi-Component Systems**: Optimize multiple interdependent text components

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

## Learn More

- **[FAQ](faq.md)** - Common questions answered
- **[Use Cases](use-cases.md)** - See real-world GEPA applications
- **[Tutorials](../tutorials/index.md)** - Step-by-step learning resources

## Community

- **Discord**: [Join the GEPA community](https://discord.gg/A7dABbtmFw)
- **Slack**: [GEPA Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)
- **Twitter/X**: Follow [@LakshyAAAgrawal](https://x.com/LakshyAAAgrawal) for updates
