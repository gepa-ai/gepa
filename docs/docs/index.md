# GEPA: System Optimization through Reflective Text Evolution

<p align="center">
  <img src="https://raw.githubusercontent.com/gepa-ai/gepa/refs/heads/main/assets/gepa_logo_with_text.svg" alt="GEPA Logo" width="400">
</p>

**GEPA** (Genetic-Pareto) is a framework for **optimizing arbitrary systems composed of text components**—like AI prompts, code snippets, or textual specs—against any evaluation metric.

[![PyPI - Version](https://img.shields.io/pypi/v/gepa)](https://pypi.org/project/gepa/)
[![PyPI Downloads](https://static.pepy.tech/badge/gepa)](https://pepy.tech/projects/gepa)

## Overview

GEPA employs LLMs to reflect on system behavior, using feedback from execution and evaluation traces to drive targeted improvements. Through iterative mutation, reflection, and Pareto-aware candidate selection, GEPA evolves robust, high-performing variants with minimal evaluations, co-evolving multiple components in modular systems for domain-specific gains.

This repository provides the official implementation of the GEPA algorithm as proposed in the paper titled "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" ([arXiv:2507.19457](https://arxiv.org/abs/2507.19457)).

## Installation

```bash
pip install gepa
```

To install the very latest from `main`:

```bash
pip install git+https://github.com/gepa-ai/gepa.git
```

## Quick Start

### The Easiest Path: DSPy Integration

!!! tip "Recommended"
    The easiest and most powerful way to use GEPA for prompt optimization is within [DSPy](https://dspy.ai/), where the GEPA algorithm is directly available through the `dspy.GEPA` API. See [dspy.GEPA Tutorials](https://dspy.ai/tutorials/gepa_ai_program/).

### Simple Prompt Optimization Example

GEPA can be run in just a few lines of code. Here's an example optimizing a system prompt for math problems:

```python
import gepa

# Load AIME dataset
trainset, valset, _ = gepa.examples.aime.init_dataset()

seed_prompt = {
    "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
}

# Run GEPA optimization
gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",  # Model being optimized
    max_metric_calls=150,            # Budget
    reflection_lm="openai/gpt-5",    # Strong model for reflection
)

print("Optimized Prompt:", gepa_result.best_candidate['system_prompt'])
```

This achieves **+10% improvement** (46.6% → 56.6%) on AIME 2025 with GPT-4.1 Mini!

## Key Concepts

### Adapter Architecture

GEPA is built around a flexible [`GEPAAdapter`](api/core/GEPAAdapter.md) abstraction that lets it plug into any system:

- **System**: A harness that uses text components to perform a task
- **Candidate**: A mapping from component names to component text
- **Trajectory**: Execution traces captured during evaluation

### Built-in Adapters

| Adapter | Description |
|---------|-------------|
| [`DefaultAdapter`](api/adapters/DefaultAdapter.md) | Single-turn LLM with system prompt optimization |
| [`RAGAdapter`](api/adapters/RAGAdapter.md) | RAG systems with any vector store |
| [`MCPAdapter`](api/adapters/MCPAdapter.md) | Model Context Protocol tool optimization |

## How GEPA Works

GEPA optimizes text components using an evolutionary search algorithm with LLM-based reflection for mutations. Key features:

1. **Reflective Mutation**: Uses task-specific textual feedback (compiler errors, profiler reports, etc.) to guide improvements
2. **Pareto Frontier Tracking**: Maintains candidates that excel on different subsets of the validation data
3. **Multi-Component Optimization**: Co-evolves multiple text components simultaneously

For details, see the paper: [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)

## Further Resources

- **Paper**: [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
- **Tutorials**: [dspy.GEPA Tutorials](https://dspy.ai/tutorials/gepa_ai_program/)
- **Discord**: [Join our community](https://discord.gg/A7dABbtmFw)
- **Slack**: [GEPA Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)

## Citation

```bibtex
@misc{agrawal2025gepareflectivepromptevolution,
      title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning}, 
      author={Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab},
      year={2025},
      eprint={2507.19457},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.19457}, 
}
```
