# Tutorials

Welcome to the GEPA tutorials! These hands-on notebooks will help you learn GEPA through practical examples.

## Available Tutorials

### DSPy Full Program Evolution

Learn how to use GEPA to evolve entire DSPy programs, including custom signatures, modules, and control flow logic.

- **[DSPy Full Program Evolution](dspy_full_program_evolution.ipynb)** - Evolve a complete DSPy program from a basic `ChainOfThought` to a sophisticated multi-step reasoning system. This tutorial demonstrates how GEPA can improve a program from 67% to 93% accuracy on the MATH benchmark.

### ARC AGI Example

- **[ARC AGI Example](arc_agi.ipynb)** - Apply GEPA to the ARC (Abstraction and Reasoning Corpus) challenge, demonstrating how to optimize programs for complex reasoning tasks.

## External Tutorials

For more tutorials, especially those focused on the DSPy integration, see:

- [dspy.GEPA Tutorials](https://dspy.ai/tutorials/gepa_ai_program/) - Official DSPy tutorials with executable notebooks
- [GEPA for AIME (Math)](https://dspy.ai/tutorials/gepa_aime/) - Optimize prompts for competition math
- [GEPA for Structured Information Extraction](https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/) - Enterprise task optimization
- [GEPA for Privacy-Conscious Delegation](https://dspy.ai/tutorials/gepa_papillon/) - Papillon benchmark
- [GEPA for Code Backdoor Classification](https://dspy.ai/tutorials/gepa_trusted_monitor/) - AI control applications

## Video Tutorials

- [Video tutorial by @weaviate on using dspy.GEPA](https://www.youtube.com/watch?v=H4o7h6ZbA4o) - Optimize a listwise reranker
- [Matei Zaharia - Reflective Optimization of Agents](https://www.youtube.com/watch?v=rrtxyZ4Vnv8) - High-level overview

## Running Tutorials Locally

To run these tutorials locally:

```bash
# Install GEPA with full dependencies
pip install gepa[full]

# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook
```

Then navigate to the tutorial notebook you want to run.

## Prerequisites

Before starting the tutorials, ensure you have:

1. **API Keys**: Most tutorials require an OpenAI API key (or other LLM provider)
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Python Environment**: Python 3.10+ with GEPA installed
   ```bash
   pip install gepa[full]
   ```

3. **Optional**: Install DSPy for the DSPy-specific tutorials
   ```bash
   pip install dspy
   ```
