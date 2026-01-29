# KernelBench Optimization with GEPA

This experiment uses GEPA (Generative Evolutionary Programming Algorithm) to optimize CUDA kernels for the KernelBench dataset.

## Structure

- `main.py`: Entry point for GEPA optimization.
- `eval.py`: Evaluation utilities for executing kernels and computing scores.
- `prompts.py`: DSPy signatures and initial prompts.
- `agentic_rag.py`: RAG implementation for retrieving CUDA documentation.
- `KernelBench/`: Git submodule of the [KernelBench](https://github.com/ScalingIntelligence/KernelBench) repository (dataset and reference implementations).

## Setup

1. Ensure you have `uv` installed.
2. Initialize and update the submodule:
   ```bash
  git submodule update --init external/gepa_blog/examples/kernelbench/KernelBench
   ```
3. Install dependencies:
   ```bash
   uv pip install torch --index-url https://download.pytorch.org/whl/cu124
   uv pip install ninja fasteners dspy-ai llama-index-core llama-index-embeddings-huggingface llama-index-llms-openai
   ```

## Running Optimization

To run the optimization:

```bash
uv run python -m examples.kernelbench.main --levels level1 --max-metric-calls 100
```

Arguments:
- `--levels`: Comma-separated levels to optimize (e.g., `level1,level2`).
- `--max-metric-calls`: Maximum number of kernel evaluations.
- `--no-rag`: Disable RAG-augmented generation.
- `--max-refinements`: Maximum number of refinement attempts per problem.

## Evaluation Logic

The evaluation logic is isolated in subprocesses to prevent CUDA crashes from affecting the main process. It uses `kernelbench.eval.eval_kernel_against_ref` from the local `KernelBench` submodule.

Scores are computed based on:
1. Compilation success.
2. Correctness (output matching reference).
3. Performance (speedup relative to PyTorch baseline).
