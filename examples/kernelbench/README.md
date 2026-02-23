# KernelBench Optimization with GEPA

Evolves LLM prompts that produce fast, correct CUDA kernels outperforming PyTorch baselines, using GEPA's seedless mode.

## Structure

- `main.py`: Entry point — defines the evaluator and calls `optimize_anything`.
- `utils/`
  - `eval.py`: Kernel execution, scoring, baseline measurement, dataset loading.
  - `agentic_rag.py`: Targeted CUDA doc retrieval based on eval outcomes.
  - `background.py`: Domain background passed to GEPA.
  - `KernelBench/`: Cloned [KernelBench](https://github.com/ScalingIntelligence/KernelBench) dataset.
  - `rag_content/`: Local CUDA documentation for RAG.
  - `tests/`: Test suite.

## Setup

1. Clone the KernelBench dataset:
   ```bash
   cd examples/kernelbench/utils
   git clone https://github.com/ScalingIntelligence/KernelBench.git KernelBench
   ```
2. Install dependencies:
   ```bash
   uv pip install torch --index-url https://download.pytorch.org/whl/cu124
   uv pip install ninja dspy
   ```

## Running

```bash
uv run python -m examples.kernelbench.main
```

No CLI arguments — configuration is set directly in `main.py` (3000 metric calls, all 3 levels).

## Tests

```bash
uv run python -m examples.kernelbench.utils.tests.test_setup
```

## How It Works

1. GEPA evolves a generation prompt (seedless mode — no seed candidate).
2. The prompt is used by an inner LM to produce a CUDA kernel.
3. The evaluator (`main.py:evaluate`) runs the kernel in a subprocess, computes a score, and retrieves targeted CUDA docs via RAG.
4. Score + side info (error details, cuda docs, runtime) feed back into GEPA's reflection and refinement loop.

Scoring is based on 6 stages (compilation, init, runtime, shape, correctness, perf) plus a performance bonus for speedup over the PyTorch baseline. Baselines are lazily measured and cached on first access.
