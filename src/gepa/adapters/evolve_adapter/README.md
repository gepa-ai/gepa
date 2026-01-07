# EvolveAdapter

The `EvolveAdapter` integrates OpenEvolve projects with GEPA's optimization engine.

## How It Works

The `EvolveAdapter` lets you evolve using GEPA while allowing you to keep your existing OpenEvolve project structure with minimal modifications. The adapter takes a path to your existing OpenEvolve project containing your `initial_program.py`, `evaluator.py`, and `config.yaml`. The adapter uses the LLMs and system prompts from your `config.yaml` for program proposals. To fit with GEPA's fine-grained evolution strategy, you must modify your `evaluate` function (and cascade evaluation functions if used) to accept batches of training data and return lists of per-data instance results.

## Required Changes

Your `evaluate` function must be modified to accept a `batch` parameter and return a list of results:

```python
def evaluate(program_path: str, batch: list) -> list[EvaluationResult]:
    results = []
    for batch_item in batch:
        # Extract parameters from batch_item
        # Evaluate program for this batch item
        results.append(EvaluationResult(metrics={...}, artifacts={...}))
    return results
```

If you use cascade evaluation, your stage functions (`evaluate_stage1`, `evaluate_stage2`, etc.) must also be modified similarly.

## Tutorial

For detailed migration instructions and an example, see:

- **[Tutorial Notebook](tutorial_evolve_adapter.ipynb)**: Step-by-step guide with the function minimization example
- **[Tutorial Script](tutorial_evolve_adapter.py)**: Standalone Python script version

The tutorial covers:
- How to modify your `evaluate` function for batch evaluation
- How to structure batch data
- How to modify cascade evaluation functions
- How to run GEPA evolution with `EvolveAdapter`

## Project Structure

Your OpenEvolve project directory should contain:

- `config.yaml`: LLM and evaluation configuration
- `evaluator.py`: Modified `evaluate` function (and stage functions if using cascade evaluation)
- `initial_program.py`: Initial program with `EVOLVE-BLOCK` markers

## Batch Evaluation

GEPA evaluates programs on batches of data instances. Each batch item represents a distinct problem instance (e.g., different function minimization problems, different test cases). The adapter:

- Calls your `evaluate` function with the full batch
- Converts your `list[EvaluationResult]` into GEPA's `EvaluationBatch` format
- Uses per-instance results to generate richer feedback for program improvement

## Configuration

The adapter reads your OpenEvolve `config.yaml` for LLM and evaluation settings. 

```yaml
llm:
  primary_model: "gemini-2.5-flash-lite"
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
  temperature: 0.7
  max_tokens: 16000

prompt:
  system_message: "Your system message here"

evaluator:
  cascade_evaluation: true
  cascade_thresholds: [1.3]
  timeout: 60
```

### Program Proposal
The `system_message` in `config["prompt"]` is prepended to the default proposal prompt (see `evolve_program_proposal_signature.py`).

## Usage

```python
from pathlib import Path
from gepa import optimize
from gepa.adapters.generic_evolve_adapter import EvolveAdapter

# Point to your OpenEvolve project directory
adapter = EvolveAdapter(path=Path("your-openevolve-project"))

# Use with GEPA's optimize function
result = optimize(
    seed_candidate={"program": initial_program},
    trainset=trainset,  # List of batch items
    adapter=adapter,
    max_metric_calls=60
)
```
