# EvolveAdapter

The `EvolveAdapter` integrates OpenEvolve projects with GEPA's optimization engine.

## How It Works

The `EvolveAdapter` lets you evolve using GEPA while allowing you to keep your existing OpenEvolve project structure with minimal modifications. The adapter takes a path to your existing OpenEvolve project containing your `initial_program.py`, `evaluator.py`, and `config.yaml`. The adapter uses the LLMs and system prompts from your `config.yaml` for program proposals. 

To fit with GEPA's training data-based evolution strategy, you must provide a batch of training data, and the adapter will call your `evaluate` function on each data instance. This means you must modify your `evaluate` function (and cascade evaluation functions if used) to take in a single data instance and return an `EvaluateResult`.

## Required Changes

Your `evaluate` function must be modified to accept a single data instance:

```python
def evaluate(program_path: str, data_instance: Any) -> EvaluationResult:
    # Extract parameters from data_instance
    # Evaluate program for this specific instance
    return EvaluationResult(metrics={...}, artifacts={...})
```

The `data_instance` parameter can be of any type (dict, string, custom object, tuple, etc.) - use whatever type best suits your original OpenEvolve project setup. The adapter simply passes each data instance to your `evaluate` function as-is.

The adapter will automatically call your function for each instance in the batch. If you use cascade evaluation, your stage functions (`evaluate_stage1`, `evaluate_stage2`, etc.) must also be modified similarly.

## Tutorial

For detailed migration instructions and an example, see:

- **[Tutorial Notebook](../../examples/evolve_adapter/function_minimization/tutorial.ipynb)**: Step-by-step guide with the OpenEvolve function minimization example
- **[Tutorial Script](../../examples/evolve_adapter/function_minimization/tutorial.py)**: Standalone Python script version

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

GEPA evaluates programs on batches of data instances. Each batch item represents a distinct problem instance (e.g., different test cases, etc.). The adapter:

- Loops over the batch and calls your `evaluate` function for each data instance
- Aggregates the per-instance `EvaluationResult` objects into GEPA's `EvaluationBatch` format
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
from gepa.adapters.evolve_adapter.evolve_adapter import EvolveAdapter

# Point to your OpenEvolve project directory
adapter = EvolveAdapter(path=Path("your-openevolve-project"))

# Use with GEPA's optimize function
result = optimize(
    seed_candidate={"program": initial_program},
    trainset=trainset,  # List of batch items
    adapter=adapter,
    max_metric_calls=500 # Change as needed
)
```
