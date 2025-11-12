# Mini-SWE-Agent GEPA Training Example

This example demonstrates how to use GEPA to optimize mini-swe-agent prompts on SWE-bench tasks.

## Overview

The training script:
1. Loads the default `swebench.yaml` configuration from mini-swe-agent
2. Extracts `system_template` and `instance_template` as the seed candidate
3. Splits SWE-bench instances into train/val/test sets
4. Runs GEPA optimization to improve the prompts
5. Evaluates the optimized prompts on the test set

## Prerequisites

### 1. Docker

SWE-bench evaluation requires Docker:

```bash
# Check if Docker is installed
docker --version

# If not installed, see: https://docs.docker.com/get-docker/
```

### 2. Python Dependencies

```bash
# Install GEPA with mini-swe-agent
cd /path/to/gepa-swe
pip install -e .
pip install -e mini-swe-agent/

# Install additional dependencies
pip install litellm datasets pyyaml
```

### 3. API Keys

Set up API keys for the models:

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"  # For reflection model
```

## Quick Start

### Basic Usage

```bash
python train_minisweagent.py --model anthropic/claude-sonnet-4 --n-instances 10
```

This will:
- Use 10 SWE-bench instances (4 train, 3 val, 3 test)
- Run optimization with default settings
- Save results to `gepa_minisweagent/`

### Recommended Settings for Testing

For quick testing (5-10 minutes):

```bash
python train_minisweagent.py \
    --model anthropic/claude-sonnet-4 \
    --reflection-model openai/gpt-4o \
    --n-instances 6 \
    --max-metric-calls 10 \
    --environment docker
```

### Full Training Run

For a more comprehensive run (several hours):

```bash
python train_minisweagent.py \
    --model anthropic/claude-sonnet-4 \
    --reflection-model openai/gpt-4o \
    --dataset princeton-nlp/SWE-bench_Verified \
    --n-instances 30 \
    --max-metric-calls 100 \
    --reflection-batch-size 2 \
    --environment docker \
    --output-dir gepa_minisweagent_full
```

### With Validation

To enable SWE-bench validation (requires `swebench` package, very slow):

```bash
# Install swebench
pip install swebench

# Run with validation
python train_minisweagent.py \
    --model anthropic/claude-sonnet-4 \
    --n-instances 10 \
    --run-validation
```

**Warning:** Validation runs actual tests in Docker and can take 20+ minutes per batch.

## Command-Line Options

```
--model MODEL                 Model for agent execution (default: anthropic/claude-sonnet-4)
--reflection-model MODEL      Model for optimization (default: openai/gpt-4o)
--dataset DATASET             SWE-bench dataset (default: princeton-nlp/SWE-bench_Verified)
--n-instances N               Total instances to use (default: 30)
--environment ENV             Environment type: docker or local (default: docker)
--run-validation              Enable SWE-bench validation (slow but accurate)
--config PATH                 Path to base agent config (default: mini-swe-agent's swebench.yaml)
--output-dir DIR              Output directory (default: gepa_minisweagent)
--max-metric-calls N          Max optimization iterations (default: 100)
--reflection-batch-size N     Batch size for reflection (default: 2)
```

## Output Files

The script creates several output files in the output directory:

```
gepa_minisweagent/
├── testset_results_no_prompt.json      # Baseline (empty prompts)
├── testset_results_before_opt.json     # Before optimization
├── testset_results_after_opt.json      # After optimization
├── best_candidate.json                 # Optimized prompts (JSON)
├── best_candidate.txt                  # Optimized prompts (readable)
├── summary.json                        # Overall summary
└── temp/                               # Temporary files
```

### Key Files

- **`best_candidate.json`**: The optimized prompts you can use in production
- **`summary.json`**: Performance metrics and comparisons
- **`best_candidate.txt`**: Human-readable version of optimized prompts

## Understanding Results

The script evaluates three versions on the test set:

1. **Baseline (empty)**: Minimal prompts to establish a baseline
2. **Before optimization**: The original prompts from `swebench.yaml`
3. **After optimization**: GEPA-optimized prompts

Example output:

```
Test Set Results:
  - Baseline (empty):  0/9 = 0.0%
  - Before opt:        2/9 = 22.2%
  - After opt:         4/9 = 44.4%

Improvement:
  - vs Baseline: +4.0 (+44.4%)
  - vs Pre-opt:  +2.0 (+22.2%)
```

## What Gets Optimized

GEPA optimizes two key prompts:

1. **`system_template`**: The system prompt that defines the agent's role and behavior
2. **`instance_template`**: The template for presenting tasks to the agent

The optimization process:
- Tries variations of these prompts
- Evaluates them on training instances
- Selects improvements based on validation set
- Returns the best-performing version

## Using Optimized Prompts

After optimization, you can use the best prompts in several ways:

### 1. Direct Use in mini-swe-agent

Update your config file with the optimized prompts:

```yaml
agent:
  system_template: |
    <paste optimized system_template from best_candidate.txt>
  instance_template: |
    <paste optimized instance_template from best_candidate.txt>
```

### 2. Programmatic Use

```python
import json
from pathlib import Path

# Load optimized candidate
with open("gepa_minisweagent/best_candidate.json") as f:
    best_candidate = json.load(f)

# Use with MiniSWEAgentAdapter
from gepa.adapters.minisweagent_adapter import MiniSWEAgentAdapter

adapter = MiniSWEAgentAdapter(
    model_name="anthropic/claude-sonnet-4",
    environment_class="docker",
)

# Evaluate with optimized prompts
results = adapter.evaluate(test_instances, best_candidate)
```

## Performance Tips

### 1. Start Small
Begin with a small number of instances (5-10) to test your setup:
```bash
python train_minisweagent.py --n-instances 6 --max-metric-calls 10
```

### 2. Use Parallel Evaluation
SWE-bench instances can be slow. Consider using a smaller, faster subset for development.

### 3. Monitor Costs
Each instance can make up to 250 model calls. Monitor your API costs:
- Train on fewer instances initially
- Use cheaper models for testing (e.g., gpt-4o-mini)
- Set lower `--max-metric-calls` for quick iterations

### 4. Disable Validation for Iteration
Validation is very slow. Disable it during development:
```bash
python train_minisweagent.py --n-instances 10  # No --run-validation
```

Only enable validation for final evaluation.

## Troubleshooting

### Docker Issues

```bash
# Check Docker is running
docker ps

# Pull SWE-bench images manually (optional)
docker pull swebench/sweb.eval.x86_64.django_1776_django_1776_4.0:latest
```

### Timeout Errors

If instances timeout frequently, adjust in the script or config:
```python
adapter = MiniSWEAgentAdapter(
    ...,
    timeout=600,  # Increase to 10 minutes
)
```

### Out of Memory

SWE-bench evaluation uses Docker containers which can consume significant memory. If you encounter OOM:
- Reduce `--n-instances`
- Close other Docker containers
- Increase Docker memory limit in Docker Desktop settings

## Example Workflow

1. **Quick test** (2-5 minutes):
   ```bash
   python train_minisweagent.py --n-instances 6 --max-metric-calls 5
   ```

2. **Development run** (30-60 minutes):
   ```bash
   python train_minisweagent.py --n-instances 12 --max-metric-calls 20
   ```

3. **Production run** (several hours):
   ```bash
   python train_minisweagent.py --n-instances 50 --max-metric-calls 200
   ```

## References

- [Mini-SWE-Agent](https://mini-swe-agent.com/)
- [SWE-bench](https://www.swebench.com/)
- [GEPA Documentation](https://github.com/gepa-ai/gepa)


