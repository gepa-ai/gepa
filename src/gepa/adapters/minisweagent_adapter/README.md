# Mini-SWE-Agent Adapter for GEPA

This adapter integrates [mini-swe-agent](https://mini-swe-agent.com/) with GEPA for optimizing agent configurations on [SWE-bench](https://www.swebench.com/) tasks.

## Overview

The Mini-SWE-Agent adapter enables GEPA to optimize agent prompts and configurations for software engineering tasks. It:

1. **Runs mini-swe-agent** on SWE-bench instances with different configurations
2. **Evaluates results** by checking if the agent successfully generates patches
3. **Optionally validates** patches using the SWE-bench harness
4. **Provides reflective feedback** to improve agent configurations

## Installation

### Required Dependencies

```bash
# Mini-swe-agent is included in this repo at mini-swe-agent/
# Make sure GEPA is installed
pip install -e .
```

### Optional: SWE-bench Validation

For validation using the SWE-bench harness:

```bash
pip install swebench
```

### Docker Setup

SWE-bench evaluation requires Docker for running tests in isolated environments:

```bash
# Make sure Docker is installed and running
docker --version

# Pull SWE-bench images (optional, they'll be pulled automatically)
# For example:
docker pull swebench/sweb.eval.x86_64.django_1776_django_1776_4.0:latest
```

## Usage

### Basic Example

```python
from gepa.adapters.minisweagent_adapter import (
    MiniSWEAgentAdapter,
    load_swebench_instances,
)
from gepa import GEPAEngine, DataLoaderList

# Load SWE-bench instances
train_instances = load_swebench_instances(
    dataset="princeton-nlp/SWE-bench_Verified",
    split="test",
    slice_spec="0:10",  # Use first 10 instances for training
)

val_instances = load_swebench_instances(
    dataset="princeton-nlp/SWE-bench_Verified",
    split="test",
    slice_spec="10:20",  # Use next 10 for validation
)

# Create adapter
adapter = MiniSWEAgentAdapter(
    model_name="anthropic/claude-sonnet-4",
    environment_class="docker",
    run_validation=False,  # Set to True to run SWE-bench validation
)

# Define seed candidate with the text components to optimize
# The candidate contains the actual prompt text that GEPA will optimize
# These will be merged with the base agent config (model settings, environment, etc.)
seed_candidate = {
    "system_template": """You are a helpful assistant that can interact with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command.""",
    
    "instance_template": """Consider the following issue:
{{task}}

Please solve this issue by making appropriate changes to the codebase.
When done, submit your solution with:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
```""",
}

# GEPA will optimize these text components while keeping the base config (model, environment) fixed

# Setup data loaders
train_loader = DataLoaderList(train_instances)
val_loader = DataLoaderList(val_instances)

# Run GEPA optimization
engine = GEPAEngine(
    adapter=adapter,
    train_loader=train_loader,
    val_loader=val_loader,
    seed_candidate=seed_candidate,
    components_to_update=["system_template", "instance_template"],
)

result = engine.optimize(
    num_iterations=10,
    train_batch_size=2,
    proposal_batch_size=2,
)

# Get the best candidate
best_candidate = result.best_program()
print("Best system template:", best_candidate["system_template"])
print("Best instance template:", best_candidate["instance_template"])
```

### With Custom Configuration

You can also start from a custom agent configuration file:

```python
adapter = MiniSWEAgentAdapter(
    model_name="anthropic/claude-sonnet-4",
    agent_config_path="path/to/swebench.yaml",
    environment_class="docker",
    timeout=300,  # 5 minute timeout per instance
)
```

### With Validation

Enable SWE-bench validation to get accurate test-based scores:

```python
adapter = MiniSWEAgentAdapter(
    model_name="anthropic/claude-sonnet-4",
    environment_class="docker",
    run_validation=True,  # Run SWE-bench harness
    validation_max_workers=4,  # Parallel validation workers
)
```

**Note:** Validation can be slow (20+ minutes per batch) as it runs actual tests in Docker containers.

## How Candidates Work

The `candidate` dictionary passed to GEPA contains the **text components** you want to optimize:

```python
candidate = {
    "system_template": "Your system prompt text...",
    "instance_template": "Your task template with {{task}} placeholder...",
}
```

These components are **merged** with the base agent configuration (model, environment, etc.) to build the complete agent. The adapter:

1. Takes the base configuration (set during initialization)
2. Overlays the candidate components onto it
3. Runs the agent with the resulting configuration

This separation allows GEPA to optimize prompts while keeping infrastructure settings (model, Docker config, etc.) fixed.

## Configuration Components

You can optimize any of these components:

### Agent Configuration
- `system_template`: System prompt for the agent
- `instance_template`: Task-specific prompt template
- `format_error_template`: Message shown when agent output is malformed
- `action_observation_template`: How observations are formatted
- `timeout_template`: Message shown when commands timeout
- `step_limit`: Maximum number of steps (as string, e.g., "250")
- `cost_limit`: Maximum cost in dollars (as string, e.g., "3.0")

### Example Multi-Component Optimization

```python
seed_candidate = {
    "system_template": "...",
    "instance_template": "...",
    "format_error_template": "...",
    "step_limit": "250",
    "cost_limit": "3.0",
}

engine = GEPAEngine(
    adapter=adapter,
    train_loader=train_loader,
    val_loader=val_loader,
    seed_candidate=seed_candidate,
    components_to_update=[
        "system_template",
        "instance_template",
        "format_error_template",
    ],
)
```

## Data Format

### Input: MiniSWEAgentDataInst

Each instance contains:
- `instance_id`: Unique identifier (e.g., "django__django-12345")
- `problem_statement`: The issue description
- `base_commit`, `patch`, `test_patch`: Version control information
- `repo`, `version`: Repository details
- Other SWE-bench metadata

Use `load_swebench_instances()` to load from Hugging Face datasets.

### Output: MiniSWEAgentRolloutOutput

Each rollout produces:
- `model_patch`: The generated patch/diff
- `exit_status`: How the agent terminated (e.g., "Submitted", "LimitsExceeded")
- `n_calls`, `cost`: Resource usage
- `validation_result`: Optional test results from SWE-bench harness

### Trajectory: MiniSWEAgentTrajectory

For reflective learning, captures:
- Full message history (system, user, assistant)
- Agent execution details
- Error messages if any
- Validation results

## Scoring

### Without Validation
- Success (Submitted): 1.0
- Failure: 0.0 (configurable via `failure_score`)

### With Validation
- All tests passed (resolved): 1.0
- Partial: `tests_passed / total_tests`
- No tests passed: 0.0

## Performance Tips

1. **Start without validation**: Validation is slow. Use `run_validation=False` initially to iterate quickly on prompts.

2. **Use smaller batches**: SWE-bench instances can take 1-5 minutes each. Start with `train_batch_size=1-2`.

3. **Filter instances**: Use `slice_spec` to work with a subset of instances during development.

4. **Use local environment**: For debugging, you can use `environment_class="local"` (but be careful, this runs commands on your actual system).

5. **Cache Docker images**: Pre-pull Docker images to avoid delays during evaluation.

## Troubleshooting

### Docker Issues

If you encounter Docker permission errors:
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Timeout Errors

If instances timeout frequently, increase the timeout:
```python
adapter = MiniSWEAgentAdapter(
    model_name="...",
    timeout=600,  # 10 minutes
)
```

### Validation Not Found

If you get "swebench package is required":
```bash
pip install swebench
```

## Examples

See `src/gepa/examples/` for complete examples:
- Basic optimization without validation
- Full optimization with validation
- Multi-component optimization

## References

- [Mini-SWE-Agent Documentation](https://mini-swe-agent.com/)
- [SWE-bench](https://www.swebench.com/)
- [GEPA Documentation](https://github.com/gepa-ai/gepa)

