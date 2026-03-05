# gskill: Learning Repository-Specific Skills

gskill learns repository-specific skills for coding agents. Given any GitHub repository, it discovers common patterns, structures, and debugging strategies, then produces a skill file that makes agents better at fixing bugs in that repo.

It uses [SWE-smith](https://swesmith.com) for task generation and GEPA's `optimize_anything` API for skill optimization.

## How It Works

1. **Generate tasks.** SWE-smith mines real commits from the target repo, introduces bugs, and produces verifiable task instances, each with a problem statement, a Docker environment, and tests.

2. **Optimize skills.** GEPA starts with empty skills and runs the agent on batches of tasks in parallel Docker containers. Pass/fail results, agent traces, and test output go to a reflection model that proposes better skills. This repeats until the budget is exhausted.

3. **Deploy.** The output is `best_skills.txt`, which gets injected into the agent's system prompt. Skills transfer directly to other agents without retraining. (This means you can train on  a cheap model and transfer to a more expensive model!)

## Installation

```bash
pip install gepa[gskill]
pip install mini-swe-agent swebench
```

Set up API keys and Docker:

```bash
export OPENAI_API_KEY=<your-key>

# Docker must be running
docker ps

# Download SWE-smith images for your target repo
python -m swesmith.build_repo.download_images
```

## Training

### Smoke Test

Run a quick validation to make sure everything is wired up:

```bash
python -m gepa.gskill.train_optimize_anything \
  --smoke-test --model "gpt-5-mini"
```

This runs 3 training tasks with a small budget so you can verify Docker, API keys, and the pipeline work end to end.

### Full Run

```bash
python -m gepa.gskill.train_optimize_anything \
  --repo pygments__pygments \
  --train-size 200 --val-size 50 --test-size 100 \
  --model gpt-5-mini --reflection-model gpt-5.2-pro \
  --workers 6 --max-metric-calls 600 \
  --proposer loop --wandb
```

### Resume a Previous Run

If a run gets interrupted, resume from its saved state:

```bash
python -m gepa.gskill.train_optimize_anything \
  --resume gepa_results/logs/run_XXXXXXXX
```

The repo and seed are loaded from the previous run's config to keep data splits consistent.

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--repo` | `pygments__pygments` | Target repository in SWE-smith |
| `--model` | `gpt-5-mini` | Agent model (runs inside Docker) |
| `--reflection-model` | `gpt-5.2-pro` | Model for reflection and skill proposal |
| `--workers` | 6 | Number of parallel Docker containers |
| `--max-metric-calls` | 600 | Total rollout budget |
| `--proposer` | `batch` | `batch` (all at once) or `loop` (one at a time then merge) |
| `--train-size` | 200 | Number of training examples |
| `--val-size` | 50 | Number of validation examples for Pareto selection |
| `--test-size` | 100 | Number of test examples for final evaluation |
| `--run-testset` | off | Evaluate before and after optimization |
| `--resume` | None | Resume from a previous run directory |
| `--smoke-test` | off | Quick validation with 3 tasks |
| `--wandb` | off | Enable Weights & Biases tracking |
| `--timeout` | 43200 | Max seconds to run (default 12 hours) |
| `--seed` | 42 | Random seed for reproducibility |

## Writing a Fitness Function

gskill uses `optimize_anything`'s evaluator protocol. The fitness function scores a single candidate on a single task and returns feedback for the reflection model.

### Evaluator Signature

Your fitness function must accept `candidate` and `example` and return `(score, side_info)`:

```python
def fitness_fn(
    candidate: dict[str, str],
    example: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """
    Args:
        candidate: Dict of optimizable text parameters (e.g. {"skills": "..."})
        example: A single task instance from the dataset

    Returns:
        score: Float, higher is better. 1.0 for pass, 0.0 for fail.
        side_info: Dict of diagnostic info for the reflection model.
    """
```

!!! danger "The second parameter must be named `example`"
    GEPA's `EvaluatorWrapper` inspects your function's signature and passes each task as a kwarg called `example`. It only forwards kwargs that match your function's parameter names. If you name the parameter something else (like `batch` or `task`), the kwarg gets filtered out and the call will crash with a missing argument error.

### Side Info Structure

The `side_info` dict is what the reflection model sees when proposing improved skills. The more useful information you put here, the better the reflection will be:

```python
side_info = {
    "Input": {
        "Task ID": "instance-123",
        "Problem": "Fix the off-by-one error in parser.py...",
    },
    "Generated Outputs": {
        "Patch": "diff --git a/parser.py ...",
        "Agent Trace": "Step 1: Read parser.py\nStep 2: ...",
    },
    "Feedback": {
        "Status": "f2p_failed",  # or "all_passed", "no_patch", etc.
        "Test Output": "FAILED test_parser.py::test_edge_case ...",
    },
    "scores": {
        "correctness": 0.0,
    },
}
```

Key fields:

- **Input**: what the agent was asked to do. helps the reflection model understand the task.
- **Generated Outputs**: what the agent actually produced. include the patch and a trace of agent actions.
- **Feedback**: why it passed or failed. include test output so the reflection model can see the actual errors.
- **scores**: numeric scores for multi-objective tracking. `optimize_anything` uses the `"scores"` key for Pareto frontier tracking.

### Parallelism

GEPA handles parallelism for you. Set `parallel=True` and `max_workers` in `EngineConfig` and GEPA's adapter will call your fitness function concurrently across examples:

```python
config = GEPAConfig(
    engine=EngineConfig(
        parallel=True,
        max_workers=6,  # number of concurrent evaluations
        max_metric_calls=600,
    ),
)
```

You don't need to implement your own `ThreadPoolExecutor` inside the fitness function. If your evaluation uses a resource pool (like gskill's Docker harness pool), GEPA's threads will access it concurrently.

### Complete Example

Here's a simplified version of gskill's fitness function showing the full pattern:

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

def create_fitness_fn(model_name: str, n_workers: int):
    """Create a fitness function with a pool of Docker harnesses."""
    harness_pool = [SWEHarness() for _ in range(n_workers)]
    # ... pool management code ...

    def fitness_fn(candidate: dict[str, str], example: dict[str, Any]) -> tuple[float, dict]:
        skills = candidate["skills"]
        harness = get_available_harness()

        try:
            # 1. Setup Docker container with the task
            harness.setup_task(task_instance=example)

            # 2. Run the agent with current skills
            patch, trace, metrics = harness.run_agent(
                example["problem_statement"], skills, model_name=model_name
            )

            # 3. Verify the patch with tests
            passed, test_output = harness.verify_with_patch(patch)
            score = 1.0 if passed else 0.0

            # 4. Return score + diagnostic info for reflection
            side_info = {
                "Input": {"Problem": example["problem_statement"][:200]},
                "Generated Outputs": {"Patch": patch[:500], "Agent Trace": trace},
                "Feedback": {"Status": "passed" if passed else "failed", "Test Output": test_output},
                "scores": {"correctness": score},
            }
            return score, side_info

        finally:
            harness.cleanup()
            release_harness(harness)

    return fitness_fn

# Use it
fitness_fn = create_fitness_fn("gpt-5-mini", n_workers=6)

result = optimize_anything(
    seed_candidate={"skills": ""},
    evaluator=fitness_fn,
    dataset=train_data,
    valset=val_data,
    config=GEPAConfig(
        engine=EngineConfig(parallel=True, max_workers=6, max_metric_calls=600),
    ),
)
```

## Custom Proposers

By default, GEPA uses its built-in reflection to propose improved skills. gskill overrides this with custom proposers that are tuned for the skill learning task.

### Batch Proposer

The `batch` proposer (default) sends all evaluation results to the reflection model at once:

```bash
python -m gepa.gskill.train_optimize_anything --proposer batch
```

### Loop Proposer

The `loop` proposer processes each evaluation result one at a time, then merges the intermediate skills into a final set:

```bash
python -m gepa.gskill.train_optimize_anything --proposer loop
```

This tends to produce more detailed skills since the reflection model focuses on one failure at a time, but it uses more LLM calls.

### Writing Your Own Proposer

A custom proposer is a callable that takes the current candidate, reflective dataset, and components to update, and returns a dict of new values:

```python
def my_proposer(
    candidate: dict[str, str],
    reflective_dataset: dict[str, list[dict]],
    components_to_update: list[str],
) -> dict[str, str]:
    """
    Args:
        candidate: Current candidate values (e.g. {"skills": "..."})
        reflective_dataset: Per-component list of evaluation side_info dicts
        components_to_update: Which components to propose new values for

    Returns:
        Dict mapping component names to their new proposed values
    """
    results = {}
    for component in components_to_update:
        current_value = candidate[component]
        feedback = reflective_dataset[component]
        # ... call an LLM to propose improvements ...
        results[component] = new_value
    return results
```

Pass it to GEPA via `ReflectionConfig`:

```python
config = GEPAConfig(
    reflection=ReflectionConfig(
        custom_candidate_proposer=my_proposer,
    ),
)
```

## Evaluation

After training, evaluate the learned skills on the held-out test set.

### Mini-SWE-agent

```bash
# Runs both with-skills and without-skills conditions
python -m gepa.gskill.gskill.evaluate.mini_swe_agent \
  --config gepa_results/logs/run_xxx/config.json \
  --workers 16
```

### Claude Code

```bash
# Baseline (no skills)
python -m gepa.gskill.gskill.evaluate.claude_code \
  --config gepa_results/logs/run_xxx/config.json \
  --model haiku --workers 4

# With skills (copies best_skills.txt as CLAUDE.md)
python -m gepa.gskill.gskill.evaluate.claude_code \
  --config gepa_results/logs/run_xxx/config.json \
  --model haiku --workers 4 --use-skills

# With Claude Code Skills (.claude/skills/<repo>/SKILL.md)
python -m gepa.gskill.gskill.evaluate.claude_code_skills \
  --config gepa_results/logs/run_xxx/config.json \
  --model sonnet --workers 4 --use-skills
```

## Output

Results are saved to `gepa_results/logs/run_<timestamp>_<id>/`:

| File | Description |
|------|-------------|
| `prompts/best_skills.txt` | The learned skills |
| `config.json` | Experiment configuration |
| `iterations.jsonl` | Per-evaluation batch metrics |
| `proposer_calls/` | Full proposer call logs (input/output for each reflection) |
| `prompts/` | Each unique prompt version, saved by hash |
| `cost_summary.txt` | Cost breakdown (agent vs reflection) |
| `gepa_state.bin` | Checkpoint for resuming |
| `terminal.log` | Full terminal output |

## References

- [SWE-smith](https://swesmith.com) - Task generation from real repositories
- [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) - Lightweight coding agent
- [optimize_anything API](quickstart.md) - GEPA's main optimization interface
