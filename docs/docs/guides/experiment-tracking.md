# Experiment Tracking with WandB and MLflow

GEPA integrates with [Weights & Biases](https://wandb.ai) and [MLflow](https://mlflow.org) to log metrics, prompts, and structured tables throughout optimization.  Both backends can be enabled simultaneously.

---

## Quick Start

=== "Weights & Biases"

    ```python
    import gepa

    result = gepa.optimize(
        seed_candidate={"system_prompt": "You are a helpful assistant."},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm="openai/gpt-5",
        max_metric_calls=200,
        use_wandb=True,
        wandb_init_kwargs={"project": "my-gepa-run", "name": "experiment-1"},
    )
    ```

=== "MLflow"

    ```python
    import gepa

    result = gepa.optimize(
        seed_candidate={"system_prompt": "You are a helpful assistant."},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm="openai/gpt-5",
        max_metric_calls=200,
        use_mlflow=True,
        mlflow_tracking_uri="./mlruns",         # local directory
        mlflow_experiment_name="my-gepa-run",
    )
    ```

=== "Both at once"

    ```python
    result = gepa.optimize(
        ...
        use_wandb=True,
        use_mlflow=True,
        mlflow_experiment_name="my-gepa-run",
    )
    ```

=== "optimize_anything"

    ```python
    from gepa.optimize_anything import optimize_anything, GEPAConfig, TrackingConfig

    result = optimize_anything(
        seed_candidate="...",
        evaluator=my_evaluator,
        config=GEPAConfig(
            tracking=TrackingConfig(
                use_wandb=True,
                wandb_init_kwargs={"project": "my-gepa-run"},
            )
        ),
    )
    ```

---

## What Gets Logged

### Scalar Metrics (line charts)

Every iteration GEPA logs scalars you can plot over time:

| Metric | Description |
|--------|-------------|
| `val_program_average` | Average valset score for the current candidate |
| `best_score_on_valset` | Best valset score found so far |
| `valset_pareto_front_agg` | Pareto-frontier aggregate score |
| `subsample/before` | Reflection minibatch score before mutation |
| `subsample/after` | Reflection minibatch score after mutation |
| `total_metric_calls` | Cumulative evaluator calls |

### Structured Tables

GEPA logs structured data as tables (WandB `Table` / MLflow `log_table`):

#### `candidates` table

One row per **accepted** candidate.  Join on `candidate_idx` to link to other tables.

| Column | Description |
|--------|-------------|
| `iteration` | Iteration number |
| `candidate_idx` | Candidate index (0 = seed) |
| `parent_ids` | Parent candidate indices |
| `valset_score` | Aggregate validation score |
| `is_best` | Whether this is the current best |
| `text:<component>` | The full text of each optimized component |

#### `proposals` table

One row per **reflection LM call** — both accepted and rejected proposals.  This is the key table for understanding what the LM generated and why it was kept or discarded.

| Column | Description |
|--------|-------------|
| `iteration` | Iteration number |
| `component` | Which prompt component was updated |
| `status` | `"accepted"` or `"rejected"` |
| `candidate_idx` | Accepted: index in `candidates` table.  Rejected: `-1` |
| `parent_ids` | Parent candidate the proposal was based on |
| `subsample_score_before` | Minibatch score before mutation |
| `subsample_score_after` | Minibatch score after mutation |
| `prompt` | Full prompt sent to the reflection LM |
| `raw_lm_output` | Raw response from the reflection LM |
| `proposed_text` | The extracted new instruction text |

!!! tip "Joining accepted proposals to candidates"
    Filter `proposals` where `status == "accepted"` then join on `candidate_idx`
    to see which LM call produced each candidate in the `candidates` table.

#### `valset_scores`, `valset_pareto_front`, `objective_scores`, `objective_pareto_front`

Per-example and per-objective scores logged as growing matrix tables.  See the
[GEPA paper](https://arxiv.org/abs/2507.19457) for how Pareto frontiers are used.

### Run Summary

At the end of optimization GEPA logs:

- `best_candidate_idx`, `best_valset_score`, `total_iterations`, `total_candidates`
- `seed/<component>` — the original seed text for each component
- `best/<component>` — the best-found text for each component

### Candidate Tree Visualization (HTML artifact)

After each accepted candidate, GEPA logs an interactive HTML visualization of the full candidate lineage tree.  In WandB this appears as a `wandb.Html` artifact; in MLflow as an HTML artifact file.

---

## WandB Tips

### Using the `proposals` table to debug rejections

```python
import wandb

api = wandb.Api()
run = api.run("my-entity/my-project/run-id")
proposals = run.history(pandas=True)  # or use the Tables tab in the UI
```

In the WandB UI, navigate to **Tables → proposals**.  Filter by `status = "rejected"` to see what the LM proposed that didn't pass the subsample acceptance test.  The `raw_lm_output` column shows exactly what the LM generated.

### Custom WandB init kwargs

Any kwargs accepted by [`wandb.init()`](https://docs.wandb.ai/ref/python/init) can be passed:

```python
gepa.optimize(
    ...
    use_wandb=True,
    wandb_init_kwargs={
        "project": "prompt-optimization",
        "name": f"run-{my_experiment_id}",
        "tags": ["aime", "gpt-5"],
        "notes": "Testing new seed prompt",
        "config": {"dataset": "aime_2025", "model": "gpt-5"},
    },
)
```

---

## MLflow Tips

### Tracking URI formats

```python
# Local filesystem
mlflow_tracking_uri="./mlruns"

# Remote server
mlflow_tracking_uri="http://localhost:5000"

# Databricks
mlflow_tracking_uri="databricks"
```

### Resuming an existing run

Wrap GEPA inside an active MLflow run to log to it:

```python
import mlflow

with mlflow.start_run(run_name="my-run"):
    result = gepa.optimize(
        ...
        use_mlflow=True,  # GEPA detects the active run and logs into it
    )
```

### Viewing the proposals table

MLflow logs tables as JSON artifacts.  View them in the MLflow UI under **Artifacts → proposals.json**, or load programmatically:

```python
import mlflow

client = mlflow.MlflowClient()
# List artifacts
artifacts = client.list_artifacts(run_id, "proposals")
# Download and read
local_path = client.download_artifacts(run_id, "proposals/proposals.json")
import json
with open(local_path) as f:
    proposals = json.load(f)
```

---

## Logging Into an Already-Active Run

When GEPA is embedded inside a larger workflow that already has an active
wandb/MLflow run, use `wandb_attach_existing=True` or `mlflow_attach_existing=True`
to log GEPA metrics into it **without touching the run lifecycle**.

Without these flags, GEPA calls `wandb.init()` on entry and `wandb.finish()` on exit —
terminating the caller's run and causing all subsequent `wandb.log()` calls to
silently fail.

=== "WandB"

    ```python
    import wandb
    import gepa

    wandb.init(project="my-project")  # caller owns this run

    result = gepa.optimize(
        seed_candidate={"system_prompt": "..."},
        ...
        use_wandb=True,
        wandb_attach_existing=True,   # skip init() + finish()
    )

    # Still works — run was never closed by GEPA
    wandb.log({"final_metric": compute_metric()})
    wandb.finish()
    ```

    Or via `TrackingConfig` with `optimize_anything`:

    ```python
    from gepa.optimize_anything import optimize_anything, GEPAConfig, TrackingConfig

    result = optimize_anything(
        ...,
        config=GEPAConfig(
            tracking=TrackingConfig(
                use_wandb=True,
                wandb_attach_existing=True,
            )
        ),
    )
    ```

=== "MLflow"

    ```python
    import mlflow
    import gepa

    with mlflow.start_run():           # caller owns this run
        mlflow.log_param("model", "gpt-5")

        result = gepa.optimize(
            ...,
            use_mlflow=True,
            mlflow_attach_existing=True,   # skip start_run() + end_run()
        )

        # Still works — run was never closed by GEPA
        mlflow.log_metric("final_score", result.val_aggregate_scores[result.best_idx])
    ```

!!! tip "What gets logged in attach mode"
    All of GEPA's normal logging still works — `log_metrics()`, `log_table()`,
    `log_summary()`, `log_html()` — it just doesn't touch `init` / `finish`.
    GEPA's metrics appear alongside the caller's metrics in the same run.

---

## Namespacing Keys with a Prefix

When logging into a shared run, use `key_prefix` to namespace all GEPA keys
so they don't collide with the caller's metrics:

=== "optimize_anything"

    ```python
    from gepa.optimize_anything import GEPAConfig, TrackingConfig

    result = optimize_anything(
        ...,
        config=GEPAConfig(
            tracking=TrackingConfig(
                use_wandb=True,
                wandb_attach_existing=True,
                key_prefix="gepa/",        # all keys become gepa/<key>
            )
        ),
    )
    # wandb will show: gepa/val_score, gepa/candidates (table), gepa/candidate_tree, …
    ```

=== "gepa.optimize"

    ```python
    result = gepa.optimize(
        ...,
        use_wandb=True,
        wandb_attach_existing=True,
        tracking_key_prefix="gepa/",
    )
    ```

`key_prefix` applies uniformly to **all** logged data: scalar metrics,
config params, summary values, table names, and HTML artifact keys.

---

## Installation

WandB and MLflow are optional dependencies included in `gepa[full]`:

```bash
pip install "gepa[full]"         # includes wandb + mlflow
pip install wandb mlflow         # or install individually
```
