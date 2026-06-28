# Experiment tracking

GEPA logs to **Weights & Biases** and **MLflow** via `TrackingConfig`. Configure it in the GEPA
engine's config block:

```python
result = optimize_anything(
    seed_candidate=SEED, evaluator=evaluate, dataset=trainset, valset=valset,
    config=OptimizeAnythingConfig(
        engine="gepa", max_evals=300,
        engine_config={
            "reflection": {"reflection_lm": "anthropic/claude-sonnet-4-6"},
            "tracking": {                       # -> TrackingConfig
                "use_wandb": True,
                "wandb_init_kwargs": {"project": "my-gepa-run", "name": "experiment-1"},
            },
        },
    ),
)
```

## `TrackingConfig` fields
- `use_wandb`, `wandb_init_kwargs`, `wandb_attach_existing`
- `use_mlflow`, `mlflow_tracking_uri`, `mlflow_experiment_name`, `mlflow_attach_existing`
- `key_prefix` — namespaces all logged metrics (e.g. `"gepa/"` → `gepa/val_score`)

## What gets logged
- **scalars** — `val_program_average`, `best_score_on_valset`, `total_metric_calls`, …
- **tables** — `candidates`, `proposals`, `valset_scores`, `valset_pareto_front`
- **run summary** — seed and best-found component text
- **interactive candidate-tree HTML**

## Attaching to an existing run
If you've already called `wandb.init()` (or started an MLflow run) yourself — e.g. a parent script
that also logs other things — set `wandb_attach_existing=True` (or `mlflow_attach_existing=True`).
GEPA then logs **into your active run** without calling `init()`/`finish()`, so it won't disrupt the
run lifecycle. Combine with `key_prefix="gepa/"` to keep GEPA's metrics in their own namespace and
avoid step collisions.

## Custom logging
You can also pass a thin tracker via `OptimizeAnythingConfig.tracker` to receive per-eval and
per-iteration callbacks (`log_eval(used, score, best_score, cost)` / `log_metrics(metrics, step)`) —
useful for piping into a logging system `TrackingConfig` doesn't cover. For full GEPA dashboards
(Pareto/candidate tables, candidate tree), prefer `TrackingConfig` above.

See the official guide for the authoritative list of fields and behaviors:
<https://gepa-ai.github.io/gepa/guides/experiment-tracking/>.
