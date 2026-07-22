# Experiment tracking

The gepa backend logs to **Weights & Biases** and **MLflow** via `TrackingConfig` — configure it
under `engine_config["tracking"]` (a valid `GEPAConfig` field, coerced to `TrackingConfig`). The
other backends have no built-in tracker; for them, rely on the eval server's `output_dir` records
(always written, whatever the backend):

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
- `use_wandb`, `wandb_api_key`, `wandb_init_kwargs`, `wandb_attach_existing`, `wandb_step_metric`
- `use_mlflow`, `mlflow_tracking_uri`, `mlflow_experiment_name`, `mlflow_attach_existing`
- `key_prefix` — prepended to every logged key/name (e.g. `"gepa/"` → `gepa/val_score`)
- `logger` — a custom `LoggerProtocol` for plain-text log lines

## What gets logged
- **scalars** — `val_program_average`, `best_score_on_valset`, `total_metric_calls`, …
- **tables** — `candidates`, `proposals`, `valset_scores`, `valset_pareto_front`
- **run summary** — seed and best-found component text
- **interactive candidate-tree HTML**

## Attaching to an existing run
If you've already called `wandb.init()` (or started an MLflow run) yourself — e.g. a parent script
that also logs other things — set `wandb_attach_existing=True` (or `mlflow_attach_existing=True`).
The tracker then logs **into your active run** without calling `init()`/`finish()`, so it won't
disrupt the run lifecycle. Combine with `key_prefix="gepa/"` to keep its metrics in their own
namespace.

When the host loop manages its own wandb step counter, also set `wandb_step_metric` (e.g.
`"gepa/iteration"`): the tracker then declares its own x-axis via `wandb.define_metric` instead of
passing `step=`, so its `1, 2, 3, …` steps don't collide with the host's counter (a collision makes
wandb silently drop the data).

## Custom hooks
For programmatic observation beyond metric logging, pass `engine_config={"callbacks": [...]}` —
`GEPACallback` objects receive events like `on_optimization_start`, `on_iteration_end`,
`on_candidate_accepted`, `on_proposal_end`. The eval server also always writes a flat record
(per-eval JSON + `progress_log.jsonl` + `summary.json`) under `output_dir`, whatever tracking is
configured.

See the official guide for the authoritative list of fields and behaviors:
<https://gepa-ai.github.io/gepa/guides/experiment-tracking/>.
