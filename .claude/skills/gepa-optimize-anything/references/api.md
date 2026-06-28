# API reference — `optimize_anything`

The public entry point is `gepa.optimize_anything.optimize_anything`.

## Signature
```python
from gepa.optimize_anything import optimize_anything, OptimizeAnythingConfig

result = optimize_anything(
    seed_candidate: str | None = None,   # starting text; None = seedless (engine bootstraps from objective/background)
    *,
    evaluator: Callable[..., tuple[float, dict]],  # (candidate) or (candidate, example) -> (score, info)
    dataset:  list | None = None,        # training examples used DURING optimization
    valset:   list | None = None,        # validation examples used to SELECT the best candidate
    objective: str | None = None,        # short goal, surfaced verbatim to the engine
    background: str | None = None,       # long-form rules/constraints/domain notes
    test_set: list | None = None,        # reporting-only: seed + final candidate scored here at the end
    config: OptimizeAnythingConfig | None = None,
) -> Result
```
Examples in `dataset`/`valset`/`test_set` are opaque — any object your `evaluator` understands.
`seed_candidate` may be a `str` **or a `dict[str, str]`** of named components to optimize jointly
(the evaluator then receives the dict).

### The three optimization modes (set by `dataset` / `valset`)
| mode | pass | evaluator called | use when |
|---|---|---|---|
| **Single-task** | `dataset=None, valset=None` | `evaluator(candidate)` | solving one hard problem; the candidate *is* the solution (one kernel, a packing layout, one document). |
| **Multi-task** | `dataset=<list>, valset=None` | `evaluator(candidate, example)` | one shared candidate must do well across a batch of related problems; insight transfers across them. |
| **Generalization** | `dataset=<list>, valset=<list>` | `evaluator(candidate, example)` | the candidate must transfer to **unseen** problems: optimize on `dataset`, select on `valset`. |

**`test_set` is reporting-only and orthogonal to the mode.** Only `dataset` and `valset` affect
optimization (search and selection). If you pass `test_set`, GEPA scores the **seed** and the **final
chosen candidate** on it *after* optimization and reports them in `result.metadata` — it never enters
the search, selection, or budget. Use it when you want an unbiased number to report; skip it otherwise.

## `OptimizeAnythingConfig` (top-level fields)
| field | meaning |
|---|---|
| `engine` | `"gepa"` (default) / `"best_of_n"` / `"autoresearch"` / `"meta_harness"`, or a constructed `Engine`. |
| `name` | run id (logging + default output dir). Auto-generated if `None`. |
| `max_evals` | server-side cap on eval calls. `None` = unlimited. |
| `max_token_cost` | USD cap on the engine's **own** optimizer-LLM spend (reflection/agent). Enforced by the engine, not the server. |
| `max_concurrency` | eval-server thread-pool size. |
| `output_dir` | where the server writes per-eval JSON, `progress_log.jsonl`, `summary.json`. |
| `tracker` | optional **thin** experiment tracker (see `tracking.md`) — NOT GEPA's native wandb. |
| `run_dir` | engine workspace (`iterations/`, `pareto/`, agent state). Distinct from `output_dir`. |
| `stop_at_score` | early-stop score threshold. |
| `effort` | `claude --effort low|medium|high|max` for Claude-backed engines/proposers. |
| `max_thinking_tokens` | fixed thinking-token budget for the proposer LM. |
| `sandbox` | wrap subprocess engines in bwrap (Linux). Default `False`. |
| `engine_config` | dict of **engine-specific** options (see below). Unknown keys are warn-and-dropped. |

**At least one of `max_evals` / `max_token_cost` must be set**, else the run is unbounded (only a
`warnings.warn`).

## `engine_config` for the `gepa` engine
The GEPA engine reads only these keys (`_GEPA_CONFIG_KEYS`): `engine`, `reflection`, `merge`,
`refiner`, `objective`, `background`, `reflection_lm_kwargs`, `callbacks`, `claude_code_agent`.
`engine` and `reflection` are **opaque kwarg dicts** forwarded to core dataclasses:

```python
engine_config={
    "reflection": {                     # -> ReflectionConfig
        "reflection_lm": "anthropic/claude-sonnet-4-6",  # see "Proposer LM" below
        "reflection_minibatch_size": 5,
    },
    "engine": {                          # -> EngineConfig (all optional, sensible defaults)
        "max_workers": 32,
        "seed": 0,                       # reproducibility
        "frontier_type": "hybrid",       # multi-objective Pareto (see writing_evaluators.md)
        "candidate_selection_strategy": ...,           # guide: candidate-selection
        "acceptance_criterion": "strict_improvement",  # guide: acceptance-criterion
        "cache_evaluation": True,        # cache identical (candidate, example) evals
    },
    "merge":  {...},                     # -> MergeConfig (cross-candidate merging) or omit
    "refiner": {...},                    # -> RefinerConfig (auto per-eval refinement) or omit
    "tracking": {"use_wandb": True},     # -> TrackingConfig (see references/tracking.md)
    # "callbacks": [GEPACallback, ...],  # guide: callbacks
    # "claude_code_agent": {...},        # replace the reflection LM with a Claude Code subprocess proposer
}
```
The `engine`/`reflection`/etc. dicts map onto core dataclasses (`gepa.gepa_launcher`: `EngineConfig`,
`ReflectionConfig`, `MergeConfig`, `RefinerConfig`, `TrackingConfig`). Their knobs are documented in
the GEPA guides — e.g. **candidate-selection**, **acceptance-criterion**, **batch-sampling**,
**callbacks**, **cost-tracking**, **experiment-tracking** — see <https://gepa-ai.github.io/gepa/guides/>.

### Proposer LM (not limited to LiteLLM)
`reflection_lm` accepts either:
- a **model-id string** resolved through LiteLLM (`"anthropic/claude-sonnet-4-6"`,
  `"openai/gpt-5"`, a Bedrock inference-profile ARN, …) — the convenience default; set the
  provider's credentials; or
- **any object/callable implementing GEPA's LM protocol** — `__call__(prompt) -> str` (a
  `str`-or-messages prompt in, completion text out). This is how you plug a **self-hosted or custom
  inference engine** (vLLM, a local server, your own client) instead of LiteLLM.

## Engines and their prerequisites
All four are selected with `engine=` and run the **same** task/evaluator — switch optimizer by
changing one argument. `optimize_anything`'s purpose is this common interface for comparing them.
| engine | how it proposes candidates | needs |
|---|---|---|
| `gepa` | an LLM reflects on evaluator feedback and mutates the candidate; keeps a Pareto frontier | proposer-LM creds (or a custom LM) |
| `best_of_n` | samples N candidates from a model and keeps the best | LLM creds (`model`, `temperature`, `max_n`, `lm_kwargs`) |
| `autoresearch` | an agentic optimizer driving a `claude` CLI subprocess (researcher-style iteration) | `claude` on PATH + headless auth, `run_dir` |
| `meta_harness` | an agentic proposer (`claude` subprocess) that reads frontier/history and writes candidates | `claude` on PATH + headless auth, `run_dir` |

Each engine reads its own keys from `engine_config` (`_<ENGINE>_CONFIG_KEYS`); unrecognized keys are
warned and ignored, so swapping `engine=` also means swapping the `engine_config` block.
`scripts/preflight.py` checks an engine's prerequisites before a long run.

## `Result`
```python
result.best_candidate   # str
result.best_score       # float (on valset/selection set)
result.total_evals      # int
result.eval_log         # list[dict]
result.metadata         # dict (verified keys):
#   "gepa_result"            full GEPAResult (all candidates + per-instance val scores)
#   "test_score"             avg over test_set         } only present
#   "test_scores"            per-example dict          } if you passed
#   "baseline_test_score"    seed avg over test_set    } a test_set
#   "baseline_test_scores"   seed per-example dict     }
#   "budget", "total_cost", "adapter_cost", "wall_time", "engine", "output_dir",
#   "progress_log", "reflection_cost_initial"/"reflection_cost_final"/"reflection_cost_log"
```
GEPA also writes `run_dir/iterations/<uuid>/`, `run_dir/pareto/`, and `output_dir/summary.json`.
The `gepa_result` in metadata is the richest artifact — keep it for post-hoc analysis.
