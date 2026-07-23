# API reference — `optimize_anything`

The public entry point is `gepa.optimize_anything.optimize_anything`.

## Signature
```python
from gepa.optimize_anything import optimize_anything, OptimizeAnythingConfig

result = optimize_anything(
    seed_candidate: str | None = None,   # starting text; None = seedless (engine bootstraps from objective/background)
    *,
    evaluator: Callable | None = None,       # (candidate) or (candidate, example) -> (score, info)
    batch_evaluator: Callable | None = None, # list[(candidate, example)] -> one result per pair (see below)
    dataset:  list | None = None,        # training examples used DURING optimization
    valset:   list | None = None,        # validation examples used to SELECT the best candidate
    objective: str | None = None,        # short goal, surfaced verbatim to the engine
    background: str | None = None,       # long-form rules/constraints/domain notes
    test_set: list | None = None,        # reporting-only: seed + final candidate scored here at the end
    config: OptimizeAnythingConfig | None = None,
) -> Result
```
Examples in `dataset`/`valset`/`test_set` are opaque — any object your `evaluator` understands.
`seed_candidate` is a **single string** at this API (multi-component `dict[str, str]` candidates
exist only in the lower-level `gepa.gepa_launcher.optimize_anything`).

**`batch_evaluator`** — grouped scoring for when evals batch better than they stream (a provider
batch API, fan-out over your own cluster): it receives ALL `(candidate, example)` pairs of an
evaluation stage (minibatch, valset, held-out test pass) in **one call** and returns one result per
pair (`score` or `(score, info)`). At least one of `evaluator` / `batch_evaluator` is required.
When both are given, multi-pair stages use the batch function and single-pair evaluations use
`evaluator`; with only `batch_evaluator`, singles route through it as singleton batches.

### The three optimization modes (set by `dataset` / `valset`)
| mode | pass | evaluator called | use when |
|---|---|---|---|
| **Single-task** | `dataset=None, valset=None` | `evaluator(candidate)` | solving one hard problem; the candidate *is* the solution (one kernel, a packing layout, one document). |
| **Multi-task** | `dataset=<list>, valset=None` | `evaluator(candidate, example)` | one shared candidate must do well across a batch of related problems; insight transfers across them. |
| **Generalization** | `dataset=<list>, valset=<list>` | `evaluator(candidate, example)` | the candidate must transfer to **unseen** problems: optimize on `dataset`, select on `valset`. |

*Note on `valset`:* held-out selection is currently implemented only by the gepa backend (GEPA is
designed around this mode — `valset` selection plus a Pareto frontier over val instances). The
other backends fold `valset` into the pool the eval server exposes (`train_set + val_set` merged),
so they optimize over the combined set with no separate selection split — they may still produce
candidates that generalize.

**`test_set` is reporting-only and orthogonal to the mode.** Only `dataset` and `valset` affect
optimization (search and selection). If you pass `test_set`, the seed and the final chosen candidate
are scored on it *after* optimization, outside the budget, and reported in `result.metadata`. The
test set never enters the eval server's HTTP surface, so agentic backends cannot see it. Use it when
you want an unbiased number to report; skip it otherwise.

## `OptimizeAnythingConfig` (top-level fields)
| field | default | meaning |
|---|---|---|
| `engine` | `"gepa"` | `"gepa"` / `"autoresearch"` / `"meta_harness"` (or `"best_of_n"`, a baseline), or a constructed `Engine` instance (custom engines can be added via `gepa.oa.registry.register_engine`). |
| `name` | `None` | run id (logging + default output dir). Auto-generated (`<engine>-<uuid>-<timestamp>`) if `None`. |
| `max_evals` | **`100`** | server-side cap on eval calls. `None` = unlimited. Size it — the default is rarely right (see SKILL.md). |
| `max_token_cost` | `None` | USD cap on the engine's **own** optimizer-LLM spend (reflection/agent). Enforced by the engine (gepa: `max_reflection_cost` stopper; agent engines: `--max-budget-usd`), not the eval server. |
| `max_concurrency` | `8` | eval-server thread-pool size. |
| `output_dir` | `None` | where the eval server writes per-eval JSON, `progress_log.jsonl`, `summary.json`. `None` → `outputs/optimize_anything/<task>/<engine>/<timestamp>/`. |
| `run_dir` | `None` | engine workspace (gepa run dir / agent work dir; with `engine.write_agent_state=True` the gepa backend writes an agent-readable `iterations/` + `pareto/` tree here). Distinct from `output_dir`. `None` → subprocess engines use a tempdir; set it to persist artifacts. |
| `stop_at_score` | `None` | early-stop score threshold; each engine interprets it. |
| `sandbox` | **`True`** | OS-jail the subprocess engines' `claude` sessions: bwrap on Linux (needs the `bubblewrap` package — the run aborts at launch if `bwrap` is missing), Claude Code's Seatbelt sandbox on macOS. Also forces the work dir to a tempdir even when `run_dir` is set (artifacts are mirrored back). `False` prints a loud warning and runs the agent unsandboxed (`bypassPermissions`, full host access). |
| `engine_config` | `{}` | dict of **engine-specific** options — see the per-backend sections below. Parsed into a typed per-engine config dataclass; **an unknown key raises `TypeError` immediately** (fail fast, not warn-and-drop). |

**If both `max_evals` and `max_token_cost` are `None`, the run is unbounded** (only a
`warnings.warn`). Note the former fields `tracker`, `effort`, and `max_thinking_tokens` no longer
exist here: tracking is configured via the gepa backend's `engine_config["tracking"]`
(see `tracking.md`), and `effort` / `max_thinking_tokens` are per-agent-engine `engine_config` keys.

## The backends

All are selected with `engine=` and run the **same** task/evaluator — switch optimizer by
changing one argument. Each backend parses `engine_config` into its own typed dataclass, so swapping
`engine=` also means swapping the `engine_config` block (leftover keys → `TypeError`).

| engine | how it proposes candidates | runs | needs |
|---|---|---|---|
| `gepa` | an LLM reflects on evaluator feedback and mutates the candidate; keeps a Pareto frontier | in-process | reflection-LM creds (default `openai/gpt-5.1`) or a custom LM |
| `autoresearch` | one Claude Code subprocess iterates in a work dir (`program.md`, `candidate.txt`, `eval.sh` → HTTP eval server) | subprocess | `claude` on PATH + headless auth, `jq` |
| `meta_harness` | a Claude subprocess reads frontier/history and writes `pending_eval.json` candidates; the engine benchmarks each | subprocess | `claude` on PATH + headless auth |
| `best_of_n` *(baseline)* | independent single-shot samples from one LLM; keep the best — no feedback, no history | in-process | LiteLLM creds for `model` (default `claude-sonnet-4-6`) |

`scripts/preflight.py` checks a backend's prerequisites before a long run.

### `gepa` — `engine_config` is a `GEPAConfig`-shaped dict (1-to-1 passthrough)
The dict is passed to `gepa.gepa_launcher.GEPAConfig(**engine_config)`; valid top-level keys are
exactly its fields — `engine`, `reflection`, `tracking`, `merge`, `refiner`, `callbacks`,
`stop_callbacks` — and nested dicts are coerced to the matching dataclasses. The OA layer only
overlays the eval budget (`max_evals` → `engine.max_metric_calls`), `run_dir`, `stop_at_score`, and
the `max_token_cost` → `engine.max_reflection_cost` cap.

```python
engine_config={
    "reflection": {                     # -> ReflectionConfig
        "reflection_lm": "anthropic/claude-sonnet-4-6",  # default "openai/gpt-5.1"; see "Proposer LM"
        "reflection_lm_kwargs": {"reasoning_effort": "high"},  # litellm kwargs (temperature, thinking, …)
        "reflection_minibatch_size": 5,  # default: 1 single-task, 3 otherwise
        # "custom_candidate_proposer": ClaudeCodeAgentProposer(...),  # replace the reflection LM with
        #                                # a Claude Code proposer (from gepa.oa.proposers)
        # "reflection_strategy": ...,    # advanced: a ReflectionLM impl owning how reflection is called
    },
    "engine": {                          # -> EngineConfig (all optional, sensible defaults)
        "max_workers": 32,               # parallel eval workers (default: cpu_count or 32)
        "seed": 0,                       # reproducibility
        "frontier_type": "hybrid",       # "instance" | "objective" | "hybrid" (default) | "cartesian"
        "candidate_selection_strategy": "pareto",      # | "current_best" | "epsilon_greedy" | "top_k_pareto"
        "acceptance_criterion": "strict_improvement",  # | "improvement_or_equal"
        "cache_evaluation": False,       # opt-in: cache identical (candidate, example) evals
        "capture_stdio": False,          # opt-in: route evaluator print() output into feedback
        "raise_on_exception": True,      # False → evaluator exceptions become score 0 + info["error"]
        # "write_agent_state": True,     # agent-readable iterations/ + pareto/ tree under run_dir
    },
    "tracking": {"use_wandb": True},     # -> TrackingConfig (see references/tracking.md)
    "merge":  {...},                     # -> MergeConfig (cross-candidate merging) or omit
    "refiner": {...},                    # -> RefinerConfig (auto per-eval refinement) or omit
    # "callbacks": [GEPACallback, ...],  # observation hooks (on_iteration_end, on_candidate_accepted, …)
    # "stop_callbacks": [...],           # custom StopperProtocol stop conditions
}
```
The nested dataclasses live in `gepa.gepa_launcher` (`EngineConfig`, `ReflectionConfig`,
`TrackingConfig`, `MergeConfig`, `RefinerConfig`); their knobs are documented in the GEPA guides —
e.g. **candidate-selection**, **acceptance-criterion**, **batch-sampling**, **callbacks**,
**cost-tracking**, **experiment-tracking** — see <https://gepa-ai.github.io/gepa/guides/>.

### `autoresearch` — `engine_config` → `AutoResearchConfig`
| key | default | meaning |
|---|---|---|
| `model` | `"claude-sonnet-4-6"` | Claude model id (pass `"sonnet"`/`"opus"` to track the current default). |
| `ralph` | `True` | keep resuming one Claude session (`claude --resume`) while budget remains. |
| `max_no_eval_seconds` | `None` | kill the subprocess after this long with no eval call. |
| `handoffs` | `None` | prior-stage artifacts for sequential compositions (materialized under `handoff/`). |
| `effort` | `None` | `claude --effort` value. |
| `max_thinking_tokens` | `None` | fixed thinking-token budget (`MAX_THINKING_TOKENS`). |

The engine lays out a work dir (`program.md`, `candidate.txt`, `best_candidate.txt`, `eval.sh`) and
launches `claude --print`; `eval.sh` POSTs candidates to the eval server, which enforces the budget
server-side (HTTP 429 on exhaustion) and caps LLM spend via `--max-budget-usd` (from
`max_token_cost`). Train and val are presented to the agent as one combined pool; the test set is
unreachable over HTTP.

### `meta_harness` — `engine_config` → `MetaHarnessConfig`
| key | default | meaning |
|---|---|---|
| `model` | `"claude-sonnet-4-6"` | proposer model id. |
| `max_iterations` | `None` | hard cap on proposer sessions; `None` = until budget. |
| `max_candidates_per_iter` | `3` | upper bound on candidates proposed per iteration. |
| `effort` | `None` | `claude --effort` value. |
| `max_thinking_tokens` | `None` | fixed thinking-token budget. |

Each iteration the proposer subprocess reads the frontier + history state files, writes
`pending_eval.json` with 1+ candidates, and the engine benchmarks each through the eval server.

### `best_of_n` (baseline) — `engine_config` → `BestOfNConfig`
Deliberately naive: each sample is one independent LLM call — no feedback, no history, no
selection pressure beyond keep-the-best. Use it as a **comparison floor** when evaluating an
optimizer, not as the optimizer. Stops on budget exhaustion, `stop_at_score`, or `max_n`.

| key | default | meaning |
|---|---|---|
| `model` | `"claude-sonnet-4-6"` | LiteLLM model id used to sample candidates. |
| `temperature` | `1.0` | sampling on by default so N calls don't collapse to one response. |
| `max_n` | `None` | optional hard cap on samples; `None` = run until budget out. |
| `lm_kwargs` | `{}` | extra kwargs forwarded to `gepa.lm.LM`. |
| `effort` | `None` | threaded into the LM as `reasoning_effort`. |
| `max_thinking_tokens` | `None` | fixed thinking-token budget for the LM. |

## Proposer LM (not limited to LiteLLM)
`reflection.reflection_lm` (gepa) accepts either:
- a **model-id string** resolved through LiteLLM (`"openai/gpt-5.1"` — the default —
  `"anthropic/claude-sonnet-4-6"`, a Bedrock inference-profile ARN, …); set the provider's
  credentials, and pass litellm kwargs via `reflection.reflection_lm_kwargs`; or
- **any object/callable implementing GEPA's LM protocol** — `__call__(prompt) -> str` (a
  `str`-or-messages prompt in, completion text out). This is how you plug a **self-hosted or custom
  inference engine** (vLLM, a local server, your own client) instead of LiteLLM.
  `reflection_lm_kwargs` is ignored for callables.

## Composition / pipeline helpers (ensembles)
`gepa.optimize_anything` also exports helpers that compose several backends over the same
task/evaluator. Each takes the same flat task fields as `optimize_anything` (`seed_candidate`
positional; `evaluator=`, `dataset=`, `valset=`, `objective=`, `background=`, `test_set=`, `name=`
keywords) plus `configs=` — a list of `OptimizeAnythingConfig`, one per stage/branch, each carrying
its own budget. Budgets are pre-partitioned per config; an early-finishing stage's leftover is not
redistributed.

- `optimize_sequential` — **pipeline**: run configs in order; each stage's best becomes the next
  stage's seed. Monotonic — the running best is carried forward, so a regressing stage doesn't
  poison the chain.
- `optimize_parallel` — run all configs concurrently; returns the list of `Result`s (config order).
- `optimize_best_of` — parallel, then keep the result with the highest `best_score` (each engine's
  own number — no re-scoring).
- `optimize_vote` — parallel, then re-score each branch's best candidate once via `evaluator`
  (outside any budget) and return the highest re-score. Use when engines score with different
  quirks and you want a fair cross-engine comparison.
- `optimize_adaptive_sequential` — **plateau scheduler**: give the active engine bounded eval
  slices (`plateau_evals`) and rotate to the next config after `patience` non-improving slices,
  always feeding the running best forward. The exception to per-config budgets: one shared eval
  pool set by its `max_evals` keyword (per-config `max_evals` is ignored; per-config
  `max_token_cost` still applies). Extra knobs: `min_evals_per_stage`, `improvement_epsilon`,
  `cycle`, `max_switches`.

```python
from gepa.optimize_anything import optimize_sequential, OptimizeAnythingConfig

result = optimize_sequential(
    SEED,
    evaluator=evaluate,
    dataset=trainset, valset=valset,
    objective="...",
    configs=[
        OptimizeAnythingConfig(engine="autoresearch", max_evals=100, max_token_cost=5.0),
        OptimizeAnythingConfig(engine="gepa", max_evals=200),  # refines autoresearch's best
    ],
)
```

`*_with_server` variants (`optimize_sequential_with_server`, `optimize_parallel_with_server`, …)
take caller-owned `EvalServer`s — one per config (one shared server for
`optimize_adaptive_sequential_with_server`) — for embedding in outer frameworks that route every
eval through their own server. Related: the autoresearch backend's `handoffs` key materializes
prior-stage artifacts into the agent's work dir for hand-rolled sequential compositions.

## `Result`
```python
result.best_candidate   # str
result.best_score       # float (on valset/selection set)
result.total_evals      # int
result.eval_log         # list[dict]
result.metadata         # dict (verified keys):
#   "gepa_result"            full GEPAResult (all candidates + per-instance val scores) — gepa engine only
#   "test_score"             avg over test_set         } only present
#   "test_scores"            per-example dict          } if you passed
#   "baseline_test_score"    seed avg over test_set    } a test_set
#   "baseline_test_scores"   seed per-example dict     }
#   "budget", "total_cost", "adapter_cost", "wall_time", "engine", "output_dir", "progress_log"
```
The gepa engine also writes `run_dir/` artifacts, and the eval server writes
`output_dir/summary.json`. The `gepa_result` in metadata is the richest artifact — keep it for
post-hoc analysis. (If the best candidate equals the seed, the test scores are reported from the
seed's single scoring pass rather than re-scored.)
