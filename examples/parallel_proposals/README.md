# Offline parallel proposal benchmark

Run a reproducible latency sweep through `gepa.optimize()` to explore how P×N,
reflection minibatch size, evaluation latency, and worker count interact. This
complements the [parallel proposal guide](https://gepa-ai.github.io/gepa/guides/parallel-proposals/)
and the runtime measurements requested in [#438](https://github.com/gepa-ai/gepa/issues/438).

## Run

From the repository root, install GEPA and run the benchmark:

```bash
uv sync
uv run python examples/parallel_proposals/benchmark.py \
  --workers 1 4 8 --pxn 1x1 1x4 2x4 --minibatches 2 8 \
  --eval-latency-ms 5 20 --reflection-latency-ms 10 \
  --metric-budget 192 --seeds 0 1 2 \
  --output /tmp/parallel-proposals.json
```

The script writes JSON and CSV after each completed run. JSON includes the Git
revision, whether the working tree has changes, the script's hash, and Python,
OS, and CPU information. The sweep order is shuffled with a fixed seed to reduce
correlation between a configuration and machine drift. Each run has a fresh
optimizer and worker pool.

For a quick smoke run:

```bash
uv run python examples/parallel_proposals/benchmark.py \
  --workers 1 2 --pxn 1x1 1x2 --minibatches 2 \
  --eval-latency-ms 0 --reflection-latency-ms 0 --metric-budget 24 \
  --train-size 8 --val-size 8 --test-size 8 --categories 8 --seeds 0 \
  --output /tmp/parallel-proposals-smoke.json
```

## What the benchmark measures

The benchmark uses GEPA's actual sampling, reflection, minibatch acceptance,
validation, and candidate selection paths. Its adapter implements
`batch_evaluate`; its reflector implements `reflect_many`. They share one bounded
`ThreadPoolExecutor`. Evaluation schedules one task per candidate/example pair;
reflection schedules one task per proposal. Results preserve submission order.
Mutations are planned before worker dispatch, so changing worker count does not
change the optimization trace for a fixed configuration and seed.

Each evaluation sleeps for `--eval-latency-ms` and each reflection sleeps for
`--reflection-latency-ms`. These delays model waiting for external services.
GEPA still executes evaluation and reflection stages sequentially; the benchmark
does not add a pipeline or background optimizer. A wider P×N creates more
independent work within those stages. The `1x1` case is the serial proposal
baseline, with the same adapter and evaluation worker count.

The local task learns a category-to-label mapping from feedback. A reflection
corrects one wrong category observed in its minibatch. Labels, split ordering,
mutation choices, and GEPA's seed are deterministic. Train, validation, and test
examples have distinct IDs and follow the same finite category rules. The test
split is scored once after selecting the best validation candidate, outside
optimization timing and its metric budget.

No provider, API key, network request, or paid model call is used. This is a
synthetic scheduling benchmark. Its validation and test accuracy fields check
the fixture; they do **not** establish prompt quality or reproduce the published
LiveBench/HoVer results. CPU-bound tasks, variable service latency, token costs,
rate limits, and real model reflection can behave differently.

## Compare results

Keep dataset sizes, requested metric budget, evaluation/reflection latency, and
seeds fixed when comparing configurations. Report medians across seeds and
retain the per-run rows. A useful comparison includes:

| Field | Meaning |
| --- | --- |
| `wall_seconds` | Total `gepa.optimize` duration, excluding test scoring and file writing |
| `evaluation_wall_seconds` | Time spent inside the adapter's evaluation batches |
| `reflection_wall_seconds` | Time spent inside the reflector's batches |
| `actual_metric_calls` | Candidate/example evaluations observed by the adapter |
| `engine_metric_calls` | GEPA's reported metric-call count |
| `budget_overshoot` | Actual calls beyond the requested budget |
| `metric_calls_per_second` | Actual calls divided by optimization duration |
| `proposals` | Reflections attempted, including proposals not accepted |
| `accepted_candidates` | Candidates added after the seed |
| `validation_fixture_accuracy` | Selected candidate's validation score |
| `test_fixture_accuracy` | Selected candidate's accuracy on the local test fixture |

GEPA checks its metric budget between iterations. A wider proposal batch can
overshoot further, so requested budgets are equal but actual work may differ.
Consider elapsed time together with actual calls, throughput, proposals, and
fixture accuracy. Do not treat a runtime ratio alone as a quality-equivalent
speedup. Evaluation caching and perfect-score skipping are disabled here, making
the workload and accounting visible.

For real-task evidence, replace `LocalAdapter` and `LocalReflection` with the
task's evaluator and reflection model while preserving these measurements.
Report held-out task scores and model/token costs as well as latency and actual
evaluation counts. The offline results alone cannot select the best P×N for
that task.
