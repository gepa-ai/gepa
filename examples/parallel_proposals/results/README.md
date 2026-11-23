# Example offline measurements

Collected on November 22, 2026: 72 runs, two seeds per configuration. These are synthetic scheduling measurements and fixture accuracies, not real model benchmarks.

Command (from the repository root):

```bash
uv run python examples/parallel_proposals/benchmark.py \
  --workers 1 4 8 --pxn 1x1 1x4 2x4 --minibatches 2 8 \
  --eval-latency-ms 5 20 --reflection-latency-ms 10 \
  --metric-budget 192 --seeds 0 1 \
  --output /tmp/parallel-proposals-results.json
```

Environment: Python 3.13.12, macOS 26.7.3, arm64, 14 logical CPUs. GEPA base revision: `ce5b0d6e3d6af6c6dbec6de2c54536bbb961d309`. The working tree contained the benchmark; its SHA-256 is `365cf2fc11ea7faff8f18f26caa640522daa874c9402008b886934bbc796596c`.

Raw measurements: [JSON](2026-11-22-macos.json) and [CSV](2026-11-22-macos.csv).

Every row below is a median over seeds 0 and 1. All 72 runs matched the engine's metric-call count. For a fixed P×N, minibatch, latency, and seed, changing worker count preserved the best rules, consumed calls, proposal count, and accepted candidate count.

At 20 ms evaluation latency and minibatch 2, increasing workers from 1 to 8 reduced median runtime from 5.209 s to 0.996 s for 1×1 and from 5.652 s to 0.719 s for 2×4. Wider proposals were faster at 8 workers, but they consumed more evaluations and produced lower fixture accuracy at this requested budget. These results demonstrate concurrency and its limits, not a quality-equivalent prompt optimization speedup.

Two seeds are a small sample. OS scheduling and timer granularity affect measured latency, especially the 5 ms setting. See the [benchmark README](../README.md) for the workload, budget accounting, and limitations.

| Eval ms | Minibatch | Workers | P×N | Seconds | Actual calls | Overshoot | Proposals | Validation fixture | Test fixture |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 2 | 1 | 1×1 | 1.405 | 220 | 28 | 7 | 0.656 | 0.656 |
| 5 | 2 | 1 | 1×4 | 1.554 | 240 | 48 | 8 | 0.562 | 0.562 |
| 5 | 2 | 1 | 2×4 | 1.528 | 240 | 48 | 8 | 0.531 | 0.531 |
| 5 | 2 | 4 | 1×1 | 0.464 | 220 | 28 | 7 | 0.656 | 0.656 |
| 5 | 2 | 4 | 1×4 | 0.389 | 240 | 48 | 8 | 0.562 | 0.562 |
| 5 | 2 | 4 | 2×4 | 0.394 | 240 | 48 | 8 | 0.531 | 0.531 |
| 5 | 2 | 8 | 1×1 | 0.325 | 220 | 28 | 7 | 0.656 | 0.656 |
| 5 | 2 | 8 | 1×4 | 0.213 | 240 | 48 | 8 | 0.562 | 0.562 |
| 5 | 2 | 8 | 2×4 | 0.203 | 240 | 48 | 8 | 0.531 | 0.531 |
| 5 | 8 | 1 | 1×1 | 1.393 | 224 | 32 | 4 | 0.625 | 0.625 |
| 5 | 8 | 1 | 1×4 | 1.403 | 224 | 32 | 4 | 0.531 | 0.531 |
| 5 | 8 | 1 | 2×4 | 2.233 | 352 | 160 | 8 | 0.531 | 0.531 |
| 5 | 8 | 4 | 1×1 | 0.396 | 224 | 32 | 4 | 0.625 | 0.625 |
| 5 | 8 | 4 | 1×4 | 0.356 | 224 | 32 | 4 | 0.531 | 0.531 |
| 5 | 8 | 4 | 2×4 | 0.558 | 352 | 160 | 8 | 0.531 | 0.531 |
| 5 | 8 | 8 | 1×1 | 0.224 | 224 | 32 | 4 | 0.625 | 0.625 |
| 5 | 8 | 8 | 1×4 | 0.187 | 224 | 32 | 4 | 0.531 | 0.531 |
| 5 | 8 | 8 | 2×4 | 0.282 | 352 | 160 | 8 | 0.531 | 0.531 |
| 20 | 2 | 1 | 1×1 | 5.209 | 220 | 28 | 7 | 0.656 | 0.656 |
| 20 | 2 | 1 | 1×4 | 5.747 | 240 | 48 | 8 | 0.562 | 0.562 |
| 20 | 2 | 1 | 2×4 | 5.652 | 240 | 48 | 8 | 0.531 | 0.531 |
| 20 | 2 | 4 | 1×1 | 1.546 | 220 | 28 | 7 | 0.656 | 0.656 |
| 20 | 2 | 4 | 1×4 | 1.460 | 240 | 48 | 8 | 0.562 | 0.562 |
| 20 | 2 | 4 | 2×4 | 1.470 | 240 | 48 | 8 | 0.531 | 0.531 |
| 20 | 2 | 8 | 1×1 | 0.996 | 220 | 28 | 7 | 0.656 | 0.656 |
| 20 | 2 | 8 | 1×4 | 0.742 | 240 | 48 | 8 | 0.562 | 0.562 |
| 20 | 2 | 8 | 2×4 | 0.719 | 240 | 48 | 8 | 0.531 | 0.531 |
| 20 | 8 | 1 | 1×1 | 5.303 | 224 | 32 | 4 | 0.625 | 0.625 |
| 20 | 8 | 1 | 1×4 | 5.271 | 224 | 32 | 4 | 0.531 | 0.531 |
| 20 | 8 | 1 | 2×4 | 8.305 | 352 | 160 | 8 | 0.531 | 0.531 |
| 20 | 8 | 4 | 1×1 | 1.366 | 224 | 32 | 4 | 0.625 | 0.625 |
| 20 | 8 | 4 | 1×4 | 1.345 | 224 | 32 | 4 | 0.531 | 0.531 |
| 20 | 8 | 4 | 2×4 | 2.111 | 352 | 160 | 8 | 0.531 | 0.531 |
| 20 | 8 | 8 | 1×1 | 0.709 | 224 | 32 | 4 | 0.625 | 0.625 |
| 20 | 8 | 8 | 1×4 | 0.681 | 224 | 32 | 4 | 0.531 | 0.531 |
| 20 | 8 | 8 | 2×4 | 1.046 | 352 | 160 | 8 | 0.531 | 0.531 |
