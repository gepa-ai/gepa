# Benchmark Results

## 2024-11-11 — OG GEPA (Classic)

- **Script**: `examples/aime_benchmark_v2.py --mode gepa`
- **Models**: task `openrouter/openai/gpt-oss-20b:nitro`, reflection `openrouter/x-ai/grok-4-fast`
- **Dataset**: AIME subset (train=30, val=30)
- **Max metric calls**: 150
- **Runtime**: **598.7 s** (~10.0 min)
- **Evaluations**: 150
- **Best quality**: 0.733 (baseline was 0.0 before optimization)
- **Best prompt snapshot**: “You are an expert assistant specialized in solving American Invitational Mathematics Examination (AIME) problems…” (see `og_gepa_benchmark.log` for full text)

Raw console output is saved in `og_gepa_benchmark.log`.

## 2024-11-11 — TurboGEPA (Clean Run)

- **Script**: `examples/aime_benchmark_v2.py --mode turbo --turbo-eval-concurrency 2`
- **Models**: task `openrouter/openai/gpt-oss-20b:nitro`, reflection `openrouter/x-ai/grok-4-fast`
- **Dataset**: AIME subset (train=30, val=30)
- **Max evaluations**: 150 (43 used)
- **Runtime**: **170.5 s**
- **Best full-shard quality**: 0.767 (seed hit 76.7 % on the 30-example shard; later mutations underperformed)
- **Best prompt snapshot**: “You are an expert in advanced mathematical problem-solving, specializing in complex numbers, roots of unity, algebraic manipulations…” (saved to `.turbo_gepa/best_prompt.txt`)

Artifacts: `turbo_gepa_benchmark_clean.log` (console log) and `.turbo_gepa/metrics/metrics_1762894099.txt`.

## 2024-11-11 — Prompt Verification (Train Split)

- **Script**: `scripts/verify_prompt.py --split train --dataset-size 30 --eval-concurrency 8`
- **Dataset**: AIME train subset (30 problems)
- **Result**: Accuracy **0.73 (22/30)**, coverage = 100 %. The remaining misses stem from combinatorics/base-conversion tasks (see JSON for IDs).
- **Artifacts**: `.turbo_gepa/verification/train_verification.json`.
