# Offline Artifacts for Zero-API Review

This note is for ACM artifact reviewers who want to inspect saved evidence without spending API credits.

## Strong Offline Domains

These domains already include concrete saved trajectories or checkpoints:

- `domains/aime_math/logs/`
  - `run.log` records the paper-matching improvement from `46.67%` to `60.00%`.
  - `best_prompt.txt` contains the final optimized prompt.
- `domains/arc_agi/logs/`
  - `test_run.log` records `32.5% -> 89.5%`.
  - `best_agent.py`, `all_candidates.json`, and `test_results.json` expose the evolved agent and trajectory.
- `domains/blackbox/logs/`
  - Per-problem `gepa_state.bin`, `eval_log.jsonl`, and Optuna comparison outputs are bundled.
- `domains/circle_packing/logs/`
  - `state_tracker_logs.json` contains per-iteration scores and solver code snapshots.
- `domains/cloud_scheduling/cloudcast/offline_logs/cloudcast_output.log`
  - Saved optimization log from a late-stage CloudCast run.
  - Shows score improvements and proposed solver code without requiring rerun.
- `domains/gskill/offline_runs/`
  - Saved GEPA training runs for `blevesearch__bleve` and `pallets__jinja`.
  - Saved Claude Code post-hoc evaluations with and without learned skills.

## Recommended Review Flow

1. Read the domain `README.md`.
2. Inspect the saved logs/checkpoints listed above.
3. Only if needed, try the live rerun path from the domain README.

## Quick Numbers Reviewers Can Verify Offline

### AIME

In `domains/aime_math/logs/run.log`:

- `Iteration 0: Base program full valset score: 0.4666666666666667`
- `Iteration 33: Best valset aggregate score so far: 0.5777777777777777`
- `Average Metric: 14.0 / 30 (46.7%)`
- `Average Metric: 18.0 / 30 (60.0%)`
- `Test score improved from 46.67% to 60.00%!`

This single `run.log` contains both the validation trajectory and the paper-matching test-improvement line.

### ARC-AGI

In `domains/arc_agi/logs/run.log`:

- `Iteration 0: Base program full valset score: 0.565 over 200 / 200 examples`
- `Iteration 30: Best valset aggregate score so far: 0.935`

In `domains/arc_agi/logs/test_run.log`:

- `Seed: 130/400 solved (32.5%), cost=$26.65`
- `Best: 358/400 solved (89.5%), cost=$57.82`
- `Seed:  130/400 (32.5%)`
- `Best:  358/400 (89.5%)`
- `Δ:     +57.0%`

For ARC-AGI, the validation trajectory is in `run.log`, while the headline paper claim is in `test_run.log`.

### CloudCast

In `domains/cloud_scheduling/cloudcast/offline_logs/cloudcast_output.log`:

- `Iteration 143: Base program full valset score: 0.00519955683867755`
- `Iteration 165: Found a better program on the valset with score 0.009008762504180836`
- `Iteration 165: Objective aggregate scores for new program: {'cost_score': 0.009008762504180836, 'raw_cost': 128.8043592}`
- `Iteration 147: Objective pareto front scores: {'cost_score': 0.008792180808885988, 'raw_cost': 209.172081}`

This gives a concrete offline trajectory showing improvement in both score and raw cost.

### gskill Training

In `domains/gskill/offline_runs/gepa_skills_training/`:

- `run_blevesearch_20260131_131944_d7b877/summary.json`
  - baseline test score `0.19`
  - optimized test score `0.85`
  - metric calls `300`
- `run_pallets_20260131_152447_4347c2/summary.json`
  - baseline test score `0.38`
  - optimized test score `0.59`
  - metric calls `307`

### Claude Code Evaluation of Learned Skills

In `domains/gskill/offline_runs/claude_code_eval/`:

- `run_blevesearch_haiku/summary.json`: `46 / 58` passed
- `run_blevesearch_haiku_with_real_skills/summary.json`: `58 / 58` passed
- `run_blevesearch_sonnet/summary.json`: `55 / 58` passed
- `run_blevesearch_sonnet_with_real_skills/summary.json`: `58 / 58` passed
- `run_pallets_haiku/summary.json`: `62 / 66` passed
- `run_pallets_haiku_with_real_skills/summary.json`: `65 / 66` passed
- `run_pallets_sonnet/summary.json`: `66 / 66` passed
- `run_pallets_sonnet_with_real_skills/summary.json`: `66 / 66` passed

Each run also includes `results.jsonl`, `test_ids.txt`, and the exact skill text when applicable.

## What Is Still Missing

These areas are less complete as offline-only artifacts:

- `domains/cloud_scheduling/can_be_late/`
  - Has code, traces, and paper plots.
  - Does not yet include a comparably rich saved GEPA trajectory bundle like CloudCast.
- `domains/cloud_scheduling/cloudcast/`
  - Now has a saved run log, but not yet a full `gepa_state.bin` checkpoint bundle in this artifact subfolder.
- `domains/gskill/`
  - Offline evidence is now strong for analysis, but full rerun still depends on Docker plus external model access.
- `domains/kernelbench/`
  - Logs are bundled, but performance reproduction still depends on the right GPU.
