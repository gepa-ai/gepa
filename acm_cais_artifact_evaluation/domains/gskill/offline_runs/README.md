# Offline gskill Runs

This directory contains two kinds of saved offline artifacts:

- `gepa_skills_training/` — saved GEPA skill-optimization runs
- `claude_code_eval/` — saved Claude Code evaluations on held-out bug-fixing tasks

## Training Runs

### `run_blevesearch_20260131_131944_d7b877`

- Repo: `blevesearch__bleve`
- Baseline test score: `0.19`
- Optimized test score: `0.85`
- Best score during search: `0.84`
- Metric calls: `300`

### `run_pallets_20260131_152447_4347c2`

- Repo: `pallets__jinja`
- Baseline test score: `0.38`
- Optimized test score: `0.59`
- Best score during search: `0.76`
- Metric calls: `307`

## Claude Code Evaluations

These runs compare task success with and without learned skills.

### `blevesearch__bleve`

- Haiku without skills: `46 / 58`
- Haiku with learned skills: `58 / 58`
- Sonnet without skills: `55 / 58`
- Sonnet with learned skills: `58 / 58`

### `pallets__jinja`

- Haiku without skills: `62 / 66`
- Haiku with learned skills: `65 / 66`
- Sonnet without skills: `66 / 66`
- Sonnet with learned skills: `66 / 66`

## Files to inspect

- `summary.json` — aggregate metrics
- `results.jsonl` — per-instance records
- `source_config.json` — evaluation configuration
- `skills_raw.txt` / `SKILL.md` — learned skill text
- `terminal.log` — training trace
- `proposer_calls/` — detailed reflection/proposal calls
