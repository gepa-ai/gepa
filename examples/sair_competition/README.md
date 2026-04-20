# SAIR Equational Theories — Cheatsheet Optimization

Uses GEPA's `optimize_anything` to evolve a cheatsheet for the [SAIR Mathematics Distillation Challenge: Equational Theories Stage 1](https://competition.sair.foundation/competitions/mathematics-distillation-challenge-equational-theories-stage1).

## Task

Given two equations over magmas, determine whether Equation 1 implies Equation 2 (TRUE/FALSE). The cheatsheet is inserted into a fixed prompt template and evaluated across three models with equal weight:

- `openai/gpt-oss-120b`
- `meta-llama/llama-3.3-70b-instruct`
- `google/gemma-4-31b-it`

GEPA optimizes the cheatsheet text — the template (task description + output format) is fixed.

## Setup

```bash
# Install dependencies
pip install "gepa[full]" httpx datasets

# Set API keys
export OPENROUTER_API_KEY="sk-or-..."   # for model evaluation
export OPENAI_API_KEY="sk-..."          # for reflection LM (gpt-4.1-mini)
export WANDB_API_KEY="..."              # optional, for experiment tracking
```

## Usage

```bash
# 1. Prepare train/val splits (downloads from HuggingFace)
uv run python -m examples.sair_competition.prepare_data

# 2. Run optimization
uv run python -m examples.sair_competition.main
```

## Data

`prepare_data.py` downloads the public problem subsets from [HuggingFace](https://huggingface.co/datasets/SAIRfoundation/equational-theories-selected-problems) and builds:

- **Valset (300 problems × 3 models = 900 examples)**: Balanced 50/50 TRUE/FALSE, sampled proportionally across difficulty levels (normal, hard1, hard2, hard3). Designed to approximate the private final evaluation distribution.
- **Trainset (1369 problems × 3 models = 4107 examples)**: All remaining problems.

The train/val split is done before model expansion — no equation pair appears in both splits.

## Outputs

- `run/` — GEPA run directory (state, checkpoints, logs). Allows resuming interrupted runs.
- `best_cheatsheet.txt` — The optimized cheatsheet text.
- `submission_prompt.txt` — Complete prompt with `{{equation1}}`/`{{equation2}}` placeholders, ready for submission.
- Wandb dashboard — Live optimization metrics.
