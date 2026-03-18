# ConfidenceAdapter vs DefaultAdapter — Classification Benchmark

Reproducible comparison of two GEPA adapters for LLM text classification:

- **DefaultAdapter** — binary scoring (correct = 1.0, wrong = 0.0)
- **ConfidenceAdapter** — logprob-based confidence scoring + richer reflection feedback via [`llm-structured-confidence`](https://github.com/rodolfonobrega/llm-structured-confidence)

Both adapters get identical data, budget, and reflection model. The only difference is how each adapter scores predictions and what feedback it passes to the reflection loop.

For a detailed write-up of the methodology, analysis, and charts, see the **[blog post](https://gepa-ai.github.io/gepa/blog/2026/03/17/confidence-adapter-benchmark/)**.

## Datasets

| Dataset | Type | Classes | Train/class | Val/class | Test total |
|---------|------|---------|-------------|-----------|------------|
| AG News | Multiclass | 4 | 120 | 40 | 2000 |
| Emotion | Multiclass | 6 | 120 | 40 | 1390 |
| Rotten Tomatoes | Binary | 2 | 120 | 40 | 1066 |

## Quick start

From the repo root (`gepa/`):

```bash
uv venv
uv pip install -e ".[confidence]"
uv pip install datasets matplotlib scikit-learn python-dotenv
```

```bash
export OPENAI_API_KEY=...
export OPENAI_API_BASE=...   # if using a proxy/gateway
PYTHONUNBUFFERED=1 uv run python -m examples.confidence_adapter.main
```

The script runs end-to-end (~2–3 hours with API calls) and writes all outputs to `outputs/`.

## Results (test accuracy)

| Dataset | Baseline | DefaultAdapter | ConfidenceAdapter | Delta |
|---------|----------|----------------|-------------------|-------|
| AG News | 83.70% | 85.80% | **87.90%** | **+2.10pp** |
| Emotion | 54.03% | 58.42% | **60.22%** | **+1.80pp** |
| Rotten Tomatoes | 91.46% | 93.15% | 93.15% | 0.00pp |

## Pre-computed reference data

The `outputs/` directory contains pre-computed results from our run, so you can inspect them without re-running the experiment:

```
outputs/
├── summaries.json       # per-dataset accuracy summary
├── optimization/        # GEPA optimization results (best prompt, convergence history)
│   ├── ag_news_default.json
│   ├── ag_news_confidence.json
│   └── ...
└── evaluation/          # side-by-side predictions (text, expected, predicted, probability)
    ├── ag_news_combined.csv
    ├── emotion_combined.csv
    └── rotten_tomatoes_combined.csv
```

When you run `main.py`, it regenerates everything including charts in `outputs/charts/`.
