# ConfidenceAdapter vs DefaultAdapter — Classification Benchmark

Compare two GEPA adapters for LLM text classification: the default binary scorer and a confidence-aware scorer that uses token-level log-probabilities.

## How it works

- **DefaultAdapter** — binary scoring: correct = 1.0, wrong = 0.0
- **ConfidenceAdapter** — extracts per-token logprobs via [`llm-structured-confidence`](https://github.com/rodolfonobrega/llm-structured-confidence), scores with `LinearBlendScoring` (continuous 0–1 scale), and sends the reflection LLM tiered feedback that includes probability values and top alternatives

Both adapters get **identical data, budget, and reflection model**. The only difference is how each adapter scores predictions and what feedback it passes to the reflection loop.

### Why confidence scoring helps

With binary scoring the optimizer sees no difference between a 51%-confident correct answer and a 99%-confident one — both score 1.0. ConfidenceAdapter fixes this by:

1. **Continuous scoring** — correct answers below a configurable threshold (default 0.99) receive proportionally lower scores, giving GEPA a gradient to distinguish "barely correct" from "confidently correct."
2. **Rich feedback** — high-conviction errors get strong signals (*"WRONG — model has 99% certainty on 'Business' but the correct answer is 'Sci/Tech'. The prompt must add explicit disambiguation rules."*), while low-confidence correct answers get a warning about fragility.
3. **Threshold calibration** — modern LLMs with enum-constrained structured output produce very high probabilities (typically 95–100%) even for uncertain answers. A 0.99 `high_confidence_threshold` creates meaningful score gradients in this regime.

## Datasets

| Dataset | Type | Classes | Train/class | Val/class | Test total |
|---------|------|---------|-------------|-----------|------------|
| AG News | Multiclass | 4 | 120 | 40 | 2000 |
| Emotion | Multiclass | 6 | 120 | 40 | 1390* |
| Rotten Tomatoes | Binary | 2 | 120 | 40 | 1066* |

\*Target was 2000 balanced examples. Emotion and Rotten Tomatoes have smaller HuggingFace test splits, so we used as many balanced examples as available.

## Setup

From the repo root (`gepa/`):

```bash
uv venv
uv pip install -e ".[confidence]"
uv pip install datasets matplotlib scikit-learn python-dotenv
```

## Run

```bash
export OPENAI_API_KEY=...
export OPENAI_API_BASE=...   # if using a proxy/gateway
PYTHONUNBUFFERED=1 uv run python -m examples.confidence_adapter.main
```

The script runs end-to-end (~2–3 hours with API calls) and writes all outputs to `examples/confidence_adapter/outputs/`.

Pre-computed results from our run are included in `outputs/` for reference.

## Results

### Test-Set Accuracy

| Dataset | Baseline | DefaultAdapter | ConfidenceAdapter | Delta (Conf − Default) |
|---------|----------|----------------|-------------------|------------------------|
| AG News | 83.70% | 85.80% | **87.90%** | **+2.10pp** |
| Emotion | 54.03% | 58.42% | **60.22%** | **+1.80pp** |
| Rotten Tomatoes | 91.46% | 93.15% | 93.15% | 0.00pp |

### Weighted-Average Metrics

| Dataset | Condition | Precision | Recall | F1 |
|---------|-----------|-----------|--------|-----|
| AG News | Baseline | 86.47% | 83.70% | 83.02% |
| | DefaultAdapter | 87.31% | 85.80% | 85.59% |
| | **ConfidenceAdapter** | **87.83%** | **87.90%** | **87.84%** |
| Emotion | Baseline | 56.05% | 54.03% | 53.03% |
| | DefaultAdapter | 58.66% | 58.42% | 58.16% |
| | **ConfidenceAdapter** | **61.75%** | **60.22%** | **60.30%** |
| Rotten Tomatoes | Baseline | 91.55% | 91.46% | 91.46% |
| | DefaultAdapter | 93.16% | 93.15% | 93.15% |
| | ConfidenceAdapter | 93.19% | 93.15% | 93.15% |

### Key Findings

**AG News (4-class):** The largest impact was on the Sci/Tech ↔ Business confusion axis. The baseline had strong bias — it aggressively predicted Business (95.2% recall) at the expense of Sci/Tech (52.8% recall). ConfidenceAdapter resolved this: Sci/Tech recall jumped to 81.8% and Business precision rose to 84.9%. Per-class F1 converged to ~83.5% for both classes. The confidence-aware feedback prompted the reflection LLM to add rules like *"Technology company news about SOFTWARE, HARDWARE, or TECH PRODUCTS → Sci/Tech, NOT Business, even if financial aspects are mentioned."*

**Emotion (6-class):** ConfidenceAdapter improved F1 on 5 of 6 classes, with the largest gains on minority classes: love (+10.4pp), surprise (+7.2pp), anger (+2.9pp). The one lower class (joy: 61.7% → 57.7% F1) reflects a generalization tradeoff — the optimizer learned stricter joy criteria to avoid confusing it with love/surprise, producing a more balanced classifier overall.

**Rotten Tomatoes (binary):** Both adapters tied at 93.15%. For binary classification with clearly distinct categories, the confidence signal has limited room to differentiate — the model already reaches high probabilities on both classes.

### Charts

All charts are in [`outputs/charts/`](outputs/charts/). Key visualizations:

| Chart | Description |
|-------|-------------|
| `accuracy_comparison.png` | Grouped bar chart: baseline vs default vs confidence accuracy |
| `metrics_comparison.png` | Precision / Recall / F1 (weighted) for all conditions |
| `convergence_*.png` | Validation score per candidate — shows optimization convergence |
| `per_class_recall_*.png` | Per-class recall (multiclass datasets) |
| `per_class_f1_*.png` | Per-class F1 (multiclass datasets) |
| `roc_rotten_tomatoes.png` | ROC curves for binary sentiment classification |
| `confidence_distribution.png` | ECDF of probability scores: correct vs incorrect predictions |

### When to use ConfidenceAdapter

**Use it when:**
- The task has enum-constrained structured output and the model exposes logprobs
- Categories are confusable (e.g., fine-grained topics, emotions)
- You want the optimizer to generate targeted disambiguation rules

**Stick with DefaultAdapter when:**
- The task is simple binary classification with clearly distinct categories
- The model does not support logprobs
- You need minimal API overhead

## Output structure

```
outputs/
├── summaries.json              # per-dataset accuracy and scores
├── datasets/                   # train/val/test CSVs (generated)
├── optimization/               # GEPA results: best prompt, convergence history
│   ├── ag_news_default.json
│   ├── ag_news_confidence.json
│   └── ...
├── evaluation/                 # side-by-side prediction CSVs
│   ├── ag_news_combined.csv
│   └── ...
└── charts/                     # all PNG visualizations
```
