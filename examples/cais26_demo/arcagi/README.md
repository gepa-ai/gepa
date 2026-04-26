# ARC-AGI × GEPA — CAIS'26 Demo Walkthrough

A self-contained Jupyter notebook showing how GEPA's `optimize_anything` API
evolved a 12-line ARC-AGI seed agent into an 89.5%-accuracy code-synthesis
solver — with no human in the loop.

## 30-second pitch

- **Seed agent:** 12 lines, one LLM call, predicts grids directly. Paper accuracy: **32.5%**.
- **Evolved agent (GEPA):** four-stage code-synthesis pipeline (programmer → validator → fixer → fallback). Paper accuracy: **89.5%**, **$0.14/task** at inference (Gemini 3 Flash via OpenRouter).
- **One-time optimization cost:** **$144.70** ($0.70 reflection + $144 evaluation, paper Table). Multi-hour run; not executed in this notebook.
- **API surface:** three function calls. The notebook shows the call site, the evaluator contract, and the diagnostic `SideInfo` GEPA's reflection LM consumes.

## Reviewer mode (default — zero setup)

```bash
pip install -r requirements.txt   # Python 3.10–3.12; reviewer mode = matplotlib + numpy + jupyter
jupyter lab walkthrough.ipynb     # or render statically on GitHub / nbviewer
```

Reviewer mode does **not** require the `gepa` package — the notebook only reads JSON artifacts. The `optimize_anything` call in §1 is shown as code, never executed.

Runs top-to-bottom in **<5 s** locally (CI cap: 60 s), no network calls. Backed by JSON artifacts in `artifacts/`.

Smoke-test from the command line:

```bash
jupyter nbconvert --to notebook --execute walkthrough.ipynb \
    --output walkthrough.ipynb --ExecutePreprocessor.timeout=60
```

The same check runs on every push via `.github/workflows/smoke.yml` (paths assume `cais/arcagi/` is the repo root — relocate to `<repo>/.github/workflows/` if it lives one level up).

## Optimize-from-scratch mode (live, ~$0.50, ~5 min)

Set `OPTIMIZE_LIVE=True` in §10 and export `OPENROUTER_API_KEY`. Runs `optimize_anything` against an 8-train / 8-val ARC slice with a 30-metric-call budget. Won't reproduce 89.5% — demonstrates the optimization loop closing in real time.

```bash
pip install gepa==0.1.1
export OPENROUTER_API_KEY=sk-or-...
# In the notebook: §10 → OPTIMIZE_LIVE = True
```

## Booth mode (live, ~$0.005, ~20 s)

Set `DEMO=True` in the CONFIG cell and export `OPENROUTER_API_KEY`. The "live or cached run" cell will execute the evolved agent against the running-example puzzle in real time.

```bash
export OPENROUTER_API_KEY=sk-or-...
# In the notebook: CONFIG cell → DEMO = True
```

## What's inside

| Path | Purpose |
|------|---------|
| `walkthrough.ipynb` | The notebook |
| `arc_runtime.py` | Vendored `run_agent` + `TrackedLLM` for booth mode (no gepa-checkout dependency) |
| `best_agent.py` | The GEPA-evolved agent (verbatim) |
| `agent_architecture.md` | Architecture diagram + prompt summaries |
| `figures/optimized_candidate.svg` | Rendered architecture |
| `artifacts/example_puzzle.json` | Running-example puzzle + cached seed/evolved predictions |
| `artifacts/all_candidates.json` | GEPA search trajectory (score vs. metric calls) |
| `artifacts/test_results.json` | 80-puzzle dev-subset results (caveat data) |

## References

- **Paper:** GEPA (camera-ready). ARC-AGI: 32.5% → 89.5% on full eval split, Gemini 3 Flash.
- **GEPA repo:** https://github.com/gepa-ai/gepa
- **Canonical optimization script:** `gepa/examples/arc_agi/main.py` (132 lines).
