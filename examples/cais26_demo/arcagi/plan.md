# ARC-AGI Walkthrough Notebook — Plan

CAIS'26 Demo Track submission. This document is the spec for `walkthrough.ipynb`.

## Goal

A notebook that, in under 5 minutes of reading and ≤60 seconds of compute, makes a reviewer believe:

> **"GEPA can take a 12-line agent and evolve it into an 89.5% ARC solver, and the API to do it is three function calls."**

Three jobs, in priority order:

1. **Demonstrate the API** (`optimize_anything`) — this is the demo-track contribution.
2. **Show the evolution is real** — seed code → evolved code, with a working example.
3. **Anchor to paper results** — 32.5% → 89.5%, $0.14/task, Gemini 3 Flash.

## Modes

- **Reviewer mode (default, zero setup):** opens, renders, tells the story. No API key. `pip install gepa==0.1.1 matplotlib numpy`. Everything backed by `artifacts/`.
- **Booth mode (`DEMO=True`, `OPENROUTER_API_KEY` set):** runs the evolved agent live on one puzzle. ~20s, ~$0.005. The "live magic" moment.
- **No re-optimization mode.** A 6-hour, $80 run is not a demo. The optimization call is *shown* as code, not executed.

## Cells

1. **Hook + how to read this** — Headline: 32.5% → 89.5%, evolved by GEPA, no human in the loop. Paragraph on modes (reviewer default; booth via `DEMO=True`). Links to paper + GitHub. Reviewer should be sold by the end of this cell.

2. **The API in 10 lines** — Pull this *before* the puzzle. Demo-track reviewers are evaluating the tool.
   ```python
   from gepa.optimize_anything import optimize_anything, GEPAConfig, ...
   result = optimize_anything(
       seed_candidate=SEED_AGENT_CODE,
       evaluator=evaluate,           # (candidate, example) -> (score, SideInfo)
       dataset=train_set, valset=val_set,
       objective=OBJECTIVE, background=BACKGROUND,
       config=GEPAConfig(...),
   )
   ```
   Caption: "Optimize any text artifact. The `evaluator` returns a score plus `SideInfo` — diagnostics the reflection LM uses to propose better candidates. That's the whole interface."

3. **The task** — One ARC puzzle visualized as ARC-palette grids (train pairs + test input + ground-truth test output). Loaded from `artifacts/example_puzzle.json` (offline-safe). This is the running example throughout.

4. **The seed (12 lines)** — `SEED_AGENT_CODE` verbatim. Cached seed prediction on the puzzle from cell 3 → side-by-side wrong vs. truth.

5. **The evaluator + a peek at SideInfo** — Condensed `evaluate()` body. Pretty-print one real `SideInfo` dict (problem_id, training_score, test_score, error, traces) so reviewers see *what GEPA's reflection LM actually consumes*. This is the demo-track money shot for the API.

6. **What GEPA discovered** — Print `best_agent.py`. Render `figures/optimized_candidate.svg` inline (mermaid as markdown fallback). Caption the four stages: programmer → validator → executor → fallback.

7. **Live or cached run** —
   - **Booth mode** (`DEMO=True`): live `run_agent` on cell-3's puzzle, LLM call trace streamed (1–4 calls).
   - **Reviewer mode**: cached prediction → correct, side-by-side with truth.
   Same puzzle as cell 4 — direct contrast.

8. **Trajectory** — Validation accuracy vs. metric calls from `artifacts/all_candidates.json`. Mark seed and best. No speculative milestone annotations.

9. **Results** — Bar chart with paper numbers (32.5% → 89.5%, full eval split). One line on cost ($0.14/task, Gemini 3 Flash via OpenRouter). One-line caveat about the 80-puzzle dev subset in `test_results.json`.

10. **Reproduce + extend** — `pip install gepa==0.1.1`; point to `examples/arc_agi/main.py`; one sentence on swapping `evaluate` to optimize *your* artifact.

## Reviewer-runnability requirements (demo-track-specific)

- `requirements.txt` next to the notebook: `gepa==0.1.1`, `matplotlib`, `numpy`. (No `datasets` — puzzle is pickled offline.)
- `README.md` at `cais/arcagi/` with: 30-second pitch, how to render statically, how to enable booth mode, expected runtime/cost.
- Top of notebook: one `CONFIG` cell — `DEMO = False`, paths, model id. Everything else parameterized.
- All artifact paths relative; no absolute paths.
- **Hard constraint:** notebook must execute top-to-bottom in <60s in reviewer mode with no network call. Pickle the running-example puzzle, seed's cached prediction, and evolved agent's cached prediction into `artifacts/`.

## Anti-goals (what this is NOT)

- ❌ Not the current `walkthrough.ipynb` — dumps internals, no narrative, no API surface, no grid viz.
- ❌ No `LegacyStateUnpickler` / `make_stub` / `gepa_state.bin` dependence. JSON only.
- ❌ No "candidate index 5", "parent program", or other internal jargon without a one-line gloss.
- ❌ No HuggingFace dataset load in reviewer mode.
- ❌ No re-optimization, ever, in the notebook.
- ❌ No code dump of `utils.py` plumbing — link to it.
- ❌ No promising live results in static text — every claim qualified by "paper" or "cached run."

## Canonical references

- **Code pattern:** `/Users/lukedhlee/gepa/examples/arc_agi/main.py` (132 lines) — adapt as notebook.
- **Plumbing (link, don't copy):** `/Users/lukedhlee/gepa/examples/arc_agi/utils.py` — `TrackedLLM`, `run_agent`, `BACKGROUND`, `OBJECTIVE`.
- **API surface:** `gepa.optimize_anything` (v0.1.1) — installed and inspected.
- **Architecture asset:** `cais/arcagi/agent_architecture.md` + `cais/arcagi/figures/optimized_candidate.svg`.
- **Evolved agent:** `cais/arcagi/best_agent.py`.
- **Numbers:** paper (32.5% → 89.5%, full eval split). `artifacts/test_results.json` (42.5% → 91.25%, 80-puzzle dev subset) used only as a caveat.

## Pre-work before writing the notebook

1. **Inventory `artifacts/`:** confirm `all_candidates.json` has the score-vs-metric-calls series; confirm `generated_best_outputs_valset/` has cached predictions; verify `best_agent.py` matches the paper's evolved agent.
2. **Pick the running-example puzzle:** seed-fails / evolved-succeeds, visually legible transformation (symmetry / color-mapping / object-counting). Save the puzzle, the seed's cached output, and the evolved agent's cached output to `artifacts/example_puzzle.json`.
3. **Test renderers:** mermaid vs. SVG-only in JupyterLab + nbviewer + GitHub. Default to SVG.
4. **Draft `README.md` + `requirements.txt` skeletons.**

## Deliverables

- `cais/arcagi/walkthrough.ipynb` — the notebook (rewritten from scratch)
- `cais/arcagi/README.md` — reviewer-facing entry point
- `cais/arcagi/requirements.txt` — pinned deps
- `cais/arcagi/artifacts/example_puzzle.json` — offline running example with cached predictions
