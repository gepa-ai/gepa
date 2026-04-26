# CAIS'26 Demo Track

Self-contained demo notebooks for the CAIS'26 demo track, showing GEPA's
`optimize_anything` API on two end-to-end tasks:

- [`arcagi/`](./arcagi) — ARC-AGI: a 12-line seed agent evolved into an 89.5% solver. Three modes (reviewer/booth/optimize-live).
- [`circle_packing/`](./circle_packing) — circle-packing optimization walkthrough with live recording/replay.

Each subdirectory ships its own `README.md`, `requirements.txt`, and
`walkthrough.ipynb`. Reviewer mode runs offline in seconds; booth and
optimize-live modes need an `OPENROUTER_API_KEY`.

Branch: `demo/cais26`.
