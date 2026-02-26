# Circle Packing

Uses GEPA to evolve Python code that packs 26 non-overlapping circles inside a unit square, maximizing the sum of radii.

## How it works

- GEPA evolves the packing algorithm (full Python code, not a prompt)
- Each candidate runs in a subprocess with a 600s timeout
- Warm-starting: the best circle configuration so far is passed to new candidates
- A refiner LLM iterates on candidates between GEPA generations

## Setup

Our `requirements.txt` contains diverse libraries to support optimization for the circle packing problem.
Although LLMs only use a handful of them, we provide diverse options regardless for the optimization engine.
```bash
uv pip install -r requirements.txt
```

## Run

From the repo root (`gepa-optimize-anything/`):

```bash
export OPENAI_API_KEY=...
python -m examples.circle_packing.main
```

Results are saved to `outputs/circle_packing/`.
