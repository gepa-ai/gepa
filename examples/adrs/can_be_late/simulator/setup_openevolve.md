## OpenEvolve + Data Quickstart (bash + uv)

This document shows a fully reproducible local setup using bash and uv only. No pip is used.

### 1) Prerequisites

- uv (package manager). Install from the official docs: [uv installation](https://docs.astral.sh/uv/). Quick install:

```bash
curl -LsSf https://astral.sh/uv/install.sh | bash
exec "$SHELL"  # reload your shell so `uv` is on PATH
uv --version
```

- gsutil (for data). If you don't have it, install the Google Cloud SDK and enable gsutil: [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)

Change to the project root:

```bash
git clone https://github.com/cblmemo/cant-be-late.git
cd cant-be-late
git checkout single-region-evolve
uv sync
```

Note: Weights & Biases is NOT required for local runs. You can ignore it.

### 2) Clone OpenEvolve into the local temp directory

The evolution script expects `@temp/openevolve/openevolve-run.py`.

```bash
mkdir -p @temp
git clone https://github.com/andylizf/openevolve.git @temp/openevolve
```

### 3) Download data (required)

The simulator and evaluator depend on real traces. Use gsutil to fetch them:

```bash
mkdir -p data/real/
gsutil -m cp -r gs://spot-vs-demand-exp/data/real/ping_based/ data/real/
```

After this, you should have `data/real/ping_based/...` with the trace files.

### 4) Sanity-check the baseline (rc_cr_threshold)

Make sure the command prints `mean:` and `std:` in the output.

```bash
uv run python main.py \
  --strategy=rc_cr_threshold \
  --env=trace \
  --trace-file data/real/ping_based/random_start_time/us-west-2a_k80_1/0.json \
  --task-duration-hours=48 \
  --deadline-hours=52 \
  --restart-overhead-hours=0.2
```

Should print something like:
```
[simulate] mean: 47.681900000000006; std: 0.0; worst 1%: 47.681900000000006; worst 10%: 47.681900000000006
```

If you see something like `mean: ...; std: ...`, you're ready.

### 5) Run the single-region evaluator (baseline comparison + scoring)

```bash
cd openevolve_single_region_strategy
uv run python evaluator.py
```

This will:
- Compute and cache the rc_cr_threshold baseline into `openevolve_single_region_strategy/baseline_cache.json`
- Evaluate the initial program across multiple traces/configs and print average cost, savings, and a score

### 6) (Optional) Run evolution

```bash
cd openevolve_single_region_strategy
uv run bash run_evolution.sh
```

The script has no hardcoded absolute paths. It auto-detects the repo root and uses `@temp/openevolve`.
Outputs are saved under `openevolve_output/` and the best program under `openevolve_output/best/`.

### Troubleshooting

- Missing `gsutil`: install the Google Cloud SDK and ensure `gsutil` is on your PATH.
- Data path assumptions: the evaluator and scripts assume data under `data/real/ping_based/`.
