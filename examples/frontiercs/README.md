# Frontier-CS Optimization with GEPA

Evolves LLM prompts that produce correct, efficient C++ solutions for [Frontier-CS](https://github.com/FrontierCS/Frontier-CS) competitive programming problems, using GEPA's seedless mode.

## Structure

- `main.py`: Entry point — defines the evaluator and calls `optimize_anything`.
- `utils/`
  - `judge.py`: Thread-safe client for the Frontier-CS judge API (submit + poll).
  - `dataset.py`: Loads problem statements, metadata, and samples (auto-clones from GitHub).
  - `background.py`: Domain background passed to GEPA's reflection LM.

## Prerequisites

1. **Frontier-CS judge server** running locally (default: `http://localhost:8081`). See [LightCPVerifier](https://github.com/FrontierCS/LightCPVerifier) for setup.

2. **Install dependencies:**
   ```bash
   uv sync
   uv pip install requests
   ```

3. The Frontier-CS dataset is auto-cloned to `~/.cache/gepa/frontier-cs/` on first run, or pass `--problems_dir` to use a local copy.

## Running

```bash
# All 172 problems, 25 metric calls each, gpt-5 reflection
uv run python -m examples.frontiercs.main

# Specific problems with more budget
uv run python -m examples.frontiercs.main \
  --problem_ids 11 64 93 \
  --max_metric_calls 100 \
  --reflection_lm openai/gpt-5

# See all options
uv run python -m examples.frontiercs.main --help
```

## How It Works

1. GEPA evolves prompt templates in seedless mode (no seed candidate).
2. For each problem, `prompt_candidate_lm` generates C++ code from the evolved template.
3. The evaluator extracts the C++ code, submits it to the judge, and polls for results.
4. Score + side info (per-case AC/WA/TLE/RE feedback) feed back into GEPA's reflection and refinement loop.

Each problem is optimized independently and parallelized across problems (default: 64 concurrent).

### Scoring

The judge returns a raw score (0-100) based on test cases passed. The evaluator also computes 4 sub-objectives for Pareto search:

- **correctness** -- fraction of AC test cases
- **compilation** -- 1.0 if compiles, 0.0 otherwise
- **time_efficiency** -- fraction without TLE
- **stability** -- fraction without RE

### Refiner

Per-evaluation refinement is enabled by default. After each evaluation, the refiner LM gets one shot to fix the candidate based on judge feedback before the score is finalized. This rescues broken proposals (e.g., compilation errors, wrong output format) without consuming an extra iteration.

## Output

Results are saved to `gepa_frontiercs_results/` by default:

- `scores.json` -- per-problem scores (updated incrementally as problems complete)
- `results.json` -- full summary with config and best prompts
- `prompts/<problem_id>.txt` -- best prompt template per problem
- `runs/<problem_id>/` -- GEPA run state per problem (for resuming)
