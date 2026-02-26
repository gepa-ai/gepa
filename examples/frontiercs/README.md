# Frontier-CS Example

Optimizes C++ solutions for [Frontier-CS](https://github.com/FrontierCS/Frontier-CS) competitive programming problems using `optimize_anything()` in **seedless mode**.

## How it works

1. GEPA evolves internal prompt templates via reflection
2. `prompt_candidate_lm` generates C++ code from those templates
3. The evaluator submits the code to a local judge server and scores it

Each problem is optimized independently and parallelized across problems.

### Scoring

The judge returns a score (0-100) based on test cases passed. The evaluator also computes 4 objectives for Pareto search:

- **correctness** — fraction of AC test cases
- **compilation** — 1.0 if compiles, 0.0 otherwise
- **time_efficiency** — fraction without TLE
- **stability** — fraction without RE

## Usage

Requires the Frontier-CS judge server running locally.

```bash
# All 172 problems, 25 metric calls each
uv run python -m examples.frontiercs.main

# Specific problems with more budget
uv run python -m examples.frontiercs.main \
  --problem_ids 11 64 93 \
  --max_metric_calls 100 \
  --reflection_lm openai/gpt-5

# See all options
uv run python -m examples.frontiercs.main --help
```

## Output

Results are saved to `gepa_frontiercs_results/` by default:

- `scores.json` — per-problem scores (updated incrementally)
- `results.json` — full summary with config
- `prompts/` — best prompt template per problem
