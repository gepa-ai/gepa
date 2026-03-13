# Glean Adapter Usage

The Glean adapter optimizes prompts for the Glean Assistant by running eval sets through the Glean API and using evolutionary optimization.

## How It Works

1. **No trainset file needed**: The adapter runs entire eval sets via the Glean API
2. **Two eval set sizes**:
   - **Screening/Mini-batch**: Uses "Glean Chat Multiturn V2 Small" for fast candidate screening
   - **Full evaluation**: Uses "Glean Chat Multiturn V2 Medium" for thorough evaluation
3. **One API call per eval set**: Results are cached to avoid redundant API calls
4. **Evolutionary optimization**: GEPA evolves prompt components (GLOBAL_ROLE, TOOL_USAGE, PERSISTENCE, etc.)

## Prerequisites

1. **Cookie authentication**: Extract your cookie from the browser when logged into Glean
   - Open Developer Tools → Network tab
   - Make a request to glean.com
   - Copy the `cookie` header value

2. **Seed candidate**: A JSON file with initial prompt components (see `data/seed_candidate.json`)

## Running

```bash
uv run python -m gepa.api \
  --seed_candidate data/seed_candidate.json \
  --cookie "your_cookie_string_from_browser" \
  --eval_set_version "20260308" \
  --model claude \
  --teacher_model gpt \
  --max_metric_calls 20 \
  --run_dir ./gepa_runs
```

## Parameters

- `--seed_candidate`: Path to JSON file with initial prompt components
- `--cookie`: Authentication cookie from your browser (required for API access, includes identity)
- `--eval_set_version`: Eval set version to use (e.g., "20260308")
- `--model`: Student model to optimize (default: "claude")
- `--teacher_model`: Teacher model to compare against (default: "gpt")
- `--max_metric_calls`: Maximum number of evaluations (default: 10)
- `--run_dir`: Directory to save results and state

## Eval Set Sizes

The adapter automatically uses two different eval set sizes:
- **Screening/Mini-batch**: "Glean Chat Multiturn V2 Small" - Used for fast candidate screening during optimization
- **Full Evaluation**: "Glean Chat Multiturn V2 Medium" - Used for thorough evaluation of final candidates

Both use the same version specified by `--eval_set_version`.

## Seed Candidate Format

The seed candidate is a JSON object with prompt components:

```json
{
  "GLOBAL_ROLE": "You are an AI assistant...",
  "PERSISTENCE": "For complex tasks, follow this approach...",
  "FORMATTING": "",
  "TOOL_USAGE": [
    "First tool usage guideline",
    "Second tool usage guideline",
    "Third tool usage guideline",
    "Fourth tool usage guideline"
  ]
}
```

- Most components are strings
- `TOOL_USAGE` is a list of 4 strings that get flattened to `TOOL_USAGE_1` through `TOOL_USAGE_4` internally

## How the API Integration Works

1. **First evaluation**: Makes one API call to run the entire eval set for a candidate
2. **Caching**: All query results are cached by (model, system_prompt, eval_set_name, eval_set_version)
3. **Subsequent queries**: Returned instantly from cache without additional API calls
4. **Teacher model**: Uses GPT with production prompt (no custom prompt) for comparison
5. **Student model**: Uses Claude with evolved prompt components
6. **Screening vs Full**: Small eval set for quick candidate screening, Medium eval set for final evaluation

## Output

Results are saved to the `run_dir`:
- `state.pkl`: Optimization state (can resume from this)
- `best_candidate.json`: Best performing prompt discovered
- `pareto_front.json`: All non-dominated candidates
- Logs and metrics for each iteration
