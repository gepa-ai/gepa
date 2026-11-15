# Can't Be Late: Cloud Scheduling Strategy Optimization

This example uses GEPA's `optimize_anything` API to evolve cloud scheduling strategies for the "Can't Be Late" problem from [NSDI'24](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao).

## Problem Overview

The strategy decides when to use:
- **SPOT instances**: Cheap (~$0.3/hour) but can be preempted at any time
- **ON_DEMAND instances**: Expensive (~$1/hour) but guaranteed availability
- **NONE**: Wait without using any instances

**Goal**: Minimize cost while ensuring task completion before the deadline.

## Setup

### 1. Download and Extract Trace Data

Download the trace dataset from the ADRS repository and extract it:

```bash
cd examples/adrs/can_be_late/utils/simulator
curl -L -o real_traces.tar.gz https://github.com/UCB-ADRS/ADRS/raw/main/openevolve/examples/ADRS/cant-be-late/simulator/real_traces.tar.gz
tar -xzf real_traces.tar.gz
```

This creates the `real/` directory containing spot availability traces from AWS.

### 2. Install Dependencies

From the repository root:

```bash
pip install -e .
```

Install example dependencies:
```bash
pip install -r examples/adrs/can_be_late/requirements.txt
```

For Gemini models, also install:
```bash
pip install google-genai
```

## Running the Example

```bash
python examples/adrs/can_be_late/main.py --model <model_name>
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model` | LLM model name (e.g., `gpt-4o`, `gemini-1.5-pro`) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-metric-calls` | 100 | Maximum fitness evaluations |
| `--max-traces` | None | Limit traces per split (for quick testing) |
| `--minibatch-size` | 3 | Reflection minibatch size |
| `--dataset-root` | `utils/simulator/real` | Path to trace dataset |
| `--run-dir` | `runs/cant_be_late/<timestamp>` | Output directory |

### Example

Quick test run:
```bash
python examples/adrs/can_be_late/main.py \
    --model gpt-4o-mini \
    --max-traces 10 \
    --max-metric-calls 20
```

Full optimization:
```bash
python examples/adrs/can_be_late/main.py \
    --model gpt-4o \
    --max-metric-calls 100
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for OpenAI models |
| `GOOGLE_API_KEY` | Required for Gemini models |
| `WANDB_API_KEY` | Optional: Enable W&B logging |
| `GEPA_RUN_DIR` | Override default run directory |
| `GEPA_SKIP_TEST` | Set to `1` to skip test evaluation |

## Output

Results are saved to the run directory:
- `best_program.py` - Best evolved strategy
- `metrics.json` - Performance summary
- `candidates/` - All generated candidates

## How It Works

1. Loads real-world spot availability traces from AWS
2. Evaluates strategies by simulating them on traces with varying:
   - Job durations (48h)
   - Deadlines (52h, 70h, 92h)
   - Restart overheads (2%, 20%, 40%)
3. Uses GEPA's reflective mutation to iteratively improve the strategy
4. Returns the best-performing strategy based on cost minimization
