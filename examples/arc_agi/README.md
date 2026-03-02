# ARC-AGI

Uses GEPA to evolve agent code that solves ARC-AGI puzzles. The agent gets training input/output pairs and must predict test outputs. GEPA optimizes the agent's Python code (not a prompt) — the agent can make up to 10 LLM calls per problem.

## Dataset

- **Source**: `dataartist/arc-agi` (HuggingFace)
- **Train**: 200 problems from the training split
- **Val**: 200 problems held out from the training split
- **Test**: Full evaluation split

## Setup

From the repo root (`gepa/`):

```bash
uv venv
uv pip install datasets dspy litellm
uv pip install -e .  # must come after dspy to avoid PyPI overwrite
```

## Run

```bash
export OPENAI_API_KEY=...
uv run python -m examples.arc_agi.main
```

The evolved agent is saved to `outputs/arc_agi/best_agent.py`. After optimization, the script evaluates both the baseline and best agent on the held-out test set.
