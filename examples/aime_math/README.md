# AIME Math

Optimize a math-solving prompt for AIME competition problems. The solver LLM (GPT-4.1-mini with chain-of-thought) is fixed — GEPA optimizes only the system prompt.

## Dataset

- **Train + Val**: `AI-MO/aimo-validation-aime` (AIME 2022–2024), split 50/50
- **Test**: `MathArena/aime_2025` (AIME 2025)

## Setup

```bash
uv pip install datasets dspy litellm
```

## Run

From the repo root (`gepa-optimize-anything/`):

```bash
export OPENAI_API_KEY=...
python -m examples.aime_math.main
```

After optimization, the script evaluates both the baseline and best-found prompt on the AIME 2025 test set and prints the improvement.
