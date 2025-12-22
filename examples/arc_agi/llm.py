import dspy


def create_reflection_lm(llm_model, api_key, seed=None):
    kwargs = {
        "model": llm_model,
        "temperature": 1.0,
        "api_key": api_key,
        "max_tokens": 32000,
        "cache": False,  # Disable caching to ensure diversity in code generation
    }
    if seed is not None:
        kwargs["seed"] = seed
    return dspy.LM(**kwargs)


REFLECTION_PROMPT = """
You are optimizing a DSPy program to solve ARC (Abstract Reasoning Corpus) tasks.

Current program code:
```
<curr_param>
```

Evaluation results:
```
<side_info>
```

Analyze the evaluation data and propose an improved DSPy program that addresses failures while preserving what works.

**Requirements:**
- Code must start with: `import dspy`, `from typing import List`, `import pydantic`
- Define `TrainingExample` as a pydantic BaseModel with `input: MATRIX` and `output: MATRIX` fields
- Define a `dspy.Signature` (e.g., `SolveTaskSignature`) that extends `dspy.Signature`
- Create a `program` variable that instantiates a DSPy module (e.g., `dspy.ChainOfThought(SolveTaskSignature)`)

**Common issues to watch for:**
- Missing imports (especially `import dspy`)
- Matrix validation errors (dimensions, types, staggered rows)
- Pattern recognition failures
- Token limits or malformed outputs

Goal: Improve the program to solve the ARC tasks better.

Provide complete, executable Python code within ``` blocks.
"""
