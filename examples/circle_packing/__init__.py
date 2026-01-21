"""
Circle Packing Optimization with GEPA - Direct Code Optimization.

Main components:
- main.py: Entry point for running experiments
- utils.py: Code execution, validation, dataset creation
- llms.py: Code proposer with reflection LLM
"""

from .utils import (
    execute_code,
    validate_packing,
    create_circle_packing_dataset,
    BASELINE_CODE_TEMPLATE,
)

from .llms import (
    CODE_REFLECTION_PROMPT_TEMPLATE,
    REFINEMENT_PROMPT_REFLECTION_INSTRUCTIONS,
    SEED_REFINEMENT_PROMPT,
    RefinerSignature,
)

__all__ = [
    "execute_code",
    "validate_packing",
    "create_circle_packing_dataset",
    "BASELINE_CODE_TEMPLATE",
    "CODE_REFLECTION_PROMPT_TEMPLATE",
    "REFINEMENT_PROMPT_REFLECTION_INSTRUCTIONS",
    "SEED_REFINEMENT_PROMPT",
    "RefinerSignature",
]
