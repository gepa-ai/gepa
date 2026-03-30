from typing import Dict
from .al_adapter import Candidate, ModuleSpec, MODULES
import random

def crossover(a: Candidate, b: Candidate, module_specs: Dict[str, ModuleSpec], global_cap: int) -> Candidate:
    new_modules = dict(a.prompt_modules)
    # randomly swap some modules from b
    for mid in MODULES:
        if mid in b.prompt_modules and random.random() < 0.5:
            new_modules[mid] = b.prompt_modules[mid]
    return Candidate(
        model=a.model,
        prompt_modules=new_modules,
        module_specs=module_specs,
        global_token_cap=global_cap,
        baseline_prompt_hash=a.baseline_prompt_hash,
        parent_id=a.candidate_id,
    )

def apply_single_module_edit(parent: Candidate, module_name: str, new_text: str) -> Candidate:
    pm = dict(parent.prompt_modules)
    pm[module_name] = new_text
    return Candidate(
        model=parent.model,
        prompt_modules=pm,
        module_specs=parent.module_specs,
        global_token_cap=parent.global_token_cap,
        baseline_prompt_hash=parent.baseline_prompt_hash,
        parent_id=parent.candidate_id,
    )