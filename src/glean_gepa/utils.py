from glean_gepa.al_adapter import Candidate


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
