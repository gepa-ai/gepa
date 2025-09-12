# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa



def json_default(x):
    """Default JSON encoder for objects that are not serializable by default."""
    try:
        return {**x}
    except Exception:
        return repr(x)

def idxmax(lst: list[float]) -> int:
    """Return the index of the maximum value in a list."""
    max_val = max(lst)
    return lst.index(max_val)

def is_dominated(y, programs, program_at_pareto_front_valset):
    y_fronts = [front for front in program_at_pareto_front_valset if y in front]
    for front in y_fronts:
        found_dominator_in_front = False
        for other_prog in front:
            if other_prog in programs:
                found_dominator_in_front = True
                break
        if not found_dominator_in_front:
            return False

    return True

def remove_dominated_programs(program_at_pareto_front_valset, scores=None):
    freq = {}
    for front in program_at_pareto_front_valset:
        for p in front:
            freq[p] = freq.get(p, 0) + 1

    dominated = set()
    programs = list(freq.keys())

    if scores is None:
        scores = dict.fromkeys(programs, 1)

    programs = sorted(programs, key=lambda x: scores[x], reverse=False)

    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        for y in programs:
            if y in dominated:
                continue
            if is_dominated(y, set(programs).difference({y}).difference(dominated), program_at_pareto_front_valset):
                dominated.add(y)
                found_to_remove = True
                break

    dominators = [p for p in programs if p not in dominated]
    for front in program_at_pareto_front_valset:
        assert any(p in front for p in dominators)

    new_program_at_pareto_front_valset = [{prog_idx for prog_idx in front if prog_idx in dominators} for front in program_at_pareto_front_valset]
    assert len(new_program_at_pareto_front_valset) == len(program_at_pareto_front_valset)
    for front_old, front_new in zip(program_at_pareto_front_valset, new_program_at_pareto_front_valset, strict=False):
        assert front_new.issubset(front_old)

    return new_program_at_pareto_front_valset

def find_dominator_programs(pareto_front_programs, train_val_weighted_agg_scores_for_all_programs):
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs)
    uniq_progs = []
    for front in new_program_at_pareto_front_valset:
        uniq_progs.extend(front)
    uniq_progs = set(uniq_progs)
    return list(uniq_progs)

def select_program_candidate_from_pareto_front(pareto_front_programs, train_val_weighted_agg_scores_for_all_programs, rng):
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs)
    program_frequency_in_validation_pareto_front = {}
    for testcase_pareto_front in new_program_at_pareto_front_valset:
        for prog_idx in testcase_pareto_front:
            if prog_idx not in program_frequency_in_validation_pareto_front:
                program_frequency_in_validation_pareto_front[prog_idx] = 0
            program_frequency_in_validation_pareto_front[prog_idx] += 1

    sampling_list = [prog_idx for prog_idx, freq in program_frequency_in_validation_pareto_front.items() for _ in range(freq)]
    assert len(sampling_list) > 0
    curr_prog_id = rng.choice(sampling_list)
    return curr_prog_id
