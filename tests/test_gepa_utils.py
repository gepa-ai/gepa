import random

from gepa.gepa_utils import select_program_candidate_from_pareto_front


def test_select_program_candidate_respects_frequency_bias():
    pareto_front_programs = {
        "val0": {0, 1},
        "val1": {1},
    }
    scores = [0.3, 0.6]
    rng = random.Random(0)

    selected = select_program_candidate_from_pareto_front(pareto_front_programs, scores, rng)

    assert selected == 1


def test_select_program_candidate_falls_back_to_idxmax_when_empty():
    pareto_front_programs = {"val0": set()}
    scores = [0.1, 0.9]
    rng = random.Random(0)

    selected = select_program_candidate_from_pareto_front(pareto_front_programs, scores, rng)

    assert selected == 1
