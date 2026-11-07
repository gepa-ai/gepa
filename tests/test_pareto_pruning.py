import random

from gepa.gepa_utils import remove_dominated_programs


def reference_prune(fronts, scores):
    """Original repeated-scan policy, retained as a small-case oracle."""
    programs = list(dict.fromkeys(p for front in fronts.values() for p in front))
    programs.sort(key=lambda p: scores[p])
    survivors = set(programs)
    while True:
        for program in programs:
            if program not in survivors:
                continue
            if all(front & (survivors - {program}) for front in fronts.values() if program in front):
                survivors.remove(program)
                break
        else:
            return {key: front & survivors for key, front in fronts.items()}


def test_pruning_preserves_unique_coverage_and_prefers_higher_scores():
    fronts = {"shared": {0, 1}, "unique": {0}, "other": {1, 2}, "empty": set()}
    assert remove_dominated_programs(fronts, [0.1, 0.9, 0.5]) == {
        "shared": {0, 1},
        "unique": {0},
        "other": {1},
        "empty": set(),
    }
    assert fronts["other"] == {1, 2}


def test_pruning_matches_repeated_scan_for_overlapping_fronts_and_ties():
    rng = random.Random(42)
    for _ in range(50):
        fronts = {key: {p for p in range(12) if rng.random() < 0.4} for key in range(15)}
        scores = [rng.choice([-1.0, 0.0, 0.5, 1.0]) for _ in range(12)]
        assert remove_dominated_programs(fronts, scores) == reference_prune(fronts, scores)
        assert remove_dominated_programs(fronts) == reference_prune(fronts, [1] * 12)


def test_empty_fronts_are_preserved():
    assert remove_dominated_programs({}) == {}
    assert remove_dominated_programs({"empty": set()}) == {"empty": set()}
