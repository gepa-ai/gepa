# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from types import SimpleNamespace

import pytest

from gepa.utils.stop_condition import CompositeStopper, MaxTrackedCandidatesStopper, NoImprovementStopper


def _state(scores):
    return SimpleNamespace(program_full_scores_val_set=list(scores), program_candidates=[{}] * len(scores))


def _stops_at(stopper, score_history):
    """Call the stopper once per iteration and return the first iteration that stops."""
    for iteration, scores in enumerate(score_history):
        if stopper(_state(scores)):
            return iteration
    return None


# One more candidate is tracked each iteration, and the best score never improves.
HISTORY = [[0.5] * (n + 1) for n in range(12)]


@pytest.mark.parametrize("patience_first", [True, False])
def test_all_mode_does_not_depend_on_stopper_order(patience_first):
    patience = NoImprovementStopper(max_iterations_without_improvement=3)
    enough_candidates = MaxTrackedCandidatesStopper(max_tracked_candidates=6)
    stoppers = (patience, enough_candidates) if patience_first else (enough_candidates, patience)

    stopper = CompositeStopper(*stoppers, mode="all")

    # Six candidates are tracked at iteration 5. By then the score has not
    # improved for five iterations, so both conditions hold.
    assert _stops_at(stopper, HISTORY) == 5


def test_any_mode_inside_all_mode_keeps_counting():
    patience = NoImprovementStopper(max_iterations_without_improvement=3)
    always = lambda state: True
    inner = CompositeStopper(always, patience, mode="any")
    enough_candidates = MaxTrackedCandidatesStopper(max_tracked_candidates=6)

    assert _stops_at(CompositeStopper(inner, enough_candidates, mode="all"), HISTORY) == 5
    # The stopper behind the one that fired was still consulted every iteration.
    assert patience.iterations_without_improvement == 5


def test_modes():
    yes = lambda state: True
    no = lambda state: False
    state = _state([0.5])

    assert CompositeStopper(no, yes, mode="any")(state) is True
    assert CompositeStopper(no, no, mode="any")(state) is False
    assert CompositeStopper(yes, yes, mode="all")(state) is True
    assert CompositeStopper(yes, no, mode="all")(state) is False
    with pytest.raises(ValueError, match="Unknown mode"):
        CompositeStopper(yes, mode="most")(state)  # type: ignore[arg-type]
