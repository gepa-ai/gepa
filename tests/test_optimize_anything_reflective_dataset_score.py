# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for score injection into the reflective dataset (#288).

``OptimizeAnythingAdapter.make_reflective_dataset`` used to forward only the
user's ``side_info`` fields.  The per-example scalar score -- the value GEPA
itself uses to accept or reject a candidate -- never reached the reflection LM,
so a 0.95 and a 0.0 looked identical to it unless the evaluator remembered to
duplicate the score into ``side_info`` by hand.

The adapter now injects the score as ``"Score (Higher is Better)"``, but never
overwrites a key the evaluator already supplied.
"""

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import (
    SCORE_INJECTION_KEY,
    OptimizeAnythingAdapter,
    _is_scalar_score_key,
)

COMPONENT = "instruction"
CANDIDATE = {COMPONENT: "answer the question"}


def _make_adapter(side_info_by_example):
    """Adapter whose evaluator returns a fixed score/side_info per example."""

    def _eval(candidate, example=None, opt_state=None):
        score, side_info = side_info_by_example[example]
        return score, candidate, side_info

    return OptimizeAnythingAdapter(evaluator=_eval, cache_mode="off")


def _reflective_records(side_info_by_example, examples, component=COMPONENT):
    adapter = _make_adapter(side_info_by_example)
    batch = adapter.evaluate(list(examples), CANDIDATE, capture_traces=True)
    dataset = adapter.make_reflective_dataset(CANDIDATE, batch, [component])
    return list(dataset[component])


def test_scalar_score_is_injected_when_side_info_omits_it():
    # The reported failure mode: the reflection LM sees Input/Output but has no
    # way to tell a near-pass from a total failure.
    records = _reflective_records(
        {
            "ex0": (0.25, {"Input": "ex0", "Output": "wrong"}),
            "ex1": (0.95, {"Input": "ex1", "Output": "right"}),
        },
        ["ex0", "ex1"],
    )

    assert records == [
        {"Input": "ex0", "Output": "wrong", SCORE_INJECTION_KEY: 0.25},
        {"Input": "ex1", "Output": "right", SCORE_INJECTION_KEY: 0.95},
    ]


def test_score_is_injected_when_side_info_is_empty():
    # A score-only evaluator (plus oa.log/captured stdout) gets the score too.
    records = _reflective_records({"ex0": (0.5, {})}, ["ex0"])

    assert records == [{SCORE_INJECTION_KEY: 0.5}]


def test_user_supplied_score_is_never_overwritten():
    # A user who already forwards the score (under any casing) keeps their value
    # and its key -- GEPA must not add a duplicate field for the same number.
    records = _reflective_records(
        {
            "ex0": (0.25, {"Score": 42.0}),
            "ex1": (0.95, {"SCORE": "graded by rubric"}),
        },
        ["ex0", "ex1"],
    )

    assert records == [
        {"Score": 42.0},
        {"SCORE": "graded by rubric"},
    ]


def test_multi_objective_scores_still_rename_and_do_not_suppress_injection():
    # `side_info["scores"]` is the multi-objective dict, a different thing from
    # the blended scalar score. It keeps its rename *and* the scalar is still
    # injected -- otherwise multi-objective evaluators would silently lose the
    # aggregate score the reflection LM reasons about.
    records = _reflective_records(
        {"ex0": (0.5, {"Input": "ex0", "scores": {"accuracy": 0.5, "cost": 12.0}})},
        ["ex0"],
    )

    assert records == [
        {
            "Input": "ex0",
            "Scores (Higher is Better)": {"accuracy": 0.5, "cost": 12.0},
            SCORE_INJECTION_KEY: 0.5,
        }
    ]


def test_score_is_injected_for_every_updated_component():
    # The score is per-example, not per-component, so all components being
    # updated in one iteration must carry it.
    adapter = _make_adapter({"ex0": (0.75, {"Input": "ex0"})})
    batch = adapter.evaluate(["ex0"], CANDIDATE, capture_traces=True)

    dataset = adapter.make_reflective_dataset(CANDIDATE, batch, ["instruction", "other"])

    for component in ("instruction", "other"):
        assert dataset[component] == [{"Input": "ex0", SCORE_INJECTION_KEY: 0.75}]


def test_zero_and_negative_scores_are_still_injected():
    # A hard 0.0 is the most informative score there is; it must not be mistaken
    # for "no score recorded" and dropped.
    records = _reflective_records(
        {"ex0": (0.0, {"Input": "ex0"}), "ex1": (-1.0, {"Input": "ex1"})},
        ["ex0", "ex1"],
    )

    assert [r[SCORE_INJECTION_KEY] for r in records] == [0.0, -1.0]


def test_is_scalar_score_key_recognizes_casings_and_parenthetical_qualifiers():
    assert _is_scalar_score_key(SCORE_INJECTION_KEY)
    assert _is_scalar_score_key("score")
    assert _is_scalar_score_key("Score")
    assert _is_scalar_score_key("SCORE")
    # A user-chosen qualifier still means "this is my score".
    assert _is_scalar_score_key("Score (0-10 rubric)")

    assert not _is_scalar_score_key("scores")
    assert not _is_scalar_score_key("Scores (Higher is Better)")
    assert not _is_scalar_score_key("Input")
    assert not _is_scalar_score_key("Expected Output")
    assert not _is_scalar_score_key("Scoresheet")
    assert not _is_scalar_score_key("best_scoring_attempt")
