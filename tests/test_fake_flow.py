import pytest

from glean_gepa.fake_flow import FakeFlowAdapter, fake_evalset, fake_seed_candidate
from glean_gepa.prompt import WRITING_CODE_KEY
from glean_gepa.runner import _parse_args


def test_fake_flow_flag_is_available_without_a_seed_candidate():
    assert _parse_args(["--fake_flow"]).fake_flow is True


def test_fake_evaluations_improve_for_each_fake_iteration():
    adapter = FakeFlowAdapter()
    seed = fake_seed_candidate()
    improved = {WRITING_CODE_KEY: seed[WRITING_CODE_KEY].replace("iteration=0", "iteration=2")}

    seed_result = adapter.evaluate(fake_evalset(), seed, capture_traces=True)
    improved_result = adapter.evaluate(fake_evalset(), improved, capture_traces=True)

    assert seed_result.summary == {"fake_score": pytest.approx(0.35)}
    assert improved_result.summary == {"fake_score": pytest.approx(0.75)}
    assert len(seed_result.trajectories or []) == 3


def test_fake_flow_has_distinct_train_and_validation_sets():
    trainset = fake_evalset("train")
    valset = fake_evalset("val")

    assert {item["eval_set_version"] for item in trainset}.isdisjoint(
        item["eval_set_version"] for item in valset
    )
