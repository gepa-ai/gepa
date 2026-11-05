import math

import pytest

from gepa.evidence.config import SequentialEvidenceConfig
from gepa.evidence.observations import Observation, ObservationId, candidate_identity
from gepa.evidence.statistics import RunningMoments, alpha_at, mean_interval, weighted_difference


def config(**kwargs):
    return SequentialEvidenceConfig(evaluator_fingerprint="test-v1", max_comparison_calls=200, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"score_min": math.nan}, {"score_max": math.inf}, {"score_min": 1, "score_max": 0},
    {"alpha": 0}, {"alpha": 1}, {"min_rounds": 0}, {"max_comparison_calls": True},
    {"max_calls_per_example": 0}, {"effect_margin": -1}, {"assumption": "unknown"},
    {"stratum_weights": (0.5, -0.5)},
])
def test_invalid_contract(kwargs):
    base = {"evaluator_fingerprint": "v1", "max_comparison_calls": 200}
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        SequentialEvidenceConfig(**base)


def test_identity_is_content_based_and_side_trial_context_is_distinct():
    assert candidate_identity({"b": "x", "a": "y"}) == candidate_identity({"a": "y", "b": "x"})
    original = ObservationId("e", "c", "train", "0", "trial-0", "parent")
    assert original.key != ObservationId("e", "c", "train", "0", "trial-1", "parent").key
    assert original.key != ObservationId("e", "c", "train", "0", "trial-0", "child").key
    assert Observation.from_dict(Observation(original, "completed", 0.3, 1).to_dict()).identity == original


def test_alpha_allocation_telescopes_and_accounts_for_sides_and_strata():
    # sum 1/(n(n+1)) = 1; finite partial sums are analytic, not simulation.
    alpha = 0.05
    total = sum(alpha_at(alpha, c, s, n, 3) for c in range(1, 80)
                for s in range(3) for n in range(1, 90)) * 2
    assert total == pytest.approx(alpha * (1 - 1 / 80) * (1 - 1 / 90))
    assert alpha_at(alpha, 2, 0, 4, 3) == pytest.approx(alpha / (2 * 6 * 20 * 3))


def test_moments_and_hoeffding_interval_match_oracles():
    moments = RunningMoments()
    for value in [0, 1, 0.5, 0.5]:
        moments.add(value)
    assert moments.mean == 0.5
    assert moments.variance == pytest.approx(1 / 6)
    low, high = mean_interval(moments, 0, 1, 0.1)
    radius = math.sqrt(math.log(20) / 8)
    assert (low, high) == pytest.approx((max(0, 0.5-radius), min(1, 0.5+radius)))
    assert mean_interval(RunningMoments(), -2, 5, 0.1) == (-2, 5)


def test_fixed_weight_difference_uses_both_sides_and_endpoints():
    bounds = weighted_difference([(0.2, 0.4), (0.6, 0.8)], [(0.5, 0.7), (0.1, 0.3)], [0.25, 0.75])
    assert bounds == pytest.approx((-0.5, -0.1))
    with pytest.raises(ValueError):
        weighted_difference([(0, 1)], [(0, 1)], [0.5])


def test_config_roundtrip_and_fingerprint_excludes_only_increased_global_cap():
    contract = config()
    assert SequentialEvidenceConfig.from_dict(contract.to_dict()) == contract
    assert contract.fingerprint == config().fingerprint
    assert contract.fingerprint != config(effect_margin=0.1).fingerprint
