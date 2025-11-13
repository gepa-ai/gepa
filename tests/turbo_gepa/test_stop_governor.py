import math

from turbo_gepa.stop_governor import EpochMetrics, StopGovernor


def test_stop_governor_uses_incremental_deltas() -> None:
    """Ensure hypervolume rate and ROI use per-epoch deltas, not cumulative totals."""

    governor = StopGovernor()
    alpha = governor.config.alpha

    # Seed epoch (establish baseline)
    epoch0 = EpochMetrics(
        round_num=0,
        hypervolume=0.0,
        new_evaluations=5,
        best_quality=0.50,
        best_cost=-10.0,
        frontier_ids=set(),
        total_tokens_spent=100,
    )
    governor.update(epoch0)

    # Second epoch should use delta hypervolume / delta evaluations
    epoch1 = EpochMetrics(
        round_num=1,
        hypervolume=0.2,
        new_evaluations=2,
        best_quality=0.60,
        best_cost=-9.0,
        frontier_ids=set(),
        total_tokens_spent=130,
    )
    governor.update(epoch1)

    expected_hv_rate = (0.2 - 0.0) / 2  # delta hypervolume / new evaluations
    expected_roi = (0.2 - 0.0) / (130 - 100)  # delta hypervolume / delta tokens

    assert math.isclose(governor.ewma_hv_rate, alpha * expected_hv_rate, rel_tol=1e-9)
    assert math.isclose(governor.ewma_roi, alpha * expected_roi, rel_tol=1e-9)
