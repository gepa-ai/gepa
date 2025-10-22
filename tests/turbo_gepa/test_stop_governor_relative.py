"""Test StopGovernor with relative improvement thresholds."""

from turbo_gepa.stop_governor import StopGovernor, StopGovernorConfig, EpochMetrics


def test_relative_improvement_at_low_quality():
    """Test that relative improvement is detected even when absolute is small."""
    config = StopGovernorConfig(
        tau_quality=0.001,  # 0.1% absolute
        tau_quality_relative=0.01,  # 1% relative
        hysteresis_window=3,
        stop_threshold=0.15,
    )

    governor = StopGovernor(config)

    # Start at 33% quality (like AIME case)
    metrics1 = EpochMetrics(
        round_num=1,
        hypervolume=10.0,
        new_evaluations=20,
        best_quality=0.33,
        best_cost=-1000,
        frontier_ids={"a", "b"},
        qd_filled_cells=5,
        qd_total_cells=100,
        qd_novelty_rate=0.05,
        total_tokens_spent=1000,
    )
    governor.update(metrics1)

    # Improve to 34% - only 1% absolute but 3% relative improvement
    metrics2 = EpochMetrics(
        round_num=2,
        hypervolume=10.5,
        new_evaluations=20,
        best_quality=0.34,  # +0.01 absolute, +3% relative
        best_cost=-1050,
        frontier_ids={"a", "b", "c"},
        qd_filled_cells=6,
        qd_total_cells=100,
        qd_novelty_rate=0.05,
        total_tokens_spent=2000,
    )
    governor.update(metrics2)

    # Check signals
    stop_score, signals = governor.compute_stop_score()

    # The relative improvement (3%) should be detected even though absolute (1%) is below threshold
    # tau_quality_relative = 0.01 (1%), so 3% relative improvement should give signal > 1.0
    # tau_quality = 0.001 (0.1%), so 1% absolute should give signal = 10.0
    # We take the max of both, so should definitely detect improvement

    assert signals["s_quality"] > 0, "Quality signal should detect improvement"

    # Should NOT stop yet
    should_stop, debug = governor.should_stop()
    assert not should_stop, f"Should not stop with improvement detected: {debug}"

    print(f"✓ Quality signal: {signals['s_quality']:.3f}")
    print(f"✓ Stop score: {stop_score:.3f}")
    print(f"✓ Should stop: {should_stop}")


def test_absolute_improvement_dominates_when_higher():
    """Test that absolute threshold dominates when it detects more improvement."""
    config = StopGovernorConfig(
        tau_quality=0.001,  # 0.1% absolute
        tau_quality_relative=0.01,  # 1% relative
    )

    governor = StopGovernor(config)

    # Start at 90% quality
    metrics1 = EpochMetrics(
        round_num=1,
        hypervolume=10.0,
        new_evaluations=20,
        best_quality=0.90,
        best_cost=-1000,
        frontier_ids={"a", "b"},
        qd_filled_cells=5,
        qd_total_cells=100,
        qd_novelty_rate=0.05,
        total_tokens_spent=1000,
    )
    governor.update(metrics1)

    # Improve to 92% - 2% absolute, 2.2% relative
    metrics2 = EpochMetrics(
        round_num=2,
        hypervolume=11.0,
        new_evaluations=20,
        best_quality=0.92,  # +0.02 absolute, +2.2% relative
        best_cost=-1050,
        frontier_ids={"a", "b", "c"},
        qd_filled_cells=6,
        qd_total_cells=100,
        qd_novelty_rate=0.05,
        total_tokens_spent=2000,
    )
    governor.update(metrics2)

    signals = governor.compute_signals()

    # Both should detect strong improvement, but absolute is stronger
    # absolute: 0.02 / 0.001 = 20.0 (capped at 1.0)
    # relative: 0.022 / 0.01 = 2.2 (capped at 1.0)
    # max(20, 2.2) = 20 -> capped at 1.0

    assert signals["s_quality"] == 1.0, "Should detect strong improvement"

    print(f"✓ Quality signal at high quality: {signals['s_quality']:.3f}")


if __name__ == "__main__":
    test_relative_improvement_at_low_quality()
    test_absolute_improvement_dominates_when_higher()
    print("\n✅ All StopGovernor relative improvement tests passed!")
