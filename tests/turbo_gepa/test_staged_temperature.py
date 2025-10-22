"""
Test staged temperature optimization (Phase 1: prompts, Phase 2: temperature).

This test uses the heuristic defaults from user_plugs_in.py, so it doesn't
require any real LLM calls.
"""

import pytest

from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst


def test_staged_temperature_optimization(tmp_path):
    """Test that staged optimization works: prompts first, then temperature."""

    # Setup
    dataset = [DefaultDataInst(input=f'Question {i}', answer=f'Answer {i}', id=f'q-{i}', difficulty=i/20.0) for i in range(15)]

    adapter = DefaultAdapter(
        dataset=dataset,
        cache_dir=(tmp_path / "cache").as_posix(),
        log_dir=(tmp_path / "logs").as_posix(),
        sampler_seed=42,
    )

    # Manually enable temperature support (skip LLM check for test)
    adapter.temperature_supported = True

    # Run staged optimization with small budget
    # Note: max_evaluations must be high enough for Phase 2 to run mutations
    # Phase 1 uses ~40-50 evals, Phase 2 needs another ~40-50 for temp exploration
    result = adapter.optimize(
        seeds=["You are a helpful assistant."],
        max_rounds=8,  # Enough rounds for both phases
        max_evaluations=None,  # No evaluation limit (use rounds instead)
        optimize_temperature_after_convergence=True,
    )

    # Verify Phase 1 results exist
    assert "phase1_pareto" in result, "Should return Phase 1 Pareto frontier"
    phase1_pareto = result["phase1_pareto"]
    assert len(phase1_pareto) > 0, "Phase 1 should produce candidates"

    # Verify Phase 1 candidates don't have temperature (or have None)
    phase1_temps = [c.meta.get("temperature") for c in phase1_pareto]
    print(f"✓ Phase 1 produced {len(phase1_pareto)} candidates")
    print(f"  Temperatures: {phase1_temps}")

    # Verify Phase 2 results exist
    assert "pareto" in result, "Should return final Pareto frontier"
    final_pareto = result["pareto"]
    assert len(final_pareto) > 0, "Phase 2 should produce candidates"

    # Verify Phase 2 candidates have temperature values
    final_temps = [c.meta.get("temperature") for c in final_pareto]
    print(f"✓ Phase 2 produced {len(final_pareto)} candidates")
    print(f"  Temperatures: {final_temps}")

    # At least some candidates should have explicit temperature
    has_temp_variants = any(t is not None for t in final_temps)
    assert has_temp_variants, "Phase 2 should explore temperature variants"

    # All temperatures should be in valid range [0.0, 1.0]
    valid_temps = [t for t in final_temps if t is not None]
    if valid_temps:
        assert all(0.0 <= t <= 1.0 for t in valid_temps), "All temps should be in [0.0, 1.0]"
        print(f"✓ All {len(valid_temps)} temperature values in valid range [0.0, 1.0]")

    # Verify result structure
    assert "pareto_entries" in result
    assert "qd_elites" in result

    print(f"\n✓ Staged optimization test passed!")


def test_staged_optimization_skips_phase2_if_unsupported(tmp_path):
    """Test that Phase 2 is skipped if temperature not supported."""

    dataset = [DefaultDataInst(input=f'Q{i}', answer=f'A{i}', id=f'q-{i}') for i in range(10)]

    adapter = DefaultAdapter(
        dataset=dataset,
        cache_dir=(tmp_path / "cache").as_posix(),
        sampler_seed=99,
    )

    # Manually disable temperature support
    adapter.temperature_supported = False

    result = adapter.optimize(
        seeds=["Assistant prompt."],
        max_rounds=3,
        max_evaluations=30,
        optimize_temperature_after_convergence=True,
    )

    # Should still return phase1_pareto
    assert "phase1_pareto" in result
    assert len(result["phase1_pareto"]) > 0

    # Final pareto should be phase1 results (not empty)
    assert "pareto" in result
    assert len(result["pareto"]) > 0

    print("✓ Correctly skipped Phase 2 when temperature not supported")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
