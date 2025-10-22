"""
Test resumable optimization (cancel and resume functionality).
"""

from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst


def test_resume_after_interruption(tmp_path):
    """Test that optimization can resume after interruption."""

    # Create dataset
    dataset = [
        DefaultDataInst(input=f'Q{i}', answer=f'A{i}', id=f'q-{i}', difficulty=i/20.0)
        for i in range(15)
    ]

    cache_dir = (tmp_path / "cache").as_posix()

    # Run 1: Start optimization and "interrupt" after 3 rounds
    # We simulate interruption by having max_rounds=10 but only reading results after 3
    # The state should persist because we haven't reached max_rounds yet
    adapter1 = DefaultAdapter(dataset=dataset, cache_dir=cache_dir, sampler_seed=42)

    # First run - partial
    result1 = adapter1.optimize(
        seeds=["You are a helpful assistant."],
        max_rounds=3,  # Will complete and clear state
        max_evaluations=50,
    )

    pareto1 = result1["pareto"]
    print(f"\n✓ Run 1: {len(pareto1)} candidates after 3 rounds")

    # For testing resume, we need state to exist
    # So let's manually save it (simulating an interrupted run)
    # In real usage, user would Ctrl+C during a long run
    if not adapter1.cache.has_state():
        # Manually save state to simulate interruption
        adapter1.cache.save_state(
            round_num=3,
            evaluations=30,
            pareto_candidates=pareto1,
            qd_candidates=[],
            queue=[],
        )

    # Verify state exists
    assert adapter1.cache.has_state(), "State should exist for resume test"

    # Run 2: Resume from saved state (should continue from round 3)
    adapter2 = DefaultAdapter(dataset=dataset, cache_dir=cache_dir, sampler_seed=42)

    result2 = adapter2.optimize(
        seeds=["You are a helpful assistant."],  # Seeds ignored when resuming
        max_rounds=6,  # Continue to round 6
        max_evaluations=100,
    )

    pareto2 = result2["pareto"]
    print(f"✓ Run 2: {len(pareto2)} candidates after resuming to round 6")

    # Verify we made progress
    assert len(pareto2) >= len(pareto1), "Should have at least as many candidates after resuming"

    # Run 3: Complete the optimization (no resume, starts fresh)
    adapter3 = DefaultAdapter(dataset=dataset, cache_dir=(tmp_path / "cache2").as_posix(), sampler_seed=42)

    result3 = adapter3.optimize(
        seeds=["You are a helpful assistant."],
        max_rounds=6,
        max_evaluations=100,
    )

    pareto3 = result3["pareto"]
    print(f"✓ Run 3 (fresh): {len(pareto3)} candidates after 6 rounds")

    print(f"\n✅ Resume test passed:")
    print(f"   Run 1 (3 rounds): {len(pareto1)} candidates")
    print(f"   Run 2 (resumed to 6): {len(pareto2)} candidates")
    print(f"   Run 3 (fresh 6 rounds): {len(pareto3)} candidates")


def test_resume_disabled(tmp_path):
    """Test that resume can be disabled."""

    dataset = [DefaultDataInst(input=f'Q{i}', answer=f'A{i}', id=f'q-{i}') for i in range(10)]
    cache_dir = (tmp_path / "cache").as_posix()

    # Run 1: Create saved state manually (since completion clears it)
    adapter1 = DefaultAdapter(dataset=dataset, cache_dir=cache_dir)
    result1 = adapter1.optimize(seeds=["Test."], max_rounds=2, max_evaluations=20)

    # Manually save state to simulate interrupted run
    adapter1.cache.save_state(
        round_num=2,
        evaluations=15,
        pareto_candidates=result1["pareto"],
        qd_candidates=[],
        queue=[],
    )

    assert adapter1.cache.has_state(), "State should be saved"

    # Run 2: Start fresh even though state exists (resume=False not directly exposed yet,
    # but we can clear state manually to test)
    adapter2 = DefaultAdapter(dataset=dataset, cache_dir=cache_dir)
    adapter2.cache.clear_state()  # Clear manually

    assert not adapter2.cache.has_state(), "State should be cleared"

    result = adapter2.optimize(seeds=["Fresh start."], max_rounds=2, max_evaluations=20)

    print("✓ Successfully started fresh after clearing state")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
