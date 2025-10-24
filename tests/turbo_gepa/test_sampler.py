from turbo_gepa.sampler import InstanceSampler


def test_sampler_hardness_override():
    """Test that hardness sampling reserves up to 25% of shard, overriding random sample."""
    sampler = InstanceSampler(example_ids=["ex1", "ex2", "ex3", "ex4", "ex5", "ex6", "ex7", "ex8"], seed=42)

    # Register some hard examples
    sampler.register_hard_examples(["ex2", "ex5"])

    # Sample a shard of size 8 - should include at least 1 hardness example (25% of 8 = 2)
    shard = sampler.sample_shard(round_id=0, k=8)

    assert len(shard) == 8
    assert "ex2" in shard or "ex5" in shard  # At least one hardness example should be included


def test_sampler_hardness_cap_at_25_percent():
    """Test that hardness sampling never exceeds 25% of shard size."""
    sampler = InstanceSampler(example_ids=[f"ex{i}" for i in range(20)], seed=42)

    # Register many hard examples
    sampler.register_hard_examples([f"ex{i}" for i in range(10)])

    # Sample a shard of size 8 - should include max 2 hardness examples (25% of 8)
    shard = sampler.sample_shard(round_id=0, k=8)

    assert len(shard) == 8
    # Count how many hardness examples are in the shard
    hardness_count = sum(1 for ex_id in shard if ex_id in [f"ex{i}" for i in range(10)])
    assert hardness_count <= 2  # Should be at most 25% of 8


def test_sampler_hardness_respects_shard_size():
    """Test that total shard size never exceeds k even with hardness sampling."""
    sampler = InstanceSampler(example_ids=[f"ex{i}" for i in range(20)], seed=42)

    # Register some hard examples
    sampler.register_hard_examples(["ex0", "ex1", "ex2"])

    # Sample various shard sizes
    for k in [4, 8, 12, 16]:
        shard = sampler.sample_shard(round_id=0, k=k)
        assert len(shard) == k  # Should never exceed requested size


def test_sampler_random_without_hardness():
    """Test that sampling works correctly when no hard examples are registered."""
    sampler = InstanceSampler(example_ids=[f"ex{i}" for i in range(10)], seed=42)

    # Sample without any hardness examples
    shard = sampler.sample_shard(round_id=0, k=5)

    assert len(shard) == 5
    assert all(ex_id.startswith("ex") for ex_id in shard)


def test_sampler_hardness_min_one():
    """Test that at least 1 hardness example is included when available."""
    sampler = InstanceSampler(example_ids=[f"ex{i}" for i in range(10)], seed=42)

    # Register one hard example
    sampler.register_hard_examples(["ex7"])

    # Sample a small shard - should include the hardness example (min 1)
    shard = sampler.sample_shard(round_id=0, k=3)

    assert len(shard) == 3
    assert "ex7" in shard  # Should include the hardness example
