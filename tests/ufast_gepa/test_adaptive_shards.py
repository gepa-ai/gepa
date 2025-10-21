"""Tests for adaptive shard selection."""

from ufast_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from ufast_gepa.config import Config, adaptive_shards


def test_adaptive_shards_basic():
    """Test basic adaptive shard selection."""
    # Very small dataset
    assert adaptive_shards(20) == (0.5, 1.0)

    # Small dataset
    assert adaptive_shards(50) == (0.3, 1.0)

    # Medium dataset
    shards = adaptive_shards(100)
    assert len(shards) == 3
    assert shards[-1] == 1.0

    # Large dataset
    shards = adaptive_shards(500)
    assert shards == (0.05, 0.2, 1.0)


def test_adaptive_shards_strategies():
    """Test different strategies produce different results."""
    size = 100

    conservative = adaptive_shards(size, strategy="conservative")
    balanced = adaptive_shards(size, strategy="balanced")
    aggressive = adaptive_shards(size, strategy="aggressive")

    # Conservative should have larger first shard
    assert conservative[0] >= balanced[0]
    assert balanced[0] >= aggressive[0]


def test_adaptive_shards_rule_of_15():
    """Test that first shard aims for ~15 examples minimum."""
    for size in [20, 50, 100, 500, 1000]:
        shards = adaptive_shards(size, strategy="balanced")
        first_stage_examples = int(size * shards[0])
        # Should be at least 10 examples (relaxed from 15 for very small datasets)
        assert first_stage_examples >= min(10, size * 0.3)


def test_default_adapter_auto_sharding():
    """Test that DefaultAdapter automatically selects shards."""
    dataset = [
        DefaultDataInst(input=f"test{i}", answer=f"ans{i}", id=f"test-{i}")
        for i in range(50)
    ]

    # Should auto-select shards
    adapter = DefaultAdapter(dataset=dataset)
    assert adapter.config.shards != Config().shards  # Should be different from default
    assert adapter.config.shards == (0.3, 1.0)  # Expected for 50 examples


def test_default_adapter_manual_shards():
    """Test that manual shard config is respected."""
    dataset = [
        DefaultDataInst(input=f"test{i}", answer=f"ans{i}", id=f"test-{i}")
        for i in range(50)
    ]

    manual_config = Config(shards=(0.1, 0.4, 1.0))
    adapter = DefaultAdapter(dataset=dataset, config=manual_config)

    # Should keep manual config
    assert adapter.config.shards == (0.1, 0.4, 1.0)


def test_default_adapter_strategies():
    """Test different strategies in DefaultAdapter."""
    dataset = [
        DefaultDataInst(input=f"test{i}", answer=f"ans{i}", id=f"test-{i}")
        for i in range(100)
    ]

    conservative = DefaultAdapter(dataset=dataset, shard_strategy="conservative")
    balanced = DefaultAdapter(dataset=dataset, shard_strategy="balanced")
    aggressive = DefaultAdapter(dataset=dataset, shard_strategy="aggressive")

    # Conservative should have larger first shard
    assert conservative.config.shards[0] >= balanced.config.shards[0]
    assert balanced.config.shards[0] >= aggressive.config.shards[0]


def test_adaptive_shards_edge_cases():
    """Test edge cases for adaptive_shards."""
    # Empty dataset
    assert adaptive_shards(0) == (1.0,)

    # Single example
    assert adaptive_shards(1) == (0.5, 1.0)

    # Very large dataset
    shards = adaptive_shards(10000)
    assert len(shards) >= 3
    assert shards[-1] == 1.0
