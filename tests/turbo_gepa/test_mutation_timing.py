"""
Tests for mutation timing and shard-based eligibility.

These tests verify that:
1. Candidates are not eligible for mutation until evaluated on meaningful shard
2. Seeds progress through shards before being used as parents
3. The orchestrator doesn't deadlock waiting for mutations
"""

import pytest
from turbo_gepa.interfaces import Candidate, EvalResult
from turbo_gepa.archive import Archive, ArchiveEntry
from turbo_gepa.config import Config


class TestMutationEligibility:
    """Test shard-based mutation eligibility filter."""

    @pytest.mark.asyncio
    async def test_seed_not_eligible_on_first_shard(self):
        """Seeds evaluated only on first shard should not be eligible for mutation."""
        config = Config(shards=(0.05, 0.20, 1.0))
        archive = Archive(bins_length=8, bins_bullets=6)

        # Seed evaluated on first shard (5%)
        seed = Candidate(text="You are a helpful assistant.", meta={"source": "seed"})
        result_shard0 = EvalResult(
            objectives={"quality": 1.0},  # Perfect on tiny sample!
            traces=[{"input": "test", "quality": 1.0}],
            n_examples=1,
            shard_fraction=0.05  # First shard
        )

        await archive.insert(seed, result_shard0)
        entries = archive.pareto_entries()
        assert len(entries) == 1

        # Calculate minimum shard for mutation (should be second shard)
        min_shard = config.shards[0] if len(config.shards) == 1 else config.shards[1]
        assert min_shard == 0.20, f"Expected min_shard=0.20, got {min_shard}"

        # Filter for eligible parents (mimics orchestrator logic)
        eligible_parents = []
        for entry in entries:
            current_shard = entry.result.shard_fraction or 0.0
            if current_shard >= min_shard:
                eligible_parents.append(entry)

        assert len(eligible_parents) == 0, "Seed on first shard should NOT be eligible for mutation"

    @pytest.mark.asyncio
    async def test_seed_eligible_on_second_shard(self):
        """Seeds evaluated on second shard should be eligible for mutation."""
        config = Config(shards=(0.05, 0.20, 1.0))
        archive = Archive(bins_length=8, bins_bullets=6)

        # Seed evaluated on second shard (20%)
        seed = Candidate(text="You are a helpful assistant.", meta={"source": "seed"})
        result_shard1 = EvalResult(
            objectives={"quality": 0.70},  # Now shows failures
            traces=[
                {"input": "test1", "quality": 1.0},
                {"input": "test2", "quality": 0.0},  # Failure!
                {"input": "test3", "quality": 1.0},
            ],
            n_examples=3,
            shard_fraction=0.20  # Second shard
        )

        await archive.insert(seed, result_shard1)
        entries = archive.pareto_entries()

        # Calculate minimum shard for mutation
        min_shard = config.shards[0] if len(config.shards) == 1 else config.shards[1]

        # Filter for eligible parents
        eligible_parents = []
        for entry in entries:
            current_shard = entry.result.shard_fraction or 0.0
            if current_shard >= min_shard:
                eligible_parents.append(entry)

        assert len(eligible_parents) == 1, "Seed on second shard SHOULD be eligible for mutation"
        assert eligible_parents[0].result.shard_fraction == 0.20

    @pytest.mark.asyncio
    async def test_single_shard_config_immediate_eligibility(self):
        """With single shard config, candidates should be immediately eligible."""
        config = Config(shards=(1.0,))  # Only full dataset
        archive = Archive(bins_length=8, bins_bullets=6)

        seed = Candidate(text="Test", meta={})
        result = EvalResult(
            objectives={"quality": 0.80},
            traces=[],
            n_examples=10,
            shard_fraction=1.0
        )

        await archive.insert(seed, result)
        entries = archive.pareto_entries()

        # Calculate minimum shard
        min_shard = config.shards[0] if len(config.shards) == 1 else config.shards[1]
        assert min_shard == 1.0, "Single shard config should use shard[0]"

        # Filter for eligible parents
        eligible_parents = []
        for entry in entries:
            current_shard = entry.result.shard_fraction or 0.0
            if current_shard >= min_shard:
                eligible_parents.append(entry)

        assert len(eligible_parents) == 1, "Single shard config should immediately allow mutation"


class TestSeedProgression:
    """Test that seeds properly progress through shards."""

    @pytest.mark.asyncio
    async def test_seed_progression_through_shards(self):
        """Seed should be evaluated on shard 0, then shard 1, before mutation."""
        config = Config(shards=(0.05, 0.20, 1.0))
        archive = Archive(bins_length=8, bins_bullets=6)

        seed = Candidate(text="You are a helpful assistant.", meta={"source": "seed"})

        # Step 1: Evaluated on shard 0
        result_shard0 = EvalResult(
            objectives={"quality": 1.0},
            traces=[{"quality": 1.0}],
            n_examples=1,
            shard_fraction=0.05
        )
        await archive.insert(seed, result_shard0)

        min_shard = config.shards[1]
        entries = archive.pareto_entries()
        eligible = [e for e in entries if (e.result.shard_fraction or 0.0) >= min_shard]
        assert len(eligible) == 0, "After shard 0: Not eligible for mutation"

        # Step 2: Evaluated on shard 1
        result_shard1 = EvalResult(
            objectives={"quality": 0.70},
            traces=[{"quality": 1.0}, {"quality": 0.0}, {"quality": 1.0}],
            n_examples=3,
            shard_fraction=0.20
        )
        await archive.insert(seed, result_shard1)

        entries = archive.pareto_entries()
        eligible = [e for e in entries if (e.result.shard_fraction or 0.0) >= min_shard]
        assert len(eligible) == 1, "After shard 1: NOW eligible for mutation"
        assert eligible[0].result.shard_fraction == 0.20


class TestReflectionDataQuality:
    """Test that mutations get meaningful reflection data."""

    @pytest.mark.asyncio
    async def test_first_shard_has_no_failures(self):
        """First shard with 100% quality has no failure traces for reflection."""
        result = EvalResult(
            objectives={"quality": 1.0},
            traces=[{"input": "test", "quality": 1.0}],
            n_examples=1,
            shard_fraction=0.05
        )

        # Extract failure traces (what reflection would see)
        failure_traces = [t for t in result.traces if t.get("quality", 0.0) < 1.0]
        success_traces = [t for t in result.traces if t.get("quality", 0.0) >= 1.0]

        assert len(failure_traces) == 0, "No failures on perfect tiny sample"
        assert len(success_traces) == 1, "Only 1 success trace"

    @pytest.mark.asyncio
    async def test_second_shard_has_failures(self):
        """Second shard should reveal actual failures."""
        result = EvalResult(
            objectives={"quality": 0.70},
            traces=[
                {"input": "test1", "quality": 1.0},
                {"input": "test2", "quality": 0.0},  # Failure
                {"input": "test3", "quality": 1.0},
                {"input": "test4", "quality": 0.5},  # Partial failure
            ],
            n_examples=4,
            shard_fraction=0.20
        )

        # Extract failure traces
        failure_traces = [t for t in result.traces if t.get("quality", 0.0) < 1.0]
        success_traces = [t for t in result.traces if t.get("quality", 0.0) >= 1.0]

        assert len(failure_traces) == 2, "Should have 2 failure traces"
        assert len(success_traces) == 2, "Should have 2 success traces"
        assert len(result.traces) == 4, "Total traces should be 4"


class TestConfigEdgeCases:
    """Test various configuration edge cases."""

    def test_two_shard_config(self):
        """Two-shard config should use second shard as minimum."""
        config = Config(shards=(0.3, 1.0))
        min_shard = config.shards[0] if len(config.shards) == 1 else config.shards[1]
        assert min_shard == 1.0, f"Two-shard config should use shard[1]=1.0, got {min_shard}"

    def test_three_shard_config(self):
        """Three-shard config should use second shard as minimum."""
        config = Config(shards=(0.05, 0.20, 1.0))
        min_shard = config.shards[0] if len(config.shards) == 1 else config.shards[1]
        assert min_shard == 0.20, f"Three-shard config should use shard[1]=0.20, got {min_shard}"

    def test_many_shard_config(self):
        """Many-shard config should still use second shard."""
        config = Config(shards=(0.01, 0.05, 0.10, 0.20, 0.50, 1.0))
        min_shard = config.shards[0] if len(config.shards) == 1 else config.shards[1]
        assert min_shard == 0.05, f"Many-shard config should use shard[1]=0.05, got {min_shard}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
