# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.strategies.proposal_sampling import IndependentSampling, PxNSampling, SameParentSampling


@pytest.mark.parametrize("count", [0, -1])
@pytest.mark.parametrize("strategy", [SameParentSampling, IndependentSampling])
def test_single_dimension_sampling_rejects_nonpositive_counts(strategy, count):
    with pytest.raises(ValueError, match="n must be positive"):
        strategy(n=count)


@pytest.mark.parametrize("count", [0, -1])
@pytest.mark.parametrize("dimension", ["p", "n"])
def test_pxn_sampling_rejects_nonpositive_counts(dimension, count):
    counts = {"p": 1, "n": 1}
    counts[dimension] = count
    with pytest.raises(ValueError, match=f"{dimension} must be positive"):
        PxNSampling(**counts)
