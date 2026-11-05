import copy
import random
from types import SimpleNamespace

import pytest

from gepa.evidence.proposer_state import restore_proposer, snapshot_proposer
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector
from gepa.strategies.proposal_sampling import SingleMutationSampling
from test_sequential_evidence_native import run


def native_proposer(seed=0):
    rng = random.Random(seed)
    return SimpleNamespace(
        batch_sampler=EpochShuffledBatchSampler(2, rng=rng),
        candidate_selector=ParetoCandidateSelector(rng=rng),
        module_selector=RoundRobinReflectionComponentSelector(),
        sampling_strategy=SingleMutationSampling(),
        perfect_score=1, skip_perfect_score=False,
    )


def test_rng_and_epoch_sampler_continue_from_checkpoint_not_seed():
    from gepa.core.data_loader import ensure_loader
    loader = ensure_loader(list(range(7)))
    state = SimpleNamespace(i=0)
    original = native_proposer()
    original.batch_sampler.next_minibatch_ids(loader, state)
    original.candidate_selector.rng.random()
    saved = snapshot_proposer(original)
    expected = []
    for iteration in range(1, 10):
        state.i = iteration
        expected.append((original.candidate_selector.rng.random(), original.batch_sampler.next_minibatch_ids(loader, state)))
    resumed = native_proposer()
    restore_proposer(resumed, saved, list(loader.all_ids()))
    actual = []
    for iteration in range(1, 10):
        state.i = iteration
        actual.append((resumed.candidate_selector.rng.random(), resumed.batch_sampler.next_minibatch_ids(loader, state)))
    assert actual == expected
    assert resumed.batch_sampler.rng is resumed.candidate_selector.rng


@pytest.mark.parametrize("kind", ["rng", "ids", "frequency", "signature", "alias"])
def test_corrupt_proposer_snapshot_rejected_before_mutation(kind):
    proposer = native_proposer()
    before = snapshot_proposer(proposer)
    broken = copy.deepcopy(before)
    if kind == "rng": broken["rng_groups"][0]["state"] = [0]
    if kind == "ids": broken["sampler"]["shuffled_ids"] = [99, 99]
    if kind == "frequency": broken["sampler"]["id_freqs"] = [[0, -1]]
    if kind == "signature": broken["signature"]["minibatch_size"] = 20
    if kind == "alias": broken["rng_groups"][0]["roles"] = ["unknown"]
    with pytest.raises(ValueError):
        restore_proposer(proposer, broken, list(range(7)))
    assert snapshot_proposer(proposer) == before


def test_native_checkpoint_contains_proposer_rng_and_resume_rejects_changed_minibatch(tmp_path):
    from gepa.core.state import GEPAState
    run(tmp_path, budget=3)
    state = GEPAState.load(str(tmp_path))
    assert state.sequential_proposer_state["rng_groups"]
    with pytest.raises(ValueError, match="context|proposer"):
        run(tmp_path, budget=30, batch_sampler=EpochShuffledBatchSampler(2), reflection_minibatch_size=None)
