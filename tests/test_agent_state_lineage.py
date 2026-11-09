# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
from itertools import count

from test_multi_proposal_invariants import TRAIN, SyntheticAdapter, SyntheticReflectionLM

import gepa
from gepa.core.state import GEPAState
from gepa.strategies.proposal_sampling import ProposalTask
from gepa.strategies.proposal_selection import AllImprovements
from gepa.utils.stop_condition import MaxCandidateProposalsStopper


def test_write_agent_state_preserves_each_accepted_candidates_parent(tmp_path):
    class MixedParentSampling:
        def sample_tasks(self, state, candidate_selector, batch_sampler, trainset):
            ids = list(trainset.all_ids())
            return [
                ProposalTask(parent, state.program_candidates[parent], ids, trainset.fetch(ids))
                for parent in range(min(2, len(state.program_candidates)))
            ]

    levels = count(1)
    result = gepa.optimize(
        seed_candidate={"system_prompt": "LEVEL=0"},
        trainset=TRAIN,
        valset=TRAIN,
        adapter=SyntheticAdapter(),
        reflection_lm=lambda _: f"```\nLEVEL={next(levels)}\n```",
        stop_callbacks=MaxCandidateProposalsStopper(2),
        sampling_strategy=MixedParentSampling(),
        selection_strategy=AllImprovements(),
        run_dir=str(tmp_path),
        write_agent_state=True,
        seed=0,
        display_progress_bar=False,
    )

    # The second iteration accepts two candidates with different parents.
    assert result.parents == [[None], [0], [0], [1]]
    state = GEPAState.load(str(tmp_path))
    ids = state.iteration_ids_by_candidate_idx
    for candidate_idx, parents in enumerate(result.parents):
        meta_path = tmp_path / "iterations" / ids[candidate_idx] / "meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["parent_iteration_ids"] == [ids[parent] for parent in parents if parent is not None]


def test_write_agent_state_preserves_parent_of_new_seed_on_resume(tmp_path):
    settings = {
        "trainset": TRAIN,
        "valset": TRAIN,
        "adapter": SyntheticAdapter(),
        "reflection_lm": SyntheticReflectionLM(),
        "run_dir": str(tmp_path),
        "write_agent_state": True,
        "display_progress_bar": False,
    }
    gepa.optimize(seed_candidate={"system_prompt": "LEVEL=0"}, max_metric_calls=len(TRAIN), **settings)
    result = gepa.optimize(seed_candidate={"system_prompt": "LEVEL=9"}, max_metric_calls=0, **settings)

    assert result.parents == [[None], [0]]
    state = GEPAState.load(str(tmp_path))
    meta_path = tmp_path / "iterations" / state.iteration_ids_by_candidate_idx[1] / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["parent_iteration_ids"] == ["seed"]
