"""Candidate-local curricula with immutable shared prefixes; no model/API calls."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ListDataLoader
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.strategies.proposal_sampling import SingleMutationSampling
from gepa.strategies.task_curator import TaskCuratorContractError, TaskCuratorSampler


def run_and_select_latest(prompt):
    choice = "retained"
    if "Eligible successes (references" in prompt:
        eligible = json.loads(prompt.split("use its evidence above):\n", 1)[1])
        choice = "after" if "after" in eligible else "before"
    return json.dumps({"decision": "run", "selected_success": choice})


def setup(size=1, **kwargs):
    sampler = TaskCuratorSampler(minibatch_size=1, lm=Mock(side_effect=run_and_select_latest), **kwargs)
    loader = ListDataLoader([{"task": i} for i in range(size)])
    state = SimpleNamespace(i=0, full_program_trace=[{"i": 0}])
    return sampler, loader, state


def advance(state):
    state.i += 1
    state.full_program_trace.append({"i": state.i})


def evaluate(sampler, loader, state, parent, score=0):
    ids = sampler.next_minibatch_ids_for_candidate(loader, state, candidate_idx=parent)
    batch = EvaluationBatch(
        scores=[score] * len(ids),
        outputs=[None] * len(ids),
        trajectories=[{"owner": parent, "task": tid, "score": score} for tid in ids],
    )
    sampler.observe_evaluation(iteration=state.i + 1, task_ids=ids, evaluation=batch, state=state)
    return ids, batch


def mutation(sampler, state, ids, before, *, child, score=1):
    parent = sampler._active_candidate_idx
    proposal = CandidateProposal(
        candidate={"p": str(child)},
        parent_program_ids=[parent],
        subsample_indices=ids,
        eval_before=SubsampleEvaluation(scores=before.scores, outputs=before.outputs, trajectories=before.trajectories),
        eval_after=SubsampleEvaluation(
            scores=[score] * len(ids),
            outputs=[None] * len(ids),
            trajectories=[{"owner": child, "task": tid, "score": score} for tid in ids],
        ),
    )
    sampler.observe_proposal(
        iteration=state.i + 1, proposal=proposal, accepted=child is not None, candidate_idx=child, state=state
    )


def snapshot(sampler):
    current = sampler._active_candidate_idx
    result = {}
    for idx, branch in sampler._branches.items():
        sampler._active_candidate_idx = idx
        result[idx] = {
            "history": sampler.history,
            "successes": deepcopy(branch.success_states),
            "retained": dict(branch.retained_success),
            "cooldown": dict(branch.next_run_epoch),
            "skip_types": dict(branch.skip_types),
            "epoch": branch.epoch,
            "order": list(branch.epoch_order),
            "position": branch.position,
            "selected": branch.selected_in_epoch,
        }
    sampler._active_candidate_idx = current
    return result


def test_fork_reuses_prefix_without_retroactive_parent_or_sibling_updates():
    s, loader, state = setup(size=2)
    ids, before = evaluate(s, loader, state, 0)
    tid = ids[0]
    prefix = s._branch.history_heads[tid]
    mutation(s, state, ids, before, child=1)
    assert s._branches[1].history_heads[tid].previous is prefix
    assert s._branches[0].history_heads[tid] is prefix
    child_one = snapshot(s)[1]

    advance(state)
    ids, before = evaluate(s, loader, state, 0)
    assert ids[0] != tid  # returning to parent continues its own epoch cursor
    mutation(s, state, ids, before, child=2)
    assert snapshot(s)[1] == child_one
    assert s._branches[2].history_heads[tid] is prefix
    assert all("/after" not in sid for states in s._branches[0].success_states.values() for sid in states)

    # Public audit views cannot mutate any of the shared records.
    view = s.history
    view[tid][-1]["evaluator_feedback"]["owner"] = "corrupted"
    assert snapshot(s)[1] == child_one
    assert s.history[tid][-1]["evaluator_feedback"]["owner"] == 0


def test_sibling_anchors_and_mastery_cooldown_do_not_cross_branches():
    s, loader, state = setup()
    ids, before = evaluate(s, loader, state, 0, -1)
    mutation(s, state, ids, before, child=1, score=1)
    advance(state)
    ids, before = evaluate(s, loader, state, 0, -1)
    mutation(s, state, ids, before, child=2, score=0)

    advance(state)
    s.lm.side_effect = None
    s.lm.return_value = json.dumps(
        {
            "decision": "skip",
            "task_status": "mastered",
            "skip_epochs": 10,
            "skip_type": "mastered",
            "selected_success": "after",
        }
    )
    ids, before = evaluate(s, loader, state, 1)
    # One-task branch advances to its forced revisit, without extra LM calls.
    assert s._branch.epoch == 11
    assert s._branch.retained_success[0].endswith("/iteration:1/after")
    branch_one = snapshot(s)[1]

    advance(state)
    s.lm.side_effect = run_and_select_latest
    ids, before = evaluate(s, loader, state, 2)
    assert s._branch.epoch == 2  # inherited epoch 1 from its own later fork
    assert not s._branch.retained_success
    assert not s._branch.next_run_epoch
    assert all(row.get("task_status") != "mastered" for row in s._older_history_for_prompt(0))
    assert snapshot(s)[1] == branch_one

    # The adapter receives the usual singleton batch and no foreign anchor.
    result = s.build_reflective_dataset(
        candidate={"p": "2"},
        task_ids=ids,
        evaluation=before,
        components_to_update=["p"],
        state=state,
        make_reflective_dataset=lambda *_: {"p": [{"Feedback": "failure"}]},
    )
    assert "HistoricalSuccess" not in result["p"][0]

    advance(state)
    ids, before = evaluate(s, loader, state, 1)
    result = s.build_reflective_dataset(
        candidate={"p": "1"},
        task_ids=ids,
        evaluation=before,
        components_to_update=["p"],
        state=state,
        make_reflective_dataset=lambda *_: {"p": [{"Feedback": "failure"}]},
    )
    assert result["p"][0]["HistoricalSuccess"]["successful_state"]["owner"] == 1


def test_rejected_trial_is_diagnostic_only_on_the_parent_that_tried_it():
    s, loader, state = setup()
    ids, before = evaluate(s, loader, state, 0)
    mutation(s, state, ids, before, child=1, score=0)
    child_snapshot = snapshot(s)[1]
    advance(state)
    ids, before = evaluate(s, loader, state, 0)
    mutation(s, state, ids, before, child=None, score=1)
    assert snapshot(s)[1] == child_snapshot
    pair = s._prompt_view(0)[-1]
    assert pair["mutation_accepted"] is False
    assert pair["before"]["candidate_idx"] == 0
    assert pair["after"]["candidate_idx"] is None
    assert s._latest_pool_candidates(0)["after"].endswith("/iteration:2/after")
    assert len(s._execution_units(0)) == 2  # a before/after pair still counts once


def test_greedy_chain_inherits_full_progress_without_copying_or_resetting():
    s, loader, state = setup(size=3, history_window=1)
    for parent in range(6):
        ids, before = evaluate(s, loader, state, parent)
        epoch, position = s._branch.epoch, s._branch.position
        prefix = s._branch.history_heads[ids[0]]
        mutation(s, state, ids, before, child=parent + 1)
        assert s._branch.epoch == epoch
        assert s._branch.position == position
        assert s._branch.history_heads[ids[0]].previous is prefix
        if parent < 5:
            advance(state)
    assert s._branch.epoch == 1
    assert s._branch.position == 3
    assert all(len(s._execution_units(tid)) == 2 for tid in loader.all_ids())
    assert all(len(s._history_for_prompt(tid)) == 1 for tid in loader.all_ids())


def test_resume_restores_every_branch_and_continues_the_same_curriculum():
    s, loader, state = setup(size=3)
    for parent, child in [(0, 1), (0, 2), (1, 3), (2, None), (0, None)]:
        ids, before = evaluate(s, loader, state, parent)
        mutation(s, state, ids, before, child=child)
        advance(state)
    trace = deepcopy(state.full_program_trace[:-1])
    resumed, _, _ = setup(size=3)
    resumed.load_from_trace(trace, loader, require_history=True)
    assert snapshot(resumed) == snapshot(s)
    resumed_state = deepcopy(state)
    for parent in [3, 2, 1, 0, 2]:
        evaluate(s, loader, state, parent)
        evaluate(resumed, loader, resumed_state, parent)
        assert s.lm.call_args == resumed.lm.call_args
        assert state.full_program_trace[-1] == resumed_state.full_program_trace[-1]
        assert snapshot(s) == snapshot(resumed)
        advance(state)
        advance(resumed_state)


@pytest.mark.parametrize("corruption", ["parent", "child", "ownership", "legacy", "incomplete_acceptance"])
def test_resume_rejects_invalid_lineage(corruption):
    s, loader, state = setup()
    ids, before = evaluate(s, loader, state, 0)
    mutation(s, state, ids, before, child=1)
    trace = deepcopy(state.full_program_trace)
    if corruption == "parent":
        trace[0]["task_curator_parent_idx"] = 99
    elif corruption == "child":
        trace[0]["task_curator_child_idx"] = 0
        trace[0]["task_curator_mutation_results"][0]["after"]["candidate_idx"] = 0
    elif corruption == "ownership":
        trace[0]["task_curator_mutation_results"][0]["before"]["candidate_idx"] = 99
    elif corruption == "incomplete_acceptance":
        trace[0]["new_program_idx"] = 1
        trace[0].pop("task_curator_child_idx")
        trace[0].pop("task_curator_mutation_results")
    else:
        trace[0]["task_curator_settings"]["version"] = 7
    resumed, _, _ = setup()
    with pytest.raises(TaskCuratorContractError):
        resumed.load_from_trace(trace, loader, require_history=True)


def test_sampling_requires_explicit_known_candidate_identity():
    s, loader, state = setup()
    with pytest.raises(TaskCuratorContractError, match="candidate-aware"):
        s.next_minibatch_ids(loader, state)
    with pytest.raises(TaskCuratorContractError, match="No curator history"):
        s.next_minibatch_ids_for_candidate(loader, state, candidate_idx=99)
    state.program_candidates = [{"p": "seed"}]
    tasks = SingleMutationSampling().sample_tasks(state, Mock(select_candidate_idx=Mock(return_value=0)), s, loader)
    assert tasks[0].parent_idx == 0
    assert state.full_program_trace[-1]["task_curator_parent_idx"] == 0


def test_curator_rejects_merge_before_any_evaluation(tmp_path):
    s, _, _ = setup()
    adapter = Mock()
    with pytest.raises(ValueError, match="multi-parent merge"):
        gepa.optimize(
            seed_candidate={"p": "seed"},
            trainset=[{}],
            valset=[{}],
            adapter=adapter,
            batch_sampler=s,
            reflection_lm=Mock(),
            use_merge=True,
            max_metric_calls=20,
            run_dir=str(tmp_path),
            display_progress_bar=False,
        )
    adapter.evaluate.assert_not_called()


def test_child_inherits_live_cooldown_but_other_branches_do_not_advance_it():
    s, loader, state = setup(size=2)
    ids, before = evaluate(s, loader, state, 0)
    mutation(s, state, ids, before, child=1)
    advance(state)
    evaluate(s, loader, state, 1)  # finish epoch 0
    advance(state)

    def skip_zero(prompt):
        tid = json.loads(prompt.split("Task ID:\n")[1].split("\n\n", 1)[0])
        if tid == 0:
            return json.dumps(
                {
                    "decision": "skip",
                    "skip_epochs": 5,
                    "skip_type": "mastered",
                    "task_status": "mastered",
                    "selected_success": "after",
                }
            )
        return run_and_select_latest(prompt)

    s.lm.side_effect = skip_zero
    for _ in range(3):
        ids, before = evaluate(s, loader, state, 1)
        if s._branch.next_run_epoch:
            break
        advance(state)
    assert ids == [1]
    assert s._branch.next_run_epoch == {0: 6}
    mutation(s, state, ids, before, child=2)
    assert s._branch.next_run_epoch == {0: 6}
    parent_state, child_state = snapshot(s)[1], snapshot(s)[2]
    s.lm.side_effect = run_and_select_latest
    for _ in range(4):
        advance(state)
        evaluate(s, loader, state, 0)
    assert snapshot(s)[1] == parent_state
    assert snapshot(s)[2] == child_state
    restored, _, _ = setup(size=2)
    restored.load_from_trace(deepcopy(state.full_program_trace), loader, require_history=True)
    assert snapshot(restored) == snapshot(s)


@pytest.mark.parametrize("accepted", [False, True])
def test_failed_evidence_extraction_cannot_leave_a_partial_branch(accepted):
    s, loader, state = setup()
    ids, before = evaluate(s, loader, state, 0)
    old = snapshot(s)
    trace = deepcopy(state.full_program_trace)
    error = RuntimeError("extractor failed")
    s.success_extractor = Mock(side_effect=error)
    with pytest.raises(TaskCuratorContractError) as caught:
        mutation(s, state, ids, before, child=1 if accepted else None)
    assert caught.value.__cause__ is error
    assert snapshot(s) == old
    assert state.full_program_trace == trace


class LatestSelector:
    def select_candidate_idx(self, state):
        return len(state.program_candidates) - 1


class AlternatingSelector:
    def select_candidate_idx(self, state):
        return [0, 0, 1, 2, 0, 3][state.i]


class BranchAdapter:
    propose_new_texts = None

    def evaluate(self, batch, candidate, capture_traces=False):
        # Parent/child rollouts stay imperfect so every requested iteration proposes.
        # A deterministic criterion below controls acceptance independently of sampling.
        return EvaluationBatch(
            scores=[0] * len(batch),
            outputs=[None] * len(batch),
            trajectories=[{"task": row, "prompt": candidate["p"]} for row in batch] if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, evaluation, components):
        return {c: [{"Feedback": row} for row in evaluation.trajectories] for c in components}


@pytest.mark.parametrize("selector", [LatestSelector(), AlternatingSelector(), "pareto"])
def test_engine_registers_and_resumes_branches_with_actual_selection(tmp_path, selector):
    s, _, _ = setup(size=3)

    def run(sampler, stop):
        lm = Mock(side_effect=[f"```\nmutation-{i}\n```" for i in range(stop - 3, stop)])
        return gepa.optimize(
            seed_candidate={"p": "seed"},
            trainset=[{"task": i} for i in range(3)],
            valset=[{"validation": 1}],
            adapter=BranchAdapter(),
            batch_sampler=sampler,
            candidate_selection_strategy=selector,
            reflection_lm=lm,
            acceptance_criterion=Mock(should_accept=Mock(return_value=True)),
            run_dir=str(tmp_path),
            stop_callbacks=lambda st: len(st.full_program_trace) >= stop,
            display_progress_bar=False,
            raise_on_exception=True,
        )

    result = run(s, 3)
    assert len(result.candidates) == 4
    old = snapshot(s)
    restored, _, _ = setup(size=3)
    saved = GEPAState.load(str(tmp_path))
    restored.load_from_trace(
        saved.full_program_trace, ListDataLoader([{"task": i} for i in range(3)]), require_history=True
    )
    assert snapshot(restored) == old
    result = run(restored, 6)
    assert len(result.candidates) == 7
    saved = GEPAState.load(str(tmp_path))
    for entry in saved.full_program_trace:
        assert entry["task_curator_parent_idx"] == entry["selected_program_candidate"]
        assert entry["task_curator_child_idx"] == entry["new_program_idx"]
    again, _, _ = setup(size=3)
    again.load_from_trace(
        saved.full_program_trace, ListDataLoader([{"task": i} for i in range(3)]), require_history=True
    )
    assert snapshot(again) == snapshot(restored)
