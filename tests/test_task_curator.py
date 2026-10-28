"""Efficient-curator behavior with deterministic responses, without API calls."""

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
from gepa.strategies.proposal_sampling import SameParentSampling
from gepa.strategies.task_curator import TaskCuratorContractError, TaskCuratorLMError, TaskCuratorSampler


def make_sampler(response='{"decision":"run"}', **kwargs):
    return TaskCuratorSampler(minibatch_size=kwargs.pop("minibatch_size", 1), lm=Mock(return_value=response), **kwargs)


def reflect_dataset(sampler, *, state, task_ids, evaluation, reflective_dataset):
    # Fixed one-task fixture for tests of anchor content and success selection.
    assert len(task_ids) == 1
    return sampler.build_reflective_dataset(
        candidate=dict.fromkeys(reflective_dataset, "prompt"),
        task_ids=task_ids,
        evaluation=evaluation,
        components_to_update=list(reflective_dataset),
        make_reflective_dataset=lambda *_: reflective_dataset,
        state=state,
    )


def state_for(i=0):
    return SimpleNamespace(i=i, full_program_trace=[{"i": i}])


def advance(state):
    state.i += 1
    state.full_program_trace.append({"i": state.i})


def evaluate(sampler, loader, state, scores=None, trajectories=None):
    ids = sampler.next_minibatch_ids_for_candidate(loader, state, candidate_idx=getattr(state, "current_candidate", 0))
    scores = scores if scores is not None else [0] * len(ids)
    trajectories = trajectories if trajectories is not None else [{"evidence": f"task-{t}"} for t in ids]
    batch = EvaluationBatch(outputs=[None] * len(ids), scores=scores, trajectories=trajectories)
    sampler.observe_evaluation(iteration=state.i + 1, task_ids=ids, evaluation=batch, state=state)
    return ids, batch


def paired(sampler, state, ids, before, after_scores, accepted=False):
    after = SubsampleEvaluation(
        scores=after_scores, outputs=[None] * len(ids), trajectories=[{"evidence": "child-success"}] * len(ids)
    )
    proposal = CandidateProposal(
        candidate={"prompt": "new"},
        parent_program_ids=[sampler._active_candidate_idx],
        subsample_indices=ids,
        eval_before=SubsampleEvaluation(scores=before.scores, outputs=before.outputs, trajectories=before.trajectories),
        eval_after=after,
    )
    child = max(sampler._branches) + 1 if accepted else None
    sampler.observe_proposal(
        iteration=state.i + 1, proposal=proposal, accepted=accepted, state=state, candidate_idx=child
    )
    if child is not None:
        state.current_candidate = child


@pytest.mark.parametrize(
    "key,value",
    [
        ("minibatch_size", 0),
        ("minibatch_size", True),
        ("max_workers", -1),
        ("history_window", 0),
        ("history_window", False),
        ("seed", 0.5),
    ],
)
def test_invalid_settings(key, value):
    with pytest.raises(ValueError):
        make_sampler(**{key: value})


@pytest.mark.parametrize("window", [1, 4, None])
def test_paired_history_and_older_assessments(window):
    s = make_sampler(history_window=window)
    loader, state = ListDataLoader([{"task": "t"}]), state_for()
    for i in range(6):
        ids, batch = evaluate(s, loader, state)
        paired(s, state, ids, batch, [1], accepted=False)
        if i < 5:
            advance(state)
    history = deepcopy(s.history)
    units = s._prompt_view(0)
    assert len(units) == (6 if window is None else window)
    assert all(u["event"] == "mutation_comparison" for u in units)
    assert all(u["mutation_accepted"] is False for u in units)
    assert "iteration" not in str(units)
    assert "curator_decision" not in str(units)
    assert len(s._older_history_for_prompt(0)) == (0 if window is None else 6 - window)
    assert s.history == history


def test_status_summary_attaches_to_previous_execution():
    s = make_sampler(history_window=1)
    loader, state = ListDataLoader([{}]), state_for()
    evaluate(s, loader, state)
    advance(state)
    s.lm.return_value = '{"decision":"run","task_status":"prompt_sensitive"}'
    evaluate(s, loader, state)
    assert s._older_history_for_prompt(0) == [
        {"epoch_id": 0, "outcome": "fail", "task_status": "prompt_sensitive", "status_epoch_id": 1, "candidate_idx": 0}
    ]


@pytest.mark.parametrize(
    "choice,expected",
    [("before", "before"), ("after", "after"), ("retained", None), ("unknown", None), (None, None), ([], None)],
)
def test_success_pool_admission_and_pair_eligibility(choice, expected):
    s = make_sampler()
    loader, state = ListDataLoader([{}]), state_for()
    ids, before = evaluate(s, loader, state, [1], [{"evidence": "parent-success"}])
    paired(s, state, ids, before, [1], accepted=False)
    assert not s._branch.retained_success  # no automatic admission
    s.lm.return_value = json.dumps({"decision": "run", "selected_success": choice})
    advance(state)
    evaluate(s, loader, state)
    selected = s._branch.retained_success.get(0)
    assert (selected.rsplit("/", 1)[-1] if selected else None) == expected
    assert "independent clean success" not in s.lm.call_args.args[0]


@pytest.mark.parametrize("before,after,choice", [(0, 1, "before"), (1, 0, "after"), (0, 0, "after")])
def test_failed_side_cannot_enter_pool(before, after, choice):
    s = make_sampler()
    loader, state = ListDataLoader([{}]), state_for()
    ids, first = evaluate(s, loader, state, [before])
    paired(s, state, ids, first, [after])
    s.lm.return_value = json.dumps({"decision": "run", "selected_success": choice})
    advance(state)
    evaluate(s, loader, state)
    assert not s._branch.retained_success


def with_anchor(**kwargs):
    s = make_sampler(**kwargs)
    loader, state = ListDataLoader([{}]), state_for()
    evaluate(s, loader, state, [1], [{"evidence": "trusted-success"}])
    advance(state)
    s.lm.return_value = '{"decision":"run","selected_success":"before"}'
    evaluate(s, loader, state)
    return s, loader, state


def test_anchor_injection_is_for_same_failed_task_and_non_mutating():
    s, _, state = with_anchor()
    dataset = {"prompt": [{"Feedback": "failed"}]}
    original = deepcopy(dataset)
    evaluation = EvaluationBatch(outputs=[None], scores=[0], trajectories=[{}])
    enriched = reflect_dataset(s, state=state, task_ids=[0], evaluation=evaluation, reflective_dataset=dataset)
    assert enriched["prompt"][0]["HistoricalSuccess"]["successful_state"] == {"evidence": "trusted-success"}
    assert dataset == original
    evaluation.scores = [1]
    assert reflect_dataset(s, state=state, task_ids=[0], evaluation=evaluation, reflective_dataset=dataset) == original
    evaluation.scores = [0]
    assert reflect_dataset(s, state=state, task_ids=[0], evaluation=evaluation, reflective_dataset={"prompt": []}) == {}
    assert state.full_program_trace[-1]["task_curator_reflection_anchors"] == []


@pytest.mark.parametrize(
    "response",
    [
        "invalid json",
        '{"decision":"bogus"}',
        '{"decision":"skip","skip_epochs":-1}',
        '{"decision":"skip","skip_epochs":2}',
        '{"decision":"run","task_status":"bogus"}',
    ],
)
def test_invalid_decision_runs_and_preserves_anchor(response):
    s, loader, state = with_anchor()
    prior = deepcopy(s._branch.retained_success)
    s.lm.return_value = response
    advance(state)
    evaluate(s, loader, state)
    assert s._branch.retained_success == prior
    assert state.full_program_trace[-1]["task_curator_decisions"][0]["curator_failure"]


def test_model_failure_is_audited_or_raised():
    for strict in (False, True):
        s = make_sampler(raise_on_error=strict)
        s.lm.side_effect = RuntimeError("service unavailable")
        loader, state = ListDataLoader([{}]), state_for()
        if strict:
            with pytest.raises(TaskCuratorLMError, match="service unavailable"):
                evaluate(s, loader, state)
        else:
            evaluate(s, loader, state)
            assert "service unavailable" in str(state.full_program_trace[-1]["task_curator_decisions"])


@pytest.mark.parametrize("paired_result", [False, True])
def test_eligible_evidence_is_not_repeated(paired_result):
    s = make_sampler()
    loader, state = ListDataLoader([{}]), state_for()
    ids, batch = evaluate(s, loader, state, [1], [{"evidence": "UNIQUE_SUCCESS"}])
    if paired_result:
        paired(s, state, ids, batch, [0])
    advance(state)
    evaluate(s, loader, state)
    prompt = s.lm.call_args.args[0]
    assert prompt.count("UNIQUE_SUCCESS") == 1
    assert "Eligible successful trajectories" not in prompt
    assert ("History[-1].before.evaluator_feedback" if paired_result else "History[-1].evaluator_feedback") in prompt


def test_cooldown_bypasses_model_and_resume_preserves_schedule():
    s, loader, state = with_anchor()
    s.lm.return_value = '{"decision":"skip","skip_epochs":3,"skip_type":"mastered"}'
    advance(state)
    ids, _ = evaluate(s, loader, state)
    assert ids == [0] and s._branch.epoch == 5
    assert s.lm.call_count == 3
    restored = make_sampler()
    restored.load_from_trace(deepcopy(state.full_program_trace), loader, require_history=True)
    assert restored._branch.retained_success == s._branch.retained_success
    assert restored._branch.epoch == s._branch.epoch
    assert restored.history == s.history
    s.lm.return_value = '{"decision":"run"}'
    advance(state)
    state2 = deepcopy(state)
    assert evaluate(s, loader, state)[0] == evaluate(restored, loader, state2)[0]
    assert restored._branch.epoch == s._branch.epoch


@pytest.mark.parametrize("size,expected", [(1, [1, 1, 1]), (3, [2, 1, 2]), (5, [2, 2, 1])])
def test_epoch_tail_and_auto_resume(size, expected):
    s = make_sampler(minibatch_size=2)
    loader, state = ListDataLoader([{} for _ in range(size)]), state_for()
    for j, count in enumerate(expected):
        ids, _ = evaluate(s, loader, state)
        assert len(ids) == count and len(ids) == len(set(ids))
        if j < len(expected) - 1:
            s = make_sampler(minibatch_size=2)
            advance(state)


def test_resume_rejects_missing_evidence_changed_configuration_or_universe():
    s, loader, state = with_anchor()
    trace = deepcopy(state.full_program_trace)
    del trace[-1]["task_curator_results"]
    with pytest.raises(TaskCuratorContractError, match="parent evaluation"):
        make_sampler().load_from_trace(trace, loader, require_history=True)
    with pytest.raises(TaskCuratorContractError, match="settings changed"):
        make_sampler(history_window=4).load_from_trace(state.full_program_trace, loader, require_history=True)
    with pytest.raises(TaskCuratorContractError, match="without curator"):
        make_sampler().load_from_trace([{"i": 0}], loader, require_history=True)
    loader.add_items([{}])
    with pytest.raises(TaskCuratorContractError, match="universe changed"):
        s.next_minibatch_ids_for_candidate(loader, state, candidate_idx=getattr(state, "current_candidate", 0))


def test_only_latest_execution_can_replace_retained_anchor():
    s, loader, state = with_anchor()
    old = dict(s._branch.retained_success)
    # The latest unit failed; a previous success must not become a candidate again.
    advance(state)
    evaluate(s, loader, state)
    assert "selected_success" not in s.lm.call_args.args[0]
    assert s._branch.retained_success == old
    advance(state)
    evaluate(s, loader, state, [1], [{"evidence": "better-success"}])
    advance(state)
    evaluate(s, loader, state)
    assert s._branch.retained_success != old


def test_custom_evidence_and_score_semantics():
    s = make_sampler(
        task_view=lambda task: {"description": task["label"]},
        evidence_extractor=lambda t: t["safe"] if t else None,
        is_success=lambda score, t: score >= 80 and t.get("complete", False),
        is_failure=lambda score, t: score < 80,
    )
    loader, state = ListDataLoader([{"label": "task", "private": "do-not-include"}]), state_for()
    evaluate(s, loader, state, [90], [{"safe": "success-evidence", "complete": True}])
    advance(state)
    s.lm.return_value = '{"decision":"run","selected_success":"before"}'
    evaluate(s, loader, state, [50], [{"safe": "failure-evidence"}])
    prompt = s.lm.call_args.args[0]
    assert "success-evidence" in prompt and "do-not-include" not in prompt
    enriched = reflect_dataset(
        s,
        state=state,
        task_ids=[0],
        evaluation=EvaluationBatch(outputs=[None], scores=[50], trajectories=[{}]),
        reflective_dataset={"p": [{}]},
    )
    assert enriched["p"][0]["HistoricalSuccess"]["successful_state"] == "success-evidence"


def test_none_trajectory_does_not_become_anchor():
    s = make_sampler()
    loader, state = ListDataLoader([{}]), state_for()
    evaluate(s, loader, state, [1], [None])
    assert not s._branch.success_states[0]


class TinyAdapter:
    propose_new_texts = None

    def __init__(self, missing=False):
        self.parents = 0
        self.missing = missing
        self.metric_calls = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        self.metric_calls += len(batch)
        if capture_traces and candidate["p"] == "seed":
            self.parents += 1
        # Parent succeeds once, then fails; rejected child still offers a success.
        score = 1 if candidate["p"] == "child" or self.parents == 1 else 0
        return EvaluationBatch(
            outputs=[None] * len(batch),
            scores=[score] * len(batch),
            trajectories=None
            if self.missing or not capture_traces
            else [{"task": task["label"], "candidate": candidate["p"]} for task in batch],
        )

    def make_reflective_dataset(self, candidate, eval_batch, components):
        return {component: [{"Feedback": t} for t in eval_batch.trajectories] for component in components}


class RejectAll:
    def should_accept(self, proposal, state):
        return False


def run_core(s, path, *, stop=4, missing=False, custom=None, sampling=None):
    adapter = TinyAdapter(missing=missing)
    prompts = []

    def lm(prompt):
        prompts.append(prompt)
        return "```\nchild\n```"

    result = gepa.optimize(
        seed_candidate={"p": "seed"},
        trainset=[{"label": "task"}],
        valset=[{"label": "val"}],
        adapter=adapter,
        batch_sampler=s,
        reflection_lm=lm,
        run_dir=str(path),
        stop_callbacks=lambda st: len(st.full_program_trace) >= stop,
        acceptance_criterion=RejectAll(),
        display_progress_bar=False,
        reflective_dataset_enricher=custom,
        sampling_strategy=sampling,
    )
    return result, prompts, adapter


@pytest.mark.parametrize("missing", [False, True])
def test_core_integration_and_checkpoint_resume(tmp_path, missing):
    s = make_sampler('{"decision":"run","selected_success":"before"}')
    result, prompts, adapter = run_core(s, tmp_path, missing=missing)
    saved = GEPAState.load(str(tmp_path))
    entries = [r for r in saved.full_program_trace if r.get("task_curator_enabled")]
    assert len(entries) == 4
    assert all(len(e["task_curator_results"]) == 1 for e in entries)
    assert saved.total_num_evals == adapter.metric_calls
    if not missing:
        assert all(e["task_curator_mutation_results"][0]["mutation_accepted"] is False for e in entries[1:])
        assert "task_curator_mutation_results" not in entries[0]  # perfect parent skipped
        assert any("HistoricalSuccess" in p for p in prompts)
        assert len(result.candidates) == 1
    else:
        assert not prompts
        assert all("task_curator_mutation_results" not in e for e in entries)
    resumed = make_sampler('{"decision":"run","selected_success":"retained"}')
    run_core(resumed, tmp_path, stop=5, missing=missing)
    assert resumed._branch.epoch == 4
    assert len(resumed._recorded_iterations) == 5


def test_curator_composes_with_existing_enricher(tmp_path):
    seen = []

    def enrich(**kwargs):
        seen.append(deepcopy(kwargs["reflective_dataset"]))
        return kwargs["reflective_dataset"]

    run_core(make_sampler('{"decision":"run","selected_success":"before"}'), tmp_path, custom=enrich)
    assert any("HistoricalSuccess" in d["p"][0] for d in seen)


def test_multi_proposal_rejected_before_evaluation(tmp_path):
    with pytest.raises(ValueError, match="SingleMutationSampling"):
        run_core(make_sampler(), tmp_path, sampling=SameParentSampling(2))


class AcceptAll:
    def should_accept(self, proposal, state):
        return True


def test_accepted_proposal_is_recorded(tmp_path):
    s = make_sampler()
    gepa.optimize(
        seed_candidate={"p": "seed"},
        trainset=[{"label": "task"}],
        valset=[{"label": "val"}],
        adapter=TinyAdapter(),
        batch_sampler=s,
        reflection_lm=lambda _: "```\nchild\n```",
        skip_perfect_score=False,
        acceptance_criterion=AcceptAll(),
        run_dir=str(tmp_path),
        stop_callbacks=lambda state: len(state.full_program_trace) >= 1,
        display_progress_bar=False,
    )
    row = GEPAState.load(str(tmp_path)).full_program_trace[0]["task_curator_mutation_results"][0]
    assert row["mutation_accepted"] is True


def test_contract_error_in_reflection_assembly_is_not_swallowed(tmp_path):
    class BrokenAdapter(TinyAdapter):
        def make_reflective_dataset(self, candidate, eval_batch, components):
            raise RuntimeError("single-task formatting failed")

    with pytest.raises(TaskCuratorContractError, match="single-task formatting failed"):
        gepa.optimize(
            seed_candidate={"p": "seed"},
            trainset=[{"label": "task"}],
            valset=[{"label": "val"}],
            adapter=BrokenAdapter(),
            batch_sampler=make_sampler('{"decision":"run","selected_success":"before"}'),
            reflection_lm=lambda _: "```\nchild\n```",
            max_metric_calls=20,
            raise_on_exception=False,
            run_dir=str(tmp_path),
            display_progress_bar=False,
        )


def test_missing_or_misaligned_evidence_cannot_advance():
    s, loader, state = make_sampler(), ListDataLoader([{}]), state_for()
    s.next_minibatch_ids_for_candidate(loader, state, candidate_idx=getattr(state, "current_candidate", 0))
    with pytest.raises(TaskCuratorContractError, match="association mismatch"):
        s.record_first_pass(iteration=1, task_ids=[10], scores=[0], outputs=[None], trajectories=[{}], state=state)
    advance(state)
    with pytest.raises(TaskCuratorContractError, match="without first-pass results"):
        s.next_minibatch_ids_for_candidate(loader, state, candidate_idx=getattr(state, "current_candidate", 0))


@pytest.mark.parametrize("entrypoint", ["launcher", "optimize_anything"])
def test_optimizer_entrypoints_accept_curator_instance(tmp_path, entrypoint):
    from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig
    from gepa.gepa_launcher import optimize_anything as legacy_optimize
    from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything

    s = make_sampler()
    reflection = ReflectionConfig(batch_sampler=s, reflection_lm=lambda _: "```\nchild\n```")
    engine = EngineConfig(max_metric_calls=10, display_progress_bar=False)
    if entrypoint == "launcher":
        config = GEPAConfig(engine=engine, reflection=reflection)
        fn = legacy_optimize
    else:
        config = OptimizeAnythingConfig(
            engine="gepa", output_dir=str(tmp_path), engine_config={"engine": engine, "reflection": reflection}
        )
        fn = optimize_anything
    fn(
        seed_candidate="seed",
        evaluator=lambda candidate, example=None: (0, {"Feedback": "failure"}),
        dataset=[{"task": "t"}],
        valset=[{"task": "v"}],
        config=config,
    )
    assert s._recorded_iterations
    assert any(row["event"] == "mutation_comparison" for row in s.history[0])


@pytest.mark.parametrize("packet", [{"score": 1, "trajectory_summary": ["completed action"]}, {}, "evidence"])
def test_dedicated_curator_packet_is_used_for_history_and_success(packet):
    s = make_sampler('{"decision":"run","selected_success":"before"}')
    loader = ListDataLoader([{"_task_curator_view": {"label": "public task"}, "private": "PRIVATE_TASK_SENTINEL"}])
    state = state_for()
    trajectory = {
        "task_curator_specific_info": packet,
        "rollout_artifact_specific_info": {"trajectory": "RAW_ROLLOUT_SENTINEL"},
        "worker_pid": "WORKER_SENTINEL",
        # An explicit packet takes precedence over the legacy field projection.
        "Task": "LEGACY_TASK_SENTINEL",
    }
    original = deepcopy(trajectory)
    ids, before = evaluate(s, loader, state, [1], [trajectory])
    s.record_mutation_comparison(
        iteration=1,
        task_ids=ids,
        before_scores=before.scores,
        before_trajectories=before.trajectories,
        after_scores=[1],
        after_trajectories=[trajectory],
        mutation_accepted=False,
        state=state,
    )
    entry = state.full_program_trace[-1]
    assert entry["task_curator_results"][0]["evaluator_feedback"] == packet
    paired_row = entry["task_curator_mutation_results"][0]
    assert paired_row["before"]["evaluator_feedback"] == packet
    assert paired_row["after"]["evaluator_feedback"] == packet
    assert all(snapshot["success_state"] == packet for snapshot in s._branch.success_states[0].values())
    advance(state)
    _, failed = evaluate(s, loader, state, [0], [{"task_curator_specific_info": {"score": 0}}])
    for call in s.lm.call_args_list:
        prompt = call.args[0]
        assert "public task" in prompt
        assert not any(
            marker in prompt
            for marker in ("PRIVATE_TASK_SENTINEL", "RAW_ROLLOUT_SENTINEL", "WORKER_SENTINEL", "LEGACY_TASK_SENTINEL")
        )
    reflected = reflect_dataset(s, state=state, task_ids=[0], evaluation=failed, reflective_dataset={"p": [{}]})
    assert reflected["p"][0]["HistoricalSuccess"]["successful_state"] == packet
    assert trajectory == original


@pytest.mark.parametrize("judge_feedback", ["judge explanation", None])
def test_legacy_evaluation_history_and_success_use_distinct_views(judge_feedback):
    trajectory = {
        "Task": "update item",
        "Score": 1,
        "JudgeFeedback": judge_feedback,
        "Error": None,
        "EvalFailures": None,
        "ReferenceEvaluation": {"expected": "updated"},
        "Trajectory": "RAW_TRAJECTORY_SENTINEL",
        "evaluation_evidence_specific_info": {
            "evaluation_status": "success",
            "evaluation_types": ["reference", "judge"],
            "evaluation_feedback": "reference explanation",
            "error": None,
            "trajectory_complete": True,
            "execution_stage": "done",
            "debug": "PRIVATE_PACKET_SENTINEL",
        },
    }
    s, loader, state = make_sampler(), ListDataLoader([{}]), state_for()
    evaluate(s, loader, state, [1], [trajectory])
    expected_history = {
        "Task": "update item",
        "Score": 1,
        "JudgeFeedback": judge_feedback,
        "Error": None,
        "EvalFailures": None,
        "ReferenceEvaluation": {"expected": "updated"},
        "Evaluation": {
            "evaluation_status": "success",
            "evaluation_types": ["reference", "judge"],
            "evaluation_feedback": "reference explanation",
            "error": None,
            "trajectory_complete": True,
            "execution_stage": "done",
        },
    }
    assert s.history[0][-1]["evaluator_feedback"] == expected_history
    expected_success = {
        "Task": "update item",
        "Score": 1,
        "EvaluatorFeedback": judge_feedback or "reference explanation",
        "ReferenceEvaluation": {"expected": "updated"},
    }
    snapshot = next(iter(s._branch.success_states[0].values()))
    assert snapshot["success_state"] == expected_success
    advance(state)
    s.lm.return_value = '{"decision":"run","selected_success":"before"}'
    _, failed = evaluate(s, loader, state)
    prompt = s.lm.call_args.args[0]
    assert "RAW_TRAJECTORY_SENTINEL" not in prompt and "PRIVATE_PACKET_SENTINEL" not in prompt
    reflected = reflect_dataset(s, state=state, task_ids=[0], evaluation=failed, reflective_dataset={"p": [{}]})
    assert reflected["p"][0]["HistoricalSuccess"]["successful_state"] == expected_success


@pytest.mark.parametrize(
    "trajectory",
    [
        {"evidence": "generic adapter evidence"},
        "text evidence",
        ["step one", "step two"],
        {"task_curator_specific_info": None, "evidence": "fallback"},
    ],
)
def test_unrecognized_adapter_evidence_is_preserved(trajectory):
    s, loader, state = make_sampler(), ListDataLoader([{}]), state_for()
    evaluate(s, loader, state, [1], [trajectory])
    assert s.history[0][-1]["evaluator_feedback"] == trajectory
    assert next(iter(s._branch.success_states[0].values()))["success_state"] == trajectory


def test_success_extractor_can_override_history_extractor():
    s = make_sampler(
        evidence_extractor=lambda trajectory: {"history": trajectory["history"]},
        success_extractor=lambda trajectory: {"reference": trajectory["reference"]},
    )
    loader, state = ListDataLoader([{}]), state_for()
    evaluate(s, loader, state, [1], [{"history": "execution details", "reference": "success details"}])
    assert s.history[0][-1]["evaluator_feedback"] == {"history": "execution details"}
    assert next(iter(s._branch.success_states[0].values()))["success_state"] == {"reference": "success details"}


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6])
def test_resume_rejects_checkpoint_with_previous_defaults(version):
    s, loader, state = make_sampler(), ListDataLoader([{}]), state_for()
    evaluate(s, loader, state)
    state.full_program_trace[-1]["task_curator_settings"]["version"] = version
    with pytest.raises(TaskCuratorContractError, match="settings changed"):
        make_sampler().load_from_trace(state.full_program_trace, loader, require_history=True)


@pytest.mark.parametrize(
    "score,trajectory,eligible",
    [
        (0, {}, False),
        (0.9, {}, False),
        (1, {}, True),
        (2, {}, False),
        (1, "completed", True),
        (1, {"Error": "invalid execution"}, False),
        (1, {"error": "judge failed"}, False),
        (1, {"EvalFailures": ["reference error"]}, False),
        (1, {"evaluation_evidence_specific_info": {"evaluation_status": " SUCCESS "}}, True),
        (1, {"evaluation_evidence_specific_info": {"evaluation_status": "fail"}}, False),
        (1, {"evaluation_evidence_specific_info": {"evaluation_status": "timeout"}}, False),
        (1, {"evaluation_evidence_specific_info": {"trajectory_complete": False}}, False),
        (1, {"evaluation_evidence_specific_info": {"error": "judge unavailable"}}, False),
        (1, {"evaluation_evidence_specific_info": {"trajectory_complete": True}}, True),
        # Match the experiment: the eligibility check reads the raw evaluation
        # packet, not similarly named fields inside the curator's evidence view.
        (1, {"task_curator_specific_info": {"evaluation_status": "fail"}}, True),
    ],
)
def test_default_clean_success_eligibility(score, trajectory, eligible):
    s, loader, state = make_sampler(), ListDataLoader([{}]), state_for()
    evaluate(s, loader, state, [score], [trajectory])
    assert bool(s._branch.success_states[0]) is eligible
    assert bool(s._latest_pool_candidates(0)) is eligible


@pytest.mark.parametrize(
    "score,feedback,outcome",
    [
        (1, {}, "success"),
        (2, {}, "score=2"),
        (0, {}, "fail"),
        (0.5, {}, "score=0.5"),
        (float("nan"), {}, "unknown"),
        (None, {}, "unknown"),
        (1, {"infrastructure_error": True}, "infrastructure_error"),
        (0, {"evaluation_status": "TIMEOUT"}, "infrastructure_error"),
        (0, {"Error": "network timeout"}, "infrastructure_error"),
        (0, {"error": "connection reset"}, "infrastructure_error"),
        (0, {"actor_error": "invalid tool"}, "fail"),
        (0, {"actor_error": "invalid tool", "evaluation_status": "error"}, "fail"),
        (0, {"error": "judge failed"}, "error"),
        (0, {"EvalFailures": ["bad reference"]}, "error"),
        (0, {"evaluation_status": "invalid"}, "error"),
        (0, {"Evaluation": {"evaluation_status": "error"}}, "error"),
        (0, {"evaluation_evidence_specific_info": {"trajectory_complete": False}}, "incomplete"),
        (1, {"Evaluation": {"trajectory_complete": False}}, "incomplete"),
        (0, {"Evaluation": {"error": "bad evaluator", "trajectory_complete": False}}, "error"),
        (0, {"evaluation_status": "success"}, "fail"),
        # The historical summarizer checks completeness on an evaluation packet.
        (0, {"trajectory_complete": False}, "fail"),
    ],
)
def test_older_history_outcomes(score, feedback, outcome):
    # Preserve experimental error classifications; the default success label
    # uses the same exact-one boundary as success snapshots.
    assert TaskCuratorSampler._compact_outcome({"score": score, "evaluator_feedback": feedback}) == outcome


def test_retained_success_alias_is_visible_only_while_in_detailed_history():
    s = make_sampler(history_window=1)
    loader, state = ListDataLoader([{}]), state_for()
    ids, batch = evaluate(s, loader, state, [1])
    paired(s, state, ids, batch, [1], accepted=False)
    assert "success_state_id" not in str(s._prompt_view(0))
    s._branch.retained_success[0] = list(s._branch.success_states[0])[1]
    row = s._prompt_view(0)[0]
    assert row["after"]["success_state_id"] == "success-2"
    assert "success_state_id" not in row["before"]
    assert "task:" not in str(row) and "iteration" not in str(row)
    advance(state)
    evaluate(s, loader, state)
    assert "success_state_id" not in str(s._prompt_view(0))
    assert s._branch.retained_success[0].endswith("/after")


def test_reflection_audits_ids_without_exposing_them_to_reflector():
    s, _, state = with_anchor()
    evaluation = EvaluationBatch(outputs=[None], scores=[0], trajectories=[{}])
    enriched = reflect_dataset(s, state=state, task_ids=[0], evaluation=evaluation, reflective_dataset={"p": [{}]})
    assert set(enriched["p"][0]["HistoricalSuccess"]) == {"instruction", "successful_state"}
    assert state.full_program_trace[-1]["task_curator_reflection_anchors"] == [
        {"task_id": 0, "success_state_id": s._branch.retained_success[0], "supporting_success_ids": []}
    ]


@pytest.mark.parametrize("strict", [False, True])
def test_fatal_model_failure_stops_even_after_optimization_progress(tmp_path, strict):
    s = make_sampler(raise_on_error=strict)
    failure = RuntimeError("quota exhausted") if strict else TaskCuratorLMError("quota exhausted")
    s.lm.side_effect = ['{"decision":"run"}', failure]
    with pytest.raises(TaskCuratorLMError, match="quota exhausted"):
        run_core(s, tmp_path)
    assert s._recorded_iterations == {1}


@pytest.mark.parametrize("paired_result", [False, True])
def test_above_one_score_is_neither_archived_nor_offered_by_default(paired_result):
    s = make_sampler('{"decision":"run","selected_success":"before"}')
    loader, state = ListDataLoader([{}]), state_for()
    ids, batch = evaluate(s, loader, state, [2], [{"evidence": "score above one"}])
    if paired_result:
        paired(s, state, ids, batch, [2], accepted=False)
    assert not s._branch.success_states[0]
    assert not s._latest_pool_candidates(0)
    advance(state)
    evaluate(s, loader, state)
    assert not s._branch.retained_success
    assert "Eligible successes" not in s.lm.call_args.args[0]


def test_resume_rejects_switching_default_success_policy():
    s, loader, state = with_anchor()
    custom = make_sampler(is_success=lambda score, trajectory: score >= 1)
    with pytest.raises(TaskCuratorContractError, match="settings changed"):
        custom.load_from_trace(state.full_program_trace, loader, require_history=True)


@pytest.mark.parametrize("custom_extractor", [False, True])
@pytest.mark.parametrize("paired_result", [False, True])
def test_history_metadata_cleaning_preserves_evaluator_payload(custom_extractor, paired_result):
    evidence = {
        "reason": "Missing required confirmation",
        "task_id": "external-task",
        "iteration": 7,
        "iteration_id": "external-step",
        "epoch": 9,
        "anchor_reason": "Evaluator explanation",
        "success_state_id": ["an evaluator field, not a curator ID"],
        "steps": [{"reason": "User authorization not checked", "epoch": 10}],
        "before": {"reason": "Original evaluator state"},
    }
    extractor = (lambda trajectory: trajectory["custom_payload"]) if custom_extractor else None
    trajectory = {"custom_payload": evidence} if custom_extractor else evidence
    s = make_sampler(evidence_extractor=extractor)
    loader, state = ListDataLoader([{}]), state_for()
    ids, batch = evaluate(s, loader, state, [0], [trajectory])
    if paired_result:
        s.record_mutation_comparison(
            iteration=1,
            task_ids=ids,
            before_scores=batch.scores,
            before_trajectories=batch.trajectories,
            after_scores=[0],
            after_trajectories=[trajectory],
            mutation_accepted=False,
            state=state,
        )
    stored = deepcopy(s.history)
    view = s._prompt_view(0)
    assert "iteration" not in view[0] and view[0]["epoch_id"] == 0
    records = [view[0]["before"], view[0]["after"]] if paired_result else [view[0]]
    assert all(record["evaluator_feedback"] == evidence for record in records)
    records[0]["evaluator_feedback"]["steps"][0]["reason"] = "modified view"
    assert s.history == stored
    advance(state)
    evaluate(s, loader, state, [0], [trajectory])
    prompt = s.lm.call_args.args[0]
    assert "Missing required confirmation" in prompt
    assert "User authorization not checked" in prompt


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("choice", [None, "unknown", "after", [], {}])
def test_invalid_anchor_selection_uses_response_error_policy(strict, choice):
    s, loader, state = with_anchor(raise_on_error=strict)
    advance(state)
    evaluate(s, loader, state, [1], [{"evidence": "new successful execution"}])
    prior = deepcopy(s._branch.retained_success)
    advance(state)
    payload = {"decision": "skip", "skip_epochs": 3, "skip_type": "mastered"}
    if choice is not None:
        payload["selected_success"] = choice
    s.lm.return_value = json.dumps(payload)
    if strict:
        with pytest.raises(TaskCuratorLMError, match="selected_success"):
            s.next_minibatch_ids_for_candidate(loader, state, candidate_idx=getattr(state, "current_candidate", 0))
    else:
        evaluate(s, loader, state)
        decision = state.full_program_trace[-1]["task_curator_decisions"][0]
        assert decision["decision"] == "run"
        assert "selected_success" in decision["curator_failure"]["error"]
        assert not s._branch.next_run_epoch
    assert s._branch.retained_success == prior


def test_anchor_selection_uses_same_fenced_response_as_decision():
    s, loader, state = make_sampler(raise_on_error=True), ListDataLoader([{}]), state_for()
    evaluate(s, loader, state, [1], [{"evidence": "success"}])
    advance(state)
    s.lm.return_value = '```json\n{"decision":"run","selected_success":"before"}\n```\nIgnored {commentary}'
    evaluate(s, loader, state)
    assert s._branch.retained_success[0].endswith("/iteration:1/before")


def test_validation_failure_retains_child_evidence_and_resumes(tmp_path):
    class FailOnceValidationAdapter(TinyAdapter):
        def __init__(self):
            super().__init__()
            self.validation_attempts = 0
            self.child_minibatches = 0

        def evaluate(self, batch, candidate, capture_traces=False):
            if candidate["p"] == "child":
                if batch[0]["label"] == "val":
                    self.validation_attempts += 1
                    if self.validation_attempts == 1:
                        raise RuntimeError("transient validation failure")
                else:
                    self.child_minibatches += 1
            return super().evaluate(batch, candidate, capture_traces=capture_traces)

    adapter = FailOnceValidationAdapter()
    s = make_sampler('{"decision":"run","selected_success":"after"}')

    def run(sampler, stop):
        return gepa.optimize(
            seed_candidate={"p": "seed"},
            trainset=[{"label": "task"}],
            valset=[{"label": "val"}],
            adapter=adapter,
            batch_sampler=sampler,
            reflection_lm=lambda _: "```\nchild\n```",
            skip_perfect_score=False,
            acceptance_criterion=AcceptAll(),
            run_dir=str(tmp_path),
            raise_on_exception=False,
            stop_callbacks=lambda state: len(state.full_program_trace) >= stop,
            display_progress_bar=False,
        )

    result = run(s, 1)
    assert adapter.child_minibatches == adapter.validation_attempts == 1
    assert result.candidates == [{"p": "seed"}]
    saved = GEPAState.load(str(tmp_path))
    rows = saved.full_program_trace[0]["task_curator_mutation_results"]
    assert len(rows) == 1
    row = rows[0]
    assert row["mutation_accepted"] is False
    assert row["after"]["score"] == 1
    assert row["after"]["evaluator_feedback"]["candidate"] == "child"
    after_id = row["after"]["success_state_id"]
    restored = make_sampler('{"decision":"run","selected_success":"after"}')
    restored.load_from_trace(saved.full_program_trace, ListDataLoader([{"label": "task"}]), require_history=True)
    assert restored.history == s.history
    assert restored._branch.success_states == s._branch.success_states

    resumed = make_sampler('{"decision":"run","selected_success":"after"}')
    result = run(resumed, 2)
    assert resumed._branch.retained_success[0] == after_id
    assert {"p": "child"} in result.candidates
    entries = GEPAState.load(str(tmp_path)).full_program_trace
    assert len(entries) == 2
    assert [entry["task_curator_mutation_results"][0]["mutation_accepted"] for entry in entries] == [False, True]
    assert all(len(entry["task_curator_mutation_results"]) == 1 for entry in entries)
