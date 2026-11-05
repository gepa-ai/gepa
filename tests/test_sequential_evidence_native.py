import json

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.result import GEPAResult
from gepa.evidence.config import SequentialEvidenceConfig
from gepa.strategies.eval_policy import FullEvaluationPolicy


class Quiet:
    def log(self, *args):
        pass


class ScriptedAdapter:
    def __init__(self, fail_replication=False, cost=1):
        self.executions = 0
        self.replications = 0
        self.fail_replication = fail_replication
        self.cost = cost

    def evaluate(self, batch, candidate, capture_traces=False):
        self.executions += len(batch) * self.cost
        score = float(candidate["program"] == "lambda x: 1")
        return EvaluationBatch(
            outputs=[eval(candidate["program"])(x) for x in batch],
            scores=[score] * len(batch),
            trajectories=[{"real": score}] * len(batch) if capture_traces else None,
            num_metric_calls=len(batch) * self.cost,
        )

    def evaluate_replication(self, batch, candidate, *, trial_context, capture_traces):
        self.replications += 1
        if self.fail_replication and self.replications == 2:
            self.executions += len(batch) * self.cost
            raise RuntimeError("score-dependent infrastructure failure")
        return self.evaluate(batch, candidate, capture_traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {key: [{"score": eval_batch.scores[0]}] for key in components_to_update}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        return dict.fromkeys(components_to_update, "lambda x: 1")


def run(tmp_path, adapter=None, budget=300, config=None, **kwargs):
    return gepa.optimize(
        seed_candidate={"program": "lambda x: 0"},
        trainset=[1],
        valset=[2],
        adapter=adapter or ScriptedAdapter(),
        reflection_minibatch_size=kwargs.pop("reflection_minibatch_size", 1),
        max_metric_calls=budget,
        skip_perfect_score=False,
        run_dir=str(tmp_path),
        logger=Quiet(),
        sequential_evidence=config or SequentialEvidenceConfig("script-v1", 200),
        **kwargs,
    )


def test_actual_native_optimize_preserves_scores_and_exposes_evidence(tmp_path):
    raw = ScriptedAdapter()
    result = run(tmp_path, raw, stop_callbacks=lambda state: len(state.program_candidates) >= 2)
    assert result.candidates == [{"program": "lambda x: 0"}, {"program": "lambda x: 1"}]
    assert result.val_aggregate_scores == [0, 1]
    assert result.total_metric_calls == raw.executions
    report = result.sequential_evidence
    assert report["comparisons"][0]["status"] == "promoted"
    assert report["comparisons"][0]["lower"] > 0
    assert not report["validation_confidence_calibrated"]
    assert GEPAResult.from_dict(result.to_dict()).sequential_evidence == report


def test_native_accounting_does_not_double_charge_reported_cost_or_seed(tmp_path):
    raw = ScriptedAdapter(cost=2)
    result = run(tmp_path, raw, budget=9, config=SequentialEvidenceConfig("script-v1", 200, max_calls_per_example=2))
    assert result.total_metric_calls == raw.executions == 8
    assert raw.replications == 0
    assert len(result.candidates) == 1


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 8])
def test_every_native_boundary_obeys_global_cap(tmp_path, budget):
    raw = ScriptedAdapter()
    result = run(tmp_path, raw, budget=budget)
    assert result.total_metric_calls == raw.executions <= budget
    assert len(result.candidates) == 1


def test_failure_is_inconclusive_and_never_promoted(tmp_path):
    raw = ScriptedAdapter(fail_replication=True)
    result = run(tmp_path, raw, budget=7)
    assert len(result.candidates) == 1
    assert result.sequential_evidence["comparisons"][0]["status"] == "inconclusive"
    assert raw.replications == 2
    assert result.total_metric_calls == raw.executions


class Explore:
    def select(self, proposals, state, criterion):
        return proposals[:1]


def test_custom_selection_exploration_is_not_confidence_acceptance(tmp_path):
    result = run(
        tmp_path,
        budget=6,
        config=SequentialEvidenceConfig("script-v1", 2),
        selection_strategy=Explore(),
        stop_callbacks=lambda state: len(state.program_candidates) > 1,
    )
    assert len(result.candidates) == 2
    assert result.sequential_evidence["comparisons"][0]["status"] == "exploration"
    assert result.sequential_evidence["comparisons"][0]["selection_label"] == "unresolved_exploration"


def test_full_validation_shortage_defers_then_resumes_without_reracing(tmp_path):
    first_adapter = ScriptedAdapter()
    full = run(tmp_path / "measure", first_adapter, stop_callbacks=lambda state: len(state.program_candidates) > 1)
    cap = full.total_metric_calls - 1
    first = run(tmp_path / "resume", budget=cap)
    assert len(first.candidates) == 1
    assert first.sequential_evidence["comparisons"][0]["status"] == "validation_deferred"
    resumed_adapter = ScriptedAdapter()
    second = run(tmp_path / "resume", resumed_adapter, budget=cap + 1)
    assert len(second.candidates) == 2
    assert resumed_adapter.replications == 0 and resumed_adapter.executions == 1
    assert second.total_metric_calls == cap + 1


def test_merge_and_unbounded_custom_paths_rejected_before_seed(tmp_path):
    raw = ScriptedAdapter()
    with pytest.raises(ValueError, match="merge"):
        run(tmp_path / "merge", raw, use_merge=True)

    class CustomPolicy(FullEvaluationPolicy):
        pass

    with pytest.raises(ValueError, match="FullEvaluationPolicy"):
        run(tmp_path / "policy", raw, val_evaluation_policy=CustomPolicy())
    assert raw.executions == 0


def test_default_ordinary_path_unchanged(tmp_path):
    raw = ScriptedAdapter()
    result = gepa.optimize(
        {"program": "lambda x: 0"},
        [1],
        [2],
        raw,
        max_metric_calls=4,
        reflection_minibatch_size=1,
        skip_perfect_score=False,
        logger=Quiet(),
        run_dir=str(tmp_path),
    )
    assert result.total_metric_calls == 4
    assert result.val_aggregate_scores == [0, 1]
    assert result.sequential_evidence is None
    assert "sequential_evidence" not in result.to_dict()


def test_resume_disallows_disabling_evidence_or_changing_seed(tmp_path):
    run(tmp_path, budget=3)
    with pytest.raises(ValueError, match="evidence"):
        gepa.optimize(
            {"program": "lambda x: 0"},
            [1],
            [2],
            ScriptedAdapter(),
            max_metric_calls=10,
            run_dir=str(tmp_path),
            logger=Quiet(),
        )
    state = json.loads((tmp_path / "sequential_evidence.json").read_text())
    assert state["payload"]["guard"]["records"]


def test_callback_summary_is_observational_and_original_criterion_called_once(tmp_path):
    from gepa.core.callbacks import SequentialEvidenceEvent

    events: list[SequentialEvidenceEvent] = []

    class Criterion:
        def __init__(self):
            self.calls = 0

        def should_accept(self, proposal, state):
            self.calls += 1
            return True

    class InspectAndSelect:
        def select(self, proposals, state, criterion):
            for proposal in proposals:
                assert criterion.should_accept(proposal, state)
                assert criterion.should_accept(proposal, state)
            return proposals

    class Observer:
        def on_sequential_evidence(self, event):
            events.append(event)
            event["decision"]["parent_ids"].append(99)

    criterion = Criterion()
    result = run(
        tmp_path,
        acceptance_criterion=criterion,
        selection_strategy=InspectAndSelect(),
        callbacks=[Observer()],
        stop_callbacks=lambda state: len(state.program_candidates) > 1,
    )
    assert criterion.calls == 1
    assert result.parents[1] == [0]
    assert len(events) == 2
    assert result.sequential_evidence["comparisons"][0]["parent_ids"] == [0]


def test_explicit_stop_callback_prevents_resumed_pending_validation(tmp_path):
    full = run(tmp_path / "measure", stop_callbacks=lambda state: len(state.program_candidates) > 1)
    cap = full.total_metric_calls - 1
    first = run(tmp_path / "resume", budget=cap)
    raw = ScriptedAdapter()
    stopped = run(tmp_path / "resume", raw, budget=cap + 1, stop_callbacks=lambda state: True)
    assert raw.executions == 0
    assert len(stopped.candidates) == 1
    assert stopped.total_metric_calls == first.total_metric_calls


def test_repeated_resume_never_retries_a_failed_validation(tmp_path):
    class BrokenValidation(ScriptedAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            if batch == [2] and candidate["program"] == "lambda x: 1":
                self.executions += 1
                raise RuntimeError("validation infrastructure")
            return super().evaluate(batch, candidate, capture_traces)

    raw = BrokenValidation()
    result = run(tmp_path, raw, stop_callbacks=lambda state: len(state.program_candidates) > 1)
    assert len(result.candidates) == 1
    assert result.sequential_evidence["comparisons"][0]["status"] == "inconclusive"
    resumed_raw = ScriptedAdapter()
    resumed = run(tmp_path, resumed_raw, budget=301, stop_callbacks=lambda state: True)
    assert resumed_raw.executions == 0
    assert len(resumed.candidates) == 1


def test_custom_candidate_proposer_rejected_before_seed(tmp_path):
    raw = ScriptedAdapter()
    raw.propose_new_texts = None
    with pytest.raises(ValueError, match="custom proposers"):
        run(tmp_path, raw, custom_candidate_proposer=lambda *args: {"program": "lambda x: 1"})
    assert raw.executions == 0


def test_multi_proposal_order_and_cache_do_not_reuse_independent_evidence(tmp_path):
    from gepa.strategies.proposal_sampling import SameParentSampling

    raw = ScriptedAdapter()
    result = run(
        tmp_path,
        raw,
        budget=500,
        sampling_strategy=SameParentSampling(2),
        cache_evaluation=True,
        stop_callbacks=lambda state: len(state.program_candidates) > 1,
    )
    report = result.sequential_evidence
    assert len(result.candidates) == 2
    # Identical child + parent + frozen IDs are one comparison in the batch,
    # and selection deduplicates their shared candidate content.
    assert len(report["comparisons"]) == 1
    assert raw.replications == report["comparisons"][0]["completed_rounds"] * 2
    assert result.total_metric_calls == raw.executions


def test_seed_requires_complete_validation_budget_before_any_dispatch(tmp_path):
    from gepa.evidence.budget import BudgetExhaustedError

    raw = ScriptedAdapter()
    with pytest.raises(BudgetExhaustedError):
        gepa.optimize(
            {"program": "lambda x: 0"},
            [1],
            [2, 3],
            raw,
            max_metric_calls=1,
            sequential_evidence=SequentialEvidenceConfig("script-v1", 200),
            run_dir=str(tmp_path),
            logger=Quiet(),
        )
    assert raw.executions == 0


def test_native_multiple_selected_children_defer_only_unvalidated_tail(tmp_path):
    from gepa.strategies.proposal_sampling import SameParentSampling

    class Distinct(ScriptedAdapter):
        def __init__(self):
            super().__init__()
            self.proposed = 0

        def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            self.proposed += 1
            return dict.fromkeys(components_to_update, f"lambda x: 1 + {self.proposed} * 0")

        def evaluate(self, batch, candidate, capture_traces=False):
            self.executions += len(batch)
            scores = [float(eval(candidate["program"])(x)) for x in batch]
            return EvaluationBatch(
                outputs=scores, scores=scores, trajectories=[{"value": s} for s in scores] if capture_traces else None
            )

    full = run(
        tmp_path / "measure",
        Distinct(),
        budget=500,
        sampling_strategy=SameParentSampling(2),
        stop_callbacks=lambda state: len(state.program_candidates) == 3,
    )
    first = run(
        tmp_path / "resume", Distinct(), budget=full.total_metric_calls - 1, sampling_strategy=SameParentSampling(2)
    )
    assert len(first.candidates) == 2
    statuses = [r["status"] for r in first.sequential_evidence["comparisons"]]
    assert statuses == ["promoted", "validation_deferred"]
    raw = Distinct()
    resumed = run(tmp_path / "resume", raw, budget=full.total_metric_calls, sampling_strategy=SameParentSampling(2))
    assert len(resumed.candidates) == 3 and raw.executions == 1
    assert raw.replications == 0


def test_custom_stateful_reflection_path_rejected_before_seed(tmp_path):
    class CustomReflection:
        def reflect(self, *args, **kwargs):
            raise AssertionError("unsupported custom path should not execute")

    raw = ScriptedAdapter()
    raw.propose_new_texts = None
    with pytest.raises(ValueError, match="stateless reflection"):
        run(tmp_path, raw, reflection_strategy=CustomReflection())
    assert raw.executions == 0
