"""Consumer-contract tests for the DSPy integration.

These tests intentionally exercise GEPA through ``dspy.GEPA``. They protect
the API and behavioral seam used by DSPy without constraining GEPA's internal
implementation. The regular test environment does not install DSPy, so this
module is skipped there and run by the dedicated DSPy compatibility CI job.
"""

from __future__ import annotations

import inspect
import threading
from pathlib import Path

import pytest

dspy = pytest.importorskip("dspy", reason="DSPy compatibility tests run in their dedicated CI job")

from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.utils.dummies import DummyLM

from gepa import EvaluationBatch, GEPAAdapter, GEPAResult, optimize
from gepa.core.adapter import ProposalFn
from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector
from gepa.strategies.acceptance import ImprovementOrEqualAcceptance
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.strategies.proposal_sampling import SameParentSampling
from gepa.strategies.proposal_selection import AllImprovements


class TwoStageProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("question -> category")
        self.answer = dspy.Predict("question, category -> answer")

    def forward(self, question):
        category = self.classify(question=question).category
        return self.answer(question=question, category=category)


class RecordingProposer:
    """The original, metadata-free ProposalFn shape supported by DSPy 3.3.1."""

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()
        self._proposal_index = 0

    def __call__(self, candidate, reflective_dataset, components_to_update):
        with self._lock:
            self._proposal_index += 1
            proposal_index = self._proposal_index
            self.calls.append((candidate, reflective_dataset, components_to_update))
        return {
            name: f"{candidate[name]} Compatibility proposal {proposal_index}."
            for name in components_to_update
        }


def feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return dspy.Prediction(score=0.2, feedback=f"Answer {gold.answer!r} more accurately.")


def objective_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return dspy.Prediction(
        score=0.2,
        feedback="Improve accuracy while staying concise.",
        objective_scores={"accuracy": 0.2, "brevity": 1.0},
    )


def single_example():
    return dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question")


def task_lm(num_responses=200):
    return DummyLM([{"answer": "four", "category": "math"}] * num_responses)


def test_dspy_imported_contract_is_available():
    """Additive API changes are fine; DSPy's consumed subset must remain."""
    assert callable(optimize)
    assert GEPAResult is not None
    assert GEPAAdapter is not None
    assert ProposalFn is not None
    assert ReflectionComponentSelector is not None
    assert InstructionProposalSignature is not None

    optimize_parameters = set(inspect.signature(optimize).parameters)
    assert {
        "seed_candidate",
        "trainset",
        "valset",
        "adapter",
        "reflection_lm",
        "module_selector",
        "max_metric_calls",
        "run_dir",
        "track_best_outputs",
        "raise_on_exception",
        "seed",
    } <= optimize_parameters

    assert {"outputs", "scores", "trajectories", "objective_scores"} <= set(
        EvaluationBatch.__dataclass_fields__
    )
    assert {
        "candidates",
        "parents",
        "val_aggregate_scores",
        "val_subscores",
        "per_val_instance_best_candidates",
        "discovery_eval_counts",
        "best_outputs_valset",
    } <= set(GEPAResult.__dataclass_fields__)


def test_default_reflection_proposer_compiles_a_runnable_program():
    reflection_lm = DummyLM([{"new_instruction": "Answer accurately and directly."}] * 10)
    trainset = [single_example()]

    with dspy.context(lm=task_lm()):
        compiled = dspy.GEPA(
            metric=feedback_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=3,
            reflection_minibatch_size=1,
            skip_perfect_score=False,
            use_merge=False,
        ).compile(dspy.Predict("question -> answer"), trainset=trainset, valset=trainset)
        prediction = compiled(question="What is 3 + 3?")

    assert isinstance(compiled, dspy.Module)
    assert compiled.signature.instructions
    assert prediction.answer
    assert reflection_lm.history, "The default GEPA instruction proposer should invoke the reflection LM"


def test_custom_proposer_selector_and_metric_trace_contracts():
    proposer = RecordingProposer()
    selector_calls = []
    metric_calls = []

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        metric_calls.append((trace, pred_name, pred_trace))
        return dspy.Prediction(score=0.2, feedback="Improve this stage.")

    def select_all(state, trajectories, subsample_scores, candidate_idx, candidate):
        selector_calls.append((state, trajectories, subsample_scores, candidate_idx, candidate))
        return list(candidate)

    trainset = [single_example()]
    with dspy.context(lm=task_lm()):
        compiled = dspy.GEPA(
            metric=metric,
            instruction_proposer=proposer,
            component_selector=select_all,
            max_metric_calls=4,
            reflection_minibatch_size=1,
            skip_perfect_score=False,
            use_merge=False,
        ).compile(TwoStageProgram(), trainset=trainset, valset=trainset)
        prediction = compiled(question="What is 3 + 3?")

    assert proposer.calls
    assert selector_calls
    assert set(selector_calls[0][-1]) == {"classify", "answer"}
    assert any(pred_name in {"classify", "answer"} and pred_trace for _, pred_name, pred_trace in metric_calls)
    assert {name for name, _ in compiled.named_predictors()} == {"classify", "answer"}
    assert prediction.answer


def test_result_and_multi_objective_shapes_remain_dspy_compatible():
    trainset = [single_example()]
    with dspy.context(lm=task_lm()):
        compiled = dspy.GEPA(
            metric=objective_metric,
            instruction_proposer=RecordingProposer(),
            max_metric_calls=3,
            reflection_minibatch_size=1,
            skip_perfect_score=False,
            use_merge=False,
            track_stats=True,
            track_best_outputs=True,
            gepa_kwargs={"frontier_type": "objective"},
        ).compile(dspy.Predict("question -> answer"), trainset=trainset, valset=trainset)

    result = compiled.detailed_results
    assert result.candidates
    assert isinstance(result.best_candidate, dspy.Module)
    assert all(isinstance(scores, dict) for scores in result.val_subscores)
    assert isinstance(result.per_val_instance_best_candidates, dict)
    assert isinstance(result.best_outputs_valset, dict)
    assert result.val_aggregate_subscores == [{"accuracy": 0.2, "brevity": 1.0}]
    assert result.per_objective_best_candidates == {"accuracy": {0}, "brevity": {0}}
    assert result.objective_pareto_front == {"accuracy": 0.2, "brevity": 1.0}
    assert isinstance(result.to_dict()["per_val_instance_best_candidates"], dict)


def test_multi_proposal_optimization_uses_dspy_batch_evaluate(monkeypatch):
    batch_calls = []
    original_batch_evaluate = DspyAdapter.batch_evaluate

    def recording_batch_evaluate(self, items, *, capture_traces=True):
        batch_calls.append((len(items), capture_traces))
        return original_batch_evaluate(self, items, capture_traces=capture_traces)

    monkeypatch.setattr(DspyAdapter, "batch_evaluate", recording_batch_evaluate)
    trainset = [
        dspy.Example(question=f"Question {index}", answer="answer").with_inputs("question")
        for index in range(2)
    ]

    with dspy.context(lm=task_lm()):
        compiled = dspy.GEPA(
            metric=feedback_metric,
            instruction_proposer=RecordingProposer(),
            max_metric_calls=8,
            reflection_minibatch_size=1,
            skip_perfect_score=False,
            use_merge=False,
            num_threads=4,
            track_stats=True,
            gepa_kwargs={
                "sampling_strategy": SameParentSampling(n=2),
                "selection_strategy": AllImprovements(),
                "acceptance_criterion": ImprovementOrEqualAcceptance(),
            },
        ).compile(dspy.Predict("question -> answer"), trainset=trainset, valset=trainset)

    assert any(num_items == 2 and capture_traces for num_items, capture_traces in batch_calls)
    assert len(compiled.detailed_results.candidates) >= 3


def test_checkpoint_resume_preserves_dspy_candidates_and_advances_budget(tmp_path):
    run_dir = tmp_path / "gepa-run"
    trainset = [single_example()]
    proposer = RecordingProposer()

    def compile_with_budget(max_metric_calls):
        return dspy.GEPA(
            metric=feedback_metric,
            instruction_proposer=proposer,
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=1,
            skip_perfect_score=False,
            use_merge=False,
            log_dir=str(run_dir),
            track_stats=True,
            gepa_kwargs={"use_cloudpickle": True},
        ).compile(dspy.Predict("question -> answer"), trainset=trainset, valset=trainset)

    with dspy.context(lm=task_lm()):
        first = compile_with_budget(3)
        resumed = compile_with_budget(8)

    first_result = first.detailed_results
    resumed_result = resumed.detailed_results
    first_components = [
        {name: predictor.signature.instructions for name, predictor in candidate.named_predictors()}
        for candidate in first_result.candidates
    ]
    resumed_components = [
        {name: predictor.signature.instructions for name, predictor in candidate.named_predictors()}
        for candidate in resumed_result.candidates
    ]
    assert Path(resumed_result.log_dir) == run_dir
    assert resumed_result.total_metric_calls > first_result.total_metric_calls
    assert resumed_components[: len(first_components)] == first_components
    with dspy.context(lm=task_lm()):
        assert resumed_result.best_candidate(question="What is 3 + 3?").answer


def test_per_example_metric_failure_preserves_evaluation_alignment():
    student = dspy.Predict("question -> answer")
    batch = [
        dspy.Example(question="works").with_inputs("question"),
        dspy.Example(question="fails").with_inputs("question"),
    ]

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        if gold.question == "fails":
            raise ValueError("example-specific metric failure")
        return 1.0

    adapter = DspyAdapter(student, metric, {}, failure_score=-1.0, num_threads=1)
    with dspy.context(lm=DummyLM([{"answer": "ok"}, {"answer": "bad"}])):
        result = adapter.evaluate(
            batch,
            {"self": student.signature.instructions},
            capture_traces=False,
        )

    assert len(result.outputs) == len(batch)
    assert result.scores == [1.0, -1.0]
    assert result.trajectories is None
