"""Tests for DspyAdapter (full-program evolution adapter).

Covers:
1. evaluate() returned EvaluationBatch(outputs=None, ...) on build failure,
   crashing downstream zip() in cached_evaluate_full.
2. reflection_lm was typed as dspy.LM but must conform to the LanguageModel
   protocol (callable returning str, not list[str]).
3. make_reflective_dataset must collapse consecutive cumulative ReAct traces
   (issue 97) without dropping independent map calls, interleaved A-B-A calls,
   History inputs, or distinct predictors in a multi-module program.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", reason="dspy is not installed — skipping DspyAdapter tests")

from unittest.mock import MagicMock, patch

import dspy
from dspy.primitives import Example
from dspy.teleprompt.bootstrap_trace import FailedPrediction
from dspy.utils.dummies import DummyLM

from gepa.adapters.dspy_full_program_adapter.full_program_adapter import (
    DspyAdapter,
    _select_trace_instances_for_reflection,
)
from gepa.core.adapter import EvaluationBatch
from gepa.proposer.reflective_mutation.base import LanguageModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(reflection_lm=None, task_lm=None):
    """Build a DspyAdapter with mocked dependencies."""
    if task_lm is None:
        task_lm = MagicMock(spec=dspy.LM)
    metric_fn = MagicMock(return_value=1.0)
    if reflection_lm is None:
        reflection_lm = MagicMock(spec=LanguageModel)
    return DspyAdapter(
        task_lm=task_lm,
        metric_fn=metric_fn,
        reflection_lm=reflection_lm,
        failure_score=0.0,
        num_threads=1,
    )


def _make_batch(n=3):
    """Create a minimal batch of DSPy Examples."""
    return [Example(question=f"q{i}").with_inputs("question") for i in range(n)]


def _make_predictor():
    """A mock predictor whose signature.equals matches only this predictor."""
    predictor = MagicMock()
    predictor.signature.equals.side_effect = lambda other: other is predictor.signature
    return predictor


def _program_trace(adapter, trace, named_predictors):
    proposed_program = MagicMock()
    proposed_program.named_predictors.return_value = named_predictors
    adapter.build_program = MagicMock(return_value=(proposed_program, None))

    example = Example(question="What is 2+2?").with_inputs("question")
    eval_batch = MagicMock()
    eval_batch.trajectories = [
        {
            "trace": trace,
            "example": example,
            "prediction": {"answer": "4"},
            "score": 1.0,
        }
    ]
    result = adapter.make_reflective_dataset(
        candidate={"program": "dummy"},
        eval_batch=eval_batch,
        components_to_update=["program"],
    )
    return result["program"][0]["Program Trace"]


def _reflective_trace_from_live_program(candidate_src, batch, dummy_answers):
    """Run evaluate(capture_traces=True) then make_reflective_dataset with DummyLM."""
    lm = DummyLM(list(dummy_answers))
    dspy.configure(lm=lm)
    adapter = DspyAdapter(
        task_lm=lm,
        metric_fn=lambda example, pred, trace=None: 1.0,
        reflection_lm=MagicMock(spec=LanguageModel),
        failure_score=0.0,
        num_threads=1,
    )
    eval_batch = adapter.evaluate(batch, {"program": candidate_src}, capture_traces=True)
    assert eval_batch.trajectories, "expected captured traces from DummyLM program"
    result = adapter.make_reflective_dataset(
        candidate={"program": candidate_src},
        eval_batch=eval_batch,
        components_to_update=["program"],
    )
    return result["program"][0]["Program Trace"], eval_batch


# ---------------------------------------------------------------------------
# Bug 1: outputs must be a list, even on build failure
# ---------------------------------------------------------------------------


class TestEvaluateOutputsOnBuildFailure:
    """When the candidate program fails to build, evaluate() must still
    return an EvaluationBatch with a list of outputs (not None)."""

    def test_outputs_is_list_on_syntax_error(self):
        adapter = _make_adapter()
        candidate = {"program": "def foo(  # syntax error"}
        batch = _make_batch(4)

        result = adapter.evaluate(batch, candidate, capture_traces=False)

        assert isinstance(result, EvaluationBatch)
        assert isinstance(result.outputs, list), f"outputs should be a list, got {type(result.outputs)}"
        assert len(result.outputs) == len(batch)
        assert len(result.scores) == len(batch)
        assert all(s == 0.0 for s in result.scores)

    def test_outputs_is_list_on_missing_program_object(self):
        adapter = _make_adapter()
        # Valid Python but doesn't define `program`
        candidate = {"program": "x = 42"}
        batch = _make_batch(2)

        result = adapter.evaluate(batch, candidate, capture_traces=False)

        assert isinstance(result.outputs, list)
        assert len(result.outputs) == len(batch)

    def test_outputs_is_list_on_runtime_error(self):
        adapter = _make_adapter()
        candidate = {"program": "raise RuntimeError('boom')"}
        batch = _make_batch(5)

        result = adapter.evaluate(batch, candidate, capture_traces=False)

        assert isinstance(result.outputs, list)
        assert len(result.outputs) == len(batch)

    def test_outputs_zippable_with_example_ids(self):
        """Reproduce the exact crash from cached_evaluate_full:
        dict(zip(example_ids, outputs)) must not raise."""
        adapter = _make_adapter()
        candidate = {"program": "def foo(  # syntax error"}
        batch = _make_batch(3)

        result = adapter.evaluate(batch, candidate, capture_traces=False)
        example_ids = list(range(len(batch)))

        # This is the exact operation that crashed before the fix
        outputs_by_id = dict(zip(example_ids, result.outputs, strict=False))
        scores_by_id = dict(zip(example_ids, result.scores, strict=False))

        assert len(outputs_by_id) == len(batch)
        assert len(scores_by_id) == len(batch)


# ---------------------------------------------------------------------------
# Bug 2: reflection_lm must conform to LanguageModel protocol
# ---------------------------------------------------------------------------


class TestReflectionLmProtocol:
    """The reflection_lm parameter should accept any callable that returns str,
    not require a dspy.LM specifically."""

    def test_lambda_wrapper_accepted(self):
        """A lambda wrapping dspy.LM (as shown in GEPA's example notebook)
        should be accepted as reflection_lm."""
        mock_dspy_lm = MagicMock(spec=dspy.LM)
        mock_dspy_lm.return_value = ["response text"]
        wrapped = lambda x: mock_dspy_lm(x)[0]

        # Should not raise
        adapter = _make_adapter(reflection_lm=wrapped)
        assert adapter.reflection_lm is wrapped

    def test_plain_callable_accepted(self):
        """Any callable (str) -> str should work as reflection_lm."""

        def my_lm(prompt):
            return "generated response"

        adapter = _make_adapter(reflection_lm=my_lm)
        assert adapter.reflection_lm is my_lm

    def test_propose_new_texts_calls_lm_correctly(self):
        """propose_new_texts should pass the prompt to reflection_lm and use
        the str return value (not a list)."""
        mock_lm = MagicMock(return_value="<new_program>\nimport dspy\nprogram = dspy.Predict('q -> a')\n</new_program>")
        adapter = _make_adapter(reflection_lm=mock_lm)

        candidate = {"program": "import dspy\nprogram = dspy.Predict('q -> a')"}
        reflective_dataset = {"program": [{"input": "q1", "output": "a1", "score": 0.5}]}

        # The proposal signature will call lm(prompt) and expect a str back.
        # We mock the signature's run method to verify the LM is called.
        with patch(
            "gepa.adapters.dspy_full_program_adapter.dspy_program_proposal_signature.DSPyProgramProposalSignature.run",
            return_value={"new_program": "import dspy\nprogram = dspy.Predict('q -> a')"},
        ) as mock_run:
            result = adapter.propose_new_texts(candidate, reflective_dataset, ["program"])
            mock_run.assert_called_once_with(
                lm=mock_lm,
                input_dict={
                    "curr_program": candidate["program"],
                    "dataset_with_feedback": reflective_dataset["program"],
                },
            )
            assert "program" in result


# ---------------------------------------------------------------------------
# Issue #97: collapse consecutive cumulative traces only
# ---------------------------------------------------------------------------


class TestSelectTraceInstancesForReflection:
    def test_react_keeps_last_react_call_and_extract(self):
        react = object()
        extract = object()
        t0 = "thought_0: add\ntool_name_0: calculator\nobservation_0: 4"
        t1 = t0 + "\nthought_1: done\ntool_name_1: finish\nobservation_1: Completed."
        trace = [
            (
                react,
                {"question": "What is 2+2?", "trajectory": ""},
                {"next_thought": "add", "next_tool_name": "calculator"},
            ),
            (
                react,
                {"question": "What is 2+2?", "trajectory": t0},
                {"next_thought": "done", "next_tool_name": "finish"},
            ),
            (extract, {"question": "What is 2+2?", "trajectory": t1}, {"answer": "4"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == [trace[1], trace[2]]
        assert t0 in selected[0][1]["trajectory"]
        assert t0 in selected[1][1]["trajectory"]
        assert "thought_1: done" in selected[1][1]["trajectory"]

    def test_distinct_predictors_are_all_kept(self):
        reasoner = object()
        extractor = object()
        trace = [
            (reasoner, {"question": "q"}, {"reasoning": "subtract 3", "answer": "5"}),
            (extractor, {"question": "q", "reasoning": "subtract 3"}, {"answer": "4"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace

    def test_independent_same_predictor_calls_are_all_kept(self):
        classify = object()
        trace = [
            (classify, {"item": "apple"}, {"label": "fruit"}),
            (classify, {"item": "carrot"}, {"label": "veg"}),
            (classify, {"item": "salmon"}, {"label": "fish"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert [t[1]["item"] for t in selected] == ["apple", "carrot", "salmon"]

    def test_prefix_lookalike_independent_items_are_not_collapsed(self):
        classify = object()
        trace = [
            (classify, {"item": "cat"}, {"label": "animal"}),
            (classify, {"item": "catalog"}, {"label": "object"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace

    def test_prefix_lookalike_context_strings_are_not_collapsed(self):
        """Greptile: context abc then abcdef is independent, not a ReAct trajectory."""
        reader = object()
        trace = [
            (reader, {"context": "abc", "question": "q1"}, {"answer": "a1"}),
            (reader, {"context": "abcdef", "question": "q2"}, {"answer": "a2"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert [t[2]["answer"] for t in selected] == ["a1", "a2"]

    def test_multiline_context_append_is_not_collapsed(self):
        """Greptile: appending a newline section to context/document is not ReAct."""
        reader = object()
        first = "doc A"
        second = first + "\ndoc B"
        trace = [
            (reader, {"context": first}, {"answer": "a1"}),
            (reader, {"context": second}, {"answer": "a2"}),
            (reader, {"document": first + "\nintro"}, {"answer": "a3"}),
            (reader, {"document": first + "\nintro\nbody"}, {"answer": "a4"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert [t[2]["answer"] for t in selected] == ["a1", "a2", "a3", "a4"]

    def test_interleaved_draft_critique_revise_keeps_chronology(self):
        gen = object()
        crit = object()
        trace = [
            (gen, {"task": "write", "notes": ""}, {"draft": "v1"}),
            (crit, {"draft": "v1"}, {"feedback": "too vague"}),
            (gen, {"task": "write", "notes": "too vague"}, {"draft": "v2"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert [t[2].get("draft") or t[2].get("feedback") for t in selected] == ["v1", "too vague", "v2"]

    def test_empty_history_then_unrelated_history_is_not_collapsed(self):
        """Empty History is a prefix of every message list, but those calls are independent."""
        talk = object()
        h_empty = dspy.History(messages=[])
        h_other = dspy.History(messages=[{"question": "unrelated session", "answer": "x"}])
        trace = [
            (talk, {"question": "hi", "history": h_empty}, {"answer": "new-user"}),
            (talk, {"question": "yo", "history": h_other}, {"answer": "old-user"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert [t[2]["answer"] for t in selected] == ["new-user", "old-user"]

    def test_shared_prefix_histories_are_not_collapsed(self):
        """Greptile: independent Histories that share a prefix must both stay."""
        talk = object()
        shared = [{"role": "system", "content": "You are helpful"}]
        h1 = dspy.History(messages=shared)
        h2 = dspy.History(messages=shared + [{"role": "user", "content": "hello"}])
        trace = [
            (talk, {"question": "q1", "history": h1}, {"answer": "keep-first"}),
            (talk, {"question": "q2", "history": h2}, {"answer": "keep-second"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert [t[2]["answer"] for t in selected] == ["keep-first", "keep-second"]

    def test_conversation_prefix_scoring_is_not_collapsed(self):
        talk = object()
        msgs = [
            {"question": "a", "answer": "1"},
            {"question": "b", "answer": "2"},
            {"question": "c", "answer": "3"},
        ]
        trace = [
            (talk, {"history": dspy.History(messages=msgs[:1])}, {"score": "s1"}),
            (talk, {"history": dspy.History(messages=msgs[:2])}, {"score": "s2"}),
            (talk, {"history": dspy.History(messages=msgs[:3])}, {"score": "s3"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert [t[2]["score"] for t in selected] == ["s1", "s2", "s3"]

    def test_growing_chat_history_is_not_collapsed(self):
        """A real chat loop grows History as a prefix. That is not ReAct, so every turn stays."""
        talk = object()
        h0 = dspy.History(messages=[])
        first_turn = [{"question": "hi", "answer": "hello"}]
        h1 = dspy.History(messages=first_turn)
        h2 = dspy.History(messages=first_turn + [{"question": "again", "answer": "still here"}])
        trace = [
            (talk, {"question": "hi", "history": h0}, {"answer": "hello"}),
            (talk, {"question": "again", "history": h1}, {"answer": "still here"}),
            (talk, {"question": "more", "history": h2}, {"answer": "final"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert [t[2]["answer"] for t in selected] == ["hello", "still here", "final"]

    def test_independent_histories_of_different_lengths_are_not_collapsed(self):
        """Greptile: unrelated History inputs must not collapse just because one is longer."""
        talk = object()
        h_short = dspy.History(messages=[{"question": "a", "answer": "1"}])
        h_long = dspy.History(
            messages=[
                {"question": "x", "answer": "9"},
                {"question": "y", "answer": "8"},
                {"question": "z", "answer": "7"},
            ]
        )
        trace = [
            (talk, {"question": "q1", "history": h_short}, {"answer": "keep-me"}),
            (talk, {"question": "q2", "history": h_long}, {"answer": "other-session"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert [t[2]["answer"] for t in selected] == ["keep-me", "other-session"]

    def test_two_independent_react_runs_keep_each_final_step(self):
        react = object()
        trace = [
            (react, {"item": "a", "trajectory": ""}, {"next": "a1"}),
            (react, {"item": "a", "trajectory": "t0-a\n"}, {"next": "a2"}),
            (react, {"item": "b", "trajectory": ""}, {"next": "b1"}),
            (react, {"item": "b", "trajectory": "t0-b\n"}, {"next": "b2"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == [trace[1], trace[3]]
        assert [t[2]["next"] for t in selected] == ["a2", "b2"]

    def test_failed_prediction_stays_with_surrounding_calls(self):
        predictor = object()
        failed = FailedPrediction(completion_text="RAW")
        t0 = "thought_0: add\n"
        t1 = t0 + "thought_1: retry\n"
        trace = [
            (predictor, {"step": "1", "trajectory": ""}, {"thought": "first"}),
            (predictor, {"step": "2", "trajectory": t0}, failed),
            (predictor, {"step": "3", "trajectory": t1}, {"answer": "final"}),
        ]

        selected = _select_trace_instances_for_reflection(trace)

        assert selected == trace
        assert selected[1][2] is failed


class TestReflectiveDatasetSelection:
    def test_same_predictor_cumulative_trajectory_keeps_last_entry(self):
        adapter = _make_adapter()
        react = _make_predictor()
        t0 = "thought_0: add\ntool_name_0: calculator\nobservation_0: 4"
        t1 = t0 + "\nthought_1: done\ntool_name_1: finish\nobservation_1: Completed."
        trace = [
            (react, {"question": "What is 2+2?", "trajectory": ""}, {"next_thought": "add"}),
            (react, {"question": "What is 2+2?", "trajectory": t0}, {"next_thought": "use result"}),
            (react, {"question": "What is 2+2?", "trajectory": t1}, {"next_thought": "done", "answer": "4"}),
        ]

        program_trace = _program_trace(adapter, trace, [("react", react)])

        assert len(program_trace) == 1
        assert program_trace[0]["Called Module"] == "react"
        assert program_trace[0]["Generated Outputs"]["answer"] == "4"
        assert t0 in program_trace[0]["Inputs"]["trajectory"]
        assert "thought_1: done" in program_trace[0]["Inputs"]["trajectory"]

    def test_react_keeps_last_react_call_and_extract(self):
        adapter = _make_adapter()
        react = _make_predictor()
        extract = _make_predictor()
        t0 = "thought_0: add\ntool_name_0: calculator\nobservation_0: 4"
        t1 = t0 + "\nthought_1: done\ntool_name_1: finish\nobservation_1: Completed."
        trace = [
            (
                react,
                {"question": "What is 2+2?", "trajectory": ""},
                {"next_thought": "add", "next_tool_name": "calculator"},
            ),
            (
                react,
                {"question": "What is 2+2?", "trajectory": t0},
                {"next_thought": "done", "next_tool_name": "finish"},
            ),
            (
                extract,
                {"question": "What is 2+2?", "trajectory": t1},
                {"answer": "4"},
            ),
        ]

        program_trace = _program_trace(
            adapter,
            trace,
            [("react.react", react), ("react.extract.predict", extract)],
        )

        assert [entry["Called Module"] for entry in program_trace] == [
            "react.react",
            "react.extract.predict",
        ]
        assert program_trace[0]["Generated Outputs"]["next_tool_name"] == "finish"
        assert t0 in program_trace[0]["Inputs"]["trajectory"]
        assert program_trace[1]["Generated Outputs"]["answer"] == "4"
        assert "thought_1: done" in program_trace[1]["Inputs"]["trajectory"]

    def test_two_module_pipeline_keeps_both_predictors(self):
        adapter = _make_adapter()
        reasoner = _make_predictor()
        extractor = _make_predictor()
        trace = [
            (reasoner, {"question": "If x+3=7, what is x?"}, {"reasoning": "subtract 3", "answer": "5"}),
            (
                extractor,
                {"question": "If x+3=7, what is x?", "reasoning": "subtract 3"},
                {"answer": "4"},
            ),
        ]

        program_trace = _program_trace(
            adapter,
            trace,
            [("reasoner.predict", reasoner), ("extractor", extractor)],
        )

        assert [entry["Called Module"] for entry in program_trace] == ["reasoner.predict", "extractor"]
        assert program_trace[0]["Generated Outputs"]["answer"] == "5"
        assert program_trace[1]["Generated Outputs"]["answer"] == "4"

    def test_independent_modules_keep_a_side_call_that_is_not_in_the_last_inputs(self):
        adapter = _make_adapter()
        writer = _make_predictor()
        critic = _make_predictor()
        trace = [
            (writer, {"question": "Write a haiku about rain."}, {"poem": "soft rain falling now"}),
            (critic, {"rubric": "score originality 1-5"}, {"score": "5"}),
        ]

        program_trace = _program_trace(
            adapter,
            trace,
            [("writer", writer), ("critic", critic)],
        )

        assert [entry["Called Module"] for entry in program_trace] == ["writer", "critic"]
        assert program_trace[0]["Generated Outputs"]["poem"] == "soft rain falling now"
        assert "poem" not in program_trace[1]["Inputs"]

    def test_failed_prediction_in_the_middle_is_kept_with_neighbors(self):
        adapter = _make_adapter()
        predictor = _make_predictor()
        failed = FailedPrediction(completion_text="RAW")
        t0 = "thought_0: add\n"
        t1 = t0 + "thought_1: retry\n"
        trace = [
            (predictor, {"step": "1", "trajectory": ""}, {"thought": "first"}),
            (predictor, {"step": "2", "trajectory": t0}, failed),
            (predictor, {"step": "3", "trajectory": t1}, {"answer": "final"}),
        ]

        program_trace = _program_trace(adapter, trace, [("react", predictor)])

        assert len(program_trace) == 3
        assert [entry["Called Module"] for entry in program_trace] == ["react", "react", "react"]
        assert "RAW" in program_trace[1]["Generated Outputs"]
        assert program_trace[2]["Generated Outputs"]["answer"] == "final"

    def test_independent_map_calls_all_appear_in_program_trace(self):
        adapter = _make_adapter()
        classify = _make_predictor()
        trace = [
            (classify, {"item": "apple"}, {"label": "fruit"}),
            (classify, {"item": "carrot"}, {"label": "veg"}),
            (classify, {"item": "salmon"}, {"label": "fish"}),
        ]

        program_trace = _program_trace(adapter, trace, [("classify", classify)])

        assert [entry["Inputs"]["item"] for entry in program_trace] == ["apple", "carrot", "salmon"]
        assert [entry["Generated Outputs"]["label"] for entry in program_trace] == ["fruit", "veg", "fish"]

    def test_interleaved_aba_keeps_draft_then_critique_then_revision(self):
        adapter = _make_adapter()
        gen = _make_predictor()
        crit = _make_predictor()
        trace = [
            (gen, {"task": "write", "notes": ""}, {"draft": "v1"}),
            (crit, {"draft": "v1"}, {"feedback": "too vague"}),
            (gen, {"task": "write", "notes": "too vague"}, {"draft": "v2"}),
        ]

        program_trace = _program_trace(adapter, trace, [("gen", gen), ("crit", crit)])

        assert [entry["Called Module"] for entry in program_trace] == ["gen", "crit", "gen"]
        assert program_trace[0]["Generated Outputs"]["draft"] == "v1"
        assert program_trace[1]["Generated Outputs"]["feedback"] == "too vague"
        assert program_trace[2]["Generated Outputs"]["draft"] == "v2"

    def test_empty_then_unrelated_history_appears_in_program_trace(self):
        adapter = _make_adapter()
        talk = _make_predictor()
        h_empty = dspy.History(messages=[])
        h_other = dspy.History(messages=[{"question": "old", "answer": "session"}])
        trace = [
            (talk, {"question": "hi", "history": h_empty}, {"answer": "new-user"}),
            (talk, {"question": "yo", "history": h_other}, {"answer": "old-user"}),
        ]

        program_trace = _program_trace(adapter, trace, [("talk", talk)])

        assert [entry["Called Module"] for entry in program_trace] == ["talk", "talk"]
        assert [entry["Generated Outputs"]["answer"] for entry in program_trace] == ["new-user", "old-user"]


# ---------------------------------------------------------------------------
# Live DSPy programs (DummyLM, no network)
# ---------------------------------------------------------------------------


REACT_CANDIDATE = '''
import dspy

def calculator(x: str) -> str:
    """Evaluate a python arithmetic expression."""
    return str(eval(x, {"__builtins__": {}}, {}))

class CumulativeProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(
            signature="question -> answer",
            tools=[calculator],
            max_iters=4,
        )

    def forward(self, question):
        return self.react(question=question)

program = CumulativeProgram()
'''

MAP_CANDIDATE = """
import dspy

class MapProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("item -> label")

    def forward(self, items):
        labels = [self.classify(item=item).label for item in items]
        return dspy.Prediction(labels=labels)

program = MapProgram()
"""

REVISE_CANDIDATE = """
import dspy

class ReviseProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict("task, notes -> draft")
        self.crit = dspy.Predict("draft -> feedback")

    def forward(self, task):
        first = self.gen(task=task, notes="")
        critique = self.crit(draft=first.draft)
        return self.gen(task=task, notes=critique.feedback)

program = ReviseProgram()
"""

MATH_CANDIDATE = """
import dspy

class MathProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reasoner = dspy.Predict("question -> reasoning, answer")
        self.extractor = dspy.Predict("question, reasoning -> answer")

    def forward(self, question):
        reasoned = self.reasoner(question=question)
        extracted = self.extractor(question=question, reasoning=reasoned.reasoning)
        return dspy.Prediction(reasoning=reasoned.reasoning, answer=extracted.answer)

program = MathProgram()
"""

HISTORY_MAP_CANDIDATE = """
import dspy

class HistoryMapProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.talk = dspy.Predict("question, history -> answer")

    def forward(self, question_a, history_a, question_b, history_b):
        first = self.talk(question=question_a, history=history_a)
        second = self.talk(question=question_b, history=history_b)
        return dspy.Prediction(answer_a=first.answer, answer_b=second.answer)

program = HistoryMapProgram()
"""

HISTORY_CHAT_CANDIDATE = """
import dspy

class HistoryChatProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.talk = dspy.Predict("question, history -> answer")

    def forward(self, questions):
        history = dspy.History(messages=[])
        answers = []
        for question in questions:
            pred = self.talk(question=question, history=history)
            answers.append(pred.answer)
            history = dspy.History(
                messages=list(history.messages) + [{"question": question, "answer": pred.answer}]
            )
        return dspy.Prediction(answers=answers)

program = HistoryChatProgram()
"""


class TestLiveDspyTraceSelection:
    def test_live_react_collapses_repeated_react_calls(self):
        batch = [Example(question="What is 2+2?", answer="4").with_inputs("question")]
        program_trace, eval_batch = _reflective_trace_from_live_program(
            REACT_CANDIDATE,
            batch,
            [
                {"next_thought": "add", "next_tool_name": "calculator", "next_tool_args": {"x": "2+2"}},
                {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
                {"reasoning": "2+2=4", "answer": "4"},
            ],
        )

        raw_trace = eval_batch.trajectories[0]["trace"]
        assert len(raw_trace) == 3
        assert raw_trace[1][1]["trajectory"].startswith(raw_trace[0][1]["trajectory"])
        assert raw_trace[2][1]["trajectory"].startswith(raw_trace[1][1]["trajectory"])
        assert [entry["Called Module"] for entry in program_trace] == ["react.react", "react.extract.predict"]
        assert program_trace[0]["Generated Outputs"]["next_tool_name"] == "finish"
        assert program_trace[1]["Generated Outputs"]["answer"] == "4"
        assert "thought_0" in program_trace[0]["Inputs"]["trajectory"]
        assert "thought_1" in program_trace[1]["Inputs"]["trajectory"]
        react_entries = [entry for entry in program_trace if entry["Called Module"] == "react.react"]
        assert len(react_entries) == 1

    def test_live_react_collapses_three_tool_steps_to_the_final_react_call(self):
        batch = [Example(question="What is 2+2?", answer="4").with_inputs("question")]
        program_trace, eval_batch = _reflective_trace_from_live_program(
            REACT_CANDIDATE,
            batch,
            [
                {"next_thought": "add", "next_tool_name": "calculator", "next_tool_args": {"x": "2+2"}},
                {"next_thought": "check", "next_tool_name": "calculator", "next_tool_args": {"x": "4"}},
                {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
                {"reasoning": "2+2=4", "answer": "4"},
            ],
        )

        raw_trace = eval_batch.trajectories[0]["trace"]
        assert len(raw_trace) == 4
        for earlier, later in zip(raw_trace, raw_trace[1:], strict=False):
            assert later[1]["trajectory"].startswith(earlier[1]["trajectory"])
        assert [entry["Called Module"] for entry in program_trace] == ["react.react", "react.extract.predict"]
        assert program_trace[0]["Generated Outputs"]["next_tool_name"] == "finish"
        assert "thought_0" in program_trace[0]["Inputs"]["trajectory"]
        assert "thought_1" in program_trace[0]["Inputs"]["trajectory"]
        assert "thought_2" in program_trace[1]["Inputs"]["trajectory"]
        assert program_trace[1]["Generated Outputs"]["answer"] == "4"

    def test_live_map_keeps_every_independent_classify_call(self):
        batch = [Example(items=["apple", "carrot", "salmon"]).with_inputs("items")]
        program_trace, eval_batch = _reflective_trace_from_live_program(
            MAP_CANDIDATE,
            batch,
            [
                {"label": "fruit"},
                {"label": "veg"},
                {"label": "fish"},
            ],
        )

        raw_trace = eval_batch.trajectories[0]["trace"]
        assert len(raw_trace) == 3
        assert [entry["Called Module"] for entry in program_trace] == ["classify", "classify", "classify"]
        assert [entry["Inputs"]["item"] for entry in program_trace] == ["apple", "carrot", "salmon"]
        assert [entry["Generated Outputs"]["label"] for entry in program_trace] == ["fruit", "veg", "fish"]

    def test_live_draft_critique_revise_keeps_aba_order(self):
        batch = [Example(task="write a haiku").with_inputs("task")]
        program_trace, eval_batch = _reflective_trace_from_live_program(
            REVISE_CANDIDATE,
            batch,
            [
                {"draft": "v1"},
                {"feedback": "too vague"},
                {"draft": "v2"},
            ],
        )

        raw_trace = eval_batch.trajectories[0]["trace"]
        assert len(raw_trace) == 3
        assert [entry["Called Module"] for entry in program_trace] == ["gen", "crit", "gen"]
        assert program_trace[0]["Generated Outputs"]["draft"] == "v1"
        assert program_trace[1]["Generated Outputs"]["feedback"] == "too vague"
        assert program_trace[2]["Generated Outputs"]["draft"] == "v2"

    def test_live_two_module_pipeline_keeps_disagreed_answers(self):
        batch = [Example(question="If x+3=7, what is x?").with_inputs("question")]
        program_trace, eval_batch = _reflective_trace_from_live_program(
            MATH_CANDIDATE,
            batch,
            [
                {"reasoning": "subtract 3", "answer": "5"},
                {"answer": "4"},
            ],
        )

        assert len(eval_batch.trajectories[0]["trace"]) == 2
        assert [entry["Called Module"] for entry in program_trace] == ["reasoner", "extractor"]
        assert program_trace[0]["Generated Outputs"]["answer"] == "5"
        assert program_trace[1]["Generated Outputs"]["answer"] == "4"
        assert program_trace[1]["Inputs"]["reasoning"] == "subtract 3"

    def test_live_history_map_keeps_empty_then_unrelated_session(self):
        h_empty = dspy.History(messages=[])
        h_other = dspy.History(messages=[{"question": "old", "answer": "session"}])
        batch = [
            Example(
                question_a="hi",
                history_a=h_empty,
                question_b="yo",
                history_b=h_other,
            ).with_inputs("question_a", "history_a", "question_b", "history_b")
        ]
        program_trace, eval_batch = _reflective_trace_from_live_program(
            HISTORY_MAP_CANDIDATE,
            batch,
            [{"answer": "new-user"}, {"answer": "old-user"}],
        )

        raw_trace = eval_batch.trajectories[0]["trace"]
        assert len(raw_trace) == 2
        assert len(raw_trace[0][1]["history"].messages) == 0
        assert len(raw_trace[1][1]["history"].messages) == 1
        assert [entry["Called Module"] for entry in program_trace] == ["talk", "talk"]
        assert [entry["Generated Outputs"]["answer"] for entry in program_trace] == ["new-user", "old-user"]

    def test_live_history_map_keeps_shared_prefix_sessions(self):
        shared = [{"role": "system", "content": "You are helpful"}]
        h1 = dspy.History(messages=shared)
        h2 = dspy.History(messages=shared + [{"role": "user", "content": "hello"}])
        batch = [
            Example(
                question_a="q1",
                history_a=h1,
                question_b="q2",
                history_b=h2,
            ).with_inputs("question_a", "history_a", "question_b", "history_b")
        ]
        program_trace, eval_batch = _reflective_trace_from_live_program(
            HISTORY_MAP_CANDIDATE,
            batch,
            [{"answer": "keep-first"}, {"answer": "keep-second"}],
        )

        raw_trace = eval_batch.trajectories[0]["trace"]
        assert len(raw_trace) == 2
        assert (
            raw_trace[1][1]["history"].messages[: len(raw_trace[0][1]["history"].messages)]
            == raw_trace[0][1]["history"].messages
        )
        assert [entry["Generated Outputs"]["answer"] for entry in program_trace] == ["keep-first", "keep-second"]

    def test_live_chat_loop_keeps_every_growing_history_turn(self):
        batch = [Example(questions=["hi", "again", "more"]).with_inputs("questions")]
        program_trace, eval_batch = _reflective_trace_from_live_program(
            HISTORY_CHAT_CANDIDATE,
            batch,
            [{"answer": "hello"}, {"answer": "still"}, {"answer": "final"}],
        )

        raw_trace = eval_batch.trajectories[0]["trace"]
        assert len(raw_trace) == 3
        lengths = [len(t[1]["history"].messages) for t in raw_trace]
        assert lengths == [0, 1, 2]
        assert raw_trace[1][1]["history"].messages[: lengths[0]] == raw_trace[0][1]["history"].messages
        assert raw_trace[2][1]["history"].messages[: lengths[1]] == raw_trace[1][1]["history"].messages
        assert [entry["Called Module"] for entry in program_trace] == ["talk", "talk", "talk"]
        assert [entry["Generated Outputs"]["answer"] for entry in program_trace] == ["hello", "still", "final"]
