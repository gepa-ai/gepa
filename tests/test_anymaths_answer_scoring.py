"""Regression coverage for AnyMaths answer scoring (gepa-ai/gepa#149)."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gepa.adapters.anymaths_adapter.anymaths_adapter import AnyMathsAdapter


def _response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.parametrize(
    ("expected", "generated", "correct"),
    [
        ("2", "42", False),
        ("2", "-2", False),
        ("2", "2.5", False),
        ("0", "10", False),
        ("2", "2 or 3", False),
        ("2", "2", True),
        (" 42 ", "42", True),
        ("42", "42.0", True),
        ("1/2", "0.5", True),
        ("1000", "1e3", True),
        ("x + y", "x + y", True),
        ("x + y", "2(x + y)", False),
        ("NaN", "NaN", False),
        ("Infinity", "Infinity", False),
        ("1/0", "1/0", False),
        ("", "", False),
    ],
)
def test_complete_answer_agreement_controls_score_and_reflection(monkeypatch, expected, generated, correct):
    adapter = AnyMathsAdapter(model="openai/test", failure_score=0.25, api_base=None)
    monkeypatch.setattr(
        adapter.litellm,
        "batch_completion",
        lambda **_kwargs: [_response(json.dumps({"solution_pad": "Work", "final_answer": generated}))],
    )
    candidate = {"system": "Solve the problem."}
    batch = [{"input": "Question", "answer": expected, "additional_context": {}}]

    result = adapter.evaluate(batch, candidate, capture_traces=True)

    assert result.scores == [1.0 if correct else adapter.failure_score]
    assert result.trajectories[0]["full_assistant_response"] == result.outputs[0]["full_assistant_response"]
    feedback = adapter.make_reflective_dataset(candidate, result, ["system"])["system"][0]["Feedback"]
    assert feedback.startswith(
        "The generated response is correct." if correct else "The generated response is incorrect."
    )


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not json",
        "[]",
        '{"solution_pad": "Work"}',
        '{"solution_pad": null, "final_answer": "2"}',
        '{"solution_pad": "Work", "final_answer": 2}',
    ],
)
def test_malformed_structured_output_receives_failure_feedback(monkeypatch, content):
    adapter = AnyMathsAdapter(model="openai/test", failure_score=0.25, api_base=None)
    monkeypatch.setattr(adapter.litellm, "batch_completion", lambda **_kwargs: [_response(content)])
    candidate = {"system": "Solve the problem."}
    batch = [{"input": "Question", "answer": "2", "additional_context": {}}]

    result = adapter.evaluate(batch, candidate, capture_traces=True)

    assert result.scores == [adapter.failure_score]
    assert len(result.outputs) == len(result.trajectories) == len(batch)
    feedback = adapter.make_reflective_dataset(candidate, result, ["system"])["system"][0]["Feedback"]
    assert feedback.startswith("The generated response is incorrect.")


def test_standalone_evaluation_uses_the_same_complete_answer_metric(monkeypatch, capsys):
    batch = [{"input": "Question", "answer": "2", "additional_context": {}} for _ in range(3)]
    monkeypatch.setitem(sys.modules, "train_anymaths", SimpleNamespace(init_dataset=lambda _dataset: ([], [], batch)))
    adapter = AnyMathsAdapter(model="openai/test", api_base=None)
    monkeypatch.setattr(
        adapter.litellm,
        "batch_completion",
        lambda **_kwargs: [
            _response(json.dumps({"solution_pad": "Work", "final_answer": answer})) for answer in ("2", "42", "-2")
        ],
    )
    entry = Path(__file__).resolve().parents[1] / "src/gepa/examples/anymaths-bench/eval_default.py"
    monkeypatch.setattr(sys, "argv", [str(entry), "--model", "openai/test", "--batch_size", "3"])

    runpy.run_path(str(entry), run_name="__main__")

    assert "Final score >> 1.0 / 3" in capsys.readouterr().out
