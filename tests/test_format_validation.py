# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import re

import pytest

from gepa.adapters.default_adapter.default_adapter import EvaluationResult
from gepa.utils import (
    FormatValidationError,
    require_format,
    require_json_output,
    require_regex_match,
    require_xml_output,
)


def test_require_json_output_parses_before_calling_evaluator():
    seen = {}

    @require_json_output(schema={"score": int, "reasoning": str})
    def evaluator(data, response):
        seen["response"] = response
        return response["score"] / 10, {"Output": response, "Expected": data["expected"]}

    score, side_info = evaluator({"expected": 7}, '{"score": 7, "reasoning": "matched"}')

    assert score == 0.7
    assert seen["response"] == {"score": 7, "reasoning": "matched"}
    assert side_info["Output"] == {"score": 7, "reasoning": "matched"}


def test_require_json_output_malformed_json_returns_zero_and_feedback():
    called = False

    @require_json_output(schema={"score": int})
    def evaluator(data, response):
        nonlocal called
        called = True
        return 1.0, {"Output": response}

    score, side_info = evaluator({"expected": 7}, '{"score": ')

    assert score == 0.0
    assert called is False
    assert side_info["format"] == "JSON"
    assert side_info["format_valid"] is False
    assert "Malformed JSON" in side_info["Error"]
    assert "JSON validation failed" in side_info["Feedback"]
    assert side_info["Output"] == '{"score": '


def test_require_json_output_schema_failure_reports_missing_key():
    @require_json_output(schema={"score": int, "reasoning": str})
    def evaluator(data, response):
        return 1.0, {"Output": response}

    score, side_info = evaluator({}, '{"score": 7}')

    assert score == 0.0
    assert "missing required key 'reasoning'" in side_info["Error"]


def test_require_json_output_supports_named_output_argument():
    @require_json_output(output_arg="response", schema={"answer": str})
    def evaluator(data, response):
        return 1.0 if response["answer"] == data["expected"] else 0.0

    assert evaluator({"expected": "Paris"}, response='{"answer": "Paris"}') == 1.0


def test_require_json_output_failure_factory_supports_custom_return_types():
    @require_json_output(
        schema={"answer": str},
        failure_factory=lambda score, side_info: EvaluationResult(
            score=score,
            feedback=side_info["Feedback"],
            objective_scores=side_info.get("scores"),
        ),
    )
    def evaluator(data, response):
        return EvaluationResult(score=1.0, feedback="ok")

    result = evaluator({}, "{bad json")

    assert result.score == 0.0
    assert "JSON validation failed" in result.feedback


def test_require_json_output_track_format_objective_on_success_and_failure():
    @require_json_output(schema={"answer": str}, track_format=True)
    def evaluator(data, response):
        return 0.8, {"scores": {"quality": 0.8}}

    success_score, success_info = evaluator({}, '{"answer": "A"}')
    failure_score, failure_info = evaluator({}, '{"missing": "A"}')

    assert success_score == 0.8
    assert success_info["scores"] == {"quality": 0.8, "json_valid": 1.0}
    assert success_info["format_valid"] is True
    assert failure_score == 0.0
    assert failure_info["scores"] == {"json_valid": 0.0}


def test_require_regex_match_failure_and_match_passing():
    @require_regex_match(pattern=r"ID-(\d+)", pass_match=True)
    def evaluator(response: re.Match[str]):
        return int(response.group(1)), {"matched": response.group(0)}

    assert evaluator("ID-42") == (42, {"matched": "ID-42"})

    score, side_info = evaluator("bad")
    assert score == 0.0
    assert "Regex fullmatch failed" in side_info["Error"]


def test_require_xml_output_parses_and_checks_root_tag():
    @require_xml_output(root_tag="answer")
    def evaluator(response):
        return 1.0, {"value": response.text}

    assert evaluator("<answer>yes</answer>") == (1.0, {"value": "yes"})

    score, side_info = evaluator("<wrong>yes</wrong>")
    assert score == 0.0
    assert "Expected XML root tag 'answer'" in side_info["Error"]


def test_require_format_supports_custom_validator():
    def validate_uppercase(raw_output):
        if raw_output != raw_output.upper():
            raise FormatValidationError("Output must be uppercase.")
        return raw_output.lower()

    @require_format(validate_uppercase, format_name="uppercase")
    def evaluator(response):
        return 1.0, {"normalized": response}

    assert evaluator("YES") == (1.0, {"normalized": "yes"})

    score, side_info = evaluator("Yes")
    assert score == 0.0
    assert side_info["format"] == "uppercase"
    assert side_info["Error"] == "Output must be uppercase."


def test_require_format_supports_boolean_predicate_validators():
    called = False

    @require_format(lambda raw_output: raw_output.startswith("ok:"), format_name="prefix")
    def evaluator(response):
        nonlocal called
        called = True
        return 1.0, {"Output": response}

    assert evaluator("ok:value") == (1.0, {"Output": "ok:value"})

    called = False
    score, side_info = evaluator("bad:value")
    assert score == 0.0
    assert side_info["format"] == "prefix"
    assert side_info["Error"] == "Format validator returned False."
    assert side_info["Output"] == "bad:value"
    assert called is False


def test_require_format_validator_exception_returns_failure_feedback():
    called = False

    def validator(raw_output):
        raise ValueError(f"cannot validate {raw_output}")

    @require_format(validator, format_name="custom")
    def evaluator(response):
        nonlocal called
        called = True
        return 1.0, {"Output": response}

    score, side_info = evaluator("bad")

    assert score == 0.0
    assert called is False
    assert side_info["format"] == "custom"
    assert side_info["Error"] == "cannot validate bad"
    assert "custom validation failed: cannot validate bad" in side_info["Feedback"]


def test_schema_list_validates_every_item():
    @require_json_output(schema={"answers": [str]})
    def evaluator(response):
        return 1.0, {"Output": response}

    assert evaluator('{"answers": ["a", "b"]}')[0] == 1.0

    score, side_info = evaluator('{"answers": ["a", 2]}')
    assert score == 0.0
    assert "$.answers[1] expected str" in side_info["Error"]


def test_schema_predicate_can_validate_values():
    @require_json_output(schema={"score": lambda value: 0 <= value <= 1})
    def evaluator(response):
        return response["score"], {"Output": response}

    assert evaluator('{"score": 0.5}')[0] == 0.5

    score, side_info = evaluator('{"score": 2}')
    assert score == 0.0
    assert "$.score failed custom schema predicate" in side_info["Error"]


def test_schema_int_does_not_accept_json_bool():
    @require_json_output(schema={"score": int})
    def evaluator(response):
        return response["score"], {"Output": response}

    score, side_info = evaluator('{"score": true}')

    assert score == 0.0
    assert "$.score expected int, got bool" in side_info["Error"]


def test_invalid_output_index_raises_clear_type_error():
    @require_json_output(output_index=1)
    def evaluator(response):
        return 1.0, {"Output": response}

    with pytest.raises(TypeError, match="output_index 1 is out of range"):
        evaluator('{"answer": "ok"}')
