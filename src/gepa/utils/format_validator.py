# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Format-validation decorators for GEPA evaluators."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

EvaluatorResult = tuple[float, dict[str, Any]]
EvaluatorFn = Callable[..., EvaluatorResult]
FormatValidatorFn = Callable[[Any], bool]
SchemaType = type[Any] | tuple[type[Any], ...]


def require_json_output(schema: Mapping[str, SchemaType] | None = None) -> Callable[[EvaluatorFn], EvaluatorFn]:
    """
    Require the response to be parseable JSON before running the evaluator.

    When validation succeeds, the wrapped evaluator receives the parsed JSON
    value in place of the raw response string. When parsing or schema
    validation fails, the evaluator is not called and the failure reason is
    returned as feedback.
    """

    def decorator(evaluator: EvaluatorFn) -> EvaluatorFn:
        @wraps(evaluator)
        def wrapper(data: Any, response: Any, *args: Any, **kwargs: Any) -> EvaluatorResult:
            try:
                parsed_response = json.loads(response)
            except (json.JSONDecodeError, TypeError) as error:
                return _failure("Malformed JSON", str(error))

            if schema is not None:
                schema_error = _validate_schema(parsed_response, schema)
                if schema_error is not None:
                    return _failure("Schema mismatch", schema_error)

            return evaluator(data, parsed_response, *args, **kwargs)

        return wrapper

    return decorator


def require_format(validator_fn: FormatValidatorFn) -> Callable[[EvaluatorFn], EvaluatorFn]:
    """
    Require ``validator_fn(response)`` to pass before running the evaluator.

    The validator should return ``True`` for valid responses. Returning a falsey
    value or raising an exception short-circuits the evaluator with a zero score
    and feedback that includes the validation failure.
    """

    return _require_format(
        validator_fn,
        failure_prefix="Format validation failed",
        false_detail="validator returned False",
    )


def require_regex_match(pattern: str | re.Pattern[str]) -> Callable[[EvaluatorFn], EvaluatorFn]:
    """Require the response string to fully match ``pattern`` before evaluation."""

    compiled_pattern = re.compile(pattern) if isinstance(pattern, str) else pattern

    def validator_fn(response: Any) -> bool:
        return isinstance(response, str) and compiled_pattern.fullmatch(response) is not None

    return _require_format(
        validator_fn,
        failure_prefix="Regex match failed",
        false_detail=f"response did not fully match pattern {compiled_pattern.pattern}",
    )


def _require_format(
    validator_fn: FormatValidatorFn,
    failure_prefix: str,
    false_detail: str,
) -> Callable[[EvaluatorFn], EvaluatorFn]:
    def decorator(evaluator: EvaluatorFn) -> EvaluatorFn:
        @wraps(evaluator)
        def wrapper(data: Any, response: Any, *args: Any, **kwargs: Any) -> EvaluatorResult:
            try:
                is_valid = validator_fn(response)
            except Exception as error:
                return _failure(failure_prefix, f"{type(error).__name__}: {error}")

            if not is_valid:
                return _failure(failure_prefix, false_detail)

            return evaluator(data, response, *args, **kwargs)

        return wrapper

    return decorator


def _validate_schema(parsed_response: Any, schema: Mapping[str, SchemaType]) -> str | None:
    if not isinstance(parsed_response, dict):
        return "expected JSON object for schema validation"

    for key, expected_type in schema.items():
        if key not in parsed_response:
            return f"missing required key {key!r}"

        try:
            if not isinstance(parsed_response[key], expected_type):
                return f"key {key!r} expected {_type_name(expected_type)}, got {type(parsed_response[key]).__name__}"
        except TypeError as error:
            return f"invalid schema for key {key!r}: {error}"

    return None


def _type_name(expected_type: SchemaType) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(type_.__name__ for type_ in expected_type)
    return expected_type.__name__


def _failure(prefix: str, detail: str) -> EvaluatorResult:
    message = prefix if not detail else f"{prefix}: {detail}"
    return 0.0, {"feedback": message}


__all__ = [
    "require_format",
    "require_json_output",
    "require_regex_match",
]
