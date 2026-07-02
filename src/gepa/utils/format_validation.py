# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Evaluator wrappers for treating structured-output format errors as hard failures."""

from __future__ import annotations

import functools
import inspect
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from typing import Any

FailureFactory = Callable[[float, dict[str, Any]], Any]


class FormatValidationError(ValueError):
    """Raised when an output does not satisfy a required format."""


def require_format(
    validator_fn: Callable[[Any], Any],
    *,
    output_arg: str | None = None,
    output_index: int = -1,
    failure_score: float = 0.0,
    failure_factory: FailureFactory | None = None,
    format_name: str = "format",
    include_output: bool = True,
    output_key: str = "Output",
    error_key: str = "Error",
    feedback_key: str = "Feedback",
    track_format: bool = False,
    format_objective_name: str = "format_valid",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap an evaluator so invalid structured output returns a hard-failure score.

    ``validator_fn`` receives the raw output argument and should return the value
    to pass into the wrapped evaluator. If it raises an exception, the evaluator
    is not called and the wrapper returns ``(failure_score, side_info)`` by
    default.

    Args:
        validator_fn: Function that validates and optionally parses the output.
        output_arg: Named argument containing the output. If omitted, the wrapper
            validates the positional argument at ``output_index``.
        output_index: Positional argument index to validate when ``output_arg``
            is not set. Defaults to the last positional argument.
        failure_score: Score returned when validation fails.
        failure_factory: Optional factory for custom evaluator return types. It
            receives ``(failure_score, side_info)``.
        format_name: Human-readable format label used in feedback.
        include_output: Include the raw output in failure ``side_info``.
        output_key: Key used for the raw output in failure ``side_info``.
        error_key: Key used for the validation error in failure ``side_info``.
        feedback_key: Key used for reflection-facing feedback in failure
            ``side_info``.
        track_format: Add a ``scores[format_objective_name]`` objective with
            ``0.0`` on failure and ``1.0`` on success when the wrapped evaluator
            returns score/side_info.
        format_objective_name: Objective name used when ``track_format=True``.
    """

    def decorator(evaluator_fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(evaluator_fn) if output_arg is not None else None

        @functools.wraps(evaluator_fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                _, replaced_args, replaced_kwargs = _replace_output_argument(
                    args,
                    kwargs,
                    validator_fn,
                    output_arg=output_arg,
                    output_index=output_index,
                    sig=sig,
                )
            except _ValidationFailureError as failure:
                side_info = _failure_side_info(
                    raw_output=failure.raw_output,
                    message=failure.message,
                    format_name=format_name,
                    include_output=include_output,
                    output_key=output_key,
                    error_key=error_key,
                    feedback_key=feedback_key,
                    track_format=track_format,
                    format_objective_name=format_objective_name,
                )
                if failure_factory is not None:
                    return failure_factory(failure_score, side_info)
                return failure_score, side_info

            result = evaluator_fn(*replaced_args, **replaced_kwargs)
            return _attach_success_format_score(result, track_format, format_objective_name)

        return wrapped

    return decorator


def require_json_output(
    evaluator_fn: Callable[..., Any] | None = None,
    *,
    schema: Any | None = None,
    output_arg: str | None = None,
    output_index: int = -1,
    failure_score: float = 0.0,
    failure_factory: FailureFactory | None = None,
    track_format: bool = False,
    format_objective_name: str = "json_valid",
) -> Callable[..., Any]:
    """Require an evaluator output argument to be valid JSON.

    The wrapped evaluator receives the parsed JSON value. ``schema`` is an
    optional lightweight Python schema: mappings require keys, types require
    ``isinstance`` matches, and single-item lists validate every list item.
    """

    def validator(raw_output: Any) -> Any:
        if isinstance(raw_output, (str, bytes, bytearray)):
            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError as exc:
                raise FormatValidationError(f"Malformed JSON: {exc}") from exc
        else:
            parsed = raw_output

        _validate_python_schema(parsed, schema)
        return parsed

    decorator = require_format(
        validator,
        output_arg=output_arg,
        output_index=output_index,
        failure_score=failure_score,
        failure_factory=failure_factory,
        format_name="JSON",
        track_format=track_format,
        format_objective_name=format_objective_name,
    )
    if evaluator_fn is not None:
        return decorator(evaluator_fn)
    return decorator


def require_xml_output(
    evaluator_fn: Callable[..., Any] | None = None,
    *,
    root_tag: str | None = None,
    output_arg: str | None = None,
    output_index: int = -1,
    failure_score: float = 0.0,
    failure_factory: FailureFactory | None = None,
    track_format: bool = False,
    format_objective_name: str = "xml_valid",
) -> Callable[..., Any]:
    """Require an evaluator output argument to be well-formed XML.

    The wrapped evaluator receives the parsed ``xml.etree.ElementTree.Element``.
    """

    def validator(raw_output: Any) -> ET.Element:
        if not isinstance(raw_output, str):
            raise FormatValidationError(f"Expected XML output as str, got {type(raw_output).__name__}.")
        try:
            root = ET.fromstring(raw_output)
        except ET.ParseError as exc:
            raise FormatValidationError(f"Malformed XML: {exc}") from exc
        if root_tag is not None and root.tag != root_tag:
            raise FormatValidationError(f"Expected XML root tag '{root_tag}', got '{root.tag}'.")
        return root

    decorator = require_format(
        validator,
        output_arg=output_arg,
        output_index=output_index,
        failure_score=failure_score,
        failure_factory=failure_factory,
        format_name="XML",
        track_format=track_format,
        format_objective_name=format_objective_name,
    )
    if evaluator_fn is not None:
        return decorator(evaluator_fn)
    return decorator


def require_regex_match(
    evaluator_fn: Callable[..., Any] | None = None,
    *,
    pattern: str | re.Pattern[str],
    flags: int = 0,
    fullmatch: bool = True,
    pass_match: bool = False,
    output_arg: str | None = None,
    output_index: int = -1,
    failure_score: float = 0.0,
    failure_factory: FailureFactory | None = None,
    track_format: bool = False,
    format_objective_name: str = "regex_valid",
) -> Callable[..., Any]:
    """Require an evaluator output argument to match a regular expression."""

    compiled = re.compile(pattern, flags) if isinstance(pattern, str) else pattern

    def validator(raw_output: Any) -> Any:
        if not isinstance(raw_output, str):
            raise FormatValidationError(f"Expected regex input as str, got {type(raw_output).__name__}.")
        match = compiled.fullmatch(raw_output) if fullmatch else compiled.search(raw_output)
        if match is None:
            match_type = "fullmatch" if fullmatch else "match"
            raise FormatValidationError(f"Regex {match_type} failed for pattern {compiled.pattern!r}.")
        return match if pass_match else raw_output

    decorator = require_format(
        validator,
        output_arg=output_arg,
        output_index=output_index,
        failure_score=failure_score,
        failure_factory=failure_factory,
        format_name="regex",
        track_format=track_format,
        format_objective_name=format_objective_name,
    )
    if evaluator_fn is not None:
        return decorator(evaluator_fn)
    return decorator


class _ValidationFailureError(Exception):
    def __init__(self, raw_output: Any, message: str) -> None:
        self.raw_output = raw_output
        self.message = message
        super().__init__(message)


def _replace_output_argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    validator_fn: Callable[[Any], Any],
    *,
    output_arg: str | None,
    output_index: int,
    sig: inspect.Signature | None,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    if output_arg is not None:
        if output_arg in kwargs:
            raw_output = kwargs[output_arg]
            parsed = _validate_or_fail(validator_fn, raw_output)
            replaced_kwargs = {**kwargs, output_arg: parsed}
            return raw_output, args, replaced_kwargs
        if sig is None:
            raise TypeError("Internal error: output_arg requires an inspected signature.")
        bound = sig.bind_partial(*args, **kwargs)
        if output_arg not in bound.arguments:
            raise TypeError(f"Missing output argument '{output_arg}' for format validation.")
        raw_output = bound.arguments[output_arg]
        parsed = _validate_or_fail(validator_fn, raw_output)
        bound.arguments[output_arg] = parsed
        return raw_output, bound.args, bound.kwargs

    if not args:
        raise TypeError("Format validation requires a positional output argument or output_arg.")
    normalized_index = output_index if output_index >= 0 else len(args) + output_index
    if normalized_index < 0 or normalized_index >= len(args):
        raise TypeError(f"output_index {output_index} is out of range for {len(args)} positional arguments.")

    raw_output = args[normalized_index]
    parsed = _validate_or_fail(validator_fn, raw_output)
    replaced_args = list(args)
    replaced_args[normalized_index] = parsed
    return raw_output, tuple(replaced_args), kwargs


def _validate_or_fail(validator_fn: Callable[[Any], Any], raw_output: Any) -> Any:
    try:
        return validator_fn(raw_output)
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        raise _ValidationFailureError(raw_output, message) from exc


def _failure_side_info(
    *,
    raw_output: Any,
    message: str,
    format_name: str,
    include_output: bool,
    output_key: str,
    error_key: str,
    feedback_key: str,
    track_format: bool,
    format_objective_name: str,
) -> dict[str, Any]:
    feedback = f"{format_name} validation failed: {message}"
    side_info: dict[str, Any] = {
        feedback_key: feedback,
        error_key: message,
        "format": format_name,
        "format_valid": False,
    }
    if include_output:
        side_info[output_key] = raw_output
    if track_format:
        side_info["scores"] = {format_objective_name: 0.0}
    return side_info


def _attach_success_format_score(result: Any, track_format: bool, format_objective_name: str) -> Any:
    if not track_format:
        return result

    success_info: dict[str, Any] = {"format_valid": True, "scores": {format_objective_name: 1.0}}
    if isinstance(result, tuple) and len(result) == 2:
        score, side_info = result
        merged = dict(side_info) if side_info is not None else {}
        existing_scores = merged.get("scores")
        scores = dict(existing_scores) if isinstance(existing_scores, Mapping) else {}
        scores[format_objective_name] = 1.0
        merged["scores"] = scores
        merged.setdefault("format_valid", True)
        return score, merged
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return result, success_info
    return result


def _validate_python_schema(value: Any, schema: Any, path: str = "$") -> None:
    if schema is None:
        return
    if isinstance(schema, type):
        if not _matches_type(value, schema):
            raise FormatValidationError(
                f"{path} expected {schema.__name__}, got {type(value).__name__}."
            )
        return
    if _is_type_tuple(schema):
        if not any(_matches_type(value, expected_type) for expected_type in schema):
            expected = " | ".join(t.__name__ for t in schema)
            raise FormatValidationError(f"{path} expected {expected}, got {type(value).__name__}.")
        return
    if isinstance(schema, Mapping):
        if not isinstance(value, Mapping):
            raise FormatValidationError(f"{path} expected object, got {type(value).__name__}.")
        for key, expected_schema in schema.items():
            if key not in value:
                raise FormatValidationError(f"{path} missing required key {key!r}.")
            _validate_python_schema(value[key], expected_schema, f"{path}.{key}")
        return
    if isinstance(schema, list):
        if len(schema) != 1:
            raise FormatValidationError(f"{path} schema list must contain exactly one item schema.")
        if not isinstance(value, list):
            raise FormatValidationError(f"{path} expected list, got {type(value).__name__}.")
        for idx, item in enumerate(value):
            _validate_python_schema(item, schema[0], f"{path}[{idx}]")
        return
    if callable(schema):
        if not schema(value):
            raise FormatValidationError(f"{path} failed custom schema predicate.")
        return
    if value != schema:
        raise FormatValidationError(f"{path} expected {schema!r}, got {value!r}.")


def _is_type_tuple(schema: Any) -> bool:
    return isinstance(schema, tuple) and all(isinstance(item, type) for item in schema)


def _matches_type(value: Any, expected_type: type) -> bool:
    if expected_type is int and isinstance(value, bool):
        return False
    return isinstance(value, expected_type)
