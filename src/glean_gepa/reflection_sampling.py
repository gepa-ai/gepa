"""Glean-specific reflection-example sampling helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypeVar

ExampleT = TypeVar("ExampleT", bound=Mapping[str, Any])

_STDOUT_MARKER = re.compile(r"\b(?:stdout|std\s+out|standard\s+output)\s*:", re.IGNORECASE)
_NEXT_DIAGNOSTIC_MARKER = re.compile(
    r"\b(?:stderr|std\s+err|standard\s+error|error|exception)\s*:",
    re.IGNORECASE,
)


def _normalize_error(error: str) -> str:
    return " ".join(error.casefold().split())


def strip_stdout_sections(text: str) -> str:
    """Remove labeled stdout sections while retaining surrounding diagnostics."""
    kept_lines: list[str] = []
    skipping_stdout = False

    for line in text.splitlines():
        stdout_match = _STDOUT_MARKER.search(line)
        if stdout_match is not None:
            prefix = line[: stdout_match.start()].rstrip()
            if prefix:
                kept_lines.append(prefix)

            remainder = line[stdout_match.end() :]
            next_marker = _NEXT_DIAGNOSTIC_MARKER.search(remainder)
            if next_marker is not None:
                kept_lines.append(remainder[next_marker.start() :].strip())
                skipping_stdout = False
            else:
                skipping_stdout = True
            continue

        if skipping_stdout:
            next_marker = _NEXT_DIAGNOSTIC_MARKER.search(line)
            if next_marker is None:
                continue
            kept_lines.append(line[next_marker.start() :].strip())
            skipping_stdout = False
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def hamming_distance(left: str, right: str) -> int:
    """Return character Hamming distance, padding the shorter string."""
    left = _normalize_error(left)
    right = _normalize_error(right)
    distance = abs(len(left) - len(right))

    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            distance += 1
    return distance


def is_within_hamming_distance(left: str, right: str, k: int) -> bool:
    """Return whether two variable-length strings are within Hamming distance ``k``."""
    if k < 0:
        raise ValueError("k must be non-negative")
    return hamming_distance(left, right) <= k


def deduplicate_reflective_examples(
    examples: Sequence[ExampleT],
    k: int,
    log: Callable[[str], None] | None = None,
) -> list[ExampleT]:
    """Keep the first example for each error cluster within Hamming distance ``k``.

    Errors are isolated from each example's ``Execution Errors`` field. An
    example with at least one novel error is retained; an example whose errors
    are all near-duplicates is dropped. Examples without execution errors are
    retained because there is no error signal on which to deduplicate them.
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    kept: list[ExampleT] = []
    seen_errors: list[str] = []
    for example in examples:
        raw_errors = example.get("Execution Errors", [])
        if isinstance(raw_errors, str):
            raw_errors = [raw_errors]
        if not isinstance(raw_errors, Sequence):
            raw_errors = []
        errors = [error for error in raw_errors if isinstance(error, str) and error.strip()]
        if not errors:
            kept.append(example)
            continue

        novel_errors: list[str] = []
        first_match: tuple[str, str, int] | None = None
        for error in errors:
            match = next(
                (
                    (seen_error, distance)
                    for seen_error in seen_errors
                    if (distance := hamming_distance(error, seen_error)) <= k
                ),
                None,
            )
            if match is None:
                novel_errors.append(error)
            elif first_match is None:
                first_match = (error, match[0], match[1])

        if not novel_errors:
            if log is not None:
                inputs = example.get("Inputs", {})
                entry_id = inputs.get("entry_id", "<unknown>") if isinstance(inputs, Mapping) else "<unknown>"
                assert first_match is not None
                error, matched_error, distance = first_match
                log(
                    f"Reflection sampling dropped entry_id={entry_id!r}: error={error!r} matched "
                    f"prior_error={matched_error!r} at Hamming distance {distance} <= k={k}."
                )
            continue

        kept.append(example)
        seen_errors.extend(novel_errors)

    return kept
