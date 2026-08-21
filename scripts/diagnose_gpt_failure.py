#!/usr/bin/env python3
"""Reproduce and diagnose the GPT reflection-model request used by Glean GEPA.

Run from the repository root:

    uv run --extra glean python scripts/diagnose_gpt_failure.py

The script never prints the API key. It exits non-zero when configuration or the
request fails, making it suitable for local debugging and CI smoke checks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Any

import openai

from glean_gepa.openai_client import create_openai_client


def _safe(value: Any) -> Any:
    """Convert SDK values to JSON-serializable diagnostic output."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_safe(item) for item in value]
    return str(value)


def _exception_details(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) if response is not None else {}
    details: dict[str, Any] = {
        "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
        "message": str(exc),
        "status_code": getattr(exc, "status_code", None),
        "request_id": getattr(exc, "request_id", None),
        "error_body": getattr(exc, "body", None),
        "cause_chain": _cause_chain(exc),
    }
    if headers:
        details["response_headers"] = {
            key: headers.get(key)
            for key in ("date", "openai-processing-ms", "x-request-id")
            if headers.get(key) is not None
        }
    return _safe(details)


def _cause_chain(exc: BaseException) -> list[dict[str, str]]:
    """Return nested exception types/messages without request or credential data."""
    causes: list[dict[str, str]] = []
    seen: set[int] = set()
    current = exc.__cause__ or exc.__context__
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        causes.append(
            {
                "type": f"{type(current).__module__}.{type(current).__name__}",
                "message": str(current),
            }
        )
        current = current.__cause__ or current.__context__
    return causes


def _print_json(label: str, value: Any) -> None:
    print(f"{label}: {json.dumps(_safe(value), indent=2, sort_keys=True)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.1", help="Model to test (default: gpt-5.1)")
    parser.add_argument("--prompt", default="Reply with exactly: OK", help="Short prompt sent to the model")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maximum completion tokens")
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Omit reasoning_effort to distinguish a parameter error from a general API failure",
    )
    args = parser.parse_args()

    print("GPT diagnostic")
    _print_json(
        "environment",
        {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "openai_sdk": openai.__version__,
            "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
            "base_url_configured": bool(os.environ.get("OPENAI_BASE_URL")),
        },
    )

    if not os.environ.get("OPENAI_API_KEY"):
        print("configuration_error: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    request: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_completion_tokens": args.max_tokens,
    }
    if not args.minimal:
        # Match the request in glean_gepa.runner._make_reflection_lm.
        request["reasoning_effort"] = "none"

    _print_json("request", request)
    try:
        completion = create_openai_client().chat.completions.create(**request)
    except Exception as exc:
        _print_json("failure", _exception_details(exc))
        if not args.minimal:
            print(
                "next_step: rerun with --minimal; if that succeeds, reasoning_effort='none' is the incompatible parameter"
            )
        return 1

    choice = completion.choices[0] if completion.choices else None
    _print_json(
        "success",
        {
            "id": completion.id,
            "model": completion.model,
            "finish_reason": choice.finish_reason if choice else None,
            "content": choice.message.content if choice else None,
            "usage": completion.usage.model_dump() if completion.usage else None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
