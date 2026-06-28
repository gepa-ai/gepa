#!/usr/bin/env python3
"""Pre-flight checks for a GEPA / optimize_anything run — fail fast before a long job.

    python preflight.py                      # checks gepa + reflection-LM creds
    python preflight.py --engine autoresearch  # also checks the `claude` CLI
    GEPA_REFLECTION_LM=anthropic/claude-sonnet-4-6 python preflight.py --test-lm

Exit code 0 = all good; non-zero = at least one blocker.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

OK, BAD = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m"
problems: list[str] = []


def check(label: str, ok: bool, fix: str = "") -> None:
    print(f"  [{OK if ok else BAD}] {label}")
    if not ok:
        problems.append(f"{label} — {fix}" if fix else label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="gepa",
                    choices=["gepa", "best_of_n", "autoresearch", "meta_harness"])
    ap.add_argument("--test-lm", action="store_true",
                    help="make a 1-call round-trip to the reflection LM (costs a few tokens)")
    a = ap.parse_args()

    print("== GEPA preflight ==")

    # 1) import + the correct API surface
    try:
        import gepa  # noqa
        from gepa.optimize_anything import OptimizeAnythingConfig, optimize_anything  # noqa
        check(f"import gepa ({getattr(gepa, '__version__', '?')}) + optimize_anything", True)
    except Exception as e:  # noqa
        check("import gepa + optimize_anything", False, "pip install gepa")
        print(f"      {e}")
        return _report()

    # 2) reflection-LM credentials (engines that use an LLM)
    lm = os.environ.get("GEPA_REFLECTION_LM", "")
    if a.engine in ("gepa", "best_of_n"):
        is_bedrock = lm.startswith("bedrock/") or "bedrock" in lm
        has_aws = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or
                       os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"))
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if is_bedrock:
            check("Bedrock creds present (AWS_BEARER_TOKEN_BEDROCK / AWS_ACCESS_KEY_ID / AWS_PROFILE)",
                  has_aws, "export AWS creds for the Bedrock reflection LM")
        else:
            check("LLM creds present (ANTHROPIC_API_KEY or provider key)", has_anthropic or has_aws,
                  "export ANTHROPIC_API_KEY (or your LiteLLM provider's key)")
        check("GEPA_REFLECTION_LM is set", bool(lm),
              "export GEPA_REFLECTION_LM=<litellm id or Bedrock ARN>")

    # 3) agentic engines need the claude CLI
    if a.engine in ("autoresearch", "meta_harness"):
        cli = shutil.which("claude")
        check(f"`claude` CLI on PATH (required by {a.engine})", bool(cli),
              "install + authenticate the Claude Code CLI headless")
        if cli:
            print(f"      claude -> {cli}")

    # 4) optional live LM round-trip
    if a.test_lm and lm and a.engine in ("gepa", "best_of_n"):
        try:
            from gepa.lm import LM
            out = LM(lm)("Reply with the single word: ok")
            check("reflection LM 1-call round-trip", bool(out),
                  "LM returned empty; check model id / creds / region")
        except Exception as e:  # noqa
            check("reflection LM 1-call round-trip", False, str(e)[:160])

    return _report()


def _report() -> int:
    print()
    if problems:
        print(f"\033[31m{len(problems)} blocker(s):\033[0m")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\033[32mAll preflight checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
