"""CLI for RLM — called by Claude Code via Bash.

Usage:
    python -m rlm.cli "Your prompt here"
    python -m rlm.cli --model haiku "Summarize these evals"
    python -m rlm.cli --parallel "Prompt 1" "Prompt 2" --model haiku
    python -m rlm.cli -o /tmp/result.txt --model haiku "Read these files"

Note: When called from inside Claude Code, stdout may be swallowed.
Always use -o/--output to write results to a file, then read the file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from rlm.core import RLMConfig, spawn, spawn_parallel


def _write_output(text: str, output_file: str | None):
    """Write to file if specified, otherwise stdout."""
    if output_file:
        with open(output_file, "w") as f:
            f.write(text)
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(description="RLM: spawn sub-agents via claude -p")
    parser.add_argument("prompts", nargs="+", help="Prompt(s) for sub-agent(s)")
    parser.add_argument("--model", "-m", default=None, help="Model (haiku/sonnet/opus)")
    parser.add_argument("--models", nargs="+", help="Models for parallel mode (one per prompt)")
    parser.add_argument("--parallel", action="store_true", help="Run prompts in parallel")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-budget", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--add-dir", action="append", default=[])
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-o", "--output", default=None, help="Write output to file (recommended inside CC)")
    args = parser.parse_args()

    config = RLMConfig(
        max_depth=args.max_depth,
        max_budget_usd=args.max_budget,
        timeout=args.timeout,
        add_dirs=args.add_dir,
        default_model=args.model or "sonnet",
    )

    budget_file = os.environ.get("RLM_BUDGET_FILE")
    if budget_file:
        config.budget_file = budget_file

    if args.parallel and len(args.prompts) > 1:
        models = args.models or [args.model] * len(args.prompts)
        pairs = list(zip(args.prompts, models))
        results = spawn_parallel(pairs, config)

        if args.json:
            _write_output(json.dumps([{"text": r.text, "model": r.model, "cost": r.cost_usd, "error": r.error} for r in results], indent=2), args.output)
        else:
            lines = []
            for i, r in enumerate(results):
                if len(results) > 1:
                    lines.append(f"=== Result {i + 1} (model={r.model}, cost=${r.cost_usd:.4f}) ===")
                if r.error:
                    lines.append(f"ERROR: {r.error}")
                else:
                    lines.append(r.text)
            _write_output("\n".join(lines), args.output)
    else:
        result = spawn(args.prompts[0], args.model, config)

        if args.json:
            _write_output(json.dumps({"text": result.text, "model": result.model, "cost": result.cost_usd, "error": result.error}, indent=2), args.output)
        elif result.error:
            print(f"ERROR: {result.error}", file=sys.stderr)
            sys.exit(1)
        else:
            _write_output(result.text, args.output)


if __name__ == "__main__":
    main()
