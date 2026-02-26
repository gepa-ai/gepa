"""End-to-end test: run optimize_anything on a real problem with a mock judge.

Run: uv run python examples/frontiercs/test_e2e.py [--metric-calls N]

The mock judge scores C++ code based on structural heuristics:
  - Has #include → compiles
  - Has int main → runs
  - Has cin/scanf → reads input (likely correct)
  - Has cout/printf → produces output
  - Has algorithm/sort/dp patterns → bonus
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.frontiercs import main
from examples.frontiercs.utils.judge import FrontierCSJudgeClient

# ---------------------------------------------------------------------------
# A real competitive programming problem (A+B style, classic)
# ---------------------------------------------------------------------------

PROBLEM = {
    "problem_id": "0",
    "statement": """## A+B Problem

### Description
Given two integers A and B, compute their sum.

### Input
The first line contains two space-separated integers A and B (1 ≤ A, B ≤ 10^9).

### Output
Print a single integer — the sum of A and B.

### Sample Input
```
3 5
```

### Sample Output
```
8
```

### Constraints
- Time limit: 1000 ms
- Memory limit: 256 MB
""",
    "tag": "math",
    "time_limit": "1000",
    "memory_limit": "262144",
    "sample_input": "3 5",
    "sample_output": "8",
}


# ---------------------------------------------------------------------------
# Mock judge: scores C++ code by structural heuristics
# ---------------------------------------------------------------------------


def _score_cpp_code(code: str) -> dict[str, Any]:
    """Heuristic scorer for C++ code — gives GEPA a gradient to optimize."""
    score = 0
    n_cases = 10
    cases: list[dict[str, str]] = []

    has_include = bool(re.search(r"#include", code))
    has_main = bool(re.search(r"int\s+main", code))
    has_input = bool(re.search(r"(cin|scanf|getline)", code))
    has_output = bool(re.search(r"(cout|printf|puts)", code))
    has_algorithm = bool(re.search(r"(algorithm|sort|vector|map|set|queue|stack|dp|memo)", code))

    if not has_include or not has_main:
        return {
            "status": "error",
            "error": "COMPILATION_ERROR",
            "score": 0,
            "result": "CE",
            "cases": [],
            "passed": False,
        }

    # Base: compiles. Give some AC cases based on code quality.
    ac_count = 0
    if has_input and has_output:
        ac_count = 8  # Reads input + produces output → mostly correct
    elif has_output:
        ac_count = 3  # Hardcoded output → some cases pass
    elif has_input:
        ac_count = 1  # Reads but doesn't output properly

    if has_algorithm:
        ac_count = min(ac_count + 1, n_cases)

    # Build per-case results
    for i in range(n_cases):
        if i < ac_count:
            cases.append({"status": "AC"})
        elif i < ac_count + 1:
            cases.append({"status": "TLE"})
        else:
            cases.append({"status": "WA"})

    score = int(ac_count / n_cases * 100)
    passed = ac_count == n_cases

    return {
        "status": "done",
        "passed": passed,
        "score": score,
        "result": "AC" if passed else "WA",
        "cases": cases,
    }


def _make_mock_judge() -> FrontierCSJudgeClient:
    """Create a mock judge that scores C++ code by heuristics."""
    judge = FrontierCSJudgeClient("http://mock:8081")
    submitted_code: dict[str, str] = {}
    sid_counter = [0]

    def mock_submit(pid: str, code: str) -> str:
        sid_counter[0] += 1
        sid = f"mock-{sid_counter[0]}"
        submitted_code[sid] = code
        return sid

    def mock_get_result(sid: str, **kwargs: Any) -> dict[str, Any]:
        code = submitted_code.get(sid, "")
        return _score_cpp_code(code)

    judge.submit_solution = mock_submit  # type: ignore[assignment]
    judge.get_result = mock_get_result  # type: ignore[assignment]
    return judge


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(max_metric_calls: int = 50, reflection_lm: str = "openai/gpt-5") -> None:
    from examples.frontiercs.utils.background import build_background
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    main.JUDGE = _make_mock_judge()
    main.VERBOSE = True

    print(f"Running e2e test: problem={PROBLEM['problem_id']}, metric_calls={max_metric_calls}")
    print(f"Reflection LM: {reflection_lm}")
    print("-" * 60)

    result = optimize_anything(
        seed_candidate=None,
        evaluator=main.evaluate,
        dataset=[PROBLEM],
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir="outputs/frontiercs_e2e_test/",
                seed=42,
                max_metric_calls=max_metric_calls,
                frontier_type="objective",
                cache_evaluation=True,
                track_best_outputs=True,
            ),
            reflection=ReflectionConfig(reflection_lm=reflection_lm),
        ),
        objective="Generate correct, efficient C++ solutions for competitive programming problems. "
        "Maximize accepted test cases, avoid TLE and runtime errors.",
        background=build_background(PROBLEM),
    )

    best_score = result.val_aggregate_scores[result.best_idx] * 100.0
    best_candidate = result.best_candidate or ""

    print("-" * 60)
    print(f"Best score: {best_score:.2f}/100")
    print(f"Best candidate (first 500 chars):\n{best_candidate[:500]}")
    print("-" * 60)

    return best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E test for FrontierCS with mock judge")
    parser.add_argument("--metric-calls", type=int, default=50, help="Number of metric calls")
    parser.add_argument("--reflection-lm", type=str, default="openai/gpt-5", help="Reflection LM")
    args = parser.parse_args()

    run(max_metric_calls=args.metric_calls, reflection_lm=args.reflection_lm)
