"""Smoke tests for the FrontierCS evaluator with a mocked judge server.

Run: uv run pytest examples/frontiercs/test_smoke.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add repo root to path so `examples.frontiercs` resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.frontiercs import main  # noqa: E402
from examples.frontiercs.utils.judge import FrontierCSJudgeClient, extract_cpp_code  # noqa: E402


# -- extract_cpp_code ----------------------------------------------------------


def test_extract_cpp_from_fenced_block():
    text = "Here is the solution:\n```cpp\n#include <iostream>\nint main() { return 0; }\n```\nDone."
    assert extract_cpp_code(text) == "#include <iostream>\nint main() { return 0; }"


def test_extract_cpp_plain_text():
    text = "#include <cstdio>\nint main() {}"
    assert extract_cpp_code(text) == text.strip()


# -- Mock judge helper ---------------------------------------------------------


def _mock_judge(submit_result: dict, get_result_value: dict) -> FrontierCSJudgeClient:
    judge = FrontierCSJudgeClient("http://fake:8081")
    judge.submit_solution = MagicMock(return_value=submit_result.get("sid"))
    judge.get_result = MagicMock(return_value=get_result_value)
    return judge


EXAMPLE = {"problem_id": "42", "statement": "Print hello world."}


# -- evaluate: all AC ----------------------------------------------------------


def test_evaluate_all_ac():
    main.JUDGE = _mock_judge(
        submit_result={"sid": "s1"},
        get_result_value={
            "status": "done",
            "passed": True,
            "score": 100,
            "result": "AC",
            "cases": [{"status": "AC"}, {"status": "AC"}, {"status": "AC"}],
        },
    )
    main.VERBOSE = False

    candidate = '#include <iostream>\nint main() { std::cout << "hello"; }'
    score, side_info = main.evaluate(candidate, EXAMPLE)

    assert score == 1.0
    assert side_info["scores"]["correctness"] == 1.0
    assert side_info["scores"]["compilation"] == 1.0
    assert side_info["scores"]["time_efficiency"] == 1.0
    assert side_info["scores"]["stability"] == 1.0
    assert side_info["generated_code"] == candidate.strip()


# -- evaluate: partial (1 AC, 1 WA, 1 TLE) ------------------------------------


def test_evaluate_partial():
    main.JUDGE = _mock_judge(
        submit_result={"sid": "s2"},
        get_result_value={
            "status": "done",
            "passed": False,
            "score": 33,
            "result": "WA",
            "cases": [{"status": "AC"}, {"status": "WA"}, {"status": "TLE"}],
        },
    )
    main.VERBOSE = False

    candidate = "int main() { return 0; }"
    score, side_info = main.evaluate(candidate, EXAMPLE)

    assert score == 0.33
    assert side_info["scores"]["correctness"] == 1.0 / 3.0  # 1 AC out of 3
    assert side_info["scores"]["compilation"] == 1.0
    assert side_info["scores"]["time_efficiency"] == 2.0 / 3.0  # 2 non-TLE out of 3
    assert side_info["scores"]["stability"] == 1.0  # no RE


# -- evaluate: compilation error -----------------------------------------------


def test_evaluate_compilation_error():
    main.JUDGE = _mock_judge(
        submit_result={"sid": "s3"},
        get_result_value={
            "status": "error",
            "error": "COMPILATION_ERROR",
            "score": 0,
            "cases": [],
        },
    )
    main.VERBOSE = False

    candidate = "this is not valid c++"
    score, side_info = main.evaluate(candidate, EXAMPLE)

    assert score == 0.0
    assert side_info["scores"]["compilation"] == 0.0


# -- evaluate: runtime errors --------------------------------------------------


def test_evaluate_runtime_errors():
    main.JUDGE = _mock_judge(
        submit_result={"sid": "s4"},
        get_result_value={
            "status": "done",
            "passed": False,
            "score": 0,
            "result": "RE",
            "cases": [{"status": "RE"}, {"status": "RE"}],
        },
    )
    main.VERBOSE = False

    candidate = "int main() { int *p = nullptr; *p = 1; }"
    score, side_info = main.evaluate(candidate, EXAMPLE)

    assert score == 0.0
    assert side_info["scores"]["stability"] == 0.0  # all RE
    assert side_info["scores"]["compilation"] == 1.0  # compiled fine


# -- evaluate: submission failure (judge down) ---------------------------------


def test_evaluate_judge_down():
    main.JUDGE = _mock_judge(submit_result={"sid": None}, get_result_value={})
    main.VERBOSE = False

    candidate = "int main() { return 0; }"
    score, side_info = main.evaluate(candidate, EXAMPLE)

    assert score == 0.0
    assert "judge" in side_info["feedback"].lower()


# -- evaluate: no code extracted -----------------------------------------------


def test_evaluate_empty_candidate():
    main.JUDGE = _mock_judge(submit_result={"sid": "s5"}, get_result_value={})
    main.VERBOSE = False

    score, side_info = main.evaluate("", EXAMPLE)

    assert score == 0.0
    assert side_info["generated_code"] == ""


# -- scoring gradient: optimized > partial > broken ----------------------------


def test_scoring_gradient():
    """Verify score ordering: all-AC > partial > compilation error."""
    main.VERBOSE = False

    # All AC
    main.JUDGE = _mock_judge(
        submit_result={"sid": "s1"},
        get_result_value={"status": "done", "passed": True, "score": 100, "cases": [{"status": "AC"}] * 5},
    )
    score_perfect, _ = main.evaluate("int main() {}", EXAMPLE)

    # Partial
    main.JUDGE = _mock_judge(
        submit_result={"sid": "s2"},
        get_result_value={
            "status": "done",
            "passed": False,
            "score": 40,
            "cases": [{"status": "AC"}, {"status": "AC"}, {"status": "WA"}, {"status": "TLE"}, {"status": "RE"}],
        },
    )
    score_partial, _ = main.evaluate("int main() {}", EXAMPLE)

    # Compilation error
    main.JUDGE = _mock_judge(
        submit_result={"sid": "s3"},
        get_result_value={"status": "error", "error": "COMPILATION_ERROR", "score": 0, "cases": []},
    )
    score_broken, _ = main.evaluate("int main() {}", EXAMPLE)

    assert score_perfect > score_partial > score_broken
