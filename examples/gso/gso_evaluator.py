"""GSO benchmark evaluator for GEPA's optimize_anything API.

Supports multi-test mode: GSO tests are split into train/val/test sets.
Each "example" in the dataset is one test script. The evaluator runs a single
test per call and returns the speedup for that test.

The evaluator signature follows the standard optimize_anything convention:
``evaluator(candidate, example)`` where ``candidate`` is ``{repo_path: branch}``.
The CodingAdapter checks out the branch before calling the evaluator, so the
repo is already on the right branch when the evaluator runs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from typing import Any

from examples.gso.docker_utils import (
    apply_patch_in_container,
    copy_to_container,
    exec_in_container,
    reinstall_in_container,
)

logger = logging.getLogger(__name__)


def prepare_tests(instance: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse and prepare test examples from a GSO instance.

    Returns a list of dicts, each representing one test:
        {"test_idx": int, "test_script": str}
    """
    tests_raw = instance["tests"]
    tests_list: list[str] = json.loads(tests_raw) if isinstance(tests_raw, str) else list(tests_raw)
    tests_list = [_fix_fstring_syntax(t) for t in tests_list]
    return [{"test_idx": i, "test_script": t} for i, t in enumerate(tests_list)]


def split_tests(
    tests: list[dict[str, Any]],
    train_ratio: float = 0.5,
    val_ratio: float = 0.3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split tests into train/val/test sets."""
    n = len(tests)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    if n_train + n_val >= n:
        return tests, tests[:1], []

    train = tests[:n_train]
    val = tests[n_train : n_train + n_val]
    test = tests[n_train + n_val :]
    return train, val, test


def make_gso_evaluator(
    instance: dict[str, Any],
    container: str,
    repo_path: str,
    test_timeout: int = 300,
) -> Callable[..., tuple[float, dict[str, Any]]]:
    """Create an evaluator for optimize_anything with multi-test support.

    The evaluator signature: ``evaluator(candidate, example)`` where:
    - ``candidate`` is ``{repo_path: branch_name}`` (standard coding mode format)
    - ``example`` is a dict with ``test_idx`` and ``test_script``

    The CodingAdapter checks out the branch in the host repo before calling
    this evaluator. The evaluator then computes the diff and applies it to the
    Docker container.

    Args:
        instance: GSO instance dict from HuggingFace.
        container: Running Docker container ID/name.
        repo_path: Host path to the git repo.
        test_timeout: Timeout per test in seconds.

    Returns:
        Evaluator function compatible with optimize_anything.
    """
    install_commands_raw = instance["install_commands"]
    if isinstance(install_commands_raw, str):
        install_commands = json.loads(install_commands_raw)
    else:
        install_commands = list(install_commands_raw)
    install_commands = [c for c in install_commands if "uv venv" not in c and "source .venv" not in c]

    # Copy all test scripts to container once
    all_tests = prepare_tests(instance)
    _setup_scripts_in_container(container, all_tests)

    # State
    state: dict[str, Any] = {
        "baseline_times": {},  # test_idx -> baseline time
        "eval_count": 0,
        "last_applied_diff_hash": None,  # hash of last applied diff (for dedup within same candidate)
    }

    # Find the base branch name from the instance
    base_branch = "base"

    def evaluator(candidate: dict[str, str], example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        state["eval_count"] += 1
        eval_num = state["eval_count"]
        test_idx = example["test_idx"]

        side_info: dict[str, Any] = {}

        # 1. Compute diff from base to current branch and apply to container
        # The CodingAdapter already checked out the branch in the host repo
        branch = candidate[repo_path]
        diff_result = subprocess.run(
            ["git", "diff", base_branch, branch],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        patch = diff_result.stdout
        diff_hash = hash(patch)

        # Always reset and re-apply (clean state each time)
        exec_in_container(container, "cd /testbed && git reset --hard HEAD", timeout=30)

        if patch.strip():
            patch_result = apply_patch_in_container(container, patch)
            if not patch_result["success"]:
                side_info["Error"] = f"Failed to apply patch:\n{patch_result['stderr']}"
                return 0.0, side_info

            if install_commands:
                install_result = reinstall_in_container(container, install_commands)
                if install_result["returncode"] != 0:
                    side_info["Error"] = (
                        f"Reinstall failed:\n{install_result['stdout'][:1000]}\n{install_result['stderr'][:1000]}"
                    )
                    exec_in_container(container, "cd /testbed && git reset --hard HEAD", timeout=30)
                    return 0.0, side_info

        # 2. Clean previous result file
        exec_in_container(container, f"cd /testbed && rm -f result_{test_idx}.txt", timeout=10)

        # 3. Run the test
        is_baseline = test_idx not in state["baseline_times"]
        flag = "--reference" if is_baseline else "--eqcheck"

        test_result = exec_in_container(
            container,
            f"cd /testbed && source .venv/bin/activate && python /gso_test_{test_idx}.py result_{test_idx}.txt {flag} --file_prefix gso_{test_idx}",
            timeout=test_timeout,
        )

        stdout = str(test_result["stdout"])
        stderr = str(test_result["stderr"])
        side_info["test_stdout"] = stdout[:3000]
        side_info["test_stderr"] = stderr[:3000]
        side_info["test_idx"] = test_idx

        if test_result["returncode"] != 0:
            side_info["Error"] = (
                f"Test {test_idx} failed (exit {test_result['returncode']}):\n"
                f"stdout: {stdout[:2000]}\nstderr: {stderr[:2000]}"
            )
            return 0.0, side_info

        # 4. Read timing
        cat_result = exec_in_container(container, f"cat /testbed/result_{test_idx}.txt", timeout=10)
        timing_output = str(cat_result["stdout"]).strip()
        side_info["timing_raw"] = timing_output

        current_time = _extract_time(timing_output)
        side_info["execution_time"] = current_time

        if current_time <= 0:
            side_info["Error"] = f"Could not extract timing from test {test_idx}: {timing_output[:300]}"
            return 0.0, side_info

        # 5. Record baseline
        if is_baseline:
            state["baseline_times"][test_idx] = current_time
            logger.info(f"Test {test_idx} baseline: {current_time:.4f}s")

        baseline = state["baseline_times"][test_idx]
        side_info["baseline_time"] = baseline

        # 6. Compute speedup
        speedup = baseline / current_time
        side_info["speedup"] = speedup
        side_info["Feedback"] = (
            f"Test {test_idx}: execution time {current_time:.4f}s (baseline: {baseline:.4f}s), "
            f"speedup: {speedup:.2f}x"
        )

        logger.info(f"Eval #{eval_num} test {test_idx}: time={current_time:.4f}s, speedup={speedup:.2f}x")
        return speedup, side_info

    return evaluator


def _setup_scripts_in_container(container: str, tests: list[dict[str, Any]]) -> None:
    """Copy test scripts to the container."""
    with tempfile.TemporaryDirectory() as tmp:
        for test in tests:
            idx = test["test_idx"]
            test_path = os.path.join(tmp, f"gso_test_{idx}.py")
            with open(test_path, "w") as f:
                f.write(test["test_script"])
            copy_to_container(container, test_path, f"/gso_test_{idx}.py")
    logger.info(f"Copied {len(tests)} test scripts to container")


def _extract_time(output: str) -> float:
    """Extract timing from GSO test output (result_N.txt)."""
    import re

    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            val = float(line)
            if val > 0:
                return val
        except ValueError:
            pass
        try:
            data = json.loads(line)
            if isinstance(data, (int, float)):
                return float(data)
            if isinstance(data, dict):
                for key in ("time", "execution_time", "elapsed", "duration"):
                    if key in data:
                        return float(data[key])
        except (json.JSONDecodeError, ValueError):
            pass
        match = re.search(r"([\d.]+)\s*(?:seconds?|sec|s)\b", line)
        if match:
            return float(match.group(1))

    logger.warning(f"Could not extract time from: {output[:200]}")
    return 0.0


def _fix_fstring_syntax(script: str) -> str:
    """Fix Python 3.12+ f-string syntax for Python 3.9 compatibility."""
    lines = script.split("\n")
    fixed_lines = []
    for line in lines:
        if "f'" in line and "['" in line and "']" in line:
            new_line = line
            idx = 0
            while True:
                pos = new_line.find("f'", idx)
                if pos == -1:
                    break
                end = pos + 2
                while end < len(new_line) and new_line[end] != "'":
                    if new_line[end] == "{":
                        depth = 1
                        end += 1
                        while end < len(new_line) and depth > 0:
                            if new_line[end] == "{":
                                depth += 1
                            elif new_line[end] == "}":
                                depth -= 1
                            end += 1
                        continue
                    end += 1
                if end < len(new_line):
                    new_line = new_line[:pos] + 'f"' + new_line[pos + 2 : end] + '"' + new_line[end + 1 :]
                idx = end + 1
            fixed_lines.append(new_line)
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)
