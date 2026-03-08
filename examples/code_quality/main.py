"""
Code quality optimization example — M1 string-mode.

Optimizes a deliberately messy Python file (seed/solver.py) for readability
and code quality, scored by AST-based static analysis (no external deps).

Usage:
    uv run python -u -m examples.code_quality.main
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything


# ---------------------------------------------------------------------------
# AST-based code quality scorer (no external dependencies)
# ---------------------------------------------------------------------------


def _naming_score(tree: ast.AST) -> tuple[float, dict]:
    """Score function/variable naming quality (0-1). Longer, descriptive names score higher."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.append(node.name)
            names.extend(arg.arg for arg in node.args.args)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.append(node.id)

    if not names:
        return 1.0, {"names": []}

    idioms = {"i", "j", "k", "x", "y", "n", "_"}
    good = sum(1 for n in names if len(n) > 1 or n in idioms)
    score = good / len(names)
    bad_names = [n for n in names if len(n) == 1 and n not in idioms]
    return score, {"total_names": len(names), "bad_names": bad_names[:20]}


def _docstring_score(tree: ast.AST) -> tuple[float, dict]:
    """Score docstring coverage (0-1). Fraction of functions with docstrings."""
    functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    if not functions:
        return 1.0, {"total_functions": 0}

    with_docstring = sum(1 for fn in functions if ast.get_docstring(fn) is not None)
    missing = [fn.name for fn in functions if ast.get_docstring(fn) is None]
    return with_docstring / len(functions), {
        "total_functions": len(functions),
        "with_docstring": with_docstring,
        "missing_docstring": missing[:20],
    }


def _complexity_score(tree: ast.AST) -> tuple[float, dict]:
    """Score cyclomatic complexity (0-1). Lower complexity = higher score."""
    branch_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With, ast.BoolOp)
    functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    if not functions:
        return 1.0, {"avg_complexity": 0}

    complexities = {}
    for fn in functions:
        cc = 1
        for child in ast.walk(fn):
            if isinstance(child, branch_nodes):
                cc += 1
            elif isinstance(child, ast.BoolOp):
                cc += len(child.values) - 1
        complexities[fn.name] = cc

    avg_cc = sum(complexities.values()) / len(complexities)
    score = max(0.0, 1.0 - (avg_cc - 1) * 0.07)
    return score, {"avg_complexity": round(avg_cc, 2), "per_function": complexities}


def _function_length_score(tree: ast.AST) -> tuple[float, dict]:
    """Score average function length (0-1). Shorter functions score higher."""
    functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    if not functions:
        return 1.0, {"avg_lines": 0}

    lengths = {}
    for fn in functions:
        lines = set()
        for child in ast.walk(fn):
            if hasattr(child, "lineno"):
                lines.add(child.lineno)
        lengths[fn.name] = len(lines)

    avg_len = sum(lengths.values()) / len(lengths)
    score = max(0.0, 1.0 - (avg_len - 5) * 0.04)
    return score, {"avg_lines": round(avg_len, 1), "per_function": lengths}


TEST_CODE = """\
import sys
sys.path.insert(0, '.')
from solver import c, f, g, h, p, z, corr, report

data = [4, 8, 15, 16, 23, 42]
results = []

try:
    m = c(data)
    assert abs(m - 18.0) < 0.01, f"mean: {m}"
    results.append("mean:OK")
except Exception as e:
    results.append(f"mean:FAIL({e})")

try:
    s = f(data)
    assert abs(s - 12.3153) < 0.01, f"std: {s}"
    results.append("std:OK")
except Exception as e:
    results.append(f"std:FAIL({e})")

try:
    med = g(data)
    assert abs(med - 15.5) < 0.01, f"median: {med}"
    results.append("median:OK")
except Exception as e:
    results.append(f"median:FAIL({e})")

try:
    r = h(data)
    assert abs(r - 38.0) < 0.01, f"range: {r}"
    results.append("range:OK")
except Exception as e:
    results.append(f"range:FAIL({e})")

try:
    rep = report(data)
    assert "mean" in rep and "std" in rep and "median" in rep
    results.append("report:OK")
except Exception as e:
    results.append(f"report:FAIL({e})")

ok = sum(1 for r in results if ":OK" in r)
print(f"PASSED={ok}/{len(results)}")
for r in results:
    print(r)
"""


def _correctness_score(source: str) -> tuple[float, dict]:
    """Write source to a temp dir and run smoke tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "solver.py").write_text(source)
        result = subprocess.run(
            [sys.executable, "-c", TEST_CODE],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=10,
        )
    output = result.stdout.strip()
    if result.returncode != 0:
        return 0.0, {"error": result.stderr[-500:], "stdout": output}

    for line in output.splitlines():
        if line.startswith("PASSED="):
            parts = line.split("=")[1].split("/")
            passed, total = int(parts[0]), int(parts[1])
            return passed / total, {"stdout": output}

    return 0.0, {"error": "no PASSED line", "stdout": output}


# ---------------------------------------------------------------------------
# Evaluator (M1 string mode — candidate is the source code text)
# ---------------------------------------------------------------------------

W_NAMING = 0.25
W_DOCSTRING = 0.25
W_COMPLEXITY = 0.20
W_LENGTH = 0.10
W_CORRECTNESS = 0.20


def evaluate(candidate: str) -> tuple[float, dict]:
    """Evaluate code quality of a candidate solver.py source string."""
    try:
        tree = ast.parse(candidate)
    except SyntaxError as exc:
        return 0.0, {"error": f"syntax error: {exc}"}

    naming, naming_info = _naming_score(tree)
    docstring, docstring_info = _docstring_score(tree)
    complexity, complexity_info = _complexity_score(tree)
    length, length_info = _function_length_score(tree)
    correctness, correctness_info = _correctness_score(candidate)

    if correctness < 1.0:
        composite = correctness * 0.5
    else:
        composite = (
            W_NAMING * naming
            + W_DOCSTRING * docstring
            + W_COMPLEXITY * complexity
            + W_LENGTH * length
            + W_CORRECTNESS * correctness
        )

    side_info = {
        "composite": round(composite, 4),
        "naming": round(naming, 4),
        "docstring": round(docstring, 4),
        "complexity": round(complexity, 4),
        "length": round(length, 4),
        "correctness": round(correctness, 4),
        "naming_detail": naming_info,
        "docstring_detail": docstring_info,
        "complexity_detail": complexity_info,
        "length_detail": length_info,
        "correctness_detail": correctness_info,
    }
    return composite, side_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OBJECTIVE = """\
Improve the code quality of this Python statistics utility module.

The code is scored on five axes (all 0-1, higher is better):
- **Naming** (25%): Use descriptive function and variable names (not single letters).
- **Docstrings** (25%): Every function should have a docstring.
- **Complexity** (20%): Keep cyclomatic complexity low. Simplify branching.
- **Function length** (10%): Prefer shorter, focused functions.
- **Correctness** (20%): The code must still pass functional tests. Do NOT change
  the public API (function names and signatures must stay the same).

IMPORTANT: You must keep the existing function names (c, f, g, h, p, z, corr, report)
as the public API. You may rename internal variables and add helper functions, but
the top-level callable names must remain unchanged for the tests to pass."""


if __name__ == "__main__":
    seed_source = (Path(__file__).parent / "seed" / "solver.py").read_text()

    # Baseline
    baseline_score, baseline_info = evaluate(seed_source)
    print(f"Baseline score: {baseline_score:.4f}")
    for key in ["naming", "docstring", "complexity", "length", "correctness"]:
        print(f"  {key}: {baseline_info[key]:.4f}")
    print()

    # Optimize with CC (haiku for fast/cheap iterations)
    result = optimize_anything(
        seed_candidate=seed_source,
        evaluator=evaluate,
        objective=OBJECTIVE,
        config=GEPAConfig(
            engine=EngineConfig(run_dir="outputs/code_quality", max_iters=5),
            reflection=ReflectionConfig(reflection_lm="claude_code/haiku"),
        ),
    )
    print(f"\nBest score: {result.best_score:.4f}")
    print(f"Best candidate:\n{result.best_candidate[:500]}")
