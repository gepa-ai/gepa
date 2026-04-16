#!/usr/bin/env python3
"""
Rediscovering universal operators with GEPA optimize_anything.

Goal: find a binary operator f(x,y) and a companion constant c such that
{c, f} can reconstruct ALL elementary mathematical functions through
composition — the continuous analogue of NAND for Boolean logic.

The evaluator uses complex arithmetic and enumerates ALL binary expression
trees up to a given depth, then checks against 35 target functions.

Usage:
    uv run python -m examples.universal_operator.main

Requires:
    pip install "gepa[full]"
    Set OPENAI_API_KEY (or another LiteLLM-compatible key).
"""

import math
import warnings
from typing import Any

import numpy as np

import gepa.optimize_anything as oa
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Test points — real-valued, algebraically independent transcendentals.
# Stored as complex128 so compositions can pass through complex intermediates.
# ---------------------------------------------------------------------------
UNARY_TEST = np.array(
    [0.5772156649, 1.2824271291, 0.7642236536, 1.4513692349, 0.9159655942],
    dtype=np.complex128,
)
BINARY_X = np.array([0.5772156649, 1.2824271291, 0.7642236536], dtype=np.complex128)
BINARY_Y = np.array([1.4513692349, 0.9159655942, 2.2955871494], dtype=np.complex128)

MAX_TREE_OPS = 4  # Maximum operator applications in an expression tree
MATCH_TOL = 1e-5

# ---------------------------------------------------------------------------
# All 35 target functions
# ---------------------------------------------------------------------------

CONSTANT_TARGETS: dict[str, complex] = {
    "0": 0.0,
    "1": 1.0,
    "2": 2.0,
    "-1": -1.0,
    "i": 1j,
    "e": math.e,
    "pi": math.pi,
}

UNARY_TARGETS: dict[str, Any] = {
    "exp": lambda x: np.exp(x),
    "ln": lambda x: np.log(x),
    "inv": lambda x: 1.0 / x,
    "half": lambda x: x / 2.0,
    "neg": lambda x: -x,
    "sqrt": lambda x: np.sqrt(x + 0j),
    "sqr": lambda x: x ** 2,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "sin": lambda x: np.sin(x),
    "cos": lambda x: np.cos(x),
    "tan": lambda x: np.tan(x),
    "arcsin": lambda x: np.arcsin(x),
    "arccos": lambda x: np.arccos(x),
    "arctan": lambda x: np.arctan(x),
    "sinh": lambda x: np.sinh(x),
    "cosh": lambda x: np.cosh(x),
    "tanh": lambda x: np.tanh(x),
    "arsinh": lambda x: np.arcsinh(x),
    "arcosh": lambda x: np.arccosh(x + 0j),
    "artanh": lambda x: np.arctanh(x),
}

BINARY_TARGETS: dict[str, Any] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "pow": lambda x, y: x ** y,
    "log_b": lambda x, y: np.log(y) / np.log(x),
    "avg": lambda x, y: (x + y) / 2.0,
    "hypot": lambda x, y: np.sqrt(x ** 2 + y ** 2),
}

ALL_TARGET_NAMES = list(CONSTANT_TARGETS) + list(UNARY_TARGETS) + list(BINARY_TARGETS)
TOTAL_TARGETS = len(ALL_TARGET_NAMES)

TARGET_WEIGHTS: dict[str, float] = {}
for n in CONSTANT_TARGETS:
    TARGET_WEIGHTS[n] = 2.0 if n in ("i", "e", "pi") else 1.0
for n in UNARY_TARGETS:
    TARGET_WEIGHTS[n] = 2.0 if n in ("exp", "ln", "sin", "cos", "inv") else 1.5
for n in BINARY_TARGETS:
    TARGET_WEIGHTS[n] = 2.0 if n in ("mul", "div", "pow") else 1.5
MAX_WEIGHTED_SCORE = sum(TARGET_WEIGHTS.values())


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_SAFE_NS: dict[str, Any] = {
    "exp": np.exp, "log": np.log, "ln": np.log,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "pi": np.pi, "e": np.e,
}


def _eval_constant(const_str: str) -> complex | None:
    try:
        val = complex(eval(const_str, {"__builtins__": {}}, {**_SAFE_NS, "i": 1j, "I": 1j}))
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except Exception:
        return None


def _make_op(op_str: str):
    try:
        code = compile(op_str, "<operator>", "eval")
    except SyntaxError:
        return None

    def op(x, y):
        try:
            with np.errstate(all="ignore"):
                return eval(code, {"__builtins__": {}}, {**_SAFE_NS, "x": x, "y": y})
        except Exception:
            return None

    return op


# ---------------------------------------------------------------------------
# Tree-based expression enumeration
#
# Enumerates ALL binary trees with up to max_ops operator applications.
# Each tree has (max_ops + 1) leaves, each leaf is one of the atoms.
# The number of trees is bounded by Catalan(max_ops) * num_atoms^(max_ops+1),
# which is independent of how many distinct values the operator produces.
#
# We use memoization by depth and deduplication by rounded value to prune.
# ---------------------------------------------------------------------------

def _round_key(vals: np.ndarray, decimals: int = 6) -> tuple:
    return tuple(np.round(vals.real, decimals)) + tuple(np.round(vals.imag, decimals))


def _is_valid(vals: np.ndarray) -> bool:
    return not (np.any(np.isnan(vals)) or np.any(np.isinf(vals)) or np.any(np.abs(vals) > 1e10))


def _enumerate_trees(
    op_fn, atoms: list[np.ndarray], max_ops: int
) -> dict[tuple, np.ndarray]:
    """Enumerate all distinct values reachable by binary trees with ≤ max_ops
    applications of op_fn, with leaves drawn from atoms.

    Uses depth-based construction:
      trees(0) = atoms
      trees(d) = trees(d-1) ∪ {op(a, b) : a ∈ trees(i), b ∈ trees(j), i+j = d-1, i < d, j < d}

    This ensures we enumerate every tree shape without the quadratic set explosion.
    """
    # by_depth[d] = dict of {rounded_key: value_array} for trees using exactly d ops
    by_depth: list[dict[tuple, np.ndarray]] = []
    all_seen: dict[tuple, np.ndarray] = {}

    # Depth 0: atoms (0 operator applications)
    d0: dict[tuple, np.ndarray] = {}
    for a in atoms:
        a = np.asarray(a, dtype=np.complex128)
        if _is_valid(a):
            key = _round_key(a)
            if key not in all_seen:
                d0[key] = a
                all_seen[key] = a
    by_depth.append(d0)

    # Depth d: op(a, b) where depth(a) = i, depth(b) = j, max(i,j) = d-1
    # Equivalently: i+j <= d-1 with max(i,j) = d-1
    # So one of (i,j) must be d-1, the other can be 0..d-1
    for d in range(1, max_ops + 1):
        dd: dict[tuple, np.ndarray] = {}

        # Collect all values with depth < d (for the "other" operand)
        all_up_to_prev = list(all_seen.values())

        # New values at depth d-1
        new_at_prev = list(by_depth[d - 1].values())

        # Combine: one operand from depth d-1, other from depth 0..d-1
        # (plus the case where both are from depth d-1)
        pairs_to_try = []
        for a in new_at_prev:
            for b in all_up_to_prev:
                pairs_to_try.append((a, b))
                pairs_to_try.append((b, a))
        # Both from depth d-1 (avoid double-counting with above when a==b)
        for a in new_at_prev:
            for b in new_at_prev:
                pairs_to_try.append((a, b))

        for a, b in pairs_to_try:
            try:
                with np.errstate(all="ignore"):
                    r = op_fn(a, b)
                if r is not None:
                    r = np.asarray(r, dtype=np.complex128)
                    if _is_valid(r):
                        key = _round_key(r)
                        if key not in all_seen:
                            dd[key] = r
                            all_seen[key] = r
            except Exception:
                pass

        by_depth.append(dd)

    return all_seen


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _best_match(seen: dict[tuple, np.ndarray], target_vals: np.ndarray) -> tuple[bool, float]:
    """Check if target is exactly matched; also compute best partial score."""
    target_key = _round_key(target_vals)
    if target_key in seen:
        return True, 1.0

    # Partial credit: find closest expression
    best_rmse = float("inf")
    for expr_vals in seen.values():
        diff = expr_vals - target_vals
        rmse = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
        if rmse < MATCH_TOL:
            return True, 1.0
        if rmse < best_rmse:
            best_rmse = rmse

    return False, 1.0 / (1.0 + best_rmse)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_operator(op_str: str, const_val: complex = 1.0) -> tuple[float, int, dict[str, tuple[bool, float]]]:
    """Return (weighted_score_fraction, num_exact_matches, per_target_details)."""
    op = _make_op(op_str)
    if op is None:
        return 0.0, 0, {n: (False, 0.0) for n in ALL_TARGET_NAMES}

    details: dict[str, tuple[bool, float]] = {}

    # The constant itself is trivially available
    for name, val in CONSTANT_TARGETS.items():
        if abs(complex(val) - const_val) < 1e-9:
            details[name] = (True, 1.0)

    const_to_check = {n: v for n, v in CONSTANT_TARGETS.items() if n not in details}

    # --- constants (atoms = [c]) ---
    c_atoms = [np.full(3, const_val, dtype=np.complex128)]
    c_seen = _enumerate_trees(op, c_atoms, MAX_TREE_OPS + 1)
    for name, val in const_to_check.items():
        tv = np.full(3, complex(val), dtype=np.complex128)
        details[name] = _best_match(c_seen, tv)

    # --- unary targets (atoms = [c, x]) ---
    u_atoms = [np.full_like(UNARY_TEST, const_val), UNARY_TEST]
    u_seen = _enumerate_trees(op, u_atoms, MAX_TREE_OPS)
    for name, fn in UNARY_TARGETS.items():
        try:
            tv = np.asarray(fn(UNARY_TEST), dtype=np.complex128)
            if not _is_valid(tv):
                details[name] = (False, 0.0)
            else:
                details[name] = _best_match(u_seen, tv)
        except Exception:
            details[name] = (False, 0.0)

    # --- binary targets (atoms = [c, x, y]) ---
    b_atoms = [np.full_like(BINARY_X, const_val), BINARY_X, BINARY_Y]
    b_seen = _enumerate_trees(op, b_atoms, MAX_TREE_OPS)
    for name, fn in BINARY_TARGETS.items():
        try:
            tv = np.asarray(fn(BINARY_X, BINARY_Y), dtype=np.complex128)
            if not _is_valid(tv):
                details[name] = (False, 0.0)
            else:
                details[name] = _best_match(b_seen, tv)
        except Exception:
            details[name] = (False, 0.0)

    num_exact = sum(1 for ex, _ in details.values() if ex)
    weighted = sum(
        (1.0 if ex else partial) * TARGET_WEIGHTS[n]
        for n, (ex, partial) in details.items()
    )
    return weighted / MAX_WEIGHTED_SCORE, num_exact, details


# ---------------------------------------------------------------------------
# Evaluator for optimize_anything
# ---------------------------------------------------------------------------

def evaluate(candidate: dict[str, str]) -> float:
    op_str = candidate.get("operator", "x + y")
    const_str = candidate.get("constant", "1")

    const_val = _eval_constant(const_str)
    if const_val is None:
        oa.log(f"Invalid constant: {const_str}")
        return 0.0

    frac, num_exact, details = score_operator(op_str, const_val)

    exact = sorted([n for n, (ex, _) in details.items() if ex])
    partial = sorted(
        [(n, s) for n, (ex, s) in details.items() if not ex and s > 0.01],
        key=lambda x: -x[1],
    )
    missed = sorted([n for n, (ex, s) in details.items() if not ex and s <= 0.01])

    oa.log(f"Operator: {op_str}  |  Constant: {const_str} = {const_val}")
    oa.log(f"Weighted score: {frac:.3f}  |  Exact matches: {num_exact}/{TOTAL_TARGETS}")
    oa.log(f"Exact: {exact}")
    if partial:
        oa.log(f"Partial: {[(n, f'{s:.3f}') for n, s in partial[:12]]}")
    if missed:
        oa.log(f"Missed: {missed}")

    return frac


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OBJECTIVE = (
    "Find a binary operator f(x,y) and a companion constant c such that "
    "composing f with itself starting from c can reconstruct ALL elementary "
    "mathematical functions — the continuous analogue of NAND.\n\n"
    "The 'operator' is a Python expression in x and y. "
    "The 'constant' is a Python expression (e.g. '1', 'e', '2').\n\n"
    "35 targets: constants (0, 1, 2, -1, i, e, pi), "
    "unary functions (exp, ln, inv, half, neg, sqrt, sqr, sigmoid, "
    "sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, "
    "arsinh, arcosh, artanh), binary ops (+, -, *, /, pow, log_base, avg, hypot).\n\n"
    "Score: weighted combination of exact matches and partial credit. "
    "Transcendental/complex targets weighted highest."
)

BACKGROUND = (
    "You are searching for a compositionally universal binary operator and "
    "companion constant.\n\n"
    "The enumeration uses COMPLEX arithmetic and enumerates ALL binary "
    "expression trees up to 4 operator applications (no cap on set size). "
    "Generating sin/cos requires producing i via compositions, then using "
    "Euler's formula.\n\n"
    "Design principles:\n"
    "- Use transcendental functions (exp, log) to reach e, pi, i.\n"
    "- Asymmetry between x and y is critical: e.g., one through exp, the "
    "other through log. Fixing one argument to c should recover a useful "
    "primitive.\n"
    "- f(c,c) must be well-defined and productive.\n"
    "- Shorter expressions compose deeper within the tree budget.\n\n"
    "Available: exp(), log() (=ln), sqrt(), abs(), sin(), cos(), tan(), "
    "pi, e. Operators: +, -, *, /, **."
)


def main():
    result = optimize_anything(
        seed_candidate={"operator": "x + y", "constant": "1"},
        evaluator=evaluate,
        objective=OBJECTIVE,
        background=BACKGROUND,
        config=GEPAConfig(
            reflection=ReflectionConfig(
                reflection_lm="openai/gpt-4.1-mini",
                reflection_minibatch_size=1,
            ),
            engine=EngineConfig(
                max_metric_calls=300,
                seed=42,
            ),
        ),
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    best = result.best_candidate
    print(f"Best operator: {best['operator']}")
    print(f"Best constant: {best['constant']}")
    print(f"Best score:    {result.best_score:.4f}")

    const_val = _eval_constant(best["constant"]) or 1.0
    _, num_exact, details = score_operator(best["operator"], const_val)
    exact = sorted([n for n, (ex, _) in details.items() if ex])
    partial = sorted(
        [(n, s) for n, (ex, s) in details.items() if not ex and s > 0.01],
        key=lambda x: -x[1],
    )
    missed = sorted([n for n, (ex, s) in details.items() if not ex and s <= 0.01])

    print(f"\nExact matches ({num_exact}/{TOTAL_TARGETS}):")
    for m in exact:
        print(f"  [+] {m}")
    if partial:
        print(f"\nPartial:")
        for n, s in partial:
            print(f"  [~] {n}: {s:.3f}")
    if missed:
        print(f"\nMissed:")
        for u in missed:
            print(f"  [-] {u}")


if __name__ == "__main__":
    main()
