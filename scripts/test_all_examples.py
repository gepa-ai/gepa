#!/usr/bin/env python3
"""
Smoke test for all examples with max_metric_calls=10.

Runs each example's optimize_anything pipeline end-to-end with minimal
metric calls to verify nothing is broken. Uses a cheap LLM to minimize cost.

Usage:
    python -m scripts.test_all_examples
    python -m scripts.test_all_examples --examples linear circle_packing
    python -m scripts.test_all_examples --llm openai/gpt-5-nano
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_LLM = "openai/gpt-5-nano"
MAX_METRIC_CALLS = 10


# ---------------------------------------------------------------------------
# Individual example tests
# ---------------------------------------------------------------------------


def test_linear(llm: str, tmp_dir: str) -> None:
    """optimize_linear_function_params: fit y = b*x + a."""
    from examples.optimize_linear_function_params import fitness_fn, dataset
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )

    seed_candidate = {"function_params": json.dumps({"a": 1.0, "b": 0.0})}
    shuffled = list(dataset)
    random.Random(0).shuffle(shuffled)
    trainset = shuffled[: len(shuffled) // 2]

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=os.path.join(tmp_dir, "linear"),
            max_metric_calls=MAX_METRIC_CALLS,
            seed=42,
        ),
        reflection=ReflectionConfig(
            reflection_lm=llm,
            reflection_minibatch_size=15,
        ),
        refiner=None,
    )

    result = optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=fitness_fn,
        dataset=trainset,
        config=config,
    )
    assert result.best_candidate is not None, "No best candidate returned"


def test_circle_packing(llm: str, tmp_dir: str) -> None:
    """circle_packing: single-instance with RefinerConfig."""
    from examples.circle_packing.utils import (
        execute_code,
        SEED_CODE2,
        CIRCLE_PACKING_BACKGROUND,
    )
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        RefinerConfig,
        optimize_anything,
    )

    TIMEOUT = 60

    def fitness_fn(candidate, best_example_evals=None, **kwargs):
        import numpy as np

        code = candidate["code"]
        best_circles = None
        if best_example_evals:
            circles_list = [
                e["side_info"]["Circles"]
                for e in best_example_evals
                if e["side_info"].get("Circles") is not None
            ]
            if circles_list:
                best_circles = np.array(circles_list)

        result = execute_code(code, TIMEOUT, best_circles)

        if result["success"]:
            score = result["result"]["validation_details"]["sum_radii"]
        else:
            score = 0.0

        side_info = {
            "scores": {"sum_radii": score},
            "Code": code,
            "Circles": result["result"]["circles"] if result["success"] else None,
            "Stdout": result.get("stdout", ""),
            "Error": result.get("error"),
        }
        return score, side_info

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=os.path.join(tmp_dir, "circle_packing"),
            max_metric_calls=MAX_METRIC_CALLS,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=1,
            reflection_lm=llm,
        ),
        refiner=RefinerConfig(),
    )

    result = optimize_anything(
        seed_candidate={"code": SEED_CODE2},
        fitness_fn=fitness_fn,
        config=config,
        objective="Optimize circle packing code to maximize sum of radii in a unit square for N=26.",
        background=CIRCLE_PACKING_BACKGROUND,
    )
    assert result.best_candidate is not None, "No best candidate returned"


def test_polynomial(llm: str, tmp_dir: str) -> None:
    """polynomial: blackbox optimization, single-instance."""
    from examples.polynomial.utils import (
        execute_code,
        extract_best_xs,
        append_eval_history,
        SEED_CODE,
        OBJECTIVE,
        BACKGROUND,
    )
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )

    log_dir = os.path.join(tmp_dir, "polynomial")

    def fitness_fn(candidate, best_example_evals=None, **kwargs):
        code = candidate["code"]
        best_xs = extract_best_xs(best_example_evals)
        result = execute_code(
            code=code,
            problem_index=0,
            timeout=60,
            budget=50,
            best_xs=best_xs,
        )
        append_eval_history(log_dir, result["all_attempts"])
        side_info = {
            "score": result["score"],
            "top_50_attempts": result["top_50_attempts"],
            "bottom_50_attempts": result["bottom_50_attempts"],
            "Stdout": result.get("stdout", ""),
            "Error": result.get("error", ""),
            "Traceback": result.get("traceback", ""),
        }
        return result["score"], side_info

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_candidate_proposals=3,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=1,
            reflection_lm=llm,
        ),
    )

    result = optimize_anything(
        seed_candidate={"code": SEED_CODE},
        fitness_fn=fitness_fn,
        config=config,
        objective=OBJECTIVE,
        background=BACKGROUND,
    )
    assert result.best_candidate is not None, "No best candidate returned"


def test_aime_math(llm: str, tmp_dir: str) -> None:
    """aime_math: prompt optimization with dspy."""
    import dspy
    from examples.aime_math.dataset import load_math_dataset
    from examples.aime_math.main import (
        fitness_fn,
        MathSolverSignature,
        predictor,
        INITIAL_PROMPT,
    )
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )

    # Configure dspy with the test LLM
    solver_lm = dspy.LM(llm, temperature=1.0, max_tokens=16000)
    dspy.configure(lm=solver_lm)

    trainset, _, _ = load_math_dataset()
    # Use only a small subset for testing
    trainset = trainset[:5]

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=os.path.join(tmp_dir, "aime_math"),
            max_metric_calls=MAX_METRIC_CALLS,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=llm,
        ),
        refiner=None,
    )

    result = optimize_anything(
        seed_candidate={"prompt": INITIAL_PROMPT},
        fitness_fn=fitness_fn,
        dataset=trainset,
        config=config,
    )
    assert result.best_candidate is not None, "No best candidate returned"


def test_cloudcast(llm: str, tmp_dir: str) -> None:
    """cloudcast: broadcast routing optimization."""
    from examples.adrs.cloudcast.evaluator import (
        create_fitness_function,
        load_config_dataset,
    )
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )
    from examples.adrs.cloudcast.main import (
        INITIAL_PROGRAM,
        OPTIMIZATION_OBJECTIVE,
        OPTIMIZATION_BACKGROUND,
        DATASET_ROOT,
    )

    samples = load_config_dataset(config_dir=str(DATASET_ROOT))
    if not samples:
        raise RuntimeError(f"No config files found in {DATASET_ROOT}")

    fitness_fn = create_fitness_function(timeout=60)

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=os.path.join(tmp_dir, "cloudcast"),
            max_metric_calls=MAX_METRIC_CALLS,
            track_best_outputs=True,
            use_cloudpickle=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
            reflection_lm=llm,
        ),
        refiner=None,
    )

    result = optimize_anything(
        seed_candidate={"program": INITIAL_PROGRAM},
        fitness_fn=fitness_fn,
        dataset=samples,
        valset=samples,
        objective=OPTIMIZATION_OBJECTIVE,
        background=OPTIMIZATION_BACKGROUND,
        config=config,
    )
    assert result.best_candidate is not None, "No best candidate returned"


def test_can_be_late(llm: str, tmp_dir: str) -> None:
    """can_be_late: cloud scheduling optimization."""
    from examples.adrs.can_be_late.evaluator import create_fitness_function
    from examples.adrs.can_be_late.trace_dataset import load_trace_dataset
    from examples.adrs.can_be_late.main import (
        INITIAL_PROGRAM,
        OPTIMIZATION_OBJECTIVE,
        OPTIMIZATION_BACKGROUND,
        DATASET_ROOT,
    )
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )

    splits = load_trace_dataset(dataset_root=str(DATASET_ROOT), max_traces_per_split=5)
    train_set = splits["train"]
    val_set = splits["val"]

    fitness_fn = create_fitness_function(timeout=60)

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=os.path.join(tmp_dir, "can_be_late"),
            max_metric_calls=MAX_METRIC_CALLS,
            track_best_outputs=True,
            use_cloudpickle=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=3,
            reflection_lm=llm,
        ),
        refiner=None,
    )

    result = optimize_anything(
        seed_candidate={"program": INITIAL_PROGRAM},
        fitness_fn=fitness_fn,
        dataset=train_set,
        valset=val_set,
        objective=OPTIMIZATION_OBJECTIVE,
        background=OPTIMIZATION_BACKGROUND,
        config=config,
    )
    assert result.best_candidate is not None, "No best candidate returned"


def test_arc_agi(llm: str, tmp_dir: str) -> None:
    """arc_agi: agent code evolution for ARC puzzles."""
    from examples.arc_agi.evaluate import run_agent
    from examples.arc_agi.utils import BACKGROUND, OBJECTIVE, load_arc_dataset
    from examples.arc_agi.main import SEED_AGENT_CODE
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        SideInfo,
        optimize_anything,
    )

    MAX_LLM_CALLS = 5

    def fitness_fn(candidate: dict, **kwargs) -> tuple[float, SideInfo]:
        ex = kwargs["example"]
        result = run_agent(
            agent_code=candidate["agent_code"],
            train_in=ex.train_in,
            train_out=ex.train_out,
            test_in=ex.test_in,
            test_out=ex.test_out or None,
            model_id=llm,
            max_llm_calls=MAX_LLM_CALLS,
        )
        llm_tracker = result["llm"]
        score = result["test_score"]
        side_info: SideInfo = {
            "score": score,
            "problem_id": ex.problem_id,
            "agent_code": candidate["agent_code"],
            "training_score": result["training_score"],
            "test_score": result["test_score"],
            "error": result["error"],
        }
        return score, side_info

    train_set, val_set, _ = load_arc_dataset()
    # Use small subsets
    train_set = train_set[:3]
    val_set = val_set[:2]

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=os.path.join(tmp_dir, "arc_agi"),
            max_metric_calls=MAX_METRIC_CALLS,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=llm,
        ),
        refiner=None,
    )

    result = optimize_anything(
        seed_candidate={"agent_code": SEED_AGENT_CODE},
        fitness_fn=fitness_fn,
        dataset=train_set,
        valset=val_set,
        config=config,
        objective=OBJECTIVE,
        background=BACKGROUND,
    )
    assert result.best_candidate is not None, "No best candidate returned"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EXAMPLES = {
    "linear": ("optimize_linear_function_params", test_linear),
    "circle_packing": ("circle_packing (RefinerConfig)", test_circle_packing),
    "polynomial": ("polynomial (blackbox opt)", test_polynomial),
    "aime_math": ("aime_math (prompt opt)", test_aime_math),
    "cloudcast": ("adrs/cloudcast", test_cloudcast),
    "can_be_late": ("adrs/can_be_late", test_can_be_late),
    "arc_agi": ("arc_agi (agent evolution)", test_arc_agi),
    # kernelbench skipped: requires GPU
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Smoke test all GEPA examples")
    parser.add_argument(
        "--examples",
        nargs="*",
        default=None,
        help=f"Examples to run (default: all). Choices: {', '.join(EXAMPLES.keys())}",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=DEFAULT_LLM,
        help=f"LLM model for all tests (default: {DEFAULT_LLM})",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Don't delete output directories after tests",
    )
    args = parser.parse_args()

    examples_to_run = args.examples or list(EXAMPLES.keys())
    tmp_dir = tempfile.mkdtemp(prefix="gepa_test_examples_")

    print("=" * 70)
    print("GEPA Examples Smoke Test")
    print("=" * 70)
    print(f"LLM: {args.llm}")
    print(f"Max metric calls: {MAX_METRIC_CALLS}")
    print(f"Output dir: {tmp_dir}")
    print(f"Examples: {', '.join(examples_to_run)}")
    print("=" * 70)
    print()

    results = {}

    for key in examples_to_run:
        if key not in EXAMPLES:
            print(f"[SKIP] Unknown example: {key}")
            results[key] = ("SKIP", "Unknown example")
            continue

        label, test_fn = EXAMPLES[key]
        print(f"[RUN]  {label} ...", flush=True)
        t0 = time.time()

        try:
            test_fn(args.llm, tmp_dir)
            elapsed = time.time() - t0
            print(f"[PASS] {label} ({elapsed:.1f}s)")
            results[key] = ("PASS", f"{elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[FAIL] {label} ({elapsed:.1f}s)")
            print(f"       {type(e).__name__}: {e}")
            traceback.print_exc()
            results[key] = ("FAIL", str(e))
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key in examples_to_run:
        if key not in results:
            continue
        status, detail = results[key]
        label = EXAMPLES.get(key, (key,))[0]
        icon = {"PASS": "+", "FAIL": "X", "SKIP": "-"}[status]
        print(f"  [{icon}] {label}: {status} ({detail})")

    passed = sum(1 for s, _ in results.values() if s == "PASS")
    failed = sum(1 for s, _ in results.values() if s == "FAIL")
    skipped = sum(1 for s, _ in results.values() if s == "SKIP")
    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)

    if not args.keep_outputs:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"\nOutputs kept at: {tmp_dir}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
