#!/usr/bin/env python3
"""Frontier-CS optimization using GEPA's optimize_anything() in seedless mode.

Per-problem mode: optimizes a separate prompt for each problem independently.
GEPA evolves internal prompt templates; prompt_candidate_lm generates C++ code
from those templates; the evaluator submits the code to the judge and scores it.

Multi-objective scoring:
  - correctness: fraction of AC test cases (0.0-1.0)
  - compilation: 1.0 if compiles, 0.0 otherwise
  - time_efficiency: fraction without TLE (0.0-1.0)
  - stability: fraction without RE (0.0-1.0)
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from examples.frontiercs.utils.background import build_background
from examples.frontiercs.utils.dataset import get_frontiercs_problems_dir, load_all_problems
from examples.frontiercs.utils.judge import FrontierCSJudgeClient, extract_cpp_code
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)

JUDGE: FrontierCSJudgeClient | None = None
VERBOSE: bool = False


def _build_feedback(result: dict[str, Any], overall: float) -> str:
    """Build human-readable feedback from judge result."""
    status = result.get("status", "unknown")
    raw_score = result.get("score", 0)
    cases = result.get("cases", [])

    if status == "done":
        if result.get("passed"):
            return f"All test cases passed. Score: {raw_score}/100."
        result_str = result.get("result", "")
        case_msgs = [
            f"Case {i + 1}: {c.get('status', '?')}" + (f" - {c.get('msg', '')}" if c.get("msg") else "")
            for i, c in enumerate(cases[:5])
            if isinstance(c, dict)
        ]
        return f"Score: {raw_score}/100. Result: {result_str}. Case details: {'; '.join(case_msgs)}"

    err = result.get("error", "Unknown error")
    return f"Judge error: {err}"


def evaluate(candidate: str, example: dict[str, Any], **kwargs: Any) -> tuple[float, SideInfo]:
    """Evaluate generated C++ code on a single problem.

    In seedless mode, GEPA's prompt_candidate_lm already generated the C++ code.
    The evaluator just extracts it, submits to the judge, and scores.

    Args:
        candidate: LLM-generated C++ code (produced by prompt_candidate_lm).
        example: Problem dict with problem_id, statement, etc.

    Returns:
        (overall_score, side_info) where overall_score is 0.0-1.0.
    """
    assert JUDGE is not None
    problem_id = example["problem_id"]
    code = extract_cpp_code(candidate)

    if not code:
        return 0.0, {
            "scores": {"correctness": 0.0, "compilation": 0.0, "time_efficiency": 0.0, "stability": 0.0},
            "feedback": "No C++ code extracted from candidate.",
            "generated_code": "",
        }

    sid = JUDGE.submit_solution(problem_id, code)
    if not sid:
        return 0.0, {
            "scores": {"correctness": 0.0, "compilation": 0.0, "time_efficiency": 0.0, "stability": 0.0},
            "feedback": "Failed to submit solution to judge. Is the judge server running?",
            "generated_code": code,
        }

    result = JUDGE.get_result(sid) or {"status": "error", "error": "NO_RESULT", "score": 0}

    cases = result.get("cases", [])
    n = max(len(cases), 1)
    correctness = sum(1 for c in cases if isinstance(c, dict) and c.get("status") == "AC") / n
    compilation = 0.0 if result.get("error") in ("COMPILATION_ERROR", "CE") else 1.0
    time_efficiency = sum(1 for c in cases if isinstance(c, dict) and c.get("status") != "TLE") / n
    stability = sum(1 for c in cases if isinstance(c, dict) and c.get("status") != "RE") / n
    overall = max(0.0, min(1.0, result.get("score", 0) / 100.0))

    feedback = _build_feedback(result, overall)

    if VERBOSE and overall == 0.0:
        err_info = result.get("error") or result.get("result") or feedback
        print(f"    [DEBUG] problem_id={problem_id} score=0: {err_info}", flush=True)

    return overall, {
        "scores": {
            "correctness": correctness,
            "compilation": compilation,
            "time_efficiency": time_efficiency,
            "stability": stability,
        },
        "feedback": feedback,
        "generated_code": code,
        "judge_result": result,
    }


def optimize_one(
    problem: dict[str, Any],
    index: int,
    args: argparse.Namespace,
) -> tuple[str, float, str]:
    """Run optimize_anything for a single problem."""
    problem_id = problem["problem_id"]

    result = optimize_anything(
        seed_candidate=None,
        evaluator=evaluate,
        dataset=[problem],
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=f"{args.output_dir}/runs/{problem_id}/",
                seed=args.seed + index,
                max_metric_calls=args.max_metric_calls,
                frontier_type="objective",
                cache_evaluation=True,
                track_best_outputs=True,
            ),
            reflection=ReflectionConfig(reflection_lm=args.reflection_lm),
        ),
        objective="Generate correct, efficient C++ solutions for competitive programming problems. "
        "Maximize accepted test cases, avoid TLE and runtime errors.",
        background=build_background(problem),
    )

    raw_score = result.val_aggregate_scores[result.best_idx] * 100.0
    best_prompt = result.best_candidate or ""
    return problem_id, raw_score, best_prompt


def main() -> None:
    global JUDGE, VERBOSE

    parser = argparse.ArgumentParser(description="GEPA per-problem optimization for Frontier-CS (seedless mode)")
    parser.add_argument(
        "--problems_dir",
        type=str,
        default=None,
        help="Path to Frontier-CS problems directory (default: clone from github)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gepa_frontiercs_results",
        help="Directory to save results (prompts and scores)",
    )
    parser.add_argument(
        "--reflection_lm",
        type=str,
        default="openai/gpt-5",
        help="Reflection LLM for prompt evolution",
    )
    parser.add_argument(
        "--judge_url",
        type=str,
        default="http://localhost:8081",
        help="Frontier-CS judge API URL",
    )
    parser.add_argument(
        "--max_metric_calls",
        type=int,
        default=25,
        help="Optimization budget (evaluations) per problem",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (offset per problem)",
    )
    parser.add_argument(
        "--problem_ids",
        type=str,
        nargs="*",
        default=None,
        help="Only optimize these problem IDs (default: all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Number of problems to optimize in parallel",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="gepa_frontiercs.log",
        help="Log file path (default: gepa_frontiercs.log in project root)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug info when score is 0 (e.g. judge/LLM errors)",
    )
    args = parser.parse_args()

    # Set module-level state for evaluator
    JUDGE = FrontierCSJudgeClient(args.judge_url)
    VERBOSE = args.verbose

    # Load problems
    problems_dir_resolved = args.problems_dir or str(get_frontiercs_problems_dir())
    problems = load_all_problems(problems_dir=args.problems_dir)
    if args.problem_ids:
        id_set = set(args.problem_ids)
        problems = [p for p in problems if p["problem_id"] in id_set]
        if len(problems) != len(id_set):
            found = {p["problem_id"] for p in problems}
            missing = id_set - found
            raise ValueError(f"Problems not found: {missing}")

    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    log_file = None
    if args.log_file:
        log_file = open(args.log_file, "w", encoding="utf-8")  # noqa: SIM115

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()

    log("-" * 60)
    log(f"Problems dir: {problems_dir_resolved}")
    log(f"Output dir: {args.output_dir}")
    log(f"Reflection LM: {args.reflection_lm}")
    log(f"Judge: {args.judge_url}, Budget per problem: {args.max_metric_calls}")
    log("Mode: seedless (optimize_anything), frontier_type=objective")
    log(f"Total problems: {len(problems)}, Concurrency: {args.concurrency}")
    log("-" * 60)

    results: dict[str, dict[str, Any]] = {}
    log_lock = threading.Lock()
    results_lock = threading.Lock()
    completed = 0

    def _save_scores() -> None:
        scores_path = output_dir / "scores.json"
        scores_path.write_text(
            json.dumps({pid: r["score"] for pid, r in results.items()}, indent=2),
            encoding="utf-8",
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(optimize_one, problem, i, args): problem["problem_id"] for i, problem in enumerate(problems)
        }
        for future in as_completed(futures):
            problem_id = futures[future]
            try:
                problem_id, raw_score, best_prompt = future.result()
                with results_lock:
                    results[problem_id] = {"score": round(raw_score, 2), "prompt": best_prompt}
                    _save_scores()
                (prompts_dir / f"{problem_id}.txt").write_text(best_prompt, encoding="utf-8")
                completed += 1
                with log_lock:
                    log(f"[{completed}/{len(problems)}] Problem {problem_id} done: {raw_score:.2f}/100")
            except Exception as e:
                with results_lock:
                    results[problem_id] = {"score": 0.0, "prompt": ""}
                    _save_scores()
                (prompts_dir / f"{problem_id}.txt").write_text("", encoding="utf-8")
                completed += 1
                with log_lock:
                    log(f"ERROR problem {problem_id}: {e}")

    # Save final summary
    summary = {
        "config": {
            "reflection_lm": args.reflection_lm,
            "max_metric_calls_per_problem": args.max_metric_calls,
            "mode": "seedless",
            "frontier_type": "objective",
        },
        "problems": {pid: {"score": r["score"], "prompt_path": f"prompts/{pid}.txt"} for pid, r in results.items()},
        "full_prompts": {pid: r["prompt"] for pid, r in results.items()},
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log("-" * 60)
    log(f"Results saved to {output_dir}")
    avg_score = sum(r["score"] for r in results.values()) / len(results) if results else 0
    log(f"Average score: {avg_score:.2f}/100")
    log("-" * 60)

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
