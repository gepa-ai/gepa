"""
Evaluate Mini SWE Agent on test set with and without skills.

This script runs the same agent used in train_optimize_anything.py on a test set
in two modes: (1) with learned skills, and (2) without skills (baseline).
Only tasks with non-empty problem statements are evaluated.

Usage:
    python -m src.evaluate.mini_swe_agent \
        --config gepa_results/logs/run_blevesearch_20260131_131944_d7b877_final/config.json \
        --workers 16 \
        --limit 100
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Import from existing modules - no duplication!
from gskill.swe_harness import SWEHarness
from gskill.train_optimize_anything import load_and_split_data


def evaluate_single_task(
    task: dict[str, Any],
    skills: str,
    model_name: str,
    task_idx: int,
    total_tasks: int,
    output_dir: Path,
    harness_pool: list[SWEHarness],
    harness_available: list[bool],
    harness_lock: threading.Lock,
    condition_label: str = "with_skills",
) -> dict[str, Any]:
    """Evaluate a single task and save detailed trace."""

    # Get a harness from the pool
    with harness_lock:
        harness_idx = None
        for i, available in enumerate(harness_available):
            if available:
                harness_available[i] = False
                harness_idx = i
                break

    if harness_idx is None:
        raise RuntimeError("No harness available")

    harness = harness_pool[harness_idx]
    instance_id = task.get("instance_id", "unknown")
    problem_statement = task.get("problem_statement", "")

    result = {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "problem_statement_length": len(problem_statement),
        "has_problem_statement": bool(problem_statement.strip()),
        "patch": "",
        "messages": [],  # Structured messages list
        "passed": False,
        "status": "",
        "test_output": "",
        "error": None,
        "steps": 0,
    }

    result["condition"] = condition_label

    try:
        print(f"[{condition_label}] [{task_idx + 1}/{total_tasks}] {instance_id[:50]}...", flush=True)
        print(f"  Problem statement length: {len(problem_statement)} chars", flush=True)

        # 1. Setup Docker Container
        harness.setup_task(task_instance=task)

        # 2. Run Agent
        patch, _, metrics = harness.run_agent(problem_statement, skills, model_name=model_name)

        result["patch"] = patch
        result["messages"] = metrics.get("messages", [])  # Structured messages
        result["steps"] = metrics.get("steps", 0)

        # 3. Verify with tests
        has_patch = len(patch.strip()) > 0

        if not has_patch:
            result["passed"] = False
            result["status"] = "no_patch"
            result["test_output"] = "No patch to test."
        else:
            # Test FAIL_TO_PASS
            f2p_passed, f2p_output = harness.verify_with_patch(patch, f2p_only=True)
            result["test_output"] = f"=== FAIL_TO_PASS ===\n{f2p_output}"

            if not f2p_passed:
                result["passed"] = False
                result["status"] = "f2p_failed"
            else:
                # Test PASS_TO_PASS
                p2p_passed, p2p_output = harness.verify_with_patch(patch, f2p_only=False)
                result["test_output"] += f"\n\n=== FULL SUITE ===\n{p2p_output}"

                if p2p_passed:
                    result["passed"] = True
                    result["status"] = "all_passed"
                else:
                    result["passed"] = False
                    result["status"] = "p2p_failed"

        status_symbol = "✓" if result["passed"] else "✗"
        print(f"  {status_symbol} {result['status']} (steps={result['steps']}, patch={len(patch)} chars)", flush=True)

    except Exception as e:
        import traceback

        result["error"] = str(e)
        result["status"] = "error"
        result["messages"] = [{"role": "error", "content": traceback.format_exc()}]
        print(f"  ✗ ERROR: {str(e)[:100]}", flush=True)

    finally:
        try:
            harness.cleanup()
        except Exception:
            pass
        with harness_lock:
            harness_available[harness_idx] = True

    # Save individual trace under condition-specific subdirectory
    trace_file = output_dir / "traces" / condition_label / f"{instance_id}.json"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_file, "w") as f:
        json.dump(result, f, indent=2)

    # Also save human-readable markdown
    md_file = output_dir / "traces" / condition_label / f"{instance_id}.md"
    save_trace_as_markdown(result, md_file)

    return result


def save_trace_as_markdown(result: dict[str, Any], filepath: Path):
    """Save trace as human-readable markdown using structured messages."""
    md = []
    md.append(f"# Agent Trace: {result['instance_id']}")
    md.append("")
    md.append("## Summary")
    md.append(f"- **Instance ID**: `{result['instance_id']}`")
    md.append(f"- **Problem Statement Length**: {result['problem_statement_length']} chars")
    md.append(f"- **Has Problem Statement**: {'Yes' if result['has_problem_statement'] else '**NO (EMPTY)**'}")
    md.append(f"- **Condition**: `{result.get('condition', 'unknown')}`")
    md.append(f"- **Status**: `{result['status']}`")
    md.append(f"- **Passed**: {'✓ Yes' if result['passed'] else '✗ No'}")
    md.append(f"- **Steps**: {result['steps']}")
    md.append(f"- **Patch Size**: {len(result['patch'])} chars")
    md.append(f"- **Total Messages**: {len(result.get('messages', []))}")
    md.append("")

    md.append("## Problem Statement")
    md.append("```")
    if result["problem_statement"].strip():
        md.append(result["problem_statement"][:2000])
        if len(result["problem_statement"]) > 2000:
            md.append("... (truncated)")
    else:
        md.append("(EMPTY - no problem statement provided)")
    md.append("```")
    md.append("")

    md.append("## Agent Conversation")
    md.append("")

    # Use structured messages if available
    messages = result.get("messages", [])
    if messages:
        for idx, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            md.append(f"### Message {idx}: {role}")
            md.append("")

            # Truncate very long content
            if len(content) > 3000:
                content = content[:1500] + "\n\n... (truncated) ...\n\n" + content[-1000:]

            md.append(content.strip())
            md.append("")
            md.append("---")
            md.append("")
    else:
        md.append("*(No structured messages available)*")
        md.append("")

    md.append("## Patch")
    md.append("```diff")
    if result["patch"].strip():
        md.append(result["patch"][:3000])
        if len(result["patch"]) > 3000:
            md.append("... (truncated)")
    else:
        md.append("(no patch generated)")
    md.append("```")
    md.append("")

    md.append("## Test Output")
    md.append("```")
    md.append(result["test_output"][:2000] if result["test_output"] else "(none)")
    md.append("```")

    with open(filepath, "w") as f:
        f.write("\n".join(md))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mini SWE Agent with trace logging")
    parser.add_argument("--config", required=True, help="Path to GEPA run config.json")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks (default: all)")
    parser.add_argument("--model", default=None, help="Override model (default: from config)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: auto)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    repo = config["repo"]
    train_size = config["train_size"]
    val_size = config["val_size"]
    test_size = config["test_size"]
    seed = config["seed"]
    model_name = args.model or config.get("model", "gpt-5-mini")

    # Load best skills
    skills_path = config_path.parent / "prompts" / "best_skills.txt"
    if skills_path.exists():
        skills = skills_path.read_text()
        print(f"Loaded skills from: {skills_path}")
        print(f"Skills length: {len(skills)} chars")
    else:
        skills = ""
        print("WARNING: No best_skills.txt found, running with empty skills")

    # Setup output directory
    run_id = config.get("run_id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"gepa_results/mini_swe_eval/{run_id}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("Mini SWE Agent Evaluation")
    print(f"{'=' * 70}")
    print(f"Config: {config_path}")
    print(f"Repo: {repo}")
    print(f"Model: {model_name}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}\n")

    # Load test data using the same function as training
    _, _, test_data = load_and_split_data(repo, train_size, val_size, test_size, seed)

    # Filter to only tasks with non-empty problem statements
    all_count = len(test_data)
    test_data = [t for t in test_data if t.get("problem_statement", "").strip()]
    print(f"Loaded {all_count} test tasks, {len(test_data)} have non-empty problem statements")
    print(f"Filtered out {all_count - len(test_data)} tasks with empty problem_statement")

    if args.limit:
        test_data = test_data[: args.limit]

    print(f"Evaluating {len(test_data)} tasks")
    print()

    # Save config
    eval_config = {
        "source_config": str(config_path),
        "repo": repo,
        "model": model_name,
        "workers": args.workers,
        "total_test_tasks": all_count,
        "filtered_test_tasks": len(test_data),
        "skills_length": len(skills),
        "conditions": ["with_skills", "without_skills"],
        "timestamp": timestamp,
    }
    with open(output_dir / "eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)

    # Save skills
    with open(output_dir / "skills.txt", "w") as f:
        f.write(skills)

    # Define the two evaluation conditions
    conditions = [
        ("with_skills", skills),
        ("without_skills", ""),
    ]

    all_condition_results = {}

    for condition_label, condition_skills in conditions:
        print(f"\n{'=' * 70}")
        print(f"CONDITION: {condition_label}")
        print(
            f"  Skills: {'yes (' + str(len(condition_skills)) + ' chars)' if condition_skills else 'none (baseline)'}"
        )
        print(f"  Tasks: {len(test_data)}")
        print(f"{'=' * 70}\n")

        # Create harness pool
        harness_pool = [SWEHarness() for _ in range(args.workers)]
        harness_available = [True] * args.workers
        harness_lock = threading.Lock()

        # Evaluate tasks in parallel
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for idx, task in enumerate(test_data):
                future = executor.submit(
                    evaluate_single_task,
                    task,
                    condition_skills,
                    model_name,
                    idx,
                    len(test_data),
                    output_dir,
                    harness_pool,
                    harness_available,
                    harness_lock,
                    condition_label,
                )
                futures.append(future)

            for future in futures:
                results.append(future.result())

        all_condition_results[condition_label] = results

        # Print condition summary
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        print(f"\n  {condition_label}: {passed}/{total} ({100 * passed / total:.1f}%)")

    # Print comparison summary
    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE - COMPARISON")
    print(f"{'=' * 70}")

    ws_total = len(all_condition_results["with_skills"])
    ws_passed = sum(1 for r in all_condition_results["with_skills"] if r["passed"])
    wo_total = len(all_condition_results["without_skills"])
    wo_passed = sum(1 for r in all_condition_results["without_skills"] if r["passed"])

    print(f"  With skills pass rate:    {ws_passed}/{ws_total} ({100 * ws_passed / ws_total:.1f}%)")
    print(f"  Without skills pass rate: {wo_passed}/{wo_total} ({100 * wo_passed / wo_total:.1f}%)")

    # Per-task comparison
    with_skills_results = {r["instance_id"]: r for r in all_condition_results["with_skills"]}
    without_skills_results = {r["instance_id"]: r for r in all_condition_results["without_skills"]}

    skills_only = 0  # passed with skills but not without
    baseline_only = 0  # passed without skills but not with
    both_passed = 0
    neither_passed = 0

    for iid in with_skills_results:
        ws = with_skills_results[iid]["passed"]
        wo = without_skills_results[iid]["passed"]
        if ws and wo:
            both_passed += 1
        elif ws and not wo:
            skills_only += 1
        elif not ws and wo:
            baseline_only += 1
        else:
            neither_passed += 1

    total = len(with_skills_results)
    print()
    print(f"  Both passed:           {both_passed:4d} ({100 * both_passed / total:.1f}%)")
    print(f"  Skills only:           {skills_only:4d} ({100 * skills_only / total:.1f}%)")
    print(f"  Baseline only:         {baseline_only:4d} ({100 * baseline_only / total:.1f}%)")
    print(f"  Neither passed:        {neither_passed:4d} ({100 * neither_passed / total:.1f}%)")
    print(f"{'=' * 70}")

    # Save summary
    summary = {
        "total_tasks": total,
        "conditions": {},
        "comparison": {
            "both_passed": both_passed,
            "skills_only": skills_only,
            "baseline_only": baseline_only,
            "neither_passed": neither_passed,
        },
        "per_task": [],
    }

    for condition_label, results in all_condition_results.items():
        passed = sum(1 for r in results if r["passed"])
        summary["conditions"][condition_label] = {
            "total": len(results),
            "passed": passed,
            "pass_rate": passed / len(results) if results else 0,
        }

    for iid in with_skills_results:
        summary["per_task"].append(
            {
                "instance_id": iid,
                "with_skills_passed": with_skills_results[iid]["passed"],
                "with_skills_status": with_skills_results[iid]["status"],
                "with_skills_steps": with_skills_results[iid]["steps"],
                "without_skills_passed": without_skills_results[iid]["passed"],
                "without_skills_status": without_skills_results[iid]["status"],
                "without_skills_steps": without_skills_results[iid]["steps"],
            }
        )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
