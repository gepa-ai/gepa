#!/usr/bin/env python3
"""Run GEPA optimization on a single GSO benchmark instance.

Each GSO instance has multiple test scripts. These are split into
train/val/test sets for GEPA's generalization mode:
- Train: coding agent sees feedback from these tests during reflection
- Val: GEPA tracks Pareto frontier and accepts/rejects candidates
- Test: held out for final evaluation

Usage:
    uv run python -m examples.gso.gso_runner --instance_id "huggingface__datasets-5994036" --max_iters 10

    # List available instances:
    uv run python -m examples.gso.gso_runner --list

    # Use bash agent instead of Claude Code:
    uv run python -m examples.gso.gso_runner --instance_id "..." --agent bash --model "openai/gpt-5.1"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess

from examples.gso.docker_utils import (
    create_container,
    get_image_name,
    pull_image,
    stop_and_remove_container,
)
from examples.gso.gso_evaluator import make_gso_evaluator, prepare_tests, split_tests
from examples.gso.prepare import list_instances, load_instance, prepare_repo

from gepa.optimize_anything import (
    CodeCandidate,
    EngineConfig,
    GEPAConfig,
    optimize_anything,
)

logger = logging.getLogger(__name__)


def run_gso_instance(
    instance_id: str,
    max_iterations: int = 10,
    agent: str = "claude_code",
    model: str = "openai/gpt-5.1",
    work_dir: str = "gso_workdir",
    output_dir: str = "gso_outputs",
    namespace: str = "slimshetty",
    train_ratio: float = 0.5,
    val_ratio: float = 0.3,
) -> dict:
    """Run GEPA optimization on a single GSO instance.

    Args:
        instance_id: GSO instance identifier.
        max_iterations: Maximum number of GEPA metric calls.
        agent: Coding agent to use ("claude_code" or "bash").
        model: LLM model for bash agent.
        work_dir: Directory for cloned repos.
        output_dir: Directory for output files.
        namespace: Docker Hub namespace for GSO images.
        train_ratio: Fraction of tests for training.
        val_ratio: Fraction of tests for validation.

    Returns:
        Dict with instance_id, best_branch, prediction_path, speedup.
    """
    # 1. Load instance
    logger.info(f"Loading GSO instance: {instance_id}")
    instance = load_instance(instance_id)

    # 2. Prepare repo
    repo_path = prepare_repo(instance, work_dir=work_dir)

    # 3. Pull Docker image
    arch = instance.get("arch", "x86_64")
    image_name = get_image_name(instance_id, arch=arch, namespace=namespace)
    logger.info(f"Docker image: {image_name}")
    pull_image(image_name)

    # 4. Create container
    container_name = f"gepa-gso-{instance_id.replace('/', '-')}"
    container = create_container(image_name, name=container_name)

    try:
        # 5. Split tests into train/val/test
        all_tests = prepare_tests(instance)
        train_tests, val_tests, test_tests = split_tests(all_tests, train_ratio, val_ratio)
        logger.info(
            f"Test split: {len(train_tests)} train, {len(val_tests)} val, {len(test_tests)} test "
            f"(total: {len(all_tests)})"
        )

        # 6. Build evaluator
        evaluator = make_gso_evaluator(
            instance=instance,
            container=container_name,
            repo_path=repo_path,
        )

        # 7. Build objective and background
        prob_script = instance["prob_script"]
        hints = instance.get("hints_text", "")
        repo_name = instance["repo"]

        objective = (
            f"Optimize the runtime performance of the code in this repository ({repo_name}). "
            f"The performance is measured by multiple test scripts that time different operations. "
            f"Make the code run as fast as possible while maintaining correctness (tests must pass)."
        )

        background_parts = [f"Repository: {repo_name}"]
        if hints:
            background_parts.append(f"Hints from the optimization target:\n{hints}")
        background_parts.append(
            f"Performance test module (prob_script) that defines the benchmarked operations:\n"
            f"```python\n{prob_script}\n```"
        )
        background = "\n\n".join(background_parts)

        # 8. Run GEPA optimization
        os.makedirs(output_dir, exist_ok=True)
        run_dir = os.path.join(output_dir, instance_id.replace("/", "__"))

        result = optimize_anything(
            seed_candidate=CodeCandidate(
                repo_paths=repo_path,
                base_branch="base",
                coding_agent=agent,
                model=model,
                branch_prefix="gepa",
            ),
            evaluator=evaluator,
            dataset=train_tests,
            valset=val_tests,
            objective=objective,
            background=background,
            config=GEPAConfig(
                engine=EngineConfig(
                    max_metric_calls=max_iterations,
                    run_dir=run_dir,
                    parallel=False,
                    display_progress_bar=True,
                    frontier_type="instance",
                ),
            ),
        )

        # 9. Extract results
        best_candidate = result.best_candidate
        assert isinstance(best_candidate, dict)
        best_branch = best_candidate[repo_path]
        best_score = result.val_aggregate_scores[result.best_idx]

        logger.info(f"Best branch: {best_branch} (val speedup: {best_score:.2f}x)")

        # Get diff
        diff_result = subprocess.run(
            ["git", "diff", "base", best_branch],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        model_patch = diff_result.stdout

        # Write prediction JSONL
        os.makedirs(run_dir, exist_ok=True)
        prediction = {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": f"gepa-{agent}",
        }
        prediction_path = os.path.join(run_dir, "prediction.jsonl")
        with open(prediction_path, "w") as f:
            f.write(json.dumps(prediction) + "\n")

        # Evaluate on held-out test set if available
        test_speedups = {}
        avg_test_speedup = None
        if test_tests:
            logger.info(f"Evaluating best candidate on {len(test_tests)} held-out tests...")
            for test_example in test_tests:
                score, info = evaluator(best_candidate, test_example)
                test_speedups[test_example["test_idx"]] = score
                logger.info(f"  Test {test_example['test_idx']}: {score:.2f}x")
            avg_test_speedup = sum(test_speedups.values()) / len(test_speedups) if test_speedups else 0

        summary = {
            "instance_id": instance_id,
            "best_branch": best_branch,
            "val_speedup": best_score,
            "test_speedups": test_speedups,
            "avg_test_speedup": avg_test_speedup if test_tests else None,
            "prediction_path": prediction_path,
            "num_candidates": result.num_candidates,
            "repo_path": repo_path,
            "split": {
                "train": len(train_tests),
                "val": len(val_tests),
                "test": len(test_tests),
            },
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    finally:
        stop_and_remove_container(container_name)


def main():
    parser = argparse.ArgumentParser(description="Run GEPA optimization on a GSO benchmark instance")
    parser.add_argument("--instance_id", type=str, help="GSO instance identifier")
    parser.add_argument("--max_iters", type=int, default=10, help="Maximum GEPA metric calls")
    parser.add_argument("--agent", type=str, default="claude_code", choices=["claude_code", "bash"])
    parser.add_argument("--model", type=str, default="openai/gpt-5.1", help="LLM model for bash agent")
    parser.add_argument("--work_dir", type=str, default="gso_workdir", help="Directory for cloned repos")
    parser.add_argument("--output_dir", type=str, default="gso_outputs", help="Directory for outputs")
    parser.add_argument("--namespace", type=str, default="slimshetty", help="Docker Hub namespace")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Fraction of tests for training")
    parser.add_argument("--val_ratio", type=float, default=0.3, help="Fraction of tests for validation")
    parser.add_argument("--list", action="store_true", help="List available instance IDs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list:
        instances = list_instances()
        print(f"Available GSO instances ({len(instances)}):")
        for iid in instances:
            print(f"  {iid}")
        return

    if not args.instance_id:
        parser.error("--instance_id is required (use --list to see available instances)")

    summary = run_gso_instance(
        instance_id=args.instance_id,
        max_iterations=args.max_iters,
        agent=args.agent,
        model=args.model,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        namespace=args.namespace,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print("\n=== GSO Optimization Complete ===")
    print(f"Instance:    {summary['instance_id']}")
    print(f"Best branch: {summary['best_branch']}")
    print(f"Val speedup: {summary['val_speedup']:.2f}x")
    if summary.get("avg_test_speedup") is not None:
        print(f"Test speedup:{summary['avg_test_speedup']:.2f}x (avg over {summary['split']['test']} tests)")
    print(f"Candidates:  {summary['num_candidates']}")
    print(f"Split:       {summary['split']}")
    print(f"Prediction:  {summary['prediction_path']}")
    print(f"Repo:        {summary['repo_path']}")


if __name__ == "__main__":
    main()
