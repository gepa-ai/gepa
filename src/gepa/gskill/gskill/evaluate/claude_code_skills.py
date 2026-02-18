"""
Evaluate testset with Claude Code using proper Claude Code Skills.

This is a variant of evaluate_claude_code.py that installs skills as proper
Claude Code skills (.claude/skills/<repo>/SKILL.md with YAML frontmatter)
instead of plain CLAUDE.md files.

See: https://code.claude.com/docs/en/skills

Usage is identical to evaluate_claude_code.py:
  python -m src.evaluate.claude_code_skills --config gepa_results/logs/run_xxx/config.json --use-skills
"""

import argparse
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import docker

# Import all shared utilities from the original evaluate_claude_code module.
# This avoids duplicating data loading, image management, verification, etc.
from gskill.evaluate.claude_code import (
    # Result types
    ClaudeCodeResult,
    ensure_claude_code_image,
    # Docker image management
    get_base_image_name,
    load_test_ids,
    load_testset_by_ids,
    load_testset_from_config,
    save_test_ids,
    # Verification
    verify_patch_with_harness,
)
from gskill.evaluate.claude_code import (
    # Original Claude Code runner (imported as private; we wrap it)
    run_claude_code_in_container as _run_claude_code_in_container,
)
from swebench.harness.constants import (
    DOCKER_USER,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
)

# ============================================================================
# Skill Installation (the key difference from evaluate_claude_code.py)
# ============================================================================


def format_skill_md(skills_content: str, repo_name: str) -> str:
    """
    Wrap raw skills text with proper SKILL.md YAML frontmatter.

    The frontmatter follows the Claude Code skill spec:
      https://code.claude.com/docs/en/skills

    Args:
        skills_content: Raw skills text (e.g. contents of best_skills.txt)
        repo_name: Repository name (e.g. "blevesearch__bleve")

    Returns:
        Complete SKILL.md content with frontmatter
    """
    # Clean repo name for the `name` field (lowercase, hyphens only)
    skill_name = repo_name.replace("__", "-").replace("_", "-").lower()

    # Build an informative description so Claude knows when to apply the skill
    human_repo = repo_name.replace("__", "/").replace("_", "-")
    description = (
        f"Expert bug-fixing skills optimized for the {human_repo} repository. "
        f"Contains repo-specific patterns, diagnostic strategies, and "
        f"step-by-step workflows for efficiently finding and resolving bugs "
        f"in this codebase. Use when debugging issues, fixing failing tests, "
        f"or resolving reported bugs."
    )

    return f"---\nname: {skill_name}-bugfix\ndescription: {description}\n---\n\n{skills_content}"


def install_skill_in_container(
    container,
    skills_content: str,
    repo_name: str,
) -> None:
    """
    Install skills as a proper Claude Code skill inside the container.

    Creates:
        <workdir>/.claude/skills/<repo_name>/SKILL.md

    with YAML frontmatter so Claude Code discovers and loads it automatically.

    Args:
        container: Running Docker container
        skills_content: Raw skills text
        repo_name: Repository name used as the skill directory name
    """
    from swebench.harness.docker_utils import copy_to_container

    skill_md = format_skill_md(skills_content, repo_name)
    skill_dir = f"{DOCKER_WORKDIR}/.claude/skills/{repo_name}"

    # Create the skill directory structure
    container.exec_run(f"mkdir -p {skill_dir}", user="root")

    # Write SKILL.md into the container
    skill_file_local = Path(f"/tmp/skill_{uuid.uuid4().hex[:8]}.md")
    skill_file_container = Path(f"{skill_dir}/SKILL.md")
    try:
        skill_file_local.write_text(skill_md)
        copy_to_container(container, skill_file_local, skill_file_container)
        container.exec_run("sync", user="root")
        # Make the whole .claude tree readable
        container.exec_run(f"chmod -R 755 {DOCKER_WORKDIR}/.claude", user="root")
        print(f"  Installed skill at .claude/skills/{repo_name}/SKILL.md ({len(skill_md)} chars)")
    finally:
        skill_file_local.unlink(missing_ok=True)


# ============================================================================
# Wrapped Claude Code Runner
# ============================================================================


def run_claude_code_in_container(
    task: dict,
    container,
    timeout: int = 1800,
    poll_interval: int = 10,
    skills: str | None = None,
    repo_name: str | None = None,
) -> ClaudeCodeResult:
    """
    Run Claude Code with skills installed as a proper SKILL.md.

    This wraps the original run_claude_code_in_container: if skills are
    provided, they are first installed as .claude/skills/<repo>/SKILL.md,
    then the original runner is called with skills=None (so it skips
    CLAUDE.md creation).

    Args:
        task: Task instance dict
        container: Running Docker container
        timeout: Maximum time to wait (seconds)
        poll_interval: How often to check for completion (seconds)
        skills: Optional raw skills content
        repo_name: Repository name for the skill directory

    Returns:
        ClaudeCodeResult with patch and status
    """
    if skills:
        effective_repo = repo_name or task.get("repo", "").split("/")[-1] or "default"
        install_skill_in_container(container, skills, effective_repo)

    # Delegate to original runner with skills=None (skill already installed)
    return _run_claude_code_in_container(task, container, timeout=timeout, poll_interval=poll_interval, skills=None)


# ============================================================================
# Evaluation Functions (thin overrides that thread repo_name through)
# ============================================================================


def evaluate_single_task(
    task: dict,
    claude_image: str,
    timeout: int = 1800,
    verify_timeout: int = 300,
    skills: str | None = None,
    repo_name: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate a single task with Claude Code (skill-aware version).

    Identical to evaluate_claude_code.evaluate_single_task except that
    skills are installed as .claude/skills/<repo>/SKILL.md.
    """
    instance_id = task[KEY_INSTANCE_ID]
    container = None

    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        return {
            "instance_id": instance_id,
            "claude_success": False,
            "test_passed": False,
            "patch": "",
            "claude_output": "",
            "test_output": "",
            "duration_seconds": 0,
            "error": f"Docker connection failed: {e!s}",
        }

    try:
        container_name = f"claude-code-{instance_id}.{uuid.uuid4().hex[:8]}"
        container = client.containers.create(
            image=claude_image,
            name=container_name,
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",
            platform="linux/x86_64",
            mem_limit="10g",
        )
        container.start()

        val = container.exec_run(
            f"git checkout {instance_id}",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER,
        )
        if val.exit_code != 0:
            raise RuntimeError(f"Failed to checkout {instance_id}: {val.output.decode()}")

        # Uses our skill-aware run_claude_code_in_container
        result = run_claude_code_in_container(task, container, timeout=timeout, skills=skills, repo_name=repo_name)

        passed = False
        test_output = ""
        if result.patch.strip():
            passed, test_output = verify_patch_with_harness(task, result.patch, timeout=verify_timeout)
        else:
            test_output = "No patch generated"

        return {
            "instance_id": instance_id,
            "claude_success": result.success,
            "test_passed": passed,
            "patch": result.patch,
            "claude_output": result.output[:5000],
            "test_output": test_output[:5000],
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        }

    except Exception as e:
        import traceback

        return {
            "instance_id": instance_id,
            "claude_success": False,
            "test_passed": False,
            "patch": "",
            "claude_output": "",
            "test_output": "",
            "duration_seconds": 0,
            "error": f"{e!s}\n{traceback.format_exc()}",
        }

    finally:
        if container:
            try:
                container.stop()
                container.remove()
            except Exception:
                pass


def evaluate_testset(
    test_data: list[dict],
    output_dir: Path,
    n_workers: int = 4,
    timeout: int = 1800,
    verify_timeout: int = 300,
    skills: str | None = None,
    run_config: dict[str, Any] | None = None,
    repo_name: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate all tasks in testset (skill-aware version).

    Identical to evaluate_claude_code.evaluate_testset except that skills
    are installed as proper Claude Code skills.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure Claude Code images exist for all unique base images
    print("\n" + "=" * 70)
    print("Preparing Claude Code Docker images...")
    print("=" * 70)

    image_cache: dict[str, str] = {}
    for task in test_data:
        base_image = get_base_image_name(task)
        if base_image not in image_cache:
            claude_image = ensure_claude_code_image(task)
            image_cache[base_image] = claude_image

    print(f"Prepared {len(image_cache)} Claude Code image(s)")

    # Evaluate tasks
    print("\n" + "=" * 70)
    print(f"Evaluating {len(test_data)} tasks with Claude Code...")
    print(f"Workers: {n_workers}, Timeout: {timeout}s")
    if skills:
        print(f"Skills: installed as .claude/skills/{repo_name or '<auto>'}/SKILL.md")
    print("=" * 70 + "\n")

    results: list[dict[str, Any]] = []

    def process_task(task):
        base_image = get_base_image_name(task)
        claude_image = image_cache[base_image]
        return evaluate_single_task(
            task,
            claude_image,
            timeout,
            verify_timeout,
            skills=skills,
            repo_name=repo_name,
        )

    if n_workers <= 1:
        for i, task in enumerate(test_data):
            print(f"[{i + 1}/{len(test_data)}] {task[KEY_INSTANCE_ID][:50]}...")
            result = process_task(task)
            results.append(result)

            status = "✓ PASS" if result["test_passed"] else "✗ FAIL"
            print(f"  {status} ({result['duration_seconds']:.1f}s)")

            with open(output_dir / "results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_task, task): task for task in test_data}

            for i, future in enumerate(as_completed(futures)):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "instance_id": task[KEY_INSTANCE_ID],
                        "claude_success": False,
                        "test_passed": False,
                        "error": str(e),
                    }

                results.append(result)

                status = "✓ PASS" if result.get("test_passed") else "✗ FAIL"
                print(f"[{i + 1}/{len(test_data)}] {result['instance_id'][:50]}: {status}")

                with open(output_dir / "results.jsonl", "a") as f:
                    f.write(json.dumps(result) + "\n")

    # Compute summary
    passed = sum(1 for r in results if r.get("test_passed"))
    claude_success = sum(1 for r in results if r.get("claude_success"))
    total = len(results)

    summary = {
        "run_config": run_config or {},
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total > 0 else 0,
        "claude_generated_patches": claude_success,
        "claude_patch_rate": claude_success / total if total > 0 else 0,
        "avg_duration": (sum(r.get("duration_seconds", 0) for r in results) / total if total > 0 else 0),
        "results": results,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total tasks: {total}")
    print(f"Claude generated patches: {claude_success}/{total} ({summary['claude_patch_rate']:.1%})")
    print(f"Tests passed: {passed}/{total} ({summary['pass_rate']:.1%})")
    print(f"Average duration: {summary['avg_duration']:.1f}s")
    print(f"Results saved to: {output_dir}")

    return summary


# ============================================================================
# CLI (same arguments as evaluate_claude_code.py)
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate testset with Claude Code (proper Skills variant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script is identical to evaluate_claude_code.py except that --use-skills
installs skills as .claude/skills/<repo>/SKILL.md (with YAML frontmatter)
instead of a plain CLAUDE.md file.

See: https://code.claude.com/docs/en/skills

Examples:
  # Evaluate with proper Claude Code skills
  python -m src.evaluate.claude_code_skills --config gepa_results/logs/run_xxx/config.json --use-skills

  # Evaluate without skills (identical to evaluate_claude_code.py)
  python -m src.evaluate.claude_code_skills --config gepa_results/logs/run_xxx/config.json
""",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.json from a GEPA run",
    )
    parser.add_argument(
        "--extract-ids",
        type=str,
        default=None,
        help="Extract test IDs to this file and exit",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: gepa_results/claude_code/<run_id>)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout for Claude Code in seconds (default: 1800)",
    )
    parser.add_argument(
        "--verify-timeout",
        type=int,
        default=300,
        help="Timeout for test verification in seconds (default: 300)",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Load test IDs from file instead of reconstructing from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only prepare images and show what would be run, don't execute",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate (for testing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sonnet",
        choices=["sonnet", "opus", "haiku", "default"],
        help="Claude Code model to use (default: sonnet = Claude Sonnet 4.5)",
    )
    parser.add_argument(
        "--use-skills",
        action="store_true",
        help="Load best_skills.txt and install as .claude/skills/<repo>/SKILL.md in the repo",
    )

    args = parser.parse_args()

    # Verify Docker connection
    try:
        client = docker.from_env()
        client.ping()
        print(f"Docker connection OK (DOCKER_HOST={os.environ.get('DOCKER_HOST', 'default')})")
    except Exception as e:
        print(f"ERROR: Cannot connect to Docker: {e}")
        print("Please ensure Docker is running:")
        print("  Rootless: systemctl --user start docker")
        print("  Standard: sudo systemctl start docker")
        sys.exit(1)

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    run_id = config.get("run_id", Path(args.config).parent.name)
    repo = config["repo"]

    # Set model environment variable for Claude Code
    os.environ["CLAUDE_CODE_MODEL"] = args.model
    print(f"Using Claude Code model: {args.model}")

    # Load testset
    if args.ids_file:
        print(f"Loading test IDs from {args.ids_file}")
        test_ids = load_test_ids(args.ids_file)
        test_data = load_testset_by_ids(test_ids, repo)
        print(f"Loaded {len(test_data)} tasks from IDs file")
    else:
        test_data, _ = load_testset_from_config(args.config)

    # Extract IDs mode
    if args.extract_ids:
        save_test_ids(test_data, args.extract_ids)
        return

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("gepa_results/claude_code") / f"{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Apply limit
    if args.limit and args.limit < len(test_data):
        print(f"Limiting to first {args.limit} tasks (out of {len(test_data)})")
        test_data = test_data[: args.limit]

    # Save test IDs for reference
    output_dir.mkdir(parents=True, exist_ok=True)
    save_test_ids(test_data, str(output_dir / "test_ids.txt"))

    # Save config copy
    with open(output_dir / "source_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load skills from config directory if requested
    config_dir = Path(args.config).parent
    skills = None
    repo_name = repo  # e.g. "blevesearch__bleve"
    if args.use_skills:
        skills_path = config_dir / "prompts" / "best_skills.txt"
        if skills_path.exists():
            skills = skills_path.read_text()
            print(f"Loaded skills from {skills_path} ({len(skills)} chars)")
            print(f"Will install as .claude/skills/{repo_name}/SKILL.md (proper Claude Code skill)")
            # Save copy of raw skills
            with open(output_dir / "skills_raw.txt", "w") as f:
                f.write(skills)
            # Also save the formatted SKILL.md for reference
            with open(output_dir / "SKILL.md", "w") as f:
                f.write(format_skill_md(skills, repo_name))
        else:
            print(f"WARNING: Skills file not found: {skills_path}")

    # Dry run mode
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - Preparing images only")
        print("=" * 70)

        image_cache = {}
        for task in test_data:
            base_image = get_base_image_name(task)
            if base_image not in image_cache:
                claude_image = ensure_claude_code_image(task)
                image_cache[base_image] = claude_image

        print(f"\nPrepared {len(image_cache)} Claude Code image(s)")
        print(f"Would evaluate {len(test_data)} tasks")
        if skills:
            print(f"Skills mode: .claude/skills/{repo_name}/SKILL.md")
        print(f"Test IDs saved to: {output_dir / 'test_ids.txt'}")

        return {
            "dry_run": True,
            "num_tasks": len(test_data),
            "images": list(image_cache.values()),
        }

    # Build run configuration metadata
    run_config = {
        "model": args.model,
        "use_skills": args.use_skills,
        "skills_format": "claude_code_skill" if args.use_skills else None,
        "skills_path": (str(config_dir / "prompts" / "best_skills.txt") if args.use_skills else None),
        "config_path": str(Path(args.config).resolve()),
        "source_run_id": run_id,
        "repo": repo,
        "workers": args.workers,
        "timeout": args.timeout,
        "verify_timeout": args.verify_timeout,
        "num_tasks": len(test_data),
        "timestamp": datetime.now().isoformat(),
    }

    # Run evaluation
    summary = evaluate_testset(
        test_data=test_data,
        output_dir=output_dir,
        n_workers=args.workers,
        timeout=args.timeout,
        verify_timeout=args.verify_timeout,
        skills=skills,
        run_config=run_config,
        repo_name=repo_name,
    )

    return summary


if __name__ == "__main__":
    main()
