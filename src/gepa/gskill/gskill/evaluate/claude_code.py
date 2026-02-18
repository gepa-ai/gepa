"""
Evaluate testset with Claude Code.

This script:
1. Reads a config JSON and extracts testset IDs using the exact same data loader
2. Outputs test IDs to a file
3. Evaluates problems using Claude Code by:
   - Finding the corresponding Docker image
   - Installing Claude Code inside and caching the image
   - Running `claude -p "task"` inside the container
   - Monitoring completion
   - Running the original harness to check correctness
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import docker
from datasets import load_dataset
from dotenv import load_dotenv
from gskill.swe_harness import SWEHarness

# Load environment variables
load_dotenv()


# Setup Docker host for rootless Docker (same as swe_harness.py)
def setup_docker_host():
    """Set DOCKER_HOST for rootless Docker if needed."""
    if not os.getenv("DOCKER_HOST"):
        uid = os.getuid()
        xdg_runtime = os.getenv("XDG_RUNTIME_DIR")

        if xdg_runtime:
            rootless_socket = f"unix://{xdg_runtime}/docker.sock"
        else:
            rootless_socket = f"unix:///run/user/{uid}/docker.sock"

        # Check if rootless socket exists
        socket_path = rootless_socket.replace("unix://", "")
        if Path(socket_path).exists():
            os.environ["DOCKER_HOST"] = rootless_socket
            print(f"Using rootless Docker: {rootless_socket}")


setup_docker_host()

# Import SWE-smith utilities
from swebench.harness.constants import (
    DOCKER_USER,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
)
from swesmith.profiles import registry

# ============================================================================
# Data Loading (same as train_optimize_anything.py)
# ============================================================================


def load_and_split_data(repo: str, train_size: int, val_size: int, test_size: int, seed: int = 42):
    """
    Load data directly from HuggingFace and split into train/val/test.

    This is the EXACT same function from train_optimize_anything.py to ensure
    identical data splits.

    Args:
        repo: Repository name to filter (e.g., "pygments__pygments" or "swesmith/pygments__pygments")
        train_size: Number of training examples
        val_size: Number of validation examples
        test_size: Number of test examples
        seed: Random seed for shuffling

    Returns:
        (train_data, val_data, test_data) tuple
    """
    total_needed = train_size + val_size + test_size
    print(f"Loading {repo} data from HuggingFace SWE-smith (need {total_needed})...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train")

    # Normalize repo name for matching
    # Handle both "pygments__pygments" and "swesmith/pygments__pygments"
    repo_pattern = repo if "/" in repo else f"swesmith/{repo}"

    all_data = []
    for item in ds:
        repo_name = item.get("repo", "")
        if repo_pattern in repo_name:
            task_dict = dict(item)
            all_data.append(task_dict)
            if len(all_data) >= total_needed:
                break

    print(f"Loaded {len(all_data)} {repo} examples")

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(all_data)

    # Calculate split sizes
    total_needed = train_size + val_size + test_size
    if len(all_data) < total_needed:
        print(f"WARNING: Only {len(all_data)} examples available, need {total_needed}")
        # Scale down proportionally
        ratio = len(all_data) / total_needed
        train_size = int(train_size * ratio)
        val_size = int(val_size * ratio)
        test_size = len(all_data) - train_size - val_size

    # Split
    train_data = all_data[:train_size]
    val_data = all_data[train_size : train_size + val_size]
    test_data = all_data[train_size + val_size : train_size + val_size + test_size]

    test_data_with_ps = []
    skipped_empty_ps = 0
    for task in test_data:
        ps = task.get("problem_statement", "")
        if not ps or not ps.strip():
            skipped_empty_ps += 1
        else:
            test_data_with_ps.append(task)

    print(f"Skipped {skipped_empty_ps} tasks with empty problem_statement")
    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    return train_data, val_data, test_data_with_ps


def load_testset_from_config(config_path: str) -> tuple[list[dict], dict]:
    """
    Load testset using config JSON parameters.

    Args:
        config_path: Path to config.json file

    Returns:
        (test_data, config) tuple
    """
    with open(config_path) as f:
        config = json.load(f)

    repo = config["repo"]
    train_size = config["train_size"]
    val_size = config["val_size"]
    test_size = config["test_size"]
    seed = config.get("seed", 42)

    print(f"Loading data with config: repo={repo}, seed={seed}")
    print(f"  train_size={train_size}, val_size={val_size}, test_size={test_size}")

    _, _, test_data = load_and_split_data(repo, train_size, val_size, test_size, seed)

    print(f"Loaded {len(test_data)} test tasks (with valid problem_statement)")

    return test_data, config


def save_test_ids(test_data: list[dict], output_path: str):
    """Save test instance IDs to a file."""
    ids = [task["instance_id"] for task in test_data]
    with open(output_path, "w") as f:
        for instance_id in ids:
            f.write(f"{instance_id}\n")
    print(f"Saved {len(ids)} test IDs to {output_path}")
    return ids


def load_test_ids(ids_file: str) -> list[str]:
    """Load test IDs from a file."""
    with open(ids_file) as f:
        return [line.strip() for line in f if line.strip()]


def load_testset_by_ids(test_ids: list[str], repo: str) -> list[dict]:
    """
    Load tasks from HuggingFace dataset by their instance IDs.

    Args:
        test_ids: List of instance IDs to load
        repo: Repository name for filtering

    Returns:
        List of task dicts
    """
    print(f"Loading {len(test_ids)} tasks by ID from HuggingFace...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train")

    # Normalize repo name

    # Create a set for fast lookup
    id_set = set(test_ids)

    tasks = []
    skipped_empty_ps = []
    for item in ds:
        if item.get("instance_id") in id_set:
            task_dict = dict(item)
            # Skip tasks with empty problem_statement - they can't be solved
            ps = task_dict.get("problem_statement", "")
            if not ps or not ps.strip():
                skipped_empty_ps.append(task_dict["instance_id"])
                continue  # Skip this task
            tasks.append(task_dict)
            if len(tasks) >= len(test_ids):
                break

    if skipped_empty_ps:
        print(f"  Skipped {len(skipped_empty_ps)} tasks with empty problem_statement, ids: {skipped_empty_ps}")

    print(
        f"Loaded {len(tasks)} tasks with valid problem_statement (requested {len(test_ids)}, skipped {len(skipped_empty_ps)} empty)"
    )

    # Reorder to match input order
    id_to_task = {t["instance_id"]: t for t in tasks}
    ordered_tasks = [id_to_task[tid] for tid in test_ids if tid in id_to_task]

    return ordered_tasks


# ============================================================================
# Claude Code Docker Image Management
# ============================================================================


def get_base_image_name(task: dict) -> str:
    """Get the base Docker image name for a task.

    Uses RepoProfile.image_name (from swesmith registry) instead of
    task["image_name"] because the locally-built images use a different
    org name (swebench/) than the HuggingFace dataset (jyangballin/).
    """
    # Always use RepoProfile to get the correct local image name
    rp = registry.get_from_inst(task)
    return rp.image_name


def get_claude_code_image_name(base_image: str) -> str:
    """Get the Claude Code enhanced image name."""
    # Add -claude-code suffix
    return f"{base_image}-claude-code"


def check_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(["docker", "image", "inspect", image_name], capture_output=True, check=False)
        return result.returncode == 0
    except Exception:
        return False


def install_claude_code_in_image(base_image: str, claude_code_image: str) -> bool:
    """
    Create a new Docker image with Claude Code installed.

    This creates a cached image so Claude Code only needs to be installed once
    per base image. Uses the locally-built image (from swesmith.build_repo.create_images)
    which has proper git branches.

    Args:
        base_image: The original SWE-smith image (locally built)
        claude_code_image: Name for the new image with Claude Code

    Returns:
        True if successful, False otherwise
    """
    print(f"Installing Claude Code into image: {base_image} -> {claude_code_image}")

    # Check if base image exists locally - DO NOT PULL (would overwrite local build)
    if not check_image_exists(base_image):
        print(f"  ERROR: Base image {base_image} not found locally!")
        print("  Please build it first with:")
        print(
            "    DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock python -m swesmith.build_repo.create_images --repos <repo> --force -y"
        )
        return False

    print(f"  Using locally-built base image: {base_image}")

    # Create Dockerfile for Claude Code installation
    # Note: We install Node.js 20 and Claude Code globally
    dockerfile_content = f"""FROM {base_image}

# Switch to root to install packages
USER root

# Install dependencies and Node.js 20 (required for Claude Code)
RUN apt-get update && \\
    apt-get install -y curl gnupg ca-certificates && \\
    mkdir -p /etc/apt/keyrings && \\
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \\
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \\
    apt-get update && \\
    apt-get install -y nodejs && \\
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Switch back to non-root user
USER {DOCKER_USER}

# Verify installation
RUN claude --version || echo "Claude Code installed"
"""

    # Write Dockerfile
    tmp_dir = Path(f"/tmp/claude_code_docker_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dockerfile_path = tmp_dir / "Dockerfile"
    dockerfile_path.write_text(dockerfile_content)

    try:
        # Build the image
        print("  Building Claude Code image...")
        result = subprocess.run(
            ["docker", "build", "-t", claude_code_image, "-f", str(dockerfile_path), str(tmp_dir)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for build
        )

        if result.returncode != 0:
            print(f"  ERROR building image: {result.stderr}")
            print(f"  STDOUT: {result.stdout}")
            return False

        print(f"  Successfully created image: {claude_code_image}")
        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: Docker build timed out after 600 seconds")
        return False

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def ensure_claude_code_image(task: dict) -> str:
    """
    Ensure Claude Code image exists, creating it if necessary.

    Uses the locally-built base image (from swesmith.build_repo.create_images).

    Args:
        task: Task instance dict

    Returns:
        Name of the Claude Code enabled image
    """
    base_image = get_base_image_name(task)
    claude_image = get_claude_code_image_name(base_image)

    if check_image_exists(claude_image):
        print(f"  Claude Code image already exists: {claude_image}")
        return claude_image

    # Create the image from locally-built base
    if not install_claude_code_in_image(base_image, claude_image):
        raise RuntimeError(f"Failed to create Claude Code image from {base_image}")

    return claude_image


# ============================================================================
# Claude Code Execution
# ============================================================================


@dataclass
class ClaudeCodeResult:
    """Result from running Claude Code on a task."""

    instance_id: str
    success: bool
    patch: str
    output: str
    duration_seconds: float
    error: str | None = None


def run_claude_code_in_container(
    task: dict,
    container: docker.models.containers.Container,
    timeout: int = 1800,  # 30 minutes default
    poll_interval: int = 10,
    skills: str | None = None,  # Optional skills to copy as CLAUDE.md
) -> ClaudeCodeResult:
    """
    Run Claude Code inside a container and monitor completion.

    Args:
        task: Task instance dict
        container: Running Docker container
        timeout: Maximum time to wait (seconds)
        poll_interval: How often to check for completion (seconds)
        skills: Optional skills content to copy as CLAUDE.md in /testbed

    Returns:
        ClaudeCodeResult with patch and status
    """
    from swebench.harness.docker_utils import copy_to_container

    instance_id = task[KEY_INSTANCE_ID]
    problem_statement = task.get("problem_statement", "")

    # Verify we have a problem statement
    if not problem_statement or not problem_statement.strip():
        return ClaudeCodeResult(
            instance_id=instance_id,
            success=False,
            patch="",
            output="",
            duration_seconds=0,
            error="Task has no problem statement (dataset issue)",
        )

    start_time = time.time()

    # Get ANTHROPIC_API_KEY from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ClaudeCodeResult(
            instance_id=instance_id,
            success=False,
            patch="",
            output="",
            duration_seconds=0,
            error="ANTHROPIC_API_KEY not set",
        )

    # Write problem statement to a temp file and copy to container
    # This avoids shell escaping issues
    problem_file_local = Path(f"/tmp/problem_{uuid.uuid4().hex[:8]}.txt")
    problem_file_container = Path("/tmp/problem_statement.txt")

    try:
        problem_file_local.write_text(problem_statement)
        local_size = problem_file_local.stat().st_size
        if local_size == 0:
            print(f"  ERROR: Local problem file is empty for {instance_id}!")

        copy_to_container(container, problem_file_local, problem_file_container)
        # Sync filesystem and ensure file is readable
        container.exec_run("sync", user="root")
        container.exec_run(f"chmod 644 {problem_file_container}", user="root")

        # Verify the copy worked with retries
        for attempt in range(3):
            # Use shell to handle redirection, and extract just the number
            verify_result = container.exec_run(["sh", "-c", f"cat {problem_file_container} | wc -c"], user="root")
            container_size = verify_result.output.decode().strip() if verify_result.output else "0"

            if container_size != "0" and verify_result.exit_code == 0:
                break

            # File is missing or empty - wait and retry
            if attempt < 2:
                time.sleep(0.5)
                copy_to_container(container, problem_file_local, problem_file_container)
                container.exec_run("sync", user="root")
                container.exec_run(f"chmod 644 {problem_file_container}", user="root")
        else:
            print(
                f"  WARNING: Problem file copy may have failed for {instance_id} (local={local_size}, container={container_size})"
            )
    finally:
        problem_file_local.unlink(missing_ok=True)

    # Copy skills as CLAUDE.md if provided
    if skills:
        skills_file_local = Path(f"/tmp/skills_{uuid.uuid4().hex[:8]}.md")
        skills_file_container = Path("/testbed/CLAUDE.md")
        try:
            skills_file_local.write_text(skills)
            copy_to_container(container, skills_file_local, skills_file_container)
            container.exec_run("sync", user="root")
            container.exec_run(f"chmod 644 {skills_file_container}", user="root")
            print(f"  Copied skills to CLAUDE.md ({len(skills)} chars)")
        finally:
            skills_file_local.unlink(missing_ok=True)

    # Run Claude Code with the problem statement from file
    # Use -p flag to pass the prompt (print mode - non-interactive)
    # Note: --dangerously-skip-permissions cannot be used as root, so we create
    # a non-root user and run Claude Code as that user
    # Default model is sonnet (Claude Sonnet 4.5)
    model = os.environ.get("CLAUDE_CODE_MODEL", "sonnet")

    # Create a runner script to avoid shell escaping issues
    # Use stdin to pass the prompt - avoids any escaping problems with special characters
    runner_script = f"""#!/bin/bash

# Create non-root user if doesn't exist
id -u claude_user >/dev/null 2>&1 || useradd -m -s /bin/bash claude_user

# Setup permissions
chown -R claude_user:claude_user {DOCKER_WORKDIR} 2>/dev/null || true

# Copy problem statement
cp {problem_file_container} /home/claude_user/problem.txt
chown claude_user:claude_user /home/claude_user/problem.txt

# Verify problem file exists and has content
if [ ! -s /home/claude_user/problem.txt ]; then
    echo "ERROR: Problem statement file is empty or missing"
    echo "Source file status:"
    ls -la {problem_file_container} 2>&1 || echo "Source not found"
    exit 1
fi

# Create a wrapper script for claude_user that reads from files
cat > /home/claude_user/run_claude.sh << 'INNEREOF'
#!/bin/bash
cd {DOCKER_WORKDIR}
export ANTHROPIC_API_KEY=$(cat /home/claude_user/.anthropic_key)
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
cat /home/claude_user/problem.txt | claude --print --model {model} --dangerously-skip-permissions
INNEREOF

# Write API key to file (after creating wrapper to avoid key in heredoc)
# Use single quotes and escape any embedded single quotes
printf '%s' '{api_key}' > /home/claude_user/.anthropic_key
chmod 600 /home/claude_user/.anthropic_key
chown claude_user:claude_user /home/claude_user/.anthropic_key
chmod +x /home/claude_user/run_claude.sh
chown claude_user:claude_user /home/claude_user/run_claude.sh

# Run as claude_user
su - claude_user -c "/home/claude_user/run_claude.sh"
"""

    # Write runner script to container
    runner_script_path = "/tmp/run_claude.sh"
    runner_script_local = Path(f"/tmp/run_claude_{uuid.uuid4().hex[:8]}.sh")
    try:
        runner_script_local.write_text(runner_script)
        copy_to_container(container, runner_script_local, Path(runner_script_path))
        container.exec_run(f"chmod +x {runner_script_path}", user="root")
    finally:
        runner_script_local.unlink(missing_ok=True)

    claude_cmd = f"bash {runner_script_path}"

    print(f"  Running Claude Code for {instance_id}...")

    # Execute Claude Code with timeout
    # Note: Docker SDK doesn't support timeout directly on exec_run
    # We'll use a background thread approach
    import threading

    result_holder = {"output": None, "exit_code": None}

    def run_exec():
        try:
            exec_result = container.exec_run(
                ["bash", "-c", claude_cmd],
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
                demux=True,
                stream=False,
            )
            stdout, stderr = exec_result.output if exec_result.output else (b"", b"")
            result_holder["output"] = (stdout or b"").decode() + (stderr or b"").decode()
            result_holder["exit_code"] = exec_result.exit_code
        except Exception as e:
            result_holder["output"] = f"Execution error: {e!s}"
            result_holder["exit_code"] = -1

    exec_thread = threading.Thread(target=run_exec)
    exec_thread.start()
    exec_thread.join(timeout=timeout)

    duration = time.time() - start_time

    if exec_thread.is_alive():
        # Timed out - try to kill the claude process
        container.exec_run("pkill -f claude", user="root")
        exec_thread.join(timeout=10)  # Give it 10 more seconds to clean up

        return ClaudeCodeResult(
            instance_id=instance_id,
            success=False,
            patch="",
            output=result_holder.get("output", "Timeout - no output"),
            duration_seconds=duration,
            error=f"Timeout after {timeout} seconds",
        )

    output = result_holder.get("output", "")

    # Get the patch (git diff)
    # First, add the workdir to git's safe.directory (needed because ownership changed to claude_user)
    container.exec_run(
        f"git config --global --add safe.directory {DOCKER_WORKDIR}",
        user=DOCKER_USER,
    )
    diff_result = container.exec_run(
        "git diff",
        workdir=DOCKER_WORKDIR,
        user=DOCKER_USER,
    )
    patch = diff_result.output.decode() if diff_result.output else ""

    print(f"  Claude Code finished in {duration:.1f}s, patch size: {len(patch)} chars")

    return ClaudeCodeResult(
        instance_id=instance_id,
        success=len(patch.strip()) > 0,
        patch=patch,
        output=output,
        duration_seconds=duration,
    )


# ============================================================================
# Verification (using existing SWEHarness - same as train_optimize_anything)
# ============================================================================


def verify_patch_with_harness(task: dict, patch: str, timeout: int = 300) -> tuple[bool, str]:
    """
    Verify a patch using the same SWEHarness as train_optimize_anything.

    This ensures identical verification logic.

    Args:
        task: Task instance dict
        patch: The git diff patch to test
        timeout: Test timeout in seconds

    Returns:
        (passed, test_output) tuple
    """
    harness = SWEHarness()

    try:
        # Setup the task (this creates the verification container)
        harness.setup_task(task)

        # Verify using the exact same logic as swe_fitness_fn.py
        has_patch = len(patch.strip()) > 0
        passed = False
        test_output = ""

        if not has_patch:
            passed = False
            test_output = "No patch to test."
        else:
            # First test FAIL_TO_PASS only
            f2p_passed, f2p_output = harness.verify_with_patch(patch, f2p_only=True, timeout=timeout)
            test_output = f"=== FAIL_TO_PASS TESTS ===\n{f2p_output}"

            if not f2p_passed:
                passed = False
            else:
                # PASS_TO_PASS Tests (Regression Check) - run full test suite
                pass_to_pass = task.get("PASS_TO_PASS", [])

                if pass_to_pass:
                    # Run full test (includes both f2p and p2p)
                    p2p_passed, p2p_output = harness.verify_with_patch(patch, f2p_only=False, timeout=timeout)
                    test_output += f"\n\n=== FULL TEST SUITE ===\n{p2p_output}"

                    if not p2p_passed:
                        passed = False
                    else:
                        passed = True
                else:
                    passed = True

        return passed, test_output

    except Exception as e:
        import traceback

        return False, f"Verification error: {e}\n{traceback.format_exc()}"

    finally:
        harness.cleanup()


# ============================================================================
# Main Evaluation Loop
# ============================================================================


def evaluate_single_task(
    task: dict,
    claude_image: str,
    timeout: int = 1800,
    verify_timeout: int = 300,
    skills: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate a single task with Claude Code.

    Args:
        task: Task instance dict
        claude_image: Name of Claude Code enabled Docker image
        timeout: Timeout for Claude Code execution
        verify_timeout: Timeout for test verification
        skills: Optional skills content to copy as CLAUDE.md

    Returns:
        Result dict with all metrics
    """
    instance_id = task[KEY_INSTANCE_ID]
    container = None

    # Create Docker client with retry for connection issues
    try:
        client = docker.from_env()
        client.ping()  # Test connection
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
        # Create container from Claude Code image
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

        # Checkout the instance branch
        # The locally-built image (from swesmith.build_repo.create_images) has all branches
        val = container.exec_run(
            f"git checkout {instance_id}",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER,
        )
        if val.exit_code != 0:
            raise RuntimeError(f"Failed to checkout {instance_id}: {val.output.decode()}")

        # Run Claude Code
        result = run_claude_code_in_container(task, container, timeout=timeout, skills=skills)

        # Verify the patch
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
            "claude_output": result.output[:5000],  # Truncate for storage
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
) -> dict[str, Any]:
    """
    Evaluate all tasks in testset with Claude Code.

    Args:
        test_data: List of task instances
        output_dir: Directory to save results
        n_workers: Number of parallel workers
        timeout: Timeout for Claude Code execution
        verify_timeout: Timeout for test verification
        skills: Optional skills content to copy as CLAUDE.md
        run_config: Run configuration metadata for summary

    Returns:
        Summary dict with all results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure Claude Code images exist for all unique base images
    print("\n" + "=" * 70)
    print("Preparing Claude Code Docker images...")
    print("=" * 70)

    image_cache = {}  # base_image -> claude_image
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
    print("=" * 70 + "\n")

    results = []

    def process_task(task):
        base_image = get_base_image_name(task)
        claude_image = image_cache[base_image]
        return evaluate_single_task(task, claude_image, timeout, verify_timeout, skills=skills)

    if n_workers <= 1:
        # Sequential execution
        for i, task in enumerate(test_data):
            print(f"[{i + 1}/{len(test_data)}] {task[KEY_INSTANCE_ID][:50]}...")
            result = process_task(task)
            results.append(result)

            status = "✓ PASS" if result["test_passed"] else "✗ FAIL"
            print(f"  {status} ({result['duration_seconds']:.1f}s)")

            # Save intermediate results
            with open(output_dir / "results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
    else:
        # Parallel execution
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

                # Save intermediate results
                with open(output_dir / "results.jsonl", "a") as f:
                    f.write(json.dumps(result) + "\n")

    # Compute summary
    passed = sum(1 for r in results if r.get("test_passed"))
    claude_success = sum(1 for r in results if r.get("claude_success"))
    total = len(results)

    summary = {
        # Run configuration (at top for easy access)
        "run_config": run_config or {},
        # Results summary
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total > 0 else 0,
        "claude_generated_patches": claude_success,
        "claude_patch_rate": claude_success / total if total > 0 else 0,
        "avg_duration": sum(r.get("duration_seconds", 0) for r in results) / total if total > 0 else 0,
        "results": results,
    }

    # Save summary
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
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate testset with Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract test IDs from config
  python evaluate_claude_code.py --config gepa_results/logs/run_xxx/config.json --extract-ids test_ids.txt

  # Evaluate using config (will reconstruct testset)
  python evaluate_claude_code.py --config gepa_results/logs/run_xxx/config.json --output results/

  # Evaluate with custom parameters
  python evaluate_claude_code.py --config gepa_results/logs/run_xxx/config.json --workers 4 --timeout 1800
""",
    )

    parser.add_argument("--config", type=str, required=True, help="Path to config.json from a GEPA run")
    parser.add_argument("--extract-ids", type=str, default=None, help="Extract test IDs to this file and exit")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: gepa_results/claude_code/<run_id>)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout for Claude Code in seconds (default: 1800)")
    parser.add_argument(
        "--verify-timeout", type=int, default=300, help="Timeout for test verification in seconds (default: 300)"
    )
    parser.add_argument(
        "--ids-file", type=str, default=None, help="Load test IDs from file instead of reconstructing from config"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only prepare images and show what would be run, don't execute"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to evaluate (for testing)")
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
        help="Load best_skills.txt from config directory and copy as CLAUDE.md in the repo",
    )

    args = parser.parse_args()

    # Verify Docker connection first
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
        # Load IDs from file and fetch corresponding tasks
        print(f"Loading test IDs from {args.ids_file}")
        test_ids = load_test_ids(args.ids_file)
        test_data = load_testset_by_ids(test_ids, repo)
        print(f"Loaded {len(test_data)} tasks from IDs file")
    else:
        # Reconstruct testset from config
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

    # Apply limit if specified
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
    if args.use_skills:
        skills_path = config_dir / "prompts" / "best_skills.txt"
        if skills_path.exists():
            skills = skills_path.read_text()
            print(f"Loaded skills from {skills_path} ({len(skills)} chars)")
            # Save copy of skills
            with open(output_dir / "skills.txt", "w") as f:
                f.write(skills)
        else:
            print(f"WARNING: Skills file not found: {skills_path}")

    # Dry run mode - just prepare images
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
        print(f"Test IDs saved to: {output_dir / 'test_ids.txt'}")

        return {"dry_run": True, "num_tasks": len(test_data), "images": list(image_cache.values())}

    # Build run configuration metadata
    run_config = {
        "model": args.model,
        "use_skills": args.use_skills,
        "skills_path": str(config_dir / "prompts" / "best_skills.txt") if args.use_skills else None,
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
    )

    return summary


if __name__ == "__main__":
    main()
