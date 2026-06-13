"""Prepare a GSO benchmark instance for optimization."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def load_instance(instance_id: str) -> dict:
    """Load a single GSO instance from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("gso-bench/gso", split="test")
    for row in ds:
        if row["instance_id"] == instance_id:
            return dict(row)

    available = [row["instance_id"] for row in ds]
    raise ValueError(
        f"Instance '{instance_id}' not found. Available instances ({len(available)}):\n"
        + "\n".join(available[:20])
        + ("\n..." if len(available) > 20 else "")
    )


def list_instances() -> list[str]:
    """List all available GSO instance IDs."""
    from datasets import load_dataset

    ds = load_dataset("gso-bench/gso", split="test")
    return [row["instance_id"] for row in ds]


def prepare_repo(instance: dict, work_dir: str = "gso_workdir") -> str:
    """Clone the repository and checkout the base commit.

    Args:
        instance: GSO instance dict (from load_instance).
        work_dir: Directory to clone repos into.

    Returns:
        Path to the cloned repository.
    """
    repo_name = instance["repo"]  # e.g. "owner/repo"
    base_commit = instance["base_commit"]
    instance_id = instance["instance_id"]

    # Create work directory
    os.makedirs(work_dir, exist_ok=True)

    # Sanitize repo name for directory
    repo_dir_name = instance_id.replace("/", "__")
    repo_path = os.path.join(os.path.abspath(work_dir), repo_dir_name)

    if os.path.exists(repo_path):
        logger.info(f"Repo already exists at {repo_path}, resetting to base commit")
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "clean", "-fdx"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
    else:
        logger.info(f"Cloning {repo_name} to {repo_path}")
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo_name}.git", repo_path],
            check=True,
        )
        logger.info(f"Checking out base commit {base_commit}")
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

    # Create a named branch at the base commit for GEPA
    subprocess.run(
        ["git", "checkout", "-B", "base"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    logger.info(f"Repo prepared at {repo_path} on branch 'base' ({base_commit[:8]})")
    return repo_path


def get_docker_image(instance: dict, dockerhub_prefix: str = "gsobench/gso") -> str:
    """Get the Docker image tag for a GSO instance.

    Args:
        instance: GSO instance dict.
        dockerhub_prefix: Docker Hub prefix (user/repo).

    Returns:
        Full Docker image tag string.
    """
    image_tag = instance.get("instance_image_tag", "latest")
    return f"{dockerhub_prefix}:{image_tag}"
