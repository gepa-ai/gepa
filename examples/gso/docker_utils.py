"""Docker utilities for running GSO benchmark instances.

Uses the pre-built GSO Docker images from Docker Hub (slimshetty namespace).
Image naming: slimshetty/gso:gso.eval.<arch>.<instance_id>
"""

from __future__ import annotations

import logging
import subprocess
import time

logger = logging.getLogger(__name__)


def get_image_name(instance_id: str, arch: str = "x86_64", namespace: str = "slimshetty") -> str:
    """Get the full Docker image name for a GSO instance."""
    return f"{namespace}/gso:gso.eval.{arch}.{instance_id}"


def pull_image(image_name: str) -> None:
    """Pull a Docker image if not already available locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info(f"Image {image_name} already available locally")
        return

    logger.info(f"Pulling Docker image: {image_name}")
    subprocess.run(["docker", "pull", image_name], check=True)


def create_container(image_name: str, name: str = "gepa-gso-eval") -> str:
    """Create and start a detached container. Returns container ID."""
    # Remove existing container with same name if any
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            "docker", "create",
            "--name", name,
            "--user", "root",
            image_name,
            "tail", "-f", "/dev/null",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    container_id = result.stdout.strip()

    subprocess.run(["docker", "start", container_id], check=True, capture_output=True)
    logger.info(f"Container {name} started: {container_id[:12]}")
    return container_id


def exec_in_container(
    container: str,
    command: str,
    timeout: int = 300,
    workdir: str | None = None,
) -> dict[str, str | int]:
    """Execute a command in a running container.

    Returns dict with stdout, stderr, returncode.
    """
    cmd = ["docker", "exec"]
    if workdir:
        cmd.extend(["-w", workdir])
    cmd.extend([container, "bash", "-c", command])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "returncode": -1,
        }


def copy_to_container(container: str, src: str, dest: str) -> None:
    """Copy a file from host to container."""
    subprocess.run(
        ["docker", "cp", src, f"{container}:{dest}"],
        check=True,
        capture_output=True,
    )


def copy_from_container(container: str, src: str, dest: str) -> None:
    """Copy a file from container to host."""
    subprocess.run(
        ["docker", "cp", f"{container}:{src}", dest],
        check=True,
        capture_output=True,
    )


def apply_patch_in_container(container: str, patch_content: str, workdir: str = "/testbed") -> dict:
    """Apply a git diff patch inside the container.

    Returns dict with success, stdout, stderr.
    """
    import tempfile
    import os

    # Write patch to temp file and copy to container
    with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
        f.write(patch_content)
        patch_path = f.name

    try:
        copy_to_container(container, patch_path, "/tmp/patch.diff")
    finally:
        os.unlink(patch_path)

    # Reset any previous changes
    exec_in_container(container, f"cd {workdir} && git reset --hard HEAD", timeout=30)

    # Apply patch
    result = exec_in_container(
        container,
        f"cd {workdir} && git apply /tmp/patch.diff",
        timeout=30,
    )

    if result["returncode"] != 0:
        # Try with more lenient options
        result = exec_in_container(
            container,
            f"cd {workdir} && git apply --ignore-space-change /tmp/patch.diff",
            timeout=30,
        )

    return {
        "success": result["returncode"] == 0,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def run_script_in_container(
    container: str,
    script_content: str,
    script_name: str = "_gepa_script.py",
    workdir: str = "/testbed",
    timeout: int = 300,
    activate_venv: bool = True,
) -> dict[str, str | int]:
    """Write a Python script to the container and run it.

    Returns dict with stdout, stderr, returncode.
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        tmp_path = f.name

    try:
        dest = f"/tmp/{script_name}"
        copy_to_container(container, tmp_path, dest)
    finally:
        os.unlink(tmp_path)

    if activate_venv:
        cmd = f"cd {workdir} && source .venv/bin/activate && python {dest}"
    else:
        cmd = f"cd {workdir} && python {dest}"

    return exec_in_container(container, cmd, timeout=timeout)


def stop_and_remove_container(container: str) -> None:
    """Stop and remove a container."""
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True)
    logger.info(f"Container {container} removed")


def reinstall_in_container(container: str, install_commands: list[str], workdir: str = "/testbed") -> dict:
    """Run install commands to rebuild the project after code changes."""
    cmd = f"cd {workdir} && source .venv/bin/activate && " + " && ".join(install_commands)
    return exec_in_container(container, cmd, timeout=600)
