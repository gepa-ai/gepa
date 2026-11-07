# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the subprocess timeout path of ``gepa.utils.code_execution``."""

import time

import pytest

from gepa.utils.code_execution import ExecutionMode, execute_code


def test_subprocess_timeout_is_reported_as_failure():
    """A subprocess that runs past the timeout is reported as a failed timeout.

    This path does not require psutil, so it always runs.
    """
    result = execute_code(
        "import time\ntime.sleep(30)",
        timeout=1.0,
        mode=ExecutionMode.SUBPROCESS,
        kill_child_processes=False,
    )

    assert result.success is False
    assert "Timeout" in result.error


def test_subprocess_timeout_kills_spawned_grandchildren(tmp_path):
    """Regression: descendants spawned by the user code must be killed on timeout.

    Previously the subprocess was killed and reaped (``process.kill()`` +
    ``process.wait()``) *before* its children were enumerated. Once reaped, the
    subprocess PID no longer resolves to a live process, so
    ``psutil.Process(pid).children()`` found nothing to kill and the grandchildren
    leaked. Descendants are now captured before the parent is reaped.
    """
    psutil = pytest.importorskip("psutil")

    marker = tmp_path / "grandchild.pid"
    code = f"""
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open({str(marker)!r}, "w") as f:
    f.write(str(child.pid))
    f.flush()
time.sleep(120)
"""

    result = execute_code(
        code,
        timeout=5.0,
        mode=ExecutionMode.SUBPROCESS,
        kill_child_processes=True,
    )

    assert result.success is False
    assert "Timeout" in result.error

    # The user code records the grandchild PID before hanging.
    for _ in range(60):
        if marker.exists() and marker.read_text().strip():
            break
        time.sleep(0.05)
    assert marker.exists() and marker.read_text().strip(), "grandchild PID was never recorded"
    grandchild_pid = int(marker.read_text().strip())

    def _still_running(pid: int) -> bool:
        if not psutil.pid_exists(pid):
            return False
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    # Allow a brief moment for the kill to take effect.
    deadline = time.time() + 10.0
    while time.time() < deadline and _still_running(grandchild_pid):
        time.sleep(0.1)

    running = _still_running(grandchild_pid)
    if running:
        # Don't leak the process if the assertion below fails.
        try:
            psutil.Process(grandchild_pid).kill()
        except psutil.NoSuchProcess:
            pass

    assert not running, "grandchild process outlived the subprocess timeout kill"
