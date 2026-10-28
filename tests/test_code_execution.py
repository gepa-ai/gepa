# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the subprocess timeout path of ``gepa.utils.code_execution``."""

import os
import subprocess
import sys
import time

import pytest

from gepa.utils.code_execution import ExecutionMode, execute_code


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-specific")
@pytest.mark.parametrize("psutil_available", [True, False])
def test_timeout_kills_orphan_after_wrapper_has_exited(tmp_path, monkeypatch, psutil_available):
    """An exited wrapper can leave a child holding its stdout/stderr pipes open."""
    psutil = pytest.importorskip("psutil")
    marker = tmp_path / "orphan.pid"
    original_popen = subprocess.Popen

    def launch_and_reap_wrapper(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        # Reap the wrapper deterministically before communicate times out on
        # the pipes its child inherited. Tree enumeration can no longer find it.
        process.wait(timeout=10)
        return process

    monkeypatch.setattr("gepa.utils.code_execution.subprocess.Popen", launch_and_reap_wrapper)
    if not psutil_available:
        monkeypatch.setitem(sys.modules, "psutil", None)
    code = f"""
import subprocess, sys
pid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]).pid
with open({str(marker)!r}, "w") as f:
    f.write(str(pid))
"""

    try:
        result = execute_code(code, timeout=0.2, mode=ExecutionMode.SUBPROCESS, kill_child_processes=True)

        assert not result.success
        assert "Timeout" in result.error
        assert marker.exists()
        try:
            child = psutil.Process(int(marker.read_text()))
        except psutil.NoSuchProcess:
            return
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
        else:
            pytest.fail("orphaned child survived the timeout cleanup")
    finally:
        if marker.exists():
            try:
                psutil.Process(int(marker.read_text())).kill()
            except psutil.NoSuchProcess:
                pass


def test_subprocess_timeout_is_reported_as_failure(monkeypatch):
    """A subprocess that runs past the timeout is reported as a failed timeout.

    This path does not require psutil, so it always runs.
    """
    if os.name == "posix":

        def unexpected_group_signal(*args):
            pytest.fail("kill_child_processes=False must not signal the caller's process group")

        monkeypatch.setattr(os, "killpg", unexpected_group_signal)

    result = execute_code(
        "import time\ntime.sleep(30)",
        timeout=1.0,
        mode=ExecutionMode.SUBPROCESS,
        kill_child_processes=False,
    )

    assert result.success is False
    assert "Timeout" in result.error


@pytest.mark.parametrize(
    "empty_snapshot",
    [False, pytest.param(True, marks=pytest.mark.skipif(os.name != "posix", reason="POSIX group fallback"))],
)
def test_subprocess_timeout_kills_spawned_grandchildren(tmp_path, monkeypatch, empty_snapshot):
    """Regression: descendants spawned by the user code must be killed on timeout.

    Previously the subprocess was killed and reaped (``process.kill()`` +
    ``process.wait()``) *before* its children were enumerated. Once reaped, the
    subprocess PID no longer resolves to a live process, so
    ``psutil.Process(pid).children()`` found nothing to kill and the grandchildren
    leaked. Descendants are now captured before the parent is reaped.
    """
    psutil = pytest.importorskip("psutil")
    if empty_snapshot:
        # A process-tree snapshot can miss descendants born just after it.
        monkeypatch.setattr(psutil.Process, "children", lambda *args, **kwargs: [])

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
