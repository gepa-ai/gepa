"""External bubblewrap jail for ``claude --print`` subprocesses.

Wraps the whole ``claude`` invocation in our own ``bwrap`` namespace so the
agent can only write under ``work_dir`` and only read the system paths we
explicitly bind. Network namespace is shared (claude needs Anthropic API +
the local eval server). ``WebFetch`` / ``WebSearch`` are denied at the tool
layer with :data:`DENY_WEB_TOOLS`.
"""

from __future__ import annotations

import os
from pathlib import Path

_SYSTEM_PATHS: tuple[str, ...] = (
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/usr/lib32",
    "/usr/lib64",
    "/usr/local",
)

_ETC_FILES: tuple[str, ...] = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/passwd",
    "/etc/group",
    "/etc/ld.so.cache",
    "/etc/localtime",
    "/etc/ssl",
    "/etc/ca-certificates",
    "/etc/alternatives",
)

DENY_WEB_TOOLS: str = "--disallowedTools=WebFetch,WebSearch"


def _system_bind_args() -> list[str]:
    args: list[str] = []
    for path in _SYSTEM_PATHS:
        if os.path.islink(path):
            args.extend(["--symlink", os.readlink(path), path])
        elif os.path.isdir(path):
            args.extend(["--ro-bind", path, path])
    return args


def _etc_bind_args() -> list[str]:
    args: list[str] = []
    for path in _ETC_FILES:
        if os.path.exists(path) or os.path.islink(path):
            args.extend(["--ro-bind", path, path])
    return args


def bwrap_prefix(
    work_dir: Path | str,
    *,
    extra_writable: list[Path | str] | None = None,
) -> list[str]:
    """Return the ``bwrap`` argv prefix that jails everything that follows."""
    home = Path.home()
    work = Path(work_dir).resolve()

    args: list[str] = [
        "bwrap",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        *_system_bind_args(),
        *_etc_bind_args(),
        "--bind",
        str(home / ".claude"),
        str(home / ".claude"),
        "--bind",
        str(home / ".claude.json"),
        str(home / ".claude.json"),
        "--ro-bind",
        str(home / ".local"),
        str(home / ".local"),
        "--bind",
        str(home / ".cache"),
        str(home / ".cache"),
        "--bind",
        str(work),
        str(work),
        "--unshare-uts",
        "--hostname",
        "sandbox",
        "--setenv",
        "HOME",
        str(home),
        "--chdir",
        str(work),
    ]
    for p in extra_writable or ():
        resolved = str(Path(p).resolve())
        args.extend(["--bind", resolved, resolved])
    return args
