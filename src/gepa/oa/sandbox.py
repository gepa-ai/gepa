"""External bubblewrap jail for ``claude --print`` subprocesses.

Wraps the whole ``claude`` invocation in our own ``bwrap`` namespace instead
of relying on Claude Code's built-in ``sandbox.enabled: true`` settings. The
internal sandbox crashes on Ubuntu 24.04 with
``bwrap: Can't mount tmpfs on /newroot/sbin: No such file or directory``
because it tries to mount tmpfs on top of ``/sbin``, which is a symlink in
the merged-``/usr`` layout. We control the bwrap argv, so we can detect
symlinks and emit ``--symlink`` instead of ``--tmpfs``.

Layout we expose inside the jail:

- ``/usr`` and friends: read-only bind, with symlinks recreated for
  merged-``/usr`` distros (Ubuntu 24.04+, Fedora, Arch). On older Debian /
  RHEL where ``/bin`` is a real directory, those paths get ``--ro-bind``
  instead.
- ``/etc``: only the handful of files needed for DNS, certs, and user
  lookups (``resolv.conf``, ``hosts``, ``passwd``, ``group``, ``ssl``...).
- ``/proc``, ``/dev``, ``/tmp``: standard mounts.
- ``$HOME/.claude``, ``$HOME/.claude.json``, ``$HOME/.cache``: writable.
- ``$HOME/.local``: read-only — ``claude`` itself lives under here.
- ``work_dir``: the only writable path under ``/data``-style trees.

Network namespace is shared with the host so the agent can reach
``localhost:<eval-server-port>`` and ``api.anthropic.com``. ``WebFetch`` /
``WebSearch`` are denied at the tool layer via :data:`DENY_WEB_TOOLS`.

macOS fallback: bwrap is Linux-only. On macOS we use Claude Code's
built-in Seatbelt sandbox via :func:`claude_settings_args`, which the
caller appends to the ``claude --print`` argv.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Linux uses bwrap; macOS falls back to Claude Code's Seatbelt sandbox
# (see ``claude_settings_args`` below). The bug that motivated the bwrap
# rewrite is Linux-only.
_IS_MACOS = sys.platform == "darwin"

# File tools whitelisted per allowed dir on the Seatbelt path. Includes
# Glob because under ``--permission-mode default`` (which we use to make
# the allowlist enforce — see ``claude_settings_args``) every unlisted
# tool call auto-denies in ``--print`` mode.
_FILE_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob", "Edit", "Write", "NotebookEdit")

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
    """Return the ``bwrap`` argv prefix that jails everything that follows.

    Returns ``[]`` on macOS — that platform uses :func:`claude_settings_args`
    as a fallback because ``bwrap`` is Linux-only. Caller usage works on both
    platforms::

        cmd = bwrap_prefix(work_dir)               # Linux: bwrap argv. macOS: [].
        cmd += ["claude", "--print", ...]
        cmd += claude_settings_args(work_dir)      # macOS: --settings JSON. Linux: [].
    """
    if _IS_MACOS:
        return []

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


def _abs_glob(path: str) -> str:
    """Format an absolute path as Claude's ``//<path>/**`` rule pattern."""
    return f"/{path}/**"


def _build_macos_sandbox_settings(
    work_dir: Path | str,
    *,
    extra_writable: list[Path | str] | None = None,
) -> dict[str, Any]:
    """Settings JSON for Claude Code's Seatbelt sandbox.

    Two layers: ``sandbox.filesystem.*`` confines Bash subprocesses;
    ``permissions.allow`` whitelists file tools (only enforces under
    ``--permission-mode default``, see :func:`claude_settings_args`).
    """
    work_paths = [str(Path(work_dir).resolve())]
    work_paths.extend(str(Path(p).resolve()) for p in extra_writable or ())

    # Bash subprocesses need /tmp + /private/tmp writable for claude's
    # per-call script staging dir. Both forms because Seatbelt path-matches
    # literally and /tmp is a symlink to /private/tmp.
    write_paths = work_paths + ["/tmp", "/private/tmp"]

    allow_rules: list[str] = [f"{tool}({_abs_glob(p)})" for p in work_paths for tool in _FILE_TOOLS]
    allow_rules.append("Bash(*)")
    return {
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": False,
            "allowUnsandboxedCommands": False,
            "network": {
                "allowLocalBinding": True,
            },
            "filesystem": {
                "denyRead": ["~/"],
                "allowRead": work_paths,
                "allowWrite": write_paths,
            },
        },
        "permissions": {"allow": allow_rules},
    }


def claude_settings_args(
    work_dir: Path | str,
    *,
    extra_writable: list[Path | str] | None = None,
) -> list[str]:
    """Settings + permission flags for the macOS Seatbelt path. Empty on Linux.

    Includes ``--permission-mode default`` so the ``permissions.allow``
    whitelist in the settings JSON actually enforces. In ``--print`` mode
    any unlisted tool call auto-denies because there's no human to approve
    the prompt — so the allowlist becomes a strict tool-layer whitelist that
    complements the OS-level Seatbelt confinement.
    """
    if not _IS_MACOS:
        return []
    settings = _build_macos_sandbox_settings(work_dir, extra_writable=extra_writable)
    return [
        "--settings",
        json.dumps(settings),
        "--permission-mode",
        "default",
    ]


def claude_permission_args(
    work_dir: Path | str,
    *,
    sandboxed: bool,
    extra_writable: list[Path | str] | None = None,
) -> list[str]:
    """Resolve the *single* tool-permission posture for a ``claude --print`` call.

    Callers must use this instead of hardcoding ``--permission-mode`` so the
    argv never carries two conflicting modes. Exactly one mode is emitted:

    - **macOS + sandboxed** → ``--settings <seatbelt json> --permission-mode
      default`` (via :func:`claude_settings_args`). ``default`` is what makes
      the settings' ``permissions.allow`` whitelist enforce: in ``--print``
      mode every unlisted tool auto-denies (no human to approve), so the
      allowlist becomes a strict tool-layer whitelist layered on top of the
      Seatbelt filesystem confinement.
    - **Linux + sandboxed** → ``--permission-mode bypassPermissions``. The
      bwrap jail (:func:`bwrap_prefix`) is the OS-level confinement and there
      is no human to answer prompts in ``--print`` mode, so tool permissions
      are bypassed inside the jail.
    - **unsandboxed** (either platform) → ``--permission-mode bypassPermissions``.
    """
    if sandboxed:
        # On macOS this carries its own ``--permission-mode default``; on Linux
        # it is empty (bwrap handles confinement), so we fall through to bypass.
        settings = claude_settings_args(work_dir, extra_writable=extra_writable)
        if settings:
            return settings
    return ["--permission-mode", "bypassPermissions"]
