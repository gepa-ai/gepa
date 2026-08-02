"""External bubblewrap jail for coding-agent subprocesses (Claude / Codex).

Wraps the whole agent invocation in our own ``bwrap`` namespace instead of
relying on each CLI's built-in sandbox. Claude Code's internal sandbox
crashes on Ubuntu 24.04 with
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
- Agent home dirs (``agent=`` selects which):
  - ``claude``: ``$HOME/.claude``, ``$HOME/.claude.json``, ``$HOME/.cache``
    writable; ``$HOME/.local`` read-only (where ``claude`` often lives).
  - ``codex``: ``$HOME/.codex``, ``$HOME/.cache`` writable — no Claude paths.
- ``work_dir``: the only writable path under ``/data``-style trees.

Network namespace is shared with the host so the agent can reach
``localhost:<eval-server-port>`` and the model API. Claude's ``WebFetch`` /
``WebSearch`` are denied at the tool layer via :data:`DENY_WEB_TOOLS`.

macOS fallback: bwrap is Linux-only. On macOS Claude uses Seatbelt via
:func:`claude_settings_args`; Codex uses ``--sandbox workspace-write``.
"""

from __future__ import annotations

import json
import os
import shutil
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


def _bind_if_exists(args: list[str], path: Path, *, readonly: bool = False) -> None:
    """Append a bwrap bind for ``path`` when it already exists on the host."""
    if not (path.exists() or path.is_symlink()):
        return
    flag = "--ro-bind" if readonly else "--bind"
    resolved = str(path.resolve()) if path.exists() else str(path)
    args.extend([flag, resolved, resolved])


def _agent_home_bind_args(home: Path, agent: str) -> list[str]:
    """Return agent-specific ``$HOME`` binds for the bwrap jail.

    Ensures writable dirs we own exist before binding so a Codex-only host
    never needs Claude Code paths, and a Claude host never needs ``~/.codex``.
    """
    raw = (agent or "claude").strip().lower()
    if raw in ("claude", "claude-code"):
        agent_key = "claude"
    elif raw == "codex":
        agent_key = "codex"
    else:
        raise ValueError(f"bwrap_prefix agent must be 'claude' or 'codex', got {agent!r}")

    args: list[str] = []
    cache = home / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    _bind_if_exists(args, cache)

    if agent_key == "claude":
        claude_dir = home / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        _bind_if_exists(args, claude_dir)
        claude_json = home / ".claude.json"
        # Preserve prior Claude behavior: bind the credentials file when present.
        # Do not create an empty file — auth must come from a real CLI login.
        _bind_if_exists(args, claude_json)
        _bind_if_exists(args, home / ".local", readonly=True)
    else:
        codex_dir = home / ".codex"
        codex_dir.mkdir(parents=True, exist_ok=True)
        _bind_if_exists(args, codex_dir)

    return args


def bwrap_prefix(
    work_dir: Path | str,
    *,
    extra_writable: list[Path | str] | None = None,
    agent: str = "claude",
) -> list[str]:
    """Return the ``bwrap`` argv prefix that jails everything that follows.

    Returns ``[]`` on macOS — that platform uses per-CLI sandboxes (Claude
    Seatbelt via :func:`claude_settings_args`, Codex ``workspace-write``)
    because ``bwrap`` is Linux-only. Caller usage works on both platforms::

        cmd = bwrap_prefix(work_dir)               # Linux: bwrap argv. macOS: [].
        cmd += ["claude", "--print", ...]
        cmd += claude_settings_args(work_dir)      # macOS: --settings JSON. Linux: [].

    ``agent`` selects which ``$HOME`` auth/config dirs are bound (``"claude"``
    default, or ``"codex"``). Absent optional paths are skipped so a Codex-only
    host does not need Claude Code installed.
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
        *_agent_home_bind_args(home, agent),
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


def _boxed_message(title: str, lines: list[str]) -> str:
    """Render ``title`` + ``lines`` inside a big ASCII box for stderr."""
    width = max(len(title), *(len(line) for line in lines), 60)
    bar = "+" + "=" * (width + 4) + "+"
    rows = [bar, f"|  {title:<{width}}  |", f"|  {'':<{width}}  |"]
    rows += [f"|  {line:<{width}}  |" for line in lines]
    rows.append(bar)
    return "\n".join(rows)


def require_claude_cli(engine_name: str) -> None:
    """Abort with a boxed error when the ``claude`` CLI is not on PATH.

    The subprocess engines (autoresearch, meta_harness) drive their whole
    optimization loop through ``claude --print``; without the CLI the run
    would only die later with a bare ``FileNotFoundError`` from
    ``subprocess``. Fail up front with instructions instead.
    """
    if shutil.which("claude"):
        return
    print(
        _boxed_message(
            "CLAUDE CODE CLI NOT FOUND",
            [
                f"The {engine_name!r} engine drives its optimization loop with the",
                "Claude Code CLI (`claude`), but no `claude` executable is on PATH.",
                "",
                "Install Claude Code first:",
                "  npm install -g @anthropic-ai/claude-code",
                "  (or: curl -fsSL https://claude.ai/install.sh | bash)",
                "then run `claude` once to authenticate, and retry.",
            ],
        ),
        file=sys.stderr,
        flush=True,
    )
    raise RuntimeError(
        f"the {engine_name!r} engine requires the Claude Code CLI (`claude`), which was not found on PATH"
    )


def require_codex_cli(engine_name: str) -> None:
    """Abort with a boxed error when the ``codex`` CLI is not on PATH.

    ``meta_harness`` with ``proposer="codex"`` shells out to ``codex exec``;
    fail up front with install instructions instead of a mid-run
    ``FileNotFoundError``.
    """
    if shutil.which("codex"):
        return
    print(
        _boxed_message(
            "CODEX CLI NOT FOUND",
            [
                f"The {engine_name!r} engine is configured with proposer='codex',",
                "but no `codex` executable is on PATH.",
                "",
                "Install the OpenAI Codex CLI first:",
                "  npm install -g @openai/codex",
                "  (or: brew install --cask codex)",
                "then run `codex login` (or set CODEX_API_KEY) and retry.",
            ],
        ),
        file=sys.stderr,
        flush=True,
    )
    raise RuntimeError(f"the {engine_name!r} engine requires the Codex CLI (`codex`), which was not found on PATH")


def require_bwrap(engine_name: str) -> None:
    """Abort with a boxed error when ``sandbox=True`` can't be honored.

    Linux-only check: the jail is built with bubblewrap (:func:`bwrap_prefix`),
    so a missing ``bwrap`` binary means no OS confinement at all. macOS always
    passes — Seatbelt ships with the OS.
    """
    if _IS_MACOS or shutil.which("bwrap"):
        return
    print(
        _boxed_message(
            "SANDBOX UNAVAILABLE: bwrap NOT FOUND",
            [
                "sandbox=True jails the agent subprocess with bubblewrap on",
                "Linux, but no `bwrap` executable is on PATH.",
                "",
                "Install it:",
                "  sudo apt install bubblewrap   (Debian/Ubuntu)",
                "  sudo dnf install bubblewrap   (Fedora/RHEL)",
                "",
                "Or pass OptimizeAnythingConfig(sandbox=False) to run unsandboxed",
                "(the agent then gets unrestricted access to this machine).",
            ],
        ),
        file=sys.stderr,
        flush=True,
    )
    raise RuntimeError(
        f"sandbox=True on the {engine_name!r} engine but `bwrap` (bubblewrap) was not found on PATH; "
        "install bubblewrap or set sandbox=False"
    )


def warn_sandbox_disabled(engine_name: str, *, agent: str = "Claude Code") -> None:
    """Print a boxed warning when the user opts out of sandboxing. Continues."""
    print(
        _boxed_message(
            "SANDBOX DISABLED",
            [
                f"sandbox=False: the {engine_name!r} engine's {agent} subprocess",
                "runs with NO OS-level confinement — unrestricted Bash plus",
                "read/write access to your files as this user. While normally",
                "harmless, this is potentially DANGEROUS!",
                "",
                "Set sandbox=True (the default) to confine it to a throwaway",
                "work dir (bwrap on Linux; Claude Seatbelt / Codex",
                "workspace-write on macOS).",
            ],
        ),
        file=sys.stderr,
        flush=True,
    )


def preflight_claude_engine(engine_name: str, *, sandbox: bool) -> None:
    """Run all launch-time checks for a claude-subprocess engine.

    Called at the top of ``run()`` by engines that shell out to ``claude``
    (autoresearch, meta_harness): verifies the CLI exists, then either
    verifies the jail can be built (``sandbox=True``) or warns loudly that
    the agent will run unconfined (``sandbox=False``).
    """
    require_claude_cli(engine_name)
    if sandbox:
        require_bwrap(engine_name)
    else:
        warn_sandbox_disabled(engine_name, agent="Claude Code")


def preflight_codex_engine(engine_name: str, *, sandbox: bool) -> None:
    """Run all launch-time checks for a Codex-subprocess engine.

    Used by ``meta_harness`` when ``proposer="codex"``: verifies the ``codex``
    CLI exists, then either verifies the bwrap jail can be built on Linux
    (``sandbox=True``) or warns that the agent will run unconfined.
    """
    require_codex_cli(engine_name)
    if sandbox:
        require_bwrap(engine_name)
    else:
        warn_sandbox_disabled(engine_name, agent="Codex")
