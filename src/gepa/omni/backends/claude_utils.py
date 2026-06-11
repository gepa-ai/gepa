"""Shared Claude Code CLI plumbing used across omni backends and proposers.

Currently exposes :func:`copy_session_transcript`, which mirrors the
``~/.claude/projects/<slug>/<session>.jsonl`` transcript that the CLI writes
during a ``claude --print`` invocation into a caller-controlled artifact dir.
The slug is the cwd path with non-alphanumeric chars replaced — same scheme
the CLI itself uses internally.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_SLUG_RE = re.compile(r"[^A-Za-z0-9-]")


def _claude_project_slug(cwd: Path) -> str:
    return _SLUG_RE.sub("-", str(cwd.resolve()))


def copy_session_transcript(cwd: Path, session_id: str, dst_dir: Path) -> None:
    """Copy the per-session jsonl transcript (if it exists) into ``dst_dir``.

    Best-effort: silently no-ops if the source transcript was never written
    (e.g. the CLI failed before opening it) or the copy raises ``OSError``
    (e.g. the destination disk is full). Callers treat the transcript as a
    nice-to-have artifact, not a correctness signal.
    """
    src = Path.home() / ".claude" / "projects" / _claude_project_slug(cwd) / f"{session_id}.jsonl"
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst_dir / src.name)
    except OSError:
        pass
