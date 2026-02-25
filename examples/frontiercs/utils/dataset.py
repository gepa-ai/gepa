"""Frontier-CS dataset loading with enriched per-problem metadata."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, TypedDict

FRONTIERCS_REPO = "https://github.com/FrontierCS/Frontier-CS.git"


class FrontierCSProblem(TypedDict, total=False):
    problem_id: str
    statement: str
    tag: str
    time_limit: str
    memory_limit: str
    test_count: str
    sample_input: str
    sample_output: str


def get_frontiercs_problems_dir() -> Path:
    """Return path to Frontier-CS problems, cloning the repo if needed.

    Uses cache dir: $XDG_CACHE_HOME/gepa/frontier-cs or ~/.cache/gepa/frontier-cs
    """
    cache_base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    repo_dir = Path(cache_base) / "gepa" / "frontier-cs"
    problems_dir = repo_dir / "algorithmic" / "problems"

    if not problems_dir.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        if not (repo_dir / ".git").exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", FRONTIERCS_REPO, str(repo_dir)],
                check=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                ["git", "pull", "-q"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
            )

    return problems_dir


def _read_file(path: Path) -> str | None:
    """Read a file if it exists, returning None otherwise."""
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace").strip()
    return None


def _load_config_yaml(pdir: Path) -> dict[str, str]:
    """Parse config.yaml for time_limit, memory_limit, test_count."""
    config_path = pdir / "config.yaml"
    if not config_path.exists():
        return {}
    result: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "time_limit":
                result["time_limit"] = value
            elif key == "memory_limit":
                result["memory_limit"] = value
            elif key == "test_count":
                result["test_count"] = value
    return result


def load_all_problems(problems_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load all Frontier-CS problems with enriched metadata.

    Each problem dict contains:
      - problem_id, statement (always present)
      - tag, time_limit, memory_limit, test_count (if available)
      - sample_input, sample_output (from testdata/1.in, 1.ans if available)
    """
    if not problems_dir:
        problems_path = get_frontiercs_problems_dir()
    else:
        problems_path = Path(problems_dir)
    if not problems_path.exists():
        raise FileNotFoundError(f"Problems directory not found: {problems_path}")

    items: list[dict[str, Any]] = []
    for pdir in sorted(problems_path.iterdir(), key=lambda p: (len(p.name), p.name)):
        if not pdir.is_dir():
            continue
        statement_file = pdir / "statement.txt"
        if not statement_file.exists():
            continue

        problem: dict[str, Any] = {
            "problem_id": pdir.name,
            "statement": statement_file.read_text(encoding="utf-8", errors="replace"),
        }

        tag = _read_file(pdir / "tag.txt")
        if tag:
            problem["tag"] = tag

        config = _load_config_yaml(pdir)
        problem.update(config)

        sample_in = _read_file(pdir / "testdata" / "1.in")
        sample_out = _read_file(pdir / "testdata" / "1.ans")
        if sample_in is not None:
            problem["sample_input"] = sample_in
        if sample_out is not None:
            problem["sample_output"] = sample_out

        items.append(problem)

    if not items:
        raise ValueError(f"No valid problems found in {problems_path}")

    return items
