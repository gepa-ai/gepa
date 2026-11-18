# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Helix integration for ``optimize_anything``.

`Helix <https://github.com/KE7/helix>`_ is an evolutionary code optimization
framework that evolves entire codebases using LLM-driven mutations.  This
module provides helpers to use Helix's evaluation infrastructure as an
evaluator (or batch evaluator) inside ``optimize_anything``.

Install the optional dependency::

    pip install "gepa[helix]"

Quick example::

    from gepa.adapters.helix import make_helix_evaluator
    from gepa.optimize_anything import optimize_anything

    evaluator = make_helix_evaluator(
        config="helix.toml",
        target_file="src/solver.py",
    )

    result = optimize_anything(
        seed_candidate=open("src/solver.py").read(),
        evaluator=evaluator,
        objective="Maximize accuracy on the test suite.",
    )
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any


def _import_helix() -> ModuleType:
    """Import the helix package at runtime. Raises ImportError if not installed."""
    try:
        import helix

        return helix
    except ImportError:
        raise ImportError(
            "helix-evo is required for Helix integration. "
            'Install it with: pip install "gepa[helix]"'
        ) from None


def _load_config(config: Any) -> Any:
    """Accept a HelixConfig, a path string, or a Path and return a HelixConfig."""
    _import_helix()
    from helix.config import HelixConfig

    if isinstance(config, str | Path):
        import tomllib

        path = Path(config)
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return HelixConfig(**raw)
    return config


def _write_candidate_to_worktree(
    worktree_path: str,
    candidate_text: str | dict[str, str],
    target_file: str | dict[str, str] | None,
) -> None:
    """Write candidate text to the appropriate file(s) in a worktree."""
    if isinstance(candidate_text, str):
        assert target_file is not None and isinstance(target_file, str), (
            "target_file must be a string path when candidate is a string"
        )
        dst = os.path.join(worktree_path, target_file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            f.write(candidate_text)
    elif isinstance(candidate_text, dict):
        if isinstance(target_file, dict):
            for name, text in candidate_text.items():
                fpath = target_file.get(name, name)
                dst = os.path.join(worktree_path, fpath)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "w") as f:
                    f.write(text)
        else:
            for name, text in candidate_text.items():
                dst = os.path.join(worktree_path, name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "w") as f:
                    f.write(text)


def make_helix_evaluator(
    config: Any,
    target_file: str | dict[str, str] | None = None,
    split: str = "val",
    score_key: str | None = None,
    repo_path: str | None = None,
) -> Any:
    """Create an ``optimize_anything``-compatible evaluator from a Helix config.

    The returned callable writes the candidate text to the target file(s) in a
    temporary copy of the repo, runs the Helix evaluator command, and returns
    ``(score, side_info)``.

    Args:
        config: A :class:`helix.config.HelixConfig`, or a path to ``helix.toml``.
        target_file: Which file(s) the candidate text maps to.

            - ``str`` — single file path relative to the repo root
              (use when ``seed_candidate`` is a plain string).
            - ``dict[str, str]`` — mapping from candidate component name
              to file path (use when ``seed_candidate`` is a dict).
            - ``None`` — candidate dict keys are used directly as file paths.

        split: Dataset split to evaluate on (``"train"``, ``"val"``, etc.).
        score_key: Key to extract from ``EvalResult.scores``.  If ``None``,
            uses the first score or falls back to the mean of instance scores.
        repo_path: Path to the git repository.  Defaults to cwd.

    Returns:
        A callable ``(candidate, example=None) -> (float, dict)`` suitable for
        passing as ``evaluator`` to ``optimize_anything``.
    """
    _import_helix()
    from helix.executor import run_evaluator
    from helix.population import Candidate

    helix_config = _load_config(config)
    repo = os.path.abspath(repo_path or ".")

    def evaluator(candidate: str | dict[str, str], example: Any = None) -> tuple[float, dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="helix_eval_") as tmpdir:
            worktree = os.path.join(tmpdir, "repo")
            shutil.copytree(repo, worktree, symlinks=True, dirs_exist_ok=False)
            _write_candidate_to_worktree(worktree, candidate, target_file)

            helix_candidate = Candidate(
                id="gepa_eval",
                worktree_path=worktree,
                branch_name="gepa_eval",
                generation=0,
                parent_id=None,
                parent_ids=[],
                operation="evaluate",
            )

            instance_ids = None
            if example is not None:
                if isinstance(example, str):
                    instance_ids = [example]
                elif isinstance(example, dict) and "instance_id" in example:
                    instance_ids = [example["instance_id"]]

            result = run_evaluator(
                helix_candidate, helix_config, split=split, instance_ids=instance_ids
            )

            score = _extract_score(result, score_key)
            side_info: dict[str, Any] = {
                "scores": result.scores,
                "instance_scores": result.instance_scores,
                "asi": result.asi,
            }
            return score, side_info

    return evaluator


def make_helix_batch_evaluator(
    config: Any,
    target_file: str | dict[str, str] | None = None,
    split: str = "val",
    score_key: str | None = None,
    repo_path: str | None = None,
) -> Any:
    """Create an ``optimize_anything``-compatible batch evaluator from a Helix config.

    The returned callable evaluates multiple ``(candidate, example)`` pairs
    using Helix's ``run_evaluators_parallel``.

    Args:
        config: A :class:`helix.config.HelixConfig`, or a path to ``helix.toml``.
        target_file: File mapping (see :func:`make_helix_evaluator`).
        split: Dataset split to evaluate on.
        score_key: Key to extract from ``EvalResult.scores``.
        repo_path: Path to the git repository.  Defaults to cwd.

    Returns:
        A callable ``(pairs) -> list[(float, dict)]`` suitable for passing as
        ``batch_evaluator`` to ``optimize_anything``.
    """
    _import_helix()
    from helix.executor import run_evaluators_parallel
    from helix.population import Candidate

    helix_config = _load_config(config)
    repo = os.path.abspath(repo_path or ".")

    def batch_evaluator(
        pairs: list[tuple[str | dict[str, str], Any]],
    ) -> list[tuple[float, dict[str, Any]]]:
        tmpdirs: list[tempfile.TemporaryDirectory[str]] = []
        candidates: list[Any] = []

        try:
            for i, (candidate_text, _example) in enumerate(pairs):
                td = tempfile.TemporaryDirectory(prefix=f"helix_batch_{i}_")
                tmpdirs.append(td)
                worktree = os.path.join(td.name, "repo")
                shutil.copytree(repo, worktree, symlinks=True, dirs_exist_ok=False)
                _write_candidate_to_worktree(worktree, candidate_text, target_file)

                candidates.append(
                    Candidate(
                        id=f"gepa_batch_{i}",
                        worktree_path=worktree,
                        branch_name=f"gepa_batch_{i}",
                        generation=0,
                        parent_id=None,
                        parent_ids=[],
                        operation="evaluate",
                    )
                )

            results = run_evaluators_parallel(candidates, helix_config, split=split)

            output: list[tuple[float, dict[str, Any]]] = []
            for result in results:
                score = _extract_score(result, score_key)
                side_info: dict[str, Any] = {
                    "scores": result.scores,
                    "instance_scores": result.instance_scores,
                    "asi": result.asi,
                }
                output.append((score, side_info))

            return output
        finally:
            for td in tmpdirs:
                td.cleanup()

    return batch_evaluator


def _extract_score(result: Any, score_key: str | None) -> float:
    """Extract a single float score from an EvalResult."""
    if score_key is not None and score_key in result.scores:
        return float(result.scores[score_key])

    if result.scores:
        return float(next(iter(result.scores.values())))

    if result.instance_scores:
        vals = list(result.instance_scores.values())
        return sum(vals) / len(vals)

    return 0.0
