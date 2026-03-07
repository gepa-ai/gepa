"""Persist every training evaluation to a candidate-centric directory.

Records all subsample evaluations (including rejected candidates) via
``on_evaluation_end``.  Validation set evaluations are intentionally
excluded — they exist for generalizability checking, not for the
reflection loop.

Directory layout::

    {run_dir}/evals/
        c000/
            skill.md
            meta.json          # candidate_idx, parents, iteration, avg_score
            tasks/
                task_{id}_eval_{global_eval_num}.json
        c001/
            ...
"""

from __future__ import annotations

import json
import os
from typing import Any

from gepa.core.callbacks import EvaluationEndEvent


class EvalRecorderCallback:
    """GEPACallback that writes training evaluation results to ``{run_dir}/evals/``."""

    def __init__(self, run_dir: str) -> None:
        self.evals_dir = os.path.join(run_dir, "evals")
        os.makedirs(self.evals_dir, exist_ok=True)
        # Counter only for proposed-but-not-yet-accepted candidates (candidate_idx=None).
        # Accepted candidates use their program index directly: c{idx:05d}.
        self._proposed_counter = 0
        self._eval_counter = 0

    def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        """Record every training evaluation (subsample evals, including rejected candidates)."""
        candidate: dict[str, str] = event["candidate"]
        example_ids: list[Any] = event["example_ids"]
        scores: list[float] = event["scores"]
        outputs: list[Any] = event["outputs"]
        iteration: int = event["iteration"]
        candidate_idx = event["candidate_idx"]
        parent_ids = list(event["parent_ids"])

        eval_id = self._eval_counter
        self._eval_counter += 1

        # Resolve the candidate directory.
        # Accepted candidates (candidate_idx is not None) use their program index
        # directly so the directory name is stable and predictable: c{idx:05d}.
        # Proposed-but-not-yet-accepted candidates (candidate_idx=None) get a
        # sequentially-numbered proposed_{n:05d} directory that never collides with
        # accepted-candidate directories.
        if candidate_idx is not None:
            cand_dir_name = f"c{candidate_idx:05d}"
        else:
            cand_dir_name = f"proposed_{self._proposed_counter:05d}"
            self._proposed_counter += 1

        cand_dir = os.path.join(self.evals_dir, cand_dir_name)
        tasks_dir = os.path.join(cand_dir, "tasks")
        os.makedirs(tasks_dir, exist_ok=True)

        # skill.md — written once, not overwritten on re-evals
        skill_path = os.path.join(cand_dir, "skill.md")
        if not os.path.exists(skill_path):
            _write_skill_md(skill_path, candidate)

        # meta.json — update average_score to reflect latest eval
        avg_score = sum(scores) / len(scores) if scores else 0.0
        meta: dict[str, Any] = {
            "candidate_idx": candidate_idx,
            "parents": parent_ids,
            "iteration": iteration,
            "average_score": avg_score,
        }
        _write_json(os.path.join(cand_dir, "meta.json"), meta)

        # tasks/task_{id}_eval_{eval_id}.json — preserves history across re-evals
        for eid, score, output in zip(example_ids, scores, outputs, strict=False):
            task_data: dict[str, Any] = {
                "eval_id": eval_id,
                "candidate_idx": candidate_idx,
                "task_id": str(eid),
                "score": score,
                "iteration": iteration,
            }
            side_info = _extract_side_info(output)
            if side_info is not None:
                task_data["side_info"] = _make_json_serializable(side_info)
            task_id_str = f"{eid:05d}" if isinstance(eid, int) else str(eid)
            _write_json(os.path.join(tasks_dir, f"task_{task_id_str}_eval_{eval_id:05d}.json"), task_data)


def _write_skill_md(path: str, candidate: dict[str, str]) -> None:
    lines: list[str] = []
    for key, value in candidate.items():
        lines.append(f"## {key}\n\n{value}\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _extract_side_info(output: Any) -> dict[str, Any] | None:
    """Extract side_info from an optimize_anything output tuple."""
    # In optimize_anything mode, output is (score, candidate, side_info)
    if isinstance(output, tuple) and len(output) >= 3:
        side_info = output[2]
        if isinstance(side_info, dict):
            return side_info
    return None


def _write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _make_json_serializable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, str | int | float | bool | type(None)):
        return obj
    return str(obj)
