"""CodingAdapter — GEPAAdapter for optimizing code in a git repository.

Candidates are represented as ``{"_branch": "branch_name"}`` internally.
The adapter handles git checkout, evaluation, reflective dataset construction,
and delegates code generation to a pluggable coding agent.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from gepa.adapters.coding_adapter.coding_agent import CodingAgentProtocol
from gepa.adapters.coding_adapter.git_repo import GitRepo
from gepa.core.adapter import EvaluationBatch

_BRANCH_KEY = "_branch"


class CodingAdapter:
    """GEPAAdapter that optimizes code in a git repository.

    Each candidate is a git branch. Evaluation checks out the branch and runs
    the user's evaluator. Proposal delegates to a coding agent that modifies
    code and returns a new branch.
    """

    def __init__(
        self,
        repo: GitRepo,
        base_branch: str,
        evaluator: Callable[..., Any],
        coding_agent: CodingAgentProtocol,
        branch_prefix: str = "gepa",
        objective: str | None = None,
        background: str | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> None:
        self.repo = repo
        self.base_branch = base_branch
        self.evaluator = evaluator
        self.coding_agent = coding_agent
        self.branch_prefix = branch_prefix
        self.objective = objective
        self.background = background
        self.parallel = parallel
        self.max_workers = max_workers

        self._branch_counter = 0
        self._counter_lock = threading.Lock()
        self._checkout_lock = threading.Lock()

    def _next_branch_name(self) -> str:
        with self._counter_lock:
            self._branch_counter += 1
            return f"{self.branch_prefix}/iter_{self._branch_counter}"

    def _call_evaluator(self, repo_path: str, example: Any) -> tuple[float, dict[str, Any]]:
        """Call the user's evaluator and normalize the return value."""
        if example is None or (hasattr(example, "__repr__") and "SingleInstanceSentinel" in repr(example)):
            result = self.evaluator(repo_path)
        else:
            result = self.evaluator(repo_path, example)

        if isinstance(result, tuple):
            score, side_info = result
            if not isinstance(side_info, dict):
                side_info = {"info": side_info}
        else:
            score = float(result)
            side_info = {}

        return score, side_info

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate a candidate branch on a batch of examples."""
        branch = candidate[_BRANCH_KEY]

        # Checkout the branch (serialized to avoid conflicts)
        with self._checkout_lock:
            self.repo.checkout(branch)

        repo_path = self.repo.repo_path

        if self.parallel and len(batch) > 1:
            raw_results = self._evaluate_parallel(repo_path, batch)
        else:
            raw_results = [self._call_evaluator(repo_path, example) for example in batch]

        scores = [score for score, _ in raw_results]
        side_infos = [si for _, si in raw_results]
        outputs = list(raw_results)

        # Extract objective_scores from side_info["scores"] if present
        objective_scores: list[dict[str, float]] = []
        for si in side_infos:
            obj = {}
            if "scores" in si:
                obj.update(si["scores"])
            objective_scores.append(obj)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=side_infos if capture_traces else None,
            objective_scores=objective_scores if any(objective_scores) else None,
        )

    def _evaluate_parallel(self, repo_path: str, batch: list[Any]) -> list[tuple[float, dict[str, Any]]]:
        """Evaluate batch examples in parallel (same branch checkout)."""
        results: list[tuple[int, tuple[float, dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers or len(batch)) as executor:
            future_to_idx = {
                executor.submit(self._call_evaluator, repo_path, example): idx for idx, example in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results.append((idx, future.result()))
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build reflective dataset with code diffs and evaluation feedback."""
        branch = candidate[_BRANCH_KEY]
        side_infos = eval_batch.trajectories
        assert side_infos is not None

        # Get the diff from base branch to this candidate
        try:
            diff = self.repo.get_diff(self.base_branch, branch)
        except Exception:
            diff = "(diff unavailable)"

        records: list[dict[str, Any]] = []
        for score, side_info in zip(eval_batch.scores, side_infos, strict=True):
            record: dict[str, Any] = {"Code Diff from Base": diff, "Score": score}
            # Include all side_info fields
            for k, v in side_info.items():
                if k != "scores":
                    record[k] = v
            records.append(record)

        return {_BRANCH_KEY: records}

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Use the coding agent to propose code changes on a new branch."""
        parent_branch = candidate[_BRANCH_KEY]
        new_branch = self._next_branch_name()

        # Format feedback from reflective dataset
        feedback = self._format_feedback(reflective_dataset)

        # Create and checkout new branch from parent
        self.repo.create_branch(new_branch, parent_branch)
        self.repo.checkout(new_branch)

        try:
            changes_made = self.coding_agent.propose(
                repo=self.repo,
                base_branch=self.base_branch,
                feedback=feedback,
                objective=self.objective,
                background=self.background,
            )

            if changes_made:
                self.repo.commit_all(f"gepa: iteration {self._branch_counter}")
            else:
                # No changes — still return the new branch (it's identical to parent)
                pass

        except Exception:
            # On failure, checkout back to parent and re-raise
            self.repo.checkout(parent_branch)
            raise

        return {_BRANCH_KEY: new_branch}

    def _format_feedback(self, reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
        """Format reflective dataset into a readable feedback string for the coding agent."""
        records = reflective_dataset.get(_BRANCH_KEY, [])
        if not records:
            return "(no evaluation feedback available)"

        parts: list[str] = []
        for i, record in enumerate(records):
            part = f"### Example {i + 1}\n"
            for k, v in record.items():
                if k == "Code Diff from Base":
                    continue  # Don't duplicate the diff in per-example feedback
                if isinstance(v, dict | list):
                    part += f"**{k}**: {json.dumps(v, indent=2, default=str)}\n"
                else:
                    part += f"**{k}**: {v}\n"
            parts.append(part)

        return "\n".join(parts)
