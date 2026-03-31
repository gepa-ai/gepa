"""CodingAdapter — subclass of OptimizeAnythingAdapter for code optimization.

Candidates are ``{repo_path: branch_name}`` dicts where each key is a path to
a git repository and each value is the branch to check out.  Multiple repos
are supported (equivalent to multiple components in text mode).

The adapter:

1. **Before evaluation**: checks out each repo to its branch
2. **Reflective dataset**: includes code diffs per component
3. **Proposal**: delegates to a pluggable coding agent
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from gepa.adapters.coding_adapter.coding_agent import CodingAgentProtocol
from gepa.adapters.coding_adapter.git_repo import GitRepo
from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
from gepa.core.adapter import DataInst, EvaluationBatch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CodingAdapter(OptimizeAnythingAdapter):
    """Adapter for optimizing code in git repositories.

    Subclasses :class:`OptimizeAnythingAdapter` to reuse evaluation caching,
    parallel execution, ``oa.log()`` capture, and ``opt_state`` injection.

    Overrides:
        - ``evaluate``: checks out branches before calling the evaluator
        - ``make_reflective_dataset``: includes code diffs
        - ``propose_new_texts``: delegates to a coding agent
    """

    def __init__(
        self,
        *args: Any,
        coding_agent: CodingAgentProtocol,
        repos: dict[str, GitRepo],
        base_branches: dict[str, str],
        branch_prefix: str = "gepa",
        objective: str | None = None,
        background: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.coding_agent = coding_agent
        self.repos = repos  # {repo_path: GitRepo}
        self.base_branches = base_branches  # {repo_path: base_branch_name}
        self.branch_prefix = branch_prefix
        self.coding_objective = objective
        self.coding_background = background

        self._branch_counter = 0
        self._branch_counter_lock = threading.Lock()

    # --- Override: checkout before evaluation ---

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Checkout all repos to their branches, then delegate to parent evaluate."""
        for repo_path, branch in candidate.items():
            repo = self.repos[repo_path]
            repo.checkout(branch)

        return super().evaluate(batch, candidate, capture_traces)

    # --- Override: reflective dataset with code diffs ---

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build reflective dataset including code diffs for each component (repo)."""
        scores = eval_batch.scores
        side_infos = eval_batch.trajectories
        assert side_infos is not None

        ret: dict[str, list[dict[str, Any]]] = {}

        for repo_path in components_to_update:
            branch = candidate[repo_path]
            base_branch = self.base_branches[repo_path]
            repo = self.repos[repo_path]

            # Get diff for this repo
            try:
                diff = repo.get_diff(base_branch, branch)
            except Exception:
                diff = "(diff unavailable)"

            records: list[dict[str, Any]] = []
            for score, side_info in zip(scores, side_infos, strict=True):
                record: dict[str, Any] = {
                    "Code Diff from Base": diff,
                    "Score": score,
                }
                # Include side_info fields (skip "scores" key used for objectives)
                for k, v in side_info.items():
                    if k == "scores":
                        record["Scores (Higher is Better)"] = v
                    elif k == f"{repo_path}_specific_info":
                        record.update(v)
                    elif not k.endswith("_specific_info"):
                        record[k] = v
                records.append(record)

            ret[repo_path] = records

        return ret

    # --- Proposal via coding agent ---

    def _next_branch_name(self) -> str:
        with self._branch_counter_lock:
            self._branch_counter += 1
            return f"{self.branch_prefix}/iter_{self._branch_counter}"

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Use the coding agent to propose code changes.

        For each repo in ``components_to_update``, creates a new branch from
        the parent, runs the coding agent, and commits the changes.

        Returns ``{repo_path: new_branch_name}`` for updated repos.
        """
        new_candidate: dict[str, str] = {}
        feedback = self._format_feedback(reflective_dataset)
        new_branch = self._next_branch_name()

        for repo_path in components_to_update:
            parent_branch = candidate[repo_path]
            base_branch = self.base_branches[repo_path]
            repo = self.repos[repo_path]

            repo.create_branch(new_branch, parent_branch)
            repo.checkout(new_branch)

            try:
                changes_made = self.coding_agent.propose(
                    repo=repo,
                    base_branch=base_branch,
                    feedback=feedback,
                    objective=self.coding_objective,
                    background=self.coding_background,
                )
                if changes_made:
                    repo.commit_all(f"gepa: iteration {self._branch_counter}")
            except Exception:
                repo.checkout(parent_branch)
                raise

            new_candidate[repo_path] = new_branch

        return new_candidate

    def _format_feedback(self, reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
        """Format reflective dataset into readable feedback for the coding agent."""
        parts: list[str] = []

        for component, records in reflective_dataset.items():
            if not records:
                continue
            parts.append(f"## Repository: {component}\n")
            for i, record in enumerate(records):
                part = f"### Example {i + 1}\n"
                for k, v in record.items():
                    if k == "Code Diff from Base":
                        continue  # Agent computes its own diff
                    if isinstance(v, dict | list):
                        part += f"**{k}**: {json.dumps(v, indent=2, default=str)}\n"
                    else:
                        part += f"**{k}**: {v}\n"
                parts.append(part)

        return "\n".join(parts) if parts else "(no evaluation feedback available)"
