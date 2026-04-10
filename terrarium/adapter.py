"""Adapter protocol: the interface every evolution system implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from terrarium.task import Task

if TYPE_CHECKING:
    from terrarium.eval_server import EvalServer


class Adapter(Protocol):
    """Protocol that every evolution system adapter must implement.

    The adapter receives an ``EvalServer`` — the single choke point for all
    evaluations. In-process adapters call ``server.evaluate(candidate)``
    directly. External/black-box adapters point their subprocess at
    ``server.url`` for HTTP-based eval.

    Example::

        class MyEvolver:
            def evolve(self, task, server, max_evals):
                candidate = task.initial_candidate
                best, best_score = candidate, -float("inf")
                for _ in range(max_evals):
                    score, info = server.evaluate(candidate)
                    if score > best_score:
                        best_score, best = score, candidate
                    candidate = self.mutate(candidate, score, info)
                return Result(best_candidate=best, best_score=best_score)
    """

    def evolve(
        self,
        task: Task,
        server: EvalServer,
        max_evals: int,
    ) -> Result:
        """Run evolution and return the best candidate found.

        Args:
            task: Task definition (description, initial_candidate, datasets).
            server: The eval server. Call ``server.evaluate(candidate)`` for
                    in-process eval, or use ``server.url`` for HTTP-based eval.
                    Raises ``BudgetExhausted`` when the budget is used up.
            max_evals: Total eval budget (informational — enforced by server).

        Returns:
            Result containing the best candidate and metadata.
        """
        ...


@dataclass
class Result:
    """What an adapter returns after evolution."""

    best_candidate: str
    best_score: float
    total_evals: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    eval_log: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
