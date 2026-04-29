"""Backend protocol — the single interface every optimizer plugs into.

A :class:`Backend` is anything that, given a :class:`Task` and an
:class:`EvalServer`, runs an optimization loop and returns a :class:`Result`.

There is **no separate ``Adapter`` sublayer**. Backends call ``server.evaluate``
directly (in-process) or POST to ``server.url`` (subprocess). Budget /
concurrency / tracking all live in the eval server, so all backends share the
same evaluation surface for free.

Backends consume a backend-specific config at construction time. The omni api
will instantiate ``Backend(**config)`` when the caller passes ``backend="<name>"``,
or use the backend object directly when a constructed instance is passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from gepa.omni.eval_server import EvalServer
    from gepa.omni.task import Task


class Backend(Protocol):
    """Contract every backend implements.

    The api calls :meth:`run` with a task and an eval server, then optionally
    :meth:`process_result` for any post-run artifact persistence.

    Backends that need a ``run_dir`` for artifacts should declare an attribute
    of that name and let the api inject ``<output_dir>/<backend_name>``.
    Top-level config (``max_token_cost``, ``effort``, ``max_thinking_tokens``,
    ``stop_at_score``, ``sandbox``) propagates the same way: the api looks for
    matching attribute names on the backend and sets them when not already set.
    """

    name: str

    def run(self, task: Task, server: EvalServer) -> Result:
        """Run optimization and return the best candidate.

        Args:
            task: Task definition.
            server: The eval server. Call ``server.evaluate(candidate)`` for
                in-process eval, or use ``server.url`` for HTTP-based eval.
                Raises ``BudgetExhausted`` when the eval budget is exhausted.
                ``server.budget.max_token_cost`` carries the LLM-spend cap.
        """
        ...

    def process_result(self, result: Result, output_dir: Path) -> None:
        """Persist backend-specific artifacts under ``output_dir`` after :meth:`run`.

        Default is a no-op. Backends that produced files, transcripts, or
        workspaces during ``run`` should override this to copy/write them.
        """
        return


@dataclass
class Result:
    """What a backend returns after optimization."""

    best_candidate: str
    best_score: float
    total_evals: int = 0
    eval_log: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
