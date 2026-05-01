"""Backend protocol — the single interface every optimizer plugs into.

A :class:`Backend` is anything that, given a :class:`Task` and an
:class:`EvalServer`, runs an optimization loop and returns a :class:`Result`.

There is **no separate ``Adapter`` sublayer**. Backends call ``server.evaluate``
directly (in-process) or POST to ``server.url`` (subprocess). Budget /
concurrency / tracking all live in the eval server, so all backends share the
same evaluation surface for free.

Construction goes through ``__init__(config: OmniConfig)``. Each backend
reads cross-cutting fields directly (``config.run_dir``, ``config.stop_at_score``,
…) and pops backend-specific keys from ``config.config``, storing what it
needs on ``self``. ``run`` reads ``self.*``. The api never injects
attributes after construction.

Backends should warn (or raise) about unknown keys in ``config.config``
themselves — they're the only ones who know what they consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from gepa.omni.config import OmniConfig
    from gepa.omni.eval_server import EvalServer
    from gepa.omni.task import Task


class Backend(Protocol):
    """Contract every backend implements.

    Construction takes an :class:`OmniConfig`; the api calls :meth:`run` with
    a task and an eval server, then :meth:`process_result` for post-run
    artifact persistence.
    """

    name: str

    def __init__(self, config: OmniConfig) -> None:
        """Configure the backend from a single :class:`OmniConfig`.

        Read cross-cutting fields directly (``config.run_dir``,
        ``config.stop_at_score``, …) and pop backend-specific keys from
        ``config.config`` onto ``self``. Warn (or raise) on unknown keys in
        ``config.config`` to surface typos.
        """
        ...

    def run(self, task: Task, server: EvalServer) -> Result:
        """Run optimization and return the best candidate.

        Args:
            task: Task definition.
            server: The eval server. Call ``server.evaluate(candidate)`` for
                in-process eval, or use ``server.url`` for HTTP-based eval.
                ``server.budget.max_token_cost`` carries the LLM-spend cap.
        """
        ...

    def process_result(self, result: Result, output_dir: Path | None) -> None:
        """Persist backend-specific artifacts under ``output_dir`` after :meth:`run`.

        Default is a no-op. Backends that produced files, transcripts, or
        workspaces should override this to copy/write them.

        ``output_dir`` is ``None`` when the caller (typically via
        :func:`optimize_anything_with_server`) built an :class:`EvalServer`
        without an ``output_dir`` — backends that need a destination must
        no-op in that case rather than crashing.
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
