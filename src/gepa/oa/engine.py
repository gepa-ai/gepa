"""Engine protocol — the single interface every optimizer plugs into.

A :class:`Engine` is anything that, given a :class:`Task` and an
:class:`EvalServer`, runs an optimization loop and returns a :class:`Result`.

There is **no separate ``Adapter`` sublayer**. Engines call ``server.evaluate``
directly (in-process) or POST to ``server.url`` (subprocess). Budget /
concurrency / tracking all live in the eval server, so all engines share the
same evaluation surface for free.

Construction goes through ``__init__(config: OptimizeAnythingConfig)``. Each engine
reads cross-cutting fields directly (``config.run_dir``, ``config.stop_at_score``,
…) and pops engine-specific keys from ``config.engine_config``, storing what it
needs on ``self``. ``run`` reads ``self.*``. The public API never injects
attributes after construction.

Engines should warn (or raise) about unknown keys in ``config.engine_config``
themselves — they're the only ones who know what they consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from gepa.oa.config import OptimizeAnythingConfig
    from gepa.oa.eval_server import EvalServer
    from gepa.oa.task import Task


class Engine(Protocol):
    """Contract every engine implements.

    Construction takes an :class:`OptimizeAnythingConfig`; the public API calls
    :meth:`run` with a task and an eval server, then :meth:`process_result` for
    post-run artifact persistence.
    """

    name: str

    def __init__(self, config: OptimizeAnythingConfig) -> None:
        """Configure the engine from a single :class:`OptimizeAnythingConfig`.

        Read cross-cutting fields directly (``config.run_dir``,
        ``config.stop_at_score``, …) and pop engine-specific keys from
        ``config.engine_config`` onto ``self``. Warn (or raise) on unknown keys
        in ``config.engine_config`` to surface typos.

        Budget contract: the engine MUST honor ``config.max_token_cost`` (the
        proposer-cost cap — USD on the engine's own optimizer LLM tokens). The
        eval server enforces only the eval-call budget and cannot enforce the
        proposer cap: it never observes the engine's out-of-band LLM spend (and
        a subprocess's cost isn't knowable until it exits). Read the cap here
        and translate it into your native mechanism (``max_reflection_cost``,
        ``--max-budget-usd``, …).
        """
        ...

    def run(self, task: Task, server: EvalServer) -> Result:
        """Run optimization and return the best candidate.

        Args:
            task: Task definition.
            server: The eval server. Call ``server.evaluate(candidate)`` for
                in-process eval, or use ``server.url`` for HTTP-based eval.
                ``server.budget`` enforces the eval-call budget; the
                proposer-cost cap lives on ``config.max_token_cost`` (read at
                construction), not on the server.
        """
        ...

    def process_result(self, result: Result, output_dir: Path | None) -> None:
        """Persist engine-specific artifacts under ``output_dir`` after :meth:`run`.

        Default is a no-op. Engines that produced files, transcripts, or
        workspaces should override this to copy/write them.

        ``output_dir`` is ``None`` when the caller (typically via
        :func:`optimize_anything_with_server`) built an :class:`EvalServer`
        without an ``output_dir`` — engines that need a destination must
        no-op in that case rather than crashing.
        """
        return


@dataclass
class Result:
    """What an engine returns after optimization."""

    best_candidate: str
    best_score: float
    total_evals: int = 0
    eval_log: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
