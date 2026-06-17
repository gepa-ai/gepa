"""Typed configuration for :func:`gepa.optimize_anything.optimize_anything`.

One flat :class:`OptimizeAnythingConfig` carries every cross-cutting knob (budget, eval
server, cross-engine conveniences) plus a free-form ``engine_config`` dict that
each engine parses for engine-specific options.

Cross-engine fields (``run_dir``, ``stop_at_score``, ``effort``,
``max_thinking_tokens``, ``sandbox``) live at the top level. Each engine's
constructor reads them explicitly; the public API does not inject attributes
after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gepa.oa.engine import Engine


@dataclass
class OptimizeAnythingConfig:
    """Configuration for :func:`gepa.optimize_anything.optimize_anything`.

    Attributes:
        engine: Engine name (``"gepa"`` / ``"autoresearch"`` /
            ``"meta_harness"``) or a constructed :class:`Engine` instance.
            When a string, the engine class is instantiated with this
            :class:`OptimizeAnythingConfig`.
        max_evals: Server-side cap on eval calls. ``None`` = unlimited.
        max_token_cost: Cap on cumulative LLM USD spend. ``None`` = unlimited.
            Each engine reads this and configures its own enforcer
            (gepa: ``max_reflection_cost``; autoresearch/meta_harness:
            ``--max-budget-usd``).
        max_concurrency: Eval server thread-pool size.
        output_dir: Where the eval server persists per-eval JSON,
            ``progress_log.jsonl``, and ``summary.json``. When ``None``, the
            api constructs a timestamped default (``outputs/optimize_anything/<task>/
            <engine>/<timestamp>/``) so a run always has somewhere to land.
        tracker: Optional experiment tracker (wandb/mlflow wrapper).
        run_dir: Engine workspace directory. Distinct from ``output_dir``:
            this is where the engine writes its own artifacts (gepa run dir,
            autoresearch work dir, etc.). When ``None``, subprocess engines
            use a tempdir; in-process engines skip artifact writes. Set
            explicitly to persist them.
        stop_at_score: Score threshold for early stop. Each engine interprets.
        effort: ``claude --effort low|medium|high|max`` for engines that use
            Claude Code. Independent of ``max_thinking_tokens`` — both can be
            set; behavior with both set is whatever the runtime (claude CLI /
            litellm) does when both are passed.
        max_thinking_tokens: Fixed thinking-token budget. Independent of
            ``effort`` (both can be set together).
        sandbox: Wrap subprocess engines in bwrap (Linux). Default ``False``.
        engine_config: Free-form, engine-specific options. Each engine reads
            the keys it understands and warns about leftovers so typos surface.
    """

    engine: str | Engine = "gepa"

    # Budget
    max_evals: int | None = 100
    max_token_cost: float | None = None

    # Eval server
    max_concurrency: int = 8
    output_dir: str | Path | None = None
    tracker: Any | None = None

    # Cross-engine conveniences
    run_dir: str | None = None
    stop_at_score: float | None = None
    effort: str | None = None
    max_thinking_tokens: int | None = None
    sandbox: bool = False

    # Engine-specific. Each engine's factory pops what it knows; the api
    # warns about anything left over.
    engine_config: dict[str, Any] = field(default_factory=dict)
