"""Typed configuration for :func:`gepa.omni.optimize_anything`.

One flat :class:`OmniConfig` carries every cross-cutting knob (budget, eval
server, cross-backend conveniences) plus a free-form ``config`` dict that
each backend's ``from_config`` factory parses for backend-specific options.

Cross-backend fields (``run_dir``, ``stop_at_score``, ``effort``,
``max_thinking_tokens``, ``sandbox``) live at the top level. Each backend's
factory reads them explicitly — no post-hoc attribute injection by the api.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gepa.omni.backend import Backend


@dataclass
class OmniConfig:
    """Configuration for :func:`gepa.omni.optimize_anything`.

    Attributes:
        backend: Backend name (``"gepa"`` / ``"claude_code"`` /
            ``"meta_harness"``) or a constructed :class:`Backend` instance.
            When a string, the backend's ``from_config`` factory is called
            with this :class:`OmniConfig`.
        max_evals: Server-side cap on eval calls. ``None`` = unlimited.
        max_token_cost: Cap on cumulative LLM USD spend. ``None`` = unlimited.
            Each backend's factory reads this and configures its own enforcer
            (gepa: ``max_reflection_cost``; claude_code/meta_harness:
            ``--max-budget-usd``).
        max_concurrency: Eval server thread-pool size.
        output_dir: Where the eval server persists per-eval JSON,
            ``progress_log.jsonl``, and ``summary.json``. When ``None``, the
            api constructs a timestamped default (``outputs/omni/<task>/
            <backend>/<timestamp>/``) so a run always has somewhere to land.
        tracker: Optional experiment tracker (wandb/mlflow wrapper).
        run_dir: Backend workspace directory. Distinct from ``output_dir``:
            this is where the backend writes its own artifacts (gepa run dir,
            claude_code work dir, etc.). When ``None``, subprocess backends
            use a tempdir; in-process backends skip artifact writes. Set
            explicitly to persist them.
        stop_at_score: Score threshold for early stop. Each backend interprets.
        effort: ``claude --effort low|medium|high|max`` for backends that use
            Claude Code. Independent of ``max_thinking_tokens`` — both can be
            set; behavior with both set is whatever the runtime (claude CLI /
            litellm) does when both are passed.
        max_thinking_tokens: Fixed thinking-token budget. Independent of
            ``effort`` (both can be set together).
        sandbox: Wrap subprocess backends in bwrap (Linux). Default ``False``.
        config: Free-form, backend-specific options. Each backend's factory
            pops the keys it understands; the api warns about leftovers (so
            typos like ``"reflectoin"`` surface immediately).
    """

    backend: str | Backend = "gepa"

    # Budget
    max_evals: int | None = 100
    max_token_cost: float | None = None

    # Eval server
    max_concurrency: int = 8
    output_dir: str | Path | None = None
    tracker: Any | None = None

    # Cross-backend conveniences
    run_dir: str | None = None
    stop_at_score: float | None = None
    effort: str | None = None
    max_thinking_tokens: int | None = None
    sandbox: bool = False

    # Backend-specific. Each backend's factory pops what it knows; the api
    # warns about anything left over.
    config: dict[str, Any] = field(default_factory=dict)
