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
        name: Identifier for this run, used for logging and to lay out the
            default output directory. When ``None``, a name is generated on the
            fly from the engine, a short uuid, and a timestamp (e.g.
            ``"gepa-a1b2c3d4-20260623-153045"``).
        max_evals: Server-side cap on eval calls. ``None`` = unlimited.
        max_token_cost: Proposer-cost cap — cumulative USD an engine may spend
            on its *own* optimizer LLM tokens (reflection, agent). ``None`` =
            unlimited. This is **not** an eval-budget field: the eval server
            never sees proposer spend (a subprocess's cost isn't even knowable
            until it exits). Each engine reads it at construction and enforces
            it itself (gepa: ``max_reflection_cost``; autoresearch/meta_harness:
            ``--max-budget-usd``). At least one of ``max_evals`` /
            ``max_token_cost`` must be set so a run is bounded.
        max_concurrency: Eval server thread-pool size.
        output_dir: Where the eval server persists per-eval JSON,
            ``progress_log.jsonl``, and ``summary.json``. When ``None``, the
            api constructs a timestamped default (``outputs/optimize_anything/<task>/
            <engine>/<timestamp>/``) so a run always has somewhere to land.
        run_dir: Engine workspace directory. Distinct from ``output_dir``:
            this is where the engine writes its own artifacts (gepa run dir,
            autoresearch work dir, etc.). When ``None``, subprocess engines
            use a tempdir; in-process engines skip artifact writes. Set
            explicitly to persist them.
        stop_at_score: Score threshold for early stop. Each engine interprets.
        sandbox: Wrap subprocess engines in bwrap (Linux). Default ``False``.
        engine_config: Engine-specific options that don't generalize across
            engines. Each engine reads the keys it understands and warns about
            leftovers so typos surface. The keys an engine accepts are
            enumerated in that engine's class docstring:

            - ``gepa`` — a ``GEPAConfig``-shaped dict, passed through **1-to-1**.
              Its accepted keys are the live fields of
              :class:`~gepa.gepa_launcher.GEPAConfig` (``engine``,
              ``reflection``, ``merge``, ``refiner``, ``tracking``,
              ``callbacks``, ``stop_callbacks``); the OA layer only overlays the
              budget/run_dir/cost-cap and merges its own callbacks. Set the
              reflection LM, wandb/mlflow tracking, reasoning knobs (via
              ``reflection.reflection_lm_kwargs``, e.g. ``reasoning_effort`` /
              ``thinking``), and (via ``reflection.custom_candidate_proposer``)
              a Claude Code proposer here — see
              :class:`~gepa.oa.engines.gepa.GepaEngine`.
            - ``autoresearch`` — ``model``, ``ralph``, ``max_no_eval_seconds``,
              ``handoffs``, ``effort``, ``max_thinking_tokens`` (see
              ``AutoResearchEngine`` / ``_AR_CONFIG_KEYS``).
            - ``best_of_n`` — ``model``, ``temperature``, ``max_n``,
              ``lm_kwargs``, ``effort``, ``max_thinking_tokens`` (see
              ``BestOfNEngine`` / ``_BON_CONFIG_KEYS``).
            - ``meta_harness`` — ``model``, ``max_iterations``,
              ``max_candidates_per_iter``, ``effort``, ``max_thinking_tokens``
              (see ``MetaHarnessEngine`` / ``_MH_CONFIG_KEYS``).

            ``effort`` (``claude --effort``) and ``max_thinking_tokens`` are
            Claude-CLI knobs, so they live in each agent engine's
            ``engine_config``; the gepa engine takes reasoning knobs through
            ``reflection.reflection_lm_kwargs`` instead.
    """

    engine: str | Engine = "gepa"
    name: str | None = None

    # Budget
    max_evals: int | None = 100
    max_token_cost: float | None = None

    # Eval server
    max_concurrency: int = 8
    output_dir: str | Path | None = None

    # Cross-engine conveniences
    run_dir: str | None = None
    stop_at_score: float | None = None
    sandbox: bool = False

    # Engine-specific. Each engine's factory pops what it knows (see the
    # engine's _<ENGINE>_CONFIG_KEYS tuple) and warns about anything left over.
    engine_config: dict[str, Any] = field(default_factory=dict)
