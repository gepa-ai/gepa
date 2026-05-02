"""GEPA backend: runs ``gepa.optimize_anything`` against the omni eval server.

In-process — calls ``server.evaluate(candidate, example)`` directly. Budget is
enforced by the server. ``max_token_cost`` is enforced via the GEPA engine's
``max_reflection_cost`` stopper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gepa.omni._helpers import warn_unknown_config_keys
from gepa.omni.backend import Result
from gepa.omni.budget import BudgetExhausted

if TYPE_CHECKING:
    from gepa.omni.config import OmniConfig
    from gepa.omni.eval_server import EvalServer
    from gepa.omni.task import Task


# Keys this backend understands inside ``OmniConfig.config``.
_GEPA_CONFIG_KEYS: tuple[str, ...] = (
    "engine",
    "reflection",
    "merge",
    "refiner",
    "objective",
    "background",
    "reflection_lm_kwargs",
    "callbacks",
    "claude_code_agent",
)


class GepaBackend:
    """Runs GEPA's ``optimize_anything`` against an omni task.

    Backend-specific keys read from ``OmniConfig.config`` (all optional):

    - ``engine``: kwargs for :class:`gepa.optimize_anything.EngineConfig`.
    - ``reflection``: kwargs for :class:`~gepa.optimize_anything.ReflectionConfig`.
    - ``merge``: kwargs for :class:`~gepa.optimize_anything.MergeConfig` (or ``None``).
    - ``refiner``: kwargs for :class:`~gepa.optimize_anything.RefinerConfig` (or ``None``).
    - ``objective``, ``background``: override ``task.objective`` / ``task.background``.
    - ``reflection_lm_kwargs``: extra kwargs for ``gepa.lm.LM(...)``.
    - ``callbacks``: list of GEPA callbacks.
    - ``claude_code_agent``: kwargs forwarded to
      :class:`~gepa.omni.proposers.ClaudeCodeAgentProposer`. When set, GEPA's
      reflection step is replaced by a Claude Code subprocess that reads the
      agent-readable run-dir tree (``write_agent_state`` is auto-enabled);
      ``run_dir``, ``max_thinking_tokens``, ``effort``, and ``sandbox`` come
      from the surrounding :class:`OmniConfig`. Mutually exclusive with
      ``reflection.custom_candidate_proposer``.
    """

    name = "gepa"

    def __init__(self, config: OmniConfig) -> None:
        extras = config.config
        warn_unknown_config_keys(self.name, extras, _GEPA_CONFIG_KEYS)
        # Cross-cutting (read directly off OmniConfig)
        self.run_dir = config.run_dir
        self.stop_at_score = config.stop_at_score
        self.effort = config.effort
        self.max_thinking_tokens = config.max_thinking_tokens
        self.sandbox = config.sandbox
        # Backend-specific (read out of config.config)
        self.engine: dict[str, Any] = dict(extras.get("engine") or {})
        self.reflection: dict[str, Any] = dict(extras.get("reflection") or {})
        self.merge: dict[str, Any] | None = dict(extras["merge"]) if extras.get("merge") else None
        self.refiner: dict[str, Any] | None = dict(extras["refiner"]) if extras.get("refiner") else None
        self.objective: str | None = extras.get("objective")
        self.background: str | None = extras.get("background")
        self.callbacks: list[Any] = list(extras.get("callbacks") or [])
        self.reflection_lm_kwargs: dict[str, Any] = dict(extras.get("reflection_lm_kwargs") or {})
        self.claude_code_agent: dict[str, Any] | None = (
            dict(extras["claude_code_agent"]) if extras.get("claude_code_agent") else None
        )

    def run(self, task: Task, server: EvalServer) -> Result:
        from gepa.lm import LM
        from gepa.optimize_anything import (
            EngineConfig,
            GEPAConfig,
            MergeConfig,
            RefinerConfig,
            ReflectionConfig,
            optimize_anything,
        )

        budget = server.budget
        objective = self.objective or task.objective
        background = self.background or task.background

        reflection_kwargs = dict(self.reflection)
        reflection_lm: Any | None = None
        agent_proposer: Any | None = None
        if self.claude_code_agent is not None:
            if reflection_kwargs.get("custom_candidate_proposer") is not None:
                raise ValueError("claude_code_agent is mutually exclusive with reflection.custom_candidate_proposer")
            if self.run_dir is None:
                raise ValueError("claude_code_agent requires OmniConfig.run_dir to be set")
            agent_proposer = self._build_claude_code_agent(objective, background)
            reflection_kwargs["custom_candidate_proposer"] = agent_proposer
            # The proposer reads <run_dir>/iterations/, <run_dir>/pareto/.
            # The state.save() pass that writes them is gated on this flag.
            reflection_kwargs.setdefault("reflection_lm", None)

        if "reflection_lm" in reflection_kwargs:
            lm_name = reflection_kwargs["reflection_lm"]
            if isinstance(lm_name, str):
                lm_kwargs = dict(self.reflection_lm_kwargs)
                if self.max_thinking_tokens is not None:
                    lm_kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.max_thinking_tokens}
                    lm_kwargs.pop("reasoning_effort", None)
                reflection_lm = LM(lm_name, **lm_kwargs)
                reflection_kwargs["reflection_lm"] = reflection_lm

        # Default ``cache_evaluation=True`` so re-evaluating the same
        # (candidate, minibatch) pair across iterations is a no-op. Gepa's
        # EngineConfig defaults this to False, which causes the reflective
        # proposer's ``eval_curr`` to call evaluate() on the same selected
        # program every time it gets re-picked from the pareto frontier
        # (and on the seed in iteration 1, because curr_prog == seed).
        # Caller can override via ``config.engine.cache_evaluation=False``.
        engine_kwargs: dict[str, Any] = {
            "cache_evaluation": True,
            **self.engine,
            "run_dir": self.run_dir,
            "max_metric_calls": budget.max_evals,
        }
        if agent_proposer is not None:
            # Proposer reads iterations/, pareto/, history.md off run_dir.
            engine_kwargs["write_agent_state"] = True
        if budget.max_token_cost is not None and reflection_lm is None and agent_proposer is None:
            engine_kwargs["max_reflection_cost"] = budget.max_token_cost

        callbacks = list(self.callbacks)
        cost_callback: _ReflectionCostCallback | None = None
        cost_source = reflection_lm if reflection_lm is not None else agent_proposer
        if cost_source is not None:
            cost_callback = _ReflectionCostCallback(cost_source, server.tracker, output_dir=self.run_dir)
            callbacks.append(cost_callback)
        if agent_proposer is not None:
            # The adapter-curated reflective_dataset (the thing the LM
            # normally sees as ``<side_info>``) is not persisted by GEPA core.
            # Mirror it under ``iterations/NNNNN/reflective_dataset.json`` so
            # the agent proposer can browse past iterations' structured
            # feedback when planning the next mutation.
            callbacks.append(_ReflectiveDatasetDumpCallback(self.run_dir))
        if task.val_set:
            callbacks.append(_ProgressCallback(server, reflection_lm=cost_source))

        stop_callbacks: list[Any] = []
        if self.stop_at_score is not None:
            from gepa.utils.stop_condition import ScoreThresholdStopper

            stop_callbacks.append(ScoreThresholdStopper(self.stop_at_score))
        if budget.max_token_cost is not None and cost_source is not None:
            from gepa.utils.stop_condition import MaxReflectionCostStopper

            stop_callbacks.append(MaxReflectionCostStopper(budget.max_token_cost, reflection_lm=cost_source))

        # Build GEPA's nested dataclasses explicitly. GEPAConfig.__post_init__
        # also accepts dicts, but its type annotations don't, so passing dicts
        # is a runtime feature pyright can't see.
        config = GEPAConfig(
            engine=EngineConfig(**engine_kwargs),
            reflection=ReflectionConfig(**reflection_kwargs),
            merge=MergeConfig(**self.merge) if self.merge else None,
            refiner=RefinerConfig(**self.refiner) if self.refiner else None,
            callbacks=callbacks or None,
            stop_callbacks=stop_callbacks or None,
        )

        # The omni eval server (server.evaluate) is our single budget choke
        # point — gepa.optimize_anything's evaluator goes straight through it.
        def evaluator(candidate, example=None, **kwargs):
            return server.evaluate(candidate, example)

        oa_kwargs: dict[str, Any] = {
            "seed_candidate": task.initial_candidate,
            "evaluator": evaluator,
            "config": config,
        }
        if task.has_dataset:
            if task.train_set:
                oa_kwargs["dataset"] = task.train_set
            # val_set only — test_set is a held-out split reserved for
            # post-run eval and must never leak into the optimization loop.
            if task.val_set:
                oa_kwargs["valset"] = task.val_set
            if "val_set" in task.metadata:
                oa_kwargs.setdefault("valset", task.metadata["val_set"])

        if objective:
            oa_kwargs["objective"] = objective
        if background:
            oa_kwargs["background"] = background

        try:
            gepa_result = optimize_anything(**oa_kwargs)
        except BudgetExhausted:
            gepa_result = None

        adapter_cost = 0.0
        reflection_meta: dict[str, Any] = {}
        if cost_callback is not None:
            reflection_meta = {"reflection_cost_log": cost_callback.cost_log}
            if cost_callback.cost_log:
                adapter_cost = cost_callback.cost_log[-1]["reflection_cost"]

        if gepa_result is not None:
            # gepa_result.best_candidate is str | dict[str,str]; we always pass
            # a str seed_candidate, so the runtime value is str — but the typing
            # admits the dict form. Coerce so omni's Result stays str-typed.
            best = gepa_result.best_candidate
            if isinstance(best, dict):
                best = next(iter(best.values()), "")
            return Result(
                best_candidate=best,
                best_score=gepa_result.val_aggregate_scores[gepa_result.best_idx],
                total_evals=server.budget.used,
                eval_log=server.eval_log,
                metadata={"gepa_result": gepa_result, "adapter_cost": adapter_cost, **reflection_meta},
            )
        return Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={"adapter_cost": adapter_cost, **reflection_meta},
        )

    def process_result(self, result: Result, output_dir: Path | None) -> None:
        return

    def _build_claude_code_agent(self, objective: str | None, background: str | None) -> Any:
        """Construct a :class:`ClaudeCodeAgentProposer` from this backend's config.

        ``OmniConfig.run_dir``, ``effort``, ``max_thinking_tokens``, and
        ``sandbox`` flow in from the surrounding config. Anything in
        ``OmniConfig.config["claude_code_agent"]`` overrides those defaults
        (e.g. set ``"sandbox": False`` to override the global, or
        ``"max_budget_usd"`` for a per-proposer USD cap that doesn't apply
        to the eval server's budget). ``objective`` / ``background`` default
        to the values resolved against the task; explicit keys override.
        """
        from gepa.omni.proposers import ClaudeCodeAgentProposer

        cfg = dict(self.claude_code_agent or {})
        kwargs: dict[str, Any] = {
            "model": cfg.pop("model", "claude-sonnet-4-6"),
            "run_dir": cfg.pop("run_dir", self.run_dir),
            "objective": cfg.pop("objective", objective),
            "background": cfg.pop("background", background),
            "effort": cfg.pop("effort", self.effort),
            "max_thinking_tokens": cfg.pop("max_thinking_tokens", self.max_thinking_tokens),
            "sandbox": cfg.pop("sandbox", self.sandbox),
        }
        # Pass through anything else verbatim (max_budget_usd, subdir_prefix, ...).
        kwargs.update(cfg)
        return ClaudeCodeAgentProposer(**kwargs)


class _ReflectionCostCallback:
    def __init__(self, lm: Any, tracker: Any | None, output_dir: str | Path | None = None) -> None:
        self._lm = lm
        self._tracker = tracker
        self._cost_log: list[dict[str, Any]] = []
        self._log_path: Path | None = None
        if output_dir is not None:
            self._log_path = Path(output_dir) / "reflection_cost_log.jsonl"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def cost_log(self) -> list[dict[str, Any]]:
        return list(self._cost_log)

    def on_iteration_end(self, event: dict[str, Any]) -> None:
        entry = {
            "iteration": event["iteration"],
            "reflection_cost": self._lm.total_cost,
            "reflection_tokens_in": self._lm.total_tokens_in,
            "reflection_tokens_out": self._lm.total_tokens_out,
        }
        self._cost_log.append(entry)
        if self._log_path is not None:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        if self._tracker is not None:
            try:
                self._tracker.log_metrics(
                    {
                        "reflection/cost": entry["reflection_cost"],
                        "reflection/tokens_in": entry["reflection_tokens_in"],
                        "reflection/tokens_out": entry["reflection_tokens_out"],
                    },
                    step=entry["iteration"],
                )
            except Exception:
                pass


class _ProgressCallback:
    def __init__(self, server: Any, reflection_lm: Any = None) -> None:
        self._server = server
        self._reflection_lm = reflection_lm

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        candidate_dict = event.get("candidate", {})
        candidate_text = next(iter(candidate_dict.values()), None) if candidate_dict else None
        reflection_cost = self._reflection_lm.total_cost if self._reflection_lm else 0.0
        self._server.log_progress(event["average_score"], candidate=candidate_text, reflection_cost=reflection_cost)


class _ReflectiveDatasetDumpCallback:
    """Write each iteration's reflective_dataset to disk under run_dir.

    GEPA core writes per-iteration meta / components / trace under
    ``iterations/NNNNN/`` (with ``NNNNN = state.i + 1`` — seed owns id 0),
    but the adapter-curated reflective_dataset (the thing the LM normally
    sees as ``<side_info>``) is never persisted by core. This callback
    closes that gap so the agent proposer can browse past iterations'
    structured feedback via ``iterations/NNNNN/reflective_dataset.json``.

    Only installed when the file-based agent proposer is selected.
    """

    def __init__(self, run_dir: str | Path | None) -> None:
        self._run_dir = Path(run_dir) if run_dir is not None else None

    def on_reflective_dataset_built(self, event: dict[str, Any]) -> None:
        if self._run_dir is None:
            return
        iteration = event.get("iteration")
        dataset = event.get("dataset") or event.get("reflective_dataset")
        if iteration is None or dataset is None:
            return
        # ``event["iteration"]`` is ``ctx.iteration`` — the 1-indexed
        # on-disk iteration id (seed owns 0, first loop proposal is 1).
        target = self._run_dir / "iterations" / f"{int(iteration):05d}" / "reflective_dataset.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dataset, indent=2, default=str))
