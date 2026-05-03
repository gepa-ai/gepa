# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Literal, TypeAlias

from gepa.core.adapter import RolloutOutput
from gepa.core.data_loader import DataId
from gepa.gepa_utils import json_default
from gepa.logging.logger import LoggerProtocol

# Types for GEPAState
ProgramIdx = int

# Type aliases
ObjectiveScores: TypeAlias = dict[str, float]
FrontierType: TypeAlias = Literal["instance", "objective", "hybrid", "cartesian"]
"""Strategy for tracking Pareto frontiers: 'instance' (per validation example), 'objective' (per objective metric), 'hybrid' (both), or 'cartesian' (per example x objective)."""
FrontierKey: TypeAlias = DataId | str | tuple[str, DataId] | tuple[str, DataId, str]
"""Key type for frontier mappings depending on frontier_type."""

CandidateHash: TypeAlias = str
CacheKey: TypeAlias = tuple[CandidateHash, DataId]


def _candidate_hash(candidate: dict[str, str]) -> CandidateHash:
    """Compute a deterministic hash of a candidate dictionary."""
    return hashlib.sha256(json.dumps(sorted(candidate.items())).encode()).hexdigest()


_COMPONENT_NAME_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_component_name(name: str) -> str:
    """Collapse component names to filesystem-safe characters."""
    return _COMPONENT_NAME_SANITIZER.sub("_", name)


@dataclass
class CachedEvaluation(Generic[RolloutOutput]):
    """Cached evaluation result for a (candidate, example) pair."""

    output: RolloutOutput
    score: float
    objective_scores: ObjectiveScores | None


@dataclass
class EvaluationCache(Generic[RolloutOutput, DataId]):
    """Cache for storing evaluation results of (candidate, example) pairs."""

    _cache: dict[CacheKey, CachedEvaluation[RolloutOutput]] = field(default_factory=dict)

    def get(self, candidate: dict[str, str], example_id: DataId) -> CachedEvaluation[RolloutOutput] | None:
        """Retrieve cached evaluation result if it exists."""
        return self._cache.get((_candidate_hash(candidate), example_id))

    def put(
        self,
        candidate: dict[str, str],
        example_id: DataId,
        output: RolloutOutput,
        score: float,
        objective_scores: ObjectiveScores | None = None,
    ) -> None:
        """Store an evaluation result in the cache."""
        self._cache[(_candidate_hash(candidate), example_id)] = CachedEvaluation(output, score, objective_scores)

    def get_batch(
        self, candidate: dict[str, str], example_ids: list[DataId]
    ) -> tuple[dict[DataId, CachedEvaluation[RolloutOutput]], list[DataId]]:
        """Look up cached results for a batch. Returns (cached_results, uncached_ids)."""
        h = _candidate_hash(candidate)
        cached, uncached = {}, []
        for eid in example_ids:
            if entry := self._cache.get((h, eid)):
                cached[eid] = entry
            else:
                uncached.append(eid)
        return cached, uncached

    def put_batch(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
        outputs: list[RolloutOutput],
        scores: list[float],
        objective_scores_list: Sequence[ObjectiveScores] | None = None,
    ) -> None:
        """Store evaluation results for a batch of examples."""
        h = _candidate_hash(candidate)
        for i, eid in enumerate(example_ids):
            self._cache[(h, eid)] = CachedEvaluation(
                outputs[i], scores[i], objective_scores_list[i] if objective_scores_list else None
            )

    def evaluate_with_cache_full(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
        fetcher: Callable[[list[DataId]], Any],
        evaluator: Callable[[Any, dict[str, str]], tuple[Any, list[float], Sequence[ObjectiveScores] | None]],
    ) -> tuple[dict[DataId, RolloutOutput], dict[DataId, float], dict[DataId, ObjectiveScores] | None, int]:
        """
        Evaluate using cache, returning full results.

        Returns (outputs_by_id, scores_by_id, objective_scores_by_id, num_actual_evals).
        """
        cached, uncached_ids = self.get_batch(candidate, example_ids)

        outputs_by_id: dict[DataId, RolloutOutput] = {eid: c.output for eid, c in cached.items()}
        scores_by_id: dict[DataId, float] = {eid: c.score for eid, c in cached.items()}
        objective_by_id: dict[DataId, ObjectiveScores] | None = None

        # Populate objective scores from cache
        for eid, c in cached.items():
            if c.objective_scores is not None:
                objective_by_id = objective_by_id or {}
                objective_by_id[eid] = c.objective_scores

        # Evaluate uncached examples
        if uncached_ids:
            batch = fetcher(uncached_ids)
            outputs, scores, obj_scores = evaluator(batch, candidate)
            for idx, eid in enumerate(uncached_ids):
                outputs_by_id[eid] = outputs[idx]
                scores_by_id[eid] = scores[idx]
                if obj_scores is not None:
                    objective_by_id = objective_by_id or {}
                    objective_by_id[eid] = obj_scores[idx]
            self.put_batch(candidate, uncached_ids, outputs, scores, obj_scores)

        return outputs_by_id, scores_by_id, objective_by_id, len(uncached_ids)


@dataclass(slots=True)
class ValsetEvaluation(Generic[RolloutOutput, DataId]):
    """Container for evaluation results on a validation set batch."""

    outputs_by_val_id: dict[DataId, RolloutOutput]
    scores_by_val_id: dict[DataId, float]
    objective_scores_by_val_id: dict[DataId, ObjectiveScores] | None = None
    # Populated only when the engine is run with ``write_agent_state=True`` —
    # full valset trajectories are expensive, so default eval paths skip them.
    trajectories_by_val_id: dict[DataId, Any] | None = None


class GEPAState(Generic[RolloutOutput, DataId]):
    """Internal persistent state of a GEPA optimization run.

    Tracks all explored candidates, their per-example and per-objective scores,
    Pareto frontiers, evaluation budget, and optional evaluation cache.
    Saved/loaded automatically when ``EngineConfig.run_dir`` is set.

    Users interact with this indirectly via :class:`~gepa.core.result.GEPAResult`
    returned by :func:`~gepa.optimize_anything.optimize_anything`.
    """

    _VALIDATION_SCHEMA_VERSION: ClassVar[int] = 6
    # Attributes that are runtime-only and should not be serialized (e.g., callback hooks, caches)
    _EXCLUDED_FROM_SERIALIZATION: ClassVar[frozenset[str]] = frozenset({"_budget_hooks"})

    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[ProgramIdx | None]]
    prog_candidate_val_subscores: list[dict[DataId, float]]
    prog_candidate_objective_scores: list[ObjectiveScores]
    # On-disk iteration id for each candidate (indexed by candidate idx).
    # Seed = 0; accepted loop candidates get the 1-indexed iteration id
    # at which they were discovered (``ctx.iteration`` ==
    # ``state.i + 1``). Maintained in lockstep with ``program_candidates``
    # so lookups are O(1) instead of requiring a trace scan.
    iteration_ids_by_candidate_idx: list[int]

    pareto_front_valset: dict[DataId, float]
    program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]]
    objective_pareto_front: ObjectiveScores
    program_at_pareto_front_objectives: dict[str, set[ProgramIdx]]
    pareto_front_cartesian: dict[tuple[DataId, str], float]
    program_at_pareto_front_cartesian: dict[tuple[DataId, str], set[ProgramIdx]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    i: int
    num_full_ds_evals: int

    total_num_evals: int

    num_metric_calls_by_discovery: list[int]

    full_program_trace: list[dict[str, Any]]
    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None

    validation_schema_version: int

    # Optional evaluation cache for (candidate, example) pairs
    evaluation_cache: "EvaluationCache[RolloutOutput, DataId] | None"

    # Opaque bag for adapter-specific persistent state.
    # Core GEPA never inspects this; adapters read/write via get_adapter_state()/set_adapter_state().
    adapter_state: dict[str, Any]

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_evaluation: ValsetEvaluation[RolloutOutput, DataId],
        track_best_outputs: bool = False,
        frontier_type: FrontierType = "instance",
        evaluation_cache: "EvaluationCache[RolloutOutput, DataId] | None" = None,
    ):
        self.program_candidates = [dict(seed_candidate)]
        # Seed owns on-disk iteration id 0.
        self.iteration_ids_by_candidate_idx = [0]
        self.prog_candidate_val_subscores = [dict(base_evaluation.scores_by_val_id)]

        base_objective_aggregates = self._aggregate_objective_scores(base_evaluation.objective_scores_by_val_id)
        self.prog_candidate_objective_scores = [base_objective_aggregates]

        self.parent_program_for_candidate = [[None]]

        self.frontier_type: FrontierType = frontier_type
        self.pareto_front_valset = dict(base_evaluation.scores_by_val_id)
        self.program_at_pareto_front_valset = {val_id: {0} for val_id in base_evaluation.scores_by_val_id.keys()}
        self.objective_pareto_front = dict(base_objective_aggregates)
        self.program_at_pareto_front_objectives = {objective: {0} for objective in base_objective_aggregates.keys()}

        # Validate that objective scores are provided for frontier types that require them
        if frontier_type in ("objective", "hybrid", "cartesian"):
            if not base_evaluation.objective_scores_by_val_id:
                raise ValueError(
                    f"frontier_type='{frontier_type}' requires objective_scores to be provided by the evaluator, "
                    f"but none were found. Use an evaluator that returns objective_scores or use frontier_type='instance'."
                )

        # Cartesian frontier will be base_evaluation.objective_scores_by_val_id
        if frontier_type == "cartesian":
            assert base_evaluation.objective_scores_by_val_id is not None  # Already validated above
            self.pareto_front_cartesian = {
                (val_id, objective): objective_score
                for val_id, objective_scores in base_evaluation.objective_scores_by_val_id.items()
                for objective, objective_score in objective_scores.items()
            }
            self.program_at_pareto_front_cartesian = {
                (val_id, objective): {0}
                for val_id, objective_scores in base_evaluation.objective_scores_by_val_id.items()
                for objective in objective_scores.keys()
            }
        else:
            self.pareto_front_cartesian = {}
            self.program_at_pareto_front_cartesian = {}

        self.list_of_named_predictors = list(seed_candidate.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1

        self.num_metric_calls_by_discovery = [0]

        if track_best_outputs:
            self.best_outputs_valset = {
                val_id: [(0, output)] for val_id, output in base_evaluation.outputs_by_val_id.items()
            }
        else:
            self.best_outputs_valset = None

        self.full_program_trace = []
        self.validation_schema_version = self._VALIDATION_SCHEMA_VERSION
        self.evaluation_cache = evaluation_cache
        self.adapter_state: dict[str, Any] = {}

    def is_consistent(self) -> bool:
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)
        assert len(self.program_candidates) == len(self.prog_candidate_val_subscores)
        assert len(self.program_candidates) == len(self.prog_candidate_objective_scores)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)
        assert len(self.program_candidates) == len(self.iteration_ids_by_candidate_idx)

        assert len(self.pareto_front_valset) == len(self.program_at_pareto_front_valset)
        assert set(self.pareto_front_valset.keys()) == set(self.program_at_pareto_front_valset.keys())
        assert set(self.objective_pareto_front.keys()) == set(self.program_at_pareto_front_objectives.keys())

        for front in self.program_at_pareto_front_valset.values():
            for prog_idx in front:
                assert prog_idx < len(self.program_candidates), (
                    "Program index in valset pareto front exceeds number of program candidates"
                )

        return True

    # Budget Hook Mechanism
    def add_budget_hook(self, hook: Callable[[int, int], None]) -> None:
        """Register a callback to be called whenever total_num_evals changes.

        Args:
            hook: A callable that receives (new_total, delta) when evals are incremented.
        """
        if not hasattr(self, "_budget_hooks"):
            self._budget_hooks: list[Callable[[int, int], None]] = []
        self._budget_hooks.append(hook)

    def increment_evals(self, count: int) -> None:
        """Increment total_num_evals and notify all registered hooks.

        Args:
            count: Number of evaluations to add.
        """
        self.total_num_evals += count
        # Lazy init handles states loaded from disk (which won't have _budget_hooks)
        hooks = getattr(self, "_budget_hooks", None)
        if hooks:
            for hook in hooks:
                hook(self.total_num_evals, count)

    def _atomic_write_json(self, run_dir: str, rel_path: str, data: Any) -> None:
        target_path = os.path.join(run_dir, rel_path)
        parent = os.path.dirname(target_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = target_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=json_default)
        os.replace(tmp_path, target_path)

    def _atomic_write_text(self, run_dir: str, rel_path: str, data: str) -> None:
        target_path = os.path.join(run_dir, rel_path)
        parent = os.path.dirname(target_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = target_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(data)
        os.replace(tmp_path, target_path)

    def save(
        self,
        run_dir: str | None,
        *,
        use_cloudpickle: bool = False,
        write_agent_state: bool = False,
    ) -> None:
        if run_dir is None:
            return
        if use_cloudpickle:
            try:
                import cloudpickle as pickle  # type: ignore[import-not-found]
            except ModuleNotFoundError:
                import pickle
                import warnings

                warnings.warn(
                    "cloudpickle is not installed; falling back to standard pickle. "
                    "Install it with: pip install gepa[full]  or  pip install cloudpickle",
                    stacklevel=2,
                )
        else:
            import pickle
        # Exclude runtime-only attributes that can't be serialized (e.g., callback hooks)
        serialized = {k: v for k, v in self.__dict__.items() if k not in self._EXCLUDED_FROM_SERIALIZATION}
        serialized["validation_schema_version"] = GEPAState._VALIDATION_SCHEMA_VERSION
        target_path = os.path.join(run_dir, "gepa_state.bin")
        tmp_path = target_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(serialized, f)
        except Exception as e:
            if not use_cloudpickle:
                raise type(e)(
                    f"{e}\n\nHint: standard pickle failed to serialize the GEPA state. "
                    "Try setting use_cloudpickle=True in EngineConfig, which can serialize "
                    "more object types (lambdas, closures, etc.). "
                    "Install it with: pip install gepa[full]  or  pip install cloudpickle"
                ) from e
            raise
        os.replace(tmp_path, target_path)

        # Save run log and candidates as human-readable JSON
        if self.full_program_trace:
            self._atomic_write_json(run_dir, "run_log.json", self.full_program_trace)
        if self.program_candidates:
            self._atomic_write_json(run_dir, "candidates.json", self.program_candidates)

        # Opt-in directory layout for agent consumption
        if write_agent_state:
            self._save_agent_directory(run_dir)

    def _save_agent_directory(self, run_dir: str) -> None:
        """Write agent-readable state as a directory of small files.

        Partitions the run's state into per-iteration subdirs so an agent
        (e.g. Claude Code) can navigate a single timeline — both accepted
        candidates *and* rejected proposals live under ``iterations/NNNNN/``.
        ``iterations/00000/`` is always the seed (``candidate_idx=0``); subsequent
        directories correspond to ``state.i + 1`` (the loop iteration that
        produced the proposal, shifted so the seed owns id 0). The pickled
        ``gepa_state.bin`` remains the source of truth for resume; this tree
        is output-only.
        """
        cand_to_iter = self._candidate_idx_to_iteration_id()
        self._save_seed_iteration_dir(run_dir)
        self._save_proposal_iteration_dirs(run_dir, cand_to_iter)
        self._save_pareto_dir(run_dir, cand_to_iter)
        self._save_index(run_dir, cand_to_iter)

    def iteration_id_for_candidate_idx(self, candidate_idx: int) -> int | None:
        """On-disk iteration id for the accepted candidate at ``candidate_idx``.

        O(1) lookup against :attr:`iteration_ids_by_candidate_idx`, which is
        maintained in lockstep with :attr:`program_candidates` by
        :meth:`update_state_with_new_program`. Returns ``None`` for
        out-of-range indices.
        """
        if candidate_idx < 0 or candidate_idx >= len(self.iteration_ids_by_candidate_idx):
            return None
        return self.iteration_ids_by_candidate_idx[candidate_idx]

    def _candidate_idx_to_iteration_id(self) -> dict[int, int]:
        """Map candidate idx (internal) → on-disk iteration id.

        Backed by :attr:`iteration_ids_by_candidate_idx`, so this is just a
        zipped view — no trace scan needed.
        """
        return dict(enumerate(self.iteration_ids_by_candidate_idx))

    def _save_seed_iteration_dir(self, run_dir: str) -> None:
        """Write ``iterations/00000/`` — the seed candidate (candidate_idx 0)."""
        base = "iterations/00000"
        # Seed dir is immutable once written; skip on subsequent saves so
        # per-iteration cost stays O(1) instead of O(N).
        if os.path.exists(os.path.join(run_dir, base, "meta.json")):
            return
        avg_score, coverage = self.get_program_average_val_subset(0)
        metric_calls = self.num_metric_calls_by_discovery[0] if self.num_metric_calls_by_discovery else 0
        obj_scores = self.prog_candidate_objective_scores[0] if self.prog_candidate_objective_scores else {}
        self._atomic_write_json(
            run_dir,
            f"{base}/meta.json",
            {
                "iteration_id": 0,
                "is_seed": True,
                "accepted": True,
                "candidate_idx": 0,
                "parent_iteration_ids": [],
                "avg_val_score": avg_score,
                "num_val_scored": coverage,
                "metric_calls_so_far": metric_calls,
                "objective_scores": obj_scores,
            },
        )
        self._atomic_write_json(
            run_dir,
            f"{base}/val_scores.json",
            {str(k): v for k, v in self.prog_candidate_val_subscores[0].items()},
        )
        self._save_components_dir(run_dir, base, self.program_candidates[0])

    def _save_proposal_iteration_dirs(self, run_dir: str, cand_to_iter: dict[int, int]) -> None:
        """Write one ``iterations/NNNNN/`` per loop trace entry.

        Covers both accepted and rejected proposals. Iteration id is
        ``state.i + 1`` so the seed keeps id ``0``.
        """
        for entry in self.full_program_trace:
            trace_i = entry.get("i")
            if trace_i is None:
                continue
            iter_id = trace_i + 1
            base = f"iterations/{iter_id:05d}"
            # Trace entries are finalized in the body of iteration K-1 before
            # the iteration-K save runs (state.save fires at the top of the
            # loop, after the previous body has fully populated its entry).
            # Once meta.json exists the dir is immutable — skip rewriting so
            # save cost stays O(1) per iteration instead of O(N).
            if os.path.exists(os.path.join(run_dir, base, "meta.json")):
                continue
            accepted = entry.get("proposal_accepted")
            candidate_idx = entry.get("new_program_idx") if accepted else None
            parent_candidate_idx = entry.get("selected_program_candidate")
            parent_iter_ids = (
                [cand_to_iter[parent_candidate_idx]]
                if parent_candidate_idx is not None and parent_candidate_idx in cand_to_iter
                else []
            )

            # The candidate text to archive under components/: for accepted
            # iterations this is the newly accepted program; for rejected
            # iterations it's the proposed (now discarded) candidate.
            components: dict[str, str] | None = None
            if accepted and candidate_idx is not None and candidate_idx < len(self.program_candidates):
                components = self.program_candidates[candidate_idx]
            elif "proposed_candidate" in entry and isinstance(entry["proposed_candidate"], dict):
                components = entry["proposed_candidate"]

            meta: dict[str, Any] = {
                "iteration_id": iter_id,
                "trace_i": trace_i,
                "accepted": bool(accepted) if accepted is not None else None,
                "parent_iteration_ids": parent_iter_ids,
                "candidate_idx": candidate_idx,
                "subsample_ids": entry.get("subsample_ids"),
                "subsample_scores_before": entry.get("subsample_scores"),
                "subsample_scores_after": entry.get("new_subsample_scores"),
            }
            if accepted and candidate_idx is not None and candidate_idx < len(self.prog_candidate_val_subscores):
                avg_score, coverage = self.get_program_average_val_subset(candidate_idx)
                meta["avg_val_score"] = avg_score
                meta["num_val_scored"] = coverage
                obj_scores = self.prog_candidate_objective_scores[candidate_idx]
                if obj_scores:
                    meta["objective_scores"] = obj_scores
                metric_calls = (
                    self.num_metric_calls_by_discovery[candidate_idx]
                    if candidate_idx < len(self.num_metric_calls_by_discovery)
                    else None
                )
                if metric_calls is not None:
                    meta["metric_calls_to_discover"] = metric_calls
            self._atomic_write_json(run_dir, f"{base}/meta.json", meta)

            if components is not None:
                self._save_components_dir(run_dir, base, components)

            if accepted and candidate_idx is not None and candidate_idx < len(self.prog_candidate_val_subscores):
                self._atomic_write_json(
                    run_dir,
                    f"{base}/val_scores.json",
                    {str(k): v for k, v in self.prog_candidate_val_subscores[candidate_idx].items()},
                )

    def _save_components_dir(self, run_dir: str, base: str, components: dict[str, str]) -> None:
        """Write ``<base>/components/<stem>.txt`` plus ``_index.json``."""
        index_map: dict[str, str] = {}
        used: set[str] = set()
        for i, (name, text) in enumerate(components.items()):
            safe = _sanitize_component_name(name)
            if not safe or safe in used:
                safe = f"component_{i:02d}"
            used.add(safe)
            index_map[name] = f"{safe}.txt"
            self._atomic_write_text(run_dir, f"{base}/components/{safe}.txt", text)
        self._atomic_write_json(run_dir, f"{base}/components/_index.json", index_map)

    def _save_pareto_dir(self, run_dir: str, cand_to_iter: dict[int, int]) -> None:
        """Write Pareto frontier files keyed by iteration id (not candidate idx).

        Keeping the on-disk representation iteration-id-keyed means the agent
        never sees the internal candidate numbering; everything is anchored
        on the ``iterations/`` timeline.
        """
        def _iter_ids(candidate_idxs: "set[int]") -> list[int]:
            return sorted({cand_to_iter[c] for c in candidate_idxs if c in cand_to_iter})

        if self.frontier_type in ("instance", "hybrid"):
            instance_front: dict[str, Any] = {}
            for val_id, best_score in self.pareto_front_valset.items():
                instance_front[str(val_id)] = {
                    "best_score": best_score,
                    "best_iteration_ids": _iter_ids(self.program_at_pareto_front_valset.get(val_id, set())),
                }
            self._atomic_write_json(run_dir, "pareto/instance_front.json", instance_front)

        if self.frontier_type in ("objective", "hybrid"):
            obj_front: dict[str, Any] = {}
            for obj_name, best_score in self.objective_pareto_front.items():
                obj_front[obj_name] = {
                    "best_score": best_score,
                    "best_iteration_ids": _iter_ids(self.program_at_pareto_front_objectives.get(obj_name, set())),
                }
            self._atomic_write_json(run_dir, "pareto/objective_front.json", obj_front)

        if self.frontier_type == "cartesian":
            cartesian_front: dict[str, Any] = {}
            for key, best_score in self.pareto_front_cartesian.items():
                cartesian_front[str(key)] = {
                    "best_score": best_score,
                    "best_iteration_ids": _iter_ids(self.program_at_pareto_front_cartesian.get(key, set())),
                }
            self._atomic_write_json(run_dir, "pareto/cartesian_front.json", cartesian_front)

    def _save_index(self, run_dir: str, cand_to_iter: dict[int, int]) -> None:
        """Write the small top-level index that points at the rest of the tree."""
        num_candidates = len(self.program_candidates)

        hardest: list[dict[str, Any]] = []
        if self.pareto_front_valset:
            sorted_examples = sorted(self.pareto_front_valset.items(), key=lambda x: x[1])
            for val_id, score in sorted_examples[:20]:
                best_iters = sorted({
                    cand_to_iter[c]
                    for c in self.program_at_pareto_front_valset.get(val_id, set())
                    if c in cand_to_iter
                })
                hardest.append({
                    "val_id": str(val_id),
                    "best_score": score,
                    "best_iteration_ids": best_iters,
                })

        full_scores = self.program_full_scores_val_set
        best_candidate_idx = (
            int(max(range(num_candidates), key=lambda i: full_scores[i]))
            if num_candidates > 0
            else 0
        )
        best_iter_id = cand_to_iter.get(best_candidate_idx)

        front_candidate_idxs: set[int] = set()
        for prog_set in self.get_pareto_front_mapping().values():
            front_candidate_idxs.update(prog_set)
        front_iter_ids = sorted({cand_to_iter[c] for c in front_candidate_idxs if c in cand_to_iter})

        index: dict[str, Any] = {
            "schema_version": 3,
            "iteration": self.i,
            "total_metric_calls": self.total_num_evals,
            "num_full_val_evals": self.num_full_ds_evals,
            "frontier_type": self.frontier_type,
            "component_names": self.list_of_named_predictors,
            "summary": {
                "num_iterations": len(cand_to_iter),
                "best_iteration_id": best_iter_id,
                "best_avg_score": full_scores[best_candidate_idx] if num_candidates > 0 else None,
                "pareto_front_iteration_ids": front_iter_ids,
                "hardest_examples": hardest,
            },
            "layout": {
                "iterations_dir": "iterations/",
                "pareto_dir": "pareto/",
            },
        }

        self._atomic_write_json(run_dir, "gepa_state.json", index)

    @staticmethod
    def load(run_dir: str) -> "GEPAState[RolloutOutput, DataId]":
        with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
            import pickle

            data = pickle.load(f)

        # handle schema migration
        version = data.get("validation_schema_version")
        if version is None or version < 2:
            GEPAState._migrate_from_legacy_state_v0(data)
            version = data.get("validation_schema_version")
        if version is None or version < GEPAState._VALIDATION_SCHEMA_VERSION:
            GEPAState._upgrade_state_dict(data)

        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(data)

        state.validation_schema_version = GEPAState._VALIDATION_SCHEMA_VERSION
        assert len(state.program_candidates) == len(state.prog_candidate_val_subscores)
        assert len(state.program_candidates) == len(state.prog_candidate_objective_scores)
        assert len(state.program_candidates) == len(state.num_metric_calls_by_discovery)
        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)
        assert len(state.program_candidates) == len(state.iteration_ids_by_candidate_idx)
        assert len(state.pareto_front_valset) == len(state.program_at_pareto_front_valset)
        assert set(state.pareto_front_valset.keys()) == set(state.program_at_pareto_front_valset.keys())
        assert set(state.objective_pareto_front.keys()) == set(state.program_at_pareto_front_objectives.keys())
        assert isinstance(state.adapter_state, dict)
        return state

    @staticmethod
    def _migrate_from_legacy_state_v0(d: dict[str, Any]) -> None:
        assert isinstance(d, dict)
        assert "prog_candidate_val_subscores" in d
        assert isinstance(d["prog_candidate_val_subscores"], list)
        assert all(isinstance(scores, list) for scores in d["prog_candidate_val_subscores"])
        legacy_scores: list[list[float]] = d.pop("prog_candidate_val_subscores", [])
        d["prog_candidate_val_subscores"] = [dict(enumerate(scores)) for scores in legacy_scores]

        pareto_front = d.get("pareto_front_valset")
        if isinstance(pareto_front, list):
            d["pareto_front_valset"] = dict(enumerate(pareto_front))

        program_at_front = d.get("program_at_pareto_front_valset")
        if isinstance(program_at_front, list):
            d["program_at_pareto_front_valset"] = {idx: set(front) for idx, front in enumerate(program_at_front)}

        best_outputs = d.get("best_outputs_valset")
        if isinstance(best_outputs, list):
            d["best_outputs_valset"] = {idx: list(outputs) for idx, outputs in enumerate(best_outputs)}

        d["validation_schema_version"] = 2

    @staticmethod
    def _upgrade_state_dict(d: dict[str, Any]) -> None:
        num_candidates = len(d.get("program_candidates", []))
        if "prog_candidate_objective_scores" not in d:
            d["prog_candidate_objective_scores"] = [{} for _ in range(num_candidates)]
        if "objective_pareto_front" not in d:
            d["objective_pareto_front"] = {}
        if "program_at_pareto_front_objectives" not in d:
            d["program_at_pareto_front_objectives"] = {}
        if "frontier_type" not in d:
            d["frontier_type"] = "instance"
            # Since frontier_type instance does not require "pareto_front_cartesian" and "program_at_pareto_front_cartesian", we can safely set them to empty dicts.
            d["pareto_front_cartesian"] = {}
            d["program_at_pareto_front_cartesian"] = {}
        # evaluation_cache is not persisted across runs by default; initialize to None if missing
        if "evaluation_cache" not in d:
            d["evaluation_cache"] = None
        if "adapter_state" not in d:
            d["adapter_state"] = {}
        if "iteration_ids_by_candidate_idx" not in d:
            # v5 → v6: reconstruct on-disk iteration ids by walking the
            # full trace. Seed gets 0; every accepted trace entry maps its
            # ``new_program_idx`` to ``trace_i + 1``. Missing entries fall
            # back to ``-1`` (shouldn't happen for a consistent state, but
            # keeps the list length aligned with ``program_candidates``).
            mapping = {0: 0}
            for entry in d.get("full_program_trace", []):
                if not entry.get("proposal_accepted"):
                    continue
                cand_idx = entry.get("new_program_idx")
                trace_i = entry.get("i")
                if cand_idx is None or trace_i is None:
                    continue
                mapping[cand_idx] = trace_i + 1
            d["iteration_ids_by_candidate_idx"] = [mapping.get(i, -1) for i in range(num_candidates)]
        d["validation_schema_version"] = GEPAState._VALIDATION_SCHEMA_VERSION

    @staticmethod
    def _aggregate_objective_scores(
        val_objective_scores: dict[DataId, ObjectiveScores] | None,
    ) -> ObjectiveScores:
        if not val_objective_scores:
            return {}
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for objective_dict in val_objective_scores.values():
            for objective, score in objective_dict.items():
                totals[objective] = totals.get(objective, 0.0) + score
                counts[objective] = counts.get(objective, 0) + 1
        return {
            objective: totals[objective] / counts[objective] for objective in totals.keys() if counts[objective] > 0
        }

    def get_program_average_val_subset(self, program_idx: int) -> tuple[float, int]:
        # TODO: This should be only used/handled by the val_evaluation_policy, and never used directly.
        scores = self.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf"), 0
        num_samples = len(scores)
        avg = sum(scores.values()) / num_samples
        return avg, num_samples

    @property
    def valset_evaluations(self) -> dict[DataId, list[ProgramIdx]]:
        """
        Valset examples by id and programs that have evaluated them. Keys include only validation
        ids that have been scored at least once.
        """
        result: dict[DataId, list[ProgramIdx]] = defaultdict(list)
        for program_idx, val_scores in enumerate(self.prog_candidate_val_subscores):
            for val_id in val_scores.keys():
                result[val_id].append(program_idx)
        return result

    @property
    def program_full_scores_val_set(self) -> list[float]:
        # TODO: This should be using the val_evaluation_policy instead of the get_program_average_val_subset method to calculate the scores.
        return [
            self.get_program_average_val_subset(program_idx)[0]
            for program_idx in range(len(self.prog_candidate_val_subscores))
        ]

    @property
    def per_program_tracked_scores(self) -> list[float]:
        return [
            self.get_program_average_val_subset(program_idx)[0]
            for program_idx in range(len(self.prog_candidate_val_subscores))
        ]

    def _update_objective_pareto_front(self, objective_scores: ObjectiveScores, program_idx: ProgramIdx) -> None:
        if not objective_scores:
            return
        for objective, score in objective_scores.items():
            prev_score = self.objective_pareto_front.get(objective, float("-inf"))
            if score > prev_score:
                self.objective_pareto_front[objective] = score
                self.program_at_pareto_front_objectives[objective] = {program_idx}
            elif score == prev_score:
                front = self.program_at_pareto_front_objectives.setdefault(objective, set())
                front.add(program_idx)

    def _update_pareto_front_for_val_id(
        self,
        val_id: DataId,
        score: float,
        program_idx: ProgramIdx,
        output: RolloutOutput | None,
        run_dir: str | None,
        iteration: int,
    ) -> None:
        prev_score = self.pareto_front_valset.get(val_id, float("-inf"))
        if score > prev_score:
            self.pareto_front_valset[val_id] = score
            self.program_at_pareto_front_valset[val_id] = {program_idx}
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id] = [(program_idx, output)]
                if run_dir is not None:
                    task_dir = os.path.join(run_dir, "generated_best_outputs_valset", f"task_{val_id}")
                    os.makedirs(task_dir, exist_ok=True)
                    with open(os.path.join(task_dir, f"iter_{iteration}_prog_{program_idx}.json"), "w") as fout:
                        json.dump(output, fout, indent=4, default=json_default)
        elif score == prev_score:
            pareto_front = self.program_at_pareto_front_valset.setdefault(val_id, set())
            pareto_front.add(program_idx)
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id].append((program_idx, output))

    def _update_pareto_front_for_cartesian(
        self,
        val_id: DataId,
        objective: str,
        objective_score: float,
        program_idx: ProgramIdx,
    ) -> None:
        prev_score = self.pareto_front_cartesian.get((val_id, objective), float("-inf"))
        if objective_score > prev_score:
            self.pareto_front_cartesian[(val_id, objective)] = objective_score
            self.program_at_pareto_front_cartesian[(val_id, objective)] = {program_idx}
        elif objective_score == prev_score:
            front = self.program_at_pareto_front_cartesian.setdefault((val_id, objective), set())
            front.add(program_idx)

    def update_state_with_new_program(
        self,
        parent_program_idx: list[ProgramIdx],
        new_program: dict[str, str],
        valset_evaluation: ValsetEvaluation,
        run_dir: str | None,
        num_metric_calls_by_discovery_of_new_program: int,
        iteration: int | None = None,
    ) -> ProgramIdx:
        # ``iteration`` is the 1-indexed on-disk iteration id for the
        # proposal that produced this candidate — same numbering
        # ``ctx.iteration`` uses (``state.i + 1``). Seed is iter id 0, so
        # loop iterations start at 1. Falls back to ``self.i + 1`` when
        # the caller hasn't plumbed it.
        if iteration is None:
            iteration = self.i + 1
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(dict(new_program))
        self.iteration_ids_by_candidate_idx.append(iteration)
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery_of_new_program)

        max_predictor_id = max(
            [self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx],
            default=0,
        )
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_predictor_id)
        self.parent_program_for_candidate.append(list(parent_program_idx))

        valset_scores = dict(valset_evaluation.scores_by_val_id)
        self.prog_candidate_val_subscores.append(valset_scores)
        objective_scores = self._aggregate_objective_scores(valset_evaluation.objective_scores_by_val_id)
        self.prog_candidate_objective_scores.append(objective_scores)

        for val_id, score in valset_scores.items():
            output = valset_evaluation.outputs_by_val_id.get(val_id) if valset_evaluation.outputs_by_val_id else None
            self._update_pareto_front_for_val_id(
                val_id,
                score,
                new_program_idx,
                output,
                run_dir,
                self.i + 1,
            )

        self._update_objective_pareto_front(objective_scores, new_program_idx)

        if self.frontier_type in ("objective", "hybrid", "cartesian"):
            if not valset_evaluation.objective_scores_by_val_id:
                raise ValueError(
                    f"frontier_type='{self.frontier_type}' requires objective_scores to be provided by the evaluator, "
                    f"but none were found in the evaluation result."
                )

        if self.frontier_type == "cartesian":
            assert valset_evaluation.objective_scores_by_val_id is not None  # Validated above
            for val_id, objective_scores in valset_evaluation.objective_scores_by_val_id.items():
                for objective, objective_score in objective_scores.items():
                    self._update_pareto_front_for_cartesian(
                        val_id,
                        objective,
                        objective_score,
                        new_program_idx,
                    )

        return new_program_idx

    def _get_pareto_front_mapping(self, frontier_type: FrontierType) -> dict[FrontierKey, set[ProgramIdx]]:
        if frontier_type == "instance":
            return {val_id: set(front) for val_id, front in self.program_at_pareto_front_valset.items()}
        if frontier_type == "objective":
            return {objective: set(front) for objective, front in self.program_at_pareto_front_objectives.items()}
        if frontier_type == "hybrid":
            combined: dict[FrontierKey, set[ProgramIdx]] = {
                ("val_id", val_id): set(front) for val_id, front in self.program_at_pareto_front_valset.items()
            }
            for objective, front in self.program_at_pareto_front_objectives.items():
                combined[("objective", objective)] = set(front)
            return combined
        if frontier_type == "cartesian":
            return {
                ("cartesian", val_id, objective): set(front)
                for (val_id, objective), front in self.program_at_pareto_front_cartesian.items()
            }
        raise ValueError(f"Unknown frontier_type: {frontier_type}")

    def get_pareto_front_mapping(self) -> dict[FrontierKey, set[ProgramIdx]]:
        """Return frontier key to best-program-indices mapping based on configured frontier_type."""
        return self._get_pareto_front_mapping(self.frontier_type)

    def cached_evaluate(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
        fetcher: Callable[[list[DataId]], Any],
        evaluator: Callable[[Any, dict[str, str]], tuple[Any, list[float], Sequence[ObjectiveScores] | None]],
    ) -> tuple[list[float], int]:
        """Evaluate with optional caching. Returns (scores, num_actual_evals)."""
        _, scores_by_id, _, num_actual_evals = self.cached_evaluate_full(candidate, example_ids, fetcher, evaluator)
        return [scores_by_id[eid] for eid in example_ids], num_actual_evals

    def cached_evaluate_full(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
        fetcher: Callable[[list[DataId]], Any],
        evaluator: Callable[[Any, dict[str, str]], tuple[Any, list[float], Sequence[ObjectiveScores] | None]],
    ) -> tuple[dict[DataId, RolloutOutput], dict[DataId, float], dict[DataId, ObjectiveScores] | None, int]:
        """Evaluate with optional caching, returning full results."""
        if self.evaluation_cache is not None:
            return self.evaluation_cache.evaluate_with_cache_full(candidate, example_ids, fetcher, evaluator)
        batch = fetcher(example_ids)
        outputs, scores, objective_scores = evaluator(batch, candidate)
        outputs_by_id = dict(zip(example_ids, outputs, strict=False))
        scores_by_id = dict(zip(example_ids, scores, strict=False))
        objective_by_id = dict(zip(example_ids, objective_scores, strict=False)) if objective_scores else None
        return outputs_by_id, scores_by_id, objective_by_id, len(example_ids)


def write_eval_scores_to_directory(scores: dict[DataId, float], output_dir: str) -> None:
    for val_id, score in scores.items():
        task_dir = os.path.join(output_dir, f"task_{val_id}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, f"iter_{0}_prog_0.json"), "w") as f:
            json.dump(score, f, indent=4, default=json_default)


def write_eval_outputs_to_directory(outputs, output_dir: str) -> None:
    """
    Write generated rollout outputs (not scalar scores) to disk.

    Structure:
      {output_dir}/task_{val_id}/iter_0_prog_0.json

    This directory is used to store best outputs for inspection/reuse.
    """
    for val_id, output in outputs.items():
        task_dir = os.path.join(output_dir, f"task_{val_id}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "iter_0_prog_0.json"), "w") as f:
            json.dump(output, f, indent=4, default=json_default)


def initialize_gepa_state(
    run_dir: str | None,
    logger: LoggerProtocol,
    seed_candidate: dict[str, str],
    seed_valset_evaluation: ValsetEvaluation[RolloutOutput, DataId],
    track_best_outputs: bool = False,
    frontier_type: FrontierType = "instance",
    evaluation_cache: "EvaluationCache[RolloutOutput, DataId] | None" = None,
) -> GEPAState[RolloutOutput, DataId]:
    if run_dir is not None and os.path.exists(os.path.join(run_dir, "gepa_state.bin")):
        logger.log("Loading gepa state from run dir")
        gepa_state = GEPAState.load(run_dir)
        if gepa_state.frontier_type != frontier_type:
            raise ValueError(
                f"Frontier type mismatch: requested '{frontier_type}' but loaded state has '{gepa_state.frontier_type}'. "
                f"Use a different run_dir or match the frontier_type parameter."
            )
        # Sync cache with current run's cache_evaluation setting:
        # - If caching is disabled (evaluation_cache is None), clear any loaded cache
        #   to respect the current run's cache_evaluation=False setting
        # - If caching is enabled and the loaded state has a cache, preserve it
        #   (allows resuming with cached results from previous run)
        # - If caching is enabled but no cache exists in loaded state, use the new empty cache
        if evaluation_cache is None:
            gepa_state.evaluation_cache = None
        elif gepa_state.evaluation_cache is None:
            gepa_state.evaluation_cache = evaluation_cache
        # else: keep the loaded cache (gepa_state.evaluation_cache is already set)
    else:
        if run_dir is not None:
            write_eval_outputs_to_directory(
                seed_valset_evaluation.outputs_by_val_id,
                os.path.join(run_dir, "generated_best_outputs_valset"),
            )

        gepa_state = GEPAState(
            seed_candidate,
            seed_valset_evaluation,
            track_best_outputs=track_best_outputs,
            frontier_type=frontier_type,
            evaluation_cache=evaluation_cache,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = len(seed_valset_evaluation.scores_by_val_id)

    return gepa_state
