# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from gepa.core.adapter import DataInst, invoke_batch_evaluate
from gepa.core.callbacks import GEPACallback
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.base import CandidateProposal
from glean_gepa.al_adapter import (
    MODULES,
    Candidate,
    GleanAdapterBase,
    ModuleSpec,
    within_prompt_budget,
)
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.prompt import WRITING_CODE_KEY, default_writing_code
from glean_gepa.utils import apply_single_module_edit


# TODO(Cathy): pick modules based on holistic performance of the eval
def pick_modules_to_edit() -> list[str]:
    return list(MODULES)


def make_children_for_generation(
    adapter: GleanAdapterBase,
    frontier_candidates: list[Candidate],
    frontier_evals: dict[str, GleanEvaluationBatch],
    reflection_llm: Any,
    offspring_count: int = 5,
    reflect_k: int | None = 8,
    max_attempts: int = 200,
    reflection_hamming_distance_k: int | None = None,
    children_by_root: dict[str, list[Candidate]] | None = None,
) -> list[Candidate]:
    """Create children by applying reflection-generated edits to one module.

    ``children_by_root`` retains the children already reflected from a parent.
    Reusing those candidates is intentional: a root's traces and prompt are
    unchanged while it remains on the frontier, so reflecting on it again only
    spends another LLM call to rediscover mutations we already have.
    """
    children: list[Candidate] = []
    seen_child_programs: set[str] = set()

    def append_child(child: Candidate) -> bool:
        """Append a distinct child while there is room in this generation."""
        child_key = json.dumps(child.prompt_modules, sort_keys=True)
        if child_key in seen_child_programs or len(children) >= offspring_count:
            return False
        seen_child_programs.add(child_key)
        children.append(child)
        return True

    # Pick a main parent using the concrete adapter's primary objective.
    best_quality_parent = max(
        frontier_candidates,
        key=lambda c: adapter.get_screening_score(frontier_evals[c.candidate_id]),
    )
    print(f"Best quality parent: {best_quality_parent}")

    # A cached root is never reflected again. Reuse cached children first, in
    # quality order, before generating mutations for roots we have not seen.
    # This also makes the cache useful when a parent disappears and later
    # returns to the Pareto frontier.
    if children_by_root is not None:
        ordered_roots = [best_quality_parent] + [
            parent for parent in frontier_candidates if parent.candidate_id != best_quality_parent.candidate_id
        ]
        for parent in ordered_roots:
            for child in children_by_root.get(parent.candidate_id, []):
                append_child(child)
            if len(children) >= offspring_count:
                return children

    attempts = 0
    while len(children) < offspring_count and attempts < max_attempts:
        attempts += 1
        uncached_roots = [
            parent
            for parent in frontier_candidates
            if children_by_root is None or parent.candidate_id not in children_by_root
        ]
        if not uncached_roots:
            break
        parent = (
            best_quality_parent
            if best_quality_parent in uncached_roots and random.random() < 0.7
            else random.choice(uncached_roots)
        )
        parent_eval = frontier_evals[parent.candidate_id]
        if not parent_eval.trajectories:
            # Need traces to reflect; skip mutation if missing.
            continue

        modules_to_edit = pick_modules_to_edit()

        high_signal = adapter.make_reflective_dataset(
            candidate=parent,
            eval_batch=frontier_evals[parent.candidate_id],
            components_to_update=modules_to_edit,
            k=reflect_k,
            error_hamming_distance_k=reflection_hamming_distance_k,
        )

        # Ask the reflection model for one to three small rewrite variants.
        for module in modules_to_edit:
            variants, _ = adapter.propose_new_texts(
                reflection_llm=reflection_llm,
                candidate=parent,
                components_to_update=[module],
                reflective_examples=high_signal[module],
            )
            if not variants:
                print(f"Reflection produced no variants for module {module}")
                continue

            for variant in variants[: max(1, offspring_count - len(children))]:
                print(f"Updated module {module}:\n{variant}")
                child = apply_single_module_edit(parent, module, variant)
                if children_by_root is not None:
                    cached_children = children_by_root.setdefault(parent.candidate_id, [])
                    if all(existing.prompt_modules != child.prompt_modules for existing in cached_children):
                        cached_children.append(child)
                append_child(child)
                if len(children) >= offspring_count:
                    break

    return children


def _select_screened_children(
    adapter: GleanAdapterBase,
    parent_eval: GleanEvaluationBatch,
    children: list[Candidate],
    screen_evals: list[GleanEvaluationBatch],
    *,
    use_high_signal_gate: bool,
) -> list[tuple[Candidate, GleanEvaluationBatch, float]]:
    """Keep every child eligible for GEPA's acceptance/selection stage."""
    selected: list[tuple[Candidate, GleanEvaluationBatch, float]] = []
    for child, screen_eval in zip(children, screen_evals, strict=True):
        child_score = (
            adapter.high_signal_fix_rate(parent_eval, screen_eval)
            if use_high_signal_gate
            else adapter.get_screening_score(screen_eval)
        )
        if not use_high_signal_gate or child_score > 0.5:
            selected.append((child, screen_eval, child_score))
    return selected


class EvolutionaryProposer:
    """
    Proposer that generates reflection-driven mutations from Pareto-frontier
    candidates and returns every child that passes screening. GEPA's configured
    acceptance and selection strategies remain authoritative over which of those
    children enter the candidate pool.

    Bridges between GEPA's dict[str, str] candidate format and the
    Glean AL adapter's Candidate type for reflection-driven mutation.
    """

    def __init__(
        self,
        logger: LoggerProtocol,
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        al_adapter: GleanAdapterBase,
        reflection_llm: Any,
        experiment_tracker: ExperimentTracker,
        # Candidate config (for converting dict[str,str] <-> Candidate)
        model: str,
        module_specs: dict[str, ModuleSpec],
        global_token_cap: int,
        baseline_prompt_hash: str,
        # Evolutionary hyperparameters
        offspring_count: int = 5,
        reflect_k: int | None = 8,
        callbacks: list[GEPACallback] | None = None,
        evalset_policy: UnseenEvalSetPolicy | None = None,
        reflection_hamming_distance_k: int | None = None,
    ):
        self.logger = logger
        self.trainset = ensure_loader(trainset)
        self.al_adapter = al_adapter
        self.reflection_llm = reflection_llm
        self.experiment_tracker = experiment_tracker
        self.callbacks = callbacks
        self.evalset_policy = evalset_policy

        # Candidate conversion config
        self.model = model
        self.module_specs = module_specs
        self.global_token_cap = global_token_cap
        self.baseline_prompt_hash = baseline_prompt_hash

        # Evolutionary hyperparameters
        self.offspring_count = offspring_count
        self.reflect_k = reflect_k
        self.reflection_hamming_distance_k = reflection_hamming_distance_k
        # Reflection depends on a root candidate and its fixed evaluation
        # traces. Keep its proposed children keyed by that stable root id so a
        # root revisited in a later iteration does not trigger reflection again.
        self._children_by_root: dict[str, list[Candidate]] = {}
        # Incremental training slices need fresh reflection, but each root should
        # still be reflected at most once within the same slice.
        self._children_by_root_by_train_slice: dict[tuple[DataId, ...], dict[str, list[Candidate]]] = {}

        # Store batch data for eval set (trainset is just metadata for eval set runs)
        # Extract first batch from trainset
        from typing import cast

        if isinstance(trainset, list):
            self._batch_data: list[dict[str, Any]] = cast(list[dict[str, Any]], trainset)
        else:
            # Get first batch from loader
            self._batch_data = []
            try:
                for _, batch in self.trainset:  # type: ignore
                    self._batch_data = cast(list[dict[str, Any]], batch)
                    break
            except Exception:
                # Fallback to empty batch
                self._batch_data = []

    def _to_candidate(self, program: dict[str, str]) -> Candidate:
        """Convert a GEPA program to the sole editable Glean prompt module."""
        prompt_modules = {WRITING_CODE_KEY: program.get(WRITING_CODE_KEY, default_writing_code)}
        content = json.dumps(prompt_modules, sort_keys=True)
        cand_id = hashlib.md5(content.encode()).hexdigest()[:10]
        return Candidate(
            model=self.model,
            prompt_modules=prompt_modules,
            module_specs=self.module_specs,
            global_token_cap=self.global_token_cap,
            baseline_prompt_hash=self.baseline_prompt_hash,
            candidate_id=cand_id,
        )

    def propose(self, state: GEPAState) -> list[CandidateProposal]:
        i = state.i + 1

        # 1. Get frontier program indices from Pareto front
        front_mapping = state.get_pareto_front_mapping()
        frontier_idxs: set[int] = set()
        for prog_set in front_mapping.values():
            frontier_idxs.update(prog_set)
        frontier_idxs_sorted = sorted(frontier_idxs)

        if not frontier_idxs_sorted:
            self.logger.log(f"Iteration {i}: No frontier programs found")
            return []
        else:
            self.logger.log(f"Iteration {i}: Found the following frontier programs {frontier_idxs_sorted}")

        # 2. Reveal one fresh training slice for this generation. The same slice
        # supplies failure evidence for reflection and the child-screening baseline,
        # so each iteration learns from new train data rather than reusing val IDs.
        if self.evalset_policy is not None:
            try:
                train_ids = self.evalset_policy.take_unseen(
                    self.trainset, purpose="reflection and offspring screening"
                )
            except RuntimeError as exc:
                self.logger.log(f"Iteration {i}: Training eval schedule exhausted; stopping proposals ({exc})")
                return []
            trace_batch = self.trainset.fetch(train_ids)
        else:
            train_ids = list(self.trainset.all_ids())
            trace_batch = self._batch_data

        # 3. Convert frontier programs to Candidate objects and evaluate with AL adapter.
        frontier_candidates: list[Candidate] = []
        frontier_evals: dict[str, GleanEvaluationBatch] = {}
        prog_idx_to_cand_id: dict[int, str] = {}

        for idx in frontier_idxs_sorted:
            program = state.program_candidates[idx]
            cand = self._to_candidate(program)
            prog_idx_to_cand_id[idx] = cand.candidate_id
            frontier_candidates.append(cand)

            frontier_evals[cand.candidate_id] = self.al_adapter.evaluate(
                trace_batch, cand.prompt_modules, capture_traces=True
            )

        # 4. Generate children using evolutionary strategies. Cached mutations
        # are scoped to the current training slice, so a fresh slice prompts a
        # new reflection while duplicate attempts within that slice are avoided.
        children_by_root = (
            self._children_by_root_by_train_slice.setdefault(tuple(train_ids), {})
            if self.evalset_policy is not None
            else self._children_by_root
        )
        children = make_children_for_generation(
            adapter=self.al_adapter,
            frontier_candidates=frontier_candidates,
            frontier_evals=frontier_evals,
            reflection_llm=self.reflection_llm,
            offspring_count=self.offspring_count,
            reflect_k=self.reflect_k,
            reflection_hamming_distance_k=self.reflection_hamming_distance_k,
            children_by_root=children_by_root,
        )

        if not children:
            self.logger.log(f"Iteration {i}: Evolutionary proposer generated no children")
            return []

        # 5. Filter children by prompt budget
        valid_children = [c for c in children if within_prompt_budget(c)]
        if not valid_children:
            self.logger.log(f"Iteration {i}: No children passed budget check")
            return []

        # 6. Screen children on the parent's high-signal failures first.
        # Only a child that fixes strictly more than half of those failures is
        # allowed to reach GEPA's full validation evaluation.
        best_parent_idx = max(frontier_idxs_sorted, key=lambda idx: state.program_full_scores_val_set[idx])
        best_parent_cand_id = prog_idx_to_cand_id[best_parent_idx]
        parent_eval = frontier_evals[best_parent_cand_id]
        use_high_signal_gate = getattr(self.al_adapter, "supports_high_signal_eval", False)
        if use_high_signal_gate:
            high_signal_batch = self.al_adapter.high_signal_batch(parent_eval)
            if not high_signal_batch:
                self.logger.log(f"Iteration {i}: Parent has no high-signal failures; rejecting children")
                return []
            high_signal_batch = self.al_adapter.prepare_high_signal_batch(high_signal_batch)
            if high_signal_batch is None:
                self.logger.log(f"Iteration {i}: Failed to prepare the high-signal eval set; rejecting children")
                return []
        else:
            high_signal_batch = trace_batch
        screen_evals = invoke_batch_evaluate(
            self.al_adapter,
            [(child.prompt_modules, high_signal_batch) for child in valid_children],
            capture_traces=use_high_signal_gate,
        )
        screened_children = _select_screened_children(
            self.al_adapter,
            parent_eval,
            valid_children,
            screen_evals,
            use_high_signal_gate=use_high_signal_gate,
        )
        best_child_score = max((score for _child, _eval, score in screened_children), default=float("-inf"))

        if not screened_children:
            if use_high_signal_gate:
                all_fix_rates = [self.al_adapter.high_signal_fix_rate(parent_eval, result) for result in screen_evals]
                best_fix_rate = max(all_fix_rates, default=0.0)
                self.logger.log(
                    f"Iteration {i}: No child fixed more than half of the high-signal failures "
                    f"(best={best_fix_rate:.1%})"
                )
            else:
                self.logger.log(f"Iteration {i}: No children completed screening")
            return []

        # 7. Get best parent and their evaluation (already cached)
        # The engine now runs the selected children on the full eval set.
        subsample_ids = train_ids
        parent_score = self.al_adapter.get_screening_score(parent_eval)
        child_score = best_child_score

        self.logger.log(
            f"Iteration {i}: Evolutionary proposer generated {len(children)} children, "
            f"{len(valid_children)} passed budget, {len(screened_children)} passed screening. "
            f"Best screening score={best_child_score:.3f}"
        )
        self.experiment_tracker.log_metrics(
            {
                "evolutionary_parent_eval_score": parent_score,
                "evolutionary_child_eval_score": child_score,
                "evolutionary_children_generated": len(children),
                "evolutionary_children_valid": len(valid_children),
                "evolutionary_children_screened_in": len(screened_children),
                "total_metric_calls": state.total_num_evals,
            },
            step=i,
        )

        return [
            CandidateProposal(
                candidate=child.prompt_modules,
                parent_program_ids=[best_parent_idx],
                subsample_indices=subsample_ids,
                subsample_scores_before=[parent_score],
                subsample_scores_after=[screen_score],
                tag="evolutionary",
            )
            for child, _screen_eval, screen_score in screened_children
        ]
