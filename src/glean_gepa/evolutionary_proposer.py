# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from gepa.core.adapter import DataInst
from gepa.core.callbacks import GEPACallback
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate
from glean_gepa.al_adapter import (
    MODULES,
    AssistantALAdapter,
    Candidate,
    JudgingMode,
    ModuleSpec,
    get_screening_score,
    within_prompt_budget,
)
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.utils import apply_single_module_edit


# TODO(Cathy): pick modules based on holistic performance of the eval
def pick_modules_to_edit() -> list[str]:
    return random.choices(MODULES, k=2)


def make_children_for_generation(
    adapter: AssistantALAdapter,
    frontier_candidates: list[Candidate],
    frontier_evals: dict[str, GleanEvaluationBatch],
    reflection_llm: Any,
    judging_mode: JudgingMode,
    offspring_count: int = 24,
    reflect_k: int = 8,
    max_attempts: int = 200,
) -> list[Candidate]:
    """
    Create children by applying reflection-generated edits to one module.
    """
    children: list[Candidate] = []

    # pick a "main parent" biased to best primary objective for this judging mode
    best_quality_parent = max(
        frontier_candidates,
        key=lambda c: get_screening_score(frontier_evals[c.candidate_id], judging_mode),
    )
    print(f"Best quality parent: {best_quality_parent}")

    attempts = 0
    while len(children) < offspring_count and attempts < max_attempts:
        attempts += 1
        parent = best_quality_parent if random.random() < 0.7 else random.choice(frontier_candidates)
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
        )

        # Ask the reflection model for one to three small rewrite variants.
        for module in modules_to_edit:
            variants, not_relevant = adapter.propose_new_texts(
                reflection_llm=reflection_llm,
                candidate=parent,
                components_to_update=[module],
                reflective_examples=high_signal[module],
            )
            if not_relevant or not variants:
                print(f"Not relevant or no variants for module {module}")
                continue

            for variant in variants[: max(1, offspring_count - len(children))]:
                child = apply_single_module_edit(parent, module, variant)
                children.append(child)
                if len(children) >= offspring_count:
                    break

    return children


class EvolutionaryProposer(ProposeNewCandidate[DataId]):
    """
    Proposer that generates reflection-driven mutations from Pareto-frontier
    candidates and returns the strongest screened child.

    Bridges between GEPA's dict[str, str] candidate format and the
    Glean AL adapter's Candidate type for reflection-driven mutation.
    """

    def __init__(
        self,
        logger: LoggerProtocol,
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        al_adapter: AssistantALAdapter,
        reflection_llm: Any,
        experiment_tracker: ExperimentTracker,
        # Candidate config (for converting dict[str,str] <-> Candidate)
        model: str,
        module_specs: dict[str, ModuleSpec],
        global_token_cap: int,
        baseline_prompt_hash: str,
        judging_mode: JudgingMode = "single_model",
        # Evolutionary hyperparameters
        offspring_count: int = 24,
        reflect_k: int = 8,
        callbacks: list[GEPACallback] | None = None,
    ):
        self.logger = logger
        self.trainset = ensure_loader(trainset)
        self.al_adapter = al_adapter
        self.reflection_llm = reflection_llm
        self.experiment_tracker = experiment_tracker
        self.judging_mode = judging_mode
        self.callbacks = callbacks

        # Candidate conversion config
        self.model = model
        self.module_specs = module_specs
        self.global_token_cap = global_token_cap
        self.baseline_prompt_hash = baseline_prompt_hash

        # Evolutionary hyperparameters
        self.offspring_count = offspring_count
        self.reflect_k = reflect_k

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
        """Convert a GEPA dict[str, str] program to a Glean Candidate with a deterministic id."""
        content = json.dumps(program, sort_keys=True)
        cand_id = hashlib.md5(content.encode()).hexdigest()[:10]
        return Candidate(
            model=self.model,
            prompt_modules=dict(program),
            module_specs=self.module_specs,
            global_token_cap=self.global_token_cap,
            baseline_prompt_hash=self.baseline_prompt_hash,
            candidate_id=cand_id,
        )

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        i = state.i + 1

        # 1. Get frontier program indices from Pareto front
        front_mapping = state.get_pareto_front_mapping()
        frontier_idxs: set[int] = set()
        for prog_set in front_mapping.values():
            frontier_idxs.update(prog_set)
        frontier_idxs_sorted = sorted(frontier_idxs)

        if not frontier_idxs_sorted:
            self.logger.log(f"Iteration {i}: No frontier programs found")
            return None
        else:
            self.logger.log(f"Iteration {i}: Found the following frontier programs {frontier_idxs_sorted}")

        # 2. Convert frontier programs to Candidate objects and evaluate with AL adapter
        frontier_candidates: list[Candidate] = []
        frontier_evals: dict[str, GleanEvaluationBatch] = {}
        prog_idx_to_cand_id: dict[int, str] = {}

        for idx in frontier_idxs_sorted:
            program = state.program_candidates[idx]
            cand = self._to_candidate(program)
            prog_idx_to_cand_id[idx] = cand.candidate_id
            frontier_candidates.append(cand)

            frontier_evals[cand.candidate_id] = self.al_adapter.evaluate(
                self._batch_data, program, capture_traces=True
            )

        # 3. Generate children using evolutionary strategies
        children = make_children_for_generation(
            adapter=self.al_adapter,
            frontier_candidates=frontier_candidates,
            frontier_evals=frontier_evals,
            reflection_llm=self.reflection_llm,
            judging_mode=self.judging_mode,
            offspring_count=self.offspring_count,
            reflect_k=self.reflect_k,
        )

        if not children:
            self.logger.log(f"Iteration {i}: Evolutionary proposer generated no children")
            return None

        # 4. Filter children by prompt budget
        valid_children = [c for c in children if within_prompt_budget(c)]
        if not valid_children:
            self.logger.log(f"Iteration {i}: No children passed budget check")
            return None

        # 5. Screen children with AL adapter and pick best
        best_child: Candidate | None = None
        best_child_eval: GleanEvaluationBatch | None = None
        best_child_score = float("-inf")
        for child in valid_children:
            screen_eval = self.al_adapter.evaluate(
                self._batch_data, child.prompt_modules, capture_traces=False
            )
            child_score = get_screening_score(screen_eval, self.judging_mode)
            if child_score > best_child_score:
                best_child = child
                best_child_eval = screen_eval
                best_child_score = child_score

        if best_child is None or best_child_eval is None:
            return None

        # 6. Get best parent and their evaluation (already cached)
        val_scores = state.program_full_scores_val_set
        best_parent_idx = max(frontier_idxs_sorted, key=lambda idx: val_scores[idx])
        best_parent_cand_id = prog_idx_to_cand_id[best_parent_idx]
        parent_eval = frontier_evals[best_parent_cand_id]

        # Use screen eval set as "subsample" (one eval set evaluation)
        # Both parent and child evaluated on same screen eval set
        subsample_ids = [0]  # Dummy ID since we evaluate on full eval set
        parent_score = get_screening_score(parent_eval, self.judging_mode)
        child_score = get_screening_score(best_child_eval, self.judging_mode)
        state.increment_evals(1)  # One eval set run

        self.logger.log(
            f"Iteration {i}: Evolutionary proposer generated {len(children)} children, "
            f"{len(valid_children)} passed budget. Best child score={best_child_score:.3f}"
        )
        self.experiment_tracker.log_metrics(
            {
                "evolutionary_parent_eval_score": parent_score,
                "evolutionary_child_eval_score": child_score,
                "evolutionary_children_generated": len(children),
                "evolutionary_children_valid": len(valid_children),
                "total_metric_calls": state.total_num_evals,
            },
            step=i,
        )

        return CandidateProposal(
            candidate=best_child.prompt_modules,
            parent_program_ids=[best_parent_idx],
            subsample_indices=subsample_ids,
            subsample_scores_before=[parent_score],
            subsample_scores_after=[child_score],
            tag="evolutionary",
        )
