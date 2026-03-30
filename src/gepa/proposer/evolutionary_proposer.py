# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import hashlib
import json
import random
from typing import Any
import math

from gepa.adapters.glean_adapter.al_adapter import (
    MODULES,
    AssistantALAdapter,
    Candidate,
    EvalSummary,
    ModuleSpec,
    within_prompt_budget,
)
from gepa.adapters.glean_adapter.utils import apply_single_module_edit, crossover
from gepa.core.adapter import DataInst, EvaluationBatch
from gepa.core.callbacks import GEPACallback
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState
from gepa.logging.experiment_tracker import ExperimentTracker
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate


# TODO(Cathy): pick modules based on holistic performance of the eval
def pick_modules_to_edit() -> [str]:
   return random.choices(MODULES, k=2)

def make_children_for_generation(
    adapter: AssistantALAdapter,
    frontier_candidates: list[Candidate],
    frontier_evals: dict[str, EvaluationBatch],
    reflection_llm: Any,
    offspring_count: int = 24,
    reflect_k: int = 8,
    p_mutation: float = 0.70,
    p_crossover: float = 0.25,
    p_pool_inject: float = 0.05,
    max_attempts: int = 200,
) -> list[Candidate]:
    """
    Creates children using:
      - mutation: apply_single_module_edit from reflection patches
      - crossover: mixing module texts from two parents
      - pool injection: replace a module with a 'good option' previously seen
    """
    children: list[Candidate] = []

    # pick a "main parent" biased to best-quality (doc says bias to best correctness)
    best_quality_parent = max(
        frontier_candidates, key=lambda c: frontier_evals[c.candidate_id].summary['correctness']
    )

    # helper: choose a second parent emphasizing diversity
    def pick_second_parent() -> Candidate:
        return random.choice(frontier_candidates)

    attempts = 0
    while len(children) < offspring_count and attempts < max_attempts:
        attempts += 1
        r = random.random()

        # ---- 1) mutation via reflection patches
        if r < p_mutation:
            parent = best_quality_parent if random.random() < 0.7 else random.choice(frontier_candidates)
            parent_eval = frontier_evals[parent.candidate_id]
            if not parent_eval.trajectories:
                # need traces to reflect; skip mutation if missing
                continue

            # pick a module to improve: worst module by relevance over high-signal failures
            modules_to_edit = pick_modules_to_edit()

            # select high-signal examples for this module (doc requirement)
            high_signal = adapter.make_reflective_dataset(
                candidate=parent,
                eval_batch=frontier_evals[parent.candidate_id],
                components_to_update=modules_to_edit,
                k=reflect_k,
            )

            # ask teacher/reflection model for 1–3 rewrite variants (small deltas)
            for module in modules_to_edit:
                variants, not_relevant = adapter.propose_new_texts(
                    reflection_llm=reflection_llm,
                    candidate=parent,
                    components_to_update=[module],
                    reflective_examples=high_signal[module],
                )
                # TODO(Cathy): Add logic here to add these to good variant.
                if not_relevant or not variants:
                    # module freeze logic lives outside (track streak; stop choosing module later)
                    continue

                # create one child per variant (bounded)
                for v in variants[: max(1, (offspring_count - len(children)))]:
                    child = apply_single_module_edit(parent, module, v)
                    children.append(child)
                    if len(children) >= offspring_count:
                        break

        # ---- 2) crossover: combine two parents
        elif r < p_mutation + p_crossover:
            a = random.choice(frontier_candidates)
            b = pick_second_parent()
            child = crossover(a, b, module_specs=a.module_specs, global_cap=a.global_token_cap)
            children.append(child)

        # ---- 3) pool injection: swap a module with a known-good text
        else:
            parent = random.choice(frontier_candidates)
            module_to_swap = random.choice(MODULES)
            pool = adapter.good_module_options.get(module_to_swap, [])
            if not pool:
                continue
            new_text = random.choice(pool)
            child = apply_single_module_edit(parent, module_to_swap, new_text)
            children.append(child)

    return children


class EvolutionaryProposer(ProposeNewCandidate[DataId]):
    """
    Proposer that uses evolutionary strategies (mutation via reflection,
    crossover, pool injection) to generate candidate children from the
    Pareto frontier and return the best as a CandidateProposal.

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
        # Evolutionary hyperparameters
        offspring_count: int = 24,
        reflect_k: int = 8,
        p_mutation: float = 0.70,
        p_crossover: float = 0.25,
        p_pool_inject: float = 0.05,
        callbacks: list[GEPACallback] | None = None,
    ):
        self.logger = logger
        self.trainset = ensure_loader(trainset)
        self.al_adapter = al_adapter
        self.reflection_llm = reflection_llm
        self.experiment_tracker = experiment_tracker
        self.callbacks = callbacks

        # Candidate conversion config
        self.model = model
        self.module_specs = module_specs
        self.global_token_cap = global_token_cap
        self.baseline_prompt_hash = baseline_prompt_hash

        # Evolutionary hyperparameters
        self.offspring_count = offspring_count
        self.reflect_k = reflect_k
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.p_pool_inject = p_pool_inject

        # Cache AL adapter evaluations for frontier candidates across iterations
        self._al_eval_cache: dict[str, CandidateEval] = {}

        # Store batch data for eval set (trainset is just metadata for eval set runs)
        # Extract first batch from trainset
        from typing import cast, Dict, List
        if isinstance(trainset, list):
            self._batch_data: List[Dict[str, Any]] = cast(List[Dict[str, Any]], trainset)
        else:
            # Get first batch from loader
            self._batch_data = []
            try:
                for _, batch in self.trainset:  # type: ignore
                    self._batch_data = cast(List[Dict[str, Any]], batch)
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

        # 2. Convert frontier programs to Candidate objects and evaluate with AL adapter
        frontier_candidates: list[Candidate] = []
        frontier_evals: dict[str, EvaluationBatch] = {}
        prog_idx_to_cand_id: dict[int, str] = {}

        for idx in frontier_idxs_sorted:
            program = state.program_candidates[idx]
            cand = self._to_candidate(program)
            prog_idx_to_cand_id[idx] = cand.candidate_id
            frontier_candidates.append(cand)

            if cand.candidate_id not in self._al_eval_cache:
                al_eval = self.al_adapter.evaluate(
                    self._batch_data, program, capture_traces=True
                )
                self._al_eval_cache[cand.candidate_id] = al_eval
            frontier_evals[cand.candidate_id] = self._al_eval_cache[cand.candidate_id]

        # 3. Generate children using evolutionary strategies
        children = make_children_for_generation(
            adapter=self.al_adapter,
            frontier_candidates=frontier_candidates,
            frontier_evals=frontier_evals,
            reflection_llm=self.reflection_llm,
            offspring_count=self.offspring_count,
            reflect_k=self.reflect_k,
            p_mutation=self.p_mutation,
            p_crossover=self.p_crossover,
            p_pool_inject=self.p_pool_inject,
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
        best_child_eval: EvaluationBatch | None = None
        best_child_quality = float("-inf")
        for child in valid_children:
            screen_eval = self.al_adapter.evaluate(
                self._batch_data, child.prompt_modules, capture_traces=False
            )
            if screen_eval.summary['correctness'] > best_child_quality:
                best_child = child
                best_child_eval = screen_eval
                best_child_quality = screen_eval.summary['correctness']

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
        parent_score = parent_eval.summary['correctness']
        child_score = best_child_eval.summary['correctness']
        state.increment_evals(1)  # One eval set run

        self.logger.log(
            f"Iteration {i}: Evolutionary proposer generated {len(children)} children, "
            f"{len(valid_children)} passed budget. Best child quality={best_child_quality:.3f}"
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
