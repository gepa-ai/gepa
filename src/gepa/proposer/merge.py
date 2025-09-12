# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
import random
from copy import deepcopy
from typing import Any, Callable

from gepa.core.adapter import DataInst, RolloutOutput
from gepa.core.state import GEPAState
from gepa.gepa_utils import find_dominator_programs
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate


def does_triplet_have_desirable_predictors(program_candidates: list[dict[str, str]], ancestor, id1, id2):
    found_predictors = []
    pred_names = set(program_candidates[ancestor].keys())
    for pred_idx, pred_name in enumerate(pred_names):
        pred_anc = program_candidates[ancestor][pred_name]
        pred_id1 = program_candidates[id1][pred_name]
        pred_id2 = program_candidates[id2][pred_name]
        if (
            (pred_anc == pred_id1) or
            (pred_anc == pred_id2)
        ) and (
            pred_id1 != pred_id2
        ):
            # We have a predictor that is the same as one of its ancestors, so we can update it with the other
            same_as_ancestor_id = (1 if pred_anc == pred_id1 else 2)
            found_predictors.append((pred_idx, same_as_ancestor_id))

    return len(found_predictors) > 0

def filter_ancestors(i, j, common_ancestors, merges_performed, agg_scores, program_candidates):
    filtered_ancestors = []
    for ancestor in common_ancestors:
        if (i, j, ancestor) in merges_performed[0]:
            continue

        if agg_scores[ancestor] > agg_scores[i] or agg_scores[ancestor] > agg_scores[j]:
            continue

        if not does_triplet_have_desirable_predictors(program_candidates, ancestor, i, j):
            continue

        filtered_ancestors.append(ancestor)
    return filtered_ancestors

def find_common_ancestor_pair(rng, parent_list, program_indexes, merges_performed, agg_scores, program_candidates, max_attempts=10):
    def get_ancestors(node, ancestors_found):
        parents = parent_list[node]
        for parent in parents:
            if parent is not None and parent not in ancestors_found:
                ancestors_found.add(parent)
                get_ancestors(parent, ancestors_found)

        return list(ancestors_found)

    for _ in range(max_attempts):
        if len(program_indexes) < 2:
            return None
        i, j = rng.sample(program_indexes, 2)
        if i == j:
            continue

        if j < i:
            i, j = j, i

        ancestors_i = get_ancestors(i, set())
        ancestors_j = get_ancestors(j, set())

        if j in ancestors_i or i in ancestors_j:
            # If one is an ancestor of the other, we cannot merge them
            continue

        common_ancestors = set(ancestors_i) & set(ancestors_j)
        common_ancestors = filter_ancestors(i, j, common_ancestors, merges_performed, agg_scores, program_candidates)
        if common_ancestors:
            # Select a random common ancestor
            common_ancestor = rng.choices(common_ancestors, k=1, weights=[agg_scores[ancestor] for ancestor in common_ancestors])[0]
            return (i, j, common_ancestor)

    return None

def sample_and_attempt_merge_programs_by_common_predictors(agg_scores, rng, merge_candidates, merges_performed, program_candidates: list[dict[str, str]], parent_program_for_candidate, max_attempts=10):
    if len(merge_candidates) < 2:
        return (False, None, None, None, None)
    if len(parent_program_for_candidate) < 3:
        return (False, None, None, None, None)

    for _ in range(max_attempts):
        ids_to_merge = find_common_ancestor_pair(rng, parent_program_for_candidate, list(merge_candidates), merges_performed=merges_performed, agg_scores=agg_scores, program_candidates=program_candidates, max_attempts=10)
        if ids_to_merge is None:
            continue
        id1, id2, ancestor = ids_to_merge

        assert (id1, id2, ancestor) not in merges_performed, "This pair has already been merged"

        assert agg_scores[ancestor] <= agg_scores[id1], "Ancestor should not be better than its descendants"
        assert agg_scores[ancestor] <= agg_scores[id2], "Ancestor should not be better than its descendants"
        assert id1 != id2, "Cannot merge the same program"

        # Now we have a common ancestor, which is outperformed by both its descendants

        new_program = deepcopy(program_candidates[ancestor])

        new_prog_desc = ()

        pred_names = set(program_candidates[ancestor].keys())
        assert pred_names == set(program_candidates[id1].keys()) == set(program_candidates[id2].keys()), "Predictors should be the same across all programs"
        for pred_name in pred_names:
            pred_anc = program_candidates[ancestor][pred_name]
            pred_id1 = program_candidates[id1][pred_name]
            pred_id2 = program_candidates[id2][pred_name]
            if (
                (pred_anc == pred_id1) or
                (pred_anc == pred_id2)
            ) and (
                pred_id1 != pred_id2
            ):
                # We have a predictor that is the same as one of its ancestors, so we can update it with the other
                same_as_ancestor_id = (1 if pred_anc == pred_id1 else 2)
                # new_program.named_predictors()[pred_idx][1].signature = program_candidates[id2 if same_as_ancestor_id == 1 else id1].named_predictors()[pred_idx][1].signature
                new_program[pred_name] = program_candidates[id2 if same_as_ancestor_id == 1 else id1][pred_name]
                new_prog_desc = (*new_prog_desc, id2 if same_as_ancestor_id == 1 else id1)
            elif (
                (pred_anc != pred_id1) and
                (pred_anc != pred_id2)
            ):
                # Both predictors are different from  the ancestor, and it is difficult to decide which one gives the benefits
                # We randomly select one of the descendants to update the predictor
                # The probability of selecting is proportional to the agg_scores of the descendants
                # prog_to_get_instruction_from = id1 if (rng.random() < (agg_scores[id1] / (agg_scores[id1] + agg_scores[id2]))) else id2
                prog_to_get_instruction_from = id1 if (agg_scores[id1] > agg_scores[id2]) else (id2 if agg_scores[id2] > agg_scores[id1] else rng.choice([id1, id2]))
                new_program[pred_name] = program_candidates[prog_to_get_instruction_from][pred_name]
                new_prog_desc = (*new_prog_desc, prog_to_get_instruction_from)
            elif (
                pred_id1 == pred_id2
            ):
                # Either both predictors are the same, or both are different from the ancestor
                # If both are different from the ancestor, we should use the new predictor, so selecting either one of the descendants is fine
                # If both are same as the ancesor, again selecting any one of the descendants is fine
                # So let's select id1
                new_program[pred_name] = program_candidates[id1][pred_name]
                new_prog_desc = (*new_prog_desc, id1)
            else:
                assert False, "Unexpected case in predictor merging logic"

        if (id1, id2, new_prog_desc) in merges_performed[1]:
            # This triplet has already been merged, so we skip it
            continue

        merges_performed[1].append((id1, id2, new_prog_desc))

        return (True, new_program, id1, id2, ancestor)

    return (False, None, None, None, None)

class MergeProposer(ProposeNewCandidate):
    """
    Implements current merge flow:
    - Find merge candidates among Pareto front dominators
    - Attempt a merge via sample_and_attempt_merge_programs_by_common_predictors
    - Subsample eval on valset-driven selected indices
    - Return proposal if merge's subsample score >= max(parents)
    The engine handles full eval + adding to state.
    """
    def __init__(
        self,
        logger: Any,
        valset: list[DataInst],
        evaluator: Callable[[list[DataInst], dict[str, str]], tuple[list[RolloutOutput], list[float]]],
        use_merge: bool,
        max_merge_invocations: int,
        rng: random.Random | None = None,
    ):
        self.logger = logger
        self.valset = valset
        self.evaluator = evaluator
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

        # Internal counters matching original behavior
        self.merges_due = 0
        self.total_merges_tested = 0
        self.merges_performed: tuple[list[tuple[int, int, int]], Any] = ([], [])

        # Toggle controlled by engine: set True when last iter found new program
        self.last_iter_found_new_program = False

    def schedule_if_needed(self):
        if self.use_merge and self.total_merges_tested < self.max_merge_invocations:
            self.merges_due += 1

    def select_eval_subsample_for_merged_program(
        self,
        scores1: list[float],
        scores2: list[float],
        num_subsample_ids: int = 5,
    ) -> list[int]:
        all_indices = set(range(len(scores1)))
        p1 = [i for i, (s1, s2) in enumerate(zip(scores1, scores2, strict=False)) if s1 > s2]
        p2 = [i for i, (s1, s2) in enumerate(zip(scores1, scores2, strict=False)) if s2 > s1]
        p3 = [i for i in all_indices if i not in p1 and i not in p2]

        n_each = math.ceil(num_subsample_ids / 3)
        n1 = min(len(p1), n_each)
        n2 = min(len(p2), n_each)
        n3 = min(len(p3), num_subsample_ids - (n1 + n2))
        selected = []
        if n1: selected += self.rng.sample(p1, k=n1)
        if n2: selected += self.rng.sample(p2, k=n2)
        if n3: selected += self.rng.sample(p3, k=n3)

        remaining = num_subsample_ids - len(selected)
        unused = list(all_indices - set(selected))
        if remaining > 0:
            if len(unused) >= remaining:
                selected += self.rng.sample(unused, k=remaining)
            else:
                selected += self.rng.choices(list(all_indices), k=remaining)
        return selected[:num_subsample_ids]

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        i = state.i + 1
        state.full_program_trace[-1]["invoked_merge"] = True

        # Only attempt when scheduled by engine and after a new program in last iteration
        if not (self.use_merge and self.last_iter_found_new_program and self.merges_due > 0):
            self.logger.log(f"Iteration {i}: No merge candidates scheduled")
            return None

        pareto_front_programs = state.program_at_pareto_front_valset
        merge_candidates = find_dominator_programs(pareto_front_programs, state.per_program_tracked_scores)
        merge_output = sample_and_attempt_merge_programs_by_common_predictors(
            agg_scores=state.per_program_tracked_scores,
            rng=self.rng,
            merge_candidates=merge_candidates,
            merges_performed=self.merges_performed,
            program_candidates=state.program_candidates,
            parent_program_for_candidate=state.parent_program_for_candidate,
        )

        if not merge_output[0]:
            self.logger.log(f"Iteration {i}: No merge candidates found")
            return None

        # success, new_program, id1, id2, ancestor
        success, new_program, id1, id2, ancestor = merge_output
        state.full_program_trace[-1]["merged"] = True
        state.full_program_trace[-1]["merged_entities"] = (id1, id2, ancestor)
        self.merges_performed[0].append((id1, id2, ancestor))
        self.logger.log(f"Iteration {i}: Merged programs {id1} and {id2} via ancestor {ancestor}")

        subsample_ids = self.select_eval_subsample_for_merged_program(
            state.prog_candidate_val_subscores[id1],
            state.prog_candidate_val_subscores[id2],
        )
        mini_devset = [self.valset[k] for k in subsample_ids]
        id1_sub_scores = [state.prog_candidate_val_subscores[id1][k] for k in subsample_ids]
        id2_sub_scores = [state.prog_candidate_val_subscores[id2][k] for k in subsample_ids]
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids

        _, new_sub_scores = self.evaluator(mini_devset, new_program)

        state.full_program_trace[-1]["id1_subsample_scores"] = id1_sub_scores
        state.full_program_trace[-1]["id2_subsample_scores"] = id2_sub_scores
        state.full_program_trace[-1]["new_program_subsample_scores"] = new_sub_scores

        # Count evals
        state.total_num_evals += len(subsample_ids)

        # Acceptance will be evaluated by engine (>= max(parents))
        return CandidateProposal(
            candidate=new_program,
            parent_program_ids=[id1, id2],
            subsample_indices=subsample_ids,
            subsample_scores_before=[sum(id1_sub_scores), sum(id2_sub_scores)],  # packed as [parent1_sum, parent2_sum]
            subsample_scores_after=new_sub_scores,
            tag="merge",
            metadata={"ancestor": ancestor}
        )
