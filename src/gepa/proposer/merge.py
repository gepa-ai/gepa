# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
import random
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy

from gepa.core.adapter import Candidate, DataInst, EvaluatorFn, RolloutOutput
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState, ProgramIdx
from gepa.gepa_utils import find_dominator_programs
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate

AncestorLog = tuple[int, int, int]
MergeDescription = tuple[int, int, tuple[int, ...]]
MergeAttempt = tuple[Candidate, ProgramIdx, ProgramIdx, ProgramIdx] | None


def does_triplet_have_desirable_predictors(
    program_candidates: Sequence[Candidate],
    ancestor: ProgramIdx,
    id1: ProgramIdx,
    id2: ProgramIdx,
) -> bool:
    found_predictors: list[tuple[int, int]] = []
    pred_names = list(program_candidates[ancestor].keys())
    for pred_idx, pred_name in enumerate(pred_names):
        pred_anc = program_candidates[ancestor][pred_name]
        pred_id1 = program_candidates[id1][pred_name]
        pred_id2 = program_candidates[id2][pred_name]
        if (pred_anc == pred_id1 or pred_anc == pred_id2) and pred_id1 != pred_id2:
            same_as_ancestor_id = 1 if pred_anc == pred_id1 else 2
            found_predictors.append((pred_idx, same_as_ancestor_id))

    return len(found_predictors) > 0


def filter_ancestors(
    i: ProgramIdx,
    j: ProgramIdx,
    common_ancestors: Iterable[ProgramIdx],
    merges_performed: tuple[list[AncestorLog], list[MergeDescription]],
    agg_scores: Sequence[float],
    program_candidates: Sequence[Candidate],
) -> list[ProgramIdx]:
    filtered_ancestors: list[ProgramIdx] = []
    for ancestor in common_ancestors:
        if (i, j, ancestor) in merges_performed[0]:
            continue

        if agg_scores[ancestor] > agg_scores[i] or agg_scores[ancestor] > agg_scores[j]:
            continue

        if not does_triplet_have_desirable_predictors(program_candidates, ancestor, i, j):
            continue

        filtered_ancestors.append(ancestor)
    return filtered_ancestors


def find_common_ancestor_pair(
    rng: random.Random,
    parent_list: Sequence[Sequence[int | None]],
    program_indexes: Sequence[int],
    merges_performed: tuple[list[AncestorLog], list[MergeDescription]],
    agg_scores: Sequence[float],
    program_candidates: Sequence[Candidate],
    max_attempts: int = 10,
) -> tuple[int, int, int] | None:
    def get_ancestors(node: int, ancestors_found: set[int]) -> list[int]:
        parents = parent_list[node]
        for parent in parents:
            if parent is not None and parent not in ancestors_found:
                ancestors_found.add(parent)
                get_ancestors(parent, ancestors_found)

        return list(ancestors_found)

    for _ in range(max_attempts):
        if len(program_indexes) < 2:
            return None
        i, j = rng.sample(list(program_indexes), 2)
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
            common_ancestor = rng.choices(
                list(common_ancestors),
                k=1,
                weights=[agg_scores[ancestor] for ancestor in common_ancestors],
            )[0]
            return (i, j, common_ancestor)

    return None


def sample_and_attempt_merge_programs_by_common_predictors(
    agg_scores: Sequence[float],
    rng: random.Random,
    merge_candidates: Sequence[int],
    merges_performed: tuple[list[AncestorLog], list[MergeDescription]],
    program_candidates: Sequence[Candidate],
    parent_program_for_candidate: Sequence[Sequence[int | None]],
    has_val_support_overlap: Callable[[ProgramIdx, ProgramIdx], bool] | None = None,
    max_attempts: int = 10,
) -> MergeAttempt:
    if len(merge_candidates) < 2:
        return None
    if len(parent_program_for_candidate) < 3:
        return None

    for _ in range(max_attempts):
        ids_to_merge = find_common_ancestor_pair(
            rng,
            parent_program_for_candidate,
            list(merge_candidates),
            merges_performed=merges_performed,
            agg_scores=agg_scores,
            program_candidates=program_candidates,
            max_attempts=max_attempts,
        )
        if ids_to_merge is None:
            continue
        id1, id2, ancestor = ids_to_merge

        assert (id1, id2, ancestor) not in merges_performed, "This pair has already been merged"
        assert agg_scores[ancestor] <= agg_scores[id1], "Ancestor should not be better than its descendants"
        assert agg_scores[ancestor] <= agg_scores[id2], "Ancestor should not be better than its descendants"
        assert id1 != id2, "Cannot merge the same program"

        # Now we have a common ancestor, which is outperformed by both its descendants

        new_program: Candidate = deepcopy(program_candidates[ancestor])

        new_prog_desc: tuple[int, ...] = ()

        pred_names = set(program_candidates[ancestor].keys())
        assert pred_names == set(program_candidates[id1].keys()) == set(program_candidates[id2].keys()), (
            "Predictors should be the same across all programs"
        )
        for pred_name in pred_names:
            pred_anc = program_candidates[ancestor][pred_name]
            pred_id1 = program_candidates[id1][pred_name]
            pred_id2 = program_candidates[id2][pred_name]
            if (pred_anc == pred_id1 or pred_anc == pred_id2) and pred_id1 != pred_id2:
                # We have a predictor that is the same as one of its ancestors, so we can update it with the other
                same_as_ancestor_id = 1 if pred_anc == pred_id1 else 2
                new_value_idx = id2 if same_as_ancestor_id == 1 else id1
                new_program[pred_name] = program_candidates[new_value_idx][pred_name]
                new_prog_desc = (*new_prog_desc, new_value_idx)
            elif pred_anc != pred_id1 and pred_anc != pred_id2:
                # Both predictors are different from  the ancestor, and it is difficult to decide which one gives the benefits
                # We randomly select one of the descendants to update the predictor
                # The probability of selecting is proportional to the agg_scores of the descendants
                # prog_to_get_instruction_from = id1 if (rng.random() < (agg_scores[id1] / (agg_scores[id1] + agg_scores[id2]))) else id2
                prog_to_get_instruction_from = (
                    id1
                    if agg_scores[id1] > agg_scores[id2]
                    else (id2 if agg_scores[id2] > agg_scores[id1] else rng.choice([id1, id2]))
                )
                new_program[pred_name] = program_candidates[prog_to_get_instruction_from][pred_name]
                new_prog_desc = (*new_prog_desc, prog_to_get_instruction_from)
            elif pred_id1 == pred_id2:
                # Either both predictors are the same, or both are different from the ancestor
                # If both are different from the ancestor, we should use the new predictor, so selecting either one of the descendants is fine
                # If both are same as the ancesor, again selecting any one of the descendants is fine
                # So let's select id1
                new_program[pred_name] = program_candidates[id1][pred_name]
                new_prog_desc = (*new_prog_desc, id1)
            else:  # pragma: no cover - defensive
                raise AssertionError("Unexpected case in predictor merging logic")

        if (id1, id2, new_prog_desc) in merges_performed[1]:
            # This triplet has already been merged, so we skip it
            continue

        if has_val_support_overlap and not has_val_support_overlap(id1, id2):
            # not enough overlapping validation support for candidates
            continue

        merges_performed[1].append((id1, id2, new_prog_desc))

        return (new_program, id1, id2, ancestor)

    return None


class MergeProposer(ProposeNewCandidate[DataId]):
    """
    Implements merge flow that combines compatible descendants of a common ancestor.

    - Find merge candidates among Pareto front dominators
    - Attempt a merge via sample_and_attempt_merge_programs_by_common_predictors
    - Subsample eval on valset-driven selected indices
    - Return proposal if merge's subsample score >= max(parents)
    The engine handles full eval + adding to state.
    """

    def __init__(
        self,
        logger: LoggerProtocol,
        valset: DataLoader[DataId, DataInst],
        evaluator: EvaluatorFn,
        use_merge: bool,
        max_merge_invocations: int,
        val_overlap_floor: int = 5,
        rng: random.Random | None = None,
    ):
        self.logger = logger
        self.valset = valset
        self.evaluator = evaluator
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.rng = rng if rng is not None else random.Random(0)

        if val_overlap_floor <= 0:
            raise ValueError("val_overlap_floor should be a positive integer")
        self.val_overlap_floor = val_overlap_floor
        # Internal counters matching original behavior
        self.merges_due = 0
        self.total_merges_tested = 0
        self.merges_performed: tuple[list[AncestorLog], list[MergeDescription]] = ([], [])

        # Toggle controlled by engine: set True when last iter found new program
        self.last_iter_found_new_program = False

    def schedule_if_needed(self) -> None:
        if self.use_merge and self.total_merges_tested < self.max_merge_invocations:
            self.merges_due += 1

    def select_eval_subsample_for_merged_program(
        self,
        scores1: dict[DataId, float],
        scores2: dict[DataId, float],
        num_subsample_ids: int = 5,
    ) -> list[DataId]:
        common_ids = list(set(scores1.keys()) & set(scores2.keys()))

        p1 = [idx for idx in common_ids if scores1[idx] > scores2[idx]]
        p2 = [idx for idx in common_ids if scores2[idx] > scores1[idx]]
        p3 = [idx for idx in common_ids if idx not in p1 and idx not in p2]

        n_each = max(1, math.ceil(num_subsample_ids / 3))
        selected: list[DataId] = []
        for bucket in (p1, p2, p3):
            if len(selected) >= num_subsample_ids:
                break
            available = [idx for idx in bucket if idx not in selected]
            take = min(len(available), n_each, num_subsample_ids - len(selected))
            if take > 0:
                selected += self.rng.sample(available, k=take)

        remaining = num_subsample_ids - len(selected)
        if remaining > 0:
            unused = [idx for idx in common_ids if idx not in selected]
            if len(unused) >= remaining:
                selected += self.rng.sample(unused, k=remaining)
            else:
                selected += self.rng.choices(common_ids, k=remaining)

        return selected[:num_subsample_ids]

    def propose(self, state: GEPAState[RolloutOutput, DataId]) -> CandidateProposal[DataId] | None:
        i = state.i + 1
        state.full_program_trace[-1]["invoked_merge"] = True

        # Only attempt when scheduled by engine and after a new program in last iteration
        if not (self.use_merge and self.last_iter_found_new_program and self.merges_due > 0):
            self.logger.log(f"Iteration {i}: No merge candidates scheduled")
            return None

        def has_val_support_overlap(id1: ProgramIdx, id2: ProgramIdx) -> bool:
            common_ids = set(state.prog_candidate_val_subscores[id1].keys()) & set(
                state.prog_candidate_val_subscores[id2].keys()
            )
            return len(common_ids) >= self.val_overlap_floor

        pareto_front_programs = state.program_at_pareto_front_valset
        merge_candidates = find_dominator_programs(pareto_front_programs, state.program_full_scores_val_set)
        merge_output = sample_and_attempt_merge_programs_by_common_predictors(
            agg_scores=state.program_full_scores_val_set,
            rng=self.rng,
            merge_candidates=merge_candidates,
            merges_performed=self.merges_performed,
            program_candidates=state.program_candidates,
            parent_program_for_candidate=state.parent_program_for_candidate,
            has_val_support_overlap=has_val_support_overlap,
        )

        if merge_output is None:
            self.logger.log(f"Iteration {i}: No merge candidates found")
            return None

        # success, new_program, id1, id2, ancestor
        new_program, id1, id2, ancestor = merge_output
        state.full_program_trace[-1]["merged"] = True
        state.full_program_trace[-1]["merged_entities"] = (id1, id2, ancestor)
        self.merges_performed[0].append((id1, id2, ancestor))
        self.logger.log(f"Iteration {i}: Merged programs {id1} and {id2} via ancestor {ancestor}")

        subsample_ids = self.select_eval_subsample_for_merged_program(
            state.prog_candidate_val_subscores[id1],
            state.prog_candidate_val_subscores[id2],
        )
        mini_devset = self.valset.fetch(subsample_ids)
        # below is a post condition of `select_eval_subsample_for_merged_program`
        assert set(subsample_ids).issubset(state.prog_candidate_val_subscores[id1].keys())
        assert set(subsample_ids).issubset(state.prog_candidate_val_subscores[id2].keys())
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
            subsample_scores_before=[sum(id1_sub_scores), sum(id2_sub_scores)],
            subsample_scores_after=new_sub_scores,
            tag="merge",
            metadata={"ancestor": ancestor},
        )
