# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any

from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate
from gepa.proposer.reflective_mutation.base import (
    BatchSampler,
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)


class ReflectiveMutationProposer(ProposeNewCandidate):
    """
    Implements current reflective mutation flow:
    - Select candidate via selector
    - Select minibatch via sampler
    - capture_traces_and_eval -> trajectories, subsample_scores
    - skip if all scores==perfect and skip_perfect_score
    - reflection + mutate -> new candidate
    - evaluate new candidate on same minibatch -> new_subsample_scores
    - Return proposal if improved; else None
    """

    def __init__(
        self,
        logger: Any,
        trainset: list[DataInst],
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        candidate_selector: CandidateSelector,
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler,
        perfect_score: float,
        skip_perfect_score: bool,
        experiment_tracker: Any,
        reflection_lm: LanguageModel | None = None,
    ):
        self.logger = logger
        self.trainset = trainset
        self.adapter = adapter
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.experiment_tracker = experiment_tracker
        self.reflection_lm = reflection_lm

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)

        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        new_texts: dict[str, str] = {}
        for name in components_to_update:
            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]
            new_texts[name] = InstructionProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                },
            )["new_instruction"]
        return new_texts

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        i = state.i + 1

        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        self.logger.log(
            f"Iteration {i}: Selected program {curr_prog_id} score: {state.per_program_tracked_scores[curr_prog_id]}"
        )

        self.experiment_tracker.log_metrics({"iteration": i, "selected_program_candidate": curr_prog_id}, step=i)

        subsample_ids = self.batch_sampler.next_minibatch_indices(len(self.trainset), i - 1)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = [self.trainset[j] for j in subsample_ids]

        # 1) Evaluate current program with traces
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            return None

        state.total_num_evals += len(subsample_ids)
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores

        if self.skip_perfect_score and all(s >= self.perfect_score for s in eval_curr.scores):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            return None

        self.experiment_tracker.log_metrics({"subsample_score": sum(eval_curr.scores)}, step=i)

        # 2) Decide which predictors to update
        predictor_names_to_update = self.module_selector(
            state, eval_curr.trajectories, eval_curr.scores, curr_prog_id, curr_prog
        )

        # 3) Build reflective dataset and propose texts
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(curr_prog, eval_curr, predictor_names_to_update)
            new_texts = self.propose_new_texts(curr_prog, reflective_dataset, predictor_names_to_update)
            for pname, text in new_texts.items():
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")
            self.experiment_tracker.log_metrics(
                {f"new_instruction_{pname}": text for pname, text in new_texts.items()}, step=i
            )
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            import traceback

            self.logger.log(traceback.format_exc())
            return None

        # 4) Create candidate, evaluate on same minibatch (no need to capture traces)
        new_candidate = curr_prog.copy()
        for pname, text in new_texts.items():
            assert pname in new_candidate, f"{pname} missing in candidate"
            new_candidate[pname] = text

        eval_new = self.adapter.evaluate(minibatch, new_candidate, capture_traces=False)
        state.total_num_evals += len(subsample_ids)
        state.full_program_trace[-1]["new_subsample_scores"] = eval_new.scores

        new_sum = sum(eval_new.scores)
        self.experiment_tracker.log_metrics({"new_subsample_score": new_sum}, step=i)

        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=eval_new.scores,
            tag="reflective_mutation",
        )
