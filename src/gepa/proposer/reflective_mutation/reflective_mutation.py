# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections.abc import Mapping, Sequence
from typing import Any

from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.callbacks import notify_callbacks
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.instruction_proposal import InstructionProposalSignature


class ReflectiveMutationProposer(ProposeNewCandidate[DataId]):
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
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        candidate_selector: CandidateSelector,
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        perfect_score: float,
        skip_perfect_score: bool,
        experiment_tracker: Any,
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | None = None,
        callbacks: list | None = None,
    ):
        self.logger = logger
        self.trainset = ensure_loader(trainset)
        self.adapter = adapter
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.experiment_tracker = experiment_tracker
        self.reflection_lm = reflection_lm
        self.callbacks = callbacks

        InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)
        self.reflection_prompt_template = reflection_prompt_template

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)

        if self.reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")
        new_texts: dict[str, str] = {}
        for name in components_to_update:
            # Gracefully handle cases where a selected component has no data in reflective_dataset
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(
                    f"Component '{name}' is not in reflective dataset. Skipping."
                )
                continue

            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]
            new_texts[name] = InstructionProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                    "prompt_template": self.reflection_prompt_template,
                },
            )["new_instruction"]
        return new_texts

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        i = state.i + 1

        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        self.logger.log(
            f"Iteration {i}: Selected program {curr_prog_id} score: {state.program_full_scores_val_set[curr_prog_id]}"
        )

        # Notify candidate selected
        notify_callbacks(
            self.callbacks,
            "on_candidate_selected",
            iteration=i,
            candidate_idx=curr_prog_id,
            candidate=curr_prog,
            score=state.program_full_scores_val_set[curr_prog_id],
        )

        self.experiment_tracker.log_metrics({"iteration": i, "selected_program_candidate": curr_prog_id}, step=i)

        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = self.trainset.fetch(subsample_ids)

        # Notify minibatch sampled
        notify_callbacks(
            self.callbacks,
            "on_minibatch_sampled",
            iteration=i,
            minibatch_ids=subsample_ids,
            trainset_size=len(self.trainset),
        )

        # 1) Evaluate current program with traces
        curr_parent_ids = [p for p in state.parent_program_for_candidate[curr_prog_id] if p is not None]
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            iteration=i,
            candidate_idx=curr_prog_id,
            batch_size=len(minibatch),
            capture_traces=True,
            parent_ids=curr_parent_ids,
        )
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
        state.increment_evals(len(subsample_ids))
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores
        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            iteration=i,
            candidate_idx=curr_prog_id,
            scores=eval_curr.scores,
            has_trajectories=bool(eval_curr.trajectories),
            parent_ids=curr_parent_ids,
        )

        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                iteration=i,
                candidate_idx=curr_prog_id,
                reason="no_trajectories",
                scores=eval_curr.scores,
            )
            return None

        if self.skip_perfect_score and all(s >= self.perfect_score for s in eval_curr.scores):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                iteration=i,
                candidate_idx=curr_prog_id,
                reason="all_scores_perfect",
                scores=eval_curr.scores,
            )
            return None

        self.experiment_tracker.log_metrics({"subsample_score": sum(eval_curr.scores)}, step=i)

        # 2) Decide which predictors to update
        predictor_names_to_update = self.module_selector(
            state, eval_curr.trajectories, eval_curr.scores, curr_prog_id, curr_prog
        )

        # 3) Build reflective dataset and propose texts
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(curr_prog, eval_curr, predictor_names_to_update)

            # Notify reflective dataset built
            notify_callbacks(
                self.callbacks,
                "on_reflective_dataset_built",
                iteration=i,
                candidate_idx=curr_prog_id,
                components=predictor_names_to_update,
                dataset=dict(reflective_dataset),
            )

            # Notify proposal start
            notify_callbacks(
                self.callbacks,
                "on_proposal_start",
                iteration=i,
                parent_candidate=curr_prog,
                components=predictor_names_to_update,
                reflective_dataset=dict(reflective_dataset),
            )

            new_texts = self.propose_new_texts(curr_prog, reflective_dataset, predictor_names_to_update)

            # Notify proposal end
            notify_callbacks(
                self.callbacks,
                "on_proposal_end",
                iteration=i,
                new_instructions=new_texts,
            )

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

        # Evaluate new candidate (not yet in state)
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            iteration=i,
            candidate_idx=None,
            batch_size=len(minibatch),
            capture_traces=False,
            parent_ids=[curr_prog_id],
        )
        eval_new = self.adapter.evaluate(minibatch, new_candidate, capture_traces=False)
        state.increment_evals(len(subsample_ids))
        state.full_program_trace[-1]["new_subsample_scores"] = eval_new.scores
        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            iteration=i,
            candidate_idx=None,
            scores=eval_new.scores,
            has_trajectories=False,
            parent_ids=[curr_prog_id],
        )

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
