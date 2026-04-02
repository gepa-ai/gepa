# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from gepa.core.adapter import DataInst, GEPAAdapter, ProposalFn, RolloutOutput, Trajectory
from gepa.core.callbacks import (
    CandidateSelectedEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    GEPACallback,
    MinibatchSampledEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    notify_callbacks,
)
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

_DEFAULT_COMBEE_AGGREGATION_PROMPT = """I provided an assistant with the following current instruction:
```
<curr_param>
```

Multiple parallel processes each independently proposed an updated instruction based on a different subset of evaluation examples. Here are the proposed updates:
```
<side_info>
```

Your task is to synthesize these proposed instruction updates into a single, comprehensive instruction. Incorporate the key improvements and specific insights from all proposals. When proposals offer complementary guidance, include all relevant details. When they conflict, prefer more specific and task-relevant guidance.

Provide the final synthesized instruction within ``` blocks."""


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
        perfect_score: float | None,
        skip_perfect_score: bool,
        experiment_tracker: Any,
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | dict[str, str] | None = None,
        custom_candidate_proposer: ProposalFn | None = None,
        callbacks: list[GEPACallback] | None = None,
        # ComBEE aggregation options
        aggregation_method: Literal["naive", "combee"] = "naive",
        combee_duplication_factor: int = 2,
        combee_aggregation_prompt: str | None = None,
        rng: random.Random | None = None,
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
        self.custom_candidate_proposer = custom_candidate_proposer
        self.callbacks = callbacks

        self.reflection_prompt_template = reflection_prompt_template
        # Track parameters for which we've already logged missing template warnings
        self._missing_template_warnings: set[str] = set()

        if aggregation_method not in ("naive", "combee"):
            raise ValueError(f"aggregation_method must be 'naive' or 'combee', got {aggregation_method!r}")
        self.aggregation_method = aggregation_method
        self.combee_duplication_factor = combee_duplication_factor
        self._combee_aggregation_prompt = combee_aggregation_prompt or _DEFAULT_COMBEE_AGGREGATION_PROMPT
        self._rng = rng if rng is not None else random.Random(0)

        if isinstance(reflection_prompt_template, dict):
            for _param_name, template in reflection_prompt_template.items():
                InstructionProposalSignature.validate_prompt_template(template)
        else:
            InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)

        if self.skip_perfect_score and self.perfect_score is None:
            raise ValueError(
                "perfect_score must be provided when skip_perfect_score is True. "
                "If you do not have a perfect target score, set skip_perfect_score=False."
            )

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str]]:
        """Propose new instruction texts for the given components.

        Returns:
            A tuple of (new_texts, prompts, raw_lm_outputs) where each is a
            dict keyed by component name.  When the adapter or a custom proposer
            handles the call, prompts and raw_lm_outputs are empty dicts.
        """
        empty: dict[str, str | list[dict[str, Any]]] = {}
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update), empty, {}

        if self.custom_candidate_proposer is not None:
            return self.custom_candidate_proposer(candidate, reflective_dataset, components_to_update), empty, {}

        if self.reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")

        new_texts: dict[str, str] = {}
        prompts: dict[str, str | list[dict[str, Any]]] = {}
        raw_lm_outputs: dict[str, str] = {}
        for name in components_to_update:
            # Gracefully handle cases where a selected component has no data in reflective_dataset
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(f"Component '{name}' is not in reflective dataset. Skipping.")
                continue

            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]

            # Determine which prompt template to use for this parameter
            prompt_template = None
            if isinstance(self.reflection_prompt_template, dict):
                # Use parameter-specific template if available
                prompt_template = self.reflection_prompt_template.get(name)
                if prompt_template is None and name not in self._missing_template_warnings:
                    self.logger.log(
                        f"No reflection_prompt_template found for parameter '{name}'. Using default template."
                    )
                    self._missing_template_warnings.add(name)
            else:
                # Use the single template for all parameters
                prompt_template = self.reflection_prompt_template

            result, prompt, raw_output = InstructionProposalSignature.run_with_metadata(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                    "prompt_template": prompt_template,
                },
            )
            new_texts[name] = result["new_instruction"]
            prompts[name] = prompt
            raw_lm_outputs[name] = raw_output
        return new_texts, prompts, raw_lm_outputs

    def _propose_new_texts_combee(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[dict[str, str], dict[str, str | list[dict[str, Any]]], dict[str, str]]:
        """ComBEE parallel scan aggregation with augmented shuffling (paper §3.1-3.2).

        For each component:
          1. Augmented shuffle: duplicate each reflection p times, shuffle.
          2. Split into k = ⌊√n⌋ groups (n = number of original reflections).
          3. Level-1 (Map): call the reflection LM once per group to get k intermediate instructions.
          4. Level-2 (Reduce): aggregate the k intermediate instructions into one final instruction.
        """
        assert self.reflection_lm is not None, "reflection_lm is required for ComBEE aggregation"
        p = self.combee_duplication_factor

        new_texts: dict[str, str] = {}
        prompts: dict[str, str | list[dict[str, Any]]] = {}
        raw_lm_outputs: dict[str, str] = {}

        for comp in components_to_update:
            if comp not in reflective_dataset or not reflective_dataset.get(comp):
                self.logger.log(f"Component '{comp}' is not in reflective dataset. Skipping.")
                continue

            if comp not in candidate:
                self.logger.log(f"Component '{comp}' is missing from candidate. Skipping.")
                continue

            records = list(reflective_dataset[comp])
            n = len(records)

            # k = ⌊√n⌋ — default from paper §3.1
            k = max(1, int(n**0.5))

            if k <= 1:
                # Degenerate case: fall back to the standard single-group reflection
                naive_texts, naive_prompts, naive_raw = self.propose_new_texts(candidate, {comp: records}, [comp])
                if comp in naive_texts:
                    new_texts[comp] = naive_texts[comp]
                    prompts[comp] = naive_prompts.get(comp, "")
                    raw_lm_outputs[comp] = naive_raw.get(comp, "")
                continue

            # --- Augmented shuffle (paper §3.2) ---
            # Duplicate each reflection p times, then shuffle the full augmented list.
            augmented: list[Mapping[str, Any]] = records * p
            self._rng.shuffle(augmented)
            total_aug = len(augmented)  # = p * n

            # Determine per-component prompt template (level-1 uses the user template)
            if isinstance(self.reflection_prompt_template, dict):
                level1_prompt_template = self.reflection_prompt_template.get(comp)
            else:
                level1_prompt_template = self.reflection_prompt_template

            # --- Level-1 (Map): one LM call per group ---
            base_group_size = total_aug // k
            remainder = total_aug % k
            group_proposals: list[str] = []
            offset = 0
            for i in range(k):
                size = base_group_size + (1 if i < remainder else 0)
                group_records = augmented[offset : offset + size]
                offset += size

                if not group_records:
                    continue

                result, _, _ = InstructionProposalSignature.run_with_metadata(
                    lm=self.reflection_lm,
                    input_dict={
                        "current_instruction_doc": candidate[comp],
                        "dataset_with_feedback": group_records,
                        "prompt_template": level1_prompt_template,
                    },
                )
                group_proposals.append(result["new_instruction"])

            if not group_proposals:
                continue

            if len(group_proposals) == 1:
                # Only one valid group produced a proposal — skip the aggregation step
                new_texts[comp] = group_proposals[0]
                continue

            # --- Level-2 (Reduce): aggregate k proposals into one final instruction ---
            # Each proposal is treated as a single-field record so it renders cleanly
            # inside the aggregation prompt's <side_info> block.
            agg_records: list[Mapping[str, Any]] = [
                {"Proposed Instruction Update": prop} for prop in group_proposals
            ]

            result, prompt, raw_output = InstructionProposalSignature.run_with_metadata(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": candidate[comp],
                    "dataset_with_feedback": agg_records,
                    "prompt_template": self._combee_aggregation_prompt,
                },
            )
            new_texts[comp] = result["new_instruction"]
            prompts[comp] = prompt
            raw_lm_outputs[comp] = raw_output

        return new_texts, prompts, raw_lm_outputs

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
            CandidateSelectedEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                candidate=curr_prog,
                score=state.program_full_scores_val_set[curr_prog_id],
            ),
        )

        self.experiment_tracker.log_metrics(
            {"iteration": i, "selected_program_candidate": curr_prog_id, "total_metric_calls": state.total_num_evals},
            step=i,
        )

        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = self.trainset.fetch(subsample_ids)

        # Notify minibatch sampled
        notify_callbacks(
            self.callbacks,
            "on_minibatch_sampled",
            MinibatchSampledEvent(
                iteration=i,
                minibatch_ids=subsample_ids,
                trainset_size=len(self.trainset),
            ),
        )

        # 1) Evaluate current program with traces
        # Note: We don't use cache for capture_traces=True evaluations since we need fresh traces for reflection
        curr_parent_ids = [p for p in state.parent_program_for_candidate[curr_prog_id] if p is not None]
        is_seed_candidate = curr_prog_id == 0
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                batch_size=len(minibatch),
                capture_traces=True,
                parent_ids=curr_parent_ids,
                inputs=minibatch,
                is_seed_candidate=is_seed_candidate,
            ),
        )
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
        state.increment_evals(len(subsample_ids))
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores
        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                scores=eval_curr.scores,
                has_trajectories=bool(eval_curr.trajectories),
                parent_ids=curr_parent_ids,
                outputs=eval_curr.outputs,
                trajectories=eval_curr.trajectories,
                objective_scores=eval_curr.objective_scores,
                is_seed_candidate=is_seed_candidate,
            ),
        )

        # Update cache with current program evaluation results (for future reuse when capture_traces=False)
        if state.evaluation_cache is not None:
            objective_scores_list = list(eval_curr.objective_scores) if eval_curr.objective_scores else None
            state.evaluation_cache.put_batch(
                curr_prog, subsample_ids, eval_curr.outputs, eval_curr.scores, objective_scores_list
            )

        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    reason="no_trajectories",
                    scores=eval_curr.scores,
                    is_seed_candidate=is_seed_candidate,
                ),
            )
            return None

        if (
            self.skip_perfect_score
            and self.perfect_score is not None
            and all(s is not None and s >= self.perfect_score for s in eval_curr.scores)
        ):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    reason="all_scores_perfect",
                    scores=eval_curr.scores,
                    is_seed_candidate=is_seed_candidate,
                ),
            )
            return None

        subsample_before = sum(eval_curr.scores)

        # 2) Decide which predictors to update
        predictor_names_to_update = self.module_selector(
            state, eval_curr.trajectories, eval_curr.scores, curr_prog_id, curr_prog
        )

        # 3) Build reflective dataset and propose texts
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(curr_prog, eval_curr, predictor_names_to_update)

            # Convert to concrete types for callback
            reflective_dataset_concrete: dict[str, list[dict[str, Any]]] = {
                k: [dict(item) for item in v] for k, v in reflective_dataset.items()
            }

            # Notify reflective dataset built
            notify_callbacks(
                self.callbacks,
                "on_reflective_dataset_built",
                ReflectiveDatasetBuiltEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    components=predictor_names_to_update,
                    dataset=reflective_dataset_concrete,
                ),
            )

            # Notify proposal start
            notify_callbacks(
                self.callbacks,
                "on_proposal_start",
                ProposalStartEvent(
                    iteration=i,
                    parent_candidate=curr_prog,
                    components=predictor_names_to_update,
                    reflective_dataset=reflective_dataset_concrete,
                ),
            )

            # ComBEE is only used when reflection_lm is available and no custom
            # adapter / proposer overrides the proposal step (§3.1-3.2).
            use_combee = (
                self.aggregation_method == "combee"
                and self.reflection_lm is not None
                and self.adapter.propose_new_texts is None
                and self.custom_candidate_proposer is None
            )
            if use_combee:
                new_texts, prompts, raw_lm_outputs = self._propose_new_texts_combee(
                    curr_prog, reflective_dataset, predictor_names_to_update
                )
            else:
                new_texts, prompts, raw_lm_outputs = self.propose_new_texts(
                    curr_prog, reflective_dataset, predictor_names_to_update
                )

            # Notify proposal end
            notify_callbacks(
                self.callbacks,
                "on_proposal_end",
                ProposalEndEvent(
                    iteration=i,
                    new_instructions=new_texts,
                    prompts=prompts,
                    raw_lm_outputs=raw_lm_outputs,
                ),
            )

            # Stash LM call data in proposal metadata so the engine can log
            # it to experiment tracking once it knows accept/reject + candidate_idx.
            _lm_metadata: dict[str, Any] = {}
            for comp in new_texts:
                _lm_metadata[f"prompt:{comp}"] = prompts.get(comp, "")
                _lm_metadata[f"raw_lm_output:{comp}"] = raw_lm_outputs.get(comp, "")

            for pname, text in new_texts.items():
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")
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

        def evaluator(b, c):
            r = self.adapter.evaluate(b, c, capture_traces=False)
            return r.outputs, r.scores, list(r.objective_scores) if r.objective_scores else None

        # Evaluate new candidate (not yet in state)
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=None,
                batch_size=len(minibatch),
                capture_traces=False,
                parent_ids=[curr_prog_id],
                inputs=minibatch,
                is_seed_candidate=False,
            ),
        )

        outputs_by_id, scores_by_id, objective_by_id, actual_evals_count = state.cached_evaluate_full(
            new_candidate, subsample_ids, self.trainset.fetch, evaluator
        )
        new_scores = [scores_by_id[eid] for eid in subsample_ids]
        outputs = [outputs_by_id[eid] for eid in subsample_ids]

        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=None,
                scores=new_scores,
                has_trajectories=False,
                parent_ids=[curr_prog_id],
                outputs=outputs,
                trajectories=None,
                objective_scores=[objective_by_id[eid] for eid in subsample_ids] if objective_by_id else None,
                is_seed_candidate=False,
            ),
        )

        state.increment_evals(actual_evals_count)
        state.full_program_trace[-1]["new_subsample_scores"] = new_scores

        new_sum = sum(new_scores)
        self.experiment_tracker.log_metrics(
            {
                "subsample/before": subsample_before,
                "subsample/after": new_sum,
                "total_metric_calls": state.total_num_evals,
            },
            step=i,
        )

        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=new_scores,
            tag="reflective_mutation",
            metadata=_lm_metadata,
        )
