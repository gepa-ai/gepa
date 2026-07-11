# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from gepa.core.adapter import DataInst, RolloutOutput
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState, ObjectiveScores, ProgramIdx
from gepa.gepa_utils import find_dominator_programs
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate
from gepa.proposer.reflective_mutation.base import LanguageModel, Signature


class CrossoverSignature(Signature):
    """Synthesize a single instruction from two parents given where each one wins.

    Unlike merge (which swaps whole component texts along a shared genealogy),
    crossover asks the reflection LM to *integrate* two divergent instruction
    texts for the same component into one that keeps both strengths.
    """

    default_prompt_template = """Two variants of the same instruction each perform better on different inputs.

VARIANT A:
```
<parent_a>
```

VARIANT B:
```
<parent_b>
```

Evidence that their strengths are complementary:
```
<contrastive_evidence>
```

Your task is to write a SINGLE new instruction that keeps the strengths responsible
for A's wins AND the strengths responsible for B's wins, without regressing either.
Do not merely concatenate the two — integrate their underlying strategies into one
coherent instruction. Provide the new instruction within ``` blocks."""

    input_keys: ClassVar[list[str]] = ["parent_a", "parent_b", "contrastive_evidence", "prompt_template"]
    output_keys: ClassVar[list[str]] = ["new_instruction"]

    @classmethod
    def prompt_renderer(cls, input_dict: Mapping[str, Any]) -> str:
        parent_a = input_dict.get("parent_a")
        parent_b = input_dict.get("parent_b")
        evidence = input_dict.get("contrastive_evidence", "")
        if not isinstance(parent_a, str) or not isinstance(parent_b, str):
            raise TypeError("parent_a and parent_b must be strings")

        template = input_dict.get("prompt_template") or cls.default_prompt_template
        return (
            template.replace("<parent_a>", parent_a)
            .replace("<parent_b>", parent_b)
            .replace("<contrastive_evidence>", str(evidence))
        )

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        # Reuse the fenced-block extraction used by reflective mutation.
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        return InstructionProposalSignature.output_extractor(lm_out)


class CrossoverProposer(ProposeNewCandidate[DataId]):
    """LLM-synthesized recombination across lineages.

    Differs from :class:`~gepa.proposer.merge.MergeProposer` in three ways:

    - **No common-ancestor gate.** Parents are chosen by complementarity of their
      per-instance win-sets, not by shared genealogy.
    - **Synthesis, not swap.** A divergent component is recombined by asking the
      reflection LM to integrate both instruction texts (merge picks one wholesale).
    - **Reflection-cost, not rollout-cost.** The synthesis is one LM call on the
      reflection axis; the only metric calls are the child subsample eval (plus the
      full valset eval that the engine runs on acceptance).

    Scheduling mirrors ``MergeProposer``: the engine bumps ``crossovers_due`` and
    ``last_iter_found_new_program`` after a reflective proposal is accepted, and the
    engine decides acceptance (``sum(after) >= max(parents)``).
    """

    def __init__(
        self,
        logger: LoggerProtocol,
        valset: DataLoader[DataId, DataInst],
        evaluator: Callable[
            [list[DataInst], dict[str, str]],
            tuple[list[RolloutOutput], list[float], Sequence[ObjectiveScores] | None],
        ],
        reflection_lm: LanguageModel,
        use_crossover: bool,
        max_crossover_invocations: int = 5,
        num_subsample_ids: int = 5,
        min_win_asymmetry: int = 2,
        crossover_prompt_template: str | None = None,
        rng: random.Random | None = None,
        callbacks: list[Any] | None = None,
    ):
        if num_subsample_ids <= 0:
            raise ValueError("num_subsample_ids should be a positive integer")
        if min_win_asymmetry <= 0:
            raise ValueError("min_win_asymmetry should be a positive integer")

        self.logger = logger
        self.valset = valset
        self.evaluator = evaluator
        self.reflection_lm = reflection_lm
        self.use_crossover = use_crossover
        self.max_crossover_invocations = max_crossover_invocations
        self.num_subsample_ids = num_subsample_ids
        self.min_win_asymmetry = min_win_asymmetry
        self.crossover_prompt_template = crossover_prompt_template
        self.rng = rng if rng is not None else random.Random(0)
        self.callbacks = callbacks

        # Scheduling counters, mirroring MergeProposer.
        self.crossovers_due = 0
        self.total_crossovers_tested = 0
        self.last_iter_found_new_program = False
        # (id1, id2, component) triplets already synthesized, and pairs whose
        # divergent components are all used up (so they are not re-selected).
        self._attempted: set[tuple[int, int, str]] = set()
        self._exhausted_pairs: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Parent selection: complementarity, not genealogy
    # ------------------------------------------------------------------

    def _win_sets(
        self, id1: ProgramIdx, id2: ProgramIdx, state: GEPAState
    ) -> tuple[list[DataId], list[DataId], list[DataId]]:
        """Partition shared validation instances into (A wins, B wins, ties)."""
        scores1 = state.prog_candidate_val_subscores[id1]
        scores2 = state.prog_candidate_val_subscores[id2]
        common = set(scores1.keys()) & set(scores2.keys())
        a_wins = [i for i in common if scores1[i] > scores2[i]]
        b_wins = [i for i in common if scores2[i] > scores1[i]]
        ties = [i for i in common if i not in a_wins and i not in b_wins]
        return a_wins, b_wins, ties

    def _select_complementary_pair(self, state: GEPAState) -> tuple[int, int] | None:
        """Among Pareto dominators, pick the pair whose disjoint win-sets give the
        largest *mutual* contribution (the bricolage of genuinely different strengths).

        Ranking by ``min(|A wins|, |B wins|)`` keeps both parents substantial while
        never spuriously rejecting a valid-but-lopsided pair that already cleared the
        ``min_win_asymmetry`` gate.
        """
        dominators = find_dominator_programs(
            state.get_pareto_front_mapping(),
            list(state.per_program_tracked_scores),
        )
        if len(dominators) < 2:
            return None

        best_pair: tuple[int, int] | None = None
        best_score = -1
        for x in range(len(dominators)):
            for y in range(x + 1, len(dominators)):
                i, j = sorted((dominators[x], dominators[y]))
                if (i, j) in self._exhausted_pairs:
                    continue
                a_wins, b_wins, _ = self._win_sets(i, j, state)
                if len(a_wins) < self.min_win_asymmetry or len(b_wins) < self.min_win_asymmetry:
                    continue
                score = min(len(a_wins), len(b_wins))
                if best_pair is None or score > best_score:
                    best_pair = (i, j)
                    best_score = score
        return best_pair

    def _pick_divergent_component(self, id1: int, id2: int, state: GEPAState) -> str | None:
        """A not-yet-tried component whose text differs between the two parents."""
        prog1 = state.program_candidates[id1]
        prog2 = state.program_candidates[id2]
        diverging = [
            name
            for name in prog1
            if prog1[name] != prog2.get(name) and (id1, id2, name) not in self._attempted
        ]
        if not diverging:
            return None
        return self.rng.choice(diverging)

    # ------------------------------------------------------------------
    # Evidence + subsample construction
    # ------------------------------------------------------------------

    def _render_contrastive_evidence(
        self,
        id1: ProgramIdx,
        id2: ProgramIdx,
        a_wins: list[DataId],
        b_wins: list[DataId],
        state: GEPAState,
    ) -> str:
        scores1 = state.prog_candidate_val_subscores[id1]
        scores2 = state.prog_candidate_val_subscores[id2]
        a_delta = sum(scores1[i] - scores2[i] for i in a_wins)
        b_delta = sum(scores2[i] - scores1[i] for i in b_wins)
        return (
            f"Variant A uniquely wins on {len(a_wins)} evaluation instances "
            f"(cumulative score advantage {a_delta:.3f}).\n"
            f"Variant B uniquely wins on {len(b_wins)} evaluation instances "
            f"(cumulative score advantage {b_delta:.3f}).\n"
            "The two variants are strong on largely disjoint inputs, so their "
            "underlying strategies are complementary rather than redundant."
        )

    def _select_contested_subsample(
        self, a_wins: list[DataId], b_wins: list[DataId], ties: list[DataId]
    ) -> list[DataId]:
        """Balanced sample across (A wins, B wins, ties), biased to contested inputs."""
        n = self.num_subsample_ids
        n_each = max(1, math.ceil(n / 3))
        selected: list[DataId] = []
        for bucket in (a_wins, b_wins, ties):
            if len(selected) >= n:
                break
            available = [i for i in bucket if i not in selected]
            take = min(len(available), n_each, n - len(selected))
            if take > 0:
                selected += self.rng.sample(available, k=take)

        if len(selected) < n:
            remaining = [i for i in (a_wins + b_wins + ties) if i not in selected]
            take = min(len(remaining), n - len(selected))
            if take > 0:
                selected += self.rng.sample(remaining, k=take)

        return selected[:n]

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def propose(self, state: GEPAState[RolloutOutput, DataId]) -> CandidateProposal[DataId] | None:
        i = state.i + 1
        state.full_program_trace[-1]["invoked_crossover"] = True

        if not (self.use_crossover and self.last_iter_found_new_program and self.crossovers_due > 0):
            self.logger.log(f"Iteration {i}: No crossover scheduled")
            return None

        pair = self._select_complementary_pair(state)
        if pair is None:
            self.logger.log(f"Iteration {i}: No complementary crossover pair found")
            return None
        id1, id2 = pair

        component = self._pick_divergent_component(id1, id2, state)
        if component is None:
            # No untried divergent component remains for this pair: retire it so it is
            # not re-selected on future crossover schedules.
            self._exhausted_pairs.add((id1, id2))
            self.logger.log(f"Iteration {i}: No untried divergent component for crossover pair, skipping")
            return None
        self._attempted.add((id1, id2, component))

        a_wins, b_wins, ties = self._win_sets(id1, id2, state)
        evidence = self._render_contrastive_evidence(id1, id2, a_wins, b_wins, state)

        # Reflection-cost axis: a single synthesis call (no rollouts).
        try:
            result, _prompt, raw_output = CrossoverSignature.run_with_metadata(
                lm=self.reflection_lm,
                input_dict={
                    "parent_a": state.program_candidates[id1][component],
                    "parent_b": state.program_candidates[id2][component],
                    "contrastive_evidence": evidence,
                    "prompt_template": self.crossover_prompt_template,
                },
            )
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during crossover synthesis: {e}")
            return None

        new_text = result["new_instruction"]
        if not new_text:
            self.logger.log(f"Iteration {i}: Crossover produced empty instruction, skipping")
            return None

        child = dict(state.program_candidates[id1])
        child[component] = new_text

        subsample_ids = self._select_contested_subsample(a_wins, b_wins, ties)
        if not subsample_ids:
            self.logger.log(f"Iteration {i}: No contested subsample available for crossover, skipping")
            return None

        # Metric-call axis: evaluate the child on the contested subsample only.
        _outputs, scores_by_id, _obj_by_id, actual_evals = state.cached_evaluate_full(
            child, subsample_ids, self.valset.fetch, self.evaluator
        )
        state.increment_evals(actual_evals)

        id1_sub_scores = [state.prog_candidate_val_subscores[id1][k] for k in subsample_ids]
        id2_sub_scores = [state.prog_candidate_val_subscores[id2][k] for k in subsample_ids]
        new_sub_scores = [scores_by_id[k] for k in subsample_ids]

        state.full_program_trace[-1]["crossover_entities"] = (id1, id2, component)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        state.full_program_trace[-1]["new_program_subsample_scores"] = new_sub_scores

        self.logger.log(f"Iteration {i}: Crossover of programs {id1} x {id2} on component '{component}'")

        # Acceptance decided by engine: sum(after) >= max(subsample_scores_before).
        return CandidateProposal(
            candidate=child,
            parent_program_ids=[id1, id2],
            subsample_indices=subsample_ids,
            subsample_scores_before=[sum(id1_sub_scores), sum(id2_sub_scores)],
            subsample_scores_after=new_sub_scores,
            tag="crossover",
            metadata={"component": component, f"raw_lm_output:{component}": raw_output},
        )
