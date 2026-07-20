# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Fitness-based PromptBreeder co-evolution for GEPA.

This module evolves two coupled artifacts:

1. GEPA's task prompt/components, which remain the candidates evaluated by GEPA.
2. A hidden PromptBreeder genome for each task component. The genome contains
   the mutation prompt and thinking style used by the reflection model.

A breeder genome receives fitness credit from the task-prompt children it
creates. Accepted child prompts inherit the trial breeder genome; rejected
trial genomes are discarded while their parent mutators receive offspring
performance statistics. GEPA remains responsible for minibatch/full validation
execution, Pareto candidate selection, acceptance, and merging.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionJob, ReflectionProposal
from gepa.strategies.instruction_proposal import InstructionProposalSignature

PromptBreederOperator = Literal["zero_order", "hypermutation", "lineage"]

DEFAULT_MUTATION_PROMPTS: tuple[str, ...] = (
    "Identify generalizable failure patterns and rewrite the task instruction to prevent them.",
    "Preserve successful behavior, remove ambiguity, and add explicit decision rules for recurring edge cases.",
    "Convert evaluator feedback into a compact expert checklist, then encode that checklist in the task instruction.",
    "Find hidden assumptions in the current instruction and replace them with operational guidance.",
    "Improve robustness across diverse inputs without overfitting to the observed examples.",
)

DEFAULT_THINKING_STYLES: tuple[str, ...] = (
    "reason from first principles",
    "act as a skeptical error analyst",
    "use counterexamples to challenge assumptions",
    "separate invariant rules from example-specific details",
    "compare multiple rewrites before selecting one",
    "optimize for clarity, coverage, and minimal redundancy",
)


@dataclass(frozen=True)
class PromptBreederConfig:
    """Configuration for fitness-based reflection-process co-evolution."""

    mutation_prompts: tuple[str, ...] = DEFAULT_MUTATION_PROMPTS
    thinking_styles: tuple[str, ...] = DEFAULT_THINKING_STYLES
    operator_weights: Mapping[PromptBreederOperator, float] = field(
        default_factory=lambda: {
            "zero_order": 0.30,
            "hypermutation": 0.40,
            "lineage": 0.30,
        }
    )
    lineage_size: int = 6
    elite_pool_size: int = 8
    exploration_rate: float = 0.15
    adaptation_strength: float = 1.5
    history_limit: int = 250
    max_meta_feedback_chars: int = 6000
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.mutation_prompts:
            raise ValueError("mutation_prompts must contain at least one prompt")
        if not self.thinking_styles:
            raise ValueError("thinking_styles must contain at least one style")
        if self.lineage_size < 1:
            raise ValueError("lineage_size must be at least 1")
        if self.elite_pool_size < 1:
            raise ValueError("elite_pool_size must be at least 1")
        if not 0 <= self.exploration_rate <= 1:
            raise ValueError("exploration_rate must be between 0 and 1")
        if self.adaptation_strength < 0:
            raise ValueError("adaptation_strength must be non-negative")
        if self.history_limit < 1:
            raise ValueError("history_limit must be at least 1")
        if self.max_meta_feedback_chars < 256:
            raise ValueError("max_meta_feedback_chars must be at least 256")

        supported: set[PromptBreederOperator] = {"zero_order", "hypermutation", "lineage"}
        unknown = set(self.operator_weights) - supported
        if unknown:
            raise ValueError(f"Unknown PromptBreeder operator(s): {sorted(unknown)}")
        if any(weight < 0 for weight in self.operator_weights.values()):
            raise ValueError("operator weights must be non-negative")
        if sum(self.operator_weights.values()) <= 0:
            raise ValueError("at least one PromptBreeder operator must have positive weight")


@dataclass
class BreederGenome:
    """The inheritable reflection process paired with one task component."""

    genome_id: str
    component: str
    mutation_prompt: str
    thinking_style: str
    generation: int
    parent_genome_ids: tuple[str, ...]
    operator: str
    creation_reward: float = 0.0
    offspring_attempts: int = 0
    offspring_accepts: int = 0
    offspring_rejects: int = 0
    offspring_reward: float = 0.0
    task_lineage: tuple[str, ...] = ()

    @property
    def mean_offspring_reward(self) -> float:
        if self.offspring_attempts == 0:
            return 0.0
        return self.offspring_reward / self.offspring_attempts

    @property
    def acceptance_rate(self) -> float:
        if self.offspring_attempts == 0:
            return 0.0
        return self.offspring_accepts / self.offspring_attempts

    @property
    def fitness(self) -> float:
        return self.creation_reward + 0.5 * self.mean_offspring_reward + 0.025 * self.acceptance_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "component": self.component,
            "mutation_prompt": self.mutation_prompt,
            "thinking_style": self.thinking_style,
            "generation": self.generation,
            "parent_genome_ids": list(self.parent_genome_ids),
            "operator": self.operator,
            "creation_reward": self.creation_reward,
            "offspring_attempts": self.offspring_attempts,
            "offspring_accepts": self.offspring_accepts,
            "offspring_rejects": self.offspring_rejects,
            "offspring_reward": self.offspring_reward,
            "task_lineage": list(self.task_lineage),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BreederGenome:
        return cls(
            genome_id=str(value["genome_id"]),
            component=str(value["component"]),
            mutation_prompt=str(value["mutation_prompt"]),
            thinking_style=str(value["thinking_style"]),
            generation=int(value.get("generation", 0)),
            parent_genome_ids=tuple(str(x) for x in value.get("parent_genome_ids", ())),
            operator=str(value.get("operator", "seed")),
            creation_reward=float(value.get("creation_reward", 0.0)),
            offspring_attempts=int(value.get("offspring_attempts", 0)),
            offspring_accepts=int(value.get("offspring_accepts", 0)),
            offspring_rejects=int(value.get("offspring_rejects", 0)),
            offspring_reward=float(value.get("offspring_reward", 0.0)),
            task_lineage=tuple(str(x) for x in value.get("task_lineage", ())),
        )


class PromptBreederReflectionLM:
    """ReflectionLM that co-evolves task prompts and reflection genomes."""

    STATE_VERSION = 2

    def __init__(
        self,
        lm: LanguageModel,
        config: PromptBreederConfig | None = None,
        logger: Any | None = None,
        reflection_prompt_template: str | dict[str, str] | None = None,
        *,
        _state: Mapping[str, Any] | None = None,
    ):
        self.lm = lm
        self.config = config or PromptBreederConfig()
        self.logger = logger
        self.reflection_prompt_template = reflection_prompt_template
        self._rng = random.Random(self.config.seed)
        self._genomes: dict[str, BreederGenome] = {}
        self._candidate_genomes: dict[str, dict[str, str]] = {}
        self._candidate_index_genomes: dict[int, dict[str, str]] = {}
        self._operator_stats: dict[str, dict[str, float]] = {
            name: {"attempts": 0.0, "accepts": 0.0, "rejects": 0.0, "reward": 0.0}
            for name in ("zero_order", "hypermutation", "lineage")
        }
        self._history: list[dict[str, Any]] = []
        self._next_genome_number = 1
        if _state is not None:
            self.set_state(_state)

    @property
    def total_cost(self) -> float:
        cost = getattr(self.lm, "total_cost", 0.0)
        return float(cost) if cost is not None else 0.0

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)

    @staticmethod
    def _candidate_key(candidate: Mapping[str, str]) -> str:
        payload = json.dumps(sorted(candidate.items()), separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _tupleize(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(PromptBreederReflectionLM._tupleize(v) for v in value)
        if isinstance(value, dict):
            return {k: PromptBreederReflectionLM._tupleize(v) for k, v in value.items()}
        return value

    def get_state(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "rng_state": self._rng.getstate(),
            "next_genome_number": self._next_genome_number,
            "genomes": {gid: genome.to_dict() for gid, genome in self._genomes.items()},
            "candidate_genomes": copy.deepcopy(self._candidate_genomes),
            "candidate_index_genomes": copy.deepcopy(self._candidate_index_genomes),
            "operator_stats": copy.deepcopy(self._operator_stats),
            "history": copy.deepcopy(self._history),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        version = int(state.get("version", 1))
        if version not in (1, self.STATE_VERSION):
            raise ValueError(f"Unsupported PromptBreeder state version: {version}")
        self._rng.setstate(self._tupleize(state["rng_state"]))
        self._next_genome_number = int(state.get("next_genome_number", 1))
        self._genomes = {str(gid): BreederGenome.from_dict(raw) for gid, raw in dict(state.get("genomes", {})).items()}
        self._candidate_genomes = {
            str(candidate_key): {str(component): str(gid) for component, gid in mapping.items()}
            for candidate_key, mapping in dict(state.get("candidate_genomes", {})).items()
        }
        self._candidate_index_genomes = {
            int(candidate_idx): {str(component): str(gid) for component, gid in mapping.items()}
            for candidate_idx, mapping in dict(state.get("candidate_index_genomes", {})).items()
        }
        self._operator_stats = {
            str(operator): {
                "attempts": float(stats.get("attempts", 0.0)),
                "accepts": float(stats.get("accepts", 0.0)),
                "rejects": float(stats.get("rejects", 0.0)),
                "reward": float(stats.get("reward", 0.0)),
            }
            for operator, stats in dict(state.get("operator_stats", {})).items()
        }
        for operator in ("zero_order", "hypermutation", "lineage"):
            self._operator_stats.setdefault(operator, {"attempts": 0.0, "accepts": 0.0, "rejects": 0.0, "reward": 0.0})
            self._operator_stats[operator].setdefault("rejects", 0.0)
        self._history = [dict(event) for event in state.get("history", [])][-self.config.history_limit :]

    def synchronize_state(self, state: Any) -> None:
        program_candidates = getattr(state, "program_candidates", None)
        if not isinstance(program_candidates, list):
            return
        for idx, candidate in enumerate(program_candidates):
            if not isinstance(candidate, Mapping):
                continue
            mapping = self._ensure_candidate_genomes(candidate)
            self._candidate_index_genomes.setdefault(idx, dict(mapping))

    def _fork(self) -> PromptBreederReflectionLM:
        return PromptBreederReflectionLM(
            lm=self.lm,
            config=self.config,
            logger=self.logger,
            reflection_prompt_template=self.reflection_prompt_template,
            _state=self.get_state(),
        )

    def _new_genome_id(self) -> str:
        genome_id = f"pb-{self._next_genome_number:08d}"
        self._next_genome_number += 1
        return genome_id

    def _seed_genome(self, component: str, task_text: str) -> BreederGenome:
        genome = BreederGenome(
            genome_id=self._new_genome_id(),
            component=component,
            mutation_prompt=self._rng.choice(self.config.mutation_prompts),
            thinking_style=self._rng.choice(self.config.thinking_styles),
            generation=0,
            parent_genome_ids=(),
            operator="seed",
            task_lineage=(task_text,),
        )
        self._genomes[genome.genome_id] = genome
        return genome

    def _ensure_candidate_genomes(self, candidate: Mapping[str, str]) -> dict[str, str]:
        key = self._candidate_key(candidate)
        mapping = self._candidate_genomes.get(key)
        if mapping is None:
            mapping = {}
            for component, text in candidate.items():
                mapping[component] = self._seed_genome(component, text).genome_id
            self._candidate_genomes[key] = mapping
        else:
            for component, text in candidate.items():
                if component not in mapping or mapping[component] not in self._genomes:
                    mapping[component] = self._seed_genome(component, text).genome_id
        return dict(mapping)

    def _choose_operator(self) -> PromptBreederOperator:
        operators: list[PromptBreederOperator] = []
        base_weights: list[float] = []
        adaptive_weights: list[float] = []
        for operator in ("zero_order", "hypermutation", "lineage"):
            base = float(self.config.operator_weights.get(operator, 0.0))
            if base <= 0:
                continue
            stats = self._operator_stats[operator]
            attempts = stats["attempts"]
            mean_reward = stats["reward"] / attempts if attempts else 0.0
            success_rate = stats["accepts"] / attempts if attempts else 0.0
            signal = max(-2.0, min(2.0, mean_reward + 0.05 * success_rate))
            adaptive = base * math.exp(self.config.adaptation_strength * signal)
            operators.append(operator)
            base_weights.append(base)
            adaptive_weights.append(adaptive)

        weights = base_weights if self._rng.random() < self.config.exploration_rate else adaptive_weights
        return self._rng.choices(operators, weights=weights, k=1)[0]

    def _elite_mate(self, component: str, excluded_id: str) -> BreederGenome | None:
        candidates = [
            genome
            for genome in self._genomes.values()
            if genome.component == component and genome.genome_id != excluded_id
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda genome: genome.fitness, reverse=True)
        pool = candidates[: self.config.elite_pool_size]
        return self._rng.choice(pool)

    def _feedback_text(self, feedback: Sequence[Mapping[str, Any]]) -> str:
        text = json.dumps(list(feedback), default=str, ensure_ascii=False)
        return text[: self.config.max_meta_feedback_chars]

    @staticmethod
    def _extract_task_prompt(raw_output: str) -> str:
        return InstructionProposalSignature.output_extractor(raw_output.strip())["new_instruction"]

    @staticmethod
    def _extract_evolved_genes(
        raw_output: str,
        fallback_mutation_prompt: str,
        fallback_thinking_style: str,
    ) -> tuple[str, str]:
        mutation_match = re.search(
            r"<mutation_prompt>\s*(.*?)\s*</mutation_prompt>", raw_output, flags=re.IGNORECASE | re.DOTALL
        )
        style_match = re.search(
            r"<thinking_style>\s*(.*?)\s*</thinking_style>", raw_output, flags=re.IGNORECASE | re.DOTALL
        )
        mutation_prompt = mutation_match.group(1).strip() if mutation_match else fallback_mutation_prompt
        thinking_style = style_match.group(1).strip() if style_match else fallback_thinking_style
        return mutation_prompt or fallback_mutation_prompt, thinking_style or fallback_thinking_style

    def _hypermutate_genome(
        self,
        parent: BreederGenome,
        feedback: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str, str, str]:
        prompt = f"""You are evolving the reflection process used to improve task prompts.

Parent mutation prompt:
<parent_mutation_prompt>
{parent.mutation_prompt}
</parent_mutation_prompt>

Parent thinking style:
<parent_thinking_style>
{parent.thinking_style}
</parent_thinking_style>

Recent task-evaluation evidence:
<feedback>
{self._feedback_text(feedback)}
</feedback>

Create a more effective, general mutation prompt and thinking style. The mutation prompt must instruct another
model how to revise a task prompt; it must not solve the task itself. Preserve useful behavior while changing weak
assumptions or search strategy.

Return exactly:
<mutation_prompt>...</mutation_prompt>
<thinking_style>...</thinking_style>
"""
        raw = self.lm(prompt)
        mutation_prompt, thinking_style = self._extract_evolved_genes(
            raw, parent.mutation_prompt, parent.thinking_style
        )
        return mutation_prompt, thinking_style, prompt, raw

    def _crossover_genomes(
        self,
        parent: BreederGenome,
        mate: BreederGenome,
        feedback: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str, str, str]:
        prompt = f"""Recombine two successful reflection-process genomes into one descendant.

Genome A mutation prompt:
{parent.mutation_prompt}
Genome A thinking style:
{parent.thinking_style}
Genome A fitness: {parent.fitness:.6f}

Genome B mutation prompt:
{mate.mutation_prompt}
Genome B thinking style:
{mate.thinking_style}
Genome B fitness: {mate.fitness:.6f}

Recent task-evaluation evidence:
{self._feedback_text(feedback)}

Synthesize compatible strengths, discard redundant or conflicting rules, and produce a general reflection process
rather than a task answer.

Return exactly:
<mutation_prompt>...</mutation_prompt>
<thinking_style>...</thinking_style>
"""
        raw = self.lm(prompt)
        mutation_prompt, thinking_style = self._extract_evolved_genes(
            raw, parent.mutation_prompt, parent.thinking_style
        )
        return mutation_prompt, thinking_style, prompt, raw

    def _resolve_base_template(self, component: str) -> str:
        if isinstance(self.reflection_prompt_template, dict):
            template = self.reflection_prompt_template.get(component)
        else:
            template = self.reflection_prompt_template
        if template is not None:
            InstructionProposalSignature.validate_prompt_template(template)
            return template
        return """Current task instruction:
```
<curr_param>
```

Evaluation traces and feedback:
```
<side_info>
```"""

    def _render_task_mutation_prompt(
        self,
        *,
        genome: BreederGenome,
        current_text: str,
        feedback: Sequence[Mapping[str, Any]],
        operator: PromptBreederOperator,
        component: str,
    ) -> str | list[dict[str, Any]]:
        lineage = "\n\n".join(
            f"Ancestor {idx + 1}:\n```\n{text}\n```"
            for idx, text in enumerate(genome.task_lineage[-self.config.lineage_size :])
        )
        template = f"""You are the reflection model in a PromptBreeder + GEPA optimizer.

Your inherited reflection-process genome is:

Mutation prompt:
```
{genome.mutation_prompt}
```

Thinking style:
```
{genome.thinking_style}
```

Mutation operator: {operator}
Reflection generation: {genome.generation}

Task-prompt lineage:
{lineage}

{self._resolve_base_template(component)}

Apply the inherited mutation prompt while following the thinking style. Analyze failures internally, preserve
demonstrated strengths, and avoid overfitting to individual examples. Return only the complete replacement task
instruction inside triple backticks.
"""
        return InstructionProposalSignature.prompt_renderer(
            {
                "current_instruction_doc": current_text,
                "dataset_with_feedback": feedback,
                "prompt_template": template,
            }
        )

    def _trial_genome(
        self,
        *,
        parent: BreederGenome,
        current_text: str,
        feedback: Sequence[Mapping[str, Any]],
        operator: PromptBreederOperator,
    ) -> tuple[BreederGenome, dict[str, Any]]:
        mutation_prompt = parent.mutation_prompt
        thinking_style = parent.thinking_style
        parent_ids: tuple[str, ...] = (parent.genome_id,)
        meta: dict[str, Any] = {}

        if operator == "hypermutation":
            mutation_prompt, thinking_style, meta_prompt, meta_raw = self._hypermutate_genome(parent, feedback)
            meta["meta_prompt"] = meta_prompt
            meta["meta_raw_output"] = meta_raw
        elif operator == "lineage":
            mate = self._elite_mate(parent.component, parent.genome_id)
            if mate is None:
                mutation_prompt, thinking_style, meta_prompt, meta_raw = self._hypermutate_genome(parent, feedback)
                meta["lineage_fallback"] = "hypermutation"
            else:
                mutation_prompt, thinking_style, meta_prompt, meta_raw = self._crossover_genomes(parent, mate, feedback)
                parent_ids = (parent.genome_id, mate.genome_id)
                meta["mate_genome_id"] = mate.genome_id
            meta["meta_prompt"] = meta_prompt
            meta["meta_raw_output"] = meta_raw

        lineage = tuple((list(parent.task_lineage) + [current_text])[-self.config.lineage_size :])
        genome = BreederGenome(
            genome_id=self._new_genome_id(),
            component=parent.component,
            mutation_prompt=mutation_prompt,
            thinking_style=thinking_style,
            generation=max(
                [self._genomes[parent_id].generation for parent_id in parent_ids if parent_id in self._genomes],
                default=parent.generation,
            )
            + 1,
            parent_genome_ids=parent_ids,
            operator=operator,
            task_lineage=lineage,
        )
        return genome, meta

    def _reflect_in_place(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> ReflectionProposal:
        candidate_key = self._candidate_key(candidate)
        mapping = self._ensure_candidate_genomes(candidate)
        proposal = ReflectionProposal(new_texts={}, prompts={}, raw_lm_outputs={}, metadata={})
        trial_genomes: dict[str, dict[str, Any]] = {}
        parent_genome_ids: dict[str, str] = {}
        operators: dict[str, str] = {}
        meta_prompts: dict[str, str] = {}
        meta_outputs: dict[str, str] = {}

        for component in components_to_update:
            feedback = reflective_dataset.get(component)
            if not feedback:
                self._log(f"Component '{component}' is not in reflective dataset. Skipping.")
                continue
            parent_genome = self._genomes[mapping[component]]
            operator = self._choose_operator()
            trial, meta = self._trial_genome(
                parent=parent_genome,
                current_text=candidate[component],
                feedback=feedback,
                operator=operator,
            )
            task_prompt = self._render_task_mutation_prompt(
                genome=trial,
                current_text=candidate[component],
                feedback=feedback,
                operator=operator,
                component=component,
            )
            raw_output = self.lm(task_prompt)
            new_text = self._extract_task_prompt(raw_output)

            parent_genome_ids[component] = parent_genome.genome_id
            operators[component] = operator
            trial_genomes[component] = trial.to_dict()
            proposal.prompts[component] = task_prompt
            proposal.raw_lm_outputs[component] = raw_output
            if "meta_prompt" in meta:
                meta_prompts[component] = str(meta["meta_prompt"])
                meta_outputs[component] = str(meta["meta_raw_output"])
            if "mate_genome_id" in meta:
                trial_genomes[component]["mate_genome_id"] = meta["mate_genome_id"]
            if "lineage_fallback" in meta:
                trial_genomes[component]["lineage_fallback"] = meta["lineage_fallback"]

            if new_text and new_text != candidate[component]:
                proposal.new_texts[component] = new_text
            else:
                self._log(f"PromptBreeder produced no task-prompt change for '{component}'.")

        proposal.metadata.update(
            {
                "promptbreeder:parent_candidate_key": candidate_key,
                "promptbreeder:parent_genome_ids": parent_genome_ids,
                "promptbreeder:operators": operators,
                "promptbreeder:trial_genomes": trial_genomes,
            }
        )
        if meta_prompts:
            proposal.metadata["promptbreeder:meta_prompts"] = meta_prompts
            proposal.metadata["promptbreeder:meta_outputs"] = meta_outputs
        return proposal

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[ReflectionProposal, PromptBreederReflectionLM]:
        successor = self._fork()
        proposal = successor._reflect_in_place(candidate, reflective_dataset, components_to_update)
        return proposal, successor

    def reflect_many(self, jobs: list[ReflectionJob]) -> list[tuple[ReflectionProposal, PromptBreederReflectionLM]]:
        results: list[tuple[ReflectionProposal, PromptBreederReflectionLM]] = []
        current: PromptBreederReflectionLM = self
        for candidate, reflective_dataset, components_to_update in jobs:
            proposal, current = current.reflect(candidate, reflective_dataset, components_to_update)
            results.append((proposal, current))
        return results

    @staticmethod
    def _subsample_reward(proposal: CandidateProposal) -> float:
        before = proposal.subsample_scores_before or []
        after = proposal.subsample_scores_after or []
        if not before or not after:
            return 0.0
        return sum(after) / len(after) - sum(before) / len(before)

    @staticmethod
    def _full_reward(proposal: CandidateProposal, new_candidate_idx: int, state: Any) -> float:
        try:
            child_score = state.get_program_average_val_subset(new_candidate_idx)[0]
            parent_scores = [state.get_program_average_val_subset(idx)[0] for idx in proposal.parent_program_ids]
            finite_parent_scores = [score for score in parent_scores if math.isfinite(score)]
            if math.isfinite(child_score) and finite_parent_scores:
                return child_score - max(finite_parent_scores)
        except (AttributeError, IndexError, TypeError):
            pass
        return PromptBreederReflectionLM._subsample_reward(proposal)

    def _record_operator_result(self, operator: str, reward: float, accepted: bool) -> None:
        stats = self._operator_stats.setdefault(
            operator, {"attempts": 0.0, "accepts": 0.0, "rejects": 0.0, "reward": 0.0}
        )
        stats["attempts"] += 1.0
        stats["reward"] += reward
        if accepted:
            stats["accepts"] += 1.0
        else:
            stats["rejects"] += 1.0

    def _credit_parent_genome(self, genome_id: str, reward: float, accepted: bool) -> None:
        parent = self._genomes.get(genome_id)
        if parent is None:
            return
        parent.offspring_attempts += 1
        parent.offspring_reward += reward
        if accepted:
            parent.offspring_accepts += 1
        else:
            parent.offspring_rejects += 1

    def _append_history(self, event: dict[str, Any]) -> None:
        self._history.append(event)
        if len(self._history) > self.config.history_limit:
            del self._history[: len(self._history) - self.config.history_limit]

    def on_proposal_accepted(
        self,
        proposal: CandidateProposal,
        new_candidate_idx: int,
        state: Any,
        valset_evaluation: Any | None = None,
    ) -> None:
        del valset_evaluation
        metadata = proposal.metadata or {}
        raw_trials = metadata.get("promptbreeder:trial_genomes", {})
        if not raw_trials:
            return
        reward = self._full_reward(proposal, new_candidate_idx, state)
        metadata["promptbreeder:status"] = "accepted"
        metadata["promptbreeder:reward"] = reward
        parent_key = str(metadata.get("promptbreeder:parent_candidate_key", ""))
        parent_mapping = dict(self._candidate_genomes.get(parent_key, {}))
        child_mapping = dict(parent_mapping)

        accepted_genome_ids: dict[str, str] = {}
        for component, raw_trial in dict(raw_trials).items():
            trial = BreederGenome.from_dict(raw_trial)
            trial.creation_reward = reward
            child_text = proposal.candidate.get(component)
            if child_text:
                trial.task_lineage = tuple((list(trial.task_lineage) + [child_text])[-self.config.lineage_size :])
            self._genomes[trial.genome_id] = trial
            child_mapping[str(component)] = trial.genome_id
            accepted_genome_ids[str(component)] = trial.genome_id
            operator = str((metadata.get("promptbreeder:operators", {}) or {}).get(component, trial.operator))
            self._record_operator_result(operator, reward, accepted=True)
            for parent_id in trial.parent_genome_ids:
                self._credit_parent_genome(parent_id, reward, accepted=True)

        child_key = self._candidate_key(proposal.candidate)
        self._candidate_genomes[child_key] = child_mapping
        self._candidate_index_genomes[new_candidate_idx] = dict(child_mapping)
        self._append_history(
            {
                "status": "accepted",
                "candidate_idx": new_candidate_idx,
                "candidate_key": child_key,
                "reward": reward,
                "genomes": accepted_genome_ids,
            }
        )

    def on_proposal_rejected(
        self,
        proposal: CandidateProposal,
        state: Any,
        reason: str | None = None,
    ) -> None:
        del state
        metadata = proposal.metadata or {}
        raw_trials = metadata.get("promptbreeder:trial_genomes", {})
        if not raw_trials:
            return
        reward = self._subsample_reward(proposal)
        metadata["promptbreeder:status"] = "rejected"
        metadata["promptbreeder:reward"] = reward
        for component, raw_trial in dict(raw_trials).items():
            trial = BreederGenome.from_dict(raw_trial)
            operator = str((metadata.get("promptbreeder:operators", {}) or {}).get(component, trial.operator))
            self._record_operator_result(operator, reward, accepted=False)
            for parent_id in trial.parent_genome_ids:
                self._credit_parent_genome(parent_id, reward, accepted=False)
        self._append_history(
            {
                "status": "rejected",
                "reward": reward,
                "reason": reason or "rejected",
                "trial_genome_ids": [str(raw["genome_id"]) for raw in raw_trials.values()],
            }
        )

    def on_candidate_imported(
        self,
        candidate: Mapping[str, str],
        parent_candidates: Sequence[Mapping[str, str]],
        *,
        new_candidate_idx: int | None = None,
        state: Any | None = None,
    ) -> None:
        del state
        child_mapping: dict[str, str] = {}
        parent_mappings = [self._candidate_genomes.get(self._candidate_key(parent), {}) for parent in parent_candidates]
        for component, text in candidate.items():
            candidate_genomes = [
                self._genomes[mapping[component]]
                for mapping in parent_mappings
                if component in mapping and mapping[component] in self._genomes
            ]
            if candidate_genomes:
                winner = max(candidate_genomes, key=lambda genome: genome.fitness)
                child_mapping[component] = winner.genome_id
            else:
                child_mapping[component] = self._seed_genome(component, text).genome_id
        candidate_key = self._candidate_key(candidate)
        self._candidate_genomes[candidate_key] = child_mapping
        if new_candidate_idx is not None:
            self._candidate_index_genomes[new_candidate_idx] = dict(child_mapping)

    def diagnostics(self) -> dict[str, Any]:
        elites = sorted(self._genomes.values(), key=lambda genome: genome.fitness, reverse=True)
        return {
            "num_genomes": len(self._genomes),
            "num_candidate_assignments": len(self._candidate_genomes),
            "num_candidate_index_assignments": len(self._candidate_index_genomes),
            "operator_stats": copy.deepcopy(self._operator_stats),
            "elite_genomes": [genome.to_dict() | {"fitness": genome.fitness} for genome in elites[:10]],
            "recent_history": copy.deepcopy(self._history[-20:]),
        }


def make_prompt_breeder_strategy(
    reflection_lm: LanguageModel | str,
    *,
    config: PromptBreederConfig | None = None,
    logger: Any | None = None,
    reflection_prompt_template: str | dict[str, str] | None = None,
    reflection_lm_kwargs: dict[str, Any] | None = None,
) -> PromptBreederReflectionLM:
    """Build a co-evolving PromptBreeder strategy for GEPA."""

    lm: LanguageModel
    if isinstance(reflection_lm, str):
        from gepa.lm import LM

        lm = LM(reflection_lm, **(reflection_lm_kwargs or {}))
    elif hasattr(reflection_lm, "total_cost"):
        lm = reflection_lm
    else:
        from gepa.lm import TrackingLM

        lm = TrackingLM(reflection_lm)

    return PromptBreederReflectionLM(
        lm=lm,
        config=config,
        logger=logger,
        reflection_prompt_template=reflection_prompt_template,
    )
