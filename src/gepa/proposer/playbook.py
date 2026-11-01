# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Incremental proposals that preserve existing candidate text."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel


class PlaybookProposer:
    """Append new rule bullets through the ``custom_candidate_proposer`` hook.

    The LM proposes additions, never replacement text. This class appends them
    locally, preserving each selected component's existing text verbatim.

    Args:
        reflection_lm: Callable that accepts a prompt and returns a JSON object
            mapping selected component names to lists of new rule strings.
            Pass the callable here; the custom proposer hook does not receive
            the optimizer's ``reflection_lm`` argument.
        max_additions_per_step: Positive maximum number of proposed additions
            across all components in one call. Over-limit responses are rejected.
        avoid_redundancy: Ask the LM to avoid near-duplicate rules and skip exact
            duplicates of existing lines/bullets or other additions locally.
            Semantic deduplication is guidance to the LM, not a guarantee.

    Invalid JSON, unselected component names, and empty or non-string additions
    raise ``ValueError``. Missing component keys or empty lists mean no change.
    Components without reflective examples are skipped without an LM call.
    """

    def __init__(
        self,
        reflection_lm: LanguageModel,
        *,
        max_additions_per_step: int = 3,
        avoid_redundancy: bool = True,
    ):
        if not isinstance(max_additions_per_step, int) or isinstance(max_additions_per_step, bool):
            raise ValueError("max_additions_per_step must be a positive integer")
        if max_additions_per_step < 1:
            raise ValueError("max_additions_per_step must be a positive integer")
        self.reflection_lm = reflection_lm
        self.max_additions_per_step = max_additions_per_step
        self.avoid_redundancy = avoid_redundancy

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        """Return selected component texts with validated additions appended."""
        components = list(dict.fromkeys(c for c in components_to_update if reflective_dataset.get(c)))
        if not components:
            return {}
        for component in components:
            if component not in candidate:
                raise ValueError(f"Unknown candidate component: {component!r}")

        context = {
            "candidate": candidate,
            "components_to_update": components,
            "reflective_dataset": {c: reflective_dataset[c] for c in components},
        }
        prompt = (
            "Improve this candidate using the evaluation feedback. "
            "Do not rewrite or remove existing content. Identify only what is missing. "
            f"Propose at most {self.max_additions_per_step} new rule bullets in total across all selected components. "
            "Return only a JSON object mapping selected component names to lists of addition strings, "
            "without Markdown fences or bullet prefixes. Use an empty list when no addition is useful. "
        )
        if self.avoid_redundancy:
            prompt += "Avoid rules that duplicate or paraphrase existing content or another proposed addition. "
        prompt += "\n\n" + json.dumps(context, ensure_ascii=False, default=str)
        additions = json.loads(self.reflection_lm(prompt))
        if not isinstance(additions, dict) or any(c not in components for c in additions):
            raise ValueError("Expected a JSON object containing only selected component names")
        for rules in additions.values():
            if not isinstance(rules, list) or any(not isinstance(rule, str) or not rule.strip() for rule in rules):
                raise ValueError("Each component must contain a list of non-empty addition strings")
        if sum(len(rules) for rules in additions.values()) > self.max_additions_per_step:
            raise ValueError("LM response exceeds max_additions_per_step")

        proposed = {}
        for component in components:
            original = candidate[component]
            seen = {line.strip().removeprefix("- ").strip() for line in original.splitlines()}
            bullets = []
            for rule in additions.get(component, []):
                rule = rule.strip()
                if self.avoid_redundancy and rule in seen:
                    continue
                bullets.append(f"- {rule}")
                seen.add(rule)
            if bullets:
                original += ("\n\n" if original else "") + "\n".join(bullets)
            proposed[component] = original
        return proposed
