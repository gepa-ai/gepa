# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""``gepa.optimize`` accepts positional arguments, so new parameters must not shift old ones.

``reflective_dataset_enricher`` was first inserted just before ``module_selector``. A
caller passing ``"all"`` positionally for ``module_selector`` then handed the string to
the enricher; every proposal raised, was skipped, and the run spent its budget without
ever calling reflection.
"""

import inspect

import gepa
from gepa.core.adapter import EvaluationBatch

# The positional order of gepa.optimize, up to module_selector, before the enricher existed.
PRE_ENRICHER_ORDER = (
    "seed_candidate",
    "trainset",
    "valset",
    "adapter",
    "task_lm",
    "evaluator",
    "reflection_lm",
    "reflection_lm_kwargs",
    "candidate_selection_strategy",
    "frontier_type",
    "skip_perfect_score",
    "batch_sampler",
    "reflection_minibatch_size",
    "perfect_score",
    "reflection_prompt_template",
    "custom_candidate_proposer",
    "module_selector",
)


def test_enricher_is_keyword_only():
    parameter = inspect.signature(gepa.optimize).parameters["reflective_dataset_enricher"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_existing_positional_parameters_keep_their_positions():
    names = tuple(inspect.signature(gepa.optimize).parameters)
    assert names[: len(PRE_ENRICHER_ORDER)] == PRE_ENRICHER_ORDER


class _TwoComponentAdapter:
    propose_new_texts = None

    def evaluate(self, batch, candidate, capture_traces=False):
        return EvaluationBatch(
            outputs=["out"] * len(batch),
            scores=[0.5] * len(batch),
            trajectories=[{"i": i} for i in range(len(batch))] if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {c: [{"Inputs": "i", "Generated Outputs": "o", "Feedback": "f"}] for c in components_to_update}


def test_positional_module_selector_still_reaches_reflection():
    reflection_calls = []

    def reflect(prompt):
        reflection_calls.append(prompt)
        return "```\nrevised instruction\n```"

    values = {
        "seed_candidate": {"a": "seed a", "b": "seed b"},
        "trainset": [{"q": f"q{i}"} for i in range(4)],
        "valset": [{"q": f"v{i}"} for i in range(4)],
        "adapter": _TwoComponentAdapter(),
        "reflection_lm": reflect,
        "reflection_minibatch_size": 2,
        "module_selector": "all",
    }
    defaults = {name: p.default for name, p in inspect.signature(gepa.optimize).parameters.items()}
    positional = [values.get(name, defaults[name]) for name in PRE_ENRICHER_ORDER]

    gepa.optimize(*positional, max_metric_calls=40, display_progress_bar=False)

    assert reflection_calls, "a positional module_selector must not silence reflection"
