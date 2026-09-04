# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests that the reflective dataset enricher reaches the optimize_anything path.

``gepa.optimize`` threads the enricher into the reflective proposer directly.
``optimize_anything`` builds the same proposer from a ``GEPAConfig``, so the
hook has to travel as a ``ReflectionConfig`` field or it is silently
unavailable to every pipeline built on that entry point.
"""

import io
from contextlib import redirect_stderr, redirect_stdout

from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from gepa.oa.config import OptimizeAnythingConfig

FINDING = "Spurious_Fact_Introduced"


def _enricher(record):
    """Adds one field to every reflective record, and counts its own calls."""

    def enrich(*, candidate, eval_batch, components_to_update, reflective_dataset):
        record["calls"] += 1
        record["components"].append(sorted(reflective_dataset))
        return {
            component: [{**row, "failure_modes": [{"name": FINDING, "evidence": "e"}]} for row in rows]
            for component, rows in reflective_dataset.items()
        }

    return enrich


def _reflection_lm(prompts):
    def reflect(prompt):
        prompts.append(prompt)
        return "```\nrevised instruction\n```"

    return reflect


def _evaluate(candidate, example=None, **kwargs):
    return 0.5, {"Feedback": "too vague"}


def _config(record, prompts, enricher=True):
    return GEPAConfig(
        engine=EngineConfig(max_metric_calls=20),
        reflection=ReflectionConfig(
            reflection_lm=_reflection_lm(prompts),
            reflective_dataset_enricher=_enricher(record) if enricher else None,
        ),
    )


def _run(config):
    dataset = [{"q": f"q{i}"} for i in range(4)]
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        optimize_anything(
            seed_candidate="write a haiku",
            evaluator=_evaluate,
            dataset=dataset,
            valset=dataset,
            config=config,
        )


class TestReflectionConfigField:
    def test_enricher_is_invoked_and_reaches_the_reflection_prompt(self):
        record, prompts = {"calls": 0, "components": []}, []
        _run(_config(record, prompts))

        assert record["calls"] > 0, "the enricher never ran"
        assert prompts, "reflection was never called"
        assert all(FINDING in prompt for prompt in prompts), "the finding did not reach the reflection prompt"

    def test_enricher_sees_the_components_under_update(self):
        record, prompts = {"calls": 0, "components": []}, []
        _run(_config(record, prompts))

        assert all(components for components in record["components"])

    def test_without_the_field_nothing_is_injected(self):
        record, prompts = {"calls": 0, "components": []}, []
        _run(_config(record, prompts, enricher=False))

        assert record["calls"] == 0
        assert prompts
        assert not any(FINDING in prompt for prompt in prompts)

    def test_default_is_none(self):
        assert ReflectionConfig().reflective_dataset_enricher is None


class TestOptimizeAnythingEngine:
    """The OA layer forwards engine_config verbatim, so the field arrives with it."""

    def test_enricher_survives_the_engine_config_round_trip(self):
        from gepa.optimize_anything import optimize_anything as oa_optimize_anything

        record, prompts = {"calls": 0, "components": []}, []
        config = OptimizeAnythingConfig(
            engine="gepa",
            engine_config={
                "engine": EngineConfig(max_metric_calls=20),
                "reflection": ReflectionConfig(
                    reflection_lm=_reflection_lm(prompts),
                    reflective_dataset_enricher=_enricher(record),
                ),
            },
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            oa_optimize_anything(
                "write a haiku", evaluator=lambda candidate: (0.5, {"Feedback": "too vague"}), config=config
            )

        assert record["calls"] > 0
        assert prompts and all(FINDING in prompt for prompt in prompts)
