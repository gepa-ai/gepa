# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


"""The launcher must not write run state into the caller's ``GEPAConfig``."""

import dataclasses

from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything


def _snapshot(config):
    """Every field of the config and its nested configs, by object identity."""
    snap = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        snap[f.name] = value
        if dataclasses.is_dataclass(value):
            snap.update({f"{f.name}.{g.name}": getattr(value, g.name) for g in dataclasses.fields(value)})
    return snap


def _run(config):
    calls = []

    def evaluator(candidate, example):
        calls.append((candidate, example))
        return (1.0 if "improved" in candidate else 0.2), {"note": f"scored {example}"}

    data = [f"ex{i}" for i in range(4)]
    result = optimize_anything(
        seed_candidate="seed", evaluator=evaluator, dataset=data, valset=data, objective="say improved", config=config
    )
    return calls, result


def test_reusing_a_config_repeats_the_run_and_leaves_it_untouched():
    """Reusing one config (with ``objective``, which builds a reflection template,
    and the default batch sampler) gives the same run twice, and no field of
    the config or its nested configs is replaced."""

    def reflection_lm(prompt):
        return "```\nimproved candidate\n```"

    config = GEPAConfig(
        engine=EngineConfig(max_metric_calls=12, parallel=False, seed=0),
        reflection=ReflectionConfig(reflection_lm=reflection_lm),
    )
    before = _snapshot(config)

    first_calls, first = _run(config)
    after = _snapshot(config)
    assert [k for k, v in before.items() if after[k] is not v] == []
    second_calls, second = _run(config)

    assert first_calls == second_calls
    assert first.candidates == second.candidates
    assert first.val_aggregate_scores == second.val_aggregate_scores
    assert config.reflection.reflection_lm is reflection_lm
    assert config.reflection.batch_sampler == "epoch_shuffled"
    assert config.tracking.logger is None
