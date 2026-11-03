# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Positional order of the public entry points must stay compatible with v0.1.4.

New parameters and fields go after the existing ones and are keyword-only.
"""

import dataclasses
import inspect

import gepa
from gepa.gepa_launcher import EngineConfig

V014_OPTIMIZE_POSITIONAL = [
    "seed_candidate", "trainset", "valset", "adapter", "task_lm", "evaluator", "reflection_lm",
    "reflection_lm_kwargs", "candidate_selection_strategy", "frontier_type", "skip_perfect_score",
    "batch_sampler", "reflection_minibatch_size", "perfect_score", "reflection_prompt_template",
    "custom_candidate_proposer", "module_selector", "use_merge", "max_merge_invocations",
    "merge_val_overlap_floor", "max_metric_calls", "max_reflection_cost", "stop_callbacks", "logger",
    "run_dir", "callbacks", "use_wandb", "wandb_api_key", "wandb_init_kwargs", "wandb_attach_existing",
    "use_mlflow", "mlflow_tracking_uri", "mlflow_experiment_name", "mlflow_attach_existing",
    "tracking_key_prefix", "track_best_outputs", "display_progress_bar", "use_cloudpickle",
    "cache_evaluation", "seed", "raise_on_exception", "val_evaluation_policy", "acceptance_criterion",
    "sampling_strategy", "selection_strategy", "reflection_strategy",
]  # fmt: skip

V014_ENGINE_CONFIG_POSITIONAL = [
    "run_dir", "seed", "display_progress_bar", "raise_on_exception", "use_cloudpickle", "track_best_outputs",
    "max_metric_calls", "max_candidate_proposals", "max_reflection_cost", "val_evaluation_policy",
    "candidate_selection_strategy", "frontier_type", "acceptance_criterion", "parallel", "max_workers",
    "cache_evaluation", "cache_evaluation_storage", "best_example_evals_k", "capture_stdio",
    "sampling_strategy", "selection_strategy",
]  # fmt: skip


def test_optimize_keeps_the_v014_positional_order():
    positional = [
        p.name
        for p in inspect.signature(gepa.optimize).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == V014_OPTIMIZE_POSITIONAL


def test_engine_config_keeps_the_v014_positional_order():
    positional = [f.name for f in dataclasses.fields(EngineConfig) if f.init and not f.kw_only]
    assert positional == V014_ENGINE_CONFIG_POSITIONAL
