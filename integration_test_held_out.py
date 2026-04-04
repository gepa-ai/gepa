"""Smoke-test full_eval vs heldout_eval on a small AIME slice.

Configuration is read from environment variables and optional ``.env`` files.

Example:

    uv run python integration_test_held_out.py
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from dotenv import load_dotenv

import gepa
import gepa.examples.aime
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
from gepa.lm import LM

load_dotenv()
load_dotenv(".env.integration", override=False)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_model(model: str) -> str:
    known_prefixes = (
        "openai/",
        "anthropic/",
        "google/",
        "vertex_ai/",
        "bedrock/",
        "groq/",
        "ollama/",
        "together_ai/",
    )
    return model if model.startswith(known_prefixes) else f"together_ai/{model}"


RAW_MODEL = os.getenv("GEPA_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
MODEL = _normalize_model(RAW_MODEL)
MAX_METRIC_CALLS = _env_int("GEPA_MAX_METRIC_CALLS", 50)
TRAIN_SLICE = _env_int("GEPA_TRAIN_SLICE", 6)
VAL_SLICE = _env_int("GEPA_VAL_SLICE", 6)
HELD_OUT_SLICE = _env_int("GEPA_HELD_OUT_SLICE", 4)
TEST_SLICE = _env_int("GEPA_TEST_SLICE", 6)
RUN_MODES = tuple(
    mode.strip() for mode in os.getenv("GEPA_RUN_MODES", "full_eval,heldout_eval").split(",") if mode.strip()
)
USE_WANDB = _env_bool("GEPA_USE_WANDB", False)
WANDB_PROJECT = os.getenv("GEPA_WANDB_PROJECT", "gepa-held-out-integration")
WANDB_PREFIX = os.getenv("GEPA_WANDB_RUN_PREFIX", "aime-qwen3-vl-8b-smoke")
MAX_TOKENS = _env_int("GEPA_MAX_TOKENS", 16384)

NO_REASONING = {"extra_body": {"reasoning": {"enabled": False}}, "max_tokens": MAX_TOKENS}


def _validate_modes(run_modes: Sequence[str]) -> None:
    allowed = {"full_eval", "heldout_eval"}
    unknown = [mode for mode in run_modes if mode not in allowed]
    if unknown:
        raise ValueError(f"Unknown GEPA_RUN_MODES entries: {unknown}. Allowed: {sorted(allowed)}")


def _build_adapter() -> tuple[DefaultAdapter, LM]:
    adapter = DefaultAdapter(model=MODEL, litellm_batch_completion_kwargs=NO_REASONING)
    adapter._lm.completion_kwargs["max_tokens"] = MAX_TOKENS
    reflection_lm = LM(MODEL, max_tokens=MAX_TOKENS, extra_body={"reasoning": {"enabled": False}})
    return adapter, reflection_lm


def _evaluate_testset(adapter: DefaultAdapter, candidate: dict[str, str], test_eval: list[dict]) -> tuple[float, int]:
    evaluation = adapter.evaluate(test_eval, candidate, capture_traces=False)
    correct = int(sum(evaluation.scores))
    return correct / len(evaluation.scores), correct


def main() -> None:
    _validate_modes(RUN_MODES)

    adapter, reflection_lm = _build_adapter()

    trainset, valset, testset = gepa.examples.aime.init_dataset()

    mid = len(valset) // 2
    pareto_val = valset[:mid]
    held_out = valset[mid:]

    train = trainset[:TRAIN_SLICE]
    val = pareto_val[:VAL_SLICE]
    held = held_out[:HELD_OUT_SLICE]
    test = testset[:TEST_SLICE]

    print(
        f"Model: {MODEL}\n"
        f"Modes: {', '.join(RUN_MODES)}\n"
        f"Budget per run: {MAX_METRIC_CALLS}\n"
        f"Splits — train: {len(train)}, val: {len(val)}, held-out: {len(held)}, test: {len(test)}"
    )

    seed = {
        "system_prompt": (
            "You are a helpful math assistant. Solve the problem step by step. "
            "Give your final integer answer at the end in the format '### <integer>'."
        )
    }
    test_eval = [{**ex, "additional_context": ex.get("additional_context", {})} for ex in test]

    for mode in RUN_MODES:
        print(f"\n=== Running {mode} ===")
        optimize_kwargs = {
            "seed_candidate": seed,
            "trainset": train,
            "valset": val,
            "adapter": adapter,
            "reflection_lm": reflection_lm,
            "max_metric_calls": MAX_METRIC_CALLS,
            "display_progress_bar": True,
            "use_wandb": USE_WANDB,
        }
        if USE_WANDB:
            optimize_kwargs["wandb_init_kwargs"] = {
                "project": WANDB_PROJECT,
                "name": f"{WANDB_PREFIX}-{mode}-budget{MAX_METRIC_CALLS}",
            }

        if mode == "heldout_eval":
            optimize_kwargs["held_out"] = held
            optimize_kwargs["val_evaluation_policy"] = "heldout_eval"
        else:
            optimize_kwargs["val_evaluation_policy"] = "full_eval"

        result = gepa.optimize(**optimize_kwargs)

        print("\n-- Result summary --")
        print(f"Candidates explored: {result.num_candidates}")
        print(f"Best idx:            {result.best_idx}")
        print(f"Valset best idx:     {result.valset_best_idx}")
        print(f"Best valset score:   {result.val_aggregate_scores[result.valset_best_idx]:.3f}")
        if result.held_out_scores is not None:
            print(f"Held-out scores:     {result.held_out_scores}")
            print(f"Held-out evals:      {result.num_held_out_evals}  (not counted in budget)")
        print(f"Total metric calls:  {result.total_metric_calls}")

        best_acc, best_correct = _evaluate_testset(adapter, result.best_candidate, test_eval)
        seed_acc, seed_correct = _evaluate_testset(adapter, seed, test_eval)
        valset_best = result.candidates[result.valset_best_idx]
        valset_acc, valset_correct = _evaluate_testset(adapter, valset_best, test_eval)

        print("\n-- Test set evaluation --")
        print(f"Seed accuracy:         {seed_acc:.3f}  ({seed_correct}/{len(test_eval)})")
        print(f"Policy best accuracy:  {best_acc:.3f}  ({best_correct}/{len(test_eval)})")
        print(f"Valset best accuracy:  {valset_acc:.3f}  ({valset_correct}/{len(test_eval)})")
        print(f"Policy vs seed:        {best_acc - seed_acc:+.3f}")
        print(f"Valset vs seed:        {valset_acc - seed_acc:+.3f}")


if __name__ == "__main__":
    main()
