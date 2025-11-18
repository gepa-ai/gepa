"""
Simplified AIME benchmark that compares original GEPA against TurboGEPA.

This script runs the classic `gepa.optimize` API and then our TurboGEPA
adapter on the exact same AIME subset so we can measure wall-clock speed,
evaluation counts, and quality side-by-side.

Usage:
    python examples/aime_benchmark_v2.py \
        --mode both \
        --task-lm openrouter/openai/gpt-oss-20b:nitro \
        --reflection-lm openrouter/x-ai/grok-4-fast

Environment:
    Set your OpenRouter/OpenAI credentials beforehand (e.g., `source .envrc`).
    By default, Hugging Face caches go to `.hf_cache` in the repo root; override
    with `--hf-cache` if needed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import time
from dataclasses import dataclass
import subprocess
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gepa
from turbo_gepa.adapters import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config, adaptive_config
from turbo_gepa.scoring import ScoringContext


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _configure_hf_cache(cache_root: Path | None) -> None:
    """Optionally point Hugging Face caches at a workspace-friendly directory."""

    if cache_root is None:
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    datasets_cache = cache_root / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))


def _load_aime_subset(limit: int) -> Tuple[List[dict], List[dict]]:
    """Load (train, val) splits from gepa.examples.aime with an optional cap."""

    trainset, valset, _ = gepa.examples.aime.init_dataset()
    if limit > 0:
        trainset = trainset[:limit]
        valset = valset[: min(limit, len(valset))]
    return trainset, valset


def _reset_turbo_cache() -> None:
    """Delete .turbo_gepa to guarantee clean-state benchmarks."""

    turbo_root = Path(".turbo_gepa")
    if turbo_root.exists():
        shutil.rmtree(turbo_root)
    turbo_root.mkdir(parents=True, exist_ok=True)
    print("🧹 Cleared .turbo_gepa cache for a fresh TurboGEPA run.")


def _summarize_prompt(prompt_text: str | None, max_chars: int = 160) -> str:
    if not prompt_text:
        return "<empty>"
    text = prompt_text.replace("\n", " ").strip()
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def _quality_reward(ctx: ScoringContext) -> float:
    """
    Simple reward combining accuracy and token efficiency.

    Treat quality as the primary objective, but gently penalize bloated prompts
    by subtracting 0.01 points for every additional 1k tokens consumed. This
    mirrors the style of custom reward functions users can plug into the adapter.
    """

    quality = float(ctx.result.objectives.get("quality", 0.0))
    tokens = float(ctx.result.objectives.get("tokens", 0.0))
    # Example alternative: include neg_tokens (e.g., cost savings). Uncomment to flip sign.
    # neg_tokens = float(ctx.result.objectives.get("neg_tokens", 0.0))
    # return 0.8 * quality + 0.2 * neg_tokens
    return quality - 0.01 * (tokens / 1000.0)


# --------------------------------------------------------------------------------------
# GEPA runner
# --------------------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    label: str
    elapsed: float
    evaluations: int
    quality: float
    prompt: str
    extra: Dict[str, Any]


def run_gepa(
    trainset: List[dict],
    valset: List[dict],
    *,
    task_lm: str,
    reflection_lm: str,
    seed_prompt: str,
    max_metric_calls: int,
) -> BenchmarkResult:
    """Run the legacy GEPA optimizer to establish a baseline."""

    print("\n=== Running GEPA (classic) ===")
    seed = {"system_prompt": seed_prompt}
    start = time.time()
    result = gepa.optimize(
        seed_candidate=seed,
        trainset=trainset,
        valset=valset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        display_progress_bar=True,
        raise_on_exception=False,
    )
    elapsed = time.time() - start

    quality = 0.0
    prompt = seed_prompt
    if getattr(result, "val_aggregate_scores", None):
        idx = getattr(result, "best_idx", 0)
        scores = result.val_aggregate_scores or []
        if 0 <= idx < len(scores):
            quality = float(scores[idx])
        best = result.best_candidate or {}
        prompt = best.get("system_prompt", prompt)

    evaluations = max_metric_calls
    print(
        f"GEPA completed in {elapsed:.1f}s | best quality {quality:.3f} "
        f"| evaluations {evaluations}"
    )
    return BenchmarkResult(
        label="GEPA",
        elapsed=elapsed,
        evaluations=evaluations,
        quality=quality,
        prompt=prompt,
        extra={"best_idx": getattr(result, "best_idx", None)},
    )


# --------------------------------------------------------------------------------------
# TurboGEPA runner
# --------------------------------------------------------------------------------------


def _build_turbo_config(dataset_size: int, args: argparse.Namespace) -> Config:
    config = adaptive_config(dataset_size)
    config.n_islands = max(1, args.turbo_n_islands)
    config.eval_concurrency = max(1, args.turbo_eval_concurrency)
    config.batch_size = max(1, min(dataset_size, config.eval_concurrency))
    config.max_mutations_per_round = max(1, args.turbo_max_mutations)
    config.queue_limit = max(4, args.turbo_queue_limit)
    config.log_level = args.turbo_log_level
    config.target_quality = args.turbo_target_quality
    config.target_shard_fraction = args.turbo_target_shard
    config.max_optimization_time_seconds = args.turbo_max_runtime
    config.auto_scale_eval_concurrency = args.turbo_auto_scale
    if getattr(args, "turbo_eval_timeout", None):
        try:
            config.eval_timeout_seconds = float(args.turbo_eval_timeout)
        except Exception:
            pass
    if getattr(args, "turbo_verification_speed_bias", None) is not None:
        config.verification_speed_bias = max(0.0, min(1.0, float(args.turbo_verification_speed_bias)))
    # Allow multiple full-shard candidates so slow OSS-20 calls overlap.
    final_cap = max(2, args.turbo_eval_concurrency // 2)
    config.max_final_shard_inflight = min(args.turbo_eval_concurrency, final_cap)
    if args.turbo_strategies:
        config.reflection_strategy_names = tuple(args.turbo_strategies)
    return config


def run_turbo(
    trainset: List[dict],
    *,
    task_lm: str,
    reflection_lm: str,
    seed_prompt: str,
    args: argparse.Namespace,
) -> BenchmarkResult:
    """Run TurboGEPA on the same dataset."""

    print("\n=== Running TurboGEPA ===")
    dataset = [
        DefaultDataInst(
            input=example["input"],
            answer=example["answer"],
            additional_context=example.get("additional_context"),
            id=f"aime_{idx}",
        )
        for idx, example in enumerate(trainset)
    ]
    config = _build_turbo_config(len(dataset), args)
    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        config=config,
        auto_config=False,
        scoring_fn=_quality_reward,
    )
    if getattr(args, "turbo_task_max_tokens", None):
        try:
            adapter.task_model.max_tokens = int(args.turbo_task_max_tokens)
        except Exception:
            pass

    start = time.time()
    try:
        result = adapter.optimize(
            seeds=[seed_prompt],
            max_rounds=args.turbo_max_rounds,
            max_evaluations=args.turbo_max_evaluations,
            display_progress=args.turbo_show_progress,
        )
    finally:
        asyncio.run(adapter.aclose())
    elapsed = time.time() - start

    # Prefer adapter-provided run_metadata if present (it captures the full‑shard
    # candidate that actually hit the target, even if not on the Pareto frontier).
    run_meta = result.get("run_metadata", {}) or {}
    quality = float(run_meta.get("best_quality") or 0.0)
    prompt = str(run_meta.get("best_prompt") or seed_prompt)
    shard_info = run_meta.get("best_quality_shard")
    if isinstance(shard_info, (int, float)):
        print(f"   Selected best prompt from shard {float(shard_info):.0%}")

    best_path = Path(".turbo_gepa") / "best_prompt.txt"
    try:
        best_path.write_text(prompt, encoding="utf-8")
    except Exception as exc:
        print(f"⚠️ Could not write best prompt to {best_path}: {exc}")

    stats = result.get("evolution_stats", {}) or {}
    evaluations = stats.get("total_evaluations", 0)

    print(
        f"TurboGEPA completed in {elapsed:.1f}s | best quality {quality:.3f} "
        f"| evaluations {evaluations}"
    )
    return BenchmarkResult(
        label="TurboGEPA",
        elapsed=elapsed,
        evaluations=evaluations,
        quality=quality,
        prompt=prompt,
        extra={
            # Keep keys robust to result payloads
            "mutations_generated": stats.get("mutations_generated"),
            "mutations_promoted": stats.get("mutations_promoted"),
            "unique_parents": stats.get("unique_parents"),
            "unique_children": stats.get("unique_children"),
            "evolution_edges": stats.get("evolution_edges"),
        },
    )


async def _eval_prompt_on_val_async(
    prompt: str,
    valset: List[dict],
    *,
    task_lm: str,
    reflection_lm: str,
    eval_concurrency: int,
) -> Tuple[float, float, int]:
    """
    Directly evaluate a single prompt on the full validation set.

    This bypasses TurboGEPA's orchestration/straggler logic and simply runs
    the task LLM once per validation example with bounded concurrency, then
    returns the mean quality and tokens.
    """

    from turbo_gepa.interfaces import Candidate

    dataset = [
        DefaultDataInst(
            input=example["input"],
            answer=example["answer"],
            additional_context=example.get("additional_context"),
            id=f"val_{idx}",
        )
        for idx, example in enumerate(valset)
    ]

    config = adaptive_config(len(dataset))
    config.n_islands = 1
    config.eval_concurrency = max(1, min(eval_concurrency, len(dataset)))
    config.batch_size = len(dataset)
    # Disable any target-based early stop; we want full coverage.
    config.target_quality = None

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        config=config,
        auto_config=False,
        scoring_fn=_quality_reward,
    )

    sem = asyncio.Semaphore(config.eval_concurrency)
    candidate = Candidate(text=prompt, meta={"source": "final_val"})
    qualities: List[float] = []
    tokens: List[float] = []

    async def eval_one(example_id: str) -> None:
        async with sem:
            metrics = await adapter._task_runner(candidate, example_id)  # type: ignore[attr-defined]
            q = metrics.get("quality")
            if isinstance(q, (int, float)):
                qualities.append(float(q))
            t = metrics.get("tokens")
            if isinstance(t, (int, float)):
                tokens.append(float(t))

    try:
        await asyncio.gather(*(eval_one(inst.id) for inst in dataset))
    finally:
        await adapter.aclose()

    if not qualities:
        return 0.0, 0.0, 0

    mean_quality = sum(qualities) / len(qualities)
    mean_tokens = sum(tokens) / len(tokens) if tokens else 0.0
    return mean_quality, mean_tokens, len(qualities)


def run_turbo_validation(
    valset: List[dict],
    *,
    task_lm: str,
    reflection_lm: str,
    best_prompt: str,
    eval_concurrency: int,
) -> Tuple[float, float, int]:
    """
    Public helper: evaluate a TurboGEPA-derived best prompt on the full
    validation set and return (mean_quality, mean_tokens, n_examples).

    This is intentionally separate from the main TurboGEPA run so callers
    can choose when to pay the extra validation cost.
    """

    print("\n=== TurboGEPA validation on full AIME val set ===")
    print(f"Best prompt (snippet): {_summarize_prompt(best_prompt)}")

    mean_quality, mean_tokens, count = asyncio.run(
        _eval_prompt_on_val_async(
            best_prompt,
            valset,
            task_lm=task_lm,
            reflection_lm=reflection_lm,
            eval_concurrency=eval_concurrency,
        )
    )

    print(
        f"TurboGEPA validation: quality={mean_quality:.3f} "
        f"(tokens={mean_tokens:.1f}, examples={count})"
    )
    return mean_quality, mean_tokens, count


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _print_summary(results: Dict[str, BenchmarkResult]) -> None:
    print("\n================ SUMMARY ================\n")
    for label, result in results.items():
        print(f"{label}:")
        print(f"  Time: {result.elapsed:.1f}s")
        print(f"  Evaluations: {result.evaluations}")
        print(f"  Quality: {result.quality:.3f}")
        print(f"  Prompt: { _summarize_prompt(result.prompt) }")
        print("")

    if {"GEPA", "TurboGEPA"}.issubset(results):
        gepa_res = results["GEPA"]
        turbo_res = results["TurboGEPA"]
        if turbo_res.elapsed > 0:
            speedup = gepa_res.elapsed / turbo_res.elapsed
            print(f"⚡ Speedup (GEPA vs TurboGEPA): {speedup:.1f}x")
        if turbo_res.evaluations:
            eval_ratio = gepa_res.evaluations / max(1, turbo_res.evaluations)
            print(f"📉 Evaluation reduction: {eval_ratio:.1f}x fewer evaluations")
        quality_delta = turbo_res.quality - gepa_res.quality
        print(f"🎯 Quality delta (Turbo - GEPA): {quality_delta:+.3f}")
        print("")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GEPA vs TurboGEPA on the AIME dataset.")
    parser.add_argument("--mode", choices=["gepa", "turbo", "both"], default="both")
    parser.add_argument("--task-lm", required=True, help="Model used for evaluation.")
    parser.add_argument("--reflection-lm", required=True, help="Model used for reflection/mutation.")
    parser.add_argument("--dataset-size", type=int, default=30, help="Number of AIME examples to use.")
    parser.add_argument("--seed-prompt", default="You are a helpful assistant.", help="Initial prompt.")
    parser.add_argument("--hf-cache", type=Path, default=Path(".hf_cache"), help="Optional HF cache override.")
    parser.add_argument("--max-metric-calls", type=int, default=40, help="GEPA metric calls.")
    # Turbo-specific knobs
    # Let target-quality and convergence govern stopping by default.
    # Leave these unset (None) unless the caller wants hard caps.
    parser.add_argument("--turbo-max-rounds", type=int, default=None)
    parser.add_argument("--turbo-max-evaluations", type=int, default=None)
    parser.add_argument("--turbo-max-mutations", type=int, default=8)
    parser.add_argument("--turbo-queue-limit", type=int, default=32)
    parser.add_argument("--turbo-eval-concurrency", type=int, default=20)
    parser.add_argument("--turbo-auto-scale", dest="turbo_auto_scale", action="store_true")
    parser.add_argument("--no-turbo-auto-scale", dest="turbo_auto_scale", action="store_false")
    parser.set_defaults(turbo_auto_scale=True)
    parser.add_argument("--turbo-eval-timeout", type=int, default=None, help="Per-example eval timeout seconds (overrides config)")
    parser.add_argument("--turbo-task-max-tokens", type=int, default=None, help="Max tokens for task LLM (caps generation length)")
    parser.add_argument("--turbo-target-quality", type=float, default=None)
    parser.add_argument("--turbo-target-shard", type=float, default=1.0)
    parser.add_argument("--turbo-max-runtime", type=int, default=None)
    parser.add_argument("--turbo-log-level", default="WARNING")
    parser.add_argument("--turbo-show-progress", action="store_true")
    parser.add_argument("--turbo-n-islands", type=int, default=1)
    parser.add_argument("--turbo-verification-speed-bias", type=float, default=None)
    parser.add_argument("--open-ui", action="store_true", help="Open live evolution UI (http://localhost:8080/scripts/evolution_live.html)")
    parser.add_argument(
        "--turbo-strategies",
        nargs="+",
        help=(
            "Names of reflection strategies to enable for TurboGEPA "
            "(e.g., incremental_reflection spec_induction interleaved_thinking). "
            "Defaults to all built-ins when omitted."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_hf_cache(args.hf_cache)

    trainset, valset = _load_aime_subset(args.dataset_size)
    print(f"Loaded AIME subset → train={len(trainset)}, val={len(valset)}")

    results: Dict[str, BenchmarkResult] = {}

    if args.mode in {"gepa", "both"}:
        results["GEPA"] = run_gepa(
            trainset,
            valset,
            task_lm=args.task_lm,
            reflection_lm=args.reflection_lm,
            seed_prompt=args.seed_prompt,
            max_metric_calls=args.max_metric_calls,
        )

    if args.mode in {"turbo", "both"}:
        # Optionally open the live UI with a simple local server
        if args.open_ui:
            try:
                subprocess.Popen(["python", "-m", "http.server", "8080"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                webbrowser.open("http://localhost:8080/scripts/evolution_live_v2.html")
            except Exception:
                pass
        _reset_turbo_cache()
        results["TurboGEPA"] = run_turbo(
            trainset,
            task_lm=args.task_lm,
            reflection_lm=args.reflection_lm,
            seed_prompt=args.seed_prompt,
            args=args,
        )

    if results:
        _print_summary(results)
    else:
        print("No benchmark executed. Use --mode to select gepa/turbo/both.")


if __name__ == "__main__":
    main()
