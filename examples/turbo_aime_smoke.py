"""
Tiny TurboGEPA smoke test that pings the real LLM stack on the AIME dataset.

Usage (env vars such as OPENROUTER_API_KEY must already be exported, e.g. via `.envrc`):

    python examples/turbo_aime_smoke.py \
        --task-lm openrouter/openai/gpt-oss-20b:nitro \
        --reflection-lm openrouter/x-ai/grok-4-fast \
        --strategies incremental_reflection spec_induction interleaved_thinking

This script keeps the dataset tiny but still exercises all configured reflection strategies,
clears `.turbo_gepa/` on every run, and prints the actual prompts produced so we can confirm
interleaved thinking (and other strategies) are working.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Sequence

import gepa
from turbo_gepa.adapters import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config, adaptive_config


def _configure_hf_cache(cache_root: Path | None) -> None:
    """Optionally redirect Hugging Face caches to a workspace-friendly path."""

    if cache_root is None:
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    datasets_cache = cache_root / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))


def _load_aime_subset(limit: int) -> list[DefaultDataInst]:
    """Load a small slice of the AIME train split for quick verification."""

    trainset, _, _ = gepa.examples.aime.init_dataset()
    trimmed = trainset[:limit]
    dataset: list[DefaultDataInst] = []
    for idx, example in enumerate(trimmed):
        dataset.append(
            DefaultDataInst(
                input=example["input"],
                answer=example["answer"],
                additional_context=example.get("additional_context"),
                id=f"aime_{idx}",
            )
        )
    return dataset


def _prepare_cache_dirs(root: Path) -> tuple[str, str]:
    """Clear .turbo_gepa/ between runs so cached state never masks new strategies."""

    if root.exists():
        shutil.rmtree(root)
    cache_dir = root / "cache"
    log_dir = root / "logs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir), str(log_dir)


def _build_config(dataset: Sequence[DefaultDataInst], args: argparse.Namespace) -> Config:
    """Derive a conservative config that still exercises the full strategy stack."""

    config = adaptive_config(len(dataset))
    config.n_islands = 1
    config.eval_concurrency = min(args.eval_concurrency, max(1, len(dataset)))
    config.batch_size = max(1, min(len(dataset), config.eval_concurrency))
    config.max_mutations_per_round = max(1, args.max_mutations_per_round)
    config.queue_limit = max(2, args.queue_limit)
    config.log_level = args.log_level
    config.target_quality = args.target_quality
    config.max_optimization_time_seconds = args.max_runtime
    if args.strategies:
        config.reflection_strategy_names = tuple(args.strategies)

    cache_root = Path(".turbo_gepa")
    cache_path, log_path = _prepare_cache_dirs(cache_root)
    config.cache_path = cache_path
    config.log_path = log_path
    return config


def _summarize_results(result: dict) -> None:
    pareto_entries = result.get("pareto_entries", []) or []
    if not pareto_entries:
        print("⚠️  No Pareto entries returned; check logs above for LLM failures.")
        return

    print(f"\nPareto entries: {len(pareto_entries)}")
    for idx, entry in enumerate(pareto_entries, 1):
        shard = entry.result.shard_fraction or 0.0
        quality = entry.result.objectives.get("quality", 0.0)
        snippet = entry.candidate.text.replace("\n", " ")[:200]
        print(f"  #{idx}: quality={quality:.2f}, shard={shard:.2f}, prompt_snippet={snippet!r}")

    stats = result.get("evolution_stats", {})
    print("\nEvolution stats snapshot:")
    for key in ("total_evaluations", "mutations_generated", "mutations_promoted"):
        print(f"  {key}: {stats.get(key, 'n/a')}")

    # Heuristic check: did we see any interleaved-thinking prompts?
    interleaved = []
    for entry in pareto_entries:
        text_lower = entry.candidate.text.lower()
        if "<think>" in text_lower or "&lt;think" in text_lower:
            interleaved.append(entry.candidate.text)
    if interleaved:
        print(f"\n✅ Detected {len(interleaved)} prompt(s) with interleaved <think>/<answer> structure.")
    else:
        print("\n⚠️  No interleaved prompts detected in Pareto set.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-lm", required=True, help="Model used for task execution.")
    parser.add_argument("--reflection-lm", required=True, help="Model used for reflection/mutation.")
    parser.add_argument("--limit", type=int, default=4, help="Number of AIME training examples to use.")
    parser.add_argument(
        "--seed-prompt",
        default="You are a competition math tutor. Solve the question carefully and end with '### <final answer>'.",
        help="Initial seed prompt to optimize.",
    )
    parser.add_argument("--hf-cache", type=Path, default=Path(".hf_cache"), help="Optional HF cache override.")
    parser.add_argument("--eval-concurrency", type=int, default=2, help="Evaluation concurrency cap.")
    parser.add_argument("--max-mutations-per-round", type=int, default=3, help="Mutations to request per round.")
    parser.add_argument("--queue-limit", type=int, default=8, help="Candidate queue size.")
    parser.add_argument("--target-quality", type=float, default=0.5, help="Early-stop target quality threshold.")
    parser.add_argument("--max-runtime", type=int, default=300, help="Max optimization time in seconds.")
    parser.add_argument("--log-level", default="INFO", help="TurboGEPA log level.")
    parser.add_argument("--rounds", type=int, default=5, help="Optimization rounds to run.")
    parser.add_argument("--max-evaluations", type=int, default=40, help="Max total evaluations.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        help=(
            "Reflection strategy names to enable (e.g., incremental_reflection spec_induction interleaved_thinking). "
            "Defaults to all built-in strategies when omitted."
        ),
    )
    args = parser.parse_args()

    _configure_hf_cache(args.hf_cache)
    dataset = _load_aime_subset(args.limit)
    print(f"Loaded {len(dataset)} AIME examples for smoke test.")
    config = _build_config(dataset, args)

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=args.task_lm,
        reflection_lm=args.reflection_lm,
        config=config,
        auto_config=False,
    )

    start = time.time()
    try:
        print("\nRunning TurboGEPA optimize()...\n")
        result = adapter.optimize(
            seeds=[args.seed_prompt],
            max_rounds=args.rounds,
            max_evaluations=args.max_evaluations,
            display_progress=False,
        )
    finally:
        asyncio.run(adapter.aclose())

    elapsed = time.time() - start
    print(f"\n⏱️  Optimization finished in {elapsed:.1f}s\n")
    _summarize_results(result)


if __name__ == "__main__":
    main()
