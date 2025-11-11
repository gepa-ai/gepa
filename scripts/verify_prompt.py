#!/usr/bin/env python3
"""
Verify the best TurboGEPA prompt on the AIME dataset with live LLM calls.

This script reloads the cached best prompt (or a user-supplied prompt) and
replays it across a chosen dataset split while bypassing the optimization
cache. It mirrors the same task model that TurboGEPA used so we can confirm
end-to-end quality numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Sequence

import gepa

from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config, adaptive_config

# NEVER EVER MODIFY THESE!!!
DEFAULT_TASK_LM = "openrouter/openai/gpt-oss-20b:nitro"
DEFAULT_REFLECTION_LM = "openrouter/x-ai/grok-4-fast"


def _load_aime_split(split: str, limit: int) -> Sequence[DefaultDataInst]:
    trainset, valset, _ = gepa.examples.aime.init_dataset()
    source = trainset if split == "train" else valset
    if limit > 0:
        source = source[: min(limit, len(source))]
    dataset: list[DefaultDataInst] = []
    for idx, example in enumerate(source):
        dataset.append(
            DefaultDataInst(
                input=example["input"],
                answer=example["answer"],
                additional_context=example.get("additional_context"),
                id=f"{split}_{idx:04d}",
            )
        )
    return dataset


def _summarize_prompt(prompt: str, max_chars: int = 160) -> str:
    snippet = " ".join(prompt.split())
    return snippet[: max_chars - 1] + "…" if len(snippet) > max_chars else snippet


def _configure_hf_cache(cache_root: Path | None) -> None:
    """Mirror benchmark helper so datasets stay inside the repo workspace."""
    if cache_root is None:
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    datasets_cache = cache_root / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate TurboGEPA's best prompt on the AIME dataset.")
    parser.add_argument(
        "--prompt-file",
        default=".turbo_gepa/best_prompt.txt",
        help="Path to the prompt text file saved during optimization.",
    )
    parser.add_argument(
        "--prompt-text",
        help="Inline prompt text. When provided, this overrides --prompt-file.",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Dataset split to evaluate.")
    parser.add_argument("--dataset-size", type=int, default=30, help="Number of examples to evaluate (per split).")
    parser.add_argument("--task-lm", default=DEFAULT_TASK_LM, help="Task model to use for evaluation.")
    parser.add_argument(
        "--reflection-lm",
        default=DEFAULT_REFLECTION_LM,
        help="Reflection model (required by adapter, not used during verification).",
    )
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent AIME evaluations.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="TurboGEPA log level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--failure-limit",
        type=int,
        default=4,
        help="How many failed examples to print inline.",
    )
    parser.add_argument(
        "--output-json",
        default=".turbo_gepa/verification/latest_verification.json",
        help="Where to store the verification summary as JSON.",
    )
    parser.add_argument(
        "--allow-cache",
        action="store_true",
        help="Reuse the existing TurboGEPA eval cache instead of forcing fresh calls.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Stream per-example progress logs from the evaluator.",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path(".hf_cache"),
        help="Optional Hugging Face cache directory override.",
    )
    parser.add_argument(
        "--no-hf-cache",
        action="store_true",
        help="Skip overriding Hugging Face cache directories.",
    )
    return parser.parse_args()


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_text:
        return args.prompt_text.strip()
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _configure_adapter(dataset: Sequence[DefaultDataInst], args: argparse.Namespace) -> DefaultAdapter:
    config: Config = adaptive_config(len(dataset))
    config.eval_concurrency = max(1, min(args.eval_concurrency, len(dataset)))
    config.batch_size = max(1, min(config.batch_size, len(dataset)))
    config.log_level = args.log_level
    config.enable_debug_log = args.log_level.upper() == "DEBUG"
    return DefaultAdapter(
        dataset=dataset,
        task_lm=args.task_lm,
        reflection_lm=args.reflection_lm,
        config=config,
        auto_config=False,
    )


def _summaries_from_traces(traces: Sequence[dict], failure_limit: int):
    correct = 0
    failures: list[dict] = []
    for trace in traces:
        quality = trace.get("quality", 0.0) or 0.0
        if quality >= 0.999:
            correct += 1
        else:
            if len(failures) < failure_limit:
                failures.append(trace)
    return correct, failures


async def _verify(args: argparse.Namespace) -> dict[str, object]:
    if not args.no_hf_cache:
        _configure_hf_cache(args.hf_cache)
    dataset = _load_aime_split(args.split, args.dataset_size)
    if not dataset:
        raise ValueError(f"No AIME examples available for split '{args.split}'.")

    adapter = _configure_adapter(dataset, args)
    prompt = _load_prompt(args)
    concurrency = max(1, min(args.eval_concurrency, len(dataset)))
    preview = _summarize_prompt(prompt)
    print(f"🔍 Verifying prompt on AIME {args.split} split ({len(dataset)} examples, concurrency={concurrency})")
    print(f"   Prompt preview: {preview}")

    start = time.perf_counter()
    result = await adapter.evaluate_prompt_async(
        prompt,
        concurrency=concurrency,
        bypass_cache=not args.allow_cache,
        show_progress=args.show_progress,
        label=f"{args.split}_verification",
    )
    elapsed = time.perf_counter() - start
    await adapter.aclose()

    traces = result.traces or []
    correct, failures = _summaries_from_traces(traces, args.failure_limit)
    total = len(dataset)
    coverage = result.n_examples / max(total, 1)
    accuracy = correct / max(total, 1)

    throughput = result.n_examples / elapsed if elapsed > 0 else float("inf")
    print(f"✅ Accuracy {accuracy:.1%} ({correct}/{total}) in {elapsed:.1f}s — {throughput:.2f} ex/s")
    if coverage < 0.95:
        print(f"⚠️  Coverage warning: only {coverage:.0%} of the dataset was evaluated this run.")
    if failures:
        print(f"⚠️  Showing {len(failures)} failing examples:")
        for trace in failures:
            example_id = trace.get("example_id")
            expected = trace.get("expected_answer", "").strip()
            output = (trace.get("output") or "").strip()
            print(f"   · {example_id}: expected {expected!r} — model output: {output[:120]}{'…' if output and len(output) > 120 else ''}")
    else:
        if coverage >= 0.95:
            print("🎉 All evaluated examples were answered correctly.")
        else:
            print("ℹ️  No failures observed on the evaluated subset.")

    summary = {
        "prompt_preview": preview,
        "prompt_length": len(prompt),
        "split": args.split,
        "dataset_size": len(dataset),
        "task_lm": args.task_lm,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_seconds": elapsed,
        "throughput": throughput,
        "coverage": coverage,
        "timestamp": time.time(),
        "failures": failures,
    }
    return summary


def _write_summary(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"📝 Saved verification summary → {path}")


def main() -> None:
    args = _parse_args()
    summary = asyncio.run(_verify(args))
    _write_summary(summary, Path(args.output_json))


if __name__ == "__main__":
    main()
