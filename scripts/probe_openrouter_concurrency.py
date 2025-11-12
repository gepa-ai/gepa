#!/usr/bin/env python3
"""
OpenRouter concurrency probe using LiteLLM with a shared httpx session.

- Uses the same model IDs as TurboGEPA (defaults to OSS-20)
- Configures LiteLLM's global httpx clients to match requested concurrency
- Fires N requests with bounded concurrency and reports throughput + latency stats

Usage:
  source .envrc && source .venv/bin/activate
  python scripts/probe_openrouter_concurrency.py \
    --model openrouter/openai/gpt-oss-20b:nitro \
    --requests 200 \
    --concurrency 8 200
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any, Sequence

from turbo_gepa.utils.litellm_client import configure_litellm_client


def _pctl(samples: Sequence[float], q: float) -> float:
    if not samples:
        return 0.0
    values = sorted(samples)
    idx = (len(values) - 1) * max(0.0, min(1.0, q))
    lo = int(idx)
    hi = min(len(values) - 1, lo + 1)
    if lo == hi:
        return float(values[lo])
    frac = idx - lo
    return float(values[lo] + (values[hi] - values[lo]) * frac)


async def _fire_once(acompletion, model: str, prompt: str, max_tokens: int) -> tuple[bool, float]:
    t0 = time.perf_counter()
    try:
        await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        ok = True
    except Exception:
        ok = False
    return ok, time.perf_counter() - t0


async def run_probe(model: str, total: int, concurrency: int, *, max_tokens: int = 16, prompt: str | None = None) -> dict[str, Any]:
    prompt = prompt or "Reply with the word: OK"
    # Configure LiteLLM shared client to match concurrency (pooled connections)
    configure_litellm_client(max(concurrency, 1))
    try:
        from litellm import acompletion  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("litellm is required for this probe (pip install litellm)") from exc

    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[tuple[bool, float]] = []

    async def worker(i: int) -> None:
        async with sem:
            res = await _fire_once(acompletion, model, prompt, max_tokens)
            results.append(res)

    t0 = time.perf_counter()
    await asyncio.gather(*[asyncio.create_task(worker(i)) for i in range(total)])
    elapsed = time.perf_counter() - t0

    oks = [lat for ok, lat in results if ok]
    errs = [lat for ok, lat in results if not ok]
    p50 = _pctl(oks, 0.50) if oks else 0.0
    p95 = _pctl(oks, 0.95) if oks else 0.0
    mean = statistics.fmean(oks) if oks else 0.0
    throughput = total / elapsed if elapsed > 0 else 0.0

    summary = {
        "model": model,
        "total": total,
        "concurrency": concurrency,
        "ok": len(oks),
        "errors": len(errs),
        "elapsed_sec": elapsed,
        "throughput_ex_s": throughput,
        "latency_mean_s": mean,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
    }
    return summary


def print_summary(tag: str, stats: dict[str, Any]) -> None:
    print(f"\n=== {tag} ===")
    print(
        f"model={stats['model']} total={stats['total']} conc={stats['concurrency']} "
        f"time={stats['elapsed_sec']:.1f}s thrpt={stats['throughput_ex_s']:.2f} ex/s ok={stats['ok']} err={stats['errors']}"
    )
    print(
        f"latency mean={stats['latency_mean_s']:.2f}s p50={stats['latency_p50_s']:.2f}s p95={stats['latency_p95_s']:.2f}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenRouter concurrency probe via LiteLLM")
    ap.add_argument("--model", default="openrouter/openai/gpt-oss-20b:nitro")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[8, 200])
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--prompt", default="Reply with the word: OK")
    args = ap.parse_args()

    print("🔍 Concurrency probe starting…")
    print(f"   Model: {args.model}")
    print(f"   Requests per run: {args.requests}")
    print(f"   Concurrency levels: {args.concurrency}")

    for conc in args.concurrency:
        print(f"\n🚀 Running conc={conc} …")
        stats = asyncio.run(
            run_probe(args.model, total=args.requests, concurrency=conc, max_tokens=args.max_tokens, prompt=args.prompt)
        )
        print_summary(f"conc={conc}", stats)


if __name__ == "__main__":
    main()

