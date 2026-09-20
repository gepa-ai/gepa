"""Compare execution modes on a synthetic task with realistic stage latencies. No API keys needed.

The task is a toy (an instruction must collect the right tokens), but the latencies are shaped like a
retrieval-agent workload scaled down ~200x: a slow, long-tailed reflection call, a faster rollout, a validation
set larger than the evaluator's worker pool, and a reflection LM that produces a useful edit about half the time.

Modes, all at the same metric-call budget:

- ``sync``        gepa.optimize, one proposal per iteration
- ``sync PxN``    gepa.optimize with batched parallel proposals
- ``async K=..``  gepa.async_engine.optimize with K reflection workers

Run:  uv run python examples/async_engine/simulated_latency.py
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import gepa
from gepa.async_engine import AsyncEngineConfig, StageConfig
from gepa.async_engine import optimize as async_optimize
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.strategies.proposal_sampling import PxNSampling

N_TOKENS = 60
SEED_CANDIDATE = {"instructions": "solve the task"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"k\d+", text))


def _unit(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16**12


class SimulatedAdapter(GEPAAdapter):
    """Every example occupies one of ``workers`` evaluator slots for ``rollout_seconds``."""

    propose_new_texts = None

    def __init__(self, workers: int, rollout_seconds: float):
        self.rollout_seconds = rollout_seconds
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sim-eval")

    def _one(self, _example) -> None:
        time.sleep(self.rollout_seconds)

    def _score(self, batch, candidate, capture_traces):
        have = _tokens(candidate["instructions"])
        scores = [1.0 if f"k{ex % N_TOKENS}" in have else 0.0 for ex in batch]
        traces = [{"example": ex, "score": s} for ex, s in zip(batch, scores, strict=True)] if capture_traces else None
        return EvaluationBatch(outputs=[None] * len(batch), scores=scores, trajectories=traces)

    def evaluate(self, batch, candidate, capture_traces=False):
        list(self._pool.map(self._one, batch))
        return self._score(batch, candidate, capture_traces)

    def batch_evaluate(self, items, capture_traces=True):
        list(self._pool.map(self._one, [ex for _cand, batch in items for ex in batch]))
        return [self._score(batch, cand, capture_traces) for cand, batch in items]

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [
            {
                "Inputs": f"example {t['example']}",
                "Feedback": "correct" if t["score"] >= 1.0 else f"add token k{t['example'] % N_TOKENS}",
            }
            for t in eval_batch.trajectories
        ]
        return dict.fromkeys(components_to_update, records)


class SimulatedReflectionLM:
    """Long-tailed latency; about ``hit_rate`` of calls return a useful edit."""

    def __init__(self, median_seconds: float, sigma: float, hit_rate: float, seed: int):
        self.median_seconds = median_seconds
        self.sigma = sigma
        self.hit_rate = hit_rate
        self.seed = seed
        self.total_cost = 0.0
        self.calls = 0

    def _latency(self, prompt: str) -> float:
        z = random.Random(f"{self.seed}:{prompt}").gauss(0.0, self.sigma)
        return self.median_seconds * math.exp(z)

    def _answer(self, prompt: str) -> str:
        blocks = re.findall(r"```\n(.*?)\n```", prompt, flags=re.S)
        current = blocks[0] if blocks else ""
        if "A worker proposed a revision" in prompt:
            base, revision, now = (_tokens(b) for b in blocks[:3])
            gained = revision - base - now
            return "DISCARD" if not gained else "```\n" + blocks[2] + " " + " ".join(sorted(gained)) + "\n```"
        wanted = [t for t in re.findall(r"add token (k\d+)", prompt) if t not in _tokens(current)]
        if wanted and _unit(f"{self.seed}:hit:{prompt}") < self.hit_rate:
            return f"```\n{current} {wanted[0]}\n```"
        return f"```\n{current} note{len(current)}\n```"

    def __call__(self, prompt) -> str:
        text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        self.calls += 1
        time.sleep(self._latency(text))
        return self._answer(text)

    def batch_complete(self, messages_list):
        """All calls go out together; the batch returns when the slowest one does."""
        texts = [m[-1]["content"] for m in messages_list]
        self.calls += len(texts)
        time.sleep(max(self._latency(t) for t in texts))
        return [self._answer(t) for t in texts]


class _Quiet:
    def log(self, *args, **kwargs):
        pass


def run(mode: str, args, **extra):
    adapter = SimulatedAdapter(args.workers, args.rollout_seconds)
    lm = SimulatedReflectionLM(args.reflect_seconds, args.reflect_sigma, args.hit_rate, seed=args.seed)
    common = {
        "seed_candidate": dict(SEED_CANDIDATE),
        "trainset": list(range(args.train)),
        "valset": list(range(1000, 1000 + args.val)),
        "adapter": adapter,
        "reflection_lm": lm,
        "max_metric_calls": args.budget,
        "seed": args.seed,
        "logger": _Quiet(),
    }
    t0 = time.monotonic()
    result = (async_optimize if mode == "async" else gepa.optimize)(**common, **extra)
    wall = time.monotonic() - t0
    stats = (result.metadata or {}).get("async", {})
    gaps = stats.get("staleness_at_commit", {})
    n_gaps = sum(gaps.values()) or 1
    return {
        "wall": wall,
        "best": result.val_aggregate_scores[result.best_idx],
        "candidates": len(result.candidates),
        "calls": result.total_metric_calls,
        "reflections": lm.calls,
        "overlap": stats.get("stage_overlap_fraction"),
        "mean_gap": sum(int(g) * c for g, c in gaps.items()) / n_gaps if gaps else None,
        "discarded": stats.get("outcomes", {}).get("discarded_stale"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--workers", type=int, default=64, help="evaluator slots shared by every stage")
    ap.add_argument("--rollout-seconds", type=float, default=0.085)
    ap.add_argument("--reflect-seconds", type=float, default=0.20, help="median reflection latency")
    ap.add_argument("--reflect-sigma", type=float, default=0.5, help="log-normal spread of reflection latency")
    ap.add_argument("--hit-rate", type=float, default=0.5)
    ap.add_argument("--max-staleness", type=int, default=4)
    ap.add_argument("--policy", default="guarded", choices=["full", "guarded", "reflective"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [("sync", run("sync", args))]
    for p, n in ((2, 2), (2, 4), (4, 4)):
        rows.append((f"sync {p}x{n}", run("sync", args, sampling_strategy=PxNSampling(p, n))))
    for k in (1, 2, 4, 8):
        config = AsyncEngineConfig(
            rollout=StageConfig(workers=max(1, k // 2)),
            propose=StageConfig(workers=k),
            screen=StageConfig(workers=max(1, k // 2)),
            validate=StageConfig(workers=2),
            patch=StageConfig(workers=max(1, k // 2)),
            staleness_policy=args.policy,
            max_staleness=args.max_staleness,
        )
        rows.append((f"async K={k}", run("async", args, async_config=config)))

    base = rows[0][1]["wall"]
    header = f"{'mode':<12}{'wall s':>8}{'speedup':>9}{'best val':>10}{'pool':>6}{'calls':>7}{'reflect':>9}{'overlap':>9}{'gap':>6}{'stale':>7}"
    print(header)
    print("-" * len(header))
    for name, r in rows:
        fmt = lambda v, spec: "-" if v is None else format(v, spec)  # noqa: E731
        print(
            f"{name:<12}{r['wall']:>8.2f}{base / r['wall']:>8.2f}x{r['best']:>10.3f}{r['candidates']:>6}"
            f"{r['calls']:>7}{r['reflections']:>9}{fmt(r['overlap'], '.2f'):>9}{fmt(r['mean_gap'], '.2f'):>6}"
            f"{fmt(r['discarded'], 'd'):>7}"
        )


if __name__ == "__main__":
    main()
