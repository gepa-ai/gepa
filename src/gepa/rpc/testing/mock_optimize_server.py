"""Launches a real gepa-rpc server with gepa.optimize patched to a tiny,
deterministic fake -- no LLM calls, no recorded cache needed.

Lets each SDK's tests exercise the real wire protocol for the non-Omni
RunOptimization path (request/response shapes, trajectories, the
reflective-dataset round trip) as a genuine subprocess over a real gRPC
stream, the same way tests/test_rpc_e2e/ covers the Omni path -- but without
needing a recorded LLM cache, since this "optimizer" never calls one.

The fake runs exactly one evaluate() round over the full trainset, then one
make_reflective_dataset() round (so both RPCs in the non-Omni contract are
exercised), then returns the seed candidate unchanged as the best one.

Usage::

    python -m gepa.rpc.testing.mock_optimize_server --port 8000 --runs-dir /tmp/runs
"""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace
from unittest.mock import patch

from gepa.rpc.server import serve


def _fake_optimize(*, seed_candidate, trainset, adapter, **_):
    batch = list(trainset)
    eval_result = adapter.evaluate(batch, dict(seed_candidate), capture_traces=True)
    adapter.make_reflective_dataset(dict(seed_candidate), eval_result, list(seed_candidate.keys()))
    return SimpleNamespace(
        candidates=[dict(seed_candidate)],
        best_idx=0,
        val_aggregate_scores=[sum(eval_result.scores) / max(len(eval_result.scores), 1)],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--runs-dir", required=True)
    args = parser.parse_args()

    with patch("gepa.rpc.servicer.gepa.optimize", side_effect=_fake_optimize):
        serve(port=args.port, runs_dir=args.runs_dir)


if __name__ == "__main__":
    main()
