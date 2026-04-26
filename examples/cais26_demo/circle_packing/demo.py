#!/usr/bin/env python3
"""
Replay demo for circle packing optimization.

Runs a full GEPA optimization loop instantly using pre-recorded candidates
and cached evaluation results. No API calls or code execution needed.

Usage:
    python demo.py

Prerequisites:
    python extract_recordings.py  # populate recordings/ from output/gepa_state.bin
"""

import os
import pickle
import sys
import time
import types
from pathlib import Path

# Stub dspy so llms.py can be imported without it installed
if "dspy" not in sys.modules:
    _dspy_stub = types.ModuleType("dspy")
    _dspy_stub.Signature = type("Signature", (), {})
    _dspy_stub.InputField = lambda **kw: None
    _dspy_stub.OutputField = lambda **kw: None
    sys.modules["dspy"] = _dspy_stub

from llms import CIRCLE_PACKING_BACKGROUND, SEED_REFINEMENT_PROMPT
from utils import SEED_CODE
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

from recorder import ReplayProposer, create_replay_evaluator


RECORDINGS_DIR = Path("recordings")


def main():
    # Load evaluation cache
    cache_path = RECORDINGS_DIR / "evaluation_cache.pkl"
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    print(f"Loaded evaluation cache: {len(cache_data['cache'])} entries")

    # Create replay proposer that feeds pre-recorded candidates
    replay_proposer = ReplayProposer(cache_data["candidates"])

    # Create replay evaluator that returns cached scores
    replay_eval = create_replay_evaluator(cache_data["cache"])

    # Seed candidate (same as original run)
    seed_candidate = {
        "code": SEED_CODE,
        "refiner_prompt": SEED_REFINEMENT_PROMPT,
    }

    num_proposals = len(cache_data["candidates"]) - 1  # exclude seed

    # Set up output directory for replay
    log_dir = f"outputs/artifacts/circle_packing/replay_{time.strftime('%y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            track_best_outputs=True,
            frontier_type="objective",
            cache_evaluation=True,
            use_cloudpickle=False,
            max_candidate_proposals=num_proposals,
        ),
        reflection=ReflectionConfig(
            reflection_lm=None,
            custom_candidate_proposer=replay_proposer,
        ),
    )

    print("\n" + "=" * 70)
    print("Circle Packing Demo — Replay Mode")
    print("=" * 70)
    print(f"Replaying {num_proposals} candidate proposals")
    print(f"Log directory: {log_dir}")
    print("=" * 70 + "\n")

    start = time.time()
    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=replay_eval,
        config=gepa_config,
        objective="Optimize circle packing code and refiner prompt to maximize sum of circle radii within a unit square for N=26 circles.",
        background=CIRCLE_PACKING_BACKGROUND,
    )
    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print(f"Replay Complete! ({elapsed:.1f}s)")
    print("=" * 70)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
