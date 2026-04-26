#!/usr/bin/env python3
"""
Extract recordings from a completed GEPA run for replay.

Reads gepa_state.bin and generates:
- recordings/reflection_lm_calls.jsonl — simulated LM responses for each candidate
- recordings/evaluation_cache.pkl — serialized evaluation cache for replay evaluator
"""

import json
import pickle
from pathlib import Path

from gepa.core.state import GEPAState, _candidate_hash


def extract_recordings(state_dir: str = "output", recordings_dir: str = "recordings"):
    state = GEPAState.load(state_dir)
    out_dir = Path(recordings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = state.program_candidates
    parents = state.parent_program_for_candidate
    print(f"Loaded {len(candidates)} candidates from {state_dir}/gepa_state.bin")

    # Extract reflection_lm responses
    # For each non-seed candidate, determine which component changed and wrap it
    # in ``` blocks to simulate the LM response GEPA's output_extractor expects.
    recordings = []
    for i in range(1, len(candidates)):
        parent_idx = parents[i][0]
        parent = candidates[parent_idx]
        child = candidates[i]

        changed = [k for k in parent if parent[k] != child[k]]
        assert len(changed) == 1, (
            f"Expected exactly 1 changed component for candidate {i}, got {changed}"
        )

        component = changed[0]
        new_text = child[component]

        # Wrap in ``` blocks so GEPA's InstructionProposalSignature.output_extractor
        # can parse it (it finds first and last ``` and extracts content between them)
        response = f"Here is the improved {component}:\n\n```\n{new_text}\n```"

        recordings.append({
            "call_index": len(recordings),
            "response": response,
        })

    jsonl_path = out_dir / "reflection_lm_calls.jsonl"
    with open(jsonl_path, "w") as f:
        for record in recordings:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(recordings)} reflection_lm recordings to {jsonl_path}")

    # Save evaluation cache for replay evaluator
    cache_path = out_dir / "evaluation_cache.pkl"
    cache_data = {
        "candidates": candidates,
        "cache": state.evaluation_cache._cache,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Saved evaluation cache ({len(state.evaluation_cache._cache)} entries) to {cache_path}")

    # Print summary of score progression
    print("\nScore progression from candidates:")
    hash_to_idx = {_candidate_hash(cand): idx for idx, cand in enumerate(candidates)}
    best_score = 0.0
    for (cand_hash, _), cached in state.evaluation_cache._cache.items():
        if cached.score > best_score:
            best_score = cached.score
            idx = hash_to_idx.get(cand_hash)
            if idx is not None:
                print(f"  Candidate {idx}: score={cached.score:.4f}")


if __name__ == "__main__":
    extract_recordings()
