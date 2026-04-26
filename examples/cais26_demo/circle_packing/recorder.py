"""
Recorder/Replay infrastructure for circle packing demo.

Provides RecorderLM (for reflection_lm) and ReplayEvaluator (for fitness function)
to enable instant replay of a full GEPA optimization run from pre-recorded data.
"""

import json
from pathlib import Path
from typing import Any, Literal

from gepa.core.state import _candidate_hash


class RecorderLM:
    """Records or replays reflection_lm calls.

    Implements the LanguageModel protocol: __call__(prompt: str | list[dict]) -> str

    Record mode: wraps a real LM, saves responses to JSONL.
    Replay mode: returns pre-recorded responses in order.
    """

    def __init__(
        self,
        recordings_dir: Path,
        mode: Literal["record", "replay"],
        wrapped_lm=None,
    ):
        self.recordings_dir = Path(recordings_dir)
        self.mode = mode
        self.call_index = 0

        if mode == "record":
            if wrapped_lm is None:
                raise ValueError("wrapped_lm required in record mode")
            self.wrapped_lm = wrapped_lm
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(
                self.recordings_dir / "reflection_lm_calls.jsonl", "w"
            )
        elif mode == "replay":
            self.recordings = []
            jsonl_path = self.recordings_dir / "reflection_lm_calls.jsonl"
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.recordings.append(json.loads(line))
            print(f"RecorderLM: loaded {len(self.recordings)} recordings from {jsonl_path}")
        else:
            raise ValueError(f"mode must be 'record' or 'replay', got {mode!r}")

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if self.mode == "record":
            response = self.wrapped_lm(prompt)
            record = {
                "call_index": self.call_index,
                "response": response,
            }
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()
            self.call_index += 1
            return response
        else:
            if self.call_index >= len(self.recordings):
                raise IndexError(
                    f"RecorderLM replay exhausted: requested call {self.call_index}, "
                    f"but only {len(self.recordings)} recordings available"
                )
            response = self.recordings[self.call_index]["response"]
            self.call_index += 1
            return response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.mode == "record" and hasattr(self, "_file"):
            self._file.close()


class ReplayProposer:
    """Custom candidate proposer that feeds pre-recorded candidates in order.

    Implements the ProposalFn protocol. Returns ALL components of each
    candidate, so the result is independent of which parent GEPA selects.
    """

    def __init__(self, candidates: list[dict[str, str]]):
        self.candidates = candidates
        self.index = 1  # skip seed (index 0)

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Any,
        components_to_update: list[str],
    ) -> dict[str, str]:
        if self.index >= len(self.candidates):
            return {}
        next_cand = self.candidates[self.index]
        self.index += 1
        # Return ALL components to override parent entirely
        return next_cand.copy()


def create_replay_evaluator(evaluation_cache: dict):
    """Create an evaluator that returns pre-cached results from a loaded GEPA state.

    The returned function matches the signature GEPA's EvaluatorWrapper expects:
    evaluator(candidate, **kwargs) -> (score, side_info_dict)

    Args:
        evaluation_cache: The _cache dict from GEPAState.evaluation_cache
    """
    # Build lookup: candidate_hash -> (score, objective_scores)
    lookup: dict[str, tuple[float, dict]] = {}
    for (cand_hash, _example_id), cached_eval in evaluation_cache.items():
        obj_scores = cached_eval.objective_scores or {}
        lookup[cand_hash] = (cached_eval.score, obj_scores)

    print(f"ReplayEvaluator: loaded {len(lookup)} cached evaluations")

    def evaluator(candidate: dict[str, str], **kwargs) -> tuple[float, dict]:
        h = _candidate_hash(candidate)
        if h in lookup:
            score, obj_scores = lookup[h]
            return score, {"scores": obj_scores}
        # Return 0 for unknown candidates (shouldn't happen with ReplayProposer)
        print(f"WARNING: Candidate hash {h[:16]}... not in replay cache")
        return 0.0, {"scores": {}}

    return evaluator
