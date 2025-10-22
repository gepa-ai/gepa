"""
Content-addressed cache utilities.

The cache stores:
1. Evaluation results keyed by candidate hash and example ID
2. Orchestrator state for resumable optimization (archive, queue, round number)

This enables cross-island reuse and automatic resume after cancellation.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .interfaces import Candidate, EvalResult


def candidate_key(candidate: Candidate) -> str:
    """Compute a stable hash for a candidate incorporating temperature metadata."""
    return candidate.fingerprint


class DiskCache:
    """
    JSONL-backed cache for evaluation results.

    Files are partitioned by candidate hash to minimize contention; writes are
    serialized with an asyncio lock so async evaluators can share the cache.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Use defaultdict to avoid race condition in lock creation
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # Global semaphore to limit concurrent file operations (prevent "too many open files")
        self._file_semaphore = asyncio.Semaphore(100)  # Max 100 concurrent file operations

    def _lock_for(self, key: str) -> asyncio.Lock:
        # defaultdict ensures atomic lock creation per key
        return self._locks[key]

    def _record_path(self, cand_hash: str) -> Path:
        prefix = cand_hash[:2]
        shard_dir = self.cache_dir / prefix
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{cand_hash}.jsonl"

    async def get(self, candidate: Candidate, example_id: str) -> Optional[EvalResult]:
        """Fetch a cached result if present."""
        cand_hash = candidate_key(candidate)
        path = self._record_path(cand_hash)
        if not path.exists():
            return None
        lock = self._lock_for(cand_hash)
        async with lock:
            # Use semaphore to limit concurrent file operations
            async with self._file_semaphore:
                cached = await asyncio.to_thread(self._read_record, path, example_id)
        return cached

    async def set(self, candidate: Candidate, example_id: str, result: EvalResult) -> None:
        """Persist a new evaluation record."""
        cand_hash = candidate_key(candidate)
        path = self._record_path(cand_hash)
        record = {
            "example_id": example_id,
            "objectives": dict(result.objectives),
            "traces": result.traces,
            "n_examples": result.n_examples,
            "shard_fraction": result.shard_fraction,
        }
        lock = self._lock_for(cand_hash)
        async with lock:
            # Use semaphore to limit concurrent file operations
            async with self._file_semaphore:
                await asyncio.to_thread(self._append_record, path, record)

    async def batch_set(self, writes: List[Tuple[Candidate, str, EvalResult]]) -> None:
        """Batch write multiple results, grouping by candidate for efficiency."""
        from collections import defaultdict

        # Group writes by candidate hash to minimize lock contention
        by_candidate: defaultdict[str, List[Tuple[Path, Dict[str, object]]]] = defaultdict(list)

        for candidate, example_id, result in writes:
            cand_hash = candidate_key(candidate)
            path = self._record_path(cand_hash)
            record = {
                "example_id": example_id,
                "objectives": dict(result.objectives),
                "traces": result.traces,
                "n_examples": result.n_examples,
                "shard_fraction": result.shard_fraction,
            }
            by_candidate[cand_hash].append((path, record))

        async def write_batch(cand_hash: str, records: List[Tuple[Path, Dict]]) -> None:
            # All records for same candidate go to same file
            path = records[0][0]
            record_objs = [r[1] for r in records]
            lock = self._lock_for(cand_hash)
            async with lock:
                # Use semaphore to limit concurrent file operations
                async with self._file_semaphore:
                    await asyncio.to_thread(self._append_records, path, record_objs)

        # Write all candidate batches in parallel
        await asyncio.gather(*(write_batch(k, v) for k, v in by_candidate.items()))

    async def sample_failures(self, candidate: Candidate, limit: int = 5) -> List[Tuple[str, List[Dict]]]:
        """
        Return failure traces for speculative reflection.

        Currently returns the most recent ``limit`` traces associated with the
        candidate, deferring richer heuristics to future iterations.
        """
        cand_hash = candidate_key(candidate)
        path = self._record_path(cand_hash)
        if not path.exists():
            return []
        lock = self._lock_for(cand_hash)
        async with lock:
            lines = await asyncio.to_thread(self._read_tail_lines, path, limit)
        failures: List[Tuple[str, List[Dict]]] = []
        for line in lines:
            record = json.loads(line)
            if record.get("traces"):
                failures.append((record["example_id"], record["traces"]))
        return failures

    def clear(self) -> None:
        """Remove all cached records (useful for tests)."""
        if not self.cache_dir.exists():
            return
        for root, _dirs, files in os.walk(self.cache_dir, topdown=False):
            for file in files:
                Path(root, file).unlink()
        for root, _dirs, _files in os.walk(self.cache_dir, topdown=False):
            Path(root).rmdir()

    def _read_record(self, path: Path, example_id: str) -> Optional[EvalResult]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["example_id"] == example_id:
                    return EvalResult(
                        objectives=record["objectives"],
                        traces=record["traces"],
                        n_examples=record["n_examples"],
                        shard_fraction=record.get("shard_fraction"),
                        example_ids=[record["example_id"]],
                    )
        return None

    def _append_record(self, path: Path, record: Dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _append_records(self, path: Path, records: List[Dict[str, object]]) -> None:
        """Batch write multiple records to same file."""
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def _read_tail_lines(self, path: Path, limit: int) -> List[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[-limit:]

    # State persistence for resumable optimization

    def _state_path(self) -> Path:
        """Path to orchestrator state file."""
        return self.cache_dir / "orchestrator_state.json"

    def save_state(
        self,
        round_num: int,
        evaluations: int,
        pareto_candidates: List[Candidate],
        qd_candidates: List[Candidate],
        queue: List[Candidate],
    ) -> None:
        """
        Save orchestrator state for resumable optimization.

        Atomically writes state to disk so it's safe to interrupt anytime.
        """
        state = {
            "round": round_num,
            "evaluations": evaluations,
            "pareto": [self._serialize_candidate(c) for c in pareto_candidates],
            "qd": [self._serialize_candidate(c) for c in qd_candidates],
            "queue": [self._serialize_candidate(c) for c in queue],
        }

        # Atomic write: write to temp file, then rename
        state_path = self._state_path()
        temp_path = state_path.with_suffix(".tmp")

        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        temp_path.replace(state_path)

    def load_state(self) -> Optional[Dict]:
        """
        Load saved orchestrator state, or None if no state exists.

        Returns dict with keys: round, evaluations, pareto, qd, queue
        """
        state_path = self._state_path()
        if not state_path.exists():
            return None

        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        # Deserialize candidates
        state["pareto"] = [self._deserialize_candidate(c) for c in state["pareto"]]
        state["qd"] = [self._deserialize_candidate(c) for c in state["qd"]]
        state["queue"] = [self._deserialize_candidate(c) for c in state["queue"]]

        return state

    def has_state(self) -> bool:
        """Check if saved state exists."""
        return self._state_path().exists()

    def clear_state(self) -> None:
        """Delete saved state file."""
        state_path = self._state_path()
        if state_path.exists():
            state_path.unlink()

    def _serialize_candidate(self, candidate: Candidate) -> Dict:
        """Convert Candidate to JSON-serializable dict."""
        return {
            "text": candidate.text,
            "meta": dict(candidate.meta),
        }

    def _deserialize_candidate(self, data: Dict) -> Candidate:
        """Reconstruct Candidate from dict."""
        return Candidate(
            text=data["text"],
            meta=data.get("meta", {}),
        )
