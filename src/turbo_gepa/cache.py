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
import logging
import os
from collections import defaultdict
from pathlib import Path

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
        # Dynamically determine safe limit based on system's file descriptor limit
        self._file_semaphore = asyncio.Semaphore(self._get_safe_file_limit())
        # In-memory index to avoid repeated linear scans per candidate
        self._record_cache: dict[str, dict[str, EvalResult]] = {}

    def _get_safe_file_limit(self) -> int:
        """Determine a safe file descriptor limit for concurrent operations.

        Returns a conservative limit that accounts for:
        - System's soft file descriptor limit
        - Multiple cache instances (e.g., in multi-island mode)
        - Other file operations (logging, etc.)
        """
        import resource

        try:
            # Get soft limit for open files
            soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

            # Conservative calculation:
            # - Assume up to 8 cache instances (generous for multi-island)
            # - Reserve 50% for other operations (Python internals, logging, etc.)
            # - Divide remaining by number of potential caches
            usable = int(soft_limit * 0.5)  # Use 50% of limit
            per_cache = max(10, usable // 8)  # At least 10, divided by 8 potential caches

            # Cap at 50 to avoid excessive concurrency even on high-limit systems
            return min(per_cache, 50)

        except (ValueError, OSError):
            # Fallback if resource.getrlimit fails (e.g., on Windows)
            # Use a very conservative default
            return 20

    def _lock_for(self, key: str) -> asyncio.Lock:
        # defaultdict ensures atomic lock creation per key
        return self._locks[key]

    def _record_path(self, cand_hash: str) -> Path:
        prefix = cand_hash[:2]
        shard_dir = self.cache_dir / prefix
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{cand_hash}.jsonl"

    def _clone_result(self, result: EvalResult, example_id: str) -> EvalResult:
        example_ids = list(result.example_ids) if result.example_ids else [example_id]
        if example_id not in example_ids:
            example_ids = [example_id]
        traces = [
            dict(trace) if isinstance(trace, dict) else trace  # shallow copy to avoid side-effects
            for trace in (result.traces or [])
        ]
        return EvalResult(
            objectives=dict(result.objectives),
            traces=traces,
            n_examples=result.n_examples,
            shard_fraction=result.shard_fraction,
            example_ids=example_ids,
        )

    def _ensure_candidate_cache(self, cand_hash: str, path: Path) -> dict[str, EvalResult]:
        cached = self._record_cache.get(cand_hash)
        if cached is not None:
            return cached

        records: dict[str, EvalResult] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    example_id = record["example_id"]
                    result = EvalResult(
                        objectives=dict(record["objectives"]),
                        traces=[
                            dict(trace) if isinstance(trace, dict) else trace
                            for trace in record.get("traces", [])
                        ],
                        n_examples=record.get("n_examples", 1),
                        shard_fraction=record.get("shard_fraction"),
                        example_ids=[example_id],
                    )
                    records[example_id] = result
        self._record_cache[cand_hash] = records
        return records

    async def get(self, candidate: Candidate, example_id: str) -> EvalResult | None:
        """Fetch a cached result if present."""
        cand_hash = candidate_key(candidate)
        path = self._record_path(cand_hash)
        in_memory = self._record_cache.get(cand_hash)
        if in_memory is not None and example_id in in_memory:
            return in_memory[example_id]
        if not path.exists():
            return None
        lock = self._lock_for(cand_hash)
        async with lock:
            # Use semaphore to limit concurrent file operations
            async with self._file_semaphore:
                records = await asyncio.to_thread(self._ensure_candidate_cache, cand_hash, path)
        return records.get(example_id)

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
            cache = self._record_cache.setdefault(cand_hash, {})
            cache[example_id] = self._clone_result(result, example_id)

    async def batch_set(self, writes: list[tuple[Candidate, str, EvalResult]]) -> None:
        """Batch write multiple results, grouping by candidate for efficiency."""
        from collections import defaultdict

        # Group writes by candidate hash to minimize lock contention
        by_candidate: defaultdict[str, list[tuple[Path, dict[str, object]]]] = defaultdict(list)
        updates: defaultdict[str, list[tuple[str, EvalResult]]] = defaultdict(list)

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
            updates[cand_hash].append((example_id, result))

        async def write_batch(cand_hash: str, records: list[tuple[Path, dict]]) -> None:
            # All records for same candidate go to same file
            path = records[0][0]
            record_objs = [r[1] for r in records]
            lock = self._lock_for(cand_hash)
            async with lock:
                # Use semaphore to limit concurrent file operations
                async with self._file_semaphore:
                    await asyncio.to_thread(self._append_records, path, record_objs)
            cache = self._record_cache.setdefault(cand_hash, {})
            for example_id, res in updates.get(cand_hash, []):
                cache[example_id] = self._clone_result(res, example_id)

        # Write all candidate batches in parallel
        await asyncio.gather(*(write_batch(k, v) for k, v in by_candidate.items()))

    def clear(self) -> None:
        """Remove all cached records (useful for tests)."""
        self._record_cache.clear()
        if not self.cache_dir.exists():
            return
        for root, _dirs, files in os.walk(self.cache_dir, topdown=False):
            for file in files:
                Path(root, file).unlink()
        for root, _dirs, _files in os.walk(self.cache_dir, topdown=False):
            Path(root).rmdir()

    def _read_record(self, path: Path, example_id: str) -> EvalResult | None:
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

    def _append_record(self, path: Path, record: dict[str, object]) -> None:
        """Append a single record with retry on OSError."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with path.open("a", encoding="utf-8", buffering=1) as handle:
                    handle.write(json.dumps(record) + "\n")
                return
            except OSError:
                if attempt < max_attempts - 1:
                    import time

                    time.sleep(0.1 * (2**attempt))
                else:
                    raise  # Re-raise on final attempt

    def _append_records(self, path: Path, records: list[dict[str, object]]) -> None:
        """Batch write multiple records to same file with retry on OSError."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with path.open("a", encoding="utf-8", buffering=1) as handle:
                    for record in records:
                        handle.write(json.dumps(record) + "\n")
                return
            except OSError:
                if attempt < max_attempts - 1:
                    import time

                    time.sleep(0.1 * (2**attempt))
                else:
                    raise  # Re-raise on final attempt

    # State persistence for resumable optimization

    def _state_path(self) -> Path:
        """Path to orchestrator state file."""
        return self.cache_dir / "orchestrator_state.json"

    def save_state(
        self,
        round_num: int,
        evaluations: int,
        pareto_candidates: list[Candidate],
        qd_candidates: list[Candidate],
        queue: list[Candidate],
    ) -> None:
        """
        Save orchestrator state for resumable optimization.

        Atomically writes state to disk so it's safe to interrupt anytime.
        Uses retry logic to handle temporary file system issues.
        """
        state = {
            "round": round_num,
            "evaluations": evaluations,
            "pareto": [self._serialize_candidate(c) for c in pareto_candidates],
            "qd": [self._serialize_candidate(c) for c in qd_candidates],
            "queue": [self._serialize_candidate(c) for c in queue],
        }

        # Atomic write with retry: write to temp file, then rename
        state_path = self._state_path()
        temp_path = state_path.with_suffix(".tmp")

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with temp_path.open("w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                temp_path.replace(state_path)
                return
            except OSError as e:
                if attempt < max_attempts - 1:
                    import time

                    time.sleep(0.1 * (2**attempt))
                else:
                    # On final failure, log warning but don't crash optimization
                    logging.warning("Failed to save state after %d attempts: %s", max_attempts, e)

    def load_state(self) -> dict | None:
        """
        Load saved orchestrator state, or None if no state exists.

        Returns dict with keys: round, evaluations, pareto, qd, queue
        Uses retry logic to handle temporary file system issues.
        """
        state_path = self._state_path()
        if not state_path.exists():
            return None

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with state_path.open("r", encoding="utf-8") as f:
                    state = json.load(f)

                # Deserialize candidates
                state["pareto"] = [self._deserialize_candidate(c) for c in state["pareto"]]
                state["qd"] = [self._deserialize_candidate(c) for c in state["qd"]]
                state["queue"] = [self._deserialize_candidate(c) for c in state["queue"]]

                return state
            except OSError as e:
                if attempt < max_attempts - 1:
                    import time

                    time.sleep(0.1 * (2**attempt))
                else:
                    # On final failure, log warning and return None
                    logging.warning("Failed to load state after %d attempts: %s", max_attempts, e)
                    return None
            except (json.JSONDecodeError, KeyError) as e:
                # Corrupted state file, return None to start fresh
                logging.warning("Corrupted state file, starting fresh: %s", e)
                return None

    def has_state(self) -> bool:
        """Check if saved state exists."""
        return self._state_path().exists()

    def clear_state(self) -> None:
        """Delete saved state file."""
        state_path = self._state_path()
        if state_path.exists():
            state_path.unlink()

    def _serialize_candidate(self, candidate: Candidate) -> dict:
        """Convert Candidate to JSON-serializable dict."""
        return {
            "text": candidate.text,
            "meta": dict(candidate.meta),
        }

    def _deserialize_candidate(self, data: dict) -> Candidate:
        """Reconstruct Candidate from dict."""
        return Candidate(
            text=data["text"],
            meta=data.get("meta", {}),
        )
