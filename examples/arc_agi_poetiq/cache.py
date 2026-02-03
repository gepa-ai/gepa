"""Thread/process-safe cache for agent evaluation results."""

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path

import fasteners
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class EvalCache:
    """Cache for agent evaluation results.

    Keyed on (agent_code, problem_id). Stores the full evaluation result.
    Thread/process-safe using file locking and atomic writes.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stats_file = self.cache_dir / "_stats.json"
        # Thread lock protects the process lock (InterProcessLock isn't thread-safe)
        self._thread_lock = threading.Lock()
        self._process_lock = fasteners.InterProcessLock(str(self.cache_dir / "_stats.lock"))

    def _make_key(self, agent_code: str, problem_id: str) -> str:
        data = json.dumps({"agent_code": agent_code, "problem_id": problem_id}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def _update_stats(self, hit: bool) -> None:
        """Atomically update hit/miss statistics."""
        with self._thread_lock:
            with self._process_lock:
                stats = {"hits": 0, "misses": 0}
                if self._stats_file.exists():
                    try:
                        with open(self._stats_file) as f:
                            stats = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        pass

                if hit:
                    stats["hits"] = stats.get("hits", 0) + 1
                else:
                    stats["misses"] = stats.get("misses", 0) + 1

                # Atomic write
                fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(stats, f)
                    os.rename(tmp_path, self._stats_file)
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except FileNotFoundError:
                        pass
                    raise

    def get(self, agent_code: str, problem_id: str) -> dict | None:
        key = self._make_key(agent_code, problem_id)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file) as f:
                result = json.load(f)
            self._update_stats(hit=True)
            return result
        except FileNotFoundError:
            self._update_stats(hit=False)
            return None

    def put(self, agent_code: str, problem_id: str, result: dict) -> None:
        key = self._make_key(agent_code, problem_id)
        cache_file = self.cache_dir / f"{key}.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(result, f, cls=NumpyEncoder)
            os.rename(tmp_path, cache_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def stats(self) -> dict:
        """Get accumulated hit/miss statistics across all processes."""
        with self._thread_lock:
            with self._process_lock:
                if self._stats_file.exists():
                    try:
                        with open(self._stats_file) as f:
                            stats = json.load(f)
                        hits = stats.get("hits", 0)
                        misses = stats.get("misses", 0)
                        total = hits + misses
                        return {
                            "hits": hits,
                            "misses": misses,
                            "hit_rate": hits / total if total > 0 else 0,
                        }
                    except (json.JSONDecodeError, FileNotFoundError):
                        pass
        return {"hits": 0, "misses": 0, "hit_rate": 0}
