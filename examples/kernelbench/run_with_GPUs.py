import re
import os
import threading
import time
import logging
import fasteners
import subprocess
from typing import List, Optional, Dict
from contextlib import contextmanager


def get_idle_gpus(all_gpu_indices: set) -> List[int]:
    """
    Retrieves idle GPU indices (0-7) not used by any processes reported by nvidia-smi.
    """
    try:
        # Run the standard nvidia-smi command
        smi_output = subprocess.check_output(["nvidia-smi"], encoding="utf-8")

        # Split the output to the process section
        process_section = smi_output.split("Processes:")[1]

        # Regular expression to find numbers (GPU Index) at the beginning of process lines
        gpu_index_pattern = re.compile(r"^\|\s+(\d+)\s+.*$", re.MULTILINE)

        # Find all GPU indices that are busy
        busy_gpu_indices = set(map(int, gpu_index_pattern.findall(process_section)))

        # GPUs that are not busy
        idle_gpus = sorted(all_gpu_indices - busy_gpu_indices)

        return idle_gpus

    except subprocess.CalledProcessError as e:
        print("Error while executing nvidia-smi:", e)
        return []
    except Exception as e:
        print("Unexpected error:", e)
        return []


class GPUManager:
    def __init__(self, device_list: List[int], lock_dir: str, logger=None):
        self.device_list = device_list
        self.lock_dir = lock_dir
        self.logger = logger or logging.getLogger(__name__)
        os.makedirs(lock_dir, exist_ok=True)

        # Round-robin counter for load balancing
        self._rr_counter = 0
        self._rr_lock = threading.Lock()

        # Create a mapping from device_id to lock
        self.gpu_locks: Dict[int, fasteners.InterProcessLock] = {
            device_id: fasteners.InterProcessLock(
                os.path.join(lock_dir, f"gpu_{device_id}.lock")
            )
            for device_id in device_list
        }

    def _get_next_start_idx(self) -> int:
        """Get next starting index for round-robin GPU selection."""
        with self._rr_lock:
            idx = self._rr_counter
            self._rr_counter = (self._rr_counter + 1) % len(self.device_list)
            return idx

    @contextmanager
    def acquire(
        self,
        timeout: Optional[float] = None,
        wait_interval: float = 0.1,
        preferred_gpu_seed: Optional[int] = None,
    ):
        """
        Acquire a GPU lock and yield the GPU ID.

        Uses round-robin assignment with file-based locking for coordination.
        Does NOT check nvidia-smi - relies purely on locks for correctness.

        Args:
            timeout: Maximum time to wait for a GPU (None means wait forever)
            wait_interval: Time to wait between acquisition attempts
            preferred_gpu_seed: Deprecated, ignored (use round-robin instead)

        Yields:
            int: The acquired GPU ID

        Raises:
            TimeoutError: If no GPU could be acquired within the timeout period
        """
        start_time = time.time()
        start_idx = self._get_next_start_idx()
        acquired_lock = None
        acquired_device = None

        while timeout is None or time.time() - start_time < timeout:
            # Try GPUs in round-robin order starting from start_idx
            for i in range(len(self.device_list)):
                idx = (start_idx + i) % len(self.device_list)
                device_id = self.device_list[idx]
                gpu_lock = self.gpu_locks[device_id]

                if gpu_lock.acquire(blocking=False):
                    acquired_lock = gpu_lock
                    acquired_device = device_id
                    self.logger.debug(f"Acquired GPU {device_id}")
                    break

            if acquired_lock is not None:
                break

            # If no GPU was available, wait and retry
            time.sleep(wait_interval)

        if acquired_lock is None:
            raise TimeoutError(f"Could not acquire any GPU within {timeout} seconds")

        try:
            yield acquired_device
        finally:
            self.logger.debug(f"Released GPU {acquired_device}")
            try:
                acquired_lock.release()
            except (threading.ThreadError, RuntimeError) as e:
                self.logger.warning(f"Lock release failed for GPU {acquired_device}: {e}")
