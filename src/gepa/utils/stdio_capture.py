"""
Thread-safe stdout/stderr capture utilities for GEPA evaluation.

When ``capture_stdio=True`` is set in :class:`EngineConfig`, these utilities
replace ``sys.stdout`` and ``sys.stderr`` with per-thread wrappers so that
print output produced inside an evaluator is captured without polluting the
main process output or leaking between concurrent evaluations.

Classes:
    ThreadLocalStreamCapture: A ``sys.stdout``/``sys.stderr`` replacement
        that routes writes to a private buffer on threads that have opted in,
        while passing all other threads through to the original stream.
    StreamCaptureManager: Reference-counted manager that installs/removes the
        capture wrappers, allowing multiple concurrent ``optimize_anything``
        calls to share them safely.
"""

import io
import sys
import threading
from typing import Any


class ThreadLocalStreamCapture:
    """A ``sys.stdout`` / ``sys.stderr`` replacement that captures output per-thread.

    Threads that have called :meth:`start_capture` get their writes routed to a
    private ``StringIO``; all other threads pass through to the original stream.
    """

    def __init__(self, original: Any) -> None:
        self._original = original
        self._local = threading.local()

    # -- file-like interface --------------------------------------------------

    def write(self, text: str) -> int:
        if getattr(self._local, "capturing", False):
            return self._local.buffer.write(text)
        return self._original.write(text)

    def flush(self) -> None:
        if getattr(self._local, "capturing", False):
            self._local.buffer.flush()
        self._original.flush()

    def fileno(self) -> int:
        # Always delegate to the original stream so that libraries (tqdm, rich,
        # subprocess, logging handlers, etc.) that call sys.stdout.fileno()
        # continue to work even while this thread is being captured.
        return self._original.fileno()

    @property
    def encoding(self) -> str:
        return self._original.encoding

    @property
    def errors(self) -> str | None:
        return self._original.errors

    def isatty(self) -> bool:
        if getattr(self._local, "capturing", False):
            return False
        return self._original.isatty()

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    # -- capture control ------------------------------------------------------

    def start_capture(self) -> None:
        """Start capturing for the current thread."""
        assert not getattr(self._local, "capturing", False), (
            "start_capture() called while already capturing on this thread. "
            "Call stop_capture() first to retrieve the buffered output."
        )
        self._local.capturing = True
        self._local.buffer = io.StringIO()

    def stop_capture(self) -> str:
        """Stop capturing and return the captured text for the current thread.

        Safe to call even if :meth:`start_capture` was never called on this
        thread — returns an empty string in that case.
        """
        if not getattr(self._local, "capturing", False):
            return ""
        self._local.capturing = False
        text = self._local.buffer.getvalue()
        self._local.buffer = io.StringIO()
        return text


class StreamCaptureManager:
    """Reference-counted manager for per-thread stdout/stderr capture.

    Allows multiple concurrent optimize_anything calls with capture_stdio=True
    to share the same stream wrappers. sys.stdout/stderr are only restored when
    the last user releases.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._refcount = 0
        self._stdout_capturer: ThreadLocalStreamCapture | None = None
        self._stderr_capturer: ThreadLocalStreamCapture | None = None
        self._original_stdout: Any = None
        self._original_stderr: Any = None

    def acquire(self) -> tuple[ThreadLocalStreamCapture, ThreadLocalStreamCapture]:
        """Install capture wrappers (or reuse existing). Returns (stdout_cap, stderr_cap)."""
        with self._lock:
            if self._refcount == 0:
                self._original_stdout = sys.stdout
                self._original_stderr = sys.stderr
                self._stdout_capturer = ThreadLocalStreamCapture(sys.stdout)
                self._stderr_capturer = ThreadLocalStreamCapture(sys.stderr)
                sys.stdout = self._stdout_capturer  # type: ignore[assignment]
                sys.stderr = self._stderr_capturer  # type: ignore[assignment]
            self._refcount += 1
            assert self._stdout_capturer is not None and self._stderr_capturer is not None
            return self._stdout_capturer, self._stderr_capturer

    def release(self) -> None:
        """Decrement ref count. Restores original streams when last user releases."""
        with self._lock:
            self._refcount -= 1
            if self._refcount <= 0:
                if self._original_stdout is not None:
                    sys.stdout = self._original_stdout
                if self._original_stderr is not None:
                    sys.stderr = self._original_stderr
                self._stdout_capturer = None
                self._stderr_capturer = None
                self._refcount = 0


# Module-level singleton shared across all optimize_anything calls.
stream_manager = StreamCaptureManager()
