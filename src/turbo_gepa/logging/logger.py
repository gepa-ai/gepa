"""Logger classes for TurboGEPA.

Copied from original GEPA implementation to maintain independence.
"""

import sys
from enum import IntEnum
from typing import Protocol


class LogLevel(IntEnum):
    """Log levels for filtering messages."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LoggerProtocol(Protocol):
    """Protocol for logger implementations."""

    def log(self, message: str, level: LogLevel = LogLevel.INFO):
        """Log a message at the specified level."""
        ...


class StdOutLogger(LoggerProtocol):
    """Logger with configurable log levels."""

    def __init__(self, min_level: LogLevel = LogLevel.WARNING):
        """Initialize logger with minimum level to display.

        Args:
            min_level: Minimum log level to display. Defaults to WARNING,
                      which only shows important messages and the dashboard.
        """
        self.min_level = min_level

    def log(self, message: str, level: LogLevel = LogLevel.INFO):
        """Log a message if it meets the minimum level threshold."""
        if level >= self.min_level:
            print(message)


class QuietLogger(LoggerProtocol):
    """Logger that suppresses all output (for dashboard-only mode)."""

    def log(self, message: str, level: LogLevel = LogLevel.INFO):
        """Suppress log message."""
        pass


class Tee:
    """Write to multiple file-like objects simultaneously."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            if hasattr(f, "flush"):
                f.flush()

    def isatty(self):
        # True if any of the files is a terminal
        return any(hasattr(f, "isatty") and f.isatty() for f in self.files)

    def close(self):
        for f in self.files:
            if hasattr(f, "close"):
                f.close()

    def fileno(self):
        for f in self.files:
            if hasattr(f, "fileno"):
                return f.fileno()
        raise OSError("No underlying file object with fileno")


class Logger(LoggerProtocol):
    """Logger that writes to both stdout and a file."""

    def __init__(self, filename, mode="a", min_level: LogLevel = LogLevel.WARNING):
        self.file_handle = open(filename, mode)
        self.file_handle_stderr = open(filename.replace("run_log.", "run_log_stderr."), mode)
        self.modified_sys = False
        self.min_level = min_level

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = Tee(sys.stdout, self.file_handle)
        sys.stderr = Tee(sys.stderr, self.file_handle_stderr)
        self.modified_sys = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.file_handle.close()
        self.file_handle_stderr.close()
        self.modified_sys = False

    def log(self, message: str, level: LogLevel = LogLevel.INFO):
        # Always write to file, but filter stdout by level
        if self.modified_sys:
            if level >= self.min_level:
                print(message)
            # Always write to file regardless of level
            print(message, file=self.file_handle)
        else:
            # Emulate print behavior but respect level filtering
            if level >= self.min_level:
                print(message)
            print(message, file=self.file_handle_stderr)
        self.file_handle.flush()
        self.file_handle_stderr.flush()
