# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import sys
from typing import Protocol


class LoggerProtocol(Protocol):
    def log(self, message: str):
        ...


class StdOutLogger(LoggerProtocol):
    def log(self, message: str):
        print(message)


class Tee:
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
    def __init__(self, filename, mode="a"):
        self.file_handle = open(filename, mode)
        self.file_handle_stderr = open(filename.replace("run_log.", "run_log_stderr."), mode)
        self.modified_sys = False

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

    def log(self, *args, **kwargs):
        if self.modified_sys:
            print(*args, **kwargs)
        else:
            # Emulate print(*args, **kwargs) behavior but write to the file
            print(*args, **kwargs)
            print(*args, file=self.file_handle_stderr, **kwargs)
        self.file_handle.flush()
        self.file_handle_stderr.flush()
