"""
Code execution utilities for GEPA optimization.

This module provides safe, sandboxed code execution with timeout support,
stdout/stderr capture, and structured result handling. These utilities are
designed to simplify implementing fitness functions for code evolution tasks.

Example usage:

    from gepa.utils.code_execution import execute_code, ExecutionMode

    # Simple in-process execution (fast, for trusted code)
    result = execute_code(
        code="x = 42\\nprint('hello')",
        timeout=5,
        mode=ExecutionMode.IN_PROCESS,
    )
    print(result.variables.get("x"))  # 42
    print(result.stdout)  # "hello\\n"

    # Subprocess execution (safe, for untrusted code)
    result = execute_code(
        code=user_generated_code,
        timeout=30,
        mode=ExecutionMode.SUBPROCESS,
        global_vars={"input_data": [1, 2, 3]},
    )
    if result.success:
        print(f"Result: {result.variables.get('output')}")
    else:
        print(f"Error: {result.error}")
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Module-level setting for cloudpickle usage (set by optimize_anything.py)
_USE_CLOUDPICKLE: bool = False


def set_use_cloudpickle(value: bool) -> None:
    """Configure whether subprocess execution uses cloudpickle or pickle."""
    global _USE_CLOUDPICKLE
    _USE_CLOUDPICKLE = value


class ExecutionMode(Enum):
    """Execution mode for code execution."""

    IN_PROCESS = "in_process"
    """Execute code in the current process using exec(). Fast but less isolated."""

    SUBPROCESS = "subprocess"
    """Execute code in a separate subprocess. Safer but slower."""


class TimeLimitError(Exception):
    """Raised when code execution exceeds the timeout."""

    pass


@dataclass
class CodeExecutionResult:
    """
    Result of code execution.

    Attributes:
        success: Whether execution completed without errors
        stdout: Captured standard output
        stderr: Captured standard error / logs
        error: Error message if execution failed
        traceback: Full traceback if an exception occurred
        variables: Variables from the execution context (for extracting results)
        execution_time: Time taken for execution in seconds
        code_hash: MD5 hash of the executed code (useful for caching)
    """

    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    traceback: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    code_hash: str = ""

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable from the execution context."""
        return self.variables.get(name, default)

    def to_side_info_dict(self) -> dict[str, Any]:
        """
        Convert execution result to a dict suitable for side_info.

        Returns a dict with standardized keys that can be included in side_info
        for LLM reflection.
        """
        info = {
            "Stdout": self.stdout,
            "Stderr": self.stderr,
        }
        if self.error:
            info["Error"] = self.error
        if self.traceback:
            info["Traceback"] = self.traceback
        return info


def _alarm_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeLimitError("Code execution timed out")


def _compute_code_hash(code: str) -> str:
    """Compute MD5 hash of code for caching/deduplication."""
    normalized = "\n".join(line.rstrip() for line in code.strip().split("\n"))
    return hashlib.md5(normalized.encode()).hexdigest()


def execute_code(
    code: str,
    timeout: float = 30.0,
    mode: ExecutionMode = ExecutionMode.IN_PROCESS,
    global_vars: dict[str, Any] | None = None,
    entry_point: str | None = None,
    entry_point_args: tuple = (),
    entry_point_kwargs: dict[str, Any] | None = None,
    capture_variables: list[str] | None = None,
    seed: int | None = None,
    kill_child_processes: bool = True,
) -> CodeExecutionResult:
    """
    Execute Python code safely with timeout and output capture.

    This function provides a unified interface for executing Python code strings
    with configurable isolation level, timeout handling, and result capture.

    Args:
        code: Python code string to execute
        timeout: Maximum execution time in seconds (default: 30)
        mode: Execution mode - IN_PROCESS (fast) or SUBPROCESS (safe)
        global_vars: Variables to inject into the execution context.
            These are accessible to the code being executed.
        entry_point: Optional function name to call after executing the code.
            If provided, the function will be called and its return value
            stored in variables["__return__"].
        entry_point_args: Positional arguments to pass to entry_point function
        entry_point_kwargs: Keyword arguments to pass to entry_point function
        capture_variables: List of variable names to extract from the execution
            context. If None, all non-private variables are captured.
        seed: Optional random seed to set before execution for reproducibility.
            Sets seeds for random, numpy.random, and torch if available.
        kill_child_processes: If True, attempt to kill any child processes spawned
            by the executed code when a timeout occurs. Requires psutil to be
            installed for full functionality. Default is True.

    Returns:
        CodeExecutionResult with execution outcome, captured output, and variables

    Example:
        # Execute code and extract result variable
        result = execute_code(
            code="def solve(x): return x * 2\\nresult = solve(21)",
            timeout=5,
            capture_variables=["result"],
        )
        print(result.variables["result"])  # 42

        # Execute with entry point function
        result = execute_code(
            code="def optimize(bounds): return sum(bounds) / 2",
            entry_point="optimize",
            entry_point_args=([0, 10],),
        )
        print(result.variables["__return__"])  # 5.0

        # Execute with injected variables
        result = execute_code(
            code="output = input_value * 2",
            global_vars={"input_value": 21},
        )
        print(result.variables["output"])  # 42
    """
    if mode == ExecutionMode.IN_PROCESS:
        return _execute_in_process(
            code=code,
            timeout=timeout,
            global_vars=global_vars,
            entry_point=entry_point,
            entry_point_args=entry_point_args,
            entry_point_kwargs=entry_point_kwargs,
            capture_variables=capture_variables,
            seed=seed,
            kill_child_processes=kill_child_processes,
        )
    else:
        return _execute_subprocess(
            code=code,
            timeout=timeout,
            global_vars=global_vars,
            entry_point=entry_point,
            entry_point_args=entry_point_args,
            entry_point_kwargs=entry_point_kwargs,
            capture_variables=capture_variables,
            seed=seed,
            kill_child_processes=kill_child_processes,
        )


def _set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _kill_child_processes(current_pid: int) -> None:
    """Kill all child processes of the current process using psutil if available."""
    try:
        import psutil  # type: ignore[import-not-found]

        try:
            parent = psutil.Process(current_pid)
            # Kill all children recursively
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
    except ImportError:
        pass  # psutil not available, skip child process cleanup
    except Exception:
        pass  # Ignore errors in cleanup


def _execute_in_process(
    code: str,
    timeout: float,
    global_vars: dict[str, Any] | None,
    entry_point: str | None,
    entry_point_args: tuple,
    entry_point_kwargs: dict[str, Any] | None,
    capture_variables: list[str] | None,
    seed: int | None,
    kill_child_processes: bool,
) -> CodeExecutionResult:
    """Execute code in the current process using exec()."""
    start_time = time.time()
    code_hash = _compute_code_hash(code)
    current_pid = os.getpid()

    # Set up execution context
    context: dict[str, Any] = {"__name__": "__main__"}
    if global_vars:
        context.update(global_vars)

    # Set random seeds if specified
    if seed is not None:
        _set_random_seeds(seed)

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    error = ""
    tb = ""
    success = True

    # Set up timeout using signal (Unix only)
    old_handler = None
    has_signal = hasattr(signal, "SIGALRM")

    if timeout > 0 and has_signal:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        # Use setitimer for sub-second precision if available
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)
        except AttributeError:
            signal.alarm(int(timeout))

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, context)

            # Call entry point if specified
            if entry_point and entry_point in context:
                kwargs = entry_point_kwargs or {}
                result = context[entry_point](*entry_point_args, **kwargs)
                context["__return__"] = result

    except TimeLimitError:
        error = f"TimeLimitError: Code execution exceeded {timeout} seconds."
        success = False
        # Try to kill any child processes spawned by the code
        if kill_child_processes:
            _kill_child_processes(current_pid)
    except Exception as e:
        error = str(e)
        tb = traceback.format_exc()
        success = False
    finally:
        # Disable timeout
        if timeout > 0 and has_signal:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except AttributeError:
                signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

        # Double-check: if we've exceeded timeout but no exception was raised, kill children
        elapsed = time.time() - start_time
        if kill_child_processes and timeout > 0 and elapsed > timeout and not error:
            error = f"TimeLimitError: Code execution exceeded {timeout} seconds (elapsed: {elapsed:.2f}s)."
            success = False
            _kill_child_processes(current_pid)

    execution_time = time.time() - start_time

    # Extract variables
    if capture_variables:
        variables = {k: context.get(k) for k in capture_variables}
    else:
        # Capture all variables except Python dunder variables (e.g., __name__, __builtins__)
        # Note: User-defined private variables (single _) are preserved for backwards compatibility
        variables = {
            k: v for k, v in context.items() if not k.startswith("__")
        }

    # Always include __return__ if it exists
    if "__return__" in context:
        variables["__return__"] = context["__return__"]

    return CodeExecutionResult(
        success=success,
        stdout=stdout_capture.getvalue(),
        stderr=stderr_capture.getvalue(),
        error=error,
        traceback=tb,
        variables=variables,
        execution_time=execution_time,
        code_hash=code_hash,
    )


def _execute_subprocess(
    code: str,
    timeout: float,
    global_vars: dict[str, Any] | None,
    entry_point: str | None,
    entry_point_args: tuple,
    entry_point_kwargs: dict[str, Any] | None,
    capture_variables: list[str] | None,
    seed: int | None,
    kill_child_processes: bool,
) -> CodeExecutionResult:
    """Execute code in a separate subprocess for isolation."""
    start_time = time.time()
    code_hash = _compute_code_hash(code)

    # Create temp files for communication
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as args_file:
        args_path = args_file.name
        if _USE_CLOUDPICKLE:
            import cloudpickle
            cloudpickle.dump(
                {
                    "code": code,  # Pass code through pickle to avoid escaping issues
                    "global_vars": global_vars or {},
                    "entry_point": entry_point,
                    "entry_point_args": entry_point_args,
                    "entry_point_kwargs": entry_point_kwargs or {},
                    "capture_variables": capture_variables,
                    "seed": seed,
                },
                args_file,
            )
        else:
            pickle.dump(
                {
                    "code": code,
                    "global_vars": global_vars or {},
                    "entry_point": entry_point,
                    "entry_point_args": entry_point_args,
                    "entry_point_kwargs": entry_point_kwargs or {},
                    "capture_variables": capture_variables,
                    "seed": seed,
                },
                args_file,
            )

    results_path = args_path + ".results"

    # Choose pickle module for subprocess
    pickle_import = "import cloudpickle as _pickle" if _USE_CLOUDPICKLE else "import pickle as _pickle"

    # Build wrapper script
    wrapper_script = f"""
import sys
{pickle_import}
import traceback
import random

# Load arguments
with open({args_path!r}, 'rb') as f:
    _args = _pickle.load(f)

_code = _args['code']
_global_vars = _args['global_vars']
_entry_point = _args['entry_point']
_entry_point_args = _args['entry_point_args']
_entry_point_kwargs = _args['entry_point_kwargs']
_capture_variables = _args['capture_variables']
_seed = _args['seed']

# Set random seeds
if _seed is not None:
    random.seed(_seed)
    try:
        import numpy as np
        np.random.seed(_seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
    except ImportError:
        pass

# Set up execution context
_context = {{"__name__": "__main__"}}
_context.update(_global_vars)

_output = {{"success": False, "error": "Code did not complete"}}

try:
    # Execute user code (loaded from pickle to avoid escaping issues)
    exec(_code, _context)

    # Call entry point if specified
    if _entry_point and _entry_point in _context:
        _result = _context[_entry_point](*_entry_point_args, **_entry_point_kwargs)
        _context["__return__"] = _result

    # Extract variables (exclude Python dunder variables, but keep user private vars)
    import types
    def _is_picklable(v):
        return not isinstance(v, (types.ModuleType, types.FunctionType, type))

    if _capture_variables:
        _variables = {{k: _context.get(k) for k in _capture_variables if _is_picklable(_context.get(k))}}
    else:
        _variables = {{k: v for k, v in _context.items() if not k.startswith("__") and _is_picklable(v)}}

    if "__return__" in _context:
        _variables["__return__"] = _context["__return__"]

    _output = {{"success": True, "variables": _variables}}

except Exception as _e:
    _output = {{
        "success": False,
        "error": str(_e),
        "traceback": traceback.format_exc(),
    }}

with open({results_path!r}, 'wb') as f:
    _pickle.dump(_output, f)
"""

    # Write wrapper script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as script_file:
        script_path = script_file.name
        script_file.write(wrapper_script)

    try:
        # Find Python executable
        python_executable = sys.executable
        venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
        if os.path.exists(venv_python):
            python_executable = venv_python

        # Run subprocess
        process = subprocess.Popen(
            [python_executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            execution_time = time.time() - start_time

            # Load results
            if os.path.exists(results_path):
                try:
                    if _USE_CLOUDPICKLE:
                        import cloudpickle
                        with open(results_path, "rb") as f:
                            output = cloudpickle.load(f)
                    else:
                        with open(results_path, "rb") as f:
                            output = pickle.load(f)
                except (EOFError, Exception) as e:
                    return CodeExecutionResult(
                        success=False,
                        stdout=stdout_str,
                        stderr=stderr_str,
                        error=f"Failed to load results: {e}",
                        traceback=stderr_str,
                        execution_time=execution_time,
                        code_hash=code_hash,
                    )

                return CodeExecutionResult(
                    success=output.get("success", False),
                    stdout=stdout_str,
                    stderr=stderr_str,
                    error=output.get("error", ""),
                    traceback=output.get("traceback", ""),
                    variables=output.get("variables", {}),
                    execution_time=execution_time,
                    code_hash=code_hash,
                )
            else:
                return CodeExecutionResult(
                    success=False,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    error="Script crashed before completing",
                    traceback=stderr_str,
                    execution_time=execution_time,
                    code_hash=code_hash,
                )

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            execution_time = time.time() - start_time

            # Try to kill child processes if requested and psutil available
            if kill_child_processes:
                try:
                    import psutil  # type: ignore[import-not-found]

                    try:
                        parent = psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        pass
                except ImportError:
                    pass

            return CodeExecutionResult(
                success=False,
                error=f"Timeout: execution exceeded {timeout} seconds",
                execution_time=execution_time,
                code_hash=code_hash,
            )

    finally:
        # Cleanup temp files
        for path in [args_path, results_path, script_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


def get_code_hash(code: str, length: int = 8) -> str:
    """
    Get a hash of code for caching/deduplication.

    The hash is computed from normalized code (trailing whitespace removed).

    Args:
        code: Python code string
        length: Number of characters to return from the hash (default: 8)

    Returns:
        MD5 hash string (first `length` characters)
    """
    return _compute_code_hash(code)[:length]
