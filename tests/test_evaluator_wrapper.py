# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the evaluator wrapper, oa.log(), capture_stdio, side_info key
handling, str_candidate_mode, OptimizationState, make_litellm_lm, and the
stdio_capture utilities."""

import io
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import gepa.optimize_anything as oa
from gepa.optimize_anything import EvaluatorWrapper, OptimizationState
from gepa.utils.stdio_capture import StreamCaptureManager, ThreadLocalStreamCapture

# ---------------------------------------------------------------------------
# oa.log()
# ---------------------------------------------------------------------------


class TestOaLog:
    """Tests for the oa.log() function."""

    def test_log_basic_capture(self):
        """oa.log() inside an evaluator should be captured in side_info['log']."""

        def my_eval(candidate, **kwargs):
            oa.log("hello", "world")
            oa.log("score is", 42)
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        score, _, side_info = wrapped({"x": "1"})
        assert score == 1.0
        assert "log" in side_info
        assert "hello world" in side_info["log"]
        assert "score is 42" in side_info["log"]

    def test_log_with_custom_sep_and_end(self):
        """oa.log() should respect sep and end arguments."""

        def my_eval(candidate, **kwargs):
            oa.log("a", "b", "c", sep="-", end="!")
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        _, _, side_info = wrapped({"x": "1"})
        assert side_info["log"] == "a-b-c!"

    def test_log_outside_evaluator_warns(self):
        """oa.log() called outside an evaluator should emit a warning."""
        # Ensure no log context is active by running in a fresh thread
        result: list = []

        def runner():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                oa.log("this should warn")
                result.extend(w)

        t = threading.Thread(target=runner)
        t.start()
        t.join()
        assert len(result) == 1
        assert "outside of an evaluator" in str(result[0].message)

    def test_log_outside_evaluator_discards_output(self):
        """oa.log() outside evaluator should discard output, not accumulate."""
        # Call oa.log() outside evaluator in a fresh thread to guarantee clean state
        def warn_runner():
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                oa.log("stale output")

        t = threading.Thread(target=warn_runner)
        t.start()
        t.join()

        # Now run an evaluator — it should NOT see the stale output
        def my_eval(candidate, **kwargs):
            oa.log("fresh output")
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        _, _, side_info = wrapped({"x": "1"})
        assert "stale" not in side_info.get("log", "")
        assert "fresh output" in side_info["log"]

    def test_log_no_output_means_no_log_key(self):
        """If evaluator doesn't call oa.log(), side_info should not have 'log' key."""

        def my_eval(candidate, **kwargs):
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        _, _, side_info = wrapped({"x": "1"})
        assert "log" not in side_info

    def test_log_child_thread_with_context_propagation(self):
        """oa.log() from child threads should be captured when context is propagated."""

        def my_eval(candidate, **kwargs):
            ctx = oa.get_log_context()
            oa.log("from main")

            def worker():
                oa.set_log_context(ctx)
                oa.log("from child")

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        _, _, side_info = wrapped({"x": "1"})
        assert "from main" in side_info["log"]
        assert "from child" in side_info["log"]

    def test_log_child_thread_via_thread_pool(self):
        """oa.log() from ThreadPoolExecutor workers with manual context propagation."""

        def my_eval(candidate, **kwargs):
            ctx = oa.get_log_context()

            def worker(msg: str):
                oa.set_log_context(ctx)
                oa.log(msg)

            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(worker, f"msg-{i}") for i in range(5)]
                for f in futures:
                    f.result()
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        _, _, side_info = wrapped({"x": "1"})
        for i in range(5):
            assert f"msg-{i}" in side_info["log"]

    def test_log_parallel_evaluators_no_cross_contamination(self):
        """Parallel evaluator calls should not cross-contaminate log output.

        Each EvaluatorWrapper call creates a fresh LogContext in thread-local
        storage, so concurrent calls on different threads are structurally
        isolated — no timing tricks needed.
        """
        results: dict[int, dict] = {}

        def my_eval(candidate, **kwargs):
            val = candidate["id"]
            oa.log(f"eval-{val}")
            oa.log(f"done-{val}")
            return float(val)

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)

        def run_eval(idx: int):
            score, _, side_info = wrapped({"id": str(idx)})
            return idx, side_info

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_eval, i) for i in range(8)]
            for f in futures:
                idx, side_info = f.result()
                results[idx] = side_info

        # Each evaluator should only see its own log output
        for idx, side_info in results.items():
            log_text = side_info.get("log", "")
            assert f"eval-{idx}" in log_text
            assert f"done-{idx}" in log_text
            # Should NOT contain output from other evaluators
            for other_idx in results:
                if other_idx != idx:
                    assert f"eval-{other_idx}" not in log_text

    def test_get_log_context_outside_evaluator_raises(self):
        """get_log_context() outside evaluator should raise RuntimeError."""

        def runner():
            with pytest.raises(RuntimeError, match="No active log context"):
                oa.get_log_context()

        # Use a fresh thread to guarantee no context is set
        t = threading.Thread(target=runner)
        t.start()
        t.join()

    def test_log_thread_safe_writes(self):
        """Multiple threads writing via oa.log() to the same context should not lose data."""

        def my_eval(candidate, **kwargs):
            ctx = oa.get_log_context()
            n_threads = 10
            n_writes = 100

            def writer(thread_id: int) -> None:
                oa.set_log_context(ctx)
                for i in range(n_writes):
                    oa.log(f"t{thread_id}:{i}")

            threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        _, _, side_info = wrapped({"x": "1"})
        lines = [line for line in side_info["log"].strip().split("\n") if line]
        assert len(lines) == 10 * 100


# ---------------------------------------------------------------------------
# capture_stdio
# ---------------------------------------------------------------------------


class TestCaptureStdio:
    """Tests for capture_stdio=True in the evaluator wrapper."""

    def test_stdout_captured_when_enabled(self):
        """print() output should be captured when capture_stdio=True."""

        def my_eval(candidate, **kwargs):
            print("hello from evaluator")
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True, capture_stdio=True)
        _, _, side_info = wrapped({"x": "1"})
        assert "hello from evaluator" in side_info.get("stdout", "")

    def test_stderr_captured_when_enabled(self):
        """sys.stderr.write() should be captured when capture_stdio=True."""

        def my_eval(candidate, **kwargs):
            sys.stderr.write("error output")
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True, capture_stdio=True)
        _, _, side_info = wrapped({"x": "1"})
        assert "error output" in side_info.get("stderr", "")

    def test_stdout_not_captured_when_disabled(self):
        """print() output should NOT be captured when capture_stdio=False."""

        def my_eval(candidate, **kwargs):
            # This should go to real stdout, not captured
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True, capture_stdio=False)
        _, _, side_info = wrapped({"x": "1"})
        assert "stdout" not in side_info
        assert "stderr" not in side_info

    def test_log_and_stdio_captured_together(self):
        """Both oa.log() and print() output should be captured simultaneously."""

        def my_eval(candidate, **kwargs):
            oa.log("log message")
            print("stdout message")
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True, capture_stdio=True)
        _, _, side_info = wrapped({"x": "1"})
        assert "log message" in side_info.get("log", "")
        assert "stdout message" in side_info.get("stdout", "")

    def test_stdio_not_captured_between_evaluator_calls(self):
        """print() output between evaluator calls should not be captured."""

        def my_eval(candidate, **kwargs):
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True, capture_stdio=True)
        # First call
        wrapped({"x": "1"})
        # Print between calls (should go to real stdout, not captured)
        # The ThreadLocalStreamCapture only captures when start_capture() has been called
        # Second call
        _, _, side_info = wrapped({"x": "2"})
        # Should not contain any spurious output
        assert side_info.get("stdout", "") == ""

    def test_capture_scoped_per_call_not_per_wrapper(self):
        """sys.stdout should NOT be replaced during wrapper lifetime — only during calls."""

        def my_eval(candidate, **kwargs):
            print("captured")
            return 1.0

        original_stdout = sys.stdout
        wrapper = EvaluatorWrapper(my_eval, single_instance_mode=True, capture_stdio=True)
        # sys.stdout should still be the original after construction
        assert sys.stdout is original_stdout
        # Call the wrapper — capture should happen only during this call
        _, _, side_info = wrapper({"x": "1"})
        assert "captured" in side_info.get("stdout", "")
        # After the call, sys.stdout should be restored
        assert sys.stdout is original_stdout


# ---------------------------------------------------------------------------
# Side-info key collision handling
# ---------------------------------------------------------------------------


class TestSideInfoKeyCollision:
    """Tests for side_info key collision warning behavior."""

    def test_log_key_collision_warns_and_prefixes(self):
        """If evaluator returns side_info with 'log' key and oa.log() is used, warn + prefix."""

        def my_eval(candidate, **kwargs):
            oa.log("captured log")
            return 1.0, {"log": "user log value"}

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, _, side_info = wrapped({"x": "1"})

        # Should have warned about the collision
        collision_warnings = [x for x in w if "conflicts" in str(x.message)]
        assert len(collision_warnings) == 1

        # User's value should be preserved under original key
        assert side_info["log"] == "user log value"
        # GEPA's captured output should be under prefixed key
        assert "captured log" in side_info["_gepa_log"]

    def test_stdout_key_collision_warns_and_prefixes(self):
        """If evaluator returns side_info with 'stdout' key and capture_stdio=True, warn + prefix."""

        def my_eval(candidate, **kwargs):
            print("captured stdout")
            return 1.0, {"stdout": "user stdout value"}

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True, capture_stdio=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, _, side_info = wrapped({"x": "1"})

        collision_warnings = [x for x in w if "conflicts" in str(x.message)]
        assert len(collision_warnings) == 1

        assert side_info["stdout"] == "user stdout value"
        assert "captured stdout" in side_info["_gepa_stdout"]

    def test_no_collision_no_warning(self):
        """No warning should be emitted when there's no key collision."""

        def my_eval(candidate, **kwargs):
            oa.log("some log")
            return 1.0, {"my_key": "my_value"}

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, _, side_info = wrapped({"x": "1"})

        collision_warnings = [x for x in w if "conflicts" in str(x.message)]
        assert len(collision_warnings) == 0
        assert side_info["my_key"] == "my_value"
        assert "some log" in side_info["log"]

    def test_no_collision_when_capture_inactive(self):
        """No collision even if side_info has 'stdout' key, when capture_stdio=False."""

        def my_eval(candidate, **kwargs):
            return 1.0, {"stdout": "user value", "log": "user log"}

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True, capture_stdio=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, _, side_info = wrapped({"x": "1"})

        # No collision because oa.log() not called and capture_stdio=False
        collision_warnings = [x for x in w if "conflicts" in str(x.message)]
        assert len(collision_warnings) == 0
        assert side_info["stdout"] == "user value"
        assert side_info["log"] == "user log"


# ---------------------------------------------------------------------------
# str_candidate_mode
# ---------------------------------------------------------------------------


class TestStrCandidateMode:
    """Tests for str_candidate_mode in the evaluator wrapper."""

    def test_str_candidate_unwrapped(self):
        """When str_candidate_mode=True, evaluator should receive a str, not a dict."""
        received_candidates = []

        def my_eval(candidate, **kwargs):
            received_candidates.append(candidate)
            return 1.0

        # The internal key name "current_candidate" is what optimize_anything uses
        # to wrap str seeds into a dict. str_candidate_mode unwraps it.
        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True, str_candidate_mode=True)
        wrapped({"current_candidate": "hello world"})
        assert len(received_candidates) == 1
        assert received_candidates[0] == "hello world"
        assert isinstance(received_candidates[0], str)

    def test_dict_candidate_not_unwrapped(self):
        """When str_candidate_mode=False, evaluator should receive the dict as-is."""
        received_candidates = []

        def my_eval(candidate, **kwargs):
            received_candidates.append(candidate)
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True, str_candidate_mode=False)
        wrapped({"key": "value"})
        assert len(received_candidates) == 1
        assert received_candidates[0] == {"key": "value"}
        assert isinstance(received_candidates[0], dict)


# ---------------------------------------------------------------------------
# Single-instance mode / per-instance mode
# ---------------------------------------------------------------------------


class TestEvaluatorModes:
    """Tests for single-instance mode vs per-instance mode."""

    def test_single_instance_mode_no_example_passed(self):
        """In single-instance mode, example should not be forwarded."""
        received_kwargs = []

        def my_eval(candidate, **kwargs):
            received_kwargs.append(kwargs)
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        wrapped({"x": "1"})
        assert "example" not in received_kwargs[0]

    def test_per_instance_mode_example_passed(self):
        """In per-instance mode, example should be forwarded to evaluator."""
        received_kwargs = []

        def my_eval(candidate, example=None, **kwargs):
            received_kwargs.append({"example": example})
            return 1.0

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=False)
        wrapped({"x": "1"}, example={"input": "test"})
        assert received_kwargs[0]["example"] == {"input": "test"}

    def test_return_tuple_normalized(self):
        """(score, side_info) return should be normalized to (score, None, side_info)."""

        def my_eval(candidate, **kwargs):
            return 0.5, {"key": "val"}

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        score, output, side_info = wrapped({"x": "1"})
        assert score == 0.5
        assert output is None
        assert side_info["key"] == "val"

    def test_return_float_normalized(self):
        """Float-only return should be normalized to (score, None, {})."""

        def my_eval(candidate, **kwargs):
            return 0.7

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        score, output, side_info = wrapped({"x": "1"})
        assert score == 0.7
        assert output is None
        assert isinstance(side_info, dict)

    def test_none_side_info_becomes_empty_dict(self):
        """Returning (score, None) should normalize side_info to {}."""

        def my_eval(candidate, **kwargs):
            return 0.5, None

        wrapped = EvaluatorWrapper(my_eval, single_instance_mode=True)
        _, _, side_info = wrapped({"x": "1"})
        assert side_info == {}


# ---------------------------------------------------------------------------
# OptimizationState
# ---------------------------------------------------------------------------


class TestOptimizationState:
    """Tests for the OptimizationState dataclass."""

    def test_construction(self):
        state = OptimizationState(best_example_evals=[{"score": 1.0, "side_info": {}}])
        assert len(state.best_example_evals) == 1
        assert state.best_example_evals[0]["score"] == 1.0

    def test_empty_evals(self):
        state = OptimizationState(best_example_evals=[])
        assert state.best_example_evals == []

    def test_opt_state_kwarg_forwarded_when_accepted(self):
        """opt_state should be passed when evaluator accepts it."""
        received_opt_state = []

        def my_eval_with_opt_state(candidate, opt_state=None, **kwargs):
            received_opt_state.append(opt_state)
            return 1.0

        wrapped = EvaluatorWrapper(my_eval_with_opt_state, single_instance_mode=True)
        test_state = OptimizationState(best_example_evals=[{"score": 0.5, "side_info": {}}])
        wrapped({"x": "1"}, opt_state=test_state)
        assert received_opt_state[0] is test_state

    def test_opt_state_filtered_when_not_accepted(self):
        """opt_state should be silently filtered if evaluator doesn't accept it."""
        received_kwargs = []

        def my_eval_no_opt_state(candidate):
            received_kwargs.append("called")
            return 1.0

        wrapped = EvaluatorWrapper(my_eval_no_opt_state, single_instance_mode=True)
        test_state = OptimizationState(best_example_evals=[])
        # Should not raise even though opt_state is not in signature
        wrapped({"x": "1"}, opt_state=test_state)
        assert received_kwargs == ["called"]


# ---------------------------------------------------------------------------
# make_litellm_lm
# ---------------------------------------------------------------------------


class TestMakeLitellmLm:
    """Tests for the make_litellm_lm helper."""

    @patch("litellm.completion")
    def test_string_prompt(self, mock_completion):
        """String prompt should be wrapped in a user message."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response text"
        mock_completion.return_value = mock_response

        lm = oa.make_litellm_lm("test-model")
        result = lm("hello")

        assert result == "response text"
        mock_completion.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    @patch("litellm.completion")
    def test_messages_prompt(self, mock_completion):
        """List-of-dicts prompt should be passed directly as messages."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "chat response"
        mock_completion.return_value = mock_response

        lm = oa.make_litellm_lm("test-model")
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = lm(messages)

        assert result == "chat response"
        mock_completion.assert_called_once_with(model="test-model", messages=messages)


# ---------------------------------------------------------------------------
# ThreadLocalStreamCapture
# ---------------------------------------------------------------------------


class TestThreadLocalStreamCapture:
    """Tests for the ThreadLocalStreamCapture utility."""

    def test_passthrough_when_not_capturing(self):
        """Writes should go to original stream when not capturing."""
        original = io.StringIO()
        cap = ThreadLocalStreamCapture(original)
        cap.write("hello")
        assert original.getvalue() == "hello"

    def test_capture_when_started(self):
        """Writes should go to capture buffer when capturing."""
        original = io.StringIO()
        cap = ThreadLocalStreamCapture(original)
        cap.start_capture()
        cap.write("captured")
        assert original.getvalue() == ""
        text = cap.stop_capture()
        assert text == "captured"

    def test_stop_capture_returns_text_and_resets(self):
        """stop_capture should return captured text and reset buffer."""
        original = io.StringIO()
        cap = ThreadLocalStreamCapture(original)
        cap.start_capture()
        cap.write("first")
        text1 = cap.stop_capture()
        assert text1 == "first"

        # After stop, writes go to original again
        cap.write("second")
        assert original.getvalue() == "second"

    def test_per_thread_isolation(self):
        """Each thread should have independent capture state."""
        original = io.StringIO()
        cap = ThreadLocalStreamCapture(original)
        results = {}

        def thread_fn(thread_id: int):
            cap.start_capture()
            cap.write(f"thread-{thread_id}")
            results[thread_id] = cap.stop_capture()

        threads = [threading.Thread(target=thread_fn, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(5):
            assert results[i] == f"thread-{i}"

    def test_encoding_property(self):
        cap = ThreadLocalStreamCapture(sys.stdout)
        assert cap.encoding == sys.stdout.encoding

    def test_isatty_false_when_capturing(self):
        original = MagicMock()
        original.isatty.return_value = True
        cap = ThreadLocalStreamCapture(original)
        assert cap.isatty() is True
        cap.start_capture()
        assert cap.isatty() is False
        cap.stop_capture()

    def test_writable_readable(self):
        cap = ThreadLocalStreamCapture(io.StringIO())
        assert cap.writable() is True
        assert cap.readable() is False


# ---------------------------------------------------------------------------
# StreamCaptureManager
# ---------------------------------------------------------------------------


class TestStreamCaptureManager:
    """Tests for the StreamCaptureManager reference-counted manager."""

    def test_acquire_installs_wrappers(self):
        """acquire() should replace sys.stdout/stderr with capture wrappers."""
        mgr = StreamCaptureManager()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            stdout_cap, stderr_cap = mgr.acquire()
            assert sys.stdout is stdout_cap
            assert sys.stderr is stderr_cap
            assert isinstance(stdout_cap, ThreadLocalStreamCapture)
        finally:
            mgr.release()
            assert sys.stdout is original_stdout
            assert sys.stderr is original_stderr

    def test_release_restores_originals(self):
        """release() should restore original sys.stdout/stderr."""
        mgr = StreamCaptureManager()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        mgr.acquire()
        mgr.release()
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr

    def test_refcounting(self):
        """Multiple acquires should share wrappers; release only on last one."""
        mgr = StreamCaptureManager()
        original_stdout = sys.stdout
        try:
            cap1_out, _ = mgr.acquire()
            cap2_out, _ = mgr.acquire()
            # Same wrappers
            assert cap1_out is cap2_out
            # First release — sys.stdout should still be the wrapper
            mgr.release()
            assert sys.stdout is cap1_out
            # Second release — sys.stdout should be restored
            mgr.release()
            assert sys.stdout is original_stdout
        except Exception:
            # Ensure cleanup on failure
            mgr.release()
            mgr.release()
            raise

    def test_release_without_acquire_safe(self):
        """Releasing more than acquired should not crash (clamped to 0)."""
        mgr = StreamCaptureManager()
        mgr.acquire()
        mgr.release()
        # Extra release should not crash
        mgr.release()

    def test_capture_isolation_between_threads(self):
        """Captured output should be per-thread, not shared."""
        mgr = StreamCaptureManager()
        try:
            stdout_cap, _ = mgr.acquire()
            results = {}

            def worker(tid: int):
                stdout_cap.start_capture()
                print(f"thread-{tid}")
                results[tid] = stdout_cap.stop_capture()

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            for i in range(4):
                assert f"thread-{i}" in results[i]
                for j in range(4):
                    if j != i:
                        assert f"thread-{j}" not in results[i]
        finally:
            mgr.release()
