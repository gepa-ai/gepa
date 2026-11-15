"""
Unified cost tracker for GEPA optimization.
Uses LiteLLM's built-in cost tracking for accuracy.
Thread-safe for parallel execution.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

import litellm


class UnifiedCostTracker:
    """
    Thread-safe cost tracker that uses LiteLLM's built-in cost calculation.
    Tracks agent and reflection costs separately.
    """

    def __init__(self, log_dir: str = "gepa_results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_start = datetime.now()
        self._lock = threading.Lock()

        # Counters
        self.total_cost = 0.0
        self.agent_cost = 0.0
        self.reflection_cost = 0.0
        self.call_count = 0
        self.agent_calls = 0
        self.reflection_calls = 0
        self.total_tokens = 0

        # Log files
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"cost_log_{timestamp}.jsonl"
        self.summary_file = self.log_dir / "cost_summary.txt"

        # Register LiteLLM callback
        self._register_callback()

    def _register_callback(self):
        """Register ourselves as a LiteLLM success callback."""
        # Remove any existing callback to avoid duplicates
        litellm.success_callback = [self._on_completion]

    def _on_completion(self, kwargs, completion_response, start_time, end_time):
        """Called by LiteLLM after each successful completion."""
        with self._lock:
            self.call_count += 1
            call_num = self.call_count

            # Get model name
            model = kwargs.get("model", "unknown")

            # Calculate cost using LiteLLM's built-in function
            try:
                cost = litellm.completion_cost(completion_response=completion_response)
            except Exception:
                cost = 0.0

            # Get token usage
            tokens = 0
            if hasattr(completion_response, "usage") and completion_response.usage:
                tokens = getattr(completion_response.usage, "total_tokens", 0)
            self.total_tokens += tokens

            # Categorize: reflection models typically have "pro" in name
            is_reflection = "pro" in model.lower() or "reflection" in kwargs.get("metadata", {}).get("type", "")

            if is_reflection:
                self.reflection_cost += cost
                self.reflection_calls += 1
                call_type = "REFLECTION"
            else:
                self.agent_cost += cost
                self.agent_calls += 1
                call_type = "AGENT"

            self.total_cost += cost

            # Log entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "call_number": call_num,
                "type": call_type,
                "model": model,
                "tokens": tokens,
                "cost": cost,
                "cumulative_cost": self.total_cost,
                "agent_cost": self.agent_cost,
                "reflection_cost": self.reflection_cost,
            }

            # Write to log file
            try:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass

            # Cost tracking is silent - check log file or call print_summary() for details

    def write_summary(self) -> str:
        """Write human-readable summary to file and return it."""
        with self._lock:
            elapsed = (datetime.now() - self.session_start).total_seconds()
            elapsed_min = elapsed / 60

            summary = f"""
================================================================================
GEPA COST TRACKER - Experiment Summary
================================================================================

Session Start: {self.session_start.strftime("%Y-%m-%d %H:%M:%S")}
Elapsed Time:  {elapsed_min:.1f} minutes

COST BREAKDOWN
--------------
Agent Calls:      {self.agent_calls:,} calls  |  ${self.agent_cost:.4f}
Reflection Calls: {self.reflection_calls:,} calls  |  ${self.reflection_cost:.4f}
Total Calls:      {self.call_count:,} calls  |  ${self.total_cost:.4f}

Total Tokens: {self.total_tokens:,}

RATES
-----
Cost Rate:  ${self.total_cost / elapsed_min:.4f}/min
Call Rate:  {self.call_count / elapsed_min:.1f} calls/min

PER-UNIT ESTIMATES
------------------
Per Agent Call:     ${self.agent_cost / max(self.agent_calls, 1):.4f}
Per Reflection Call: ${self.reflection_cost / max(self.reflection_calls, 1):.4f}

Log file: {self.log_file}
================================================================================
"""
            with open(self.summary_file, "w") as f:
                f.write(summary)

            return summary

    def get_stats(self) -> dict:
        """Get current stats as a dictionary."""
        with self._lock:
            elapsed = (datetime.now() - self.session_start).total_seconds()
            return {
                "total_cost": self.total_cost,
                "agent_cost": self.agent_cost,
                "reflection_cost": self.reflection_cost,
                "total_calls": self.call_count,
                "agent_calls": self.agent_calls,
                "reflection_calls": self.reflection_calls,
                "total_tokens": self.total_tokens,
                "elapsed_seconds": elapsed,
                "cost_per_minute": self.total_cost / (elapsed / 60) if elapsed > 0 else 0,
            }

    def print_summary(self):
        """Print and save final summary."""
        summary = self.write_summary()
        print(summary)


# Global tracker instance
_tracker: UnifiedCostTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker(log_dir: str = "gepa_results/logs") -> UnifiedCostTracker:
    """Get or create global tracker instance."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = UnifiedCostTracker(log_dir=log_dir)
        return _tracker


def reset_tracker(log_dir: str = "gepa_results/logs") -> UnifiedCostTracker:
    """Reset and create a new tracker instance."""
    global _tracker
    with _tracker_lock:
        _tracker = UnifiedCostTracker(log_dir=log_dir)
        return _tracker


# For backwards compatibility
def log_call(model_name, input_tokens, output_tokens, operation="unknown"):
    """Legacy function - now a no-op since LiteLLM callbacks handle tracking."""
    pass  # Tracking is now automatic via LiteLLM callbacks
