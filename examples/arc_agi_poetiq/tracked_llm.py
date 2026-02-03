"""Minimal tracked LLM wrapper."""

import time
from dataclasses import dataclass, field
from typing import Any

import dspy
import litellm
from litellm import completion

litellm.suppress_debug_info = True

@dataclass
class TrackedLLM:
    """Simple LLM wrapper that tracks calls, tokens, and costs."""

    model_id: str = "openrouter/google/gemini-3-flash-preview"
    max_llm_calls: int = 20
    reasoning_effort: str | None = None  # "low", "medium", "high" for o1/o3/gemini

    # Tracking
    calls: list[dict] = field(default_factory=list)

    @property
    def remaining_calls(self) -> int:
        return self.max_llm_calls - len(self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.get("total_tokens", 0) for c in self.calls)

    @property
    def total_cost(self) -> float:
        return sum(c.get("cost", 0.0) for c in self.calls)

    def __call__(self, prompt: str, temperature: float = 1.0) -> str:
        """Make an LLM call. Raises if budget exhausted."""
        if len(self.calls) >= self.max_llm_calls:
            raise RuntimeError(f"LLM budget exhausted ({self.max_llm_calls} calls)")

        start = time.time()

        kwargs: dict = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        # Add reasoning effort if specified
        # OpenRouter format: {"reasoning": {"effort": "high"}} for o1/o3/grok
        # For Gemini: {"reasoning": {"max_tokens": N}} for direct token control
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}

        resp = completion(**kwargs)

        duration = time.time() - start
        usage = resp.usage if hasattr(resp, "usage") else {}
        msg = resp.choices[0].message
        content = msg.content or ""

        # Extract reasoning trace if available (from thinking models)
        reasoning = getattr(msg, "reasoning_content", None) or ""


        # Get cost from litellm
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0

        call_data = {
            "prompt": prompt,
            "response": content,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
            "cost": cost,
            "duration": duration,
        }
        if reasoning:
            call_data["reasoning"] = reasoning

        self.calls.append(call_data)

        return content

    def get_side_info(self) -> dict[str, Any]:
        """Get side info for GEPA reflection. Trajectory sampled to max 10 calls."""
        import random

        calls_to_include = self.calls
        sampled = False

        if len(self.calls) > 10:
            calls_to_include = random.sample(self.calls, 10)
            sampled = True

        trajectory = []
        for c in calls_to_include:
            entry = {
                "prompt": c["prompt"],
                "response": c["response"],
                "tokens": c["total_tokens"],
                "cost": c.get("cost", 0.0),
            }
            if c.get("reasoning"):
                entry["reasoning"] = c["reasoning"]
            trajectory.append(entry)

        result = {
            "llm_calls": len(self.calls),
            "llm_budget": self.max_llm_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "trajectory": trajectory,
        }

        if sampled:
            result["trajectory_note"] = f"Randomly sampled 10 of {len(self.calls)} LLM calls"

        return result
