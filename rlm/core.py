"""RLM (Recursive Language Model) — spawn sub-agents via claude -p.

Provides recursive, parallel, budget-controlled sub-agent spawning
for Claude Code. Each spawn is a `claude -p` subprocess with model
cascade, depth limits, and shared budget tracking.

Depth is tracked via the RLM_DEPTH env var (incremented per spawn).
Budget is tracked via an atomic file at RLM_BUDGET_FILE.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpawnResult:
    """Result from a sub-agent call."""

    text: str
    model: str
    depth: int
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str | None = None


@dataclass
class RLMConfig:
    """Configuration for the RLM system."""

    max_depth: int = 3
    max_budget_usd: float = 5.0
    default_model: str = "sonnet"
    timeout: int = 300
    budget_file: str | None = None  # Shared budget tracking file
    add_dirs: list[str] = field(default_factory=list)


_budget_lock = threading.Lock()


def _get_current_depth() -> int:
    return int(os.environ.get("RLM_DEPTH", "0"))


def _read_budget_spent(budget_file: str) -> float:
    try:
        return float(Path(budget_file).read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0.0


def _update_budget_spent(budget_file: str, cost: float) -> float:
    """Atomically add cost to budget file. Returns new total."""
    with _budget_lock:
        current = _read_budget_spent(budget_file)
        new_total = current + cost
        Path(budget_file).write_text(str(new_total))
        return new_total


def _parse_stream_json(raw: str) -> tuple[str, float, int]:
    """Extract text, cost, duration from stream-json output."""
    result_text = ""
    cost = 0.0
    duration = 0
    assistant_texts: list[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        if obj.get("type") == "assistant":
            message = obj.get("message", {})
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        assistant_texts.append(text)

        elif obj.get("type") == "result":
            result_text = obj.get("result", "")
            cost = obj.get("total_cost_usd", 0.0)
            duration = obj.get("duration_ms", 0)

    text = "\n".join(assistant_texts) if assistant_texts else result_text
    return text, cost, duration


def spawn(
    prompt: str,
    model: str | None = None,
    config: RLMConfig | None = None,
) -> SpawnResult:
    """Spawn a sub-agent via claude -p.

    Checks depth and budget limits before spawning. The sub-agent
    inherits RLM_DEPTH + 1 so it knows its depth and can recurse further
    (up to max_depth).
    """
    if config is None:
        config = RLMConfig()
    if model is None:
        model = config.default_model

    depth = _get_current_depth()

    # Depth check
    if depth >= config.max_depth:
        return SpawnResult(
            text="",
            model=model,
            depth=depth,
            error=f"Max depth {config.max_depth} reached",
        )

    # Budget check
    if config.budget_file:
        spent = _read_budget_spent(config.budget_file)
        if spent >= config.max_budget_usd:
            return SpawnResult(
                text="",
                model=model,
                depth=depth,
                error=f"Budget exhausted: ${spent:.4f} >= ${config.max_budget_usd:.2f}",
            )

    # Build env: strip CC nesting vars, set depth
    cc_vars = {"CLAUDECODE", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT"}
    env = {k: v for k, v in os.environ.items() if k not in cc_vars}
    env["RLM_DEPTH"] = str(depth + 1)
    if config.budget_file:
        env["RLM_BUDGET_FILE"] = config.budget_file

    # Build command
    cmd = ["claude", "-p", "--output-format", "stream-json", "--model", model]
    cmd += ["--dangerously-skip-permissions"]
    for d in config.add_dirs:
        cmd += ["--add-dir", d]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=config.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return SpawnResult(
            text="",
            model=model,
            depth=depth + 1,
            error=f"Timeout after {config.timeout}s",
        )

    if result.returncode != 0:
        return SpawnResult(
            text="",
            model=model,
            depth=depth + 1,
            error=f"Exit {result.returncode}: {result.stderr[:500]}",
        )

    text, cost, duration = _parse_stream_json(result.stdout)

    # Track budget
    if config.budget_file and cost > 0:
        _update_budget_spent(config.budget_file, cost)

    return SpawnResult(
        text=text,
        model=model,
        depth=depth + 1,
        cost_usd=cost,
        duration_ms=duration,
    )


def spawn_parallel(
    prompts: list[tuple[str, str | None]],
    config: RLMConfig | None = None,
    max_workers: int | None = None,
) -> list[SpawnResult]:
    """Spawn multiple sub-agents in parallel.

    Args:
        prompts: List of (prompt, model) tuples. model=None uses default.
        config: Shared RLM config (budget is tracked across all calls).
        max_workers: Max concurrent sub-agents.

    Returns:
        Results in the same order as prompts.
    """
    if config is None:
        config = RLMConfig()
    if max_workers is None:
        max_workers = len(prompts)

    results: list[SpawnResult | None] = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(spawn, prompt, model, config): i
            for i, (prompt, model) in enumerate(prompts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results  # type: ignore[return-value]
