"""Claude Code runtime helpers for optimize_anything."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel


def parse_stream_json(raw_output: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse ``claude --output-format stream-json`` JSONL output."""
    import json as _json

    events: list[dict[str, Any]] = []
    result_text = ""
    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        events.append(obj)
        if obj.get("type") == "result":
            result_text = obj.get("result", "")
    return result_text, events


def extract_text_from_stream_json(raw_output: str) -> str:
    """Extract assistant text from ``claude --output-format stream-json`` output."""
    result_text, events = parse_stream_json(raw_output)

    assistant_texts: list[str] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    assistant_texts.append(text)

    if assistant_texts:
        return "\n".join(assistant_texts)
    return result_text


def build_claude_code_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build subprocess environment for nested Claude Code invocation."""
    import os
    import tempfile

    env_source = dict(base_env if base_env is not None else os.environ)

    # Nested Claude Code subprocesses reject these parent-session vars.
    cc_vars = {"CLAUDECODE", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT"}
    env = {k: v for k, v in env_source.items() if k not in cc_vars}

    env["RLM_DEPTH"] = "0"
    env["RLM_BUDGET_FILE"] = os.path.join(tempfile.gettempdir(), f"rlm_budget_{os.getpid()}.txt")

    repo_src_dir = str(Path(__file__).resolve().parents[3])
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_src_dir}:{existing_pythonpath}" if existing_pythonpath else repo_src_dir
    return env


def make_claude_code_lm(
    model: str | None = None,
    max_budget_usd: float | None = None,
    timeout: int = 300,
    add_dirs: list[str] | None = None,
    system_prompt: str | None = None,
    default_system_prompt: str | None = None,
    env_builder: Callable[[dict[str, str] | None], dict[str, str]] = build_claude_code_env,
) -> LanguageModel:
    """Create a :class:`LanguageModel` callable backed by ``claude -p``."""
    import subprocess

    if system_prompt is None and default_system_prompt is None:
        from gepa.adapters.optimize_anything_adapter.claude_code.prompts import CC_RLM_SYSTEM_PROMPT

        default_system_prompt = CC_RLM_SYSTEM_PROMPT

    def _lm(prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, list):
            prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt)

        env = env_builder()
        cmd = ["claude", "-p", "--output-format", "stream-json", "--dangerously-skip-permissions"]
        if model:
            cmd += ["--model", model]
        if max_budget_usd is not None:
            cmd += ["--max-budget-usd", str(max_budget_usd)]

        prompt_to_inject = system_prompt if system_prompt is not None else default_system_prompt
        if prompt_to_inject is not None:
            cmd += ["--append-system-prompt", prompt_to_inject]
        if add_dirs:
            for directory in add_dirs:
                cmd += ["--add-dir", directory]

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode != 0:
            # CLI may put error details in stdout (stream-json) or stderr
            details = result.stderr.strip() or result.stdout.strip()[:500]
            raise RuntimeError(f"Claude Code failed (exit {result.returncode}): {details}")
        return extract_text_from_stream_json(result.stdout)

    return _lm
