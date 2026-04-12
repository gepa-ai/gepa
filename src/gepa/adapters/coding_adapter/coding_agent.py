"""Coding agent backends for proposing code changes."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any, Protocol

from gepa.adapters.coding_adapter.git_repo import GitRepo

logger = logging.getLogger(__name__)


class CodingAgentProtocol(Protocol):
    """Protocol for coding agents that propose code changes."""

    def propose(
        self,
        repo: GitRepo,
        base_branch: str,
        feedback: str,
        objective: str | None,
        background: str | None,
    ) -> bool:
        """Propose code changes on the currently checked-out branch.

        The repo is already checked out to a new branch (forked from the parent
        candidate). The agent should modify files in-place. The caller handles
        committing.

        Args:
            repo: Git repository wrapper (already checked out to the new branch).
            base_branch: The original base branch of the optimization (e.g. "main").
                Useful for computing diffs to understand what has changed so far.
            feedback: Formatted evaluation feedback from the reflective dataset.
            objective: High-level optimization goal.
            background: Domain context and constraints.

        Returns:
            True if any file changes were made, False otherwise.
        """
        ...


BASH_AGENT_SYSTEM_PROMPT = """You are an expert software engineer. Your task is to improve code \
in a git repository based on evaluation feedback.

You have one tool available: `bash` — which executes a shell command in the repository directory \
and returns its output.

Use bash to:
- Read files (cat, head, tail, less)
- Explore the codebase (find, ls, tree, grep)
- Edit files (sed, awk, or write with cat/tee/echo)
- Run commands to verify your changes (e.g. python, pytest, make)

Make changes directly to the files. Do NOT create branches or commits — just edit the code.

When you are done making changes, respond without a tool call to signal completion."""

BASH_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a shell command in the repository directory. Returns stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}


def _execute_bash(command: str, cwd: str, timeout: int = 120) -> str:
    """Execute a bash command and return combined stdout/stderr."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"[exit code: {result.returncode}]")
        return "\n".join(output_parts) if output_parts else "(no output)"
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]"
    except Exception as e:
        return f"[error executing command: {e}]"


class BashCodingAgent:
    """Agentic coding agent that uses an LLM with a bash tool in a loop.

    The agent iteratively:
    1. Receives the objective, evaluation feedback, and repository context
    2. Calls bash commands to explore code, understand issues, and make edits
    3. Continues until it decides it's done (responds without a tool call)

    All file modifications are made directly via bash commands (sed, cat, tee, etc.).
    """

    def __init__(
        self,
        model: str = "openai/gpt-5.1",
        max_iterations: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        bash_timeout: int = 120,
    ) -> None:
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.bash_timeout = bash_timeout

    def propose(
        self,
        repo: GitRepo,
        base_branch: str,
        feedback: str,
        objective: str | None,
        background: str | None,
    ) -> bool:
        import litellm

        # Build the initial user message with all context
        user_parts: list[str] = []

        if objective:
            user_parts.append(f"## Optimization Goal\n{objective}")
        if background:
            user_parts.append(f"## Domain Context\n{background}")

        # Show diff from base branch so the agent knows what's changed
        branch = repo.current_branch()
        try:
            diff = repo.get_diff(base_branch, branch)
            if diff:
                user_parts.append(f"## Changes So Far (diff from {base_branch})\n\n```diff\n{diff}\n```")
        except Exception:
            pass

        user_parts.append(f"## Evaluation Feedback\n\n{feedback}")
        user_parts.append(
            "## Instructions\n\n"
            "Analyze the feedback and improve the code to address the issues. "
            "Use the bash tool to explore the repository and make changes directly to files. "
            f"The repository is at: {repo.repo_path}"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": BASH_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

        for i in range(self.max_iterations):
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=[BASH_TOOL_DEFINITION],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                drop_params=True,
            )

            # litellm types Choices | StreamingChoices; we use non-streaming so it's always Choices
            choice = response.choices[0]  # type: ignore[union-attr]
            message = choice.message  # type: ignore[union-attr]

            # Append assistant message to history
            messages.append(message.model_dump(exclude_none=True))

            # Check if the model wants to call a tool
            if not message.tool_calls:
                # No tool call — agent is done
                logger.info(f"BashCodingAgent finished after {i + 1} iterations")
                break

            # Execute each tool call
            for tool_call in message.tool_calls:
                if tool_call.function.name != "bash":
                    # Unknown tool — return error
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Unknown tool: {tool_call.function.name}. Only 'bash' is available.",
                        }
                    )
                    continue

                try:
                    args = json.loads(tool_call.function.arguments)
                    command = args["command"]
                except (json.JSONDecodeError, KeyError):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Invalid arguments. Expected JSON with 'command' key.",
                        }
                    )
                    continue

                logger.debug(f"BashCodingAgent executing: {command}")
                output = _execute_bash(command, cwd=repo.repo_path, timeout=self.bash_timeout)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                )
        else:
            logger.warning(f"BashCodingAgent hit max iterations ({self.max_iterations})")

        return repo.has_uncommitted_changes()


class ClaudeCodeAgent:
    """Coding agent that delegates to Claude Code CLI (``claude -p``).

    Claude Code autonomously explores the repo, reads files, and makes edits.
    Changes are written directly to disk by Claude Code.
    """

    def __init__(
        self,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        max_turns: int | None = None,
    ) -> None:
        self.model = model
        self.allowed_tools = allowed_tools
        self.max_turns = max_turns

    def propose(
        self,
        repo: GitRepo,
        base_branch: str,
        feedback: str,
        objective: str | None,
        background: str | None,
    ) -> bool:
        # Get diff from base to show Claude Code what's changed so far
        branch = repo.current_branch()
        try:
            diff = repo.get_diff(base_branch, branch)
        except Exception:
            diff = ""

        prompt_parts: list[str] = []

        if objective:
            prompt_parts.append(f"## Objective\n{objective}")
        if background:
            prompt_parts.append(f"## Background\n{background}")
        if diff:
            prompt_parts.append(f"## Changes So Far (diff from {base_branch})\n\n```diff\n{diff}\n```")

        prompt_parts.append(
            f"## Evaluation Feedback\n\nThe current code was evaluated. Here is the feedback:\n\n{feedback}"
        )
        prompt_parts.append(
            "## Instructions\n\n"
            "Analyze the evaluation feedback and improve the code to address the issues. "
            "Edit the files directly. Do not create new branches or make commits — "
            "just modify the files that need changes."
        )

        prompt = "\n\n".join(prompt_parts)

        cmd = ["claude", "-p", "--output-format", "json"]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])
        if self.max_turns:
            cmd.extend(["--max-turns", str(self.max_turns)])

        result = subprocess.run(
            cmd,
            input=prompt,
            cwd=repo.repo_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            # Try to extract useful error info
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"Claude Code failed (exit {result.returncode}): {error_msg[:500]}")

        # Check if Claude Code actually made changes
        try:
            output = json.loads(result.stdout)
            # Claude Code JSON output has a "result" field
            if isinstance(output, dict) and output.get("is_error"):
                raise RuntimeError(f"Claude Code reported error: {output.get('result', 'Unknown')}")
        except (json.JSONDecodeError, KeyError):
            pass  # Non-JSON output is fine, changes are on disk regardless

        return repo.has_uncommitted_changes()
