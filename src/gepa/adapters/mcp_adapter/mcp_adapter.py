# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import asyncio
import json
import logging
from typing import Any, Callable

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from .mcp_types import MCPDataInst, MCPOutput, MCPTrajectory

try:
    from mcp import StdioServerParameters, stdio_client
    from mcp.client.session import ClientSession
    from mcp.types import TextContent, Tool
except ImportError as e:
    raise ImportError("MCP Python SDK is required for MCPAdapter. Install it with: pip install mcp") from e

logger = logging.getLogger(__name__)


class MCPAdapter(GEPAAdapter[MCPDataInst, MCPTrajectory, MCPOutput]):
    """
    GEPA adapter for optimizing MCP tool usage.

    This adapter enables optimization of:
    - Tool descriptions (primary component)
    - System prompts for tool usage guidance
    - Tool usage guidelines

    The adapter uses a two-pass workflow:
    1. First pass: Model receives user query and decides to call tool
    2. Second pass: Model receives tool response and generates final answer

    Example:
        >>> from mcp import StdioServerParameters
        >>> adapter = MCPAdapter(
        ...     server_params=StdioServerParameters(
        ...         command="npx",
        ...         args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ...     ),
        ...     tool_name="read_file",
        ...     task_model="openai/gpt-4o-mini",
        ...     metric_fn=lambda item, output: 1.0 if item["reference_answer"] in output else 0.0,
        ... )
    """

    def __init__(
        self,
        server_params: StdioServerParameters,
        tool_name: str,
        task_model: str | Callable,
        metric_fn: Callable[[MCPDataInst, str], float],
        base_system_prompt: str = "You are a helpful assistant with access to tools.",
        enable_two_pass: bool = True,
        failure_score: float = 0.0,
        max_litellm_workers: int = 10,
    ):
        """
        Initialize MCPAdapter.

        Args:
            server_params: MCP server configuration (command, args, env)
            tool_name: Name of the tool to optimize
            task_model: Model to use for task execution (litellm model string or callable)
            metric_fn: Function to score outputs: (data_inst, output) -> float
            base_system_prompt: Base system prompt (will be extended with tool info)
            enable_two_pass: Use two-pass workflow (tool call + answer generation)
            failure_score: Score to assign when execution fails
            max_litellm_workers: Maximum parallel workers for litellm batch calls
        """
        self.server_params = server_params
        self.tool_name = tool_name
        self.base_system_prompt = base_system_prompt
        self.enable_two_pass = enable_two_pass
        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self.metric_fn = metric_fn

        # Setup model
        if isinstance(task_model, str):
            import litellm

            self.litellm = litellm
        self.task_model = task_model

    def evaluate(
        self,
        batch: list[MCPDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[MCPTrajectory, MCPOutput]:
        """
        Evaluate candidate on batch using MCP tool.

        This runs an async evaluation session for the entire batch,
        creating a new MCP server process and session.

        Args:
            batch: List of dataset items to evaluate
            candidate: Component mapping (e.g., {"tool_description": "..."})
            capture_traces: Whether to capture detailed trajectories

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        # Run async evaluation in new event loop
        return asyncio.run(self._evaluate_async(batch, candidate, capture_traces))

    async def _evaluate_async(
        self,
        batch: list[MCPDataInst],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> EvaluationBatch[MCPTrajectory, MCPOutput]:
        """Async implementation of evaluation."""
        outputs: list[MCPOutput] = []
        scores: list[float] = []
        trajectories: list[MCPTrajectory] | None = [] if capture_traces else None

        try:
            # Create MCP client session
            async with stdio_client(self.server_params) as (read, write):
                session = ClientSession(read, write)
                await session.initialize()

                # Get tool information
                tools_result = await session.list_tools()
                tool_info = self._find_tool(tools_result.tools, self.tool_name)

                # Build system prompt with optimized components
                system_prompt = self._build_system_prompt(candidate, tool_info)
                optimized_description = candidate.get("tool_description", tool_info.description or "")

                # Evaluate each item in batch
                for item in batch:
                    try:
                        # First pass: Model calls tool
                        first_pass_result = await self._first_pass(session, item, system_prompt, tool_info)

                        # Second pass: Model uses tool response (if enabled)
                        if self.enable_two_pass and first_pass_result["tool_called"]:
                            final_output = await self._second_pass(
                                session, item, system_prompt, first_pass_result["tool_response"]
                            )
                        else:
                            final_output = first_pass_result["output"]

                        # Score the output
                        score = self.metric_fn(item, final_output)

                        # Collect results
                        output: MCPOutput = {
                            "final_answer": final_output,
                            "tool_called": first_pass_result["tool_called"],
                            "tool_response": first_pass_result["tool_response"],
                        }
                        outputs.append(output)
                        scores.append(score)

                        # Capture trajectory
                        if capture_traces:
                            trajectory: MCPTrajectory = {
                                "user_query": item["user_query"],
                                "tool_name": self.tool_name,
                                "tool_called": first_pass_result["tool_called"],
                                "tool_arguments": first_pass_result["tool_arguments"],
                                "tool_response": first_pass_result["tool_response"],
                                "tool_description_used": optimized_description,
                                "system_prompt_used": system_prompt,
                                "model_first_pass_output": first_pass_result["output"],
                                "model_final_output": final_output,
                                "score": score,
                            }
                            trajectories.append(trajectory)

                    except Exception as e:
                        logger.exception(f"Failed to evaluate item: {item['user_query']}")
                        # Return failure score for this item
                        outputs.append(
                            {
                                "final_answer": "",
                                "tool_called": False,
                                "tool_response": None,
                            }
                        )
                        scores.append(self.failure_score)

                        if capture_traces:
                            trajectories.append(
                                {
                                    "user_query": item["user_query"],
                                    "tool_name": self.tool_name,
                                    "tool_called": False,
                                    "tool_arguments": None,
                                    "tool_response": None,
                                    "tool_description_used": optimized_description,
                                    "system_prompt_used": system_prompt,
                                    "model_first_pass_output": f"ERROR: {e!s}",
                                    "model_final_output": "",
                                    "score": self.failure_score,
                                }
                            )

        except Exception as e:
            logger.exception("Failed to create MCP session")
            # Return failure for entire batch
            for item in batch:
                outputs.append(
                    {
                        "final_answer": "",
                        "tool_called": False,
                        "tool_response": None,
                    }
                )
                scores.append(self.failure_score)
                if capture_traces:
                    trajectories.append(
                        {
                            "user_query": item["user_query"],
                            "tool_name": self.tool_name,
                            "tool_called": False,
                            "tool_arguments": None,
                            "tool_response": None,
                            "tool_description_used": "",
                            "system_prompt_used": "",
                            "model_first_pass_output": f"SESSION ERROR: {e!s}",
                            "model_final_output": "",
                            "score": self.failure_score,
                        }
                    )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    async def _first_pass(
        self,
        session: ClientSession,
        item: MCPDataInst,
        system_prompt: str,
        tool_info: Tool,
    ) -> dict[str, Any]:
        """
        First pass: Model receives query and calls tool if needed.

        Returns dict with:
            - output: Raw model output
            - tool_called: Whether tool was called
            - tool_arguments: Arguments passed to tool (if called)
            - tool_response: Tool response (if called)
        """
        # Build message with tool information
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["user_query"]},
        ]

        # For now, we'll use a simple approach: have the model output JSON
        # with tool call decision. In future, we can use function calling API.
        try:
            if isinstance(self.task_model, str):
                response = self.litellm.completion(
                    model=self.task_model,
                    messages=messages,
                )
                model_output = response.choices[0].message.content.strip()
            else:
                model_output = self.task_model(messages)

            # Parse tool call (simple JSON format for now)
            # Expected format: {"action": "call_tool", "arguments": {...}} or {"action": "answer", "text": "..."}
            tool_called = False
            tool_arguments = None
            tool_response = None

            try:
                parsed = json.loads(model_output)
                if parsed.get("action") == "call_tool":
                    tool_called = True
                    tool_arguments = parsed.get("arguments", {})

                    # Call the tool via MCP
                    result = await session.call_tool(
                        name=self.tool_name,
                        arguments=tool_arguments,
                    )

                    # Extract text from tool response
                    tool_response = self._extract_tool_response(result)

            except (json.JSONDecodeError, KeyError):
                # Model didn't follow JSON format, treat as direct answer
                pass

            return {
                "output": model_output,
                "tool_called": tool_called,
                "tool_arguments": tool_arguments,
                "tool_response": tool_response,
            }

        except Exception as e:
            logger.exception("First pass failed")
            return {
                "output": f"ERROR: {e!s}",
                "tool_called": False,
                "tool_arguments": None,
                "tool_response": None,
            }

    async def _second_pass(
        self,
        session: ClientSession,
        item: MCPDataInst,
        system_prompt: str,
        tool_response: str | None,
    ) -> str:
        """
        Second pass: Model receives tool response and generates final answer.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["user_query"]},
            {
                "role": "assistant",
                "content": f"I'll use the tool to help answer this. Tool response: {tool_response}",
            },
            {
                "role": "user",
                "content": "Based on the tool response, please provide your final answer.",
            },
        ]

        try:
            if isinstance(self.task_model, str):
                response = self.litellm.completion(
                    model=self.task_model,
                    messages=messages,
                )
                return response.choices[0].message.content.strip()
            else:
                return self.task_model(messages)

        except Exception as e:
            logger.exception("Second pass failed")
            return f"ERROR: {e!s}"

    def _build_system_prompt(self, candidate: dict[str, str], tool_info: Tool) -> str:
        """Build system prompt with tool information."""
        tool_description = candidate.get("tool_description", tool_info.description or "")
        custom_system_prompt = candidate.get("system_prompt", self.base_system_prompt)

        # Build tool info section
        tool_section = f"""
You have access to the following tool:

Tool: {self.tool_name}
Description: {tool_description}
Input Schema: {json.dumps(tool_info.inputSchema, indent=2)}

To use the tool, respond with JSON in this format:
{{"action": "call_tool", "arguments": {{"param1": "value1", ...}}}}

If you don't need the tool, respond with:
{{"action": "answer", "text": "your answer here"}}
"""

        return f"{custom_system_prompt}\n\n{tool_section}"

    def _find_tool(self, tools: list[Tool], tool_name: str) -> Tool:
        """Find tool by name in list."""
        for tool in tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Tool '{tool_name}' not found in MCP server. Available: {[t.name for t in tools]}")

    def _extract_tool_response(self, result: Any) -> str:
        """Extract text from MCP tool response."""
        # MCP tool results contain content array
        if hasattr(result, "content") and result.content:
            # Extract text from TextContent items
            texts = []
            for content_item in result.content:
                if isinstance(content_item, TextContent):
                    texts.append(content_item.text)
                elif hasattr(content_item, "text"):
                    texts.append(content_item.text)
            return "\n".join(texts)
        return str(result)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[MCPTrajectory, MCPOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build reflective dataset for instruction refinement.

        This extracts examples showing:
        - Successful and failed tool calls
        - Cases where tool wasn't called but should have been
        - Cases where tool response wasn't utilized well
        """
        reflective_data: dict[str, list[dict[str, Any]]] = {}

        for component in components_to_update:
            examples: list[dict[str, Any]] = []

            for traj, score, _output in zip(
                eval_batch.trajectories or [],
                eval_batch.scores,
                eval_batch.outputs,
                strict=False,
            ):
                if component == "tool_description":
                    # Focus on tool-specific feedback
                    feedback = self._generate_tool_feedback(traj, score)
                    examples.append(
                        {
                            "Inputs": {
                                "user_query": traj["user_query"],
                                "tool_description": traj["tool_description_used"],
                            },
                            "Generated Outputs": {
                                "tool_called": traj["tool_called"],
                                "tool_arguments": traj["tool_arguments"],
                                "final_answer": traj["model_final_output"],
                            },
                            "Feedback": feedback,
                        }
                    )

                elif component == "system_prompt":
                    # Focus on general guidance feedback
                    feedback = self._generate_system_prompt_feedback(traj, score)
                    examples.append(
                        {
                            "Inputs": {
                                "user_query": traj["user_query"],
                                "system_prompt": traj["system_prompt_used"],
                            },
                            "Generated Outputs": traj["model_final_output"],
                            "Feedback": feedback,
                        }
                    )

            reflective_data[component] = examples

        return reflective_data

    def _generate_tool_feedback(self, traj: MCPTrajectory, score: float) -> str:
        """Generate feedback focused on tool usage."""
        if score > 0.5:
            return (
                f"Good! The tool was used appropriately and produced a correct answer. "
                f"Tool called: {traj['tool_called']}, Score: {score:.2f}"
            )
        else:
            feedback_parts = [f"The response was incorrect (score: {score:.2f})."]

            if not traj["tool_called"]:
                feedback_parts.append(
                    "The tool was not called. Consider whether calling the tool would help answer this query."
                )
            else:
                feedback_parts.append(
                    f"The tool was called with arguments {traj['tool_arguments']}, "
                    f"but the final answer was still incorrect. The tool description might need to be clearer "
                    f"about when and how to use the tool."
                )

            return " ".join(feedback_parts)

    def _generate_system_prompt_feedback(self, traj: MCPTrajectory, score: float) -> str:
        """Generate feedback focused on system prompt guidance."""
        if score > 0.5:
            return f"The system prompt provided good guidance. Score: {score:.2f}"
        else:
            return (
                f"The system prompt may need improvement (score: {score:.2f}). "
                f"The model {'called' if traj['tool_called'] else 'did not call'} the tool, "
                f"but the final answer was incorrect. Consider providing clearer guidance on tool usage strategy."
            )
