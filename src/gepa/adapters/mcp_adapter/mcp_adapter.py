# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
MCP Adapter for GEPA - Supports local and remote MCP servers.

This adapter uses SimpleStdioMCPClient for local servers as a workaround
for the MCP Python SDK stdio transport bug:
https://github.com/modelcontextprotocol/python-sdk/issues/1452

Once the SDK bug is fixed, we'll migrate to use the official client while
maintaining the same interface.

Supports:
- Local stdio transport (subprocess-based)
- Remote SSE transport (HTTP Server-Sent Events)
- Remote StreamableHTTP transport (production-grade HTTP)
"""

import asyncio
import json
import logging
from typing import Any, Callable

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from .mcp_types import MCPDataInst, MCPOutput, MCPTrajectory

try:
    from mcp import StdioServerParameters
except ImportError as e:
    raise ImportError("MCP Python SDK is required for MCPAdapter. Install it with: pip install mcp") from e

from .simple_stdio_client import SimpleStdioMCPClient

logger = logging.getLogger(__name__)


class MCPAdapter(GEPAAdapter[MCPDataInst, MCPTrajectory, MCPOutput]):
    """
    GEPA adapter for optimizing MCP tool usage.

    This adapter enables optimization of:
    - Tool descriptions
    - System prompts for tool usage guidance
    - Tool usage guidelines

    The adapter uses a two-pass workflow:
    1. First pass: Model receives user query and decides to call tool
    2. Second pass: Model receives tool response and generates final answer

    Supports both local (stdio) and remote (SSE/StreamableHTTP) MCP servers.

    Example (Local):
        >>> from mcp import StdioServerParameters
        >>> adapter = MCPAdapter(
        ...     tool_name="read_file",
        ...     task_model="gpt-4o-mini",
        ...     metric_fn=lambda item, output: 1.0 if item["reference_answer"] in output else 0.0,
        ...     server_params=StdioServerParameters(
        ...         command="python",
        ...         args=["server.py", "/data"],
        ...     ),
        ... )

    Example (Remote):
        >>> adapter = MCPAdapter(
        ...     tool_name="search_web",
        ...     task_model="gpt-4o-mini",
        ...     metric_fn=lambda item, output: 1.0 if item["reference_answer"] in output else 0.0,
        ...     remote_url="https://mcp-server.com/sse",
        ...     remote_transport="sse",
        ... )
    """

    def __init__(
        self,
        tool_name: str,
        task_model: str | Callable,
        metric_fn: Callable[[MCPDataInst, str], float],
        # Local server params (for stdio transport)
        server_params: StdioServerParameters | None = None,
        # Remote server params (for SSE/StreamableHTTP transport)
        remote_url: str | None = None,
        remote_transport: str = "sse",  # "sse" or "streamable_http"
        remote_headers: dict[str, str] | None = None,
        remote_timeout: float = 30,
        # Common params
        base_system_prompt: str = "You are a helpful assistant with access to tools.",
        enable_two_pass: bool = True,
        failure_score: float = 0.0,
    ):
        """
        Initialize MCPAdapter.

        Args:
            tool_name: Name of the tool to optimize
            task_model: Model to use for task execution (litellm model string or callable)
            metric_fn: Function to score outputs: (data_inst, output) -> float
            server_params: MCP server configuration for local stdio (command, args, env)
            remote_url: URL for remote MCP server (SSE or StreamableHTTP)
            remote_transport: Transport type for remote server ("sse" or "streamable_http")
            remote_headers: Optional headers for remote requests (e.g., auth tokens)
            remote_timeout: Timeout for remote HTTP operations
            base_system_prompt: Base system prompt (will be extended with tool info)
            enable_two_pass: Use two-pass workflow (tool call + answer generation)
            failure_score: Score to assign when execution fails

        Note:
            Exactly one of server_params or remote_url must be provided.
        """
        # Validate transport configuration
        if server_params and remote_url:
            raise ValueError("Provide either server_params (local) or remote_url (remote), not both")
        if not server_params and not remote_url:
            raise ValueError("Must provide either server_params (local) or remote_url (remote)")

        self.server_params = server_params
        self.remote_url = remote_url
        self.remote_transport = remote_transport
        self.remote_headers = remote_headers or {}
        self.remote_timeout = remote_timeout

        self.tool_name = tool_name
        self.base_system_prompt = base_system_prompt
        self.enable_two_pass = enable_two_pass
        self.failure_score = failure_score
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

        Args:
            batch: List of dataset items to evaluate
            candidate: Component mapping (e.g., {"tool_description": "..."})
            capture_traces: Whether to capture detailed trajectories

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        return asyncio.run(self._evaluate_async(batch, candidate, capture_traces))

    async def _evaluate_async(
        self,
        batch: list[MCPDataInst],
        candidate: dict[str, str],
        capture_traces: bool,
    ) -> EvaluationBatch[MCPTrajectory, MCPOutput]:
        """Async implementation of evaluation using configured MCP client."""
        outputs: list[MCPOutput] = []
        scores: list[float] = []
        trajectories: list[MCPTrajectory] | None = [] if capture_traces else None

        client = None
        try:
            # Create MCP client based on configuration
            logger.info(f"Starting MCP session for batch of {len(batch)} items...")

            if self.server_params:
                # Local stdio transport
                from .simple_stdio_client import SimpleStdioMCPClient

                logger.info("Using local stdio transport")
                client = SimpleStdioMCPClient(command=self.server_params.command, args=self.server_params.args)
            elif self.remote_url:
                # Remote transport (SSE or StreamableHTTP)
                if self.remote_transport == "sse":
                    from .sse_client import SSEMCPClient

                    logger.info(f"Using remote SSE transport: {self.remote_url}")
                    client = SSEMCPClient(
                        url=self.remote_url,
                        headers=self.remote_headers,
                        timeout=self.remote_timeout,
                    )
                elif self.remote_transport == "streamable_http":
                    from .streamable_http_client import StreamableHTTPMCPClient

                    logger.info(f"Using remote StreamableHTTP transport: {self.remote_url}")
                    client = StreamableHTTPMCPClient(
                        url=self.remote_url,
                        headers=self.remote_headers,
                        timeout=self.remote_timeout,
                    )
                else:
                    raise ValueError(
                        f"Unknown remote transport: {self.remote_transport}. Must be 'sse' or 'streamable_http'"
                    )

            await client.start()

            # Initialize session
            init_result = await client.initialize()
            logger.info(f"MCP session initialized: {init_result.get('serverInfo', {}).get('name', 'unknown')}")

            # Get tool information
            tools_list = await client.list_tools()
            tool_info = next((t for t in tools_list if t.get("name") == self.tool_name), None)
            if not tool_info:
                raise ValueError(f"Tool '{self.tool_name}' not found. Available: {[t.get('name') for t in tools_list]}")

            # Build system prompt
            system_prompt = self._build_system_prompt(candidate, tool_info)
            optimized_description = candidate.get("tool_description", tool_info.get("description", ""))

            # Evaluate each item in batch
            for idx, item in enumerate(batch):
                try:
                    logger.info(f"Evaluating item {idx + 1}/{len(batch)}: {item['user_query'][:50]}...")

                    # First pass: Model calls tool
                    first_pass_result = await self._first_pass(client, item, system_prompt, tool_info)
                    logger.info(f"First pass complete for item {idx + 1}")

                    # Second pass: Model uses tool response (if enabled)
                    if self.enable_two_pass and first_pass_result["tool_called"]:
                        final_output = await self._second_pass(
                            client, item, system_prompt, first_pass_result["tool_response"]
                        )
                    else:
                        final_output = first_pass_result["output"]

                    # Score the output
                    score = self.metric_fn(item, final_output)

                    # Collect results
                    outputs.append(
                        {
                            "final_answer": final_output,
                            "tool_called": first_pass_result["tool_called"],
                            "tool_response": first_pass_result["tool_response"],
                        }
                    )
                    scores.append(score)

                    # Capture trajectory
                    if capture_traces:
                        trajectories.append(
                            {
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
                        )

                except Exception as e:
                    logger.exception(f"Failed to evaluate item: {item['user_query']}")
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
        finally:
            if client:
                await client.close()

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    async def _first_pass(
        self,
        client: SimpleStdioMCPClient,
        item: MCPDataInst,
        system_prompt: str,
        tool_info: dict[str, Any],
    ) -> dict[str, Any]:
        """
        First pass: Model receives query and calls tool if needed.

        Returns dict with:
            - output: Raw model output
            - tool_called: Whether tool was called
            - tool_arguments: Arguments passed to tool (if called)
            - tool_response: Tool response (if called)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["user_query"]},
        ]

        try:
            if isinstance(self.task_model, str):
                logger.debug(f"Calling model with messages: {messages}")
                response = self.litellm.completion(
                    model=self.task_model,
                    messages=messages,
                )
                model_output = response.choices[0].message.content.strip()
                logger.debug(f"Model output (raw): '{model_output}'")
                logger.debug(f"Model output length: {len(model_output)} chars")
            else:
                model_output = self.task_model(messages)

            # Parse tool call (JSON format)
            tool_called = False
            tool_arguments = None
            tool_response = None

            try:
                parsed = json.loads(model_output)
                if parsed.get("action") == "call_tool":
                    tool_called = True
                    tool_arguments = parsed.get("arguments", {})

                    # Call the tool via simple MCP client
                    result = await client.call_tool(self.tool_name, tool_arguments)

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
        client: SimpleStdioMCPClient,
        item: MCPDataInst,
        system_prompt: str,
        tool_response: str | None,
    ) -> str:
        """Second pass: Model receives tool response and generates final answer."""
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

    def _build_system_prompt(self, candidate: dict[str, str], tool_info: dict[str, Any]) -> str:
        """Build system prompt with tool information."""
        tool_description = candidate.get("tool_description", tool_info.get("description", ""))
        custom_system_prompt = candidate.get("system_prompt", self.base_system_prompt)

        # Build example arguments from schema
        input_schema = tool_info.get("inputSchema", {})
        properties = input_schema.get("properties", {})

        # Create example with actual parameter names from schema
        example_args = {}
        for param_name, param_info in properties.items():
            # Use first parameter as example, or create simple example
            if param_info.get("type") == "string":
                example_args[param_name] = "example_value"
            elif param_info.get("type") == "number":
                example_args[param_name] = 123
            elif param_info.get("type") == "boolean":
                example_args[param_name] = True
            else:
                example_args[param_name] = "value"

        # Fallback if no properties
        if not example_args:
            example_args = {"param": "value"}

        example_json = json.dumps(example_args)

        # Build tool info section
        # NOTE: Simplified format to avoid triggering "thinking" mode in reasoning models
        tool_section = f"""
You have access to the tool: {self.tool_name}
Description: {tool_description}
Input Schema: {json.dumps(input_schema, indent=2)}

When you need to use the tool, respond ONLY with JSON:
{{"action": "call_tool", "arguments": {example_json}}}

When you can answer directly, respond ONLY with JSON:
{{"action": "answer", "text": "your answer"}}

Always respond with valid JSON. No other text.
"""

        return f"{custom_system_prompt}\n{tool_section}"

    def _extract_tool_response(self, result: dict) -> str:
        """Extract text from MCP tool response."""
        if isinstance(result, dict):
            content = result.get("content", [])
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                return "\n".join(texts)
        return str(result)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[MCPTrajectory, MCPOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build reflective dataset for instruction refinement."""
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
                    f"but the final answer was still incorrect. The tool description might need to be clearer."
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
                f"but the final answer was incorrect."
            )
