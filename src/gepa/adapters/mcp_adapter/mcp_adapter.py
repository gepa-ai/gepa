# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
MCP Adapter for GEPA - Optimizes tool descriptions and system prompts.

Supports local (stdio) and remote (SSE/StreamableHTTP) MCP servers.
Enables optimization of tool descriptions, system prompts, and tool selection
across single or multiple tools.
"""

import asyncio
import json
import logging
from typing import Any, Callable, TypedDict

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

try:
    from mcp import StdioServerParameters  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError("MCP Python SDK is required. Install it with: pip install mcp") from e

from .mcp_client import create_mcp_client

logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions
# ============================================================================


class MCPDataInst(TypedDict):
    """
    Dataset item for MCP tool optimization.

    Attributes:
        user_query: The user's question or request
        tool_arguments: Expected tool arguments (for validation/guidance)
        reference_answer: Optional reference answer for scoring
        additional_context: Optional additional context
    """

    user_query: str
    tool_arguments: dict[str, Any]
    reference_answer: str | None
    additional_context: dict[str, str]


class MCPTrajectory(TypedDict):
    """
    Execution trace for MCP tool invocation.

    Captures the full workflow including tool selection, execution,
    and model behavior at each stage.
    """

    user_query: str
    tool_names: list[str]
    selected_tool: str | None
    tool_called: bool
    tool_arguments: dict[str, Any] | None
    tool_response: str | None
    tool_description_used: str
    system_prompt_used: str
    model_first_pass_output: str
    model_final_output: str
    score: float


class MCPOutput(TypedDict):
    """
    Output from MCP evaluation.

    Attributes:
        final_answer: The final answer from the model
        tool_called: Whether a tool was called
        selected_tool: Which tool was selected (if any)
        tool_response: The tool's response (if called)
    """

    final_answer: str
    tool_called: bool
    selected_tool: str | None
    tool_response: str | None


# ============================================================================
# MCP Adapter
# ============================================================================


class MCPAdapter(GEPAAdapter[MCPDataInst, MCPTrajectory, MCPOutput]):
    """
    GEPA adapter for optimizing MCP tool usage.

    This adapter enables optimization of:
    - Tool descriptions (single or multiple tools)
    - System prompts for tool usage guidance
    - Tool selection logic

    Features:
    - Multi-tool support: Optimize multiple tools simultaneously
    - Two-pass workflow: Tool call + answer generation
    - Multiple transports: stdio (local), SSE, StreamableHTTP (remote)
    - Reflective datasets: Generate training data for refinement

    Example (Local):
        >>> from mcp import StdioServerParameters
        >>> adapter = MCPAdapter(
        ...     tool_names=["read_file", "write_file"],
        ...     task_model="gpt-4o-mini",
        ...     metric_fn=lambda item, output: 1.0 if item["reference_answer"] in output else 0.0,
        ...     server_params=StdioServerParameters(
        ...         command="python",
        ...         args=["server.py"],
        ...     ),
        ... )

    Example (Remote):
        >>> adapter = MCPAdapter(
        ...     tool_names="search_web",
        ...     task_model="gpt-4o-mini",
        ...     metric_fn=accuracy_metric,
        ...     remote_url="https://mcp-server.com/sse",
        ...     remote_transport="sse",
        ... )
    """

    def __init__(
        self,
        tool_names: str | list[str],
        task_model: str | Callable,
        metric_fn: Callable[[MCPDataInst, str], float],
        # Local server configuration
        server_params: StdioServerParameters | None = None,
        # Remote server configuration
        remote_url: str | None = None,
        remote_transport: str = "sse",
        remote_headers: dict[str, str] | None = None,
        remote_timeout: float = 30,
        # Adapter configuration
        base_system_prompt: str = "You are a helpful assistant with access to tools.",
        enable_two_pass: bool = True,
        failure_score: float = 0.0,
    ):
        """
        Initialize MCPAdapter.

        Args:
            tool_names: Name(s) of tool(s) to optimize (str or list[str])
            task_model: Model for task execution (litellm string or callable)
            metric_fn: Scoring function: (data_inst, output) -> float
            server_params: Local MCP server configuration (stdio)
            remote_url: Remote MCP server URL
            remote_transport: "sse" or "streamable_http"
            remote_headers: HTTP headers for remote (e.g., auth tokens)
            remote_timeout: Timeout for remote HTTP operations
            base_system_prompt: Base system prompt template
            enable_two_pass: Use two-pass workflow (tool + answer)
            failure_score: Score assigned when execution fails
        """
        # Store transport configuration
        self.server_params = server_params
        self.remote_url = remote_url
        self.remote_transport = remote_transport
        self.remote_headers = remote_headers or {}
        self.remote_timeout = remote_timeout

        # Normalize tool_names to list
        self.tool_names = [tool_names] if isinstance(tool_names, str) else tool_names

        # Store adapter configuration
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
        Evaluate candidate on batch using MCP tools.

        Args:
            batch: Dataset items to evaluate
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
        """Async implementation of evaluation."""
        outputs: list[MCPOutput] = []
        scores: list[float] = []
        trajectories: list[MCPTrajectory] | None = [] if capture_traces else None

        client = None
        try:
            # Create MCP client using factory
            logger.info(f"Starting MCP session for batch of {len(batch)} items...")
            client = create_mcp_client(
                server_params=self.server_params,
                remote_url=self.remote_url,
                remote_transport=self.remote_transport,
                remote_headers=self.remote_headers,
                remote_timeout=self.remote_timeout,
            )

            await client.start()
            init_result = await client.initialize()
            logger.info(f"MCP session initialized: {init_result.get('serverInfo', {}).get('name', 'unknown')}")

            # Get available tools
            tools_list = await client.list_tools()
            available_tools = [t for t in tools_list if t.get("name") in self.tool_names]

            if not available_tools:
                available_names = [t.get("name") for t in tools_list]
                raise ValueError(f"Tools {self.tool_names} not found. Available: {available_names}")

            # Build system prompt with tools
            system_prompt = self._build_system_prompt(candidate, available_tools)

            # Evaluate each item
            for idx, item in enumerate(batch):
                try:
                    logger.info(f"Evaluating item {idx + 1}/{len(batch)}: {item['user_query'][:50]}...")

                    # First pass: Model calls tool
                    first_pass = await self._first_pass(client, item, system_prompt, available_tools)
                    logger.info(f"First pass complete for item {idx + 1}")

                    # Second pass: Model uses tool response (if enabled)
                    if self.enable_two_pass and first_pass["tool_called"]:
                        final_output = await self._second_pass(client, item, system_prompt, first_pass["tool_response"])
                    else:
                        final_output = first_pass["output"]

                    # Score the output
                    score = self.metric_fn(item, final_output)

                    # Collect results
                    outputs.append(
                        {
                            "final_answer": final_output,
                            "tool_called": first_pass["tool_called"],
                            "selected_tool": first_pass["selected_tool"],
                            "tool_response": first_pass["tool_response"],
                        }
                    )
                    scores.append(score)

                    # Capture trajectory
                    if capture_traces and trajectories is not None:
                        trajectories.append(
                            {
                                "user_query": item["user_query"],
                                "tool_names": self.tool_names,
                                "selected_tool": first_pass["selected_tool"],
                                "tool_called": first_pass["tool_called"],
                                "tool_arguments": first_pass["tool_arguments"],
                                "tool_response": first_pass["tool_response"],
                                "tool_description_used": candidate.get("tool_description", ""),
                                "system_prompt_used": system_prompt,
                                "model_first_pass_output": first_pass["output"],
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
                            "selected_tool": None,
                            "tool_response": None,
                        }
                    )
                    scores.append(self.failure_score)

                    if capture_traces and trajectories is not None:
                        trajectories.append(
                            {
                                "user_query": item["user_query"],
                                "tool_names": self.tool_names,
                                "selected_tool": None,
                                "tool_called": False,
                                "tool_arguments": None,
                                "tool_response": None,
                                "tool_description_used": candidate.get("tool_description", ""),
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
                        "selected_tool": None,
                        "tool_response": None,
                    }
                )
                scores.append(self.failure_score)
                if capture_traces and trajectories is not None:
                    trajectories.append(
                        {
                            "user_query": item["user_query"],
                            "tool_names": self.tool_names,
                            "selected_tool": None,
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
        client,
        item: MCPDataInst,
        system_prompt: str,
        available_tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        First pass: Model receives query and calls tool if needed.

        Returns dict with:
            - output: Raw model output
            - tool_called: Whether tool was called
            - selected_tool: Which tool was selected
            - tool_arguments: Tool arguments
            - tool_response: Tool response
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
                model_output = response.choices[0].message.content.strip()  # type: ignore[union-attr]
                logger.debug(f"Model output: '{model_output}'")
            else:
                model_output = self.task_model(messages)

            # Parse tool call (JSON format)
            tool_called = False
            selected_tool = None
            tool_arguments = None
            tool_response = None

            try:
                parsed = json.loads(model_output)
                if parsed.get("action") == "call_tool":
                    tool_called = True
                    selected_tool = parsed.get("tool")
                    tool_arguments = parsed.get("arguments", {})

                    # Validate tool selection
                    if selected_tool not in self.tool_names:
                        logger.warning(f"Invalid tool '{selected_tool}', available: {self.tool_names}")
                        tool_called = False
                        selected_tool = None
                    else:
                        # Call the tool
                        result = await client.call_tool(selected_tool, tool_arguments)
                        tool_response = self._extract_tool_response(result)

            except (json.JSONDecodeError, KeyError):
                # Model didn't follow JSON format
                pass

            return {
                "output": model_output,
                "tool_called": tool_called,
                "selected_tool": selected_tool,
                "tool_arguments": tool_arguments,
                "tool_response": tool_response,
            }

        except Exception as e:
            logger.exception("First pass failed")
            return {
                "output": f"ERROR: {e!s}",
                "tool_called": False,
                "selected_tool": None,
                "tool_arguments": None,
                "tool_response": None,
            }

    async def _second_pass(
        self,
        client,
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
                return response.choices[0].message.content.strip()  # type: ignore[union-attr]
            else:
                return self.task_model(messages)

        except Exception as e:
            logger.exception("Second pass failed")
            return f"ERROR: {e!s}"

    def _build_system_prompt(self, candidate: dict[str, str], available_tools: list[dict[str, Any]]) -> str:
        """Build system prompt with tool information."""
        custom_system_prompt = candidate.get("system_prompt", self.base_system_prompt)

        # Build tool descriptions
        tool_descriptions = {}
        for tool in available_tools:
            tool_name = tool.get("name")
            # Use optimized description if available
            # Support both tool_description_{tool_name} (multi-tool) and tool_description (single-tool)
            optimized_desc = candidate.get(f"tool_description_{tool_name}") or candidate.get("tool_description")
            tool_descriptions[tool_name] = optimized_desc or tool.get("description", "")

        # Build tools section
        tools_section = "You have access to the following tools:\n\n"

        for tool in available_tools:
            tool_name = tool.get("name")
            tool_description = tool_descriptions[tool_name]
            input_schema = tool.get("inputSchema", {})

            # Build example arguments from schema
            properties = input_schema.get("properties", {})
            example_args = {}
            for param_name, param_info in properties.items():
                if param_info.get("type") == "string":
                    example_args[param_name] = "example_value"
                elif param_info.get("type") == "number":
                    example_args[param_name] = 123
                elif param_info.get("type") == "boolean":
                    example_args[param_name] = True
                else:
                    example_args[param_name] = "value"

            if not example_args:
                example_args = {"param": "value"}

            example_json = json.dumps(example_args)

            tools_section += f"""Tool: {tool_name}
Description: {tool_description}
Input Schema: {json.dumps(input_schema, indent=2)}
Example usage: {{"action": "call_tool", "tool": "{tool_name}", "arguments": {example_json}}}

"""

        # Add usage instructions
        usage_instructions = f"""
When you need to use a tool, respond ONLY with JSON:
{{"action": "call_tool", "tool": "tool_name", "arguments": {{"param": "value"}}}}

When you can answer directly, respond ONLY with JSON:
{{"action": "answer", "text": "your answer"}}

Choose the most appropriate tool for the task. Available tools: {[t.get("name") for t in available_tools]}

Always respond with valid JSON. No other text.
"""

        return f"{custom_system_prompt}\n{tools_section}{usage_instructions}"

    def _extract_tool_response(self, result) -> str:
        """
        Extract text from MCP tool response.

        Handles multiple content types following MCP SDK best practices:
        - TextContent: Plain text responses
        - EmbeddedResource: Resource references
        - ImageContent: Image data (converted to description)
        - structuredContent: Structured JSON data

        Based on latest MCP SDK examples and DSPy implementation.
        """
        try:
            # Import MCP types for proper parsing
            from mcp.types import EmbeddedResource, ImageContent, TextContent  # type: ignore[import-untyped]

            # Check for errors first (following DSPy pattern)
            if hasattr(result, "isError") and result.isError:
                # Extract error message from content
                error_texts = []
                for content_item in result.content:
                    if isinstance(content_item, TextContent):
                        error_texts.append(content_item.text)
                error_msg = "\n".join(error_texts) if error_texts else "Tool execution failed"
                logger.warning(f"Tool returned error: {error_msg}")
                return f"ERROR: {error_msg}"

            # Try structured content first (modern MCP pattern)
            if hasattr(result, "structuredContent") and result.structuredContent:
                import json

                return json.dumps(result.structuredContent, indent=2)

            # Parse content array
            if hasattr(result, "content"):
                texts = []
                for content_item in result.content:
                    if isinstance(content_item, TextContent):
                        texts.append(content_item.text)
                    elif isinstance(content_item, EmbeddedResource):
                        # Handle embedded resources
                        resource = content_item.resource
                        if hasattr(resource, "text"):
                            texts.append(resource.text)
                        else:
                            texts.append(f"[Resource: {getattr(resource, 'uri', 'unknown')}]")
                    elif isinstance(content_item, ImageContent):
                        # Handle images with description
                        mime_type = getattr(content_item, "mimeType", "unknown")
                        data_len = len(getattr(content_item, "data", b""))
                        texts.append(f"[Image: {mime_type}, {data_len} bytes]")

                if texts:
                    return "\n".join(texts)

            # Fallback to dict access for backward compatibility
            if isinstance(result, dict):
                content = result.get("content", [])
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            texts.append(item.get("text", ""))
                    # Return empty string if content list is present but empty
                    return "\n".join(texts)

            return str(result)

        except Exception as e:
            logger.exception("Failed to extract tool response")
            return f"ERROR extracting response: {e!s}"

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[MCPTrajectory, MCPOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build reflective dataset for instruction refinement.

        Args:
            candidate: Current candidate components
            eval_batch: Evaluation results with trajectories
            components_to_update: Which components to generate data for

        Returns:
            Dictionary mapping component names to reflective examples
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
                    feedback = self._generate_tool_feedback(traj, score)
                    examples.append(
                        {
                            "Inputs": {
                                "user_query": traj["user_query"],
                                "tool_description": traj["tool_description_used"],
                            },
                            "Generated Outputs": {
                                "tool_called": traj["tool_called"],
                                "selected_tool": traj["selected_tool"],
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
        """Generate feedback focused on tool usage and selection."""
        if score > 0.5:
            if traj["tool_called"]:
                return (
                    f"Good! The tool '{traj['selected_tool']}' was used appropriately. "
                    f"Score: {score:.2f}"
                )
            else:
                return f"Good! No tool needed, direct answer was correct. Score: {score:.2f}"
        else:
            feedback_parts = [f"Incorrect response (score: {score:.2f})."]

            if not traj["tool_called"]:
                feedback_parts.append("Tool was not called. Consider if a tool would help.")
            else:
                selected_tool = traj["selected_tool"]
                available_tools = traj["tool_names"]
                feedback_parts.append(
                    f"Tool '{selected_tool}' was called with {traj['tool_arguments']}, "
                    f"but answer was incorrect."
                )
                if len(available_tools) > 1:
                    feedback_parts.append(
                        f"Consider a different tool from {available_tools} or clearer description."
                    )
                else:
                    feedback_parts.append("Tool description may need improvement.")

            return " ".join(feedback_parts)

    def _generate_system_prompt_feedback(self, traj: MCPTrajectory, score: float) -> str:
        """Generate feedback focused on system prompt guidance."""
        if score > 0.5:
            return f"System prompt provided good guidance. Score: {score:.2f}"
        else:
            return (
                f"System prompt may need improvement (score: {score:.2f}). "
                f"Model {'called' if traj['tool_called'] else 'did not call'} tool, "
                f"but answer was incorrect."
            )
