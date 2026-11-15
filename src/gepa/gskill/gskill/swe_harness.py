import gc
import logging
import os
import platform
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Suppress verbose LiteLLM logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import litellm

litellm.suppress_debug_info = True

import docker
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from minisweagent.models.litellm_model import LitellmModel
from swebench.harness.constants import KEY_INSTANCE_ID, LOG_TEST_OUTPUT, RUN_EVALUATION_LOG_DIR
from swesmith.harness.utils import run_patch_in_container
from swesmith.profiles import registry


class ExistingContainerEnvironment(DockerEnvironment):
    """DockerEnvironment subclass that uses an existing container instead of creating one."""

    def __init__(self, container_id: str, cwd: str = "/testbed", timeout: int = 120):
        # Don't call super().__init__() - it would try to create a new container
        # Just set up the minimal config needed
        self.logger = None
        self.container_id = container_id
        self.config = DockerEnvironmentConfig(
            image="unused",  # Not used since we already have a container
            cwd=cwd,
            timeout=timeout,
            env={
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        )

    def cleanup(self):
        """No cleanup - container is managed by SWEHarness."""
        pass


@dataclass
class TaskResult:
    passed: bool
    trace: str
    output: str


class SWEHarness:
    def __init__(self):
        """Initialize harness with SWE-smith Docker containers."""
        self.container = None
        self.repo_profile = None
        self.current_task = None

        # Set DOCKER_HOST environment variable for rootless Docker
        # This ensures SWE-smith's internal docker.from_env() calls work correctly
        if not os.getenv("DOCKER_HOST"):
            # Try to detect rootless Docker
            uid = os.getuid()
            xdg_runtime = os.getenv("XDG_RUNTIME_DIR")

            if xdg_runtime:
                rootless_socket = f"unix://{xdg_runtime}/docker.sock"
            else:
                rootless_socket = f"unix:///run/user/{uid}/docker.sock"

            # Check if rootless socket exists
            import pathlib

            socket_path = rootless_socket.replace("unix://", "")
            if pathlib.Path(socket_path).exists():
                os.environ["DOCKER_HOST"] = rootless_socket
                print(f"  Using rootless Docker: {rootless_socket}")

        # Try to connect to Docker
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Docker: {e}\n\n"
                f"Please ensure Docker is running:\n"
                f"  Rootless: systemctl --user start docker\n"
                f"  Standard: sudo systemctl start docker\n\n"
                f"If using rootless Docker, ensure DOCKER_HOST is set:\n"
                f"  export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock"
            ) from e

    def setup_task(self, task_instance: dict[str, Any]):
        """Setup task environment using SWE-smith Docker container.

        Args:
            task_instance: Full SWE-smith task instance
        """
        # Cleanup previous container if any
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
            except Exception:
                pass

        # Get container from SWE-smith
        # The container comes with:
        # - Repository cloned at /testbed
        # - Instance branch checked out at HEAD (tests removed - agent can't see them)
        # - All dependencies installed
        #
        # NOTE: SWE-smith branch structure:
        # - HEAD: Bug commit WITHOUT test files (for agent work)
        # - HEAD~1: Bug commit WITH test files (for evaluation)
        # The agent works at HEAD. Verification uses run_patch_in_container which
        # handles the proper checkout to HEAD~1 for testing.
        self.repo_profile = registry.get_from_inst(task_instance)
        self.current_task = task_instance
        self.container = self.repo_profile.get_container(task_instance)
        print(f"  Docker container created: {self.container.id[:12]}")

    def run_agent(
        self,
        problem_statement: str,
        skills: str,
        model_name: str = "gemini/gemini-2.0-flash-exp",
        config_path: str | None = None,
    ) -> tuple[str, str, dict]:
        """Run the agent in Docker container and return (patch, conversation_trace, metrics).

        The skills from GEPA are injected into the system template's {{ skills }} placeholder.
        This allows GEPA to evolve the agent's learned skills over time.

        Args:
            problem_statement: The problem to solve
            skills: Skills string to inject (ignored if config doesn't have {{ skills }} placeholder)
            model_name: LiteLLM model name
            config_path: Optional custom config path. Defaults to mini.yaml

        Returns:
            patch: The git diff of changes made
            trace: Full conversation trace
            metrics: Dict with 'steps' (number of agent turns) and 'tokens' (estimated)
        """

        # Load our custom mini-swe-agent config from the project directory
        # This config has {{ skills }} placeholder that GEPA will optimize
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "mini_swe_agent_config", "mini.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Extract only supported fields from agent config
        full_agent_config = config.get("agent", {})
        # DefaultAgent only accepts these template fields + limits
        supported_fields = [
            "system_template",
            "instance_template",
            "action_observation_template",
            "format_error_template",
            "timeout_template",
            "step_limit",
            "cost_limit",
        ]
        agent_config = {k: v for k, v in full_agent_config.items() if k in supported_fields}
        agent_config["step_limit"] = 50  # Max steps per task

        # Get model kwargs from config and add OpenAI regional endpoint if needed
        model_config = config.get("model", {})
        model_kwargs = model_config.get("model_kwargs", {}).copy()

        # Add OpenAI regional endpoint (us.api.openai.com) if using OpenAI models
        if "openai" in model_name.lower() or model_name.startswith("gpt-"):
            if "api_base" not in model_kwargs:
                model_kwargs["api_base"] = "https://us.api.openai.com/v1"

        # Initialize Agent with Docker container environment
        # Use ExistingContainerEnvironment to execute commands inside the container
        agent = DefaultAgent(
            model=LitellmModel(model_name=model_name, model_kwargs=model_kwargs),
            env=ExistingContainerEnvironment(
                container_id=self.container.id,
                cwd="/testbed",
                timeout=120,
            ),
            **agent_config,
        )

        try:
            # We wrap in try/except to ensure we capture trace even if it crashes
            # Build kwargs for template variables
            run_kwargs = {}

            # Pass skills if the config has the {{ skills }} placeholder
            if "{{ skills }}" in agent_config.get("system_template", ""):
                run_kwargs["skills"] = skills

            # Pass system info if the config has these placeholders (original_mini.yaml)
            instance_template = agent_config.get("instance_template", "")
            if "{{system}}" in instance_template or "{{ system }}" in instance_template:
                run_kwargs["system"] = platform.system()
                run_kwargs["release"] = platform.release()
                run_kwargs["version"] = platform.version()
                run_kwargs["machine"] = platform.machine()

            result = agent.run(problem_statement, **run_kwargs)

            # Extract the full conversation trace from agent.messages
            # This contains the agent's reasoning, actions, and tool outputs
            trace = "\n\n".join([f"[{msg['role'].upper()}]\n{msg['content']}" for msg in agent.messages])

            # Calculate metrics
            num_steps = len([m for m in agent.messages if m.get("role") == "assistant"])

            # Use LiteLLM's token counter for accurate count
            try:
                import litellm

                # token_counter expects messages with 'role' and 'content' keys
                token_count = litellm.token_counter(model=model_name, messages=agent.messages)
            except Exception:
                # Fallback to estimate if tokenizer fails
                total_chars = sum(len(m.get("content", "")) for m in agent.messages)
                token_count = total_chars // 4

            metrics = {
                "steps": num_steps,
                "estimated_tokens": token_count,
                "num_messages": len(agent.messages),
                "messages": list(agent.messages),  # Structured messages for trace logging
            }

            # Generate patch of changes from Docker container
            result = self.container.exec_run("git diff", workdir="/testbed")
            patch = result.output.decode() if result.output else ""

            # Explicit cleanup to prevent memory leaks
            del agent
            gc.collect()

            return patch, trace, metrics

        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            print(f"  AGENT ERROR: {e!s}")
            print(f"  Traceback:\n{error_trace}")
            gc.collect()  # Clean up even on error
            return (
                "",
                f"Agent crashed: {e!s}\n\nTraceback:\n{error_trace}",
                {"steps": 0, "estimated_tokens": 0, "num_messages": 0},
            )

    def verify_with_patch(self, patch: str, f2p_only: bool = True, timeout: int = 300) -> tuple[bool, str]:
        """Verify a patch using SWE-smith's run_patch_in_container.

        This is the proper way to evaluate patches - it:
        1. Creates a fresh container
        2. Checks out the correct commit with test files
        3. Applies the patch
        4. Runs the appropriate tests
        5. Cleans up

        Args:
            patch: The git diff patch to apply and test
            f2p_only: If True, only run FAIL_TO_PASS tests
            timeout: Test timeout in seconds

        Returns:
            (passed, test_output) tuple
        """
        if not self.current_task:
            return False, "No task set up"

        instance_id = self.current_task.get(KEY_INSTANCE_ID, "unknown")
        run_id = f"gepa_{uuid.uuid4().hex[:8]}"
        # IMPORTANT: Use RUN_EVALUATION_LOG_DIR to trigger is_eval=True in run_patch_in_container
        # This makes it do `git checkout HEAD~1` to restore test files
        log_dir = RUN_EVALUATION_LOG_DIR

        try:
            result = run_patch_in_container(
                instance=self.current_task,
                run_id=run_id,
                log_dir=log_dir,
                timeout=timeout,
                patch=patch if patch.strip() else None,
                commit=instance_id,  # Checkout the instance branch
                f2p_only=f2p_only,
                is_gold=False,  # We're testing a fix, not the gold solution
            )

            if result is None:
                return False, "run_patch_in_container returned None (error occurred)"

            _logger, timed_out = result

            # Read the test output from the log file
            test_output_file = Path(log_dir) / run_id / instance_id / LOG_TEST_OUTPUT
            if test_output_file.exists():
                test_output = test_output_file.read_text()
            else:
                # Test output file not found means tests never ran (e.g., patch apply failed)
                # Check the run_instance.log for more details
                run_log = Path(log_dir) / run_id / instance_id / "run_instance.log"
                if run_log.exists():
                    run_log_content = run_log.read_text()
                    if "Patch Apply Failed" in run_log_content:
                        return False, f"Patch Apply Failed. See {run_log}"
                return False, "Test output file not found - tests never ran"

            # Parse test results - check for failures in the output
            # SWE-smith uses pytest, so we look for standard pytest markers
            passed = not timed_out and "FAILED" not in test_output and "ERROR" not in test_output

            # Also check exit code from the log if available
            if "PASSED" in test_output or "passed" in test_output.lower():
                passed = True

            # Double-check for patch apply failures in test output
            if "Patch Apply Failed" in test_output:
                passed = False

            return passed, test_output

        except Exception as e:
            import traceback

            return False, f"Verification error: {e}\n{traceback.format_exc()}"

    def cleanup(self):
        """Cleanup Docker container after run."""
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
                self.container = None
            except Exception as e:
                print(f"  WARNING: Failed to cleanup container: {e}")
