import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from gepa import EvaluationBatch, GEPAAdapter
from harbor import Job
from harbor.models.job.config import (
    JobConfig,
    OrchestratorConfig,
    RegistryDatasetConfig,
)
from harbor.models.registry import RemoteRegistryInfo
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    VerifierConfig,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.orchestrator_type import OrchestratorType
from harbor.registry.client import RegistryClient


class HarborTerminus2Task(BaseModel):
    task_id: str
    model_name: str
    parser_name: str = "json"  # For Terminus 2: "json" or "xml"


def write_terminus_2_template(instruction_prompt: str, template_path: Path, parser_name: str = "json"):
    """Write the Terminus 2 template file with the instruction prompt baked in.
    
    The instruction_prompt is the ONLY optimizable part - all technical specifications
    about JSON/XML format, keystrokes, duration, etc. are kept fixed.
    """
    

    # de-format instruction_prompt
    instruction_prompt = instruction_prompt.replace("{", "{{").replace("}", "}}")

    # Fixed technical specification for JSON format
    # Based on Harbor's terminus-json-plain.txt template
    if parser_name == "json":
        template_content = f"""
Format your response as JSON with the following structure:

{{{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{{{
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }}}},
    {{{{
      "keystrokes": "cd project\\n",
      "duration": 0.1
    }}}}
  ],
  "task_complete": true
}}}}

{instruction_prompt}

Task Description:
{{instruction}}

Current terminal state:
{{terminal_state}}
"""
    else:  # xml
        template_content = f"""
Format your response as XML with the following structure:

<response>
  <analysis>Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?</analysis>
  <plan>Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.</plan>
  <commands>
    <command>
      <keystrokes>ls -la\\n</keystrokes>
      <duration>0.1</duration>
    </command>
    <command>
      <keystrokes>cd project\\n</keystrokes>
      <duration>0.1</duration>
    </command>
  </commands>
  <task_complete>true</task_complete>
</response>

Required elements:
- <analysis>: Your analysis of the current situation
- <plan>: Your plan for the next steps
- <commands>: Container for command elements

Optional elements:
- <task_complete>: Boolean indicating if the task is complete (defaults to false if not present)

Command element structure:
- <keystrokes>: String containing the exact keystrokes to send to the terminal (required)
- <duration>: Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside <keystrokes> will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- Most bash commands should end with a newline (\\n) to cause them to execute
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

The <duration> element specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running <command><keystrokes></keystrokes><duration>10.0</duration></command> on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command
- Extra text before or after the XML will generate warnings but be tolerated
- The XML must be valid - use proper escaping for special characters
- Commands container can be empty if you want to wait without taking action

Task Description:
{{instruction}}

Current terminal state:
{{terminal_state}}
"""
    
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(template_content)


def get_results_from_harbor_job(job_dir: Path) -> tuple[list[bool], list[float], list[str], list[dict]]:
    """Extract results from a completed Harbor job."""
    successes = []
    scores = []
    failed_reasons = []
    trajectories = []
    
    # Iterate through trial directories
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
            
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            print(f"Warning: No result found for {trial_dir.name}")
            continue
            
        with open(result_path) as f:
            result = json.load(f)
        
        # Extract success and score from verifier results
        # Harbor's result format:
        # - verifier_result can be None or a dict with "rewards"
        # - exception_info contains error information if task failed
        # - rewards is typically {"reward": 0.0 or 1.0} or multiple rewards
        verifier_result = result.get("verifier_result")
        exception_info = result.get("exception_info")
        
        if verifier_result is not None:
            rewards = verifier_result.get("rewards", {})
            # Extract the main "reward" field (typically 0.0 or 1.0)
            reward = rewards.get("reward", 0.0)
            
            # Success if reward is 1.0 and no exception
            success = (reward == 1.0)
            
            # Score is the reward value (0.0 or 1.0)
            score = reward
        else:
            success = False
            score = 0.0
        
        # Determine failure reason
        if exception_info:
            failed_reason = f"{exception_info.get('exception_type', 'unknown')}: {exception_info.get('exception_message', 'unknown')}"
            # Read full exception traceback for non-timeout errors
            if exception_info.get("exception_type") != "AgentTimeoutError":
                exception_file = trial_dir / "exception.txt"
                if exception_file.exists():
                    with open(exception_file) as f:
                        exception_text = f.read()
                    failed_reason += f"\n\n{exception_text}"
        elif not success and verifier_result:
            failed_reason = "Test failed. The trajectory failed to meet the task description."
        elif verifier_result is None:
            failed_reason = "Test failed. The verifier result is missing."
        else:
            failed_reason = "Test failed. Unknown reason."
        
        # Extract messages from the last episode's debug.json
        messages = []
        try:
            agent_dir = trial_dir / "agent"
            episode_dirs = sorted(
                [d for d in agent_dir.iterdir() if d.is_dir() and d.name.startswith("episode-")],
                key=lambda d: int(d.name.split("-")[1])
            )
            last_episode_dir = episode_dirs[-1]
            
            with open(last_episode_dir / "debug.json") as f:
                debug_data = json.load(f)
                messages = debug_data.get("messages", [])
                
                # For all messages, if content is a list, extract the first text content
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        msg["content"] = msg["content"][0]["text"]
                
                # Parse and add original_response as the last message
                original_response = debug_data.get("original_response")
                response_data = json.loads(original_response)
                messages.append({
                    "role": "assistant",
                    "content": response_data["content"][0]["text"]
                })

            
        except (FileNotFoundError, IndexError, json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to extract messages for {trial_dir.name}: {e}")
        
        successes.append(success)
        scores.append(score)
        failed_reasons.append(failed_reason)
        trajectories.append({
            "messages": messages,
            "success": success,
            "failed_reason": failed_reason,
        })
    
    return successes, scores, failed_reasons, trajectories


class Terminus2HarborAdapter(GEPAAdapter):
    def __init__(
        self,
        n_concurrent: int = 4,
        parser_name: str = "json",
        agent_import_path: str = "train_terminus_2_harbor:Terminus2Wrapper",
        template_dir: Path = None,
        default_dataset_name: str = "tb-lite-beta@0.0",
    ):
        self.n_concurrent = n_concurrent
        self.parser_name = parser_name
        self.agent_import_path = agent_import_path
        self.default_dataset_name = default_dataset_name
        
        # Set up template directory
        if template_dir is None:
            template_dir = Path(__file__).parent.parent.parent / "examples/terminal-bench/prompt-templates"
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Template path for Terminus 2
        self.template_path = self.template_dir / "terminus_2_harbor.txt"

    def evaluate(
        self,
        batch: list[HarborTerminus2Task],
        candidate: dict[str, str],
        capture_traces: bool = False,
        dataset_name: str | None = None,
        job_name: str = "gepa_terminus2",
    ) -> EvaluationBatch:
        """
        Evaluate a batch of tasks using Harbor.
        
        Args:
            batch: List of tasks to evaluate
            candidate: Dictionary containing "instruction_prompt" key
            capture_traces: Whether to capture execution traces
            dataset_name: Harbor dataset to use (e.g., "tb-lite-beta@0.0" or "terminal-bench@2.0")
                         If None, uses the default_dataset_name from initialization
        """
        if dataset_name is None:
            dataset_name = self.default_dataset_name
        # Write the template file with instruction prompt baked in
        write_terminus_2_template(
            candidate["instruction_prompt"],
            self.template_path,
            self.parser_name,
        )
        
        # Create unique job directory
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        job_name = f"{job_name}_{timestamp}"
        job_dir = Path("runs") / job_name
        
        # Extract task IDs from batch
        task_ids = [task.task_id for task in batch]
        model_name = batch[0].model_name
        
        # Parse dataset name (format: "name@version" or just "name")
        if "@" in dataset_name:
            ds_name, ds_version = dataset_name.split("@", 1)
        else:
            ds_name, ds_version = dataset_name, "head"
        
        # Create Harbor job config
        # Note: We filter to specific task names by passing them as exact patterns
        job_config = JobConfig(
            job_name=job_name,
            jobs_dir=job_dir.parent,
            datasets=[
                RegistryDatasetConfig(
                    registry=RemoteRegistryInfo(url="https://raw.githubusercontent.com/laude-institute/harbor/dc62fd28edc087e64fde3bfa0bfd22d5003d2184/registry.json"),
                    name=ds_name,
                    version=ds_version,
                    task_names=task_ids if task_ids else None,  # Filter to specific task names (exact match)
                )
            ],
            agents=[
                AgentConfig(
                    name="terminus-2-gepa-adapter",
                    import_path=self.agent_import_path,
                    model_name=model_name,
                    kwargs={"parser_name": self.parser_name},
                )
            ],
            environment=EnvironmentConfig(
                type=EnvironmentType.DOCKER,
            ),
            verifier=VerifierConfig(),
            orchestrator=OrchestratorConfig(
                type=OrchestratorType.LOCAL,
                n_concurrent_trials=self.n_concurrent,
            ),
        )
        
        # Run the job
        print(f"Running Harbor job: {job_name}")
        print(f"Dataset: {dataset_name}, Tasks: {len(task_ids)}, Model: {model_name}")
        print(f"Task names to filter: {task_ids[:5]}...")  # Show first 5 task names
        
        try:
            job = Job(job_config)
            asyncio.run(job.run())
        except Exception as e:
            print(f"Error running Harbor job: {e}")
            # Return empty results on failure
            return EvaluationBatch(
                outputs=[f"Error: {e}"] * len(batch),
                scores=[0.0] * len(batch),
                trajectories=[{
                    "messages": [],
                    "instruction_prompt": candidate["instruction_prompt"],
                    "failed_reason": str(e),
                    "success": False,
                }] * len(batch),
            )
        
        # Extract results
        successes, scores, failed_reasons, traj_list = get_results_from_harbor_job(job_dir)
        
        # Build outputs and trajectories
        outputs = []
        trajectories = []
        
        for i, (success, score, failed_reason, traj) in enumerate(
            zip(successes, scores, failed_reasons, traj_list, strict=False)
        ):
            outputs.append(
                f"Harbor outputs are omitted. Please see {job_dir}/{task_ids[i]}/ for detailed logging."
            )
            trajectories.append({
                "messages": traj["messages"],
                "instruction_prompt": candidate["instruction_prompt"],
                "failed_reason": failed_reason,
                "success": success,
            })
        
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ):
        reflective_dataset = {"instruction_prompt": []}
        for _score, trajectory in zip(eval_batch.scores, eval_batch.trajectories, strict=False):
            if trajectory["success"]:
                feedback = "Successfully solved the task!"
            else:
                feedback = f"Failed to solve the task. Reason: {trajectory['failed_reason']}"
            reflective_dataset["instruction_prompt"].append(
                {
                    "Message History": trajectory["messages"], # TODO make some compaction of messages
                    "Instruction Prompt": candidate["instruction_prompt"],
                    "Feedback": feedback,
                }
            )
        return reflective_dataset

