import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from terminal_bench.agents.terminus_1 import CommandBatchResponse

from gepa import EvaluationBatch, GEPAAdapter


class TerminalBenchTask(BaseModel):
    task_id: str
    model_name: str


def run_agent_tb(
    task_ids: str | list[str],
    run_id: str,
    model_name: str,
    instruction_prompt: str,
    dataset_name: str = "terminal-bench-core",
    dataset_version: str = "head",
    agent_import_path: str = "train_terminus:TerminusWrapper",
    n_concurrent: int = 6,
    prompt_template_path: str = "prompt-templates/instruction_prompt.txt",
):
    """Run the replay agent for multiple task IDs using tb run command."""

    env = os.environ.copy()
    # write instruction prompt to file
    with open(prompt_template_path, "w") as f:
        f.write(instruction_prompt)

    cmd = [
        "tb",
        "run",
        "--dataset-name",
        dataset_name,
        "--dataset-version",
        dataset_version,
        "--agent-import-path",
        agent_import_path,
        "--model-name",
        model_name,
        "--run-id",
        run_id,
        "--n-concurrent",
        str(n_concurrent),
        "--output-path",
        str(Path(os.getcwd()) / "runs"),
    ]
    if isinstance(task_ids, list):
        for task_id in task_ids:
            cmd.extend(["--task-id", task_id])
    else:
        cmd.extend(["--task-id", task_ids])

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=env, cwd=Path(prompt_template_path).parent.parent, check=True)
        print(f"Command completed successfully with return code: {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def get_results(task_id: str, run_id: str) -> tuple[int, list]:

    def _read_episode_response(episode_dir: Path) -> CommandBatchResponse | None:
        """Helper method to read and parse response.json from an episode directory."""
        response_file = episode_dir / "response.json"
        if response_file.exists():
            try:
                response_content = response_file.read_text()
                return CommandBatchResponse.model_validate_json(response_content)
            except Exception:
                pass
        return None

    def _get_logging_dir(task_id: str, run_id: str):
        logging_dir_base = Path("runs") / run_id / task_id
        for dir in logging_dir_base.iterdir():
            if dir.is_dir() and dir.name.startswith(task_id):
                return dir
        raise ValueError(
            f"No logging directory found for task {task_id} and run {run_id}"
        )

    logging_dir = _get_logging_dir(task_id, run_id)
    result_json = logging_dir / "results.json"
    with open(result_json) as f:
        result = json.load(f)
    if result.get("parser_results", None):
        score = sum(x == "passed" for x in result["parser_results"].values())
    else:
        score = 0

    if result.get("is_resolved", None):
        success = True
    else:
        success = False

    failed_reason = result.get("failure_mode", "unknown")

    trajectory_path = logging_dir / "agent-logs"
    episode_dirs = []
    for dir in trajectory_path.iterdir():
        if dir.is_dir() and dir.name.startswith("episode-"):
            episode_dirs.append(dir)

    if episode_dirs:
        # Sort by episode number to get the last one
        episode_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
        last_episode_dir = episode_dirs[-1]

    last_episode_dir_trajectory = last_episode_dir / "debug.json"
    with open(last_episode_dir_trajectory) as f:
        trajectory = json.load(f)

        if "input" in trajectory and isinstance(trajectory["input"], list):
            messages = trajectory["input"]

        # Add the last assistant response using helper method
        parsed_response = _read_episode_response(last_episode_dir)

        if parsed_response:
            assistant_message = {
                "role": "assistant",
                "content": parsed_response.model_dump_json(),
            }
            messages.append(assistant_message)

    return success, score, failed_reason, messages


class TerminusAdapter(GEPAAdapter):

    def __init__(
        self,
        n_concurrent: int = 6,
        instruction_prompt_path: str = "prompt-templates/instruction_prompt.txt",
    ):
        self.n_concurrent = n_concurrent
        self.instruction_prompt_path = instruction_prompt_path

    def evaluate(
        self,
        batch: list[TerminalBenchTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        outputs = []
        scores = []
        trajectories = []
        example_run_id = "temp_gepa_run" + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
        example_model_name = batch[0].model_name

        run_agent_tb(
            [task.task_id for task in batch],
            example_run_id,
            example_model_name,
            instruction_prompt=candidate["instruction_prompt"],
            n_concurrent=self.n_concurrent,
            prompt_template_path=self.instruction_prompt_path,
        )

        for example in batch:
            try:
                success, score, failed_reason, messages = get_results(
                    example.task_id, example_run_id
                )
            except Exception as e:
                print(f"Error running example {example.task_id} {example_run_id}: {e}")
                success = False
                score = 0
                failed_reason = str(e)
                messages = []

            outputs.append(
                f"Terminal Bench outputs are omitted. Please see runs/{example_run_id}/{example.task_id}/ for detailed logging."
            )
            scores.append(score)
            trajectories.append(
                {
                    "messages": messages,
                    "instruction_prompt": candidate["instruction_prompt"],
                    "failed_reason": failed_reason,
                    "success": success,
                }
            )
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
                feedback = (
                    f"Failed to solve the task. Reason: {trajectory['failed_reason']}"
                )
            reflective_dataset["instruction_prompt"].append(
                {
                    "Message History": trajectory["messages"],
                    "Instruction Prompt": candidate["instruction_prompt"],
                    "Feedback": feedback,
                }
            )
        return reflective_dataset
