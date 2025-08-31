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
   scores = {}
   if result.get("parser_results", None):
       parser_passed = sum(map(lambda x: x == "passed", result["parser_results"].values()))
       parser_total = len(result["parser_results"])
       scores["parser_accuracy"] = parser_passed / parser_total if parser_total > 0 else 0.0
       scores["parser_passed"] = float(parser_passed)
       scores["parser_total"] = float(parser_total)
   else:
       scores["parser_accuracy"] = 0.0
       scores["parser_passed"] = 0.0
       scores["parser_total"] = 0.0


   if result.get("is_resolved", None):
       success = True
       scores["task_resolved"] = 1.0
   else:
       success = False
       scores["task_resolved"] = 0.0


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


   return success, scores, failed_reason, messages




class TerminusAdapter(GEPAAdapter):


   def __init__(
       self,
       n_concurrent: int = 6,
       instruction_prompt_path: str = "prompt-templates/instruction_prompt.txt",
       score_weights: dict[str, float] | None = None,
       aggregation_method: str = "weighted_sum",
   ):
       self.n_concurrent = n_concurrent
       self.instruction_prompt_path = instruction_prompt_path
       self.score_weights = score_weights or {
           "parser_accuracy": 0.7,
           "task_resolved": 0.3
       }
       self.aggregation_method = aggregation_method
   def _aggregate_scores(self, scores: dict[str, float]) -> float:
       """
       this was done to convert multiple scores to one score for GEPA to be able to optimize.
       users can choose the different aggregation strategies.
       """
       if self.aggregation_method == "weighted_sum":
           return sum(
               score * self.score_weights.get(name, 1.0)
               for name, score in scores.items()
           )
       elif self.aggregation_method == "geometric_mean":
           import math
           return math.prod(scores.values()) ** (1.0 / len(scores))
       elif self.aggregation_method == "min":
           return min(scores.values())
       elif self.aggregation_method == "max":
           return max(scores.values())
       elif self.aggregation_method == "arithmetic_mean":
           return sum(scores.values()) / len(scores)
       else:
           # this is when we resort to default to weighted sum
           return sum(
               score * self.score_weights.get(name, 1.0)
               for name, score in scores.items()
           )
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
               success, multi_scores, failed_reason, messages = get_results(
                   example.task_id, example_run_id
               )
           except Exception as e:
               print(f"Error running example {example.task_id} {example_run_id}: {e}")
               success = False
               multi_scores = {
                   "parser_accuracy": 0.0,
                   "parser_passed": 0.0,
                   "parser_total": 0.0,
                   "task_resolved": 0.0
               }
               failed_reason = str(e)
               messages = []


           aggregated_score = self._aggregate_scores(multi_scores)


           outputs.append(
               f"Terminal Bench outputs are omitted. Please see runs/{example_run_id}/{example.task_id}/ for detailed logging."
           )
           scores.append(aggregated_score)
           if capture_traces:
               trajectories.append(
                   {
                       "messages": messages,
                       "instruction_prompt": candidate["instruction_prompt"],
                       "failed_reason": failed_reason,
                       "success": success,
                       "raw_scores": multi_scores,
                       "aggregated_score": aggregated_score,
                       "aggregation_method": self.aggregation_method,
                       "score_weights": self.score_weights, 
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
       for score, trajectory in zip(eval_batch.scores, eval_batch.trajectories, strict=False):
           if trajectory["success"]:
               feedback = "Successfully solved the task!"
           else:
               feedback = (
                   f"Failed to solve the task. Reason: {trajectory['failed_reason']}"
               )
           # adding the multi-score information to feedback
           raw_scores = trajectory.get("raw_scores", {})
           if raw_scores:
               score_details = ", ".join([f"{k}: {v:.3f}" for k, v in raw_scores.items()])
               feedback += f" Score breakdown: {score_details}"
              
           reflective_dataset["instruction_prompt"].append(
               {
                   "Message History": trajectory["messages"],
                   "Instruction Prompt": candidate["instruction_prompt"],
                   "Feedback": feedback,
               }
           )
       return reflective_dataset
