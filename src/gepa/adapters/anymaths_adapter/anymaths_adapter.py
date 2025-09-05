from typing import Any, TypedDict

import litellm
from pydantic import BaseModel, Field

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


class AnyMathsDataInst(TypedDict):
    input: str
    additional_context: dict[str, str]
    answer: str


class AnyMathsTrajectory(TypedDict):
    data: AnyMathsDataInst
    full_assistant_response: str


class AnyMathsRolloutOutput(TypedDict):
    full_assistant_response: str


class AnyMathsStructuredOutput(BaseModel):
    final_answer: str = Field(
        ..., description="The final answer to the mathematical problem (i.e., no units, no other text)"
    )
    solution_pad: str = Field(..., description="The solution pad containing the step-by-step solution to the problem.")


class AnyMathsAdapter(GEPAAdapter[AnyMathsDataInst, AnyMathsTrajectory, AnyMathsRolloutOutput]):
    """AnyMaths Adapter is a GEPAAdapter for any dataset that contains mathematical word problems
    of varying complexity and structure. It is designed to handle a wide range of mathematical
    tasks, including arithmetic, algebra, and more.

    Note: Ollama must be installed and configured to use this adapter.
    """

    def __init__(
        self,
        model: str,
        failure_score: float = 0.0,
        api_base: str | None = "http://localhost:11434",
        max_litellm_workers: int = 10,
    ) -> None:
        import litellm

        self.model = model
        self.failure_score = failure_score
        self.litellm = litellm
        self.max_litellm_workers = max_litellm_workers
        self.api_base = api_base

        if self.model.startswith("ollama"):
            assert self.api_base is not None, "API base URL must be provided when using Ollama."

        if self.api_base is None or self.api_base == "":
            self.api_base = None

    def evaluate(
        self,
        batch: list[AnyMathsDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[AnyMathsTrajectory, AnyMathsRolloutOutput]:
        import ast

        outputs: list[AnyMathsRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[AnyMathsTrajectory] | None = [] if capture_traces else None

        if not candidate:
            raise ValueError("Candidate must contain at least one component text.")

        system_content = next(iter(candidate.values()))

        litellm_requests = []

        for data in batch:
            user_content = f"{data['input']}"

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            litellm_requests.append(messages)

        try:
            responses = self.litellm.batch_completion(
                model=self.model,
                messages=litellm_requests,
                api_base=self.api_base,
                max_workers=self.max_litellm_workers,
                format=AnyMathsStructuredOutput.model_json_schema(),
                response_format={
                    "type": "json_object",
                    "response_schema": AnyMathsStructuredOutput.model_json_schema(),
                    "enforce_validation": True,
                },
            )
        except litellm.exceptions.JSONSchemaValidationError as e:
            raise e

        for data, response in zip(batch, responses, strict=False):
            correct_output_format = True
            try:
                assistant_response = ast.literal_eval(response.choices[0].message.content.strip())
            except Exception:
                assistant_response = "Assistant failed to respond with the correct answer or format."
                correct_output_format = False

            if correct_output_format:
                structured_assistant_response = f"Assistant's Solution: {assistant_response['solution_pad']}\n"
                structured_assistant_response += f"Final Answer: {assistant_response['final_answer']}"
                output = {"full_assistant_response": structured_assistant_response}
                score = 1.0 if data["answer"] in assistant_response["final_answer"] else self.failure_score
            else:
                output = {"full_assistant_response": assistant_response}
                score = self.failure_score

            outputs.append(output)
            scores.append(score)

            if capture_traces:
                trajectories.append({"data": data, "full_assistant_response": output["full_assistant_response"]})
        # Return results for the entire batch (not just the first item)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[AnyMathsTrajectory, AnyMathsRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        ret_d: dict[str, list[dict[str, Any]]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        items: list[dict[str, Any]] = []
        trace_instances = list(zip(eval_batch.trajectories, eval_batch.scores, eval_batch.outputs, strict=False))

        for trace_instance in trace_instances:
            traj, score, _ = trace_instance
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]

            if score > 0.0:
                feedback = f"The generated response is correct. The final answer is: {data['answer']}."
            else:
                additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
                if additional_context_str:
                    feedback = (
                        f"The generated response is incorrect. The correct answer is: {data['answer']}. "
                        "Ensure that the correct answer is included in the response exactly as it is. "
                        f"Here is some additional context that might be helpful:\n{additional_context_str}"
                    )
                else:
                    feedback = (
                        f"The generated response is incorrect. The correct answer is: {data['answer']}. "
                        "Ensure that the correct answer is included in the response exactly as it is."
                    )

            d = {"Inputs": data["input"], "Generated Outputs": generated_outputs, "Feedback": feedback}

            items.append(d)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d
