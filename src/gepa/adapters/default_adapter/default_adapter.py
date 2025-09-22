# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any, Callable, TypedDict

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


# DataInst, Trajectory, RolloutOutput
class DefaultDataInst(TypedDict):
    input: str
    additional_context: dict[str, str]
    answer: str

class DefaultTrajectory(TypedDict):
    data: DefaultDataInst
    full_assistant_response: str

class DefaultRolloutOutput(TypedDict):
    full_assistant_response: str

class DefaultAdapter(GEPAAdapter[DefaultDataInst, DefaultTrajectory, DefaultRolloutOutput]):
    def __init__(
        self,
        model: str | Callable,
        failure_score: float = 0.0,
        max_litellm_workers: int = 10,
    ):
        if isinstance(model, str):
            import litellm
            self.litellm = litellm
        self.model = model

        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers

    def evaluate(
        self,
        batch: list[DefaultDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput]:
        outputs: list[DefaultRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[DefaultTrajectory] | None = [] if capture_traces else None

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
            if isinstance(self.model, str):
                responses = [resp.choices[0].message.content.strip() for resp in self.litellm.batch_completion(model=self.model, messages=litellm_requests, max_workers=self.max_litellm_workers)]
            else:
                responses = [self.model(messages) for messages in litellm_requests]
        except Exception as e:
            raise e

        for data, assistant_response in zip(batch, responses, strict=False):
            output = {"full_assistant_response": assistant_response}
            score = 1.0 if data["answer"] in assistant_response else 0.0

            outputs.append(output)
            scores.append(score)

            if capture_traces:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput],
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
                feedback = f"The generated response is correct. The response include the correct answer '{data['answer']}'"
            else:
                additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
                feedback = f"The generated response is incorrect. The correct answer is '{data['answer']}'. Ensure that the correct answer is included in the response exactly as it is. Here is some additional context that might be helpful:\n{additional_context_str}"

            d = {
                "Inputs": data["input"],
                "Generated Outputs": generated_outputs,
                "Feedback": feedback,
            }

            items.append(d)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d
