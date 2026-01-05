# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol, TypedDict, cast

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


# DataInst, Trajectory, RolloutOutput
class DefaultDataInst(TypedDict):
    input: str
    additional_context: dict[str, str]
    answer: str


class EvaluationResult(NamedTuple):
    score: float
    feedback: str
    objective_scores: dict[str, float] | None = None


class DefaultTrajectory(TypedDict):
    data: DefaultDataInst
    full_assistant_response: str
    feedback: str


class DefaultRolloutOutput(TypedDict):
    full_assistant_response: str


DefaultReflectiveRecord = TypedDict(
    "DefaultReflectiveRecord",
    {
        "Inputs": str,
        "Generated Outputs": str,
        "Feedback": str,
    },
)


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatCompletionCallable(Protocol):
    """Protocol for chat completion callables (duck typing for custom model wrappers)."""

    def __call__(self, messages: Sequence[ChatMessage]) -> str: ...


# Callable that evaluates a response and returns (score, feedback, optional objective_scores)
class Evaluator(Protocol):
    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        """
        Evaluates a response and returns a score, feedback, and optional objective scores.
        """
        ...


class ContainsAnswerEvaluator:
    """Default evaluator that checks if the expected answer is contained in the response."""

    def __init__(self, failure_score: float = 0.0):
        self.failure_score = failure_score

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        is_correct = data["answer"] in response
        score = 1.0 if is_correct else self.failure_score

        if is_correct:
            feedback = f"The generated response is correct. The response include the correct answer '{data['answer']}'"
        else:
            additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
            feedback = (
                f"The generated response is incorrect. The correct answer is '{data['answer']}'. "
                "Ensure that the correct answer is included in the response exactly as it is."
            )
            if additional_context_str:
                feedback += f" Here is some additional context that might be helpful:\n{additional_context_str}"

        return EvaluationResult(score=score, feedback=feedback, objective_scores=None)


class DefaultAdapter(GEPAAdapter[DefaultDataInst, DefaultTrajectory, DefaultRolloutOutput]):
    def __init__(
        self,
        model: str | ChatCompletionCallable,
        evaluator: Evaluator | None = None,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
    ):
        if isinstance(model, str):
            import litellm

            self.litellm = litellm
        self.model = model
        self.evaluator = evaluator or ContainsAnswerEvaluator()
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}

    def evaluate(
        self,
        batch: list[DefaultDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput]:
        outputs: list[DefaultRolloutOutput] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float] | None] = []
        trajectories: list[DefaultTrajectory] | None = [] if capture_traces else None

        system_content = next(iter(candidate.values()))

        litellm_requests = []

        for data in batch:
            user_content = f"{data['input']}"

            messages: list[ChatMessage] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            litellm_requests.append(messages)

        if isinstance(self.model, str):
            responses = [
                resp.choices[0].message.content.strip()
                for resp in self.litellm.batch_completion(
                    model=self.model,
                    messages=litellm_requests,
                    max_workers=self.max_litellm_workers,
                    **self.litellm_batch_completion_kwargs,
                )
            ]
        else:
            responses = [self.model(messages) for messages in litellm_requests]

        for data, assistant_response in zip(batch, responses, strict=True):
            eval_result = self.evaluator(data, assistant_response)
            score = eval_result.score
            feedback = eval_result.feedback
            obj_scores = eval_result.objective_scores

            output: DefaultRolloutOutput = {"full_assistant_response": assistant_response}

            outputs.append(output)
            scores.append(score)
            objective_scores.append(obj_scores)

            if trajectories is not None:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                        "feedback": feedback,
                    }
                )

        objective_scores_arg: list[dict[str, float]] | None = None
        if objective_scores:
            all_none = all(x is None for x in objective_scores)
            all_not_none = all(x is not None for x in objective_scores)
            if not (all_none or all_not_none):
                raise ValueError("Objective scores must either be all None or all not None.")
            if all_not_none:
                objective_scores_arg = cast(list[dict[str, float]], objective_scores)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores_arg,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        ret_d: dict[str, list[DefaultReflectiveRecord]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        trajectories = eval_batch.trajectories
        assert trajectories is not None, "Trajectories are required to build a reflective dataset."

        items: list[DefaultReflectiveRecord] = []

        for traj in trajectories:
            d: DefaultReflectiveRecord = {
                "Inputs": traj["data"]["input"],
                "Generated Outputs": traj["full_assistant_response"],
                "Feedback": traj["feedback"],
            }

            items.append(d)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d
