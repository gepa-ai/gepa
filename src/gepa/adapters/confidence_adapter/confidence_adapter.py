# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Confidence-aware adapter for structured-output classification tasks.

Uses token-level logprobs from ``llm-structured-confidence`` to detect
and penalise lucky guesses, feeding rich confidence diagnostics into
the reflection LLM.  Requires structured output with enum constraints
and a model that supports ``logprobs=True``.

See the full guide: https://gepa-ai.github.io/gepa/guides/confidence-adapter/
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict

from gepa.adapters.confidence_adapter.scoring import LinearBlendScoring, ScoringStrategy
from gepa.core.adapter import EvaluationBatch, GEPAAdapter, ProposalFn

logger = logging.getLogger(__name__)

TOP_ALTERNATIVES_IN_FEEDBACK = 3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class ConfidenceDataInst(TypedDict):
    """Input data instance for confidence-aware evaluation.

    Attributes
    ----------
    input:
        The user-facing text to classify (e.g. a transaction description).
    additional_context:
        Optional key-value pairs surfaced in reflective feedback.
    answer:
        The expected classification label (must match an enum value).
    """

    input: str
    additional_context: dict[str, str]
    answer: str


class ConfidenceTrajectory(TypedDict):
    """Per-example execution trace used to build reflective feedback.

    Attributes
    ----------
    data:
        The original input data instance.
    full_assistant_response:
        Raw text returned by the LLM.
    parsed_value:
        Value extracted from the JSON response at ``answer_field``.
    logprob_score:
        Joint logprob (sum of per-token logprobs) for the target field.
        Always ``<= 0``; closer to 0 = more confident.  ``None`` when
        logprobs are unavailable.
    top_alternatives:
        Top alternative tokens with their probabilities and resolved
        enum values (when ``response_schema`` is provided).
    is_correct:
        Whether ``parsed_value`` matches the expected answer.
    score:
        Blended score produced by the scoring strategy.
    feedback:
        Human-readable feedback string including confidence details.
    """

    data: ConfidenceDataInst
    full_assistant_response: str
    parsed_value: str | None
    logprob_score: float | None
    top_alternatives: list[dict[str, Any]]
    is_correct: bool
    score: float
    feedback: str


class ConfidenceRolloutOutput(TypedDict):
    """Per-example output exposed to GEPA (opaque to the engine).

    Attributes
    ----------
    full_assistant_response:
        Raw text returned by the LLM.
    parsed_value:
        Value extracted from the JSON response at ``answer_field``.
    logprob_score:
        Joint logprob for the target field.
    """

    full_assistant_response: str
    parsed_value: str | None
    logprob_score: float | None


ConfidenceReflectiveRecord = TypedDict(
    "ConfidenceReflectiveRecord",
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
    """Protocol for callables that return the raw LLM response object.

    The callable must return the **full** response object (e.g.
    ``litellm.ModelResponse``) so that logprobs can be extracted.
    """

    def __call__(self, messages: Sequence[ChatMessage]) -> Any: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_answer_from_json(text: str, field_path: str) -> str | None:
    """Walk *field_path* (dot-separated) into parsed JSON and return the leaf value."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    for key in field_path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return str(obj) if obj is not None else None


def _build_feedback(
    *,
    is_correct: bool,
    expected: str,
    got: str | None,
    logprob_score: float | None,
    top_alternatives: list[dict[str, Any]],
    additional_context: dict[str, str],
    high_confidence_prob: float,
    low_confidence_prob: float,
) -> str:
    """Build feedback for the reflection LLM using probability only.

    Design principles:
    - **Correct + confident**: just ``"Correct."`` -- no noise.
    - **Correct + uncertain**: flag it as risky with alternatives.
    - **Incorrect**: rich detail scaled by confidence -- the higher the
      model's certainty on the wrong answer, the stronger the language.

    Parameters
    ----------
    high_confidence_prob:
        Probability threshold above which a prediction is considered
        "highly confident".
    low_confidence_prob:
        Probability threshold below which a *correct* prediction is
        labelled "unreliable".
    """
    probability = math.exp(logprob_score) if logprob_score is not None else None
    got_str = got or "<parse error>"

    if is_correct:
        if probability is None or probability >= high_confidence_prob:
            feedback = "Correct."
        elif probability < low_confidence_prob:
            alt_str = _format_alternatives(top_alternatives, exclude=expected)
            feedback = (
                f"Correct but uncertain ({probability:.0%} probability). "
                f"Model answered '{expected}' but was nearly split with alternatives."
            )
            if alt_str:
                feedback += f" Top alternatives: {alt_str}."
            feedback += " The model cannot reliably distinguish between these categories with the current prompt."
        else:
            alt_str = _format_alternatives(top_alternatives, exclude=expected)
            feedback = f"Correct ({probability:.0%} probability)."
            if alt_str:
                feedback += f" Close alternatives: {alt_str}."
    else:
        alt_str = _format_alternatives(top_alternatives, exclude=got_str)
        correct_alt_prob = _find_alternative_prob(top_alternatives, expected)

        if probability is not None and probability >= high_confidence_prob:
            feedback = (
                f"WRONG — model has {probability:.0%} certainty on '{got_str}' "
                f"but the correct answer is '{expected}'. "
                "The model has no doubt about its wrong answer; "
                "the prompt is actively misleading it for this type of input."
            )
            if correct_alt_prob is not None:
                feedback += f" The correct category '{expected}' only had {correct_alt_prob:.1%} probability."
            if alt_str:
                feedback += f" Alternatives: {alt_str}."
            feedback += f" The prompt must add explicit rules to disambiguate '{got_str}' vs '{expected}'."
        elif probability is not None and probability >= low_confidence_prob:
            feedback = f"Wrong ({probability:.0%} probability). Expected '{expected}' but got '{got_str}'."
            if alt_str:
                feedback += f" Alternatives: {alt_str}."
            feedback += " The prompt should better guide the model for this case."
        else:
            prob_str = f"{probability:.0%} probability" if probability is not None else "unknown confidence"
            feedback = (
                f"Wrong ({prob_str}). "
                f"Expected '{expected}' but got '{got_str}'. "
                "The model was uncertain — better prompt guidance could fix this."
            )
            if alt_str:
                feedback += f" Alternatives: {alt_str}."

        ctx = "\n".join(f"{k}: {v}" for k, v in additional_context.items())
        if ctx:
            feedback += f"\nAdditional context:\n{ctx}"

    return feedback


def _find_alternative_prob(
    alts: list[dict[str, Any]],
    target: str,
) -> float | None:
    """Find the probability of *target* in the alternatives list."""
    for alt in alts:
        val = alt.get("resolved_value") or alt.get("token", "")
        if val == target:
            return alt.get("probability", 0.0)
    return None


def _format_alternatives(alts: list[dict[str, Any]], exclude: str | None = None) -> str:
    parts: list[str] = []
    for alt in alts[:TOP_ALTERNATIVES_IN_FEEDBACK]:
        val = alt.get("resolved_value") or alt.get("token", "")
        prob = alt.get("probability", 0.0)
        if val and val != exclude:
            parts.append(f"'{val}' ({prob:.0%})")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# ConfidenceAdapter
# ---------------------------------------------------------------------------


class ConfidenceAdapter(GEPAAdapter[ConfidenceDataInst, ConfidenceTrajectory, ConfidenceRolloutOutput]):
    """GEPA adapter for structured-output classification with logprob confidence.


    This adapter is specifically designed for **classification tasks** where
    the LLM returns a structured JSON output with an ``enum``-constrained
    field.  The ``enum`` constraint is critical: it forces the model to
    choose from a closed set of categories, and the logprobs then represent
    the model's true probability distribution over those categories.

    The adapter owns the full LLM call lifecycle:

    1. Sends requests with ``logprobs=True`` and ``response_format``
    2. Parses the structured JSON output
    3. Extracts the **joint logprob** (sum of per-token logprobs) for the
       target field via ``llm-structured-confidence``
    4. Computes a blended score via the pluggable :class:`ScoringStrategy`
    5. Generates rich reflective feedback with confidence details

    Why joint logprob?
    ~~~~~~~~~~~~~~~~~~
    The ``joint_logprob`` is the sum of all per-token logprobs for the
    value tokens of the target field.  For example, if the model outputs
    ``"Bills/Electricity"`` and the tokens are ``["Bills", "/", "Elec",
    "tricity"]`` with logprobs ``[-0.02, -0.01, -0.10, -0.01]``, the
    joint logprob is ``-0.14``.

    This is the most natural confidence measure because:

    * It captures the **total uncertainty** across all tokens (not just
      the average), so longer values with one uncertain token are correctly
      penalised.
    * ``exp(joint_logprob)`` gives the **joint probability** -- the
      probability the model assigns to the entire value as a whole.
    * It is numerically stable and works well across different tokenisations.

    Parameters
    ----------
    model:
        Either a litellm model string (e.g. ``"openai/gpt-4.1-mini"``) or a
        callable that takes ``messages`` and returns the full response object
        (must include logprobs).
    field_path:
        Path to the target field in the JSON response, using the syntax of
        ``llm-structured-confidence``: ``"category_name"`` for a top-level
        field, ``"classification.name"`` for a nested field, or
        ``"results[].category"`` for an array of objects.
    response_format:
        JSON schema dict passed to litellm as ``response_format``.  Should
        define ``enum`` constraints on the target field for meaningful
        confidence extraction.  Required when *model* is a string.
    response_schema:
        Optional Pydantic model or dict schema passed to
        ``extract_logprobs(response_schema=…)`` for enum resolution.
        When provided, ``TopAlternative.resolved_value`` maps token
        prefixes back to full enum values (e.g. ``"Pos"`` -> ``"Positive"``).
    scoring_strategy:
        How to blend correctness and logprob confidence into a single score.
        Defaults to :class:`LinearBlendScoring`.
    answer_field:
        JSON field path used to extract the answer from the response text.
        Defaults to *field_path* (they are usually the same).
    high_confidence_threshold:
        Probability threshold (in ``(0, 1]``) above which a prediction is
        labelled "high confidence" in reflective feedback.  Models using
        structured output with enum constraints typically produce
        probabilities above 95%, so this should be set high (e.g. ``0.99``)
        to produce useful feedback gradients.  Default ``0.99``.
    low_confidence_threshold:
        Probability threshold (in ``(0, 1)``) below which a *correct*
        prediction is labelled "unreliable" in reflective feedback.
        Default ``0.90``.
    top_logprobs:
        Number of top logprobs to request from the LLM (1-20).
    failure_score:
        Score assigned when an example fails (parse error, API error, etc.).
    max_litellm_workers:
        Concurrency for litellm calls (only used when *model* is a string).
    litellm_batch_completion_kwargs:
        Extra keyword arguments forwarded to every ``litellm.batch_completion``
        call (e.g. ``temperature``, ``max_tokens``).
    """

    propose_new_texts: ProposalFn | None = None

    def __init__(
        self,
        model: str | ChatCompletionCallable,
        field_path: str,
        response_format: dict[str, Any] | None = None,
        response_schema: type | dict[str, Any] | None = None,
        scoring_strategy: ScoringStrategy | None = None,
        answer_field: str | None = None,
        high_confidence_threshold: float = 0.99,
        low_confidence_threshold: float = 0.90,
        top_logprobs: int = 5,
        failure_score: float = 0.0,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(model, str):
            import litellm

            self.litellm = litellm
            if response_format is None:
                raise ValueError(
                    "response_format is required when model is a string (LiteLLM path). "
                    "Provide a JSON schema with enum constraints for structured classification output."
                )
        self.model = model
        self.field_path = field_path
        self.response_format = response_format
        self.response_schema = response_schema
        self.scoring_strategy: ScoringStrategy = scoring_strategy or LinearBlendScoring(
            low_confidence_threshold=high_confidence_threshold,
        )
        self.answer_field = answer_field or field_path
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.top_logprobs = top_logprobs
        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def _process_response(
        self,
        response: Any,
        data: ConfidenceDataInst,
        capture_traces: bool,
    ) -> tuple[ConfidenceRolloutOutput, float, dict[str, float], ConfidenceTrajectory | None]:
        """Extract logprobs, compute score, and build feedback from a single response."""
        from llm_structured_confidence import extract_logprobs

        try:
            if isinstance(response, Exception):
                raise response

            response_text = self._extract_text(response)
            parsed_value = _extract_answer_from_json(response_text, self.answer_field)

            logprob_score: float | None = None
            top_alternatives: list[dict[str, Any]] = []

            try:
                entries = extract_logprobs(
                    response,
                    field_path=self.field_path,
                    response_schema=self.response_schema,
                )
                if entries:
                    fl = entries[0].field_logprob
                    logprob_score = fl.joint_logprob
                    top_alternatives = [
                        {
                            "token": alt.token,
                            "probability": alt.probability,
                            "resolved_value": alt.resolved_value,
                        }
                        for alt in fl.top_logprobs
                    ]
            except Exception:
                logger.debug("Logprob extraction failed for an example", exc_info=True)

            is_correct = self._check_correctness(parsed_value, data["answer"])
            score = self.scoring_strategy.score(is_correct, logprob_score)

        except Exception:
            logger.debug("LLM call or parsing failed for an example", exc_info=True)
            response_text = ""
            parsed_value = None
            logprob_score = None
            top_alternatives = []
            is_correct = False
            score = self.failure_score

        feedback = _build_feedback(
            is_correct=is_correct,
            expected=data["answer"],
            got=parsed_value,
            logprob_score=logprob_score,
            top_alternatives=top_alternatives,
            additional_context=data.get("additional_context", {}),
            high_confidence_prob=self.high_confidence_threshold,
            low_confidence_prob=self.low_confidence_threshold,
        )

        output: ConfidenceRolloutOutput = {
            "full_assistant_response": response_text,
            "parsed_value": parsed_value,
            "logprob_score": logprob_score,
        }

        probability = math.exp(logprob_score) if logprob_score is not None else 0.0
        obj_scores = {
            "accuracy": 1.0 if is_correct else 0.0,
            "probability": probability,
        }

        trajectory = None
        if capture_traces:
            trajectory = {
                "data": data,
                "full_assistant_response": response_text,
                "parsed_value": parsed_value,
                "logprob_score": logprob_score,
                "top_alternatives": top_alternatives,
                "is_correct": is_correct,
                "score": score,
                "feedback": feedback,
            }

        return output, score, obj_scores, trajectory

    def evaluate(
        self,
        batch: list[ConfidenceDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ConfidenceTrajectory, ConfidenceRolloutOutput]:
        """Run *candidate* on *batch*, extracting logprob confidence.

        Uses ``litellm.batch_completion`` for parallel LLM calls, then
        post-processes each response to extract logprobs and build feedback.
        """
        system_content = next(iter(candidate.values()))

        all_messages: list[list[ChatMessage]] = [
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data["input"]},
            ]
            for data in batch
        ]

        if isinstance(self.model, str):
            batch_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": all_messages,
                "max_workers": self.max_litellm_workers,
                "logprobs": True,
                "top_logprobs": self.top_logprobs,
                **self.litellm_batch_completion_kwargs,
            }
            if self.response_format is not None:
                batch_kwargs["response_format"] = self.response_format
            responses = list(self.litellm.batch_completion(**batch_kwargs))
        else:
            responses: list[Any] = []
            for msgs in all_messages:
                try:
                    responses.append(self.model(msgs))
                except Exception as exc:
                    responses.append(exc)

        outputs: list[ConfidenceRolloutOutput] = []
        scores: list[float] = []
        objective_scores_list: list[dict[str, float]] = []
        trajectories: list[ConfidenceTrajectory] | None = [] if capture_traces else None

        for data, response in zip(batch, responses, strict=True):
            output, score, obj_scores, trajectory = self._process_response(
                response,
                data,
                capture_traces,
            )
            outputs.append(output)
            scores.append(score)
            objective_scores_list.append(obj_scores)
            if trajectories is not None and trajectory is not None:
                trajectories.append(trajectory)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores_list,
        )

    # ------------------------------------------------------------------
    # make_reflective_dataset
    # ------------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[ConfidenceTrajectory, ConfidenceRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build reflective dataset with confidence-enriched feedback.

        The feedback tells the reflection LLM *why* the task model was
        uncertain, not just whether it was correct.  This enables GEPA to
        evolve prompts that resolve specific ambiguities between categories.

        Each record in the dataset contains:

        * **Inputs**: the original user input text.
        * **Generated Outputs**: the model's answer annotated with
          probability.
        * **Feedback**: a diagnosis including the probability, the top
          competing alternatives, and guidance for what the prompt should
          improve.
        """
        assert len(components_to_update) == 1
        comp = components_to_update[0]

        trajectories = eval_batch.trajectories
        assert trajectories is not None, "Trajectories are required to build a reflective dataset."

        items: list[ConfidenceReflectiveRecord] = []
        for traj in trajectories:
            generated = traj["parsed_value"] or traj["full_assistant_response"]
            if traj["logprob_score"] is not None:
                probability = math.exp(traj["logprob_score"])
                generated += f" ({probability:.0%} probability)"

            items.append(
                {
                    "Inputs": traj["data"]["input"],
                    "Generated Outputs": generated,
                    "Feedback": traj["feedback"],
                }
            )

        if not items:
            raise Exception("No valid predictions found for any module.")

        return {comp: items}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_text(self, response: Any) -> str:
        """Extract the text content from a response object."""
        if hasattr(response, "choices"):
            return response.choices[0].message.content.strip()
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
        raise TypeError(f"Cannot extract text from response of type {type(response).__name__}")

    @staticmethod
    def _check_correctness(parsed_value: str | None, expected: str) -> bool:
        """Check if the parsed answer matches the expected answer."""
        if parsed_value is None:
            return False
        return parsed_value.strip().lower() == expected.strip().lower()
