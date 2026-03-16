# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Confidence-aware adapter for structured-output classification tasks.

Motivation
~~~~~~~~~~
Standard prompt-optimization evaluates a candidate prompt by checking whether
the model's answer matches the expected answer: correct → 1.0, wrong → 0.0.
This binary signal **cannot distinguish** between a model that genuinely
understands a category and one that merely *guesses correctly*.

Consider a prompt for transaction categorization: the model answers
``"Bills/Electricity"`` and that happens to be correct.  But the token-level
log-probabilities reveal that the runner-up was ``"Bills/Gas & Oil"`` at
almost the same probability.  The model got lucky.  Under a purely binary
metric, GEPA would keep this prompt -- it "works".  The next random seed or
slight input variation will cause a misclassification.

This adapter solves the problem by:

1. **Extracting the joint logprob** (sum of per-token logprobs) for the
   classification field via ``llm-structured-confidence``.
2. **Blending** correctness with confidence via a pluggable
   :class:`ScoringStrategy` so that lucky guesses are penalised.
3. **Feeding confidence details into the reflective feedback** so the
   reflection LLM knows *why* the model was uncertain and can propose
   prompts that resolve specific ambiguities.

Prerequisites
~~~~~~~~~~~~~
* **Structured output with enum constraints**: the adapter works best when
  the LLM's ``response_format`` forces a JSON object with one or more
  ``enum``-constrained string fields (e.g. ``{"category": "..."}``).  This
  lets the logprobs reflect the model's distribution over the allowed
  categories, rather than arbitrary free-text.
* **Logprobs support**: the model must expose token-level logprobs.
  OpenAI (``gpt-4.1``, ``gpt-4.1-mini``, ``o3-mini``, etc.) and
  Google Gemini (``gemini-2.5-flash``, etc.) both support this via
  ``logprobs=True``.

Typical usage::

    from gepa.adapters.confidence_adapter import ConfidenceAdapter, LinearBlendScoring
    import gepa

    adapter = ConfidenceAdapter(
        model="openai/gpt-4.1-mini",
        field_path="category_name",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "category_name": {
                            "type": "string",
                            "enum": ["Bills/Electricity", "Food & Drinks/Restaurants", ...],
                        }
                    },
                    "required": ["category_name"],
                    "additionalProperties": False,
                },
            },
        },
        scoring_strategy=LinearBlendScoring(low_confidence_threshold=0.5),
    )

    result = gepa.optimize(
        seed_candidate={"system_prompt": seed_prompt},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm="openai/gpt-5",
        max_metric_calls=15000,
    )

What the reflection LLM sees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With the default :class:`~gepa.adapters.default_adapter.DefaultAdapter`:

    *"The generated response is incorrect.  The correct answer is
    'Bills/Electricity'.  Ensure that the correct answer is included in the
    response exactly as it is."*

With :class:`ConfidenceAdapter`:

    *"Correct but unreliable (-2.31 logprob, 10% probability).  Model
    answered 'Bills/Electricity' but was nearly split with alternatives.
    Top alternatives: 'Bills/Gas & Oil' (9%).  The model cannot reliably
    distinguish between these options with the current prompt."*

Or:

    *"Incorrect with high confidence (-0.05 logprob, 95% probability).
    Model confidently answered 'Shopping/Electronics' but the correct
    answer is 'Shopping/Video Games'.  The prompt is misleading the model
    for this type of input."*
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict

from gepa.adapters.confidence_adapter.scoring import LinearBlendScoring, ScoringStrategy
from gepa.core.adapter import EvaluationBatch

logger = logging.getLogger(__name__)

TOP_ALTERNATIVES_IN_FEEDBACK = 3
_HIGH_CONFIDENCE_LOGPROB = -0.36  # ~ exp(-0.36) ≈ 0.70
_LOW_CONFIDENCE_LOGPROB = -0.69   # ~ exp(-0.69) ≈ 0.50


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ConfidenceDataInst(TypedDict):
    """Input data instance for confidence-aware evaluation.

    Attributes
    ----------
    input:
        The user-facing text to classify (e.g. a transaction description).
    answer:
        The expected classification label (must match an enum value).
    additional_context:
        Optional key-value pairs surfaced in reflective feedback.
    """

    input: str
    answer: str
    additional_context: dict[str, str]


class ConfidenceTrajectory(TypedDict):
    """Per-example execution trace used to build reflective feedback.

    Attributes
    ----------
    data:
        The original input data instance.
    response_text:
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
    response_text: str
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
    response_text:
        Raw text returned by the LLM.
    parsed_value:
        Value extracted from the JSON response at ``answer_field``.
    logprob_score:
        Joint logprob for the target field.
    """

    response_text: str
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
) -> str:
    """Build rich feedback with logprob confidence for the reflection LLM.

    The feedback is structured to help the reflection LLM understand:
    - Whether the answer was correct
    - How confident the model was (logprob + probability)
    - What the close alternatives were (if any)
    - What the prompt should improve to resolve ambiguity
    """
    if logprob_score is not None:
        probability = math.exp(logprob_score)
        conf_str = f"{logprob_score:.2f} logprob, {probability:.0%} probability"
    else:
        conf_str = "unknown"
    got_str = got or "<parse error>"

    if is_correct:
        if logprob_score is not None and logprob_score >= _HIGH_CONFIDENCE_LOGPROB:
            feedback = (
                f"Correct with high confidence ({conf_str}). "
                f"Model is certain about '{expected}'."
            )
        elif logprob_score is not None and logprob_score < _LOW_CONFIDENCE_LOGPROB:
            alt_str = _format_alternatives(top_alternatives, exclude=expected)
            feedback = (
                f"Correct but unreliable ({conf_str}). "
                f"Model answered '{expected}' but was nearly split with alternatives."
            )
            if alt_str:
                feedback += f" Top alternatives: {alt_str}."
            feedback += " The model cannot reliably distinguish between these options with the current prompt."
        else:
            feedback = (
                f"Correct with moderate confidence ({conf_str}). "
                f"Model answered '{expected}'."
            )
    else:
        alt_str = _format_alternatives(top_alternatives, exclude=got_str)
        if logprob_score is not None and logprob_score >= _HIGH_CONFIDENCE_LOGPROB:
            feedback = (
                f"Incorrect with high confidence ({conf_str}). "
                f"Model confidently answered '{got_str}' but the correct answer is '{expected}'. "
                "The prompt is misleading the model for this type of input."
            )
        else:
            feedback = (
                f"Incorrect ({conf_str}). "
                f"Expected '{expected}' but got '{got_str}'."
            )
            if alt_str:
                feedback += f" Top alternatives: {alt_str}."
            feedback += " The prompt needs clearer signals to distinguish between these options."

        ctx = "\n".join(f"{k}: {v}" for k, v in additional_context.items())
        if ctx:
            feedback += f"\nAdditional context:\n{ctx}"

    return feedback


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

class ConfidenceAdapter:
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
    top_logprobs:
        Number of top logprobs to request from the LLM (1-20).
    failure_score:
        Score assigned when an example fails (parse error, API error, etc.).
    max_litellm_workers:
        Concurrency for litellm calls (only used when *model* is a string).
    litellm_completion_kwargs:
        Extra keyword arguments forwarded to every ``litellm.completion``
        call (e.g. ``temperature``, ``max_tokens``).
    """

    def __init__(
        self,
        model: str | ChatCompletionCallable,
        field_path: str,
        response_format: dict[str, Any] | None = None,
        response_schema: type | dict[str, Any] | None = None,
        scoring_strategy: ScoringStrategy | None = None,
        answer_field: str | None = None,
        top_logprobs: int = 5,
        failure_score: float = 0.0,
        max_litellm_workers: int = 10,
        litellm_completion_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(model, str):
            import litellm

            self.litellm = litellm
        self.model = model
        self.field_path = field_path
        self.response_format = response_format
        self.response_schema = response_schema
        self.scoring_strategy: ScoringStrategy = scoring_strategy or LinearBlendScoring()
        self.answer_field = answer_field or field_path
        self.top_logprobs = top_logprobs
        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self.litellm_completion_kwargs = litellm_completion_kwargs or {}

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list[ConfidenceDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ConfidenceTrajectory, ConfidenceRolloutOutput]:
        """Run *candidate* on *batch*, extracting logprob confidence.

        Each example is evaluated individually because we need the full
        response object to extract token-level logprobs.

        The joint logprob (sum of per-token logprobs) for the target field
        is used as the confidence metric.  This value is always ``<= 0``;
        closer to 0 means the model is more certain about its answer.
        """
        from llm_structured_confidence import extract_logprobs

        outputs: list[ConfidenceRolloutOutput] = []
        scores: list[float] = []
        objective_scores_list: list[dict[str, float]] = []
        trajectories: list[ConfidenceTrajectory] | None = [] if capture_traces else None

        system_content = next(iter(candidate.values()))

        for data in batch:
            messages: list[ChatMessage] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data["input"]},
            ]

            try:
                response = self._call_llm(messages)
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
            )

            output: ConfidenceRolloutOutput = {
                "response_text": response_text,
                "parsed_value": parsed_value,
                "logprob_score": logprob_score,
            }
            outputs.append(output)
            scores.append(score)

            probability = math.exp(logprob_score) if logprob_score is not None else 0.0
            objective_scores_list.append({
                "accuracy": 1.0 if is_correct else 0.0,
                "logprob": logprob_score if logprob_score is not None else 0.0,
                "probability": probability,
            })

            if trajectories is not None:
                trajectories.append({
                    "data": data,
                    "response_text": response_text,
                    "parsed_value": parsed_value,
                    "logprob_score": logprob_score,
                    "top_alternatives": top_alternatives,
                    "is_correct": is_correct,
                    "score": score,
                    "feedback": feedback,
                })

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
        """Build reflective dataset with logprob-enriched feedback.

        The feedback tells the reflection LLM *why* the task model was
        uncertain, not just whether it was correct.  This enables GEPA to
        evolve prompts that resolve specific ambiguities between categories.

        Each record in the dataset contains:

        * **Inputs**: the original user input text.
        * **Generated Outputs**: the model's answer annotated with
          logprob and probability.
        * **Feedback**: a detailed diagnosis including the joint logprob,
          the derived probability, the top competing alternatives, and
          guidance for what the prompt should improve.
        """
        assert len(components_to_update) == 1
        comp = components_to_update[0]

        trajectories = eval_batch.trajectories
        assert trajectories is not None, "Trajectories are required to build a reflective dataset."

        items: list[ConfidenceReflectiveRecord] = []
        for traj in trajectories:
            generated = traj["parsed_value"] or traj["response_text"]
            if traj["logprob_score"] is not None:
                probability = math.exp(traj["logprob_score"])
                generated += f" (logprob: {traj['logprob_score']:.2f}, probability: {probability:.0%})"

            items.append({
                "Inputs": traj["data"]["input"],
                "Generated Outputs": generated,
                "Feedback": traj["feedback"],
            })

        if not items:
            raise Exception("No valid predictions found for any module.")

        return {comp: items}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[ChatMessage]) -> Any:
        """Call the LLM and return the full response object."""
        if isinstance(self.model, str):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "logprobs": True,
                "top_logprobs": self.top_logprobs,
                **self.litellm_completion_kwargs,
            }
            if self.response_format is not None:
                kwargs["response_format"] = self.response_format
            return self.litellm.completion(**kwargs)
        return self.model(messages)

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
