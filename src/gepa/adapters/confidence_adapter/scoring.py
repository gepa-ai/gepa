# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Scoring strategies that blend correctness with logprob confidence.

Why logprobs?
~~~~~~~~~~~~~
When an LLM generates a structured JSON output constrained by an ``enum``,
the model must choose one of the allowed values.  The token-level
log-probabilities for those value tokens reveal **how certain** the model
actually is about that choice -- even when the output always *looks*
deterministic (temperature = 0, forced choice).

The sum of the per-token logprobs for a value (``joint_logprob``) is the
natural measure of confidence for the chosen classification:

* **joint_logprob = 0** means the model considered every token virtually
  certain (probability ≈ 1).  In practice values cluster between roughly
  **-0.1** (very confident) and **-5.0** (very uncertain); values below
  **-10** indicate near-random guessing.
* Because logprobs are *negative*, **closer to 0 = more confident**.

Using logprobs in GEPA
~~~~~~~~~~~~~~~~~~~~~~~
The :class:`ConfidenceAdapter` extracts ``joint_logprob`` from the
``llm-structured-confidence`` library and passes it to a pluggable
:class:`ScoringStrategy`.  Each strategy maps
``(is_correct, logprob_score)`` to a single float in ``[0, 1]`` consumed
by the GEPA engine (higher is better).

Three built-in strategies are provided:

* :class:`LinearBlendScoring` -- proportional penalty for low-confidence
  correct answers.  **Recommended default** for most classification tasks.
* :class:`ThresholdScoring` -- binary gate: ``1.0`` only if correct *and*
  logprob exceeds a cutoff.
* :class:`SigmoidScoring` -- smooth S-curve that maps logprobs to
  ``[0, 1]`` when correct.

All strategies degrade gracefully when logprobs are unavailable
(``logprob_score=None``): correct answers receive ``1.0``, incorrect
answers receive ``0.0`` -- identical to plain accuracy scoring.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class ScoringStrategy(Protocol):
    """Protocol for combining correctness and logprob confidence into a score.

    Every implementation must return a value in ``[0, 1]`` following GEPA's
    convention: **higher is better**.

    The ``logprob_score`` parameter is the **joint logprob** (sum of
    per-token logprobs) for the target field, as returned by
    ``llm-structured-confidence``.  It is always **<= 0** (closer to 0 =
    more confident).
    """

    def score(self, is_correct: bool, logprob_score: float | None) -> float:
        """Return a blended score in ``[0, 1]``.

        Parameters
        ----------
        is_correct:
            Whether the model's answer matches the expected answer.
        logprob_score:
            Joint logprob (sum of per-token logprobs) for the target field.
            Always ``<= 0``.  ``None`` when logprobs are unavailable.
        """
        ...

    def describe(self) -> str:
        """Human-readable description for logs and reflective feedback.

        .. note:: Not currently called by the adapter, but available for
           custom logging or user-facing diagnostics.
        """
        ...


class LinearBlendScoring:
    """Proportional scoring that penalises low-confidence correct answers.

    The joint logprob is converted to a probability via ``exp(logprob)``
    and then compared against a threshold:

    * **Correct + confident** (probability >= *low_confidence_threshold*)
      -> ``1.0``
    * **Correct + uncertain** (probability < threshold) -> linearly
      interpolated between *min_score_on_correct* and ``1.0``
    * **Incorrect** -> ``0.0``

    When *logprob_score* is ``None`` (logprobs unavailable), correct answers
    receive ``1.0`` so behaviour degrades gracefully to binary scoring.

    Parameters
    ----------
    low_confidence_threshold:
        Probability threshold (in ``(0, 1]``) above which a correct answer
        receives a full score of ``1.0``.  Default ``0.5``.
    min_score_on_correct:
        Minimum score assigned to a correct answer whose probability falls
        below the threshold.  Default ``0.3``.
    """

    def __init__(
        self,
        low_confidence_threshold: float = 0.5,
        min_score_on_correct: float = 0.3,
    ) -> None:
        if not 0.0 < low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be in (0, 1]")
        if not 0.0 <= min_score_on_correct < 1.0:
            raise ValueError("min_score_on_correct must be in [0, 1)")
        self.low_confidence_threshold = low_confidence_threshold
        self.min_score_on_correct = min_score_on_correct

    def score(self, is_correct: bool, logprob_score: float | None) -> float:
        if not is_correct:
            return 0.0
        if logprob_score is None:
            return 1.0
        probability = math.exp(logprob_score)
        if probability >= self.low_confidence_threshold:
            return 1.0
        t = probability / self.low_confidence_threshold
        return self.min_score_on_correct + (1.0 - self.min_score_on_correct) * t

    def describe(self) -> str:
        return f"LinearBlendScoring(threshold={self.low_confidence_threshold}, min_score={self.min_score_on_correct})"


class ThresholdScoring:
    """Binary scoring gated on a logprob threshold.

    * ``1.0`` only when correct **and** ``exp(logprob)`` >= *threshold*
    * ``0.0`` otherwise

    When *logprob_score* is ``None``, correct answers receive ``1.0``.

    Parameters
    ----------
    threshold:
        Probability threshold in ``(0, 1]``.  Default ``0.7``.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    def score(self, is_correct: bool, logprob_score: float | None) -> float:
        if not is_correct:
            return 0.0
        if logprob_score is None:
            return 1.0
        probability = math.exp(logprob_score)
        return 1.0 if probability >= self.threshold else 0.0

    def describe(self) -> str:
        return f"ThresholdScoring(threshold={self.threshold})"


class SigmoidScoring:
    """Smooth sigmoid curve mapping logprob confidence to ``[0, 1]``.

    The logprob is converted to probability via ``exp(logprob)`` and then
    passed through a sigmoid:

        ``score = sigmoid(steepness * (probability - midpoint))``

    when correct, ``0.0`` when incorrect.

    When *logprob_score* is ``None``, correct answers receive ``1.0``.

    Parameters
    ----------
    midpoint:
        Probability value where the sigmoid outputs ``0.5``.  Default ``0.5``.
    steepness:
        Controls how sharply the curve transitions.  Default ``10.0``.
    """

    def __init__(self, midpoint: float = 0.5, steepness: float = 10.0) -> None:
        if not 0.0 < midpoint < 1.0:
            raise ValueError("midpoint must be in (0, 1)")
        if steepness <= 0:
            raise ValueError("steepness must be positive")
        self.midpoint = midpoint
        self.steepness = steepness

    def score(self, is_correct: bool, logprob_score: float | None) -> float:
        if not is_correct:
            return 0.0
        if logprob_score is None:
            return 1.0
        probability = math.exp(logprob_score)
        return 1.0 / (1.0 + math.exp(-self.steepness * (probability - self.midpoint)))

    def describe(self) -> str:
        return f"SigmoidScoring(midpoint={self.midpoint}, steepness={self.steepness})"
