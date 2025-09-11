from __future__ import annotations
import re
import string
from typing import List, Union

# ---- Helpers ----
def _normalize_string(s: str) -> str:
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def _remove_end_punctuation(s: str) -> str:
    while s and (s[-1] in string.punctuation or s[-1].isspace()) and s[-1] != "%":
        s = s[:-1]
    return s

# ---- Base Metric ----
class Metric:
    @property
    def name(self) -> str:
        raise NotImplementedError
    def score(self, model_answer: str, reference_answer: Union[str, list[str]]) -> float:
        raise NotImplementedError

# ---- Relaxed Correctness Core ----
class RelaxedCorrectness(Metric):
    def _relaxed_correctness(
        self, prediction: str, targets: list[str], max_relative_change: float = 0.05
    ) -> float:
        def _to_float(text: str):
            text = text.strip()
            is_percent = text.endswith("%")
            try:
                return float(text.rstrip("%")), is_percent
            except ValueError:
                return None, False

        def _is_letter(text: str) -> bool:
            return text.isalpha() and len(text) == 1

        def _preprocess_text(text: str) -> str:
            if not any(c.isdigit() for c in text):
                return _normalize_string(text)
            return _remove_end_punctuation(text).replace(",", "").replace("$", "")

        def rel(pred: float, tgt: float) -> float:
            return abs(pred - tgt) / max(abs(tgt), 1e-10)

        def compare_numeric_with_percent(
            p: float, p_pct: bool, t: float, t_pct: bool
        ) -> float:
            # direct
            ok = rel(p, t) <= max_relative_change
            if ok:
                return 1.0
            if p_pct or t_pct:
                p_dec = p / 100 if p_pct else p
                t_dec = t / 100 if t_pct else t
                if rel(p_dec, t) <= max_relative_change or rel(p, t_dec) <= max_relative_change:
                    return 1.0
            return 0.0

        prediction = _preprocess_text(prediction)
        p_val, p_pct = _to_float(prediction)

        best = 0.0
        for tgt in targets:
            tgt_prep = _preprocess_text(tgt)
            t_val, t_pct = _to_float(tgt_prep)
            if p_val is not None and t_val is not None:
                val = compare_numeric_with_percent(p_val, p_pct, t_val, t_pct)
            elif _is_letter(tgt_prep) and prediction:
                val = 1.0 if prediction[0].lower() == tgt_prep.lower() else 0.0
            else:
                val = 1.0 if prediction.lower() == tgt_prep.lower() else 0.0
            best = max(best, val)
        return best

    def score(self, model_answer: str, reference_answer: Union[str, list[str]]) -> float:
        refs = reference_answer if isinstance(reference_answer, list) else [reference_answer]
        return self._relaxed_correctness(model_answer, refs)

# ---- Explicit Prompt Variant (ONLY ONE EXPORTED) ----
class ExplicitPromptRelaxedCorrectness(RelaxedCorrectness):
    @property
    def name(self) -> str:
        return "explicit_prompt_relaxed_correctness"

    def _get_final_answer(self, generation: str) -> str:
        # Normalize "Final Answer:" variants
        generation = re.sub(r"([fF]inal\s*)?[aA]nswer\s*[:：]", "answer:", generation)
        idx = generation.lower().rfind("answer:")
        if idx == -1:
            return ""
        segment = generation[idx + len("answer:") :]
        for line in segment.splitlines():
            line = line.strip()
            if line:
                # strip simple markdown decorations
                line = re.sub(r"[*_`\[\]\(\)]", "", line)
                return line.strip()
        return ""

    def score(self, model_answer: str, reference_answer: Union[str, list[str]]) -> float:
        parsed = self._get_final_answer(model_answer)
        if not parsed:
            return 0.0
        return super().score(parsed, reference_answer)

# ---- Factory / public API ----
def get_metric() -> ExplicitPromptRelaxedCorrectness:
    return ExplicitPromptRelaxedCorrectness()

__all__ = ["ExplicitPromptRelaxedCorrectness", "get_metric"]