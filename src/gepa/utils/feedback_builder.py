from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import EvaluationBatch


@dataclass
class FeedbackBuilder:
    """Builds reflective feedback datasets from ``EvaluationBatch``.

    Replaces the boilerplate formerly inlined in
    ``OptimizeAnythingAdapter.make_reflective_dataset()`` and adds optional
    enrichments: score-gap annotation, rationale extraction, and global
    constraint injection.
    """

    include_score_diff: bool = False
    rationale_field: str | None = None
    global_constraints: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build per-component reflective datasets from *eval_batch*.

        Returns a mapping ``{component_name: [record, ...]}``.
        """
        scores, side_infos = eval_batch.scores, eval_batch.trajectories
        assert side_infos is not None

        ret: dict[str, list[dict[str, Any]]] = {}
        for component_name in components_to_update:
            ret[component_name] = []
            for score, side_info in zip(scores, side_infos, strict=False):
                record: dict[str, Any] = {}
                for k, v in side_info.items():
                    if k == "scores":
                        record["Scores (Higher is Better)"] = v
                    elif k == self.rationale_field:
                        continue
                    elif not k.endswith("_specific_info"):
                        record[k] = v
                    elif k == f"{component_name}_specific_info":
                        record.update(v)
                    else:
                        continue
                self._enrich_record(record, score=score, side_info=side_info)
                ret[component_name].append(record)
        return ret

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enrich_record(
        self,
        record: dict[str, Any],
        *,
        score: float,
        side_info: dict[str, Any],
    ) -> None:
        """Apply optional enrichments to *record* in-place."""

        # Score diff
        if self.include_score_diff:
            gap = 1.0 - score
            line = f"Score: {score}. Gap from perfect: {gap:.2f}."
            existing = record.get("Feedback")
            if existing is not None:
                record["Feedback"] = f"{existing}\n{line}"
            else:
                record["Feedback"] = line

        # Rationale extraction
        if self.rationale_field is not None:
            value = side_info.get(self.rationale_field)
            if value is not None:
                record["Rationale"] = value

        # Global constraints
        if self.global_constraints:
            record["Constraints"] = "\n".join(f"- {c}" for c in self.global_constraints)
