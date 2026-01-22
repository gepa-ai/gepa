from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from gepa.core.adapter import DataInst, EvaluationBatch, GEPAAdapter
from gepa.proposer.reflective_mutation.base import LanguageModel

if TYPE_CHECKING:
    from gepa.optimize_anything import FitnessFn, SideInfo


class OptimizeAnythingAdapter(GEPAAdapter):
    def __init__(
        self,
        fitness_fn: "FitnessFn",
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | None = None,
        parallel: bool = True,
        max_workers: int | None = None,
    ):
        self.fitness_fn = fitness_fn
        self.parallel = parallel
        self.max_workers = max_workers

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        # Call fitness_fn once per example
        if self.parallel and len(batch) > 1:
            eval_output = self._evaluate_parallel(batch, candidate)
        else:
            eval_output = [self.fitness_fn(candidate, example) for example in batch]
        scores = [score for score, _, _ in eval_output]
        side_infos: list[SideInfo] = [info for _, _, info in eval_output]
        outputs = [output for _, output, _ in eval_output]

        objective_scores = []
        for side_info in side_infos:
            objective_score = {}
            if "scores" in side_info:
                objective_score.update(side_info["scores"])

            for param_name in candidate.keys():
                if param_name + "_specific_info" in side_info and "scores" in side_info[param_name + "_specific_info"]:
                    objective_score.update(
                        {f"{param_name}::{k}": v for k, v in side_info[param_name + "_specific_info"]["scores"].items()}
                    )

            objective_scores.append(objective_score)

        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=side_infos, objective_scores=objective_scores
        )

    def _evaluate_parallel(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
    ) -> list[tuple[float, Any, "SideInfo"]]:
        """Evaluate batch in parallel using ThreadPoolExecutor."""
        results: list[tuple[int, tuple[float, Any, SideInfo]]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers or len(batch)) as executor:
            # Submit all tasks with their index to maintain order
            future_to_idx = {
                executor.submit(self.fitness_fn, candidate, example): idx
                for idx, example in enumerate(batch)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results.append((idx, future.result()))

        # Sort by index to maintain original order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        scores, side_infos = eval_batch.scores, eval_batch.trajectories
        assert side_infos is not None
        ret: dict[str, list[dict[str, Any]]] = {}
        for component_name in components_to_update:
            ret[component_name] = []
            for score, side_info in zip(scores, side_infos, strict=False):
                ret[component_name].append({})
                for k, v in side_info.items():
                    if k == "scores":
                        ret[component_name][-1]["Scores (Higher is Better)"] = v
                    elif not k.endswith("_specific_info"):
                        ret[component_name][-1][k] = v
                    elif k == f"{component_name}_specific_info":
                        ret[component_name][-1].update(v)
                    else:
                        continue
        return ret
