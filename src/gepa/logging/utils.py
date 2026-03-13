# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.strategies.eval_policy import EvaluationPolicy


def log_detailed_metrics_after_discovering_new_program(
    logger,
    gepa_state: GEPAState,
    new_program_idx,
    valset_evaluation: ValsetEvaluation,
    objective_scores,
    experiment_tracker,
    linear_pareto_front_program_idx,
    valset_size: int,
    val_evaluation_policy: EvaluationPolicy[DataId, DataInst],
):
    # best_prog_per_agg_val_score = idxmax(gepa_state.program_full_scores_val_set)
    best_prog_per_agg_val_score = val_evaluation_policy.get_best_program(gepa_state)
    best_score_on_valset = val_evaluation_policy.get_valset_score(best_prog_per_agg_val_score, gepa_state)

    # avg, coverage = gepa_state.get_program_average_val_subset(new_program_idx)
    valset_score = val_evaluation_policy.get_valset_score(new_program_idx, gepa_state)
    valset_scores = valset_evaluation.scores_by_val_id
    coverage = len(valset_scores)
    logger.log(
        f"Iteration {gepa_state.i + 1}: Valset score for new program: {valset_score}"
        f" (coverage {coverage} / {valset_size})"
    )

    agg_valset_score_new_program = val_evaluation_policy.get_valset_score(new_program_idx, gepa_state)

    logger.log(f"Iteration {gepa_state.i + 1}: Val aggregate for new program: {agg_valset_score_new_program}")
    logger.log(f"Iteration {gepa_state.i + 1}: Individual valset scores for new program: {valset_scores}")
    if objective_scores:
        logger.log(f"Iteration {gepa_state.i + 1}: Objective aggregate scores for new program: {objective_scores}")
    logger.log(f"Iteration {gepa_state.i + 1}: New valset pareto front scores: {gepa_state.pareto_front_valset}")
    if gepa_state.objective_pareto_front:
        logger.log(f"Iteration {gepa_state.i + 1}: Objective pareto front scores: {gepa_state.objective_pareto_front}")

    pareto_scores = list(gepa_state.pareto_front_valset.values())
    assert all(score > float("-inf") for score in pareto_scores), (
        "Should have at least one valid score per validation example"
    )
    assert len(pareto_scores) > 0
    pareto_avg = sum(pareto_scores) / len(pareto_scores)

    logger.log(f"Iteration {gepa_state.i + 1}: Valset pareto front aggregate score: {pareto_avg}")
    logger.log(
        f"Iteration {gepa_state.i + 1}: Updated valset pareto front programs: {gepa_state.program_at_pareto_front_valset}"
    )
    if gepa_state.program_at_pareto_front_objectives:
        logger.log(
            f"Iteration {gepa_state.i + 1}: Updated objective pareto front programs: {gepa_state.program_at_pareto_front_objectives}"
        )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best valset aggregate score so far: {max(gepa_state.program_full_scores_val_set)}"
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best program as per aggregate score on valset: {best_prog_per_agg_val_score}"
    )
    logger.log(f"Iteration {gepa_state.i + 1}: Best score on valset: {best_score_on_valset}")
    logger.log(f"Iteration {gepa_state.i + 1}: Linear pareto front program index: {linear_pareto_front_program_idx}")
    logger.log(f"Iteration {gepa_state.i + 1}: New program candidate index: {new_program_idx}")

    # Scalar metrics go to log_metrics (creates wandb/mlflow line charts)
    metrics = {
        "iteration": gepa_state.i + 1,
        "new_program_idx": new_program_idx,
        "valset_pareto_front_agg": pareto_avg,
        "best_score_on_valset": best_score_on_valset,
        "linear_pareto_front_program_idx": linear_pareto_front_program_idx,
        "best_program_as_per_agg_score_valset": best_prog_per_agg_val_score,
        "val_evaluated_count_new_program": coverage,
        "val_total_count": valset_size,
        "val_program_average": valset_score,
        "total_metric_calls": gepa_state.total_num_evals,
    }
    if objective_scores:
        for obj_name, obj_val in objective_scores.items():
            if isinstance(obj_val, int | float):
                metrics[f"objective/{obj_name}"] = obj_val
    experiment_tracker.log_metrics(metrics, step=gepa_state.i + 1)

    # Structured data goes to log_table (creates wandb Tables / mlflow artifacts)
    # instead of log_metrics, which would flatten nested dicts into hundreds of charts.
    # Only log the new candidate's row to avoid O(candidates * valset) repeated uploads.

    all_val_ids = sorted(gepa_state.pareto_front_valset.keys(), key=str)
    val_columns = ["candidate_idx", "parent_ids"] + [str(vid) for vid in all_val_ids]
    new_scores_dict = gepa_state.prog_candidate_val_subscores[new_program_idx]
    new_parent = gepa_state.parent_program_for_candidate[new_program_idx]
    val_row = [new_program_idx, str(new_parent)] + [new_scores_dict.get(vid) for vid in all_val_ids]
    experiment_tracker.log_table("valset_scores", columns=val_columns, data=[val_row])

    # Valset pareto front: which programs are best for which val examples
    pareto_front_rows = [
        [str(val_id), score, str(sorted(gepa_state.program_at_pareto_front_valset[val_id]))]
        for val_id, score in gepa_state.pareto_front_valset.items()
    ]
    if pareto_front_rows:
        experiment_tracker.log_table(
            "valset_pareto_front",
            columns=["val_id", "best_score", "program_ids"],
            data=pareto_front_rows,
        )

    # Objective scores for the new candidate only
    new_obj_scores = gepa_state.prog_candidate_objective_scores[new_program_idx]
    if new_obj_scores:
        all_objectives = sorted(new_obj_scores.keys(), key=str)
        obj_columns = ["candidate_idx", "parent_ids"] + [str(obj) for obj in all_objectives]
        obj_row = [new_program_idx, str(new_parent)] + [new_obj_scores.get(obj) for obj in all_objectives]
        experiment_tracker.log_table("objective_scores", columns=obj_columns, data=[obj_row])

    # Objective pareto front
    if gepa_state.objective_pareto_front:
        obj_pareto_rows = [
            [str(obj), float(score), str(sorted(gepa_state.program_at_pareto_front_objectives[obj]))]
            for obj, score in gepa_state.objective_pareto_front.items()
        ]
        experiment_tracker.log_table(
            "objective_pareto_front",
            columns=["objective", "best_score", "program_ids"],
            data=obj_pareto_rows,
        )
