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
    log_individual_valset_scores_and_programs: bool = False,
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

    metrics = {
        "iteration": gepa_state.i + 1,
        "new_program_idx": new_program_idx,
        "valset_pareto_front_agg": pareto_avg,
        "valset_pareto_front_programs": {k: list(v) for k, v in gepa_state.program_at_pareto_front_valset.items()},
        "best_valset_agg_score": best_score_on_valset,
        "linear_pareto_front_program_idx": linear_pareto_front_program_idx,
        "best_program_as_per_agg_score_valset": best_prog_per_agg_val_score,
        "best_score_on_valset": best_score_on_valset,
        "val_evaluated_count_new_program": coverage,
        "val_total_count": valset_size,
        "val_program_average": valset_score,
    }
    if log_individual_valset_scores_and_programs:
        metrics.update(
            {
                "valset_pareto_front_scores": dict(gepa_state.pareto_front_valset),
                "individual_valset_score_new_program": dict(valset_scores),
            }
        )
    if objective_scores:
        metrics["objective_scores_new_program"] = dict(objective_scores)
    if valset_evaluation.objective_scores_by_val_id:
        metrics["objective_scores_by_val_new_program"] = {
            val_id: dict(scores) for val_id, scores in valset_evaluation.objective_scores_by_val_id.items()
        }
    if gepa_state.objective_pareto_front:
        metrics["objective_pareto_front_scores"] = dict(gepa_state.objective_pareto_front)
        metrics["objective_pareto_front_programs"] = {
            k: list(v) for k, v in gepa_state.program_at_pareto_front_objectives.items()
        }

    experiment_tracker.log_metrics(metrics, step=gepa_state.i + 1)
