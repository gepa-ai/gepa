# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


from gepa.core.state import GEPAState
from gepa.gepa_utils import idxmax


def log_detailed_metrics_after_discovering_new_program(
    logger,
    gepa_state: GEPAState,
    valset_score,
    new_program_idx,
    valset_subscores,
    experiment_tracker,
    linear_pareto_front_program_idx,
    valset_size: int,
):
    best_prog_as_per_agg_score = idxmax(gepa_state.per_program_tracked_scores)
    best_prog_as_per_agg_score_valset = idxmax(gepa_state.program_full_scores_val_set)

    avg, coverage = gepa_state.get_program_average(new_program_idx)
    logger.log(
        f"Iteration {gepa_state.i + 1}: Valset score for new program: {valset_score}"
        f" (coverage {coverage} / {valset_size})"
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Train/val aggregate for new program: {gepa_state.per_program_tracked_scores[new_program_idx]}"
    )
    logger.log(f"Iteration {gepa_state.i + 1}: Individual valset scores for new program: {valset_subscores}")
    logger.log(f"Iteration {gepa_state.i + 1}: New valset pareto front scores: {gepa_state.pareto_front_valset}")

    pareto_scores = [score for score in gepa_state.pareto_front_valset.values() if score != float("-inf")]
    pareto_avg = (sum(pareto_scores) / len(pareto_scores)) if pareto_scores else float("-inf")

    logger.log(f"Iteration {gepa_state.i + 1}: Valset pareto front aggregate score: {pareto_avg}")
    logger.log(
        f"Iteration {gepa_state.i + 1}: Updated valset pareto front programs: {gepa_state.program_at_pareto_front_valset}"
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best valset aggregate score so far: {max(gepa_state.program_full_scores_val_set)}"
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best program as per aggregate score on train_val: {best_prog_as_per_agg_score}"
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best program as per aggregate score on valset: {best_prog_as_per_agg_score_valset}"
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best score on valset: {gepa_state.program_full_scores_val_set[best_prog_as_per_agg_score_valset]}"
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best score on train_val: {gepa_state.per_program_tracked_scores[best_prog_as_per_agg_score]}"
    )
    logger.log(f"Iteration {gepa_state.i + 1}: Linear pareto front program index: {linear_pareto_front_program_idx}")
    logger.log(f"Iteration {gepa_state.i + 1}: New program candidate index: {new_program_idx}")

    metrics = {
        "iteration": gepa_state.i + 1,
        "new_program_idx": new_program_idx,
        "valset_pareto_front_scores": dict(gepa_state.pareto_front_valset),
        "individual_valset_score_new_program": dict(valset_subscores),
        "valset_pareto_front_agg": pareto_avg,
        "valset_pareto_front_programs": {k: list(v) for k, v in gepa_state.program_at_pareto_front_valset.items()},
        "best_valset_agg_score": max(gepa_state.program_full_scores_val_set),
        "linear_pareto_front_program_idx": linear_pareto_front_program_idx,
        "best_program_as_per_agg_score": best_prog_as_per_agg_score,
        "best_program_as_per_agg_score_valset": best_prog_as_per_agg_score_valset,
        "best_score_on_valset": gepa_state.program_full_scores_val_set[best_prog_as_per_agg_score_valset],
        "best_score_on_train_val": gepa_state.per_program_tracked_scores[best_prog_as_per_agg_score],
        "val_evaluated_count_new_program": coverage,
        "val_total_count": valset_size,
        "val_program_average": avg if avg is not None else None,
    }

    experiment_tracker.log_metrics(metrics, step=gepa_state.i + 1)
