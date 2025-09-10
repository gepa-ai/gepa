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
):
    best_prog_as_per_agg_score = idxmax(gepa_state.per_program_tracked_scores)
    best_prog_as_per_agg_score_valset = idxmax(gepa_state.program_full_scores_val_set)

    logger.log(f"Iteration {gepa_state.i + 1}: Full valset score for new program: {valset_score}")
    logger.log(
        f"Iteration {gepa_state.i + 1}: Full train_val score for new program: {gepa_state.per_program_tracked_scores[new_program_idx]}"
    )
    logger.log(f"Iteration {gepa_state.i + 1}: Individual valset scores for new program: {valset_subscores}")
    logger.log(f"Iteration {gepa_state.i + 1}: New valset pareto front scores: {gepa_state.pareto_front_valset}")
    logger.log(
        f"Iteration {gepa_state.i + 1}: Full valset pareto front score: {sum(gepa_state.pareto_front_valset) / len(gepa_state.pareto_front_valset)}"
    )
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
        # "full_valset_score": valset_score,
        # "full_train_val_score": gepa_state.per_program_tracked_scores[new_program_idx],
        "new_program_idx": new_program_idx,
        "valset_pareto_front_scores": gepa_state.pareto_front_valset,
        "individual_valset_score_new_program": valset_subscores,
        "valset_pareto_front_agg": sum(gepa_state.pareto_front_valset) / len(gepa_state.pareto_front_valset),
        "valset_pareto_front_programs": gepa_state.program_at_pareto_front_valset,
        "best_valset_agg_score": max(gepa_state.program_full_scores_val_set),
        "linear_pareto_front_program_idx": linear_pareto_front_program_idx,
        "best_program_as_per_agg_score": best_prog_as_per_agg_score,
        "best_program_as_per_agg_score_valset": best_prog_as_per_agg_score_valset,
        "best_score_on_valset": gepa_state.program_full_scores_val_set[best_prog_as_per_agg_score_valset],
        "best_score_on_train_val": gepa_state.per_program_tracked_scores[best_prog_as_per_agg_score],
    }

    experiment_tracker.log_metrics(metrics, step=gepa_state.i + 1)
