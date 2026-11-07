"""Validate persisted inference against observations and execution reservations."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

from gepa.evidence.budget import MetricBudgetGuard, Reservation
from gepa.evidence.config import SequentialEvidenceConfig, positive_integer
from gepa.evidence.observations import ObservationId, Side, candidate_identity, example_identity

if TYPE_CHECKING:
    from gepa.evidence.state import ComparisonState


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-14)


def audit_comparison(record: ComparisonState, config: SequentialEvidenceConfig) -> None:
    """Reconstruct all learned moments and intervals instead of trusting summaries."""
    from gepa.evidence.state import ComparisonState

    positive_integer(record.index, "saved comparison allocation index")
    for name in ("completed_rounds", "metric_calls"):
        value = getattr(record, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid saved {name}")
    if record.example_keys != [example_identity(eid) for eid in record.example_ids]:
        raise ValueError("saved example identities differ from frozen stratum IDs")
    if record.support != (config.assumption if config.confidence_supported else "none"):
        raise ValueError("saved statistical support differs from declared contract")
    if not record.parent_ids or len(record.parent_ids) != 1:
        raise ValueError("saved evidence must have exactly one reflective parent")
    if isinstance(record.parent_ids[0], bool) or not isinstance(record.parent_ids[0], int) or record.parent_ids[0] < 0:
        raise ValueError("invalid saved parent pool identity")
    parent_key, child_key = candidate_identity(record.parent), candidate_identity(record.child)
    rounds: dict[int, dict[str, list[Any]]] = defaultdict(lambda: {"parent": [], "child": []})
    for observation in record.observations.values():
        identity = observation.identity
        expected_candidate = parent_key if identity.side == "parent" else child_key
        if identity.candidate != expected_candidate or identity.split != "train":
            raise ValueError("saved observation candidate or split identity mismatch")
        prefix = f"{record.comparison_id}:"
        if not identity.trial.startswith(prefix):
            raise ValueError("saved observation belongs to another comparison")
        suffix = identity.trial[len(prefix) :]
        try:
            round_index = int(suffix)
        except ValueError as error:
            raise ValueError("invalid saved observation trial index") from error
        if round_index <= 0 or str(round_index) != suffix:
            raise ValueError("noncanonical saved observation trial index")
        if round_index > record.completed_rounds + 1:
            raise ValueError("saved observation skips a balanced round")
        rounds[round_index][identity.side].append(observation)
    if set(range(1, record.completed_rounds + 1)) - rounds.keys():
        raise ValueError("saved completed round has no observations")
    rebuilt = ComparisonState(
        record.comparison_id,
        record.index,
        record.parent,
        record.child,
        record.parent_ids,
        record.example_ids,
        record.example_keys,
        record.weights,
        record.support,
    )
    rebuilt.initialize_intervals(config)
    for round_index in range(1, record.completed_rounds + 1):
        if rebuilt.terminal:
            raise ValueError("saved inference continued after a terminal statistical decision")
        side_ids = []
        for side in ("parent", "child"):
            observations = rounds[round_index][side]
            if len(observations) != len(record.example_keys):
                raise ValueError("saved completed round is unbalanced")
            by_example = {o.identity.example: o for o in observations}
            if set(by_example) != set(record.example_keys) or any(o.status != "completed" for o in observations):
                raise ValueError("saved completed round has missing or failed observations")
            identities = []
            for example_key in record.example_keys:
                observation = by_example[example_key]
                rebuilt.observations[observation.identity.key] = observation
                identities.append(observation.identity)
            side_ids.append(identities)
        rebuilt.finish_round(side_ids[0], side_ids[1], config)
    for original, computed in zip(
        (*record.moments_parent, *record.moments_child), (*rebuilt.moments_parent, *rebuilt.moments_child), strict=True
    ):
        if (
            original.count != computed.count
            or not _same_number(original.mean, computed.mean)
            or not _same_number(original.m2, computed.m2)
        ):
            raise ValueError("saved moments disagree with replayed completed observations")
    for original, computed in zip(
        (*record.intervals_parent, *record.intervals_child, (record.lower, record.upper)),
        (*rebuilt.intervals_parent, *rebuilt.intervals_child, (rebuilt.lower, rebuilt.upper)),
        strict=True,
    ):
        if not all(_same_number(a, b) for a, b in zip(original, computed, strict=True)):
            raise ValueError("saved interval disagrees with replayed confidence allocation")
    if record.status == "supported_improvement" and rebuilt.status != record.status:
        raise ValueError("saved improvement has insufficient statistical support")
    if record.status == "supported_regression" and rebuilt.status != record.status:
        raise ValueError("saved regression has insufficient statistical support")
    if record.selection_label == "supported" and rebuilt.status != "supported_improvement":
        raise ValueError("selected candidate lacks confidence support")
    if record.selected and record.selection_label not in ("supported", "unresolved_exploration"):
        raise ValueError("selected candidate lacks an explicit acceptance label")
    if record.selected:
        positive_integer(record.pool_size_at_selection, "saved pool publication boundary")
    if record.status in ("promoted", "exploration") and record.validation_candidate_idx is None:
        raise ValueError("saved promoted decision lacks its validation pool index")
    if record.validation_candidate_idx is not None:
        if not record.selected or record.status not in ("promoted", "exploration"):
            raise ValueError("invalid saved candidate promotion state")
        if (
            isinstance(record.validation_candidate_idx, bool)
            or not isinstance(record.validation_candidate_idx, int)
            or record.validation_candidate_idx < 0
        ):
            raise ValueError("invalid saved validation pool index")


def audit_ledger(
    comparisons: dict[str, ComparisonState],
    guard: MetricBudgetGuard,
    saved: list[dict[str, Any]],
    config: SequentialEvidenceConfig,
) -> None:
    """Join execution costs to comparisons and account interrupted charges."""
    grouped: dict[str, list[Reservation]] = defaultdict(list)
    for reservation in guard.records.values():
        if reservation.purpose != "evidence":
            continue
        if reservation.comparison_id not in comparisons:
            raise ValueError("execution reservation belongs to an unknown comparison")
        assert reservation.comparison_id is not None
        grouped[reservation.comparison_id].append(reservation)
    original = {item["comparison_id"]: item for item in saved}
    for identity, record in comparisons.items():
        was_running = original[identity]["status"] == "running"
        seen = set()
        total = 0
        for reservation in grouped[identity]:
            key = (reservation.round_index, reservation.side)
            if key in seen:
                raise ValueError("duplicate side execution for the same evidence round")
            seen.add(key)
            if reservation.round_index is None or reservation.round_index > record.completed_rounds + 1:
                raise ValueError("execution reservation skips comparison rounds")
            expected_cost = len(record.example_keys) * config.max_calls_per_example
            if reservation.upper_cost != expected_cost:
                raise ValueError("saved execution maximum differs from the declared cost contract")
            if reservation.round_index <= record.completed_rounds and reservation.status != "completed":
                raise ValueError("learned evidence belongs to an incomplete execution")
            candidate = record.parent if reservation.side == "parent" else record.child
            for example in record.example_keys:
                observation_id = ObservationId(
                    config.evaluator_fingerprint,
                    candidate_identity(candidate),
                    "train",
                    example,
                    f"{identity}:{reservation.round_index}",
                    cast(Side, reservation.side),
                )
                if observation_id.key not in record.observations:
                    raise ValueError("execution reservation lacks scheduled observation identities")
            total += reservation.charged
        for round_index in range(1, record.completed_rounds + 1):
            if (round_index, "parent") not in seen or (round_index, "child") not in seen:
                raise ValueError("completed observations lack both execution reservations")
        if was_running:
            # Snapshot may have persisted an external completion before the
            # finally-block updated the diagnostic comparison counter.
            if record.metric_calls > total:
                raise ValueError("saved comparison cost exceeds its execution ledger")
            record.metric_calls = total
        elif record.metric_calls != total:
            raise ValueError("saved comparison cost disagrees with its execution ledger")
        if total > config.max_comparison_calls:
            raise ValueError("saved comparison exceeds its configured metric cap")
