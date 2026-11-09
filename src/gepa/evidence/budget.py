"""One scorer-attempt budget owner shared by every evidence-mode dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from gepa.evidence.config import positive_integer


class BudgetExhaustedError(Exception):
    """A complete required unit of work cannot fit without exceeding the cap."""


@dataclass
class Reservation:
    token: str
    upper_cost: int
    purpose: str
    status: Literal["reserved", "completed", "failed", "abandoned"] = "reserved"
    actual_cost: int | None = None
    failure: str | None = None
    comparison_id: str | None = None
    round_index: int | None = None
    side: str | None = None

    @property
    def charged(self) -> int:
        if self.status == "reserved":
            return 0
        return self.actual_cost if self.actual_cost is not None else self.upper_cost


class MetricBudgetGuard:
    def __init__(self, limit: int, *, initial_used: int = 0) -> None:
        self.limit = positive_integer(limit, "max_metric_calls")
        if isinstance(initial_used, bool) or not isinstance(initial_used, int) or not 0 <= initial_used <= limit:
            raise ValueError("initial metric use must fit the configured cap")
        self.initial_used = initial_used
        self.records: dict[str, Reservation] = {}
        self._used = initial_used
        self._reserved = 0
        self.sequence = 0
        self.persist: Callable[[], None] | None = None
        self.listener: Callable[[int], None] | None = None

    @property
    def used(self) -> int:
        return self._used

    @property
    def available(self) -> int:
        return self.limit - self._used - self._reserved

    def next_token(self, purpose: str) -> str:
        self.sequence += 1
        return f"{purpose}:{self.sequence}"

    def can_fit(self, upper_cost: int) -> bool:
        if isinstance(upper_cost, bool) or not isinstance(upper_cost, int) or upper_cost < 0:
            raise ValueError("upper cost must be a nonnegative integer")
        return upper_cost <= self.available

    def reserve(
        self,
        token: str,
        upper_cost: int,
        purpose: str,
        *,
        comparison_id: str | None = None,
        round_index: int | None = None,
        side: str | None = None,
    ) -> str:
        positive_integer(upper_cost, "reserved upper cost")
        if not isinstance(token, str) or not token or token in self.records:
            raise ValueError("reservation IDs must be unique; replay completed observations instead")
        if not self.can_fit(upper_cost):
            raise BudgetExhaustedError(f"{purpose} requires {upper_cost} metric calls; {self.available} available")
        if purpose == "evidence":
            if not comparison_id or side not in ("parent", "child"):
                raise ValueError("evidence reservations require comparison and side identities")
            positive_integer(round_index, "evidence round index")
        elif comparison_id is not None or round_index is not None or side is not None:
            raise ValueError("comparison metadata belongs only to evidence reservations")
        self.records[token] = Reservation(
            token,
            upper_cost,
            purpose,
            comparison_id=comparison_id,
            round_index=round_index,
            side=side,
        )
        self._reserved += upper_cost
        self._publish()
        return token

    def complete(self, token: str, actual_cost: int) -> None:
        record = self.records[token]
        if record.status == "completed":
            if record.actual_cost != actual_cost:
                raise ValueError("conflicting replayed metric cost")
            return
        if record.status != "reserved":
            raise ValueError("a failed or abandoned execution cannot become completed")
        if (
            isinstance(actual_cost, bool)
            or not isinstance(actual_cost, int)
            or actual_cost < 0
            or actual_cost > record.upper_cost
        ):
            self.fail(token, "adapter-reported cost violates declared upper bound")
            raise ValueError("adapter-reported metric cost must be an integer within the reserved upper bound")
        record.actual_cost = actual_cost
        record.status = "completed"
        self._reserved -= record.upper_cost
        self._used += actual_cost
        self._publish()

    def fail(self, token: str, failure: str) -> None:
        record = self.records[token]
        if record.status != "reserved":
            return
        record.status = "failed"
        record.actual_cost = record.upper_cost
        record.failure = failure
        self._reserved -= record.upper_cost
        self._used += record.upper_cost
        self._publish()

    def _publish(self) -> None:
        # Durable state precedes user callbacks, which can themselves raise.
        if self.persist is not None:
            self.persist()
        if self.listener is not None:
            self.listener(self.used)

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "initial_used": self.initial_used,
            "sequence": self.sequence,
            "records": [asdict(record) for record in self.records.values()],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, limit: int) -> MetricBudgetGuard:
        old_limit = positive_integer(value["limit"], "saved budget limit")
        if limit < old_limit:
            raise ValueError("resuming evidence requires the same or an explicitly increased metric cap")
        guard = cls(limit, initial_used=value["initial_used"])
        sequence = value["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("corrupt budget sequence")
        guard.sequence = sequence
        for item in value["records"]:
            record = Reservation(**item)
            if not record.token or record.token in guard.records:
                raise ValueError("duplicate budget reservation")
            positive_integer(record.upper_cost, "saved upper cost")
            if record.purpose == "evidence":
                if not isinstance(record.comparison_id, str) or not record.comparison_id:
                    raise ValueError("saved evidence reservation lacks comparison identity")
                if record.side not in ("parent", "child"):
                    raise ValueError("saved evidence reservation has invalid side")
                positive_integer(record.round_index, "saved evidence round")
            elif any(x is not None for x in (record.comparison_id, record.round_index, record.side)):
                raise ValueError("native reservation has unexpected comparison metadata")
            if record.status not in ("reserved", "completed", "failed", "abandoned"):
                raise ValueError("corrupt reservation status")
            if record.status == "reserved":
                record.status = "abandoned"
                record.actual_cost = record.upper_cost
                record.failure = "process stopped with unknown external completion"
            if (
                isinstance(record.actual_cost, bool)
                or not isinstance(record.actual_cost, int)
                or not 0 <= record.actual_cost <= record.upper_cost
            ):
                raise ValueError("corrupt saved metric cost")
            if record.status in ("failed", "abandoned") and record.actual_cost != record.upper_cost:
                raise ValueError("failed or abandoned executions must be charged conservatively")
            guard.records[record.token] = record
            guard._used += record.charged
        if guard.used > limit:
            raise ValueError("saved metric use exceeds the configured budget")
        return guard
