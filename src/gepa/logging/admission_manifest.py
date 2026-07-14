# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Admission manifest: a per-decision JSONL audit ledger under ``run_dir``.

When enabled, GEPA appends one JSON line per candidate-admission decision
(accepted or rejected), keyed by candidate hash and parent hashes, so a run can
be audited and reproduced from local files alone: *why was this mutation admitted
into the candidate pool, on what evidence?*

Every recorded field is derivable from facts GEPA already holds, so the manifest
is **replayable for free**: a verifier can recompute the candidate/parent hashes
and re-derive the decision from the recorded subsample scores without re-running
the optimizer (see :func:`verify_record` and ``tests/test_admission_manifest.py``).

Design notes:
- One complete JSON object per line, appended and flushed (not an atomic
  whole-document replace), because the manifest is an append-only log.
- The recorded ``decision`` is the *subsample acceptance* decision. The
  ``validation`` block is post-decision enrichment (full-val results for an
  accepted candidate), not a second admission gate.
- ``components_changed`` is derived from a candidate-versus-parent diff rather
  than proposer metadata, which can be absent for custom adapters/proposers.
- The acceptance criterion *and* its comparison operator are recorded on every
  line, so :func:`verify_record` re-derives the decision from local files alone
  (the caller need not know whether the run used strict-improvement,
  improvement-or-equal, or the merge gate).

Opt-in via ``EngineConfig(write_admission_manifest=True)`` or
``optimize(write_admission_manifest=True)``, both of which require ``run_dir``.
Off by default and zero-cost when off.
"""

import json
import os
from typing import Any

from gepa.core.state import _candidate_hash
from gepa.gepa_utils import json_default

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "admission_manifest.jsonl"

# Canonical names for GEPA's built-in subsample acceptance criteria. The first
# two match the string values accepted by ``optimize(acceptance_criterion=...)``;
# ``merge_gate`` is the (instance-less) rule the merge path applies. Each record
# stores its criterion name *and* the comparison operator below, so a verifier
# re-derives the decision from local files alone, without knowing how the run was
# configured or what GEPA's internal class names are.
STRICT_IMPROVEMENT = "strict_improvement"
IMPROVEMENT_OR_EQUAL = "improvement_or_equal"
MERGE_GATE = "merge_gate"

# Comparison applied between the new and old subsample score for each criterion.
_ACCEPTANCE_OPERATORS = {
    STRICT_IMPROVEMENT: ">",
    IMPROVEMENT_OR_EQUAL: ">=",
    MERGE_GATE: ">=",
}

# Built-in acceptance-criterion class name -> canonical manifest name.
_CRITERION_CLASS_TO_NAME = {
    "StrictImprovementAcceptance": STRICT_IMPROVEMENT,
    "ImprovementOrEqualAcceptance": IMPROVEMENT_OR_EQUAL,
}


def criterion_name(acceptance_criterion: object) -> str:
    """Canonical manifest name for an acceptance-criterion instance or class.

    Returns the public string name (``"strict_improvement"`` /
    ``"improvement_or_equal"``) for a built-in criterion, matching
    ``optimize(acceptance_criterion=...)``. For a custom criterion returns its
    class name, so the rule is still identified in the audit log even though its
    score decision is out of scope for replay.
    """
    cls = acceptance_criterion if isinstance(acceptance_criterion, type) else type(acceptance_criterion)
    return _CRITERION_CLASS_TO_NAME.get(cls.__name__, cls.__name__)


def acceptance_operator(acceptance_criterion: str | None) -> str | None:
    """Comparison operator (``">"`` / ``">="``) for a recorded criterion name.

    Returns ``None`` for a custom or unknown criterion, whose score decision is
    not re-derivable from the two recorded scalars and is therefore out of scope
    for replay (hashes and ``components_changed`` are still verified).
    """
    if acceptance_criterion is None:
        return None
    return _ACCEPTANCE_OPERATORS.get(acceptance_criterion)


_MISSING = object()


def components_changed(candidate: dict[str, str], parents: list[dict[str, str]]) -> list[str]:
    """Sorted component names whose text differs from at least one parent.

    Deterministic and replayable: computed from the candidate and parent
    dictionaries, not from proposer metadata. For the common single-parent
    reflective case this is exactly the set of components the mutation changed;
    for a two-parent merge it is the components that differ from some parent.
    """
    if not parents:
        return sorted(candidate)
    names: set[str] = set(candidate)
    for parent in parents:
        names |= set(parent)
    return sorted(name for name in names if any(candidate.get(name) != parent.get(name) for parent in parents))


def build_decision_record(
    *,
    iteration: int,
    proposal_kind: str,
    decision: str,
    decision_reason: str,
    acceptance_criterion: str,
    candidate: dict[str, str],
    parent_ids: list[int],
    parents: list[dict[str, str]],
    subsample_ids: list[Any] | None,
    subsample_score_before: float | None,
    subsample_score_after: float | None,
    metric_calls_before_decision: int,
) -> dict[str, Any]:
    """Build the decision-time snapshot of an admission record.

    Captures identity, lineage, the subsample acceptance inputs, and the
    decision. ``candidate_idx`` and ``validation`` are left null here; for an
    accepted candidate they are filled after the full-val evaluation.

    ``subsample_score_before`` is the acceptance *baseline* the candidate's
    subsample score was compared against: for reflective proposals the parent's
    subsample sum, for a merge ``max`` of the parent sums (the merge gate). This
    keeps the decision re-derivable from the two recorded scalars.

    ``acceptance_criterion`` is the canonical name of the rule that produced the
    decision (see :func:`criterion_name`); the matching comparison operator is
    recorded alongside it so :func:`verify_record` replays without the caller
    re-supplying the strategy.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "iteration": iteration,
        "proposal_kind": proposal_kind,
        "decision": decision,
        "decision_reason": decision_reason,
        "decision_basis": "subsample_acceptance",
        "acceptance_criterion": acceptance_criterion,
        "acceptance_operator": acceptance_operator(acceptance_criterion),
        "candidate_hash": _candidate_hash(candidate),
        "candidate_idx": None,
        "parent_candidate_ids": list(parent_ids),
        "parent_candidate_hashes": [_candidate_hash(p) for p in parents],
        "components_changed": components_changed(candidate, parents),
        "subsample_ids": list(subsample_ids) if subsample_ids is not None else None,
        "subsample_score_before": subsample_score_before,
        "subsample_score_after": subsample_score_after,
        "metric_calls_before_decision": metric_calls_before_decision,
        # Post-decision enrichment; null for rejects (see module docstring).
        "validation": None,
        # Cache hit/miss accounting is not tracked in v1.
        "cache_hits": None,
        "cache_misses": None,
    }


class AdmissionManifestWriter:
    """Appends one complete JSON record per line to ``admission_manifest.jsonl``.

    A dedicated single-record append (serialize a complete line, append, flush),
    not an atomic whole-document replace: the manifest is append-only. A process
    killed mid-append can leave a truncated final line; :func:`read_manifest`
    tolerates that.
    """

    def __init__(self, path: str):
        self.path = path

    def append_record(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=json_default)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def read_manifest(path: str) -> list[dict[str, Any]]:
    """Read all complete records, tolerating a truncated final line.

    A process killed mid-append can leave a partial final line; every prior line
    is a complete record. Parse line by line and drop only a final line that
    fails to parse; a non-final unparseable line is a real corruption and raises.
    """
    records: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if idx == len(lines) - 1:
                break  # tolerated: truncated final line from an interrupted append
            raise
    return records


def verify_record(
    record: dict[str, Any],
    candidate: dict[str, str],
    parents: list[dict[str, str]],
    *,
    acceptance: str | None = None,
) -> bool:
    """Replay check: recompute hashes and re-derive the decision from the record.

    Returns True if the record is internally consistent with the given candidate
    and parent dictionaries. Hashes and ``components_changed`` are checked for
    every record; the decision is re-derived from the recorded subsample scores
    using the operator the record itself stored (``acceptance_operator``), so the
    caller does not need to know which acceptance strategy the run used. A custom
    criterion (operator recorded as null) has its score decision left out of
    scope; its hashes and ``components_changed`` are still verified.

    ``acceptance`` is an optional override for older records written before the
    criterion was recorded; the recorded operator always takes precedence.
    """
    if record["candidate_hash"] != _candidate_hash(candidate):
        return False
    if record["parent_candidate_hashes"] != [_candidate_hash(p) for p in parents]:
        return False
    if record["components_changed"] != components_changed(candidate, parents):
        return False
    before = record["subsample_score_before"]
    after = record["subsample_score_after"]
    if before is not None and after is not None:
        op = record.get("acceptance_operator", _MISSING)
        if op is _MISSING:
            # Record predates the recorded criterion: fall back to the recorded
            # name, then a caller override, then the merge-gate / strict default.
            name = record.get("acceptance_criterion") or acceptance
            op = acceptance_operator(name)
            if op is None:
                op = ">=" if record.get("proposal_kind") == "merge" else ">"
        if op == ">=":
            improved = after >= before
        elif op == ">":
            improved = after > before
        else:
            return True  # custom/unknown criterion: score decision out of scope
        expected = "accepted" if improved else "rejected"
        if record["decision"] != expected:
            return False
    return True
