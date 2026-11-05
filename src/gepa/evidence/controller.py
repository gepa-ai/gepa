"""Serial native comparison coordinator with write-ahead scorer reservations."""

from __future__ import annotations

from typing import Any

from gepa.evidence.adapter import GuardedAdapter
from gepa.evidence.budget import MetricBudgetGuard
from gepa.evidence.config import SequentialEvidenceConfig, digest, positive_integer
from gepa.evidence.observations import TrialContext, candidate_identity, example_identity
from gepa.evidence.persistence import EvidenceStore
from gepa.evidence.state import ComparisonState
from gepa.proposer.base import CandidateProposal


class EvidenceController:
    def __init__(
        self,
        config: SequentialEvidenceConfig,
        adapter: Any,
        trainset: Any,
        valset: Any,
        *,
        max_metric_calls: int,
        run_dir: str | None,
        seed: int,
    ) -> None:
        positive_integer(max_metric_calls, "max_metric_calls")
        self.config = config
        self.trainset = trainset
        self.valset = valset
        self.seed = seed
        # Fixed data are part of the context, not just list indices. Fail
        # explicitly for opaque/non-JSON examples rather than hash repr().
        try:
            context = digest(
                {
                    "config": config.to_dict(),
                    "seed": seed,
                    "train_ids": list(trainset.all_ids()),
                    "val_ids": list(valset.all_ids()),
                    "train": trainset.fetch(list(trainset.all_ids())),
                    "val": valset.fetch(list(valset.all_ids())),
                }
            )
        except (TypeError, ValueError) as error:
            raise ValueError("initial evidence release requires stable JSON-serializable data and IDs") from error
        self.store = EvidenceStore(run_dir, context)
        self.comparisons: dict[str, ComparisonState] = {}
        self.batch_sequence = 0
        self.stop_requested = False
        self.native_state: Any = None
        saved = self.store.load()
        if saved is None:
            self.guard = MetricBudgetGuard(max_metric_calls)
        else:
            self.guard = MetricBudgetGuard.from_dict(saved["guard"], limit=max_metric_calls)
            self.batch_sequence = saved["batch_sequence"]
            for item in saved["comparisons"]:
                record = ComparisonState.from_dict(item, config)
                if record.comparison_id in self.comparisons or record.index != len(self.comparisons) + 1:
                    raise ValueError("invalid saved comparison allocation order")
                self.comparisons[record.comparison_id] = record
        self.adapter = GuardedAdapter(adapter, self.guard, config)
        self.guard.persist = self.save
        self.save()

    def snapshot(self) -> dict[str, Any]:
        return {
            "context": self.store.context,
            "guard": self.guard.to_dict(),
            "batch_sequence": self.batch_sequence,
            "comparisons": [c.to_dict() for c in self.comparisons.values()],
        }

    def save(self) -> None:
        snapshot = self.snapshot()
        self.store.save(snapshot)
        if self.native_state is not None:
            self.native_state.sequential_evidence_state = snapshot
            self.native_state.sequential_evidence_report = self.report()

    def attach_state(self, state: Any, *, resumed: bool) -> None:
        native_snapshot = getattr(state, "sequential_evidence_state", None)
        if native_snapshot is not None and native_snapshot.get("context") != self.store.context:
            raise ValueError("native checkpoint evidence context differs from requested configuration")
        if resumed and native_snapshot is not None and not self.comparisons and not self.guard.records:
            raise ValueError("evidence sidecar is missing; cannot reconstruct in-flight work from stale checkpoint")
        if resumed and native_snapshot is None and not self.guard.records:
            # Legacy checkpoints have ordinary metric use but no independent
            # evidence. Preserve their costs and start comparison index one.
            self.guard = MetricBudgetGuard(self.guard.limit, initial_used=state.total_num_evals)
            self.guard.persist = self.save
            self.adapter.guard = self.guard
        if resumed and state.total_num_evals > self.guard.used:
            raise ValueError("native checkpoint metric count is ahead of evidence ledger")
        self.native_state = state
        state._evidence_budget_guard = self.guard
        self.guard.listener = state.synchronize_evidence_budget
        state.synchronize_evidence_budget(self.guard.used)
        # A pool write may have committed just before process termination,
        # leaving its promotion journal entry pending. Content identity makes
        # that reconciliation idempotent without adding the candidate twice.
        pool = {candidate_identity(candidate): idx for idx, candidate in enumerate(state.program_candidates)}
        for record in self.comparisons.values():
            if record.selected and candidate_identity(record.child) in pool:
                record.status = "promoted" if record.selection_label == "supported" else "exploration"
                record.validation_candidate_idx = pool[candidate_identity(record.child)]
        self.save()

    def start_batch(self) -> None:
        self.batch_sequence += 1
        self.save()

    def compare(self, proposal: CandidateProposal, parent: dict[str, str]) -> ComparisonState:
        ids = proposal.subsample_indices
        if not ids:
            raise ValueError("evidence comparisons require a frozen nonempty proposal batch")
        keys = [example_identity(eid) for eid in ids]
        if len(set(keys)) != len(keys):
            raise ValueError("evidence proposal example IDs must be unique")
        if len(proposal.parent_program_ids) != 1:
            raise ValueError("initial evidence release requires exactly one reflective parent")
        weights = self.config.weights_for(len(keys))
        identity = digest(
            {
                "batch": self.batch_sequence,
                "parent": candidate_identity(parent),
                "child": candidate_identity(proposal.candidate),
                "strata": keys,
            }
        )
        existing = self.comparisons.get(identity)
        if existing is not None:
            proposal.metadata["sequential_evidence_id"] = identity
            return existing
        # Fetch before reservations so invalid IDs cannot consume scorer budget.
        examples = self.trainset.fetch(ids)
        if len(examples) != len(ids):
            raise ValueError("loader did not return all frozen evidence examples")
        record = ComparisonState(
            comparison_id=identity,
            index=len(self.comparisons) + 1,
            parent=dict(parent),
            child=dict(proposal.candidate),
            parent_ids=list(proposal.parent_program_ids),
            example_ids=list(ids),
            example_keys=keys,
            weights=weights,
            support=self.config.assumption if self.config.confidence_supported else "none",
            diagnostics_before=proposal.subsample_scores_before,
            diagnostics_after=proposal.subsample_scores_after,
        )
        record.initialize_intervals(self.config)
        self.comparisons[identity] = record
        proposal.metadata["sequential_evidence_id"] = identity
        self.save()
        while not record.terminal:
            upper_side = self.adapter.upper_cost(len(examples))
            upper_round = 2 * upper_side
            if record.metric_calls + upper_round > self.config.max_comparison_calls or not self.guard.can_fit(
                upper_round
            ):
                record.status = "budget_exhausted" if self.config.confidence_supported else "heuristic"
                record.reason = "a complete balanced round cannot fit the comparison or global metric cap"
                self.save()
                break
            parent_ids, child_ids = record.reserve_round(self.config)
            tokens = []
            # Both sides are reserved before either is dispatched.
            for side in ("parent", "child"):
                token = self.guard.next_token(f"evidence:{record.index}:{record.completed_rounds + 1}:{side}")
                tokens.append(self.guard.reserve(token, upper_side, "evidence"))
            before = self.guard.used
            try:
                for side, candidate, observation_ids, token in (
                    ("parent", parent, parent_ids, tokens[0]),
                    ("child", proposal.candidate, child_ids, tokens[1]),
                ):
                    trial = observation_ids[0].trial
                    context = TrialContext(
                        identity,
                        record.completed_rounds + 1,
                        trial,
                        side,
                        tuple(keys),
                        int(digest([self.seed, identity, record.completed_rounds + 1])[:16], 16),
                    )
                    result = self.adapter.evaluate_replication(examples, candidate, trial_context=context)
                    cost = result.num_metric_calls if result.num_metric_calls is not None else len(examples)
                    record.complete_side(observation_ids, result.scores, self.config)
                    self.guard.complete(token, cost)
                record.finish_round(parent_ids, child_ids, self.config)
            except Exception as error:
                reason = f"informative missingness or invalid outcome: {type(error).__name__}: {error}"
                for token in tokens:
                    self.guard.fail(token, reason)
                record.fail_round(parent_ids + child_ids, reason)
            finally:
                record.metric_calls += self.guard.used - before
                self.save()
        return record

    def record_for(self, proposal: CandidateProposal) -> ComparisonState:
        return self.comparisons[proposal.metadata["sequential_evidence_id"]]

    def select(self, proposal: CandidateProposal, *, supported: bool, iteration_id: str) -> None:
        record = self.record_for(proposal)
        record.selected = True
        record.selection_label = "supported" if supported else "unresolved_exploration"
        record.iteration_id = iteration_id
        if not supported:
            record.reason = "selection strategy or explicit unresolved policy overrode statistical/criterion gate"
        self.save()

    def defer_validation(self, record: ComparisonState) -> None:
        record.status = "validation_deferred"
        record.reason = "complete validation cannot fit remaining metric budget; candidate not promoted"
        self.stop_requested = True
        self.save()

    def promoted(self, record: ComparisonState, candidate_idx: int) -> None:
        record.validation_candidate_idx = candidate_idx
        record.status = "promoted" if record.selection_label == "supported" else "exploration"
        record.reason = "complete native validation published to candidate pool"
        self.save()

    def pending_validation(self) -> list[ComparisonState]:
        return [c for c in self.comparisons.values() if c.selected and c.validation_candidate_idx is None]

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": EvidenceStore.SCHEMA_VERSION,
            "context": self.store.context,
            "assumption": self.config.assumption,
            "alpha": self.config.alpha,
            "metric_limit": self.guard.limit,
            "metric_calls": self.guard.used,
            "reserved_calls": self.guard.limit - self.guard.used - self.guard.available,
            "comparisons": [record.summary() for record in self.comparisons.values()],
            "validation_confidence_calibrated": False,
        }


class EvidenceAcceptance:
    """Compose evidence before the original, externally memoized criterion."""

    def __init__(self, controller: EvidenceController, original: Any) -> None:
        self.controller = controller
        self.original = original
        self.original_verdicts: dict[int, bool] = {}

    def should_accept(self, proposal: CandidateProposal, state: Any) -> bool:
        record = self.controller.record_for(proposal)
        evidence_allows = record.eligible or (
            self.controller.config.unresolved == "explore" and record.status in ("budget_exhausted", "heuristic")
        )
        if not evidence_allows:
            return False
        key = id(proposal)
        if key not in self.original_verdicts:
            self.original_verdicts[key] = bool(self.original.should_accept(proposal, state))
        return self.original_verdicts[key]

    def confidence_supported(self, proposal: CandidateProposal) -> bool:
        return self.controller.record_for(proposal).eligible and self.original_verdicts.get(id(proposal), False)

    def reject_reason(self, proposal: CandidateProposal, state: Any) -> str:
        record = self.controller.record_for(proposal)
        return f"Sequential evidence {record.status}: {record.reason}"
