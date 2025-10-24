import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import DataId, DataInst, DataLoader, ListDataLoader
from gepa.core.state import GEPAState, ProgramIdx
from gepa.strategies.eval_policy import EvaluationPolicy


class AutoExpandingListLoader(ListDataLoader):
    def __init__(self, initial_items, staged_items):
        super().__init__(initial_items)
        self._staged = list(staged_items)
        self.expansions = 0

    def has_pending(self) -> bool:
        return bool(self._staged)

    def add_next_if_available(self) -> None:
        if self._staged:
            self.add_items([self._staged.pop(0)])
            self.expansions += 1


class DummyAdapter:
    """Simple adapter that increments an integer weight component."""

    def __init__(self, val_loader: AutoExpandingListLoader, expand_after: int = 2):
        self.val_loader = val_loader
        self.expand_after = expand_after
        self.val_eval_calls = 0
        self.propose_new_texts = self._propose_new_texts

    def evaluate(self, batch, candidate, capture_traces=False):
        weight = int(candidate["system_prompt"].split("=")[-1])
        outputs = [{"id": item["id"], "weight": weight} for item in batch]
        scores = [min(1.0, (weight + 1) / item["difficulty"]) for item in batch]

        if batch and batch[0].get("split") == "val":
            self.val_eval_calls += 1
            if self.val_eval_calls == self.expand_after and self.val_loader.has_pending():
                self.val_loader.add_next_if_available()

        trajectories = [{"score": score} for score in scores] if capture_traces else None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [{"score": score} for score in eval_batch.scores]
        return dict.fromkeys(components_to_update, records)

    def _propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        weight = int(candidate["system_prompt"].split("=")[-1])
        return dict.fromkeys(components_to_update, f"weight={weight + 1}")


class RoundRobinSampleEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that samples validation examples with fewest recorded evaluations."""

    def __init__(self, batch_size: int = 5):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size

    def get_eval_batch(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState,
        target_program_idx: ProgramIdx | None = None,
    ) -> list[DataId]:
        """Return ids sorted by how often they've been evaluated, preferring ids that have been least leveraged for eval (in particular preferring examples not yet evaluated)."""
        all_ids = list(loader.all_ids())
        if not all_ids:
            return []

        order_index = {val_id: idx for idx, val_id in enumerate(all_ids)}
        valset_evaluations = state.valset_evaluations

        def sort_key(val_id: DataId):
            eval_count = len(valset_evaluations.get(val_id, []))
            return (eval_count, order_index[val_id])

        ordered_ids = sorted(all_ids, key=sort_key)
        batch = ordered_ids[: self.batch_size] or ordered_ids

        return batch

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Pick the program whose evaluated validation scores achieve the highest average."""
        best_idx, best_score, best_coverage = -1, float("-inf"), -1
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            coverage = len(scores)
            avg = sum(scores.values()) / coverage if coverage else float("-inf")
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = program_idx
                best_coverage = coverage
        return best_idx

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        return state.get_program_average_val_subset(program_idx)[0]


def test_incremental_eval_policy_handles_dynamic_valset(tmp_path):
    trainset = [
        {"id": 0, "difficulty": 2, "split": "train"},
        {"id": 1, "difficulty": 3, "split": "train"},
        {"id": 2, "difficulty": 4, "split": "train"},
    ]
    initial_valset = [
        {"id": 0, "difficulty": 3, "split": "val"},
        {"id": 1, "difficulty": 4, "split": "val"},
    ]
    staged_val_items = [
        {"id": 2, "difficulty": 5, "split": "val"},
    ]

    val_loader = AutoExpandingListLoader(initial_valset, staged_val_items)
    adapter = DummyAdapter(val_loader=val_loader, expand_after=2)

    result = gepa.optimize(
        seed_candidate={"system_prompt": "weight=0"},
        trainset=trainset,
        valset=val_loader,
        adapter=adapter,
        reflection_lm=None,
        candidate_selection_strategy="current_best",
        max_metric_calls=12,
        run_dir=str(tmp_path / "run"),
        val_evaluation_policy=RoundRobinSampleEvaluationPolicy(batch_size=2),
    )

    assert val_loader.expansions == 1

    covered_ids = set().union(*(scores.keys() for scores in result.val_subscores))
    assert 2 in covered_ids

    # Ensure round-robin policy limited the batch size for new candidates once the loader grew.
    non_seed_batch_sizes = {len(scores) for scores in result.val_subscores[1:]}
    assert non_seed_batch_sizes
    assert max(non_seed_batch_sizes) <= 2
