import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ListDataLoader


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
        val_evaluation_policy="round_robin_sample",
        val_evaluation_sample_size=2,
    )

    assert val_loader.expansions == 1

    covered_ids = set().union(*(scores.keys() for scores in result.val_subscores))
    assert 2 in covered_ids

    # Ensure round-robin policy limited the batch size for new candidates once the loader grew.
    non_seed_batch_sizes = {len(scores) for scores in result.val_subscores[1:]}
    assert non_seed_batch_sizes
    assert max(non_seed_batch_sizes) <= 2
