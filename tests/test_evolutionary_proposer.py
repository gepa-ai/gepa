import json
from typing import ClassVar
from unittest.mock import MagicMock

from gepa.core.engine import GEPAEngine
from gepa.strategies.acceptance import StrictImprovementAcceptance
from glean_gepa.al_adapter import Candidate, ModuleSpec
from glean_gepa.batch import GleanEvaluationBatch
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.evolutionary_proposer import EvolutionaryProposer, make_children_for_generation


class _ReflectionAdapter:
    def __init__(self, variants: list[str] | None = None) -> None:
        self.reflection_calls = 0
        self.variants = variants if variants is not None else ["cached rewrite 1", "cached rewrite 2"]

    def get_screening_score(self, _eval: object) -> float:
        return 1.0

    def make_reflective_dataset(self, **_kwargs: object) -> dict[str, list[dict[str, str]]]:
        return {"WRITING_CODE": [{"feedback": "fix it"}]}

    def propose_new_texts(self, **_kwargs: object) -> tuple[list[str], object]:
        self.reflection_calls += 1
        return self.variants, None


class _Evaluation:
    def __init__(self) -> None:
        self.trajectories = [object()]


class _ProposerAdapter(_ReflectionAdapter):
    supports_high_signal_eval = False

    def __init__(self) -> None:
        super().__init__()
        self.root_evaluation_calls = 0
        self.screen_evaluation_calls = 0

    def evaluate(self, _batch, _candidate, capture_traces=False):
        self.root_evaluation_calls += 1
        return _Evaluation()

    def batch_evaluate(self, items, *, capture_traces=True):
        self.screen_evaluation_calls += 1
        return [
            GleanEvaluationBatch(outputs=[{}], scores=[1.0], summary={"objective": 1.0}) for _candidate, _batch in items
        ]

    @staticmethod
    def attach_cached_eval_run_ids(batch, _eval_run_ids):
        return batch


class _HighSignalProposerAdapter(_ProposerAdapter):
    supports_high_signal_eval = True

    def get_screening_score(self, _eval: object) -> float:
        return 0.9467353951890034

    @staticmethod
    def high_signal_batch(_parent_eval: object) -> list[dict[str, object]]:
        return [{}]

    @staticmethod
    def prepare_high_signal_batch(batch: list[dict[str, object]]) -> list[dict[str, object]]:
        return batch

    @staticmethod
    def high_signal_fix_rate(_parent_eval: object, _child_eval: object) -> float:
        return 7 / 17


class _OneSliceLoader:
    def all_ids(self):
        return [0]

    def fetch(self, _ids):
        return [{}]

    def __len__(self):
        return 1


def _candidate(candidate_id: str, text: str = "original") -> Candidate:
    return Candidate(
        model="test",
        prompt_modules={"WRITING_CODE": text},
        module_specs={"WRITING_CODE": ModuleSpec("WRITING_CODE", "free_text", 100)},
        global_token_cap=100,
        baseline_prompt_hash="baseline",
        candidate_id=candidate_id,
    )


def _proposer(adapter: _ReflectionAdapter, cache_file: str) -> EvolutionaryProposer:
    return EvolutionaryProposer(
        logger=MagicMock(),
        trainset=[],
        al_adapter=adapter,  # type: ignore[arg-type]
        reflection_llm=object(),
        experiment_tracker=MagicMock(),
        model="test",
        module_specs={"WRITING_CODE": ModuleSpec("WRITING_CODE", "free_text", 100)},
        global_token_cap=100,
        baseline_prompt_hash="baseline",
        evalset_policy=UnseenEvalSetPolicy(),
        children_cache_file=cache_file,
    )


def test_reuses_children_cached_for_root_without_rereflecting() -> None:
    adapter = _ReflectionAdapter()
    root = _candidate("root")
    children_by_root: dict[str, list[Candidate]] = {}
    frontier_evals = {root.candidate_id: _Evaluation()}

    first = make_children_for_generation(
        adapter, [root], frontier_evals, reflection_llm=object(), offspring_count=2, children_by_root=children_by_root
    )
    second = make_children_for_generation(
        adapter, [root], frontier_evals, reflection_llm=object(), offspring_count=2, children_by_root=children_by_root
    )

    assert adapter.reflection_calls == 1
    assert [child.prompt_modules for child in second] == [child.prompt_modules for child in first]
    assert children_by_root[root.candidate_id] == first


def test_prints_child_prompt_delta_against_parent(capsys) -> None:
    adapter = _ReflectionAdapter(variants=["first line\nupdated line\n"])
    root = _candidate("root", "first line\noriginal line\n")

    make_children_for_generation(
        adapter,
        [root],
        {root.candidate_id: _Evaluation()},
        reflection_llm=object(),
        offspring_count=1,
    )

    output = capsys.readouterr().out
    assert "Prompt delta for child" in output
    assert "--- parent/root/WRITING_CODE" in output
    assert "-original line" in output
    assert "+updated line" in output


def test_empty_reflection_result_marks_root_as_cached() -> None:
    adapter = _ReflectionAdapter(variants=[])
    root = _candidate("root")
    children_by_root: dict[str, list[Candidate]] = {}
    frontier_evals = {root.candidate_id: _Evaluation()}

    first = make_children_for_generation(
        adapter, [root], frontier_evals, reflection_llm=object(), children_by_root=children_by_root
    )
    second = make_children_for_generation(
        adapter, [root], frontier_evals, reflection_llm=object(), children_by_root=children_by_root
    )

    assert first == second == []
    assert adapter.reflection_calls == 1
    assert children_by_root == {root.candidate_id: []}


def test_children_cache_survives_proposer_restart(tmp_path) -> None:
    cache_file = str(tmp_path / "children.json")
    root = _candidate("root")
    frontier_evals = {root.candidate_id: _Evaluation()}
    first_adapter = _ReflectionAdapter()
    first_proposer = _proposer(first_adapter, cache_file)
    first_slice_cache = first_proposer._children_by_root_by_train_slice.setdefault((0,), {})

    first = make_children_for_generation(
        first_adapter,
        [root],
        frontier_evals,
        reflection_llm=object(),
        offspring_count=2,
        children_by_root=first_slice_cache,
    )
    first_proposer._record_eval_run_ids(
        (0,),
        first[0],
        [
            {
                "eval_set_name": "focused",
                "eval_set_version": "v1",
                "student_eval_run_id": "eval-child-1",
            }
        ],
    )
    first_proposer._save_children_cache()
    cached_child = json.loads((tmp_path / "children.json").read_text())["training_slices"][0]["roots"]["root"][0]
    assert cached_child["prompt_modules"] == first[0].prompt_modules
    assert cached_child["eval_run_ids"][0]["student_eval_run_id"] == "eval-child-1"

    second_adapter = _ReflectionAdapter()
    second_proposer = _proposer(second_adapter, cache_file)
    second = make_children_for_generation(
        second_adapter,
        [root],
        frontier_evals,
        reflection_llm=object(),
        offspring_count=2,
        children_by_root=second_proposer._children_by_root_by_train_slice[(0,)],
    )

    assert first_adapter.reflection_calls == 1
    assert second_adapter.reflection_calls == 0
    assert [child.prompt_modules for child in second] == [child.prompt_modules for child in first]
    assert second_proposer._cached_eval_run_ids((0,), second[0]) == [
        {
            "eval_set_name": "focused",
            "eval_set_version": "v1",
            "student_eval_run_id": "eval-child-1",
        }
    ]


def test_children_cache_persists_screening_result_with_eval_id(tmp_path) -> None:
    cache_file = str(tmp_path / "children.json")
    root = _candidate("root")
    frontier_evals = {root.candidate_id: _Evaluation()}
    first_adapter = _ReflectionAdapter()
    first_proposer = _proposer(first_adapter, cache_file)
    first_slice_cache = first_proposer._children_by_root_by_train_slice.setdefault((0,), {})

    children = make_children_for_generation(
        first_adapter,
        [root],
        frontier_evals,
        reflection_llm=object(),
        offspring_count=2,
        children_by_root=first_slice_cache,
    )
    first_proposer._record_eval_run_ids(
        (0,),
        children[0],
        [
            {
                "eval_set_name": "focused",
                "eval_set_version": "v1",
                "student_eval_run_id": "eval-child-1",
            }
        ],
    )
    first_proposer._record_screening_result((0,), children[0], 0.75, True)
    # Simulate a result persisted under the old 50% gate. The score must be
    # reconsidered when the threshold changes.
    first_proposer._record_screening_result((0,), children[1], 1 / 3, False)
    first_proposer._save_children_cache()

    second_proposer = _proposer(_ReflectionAdapter(), cache_file)
    cached = second_proposer._cached_screening_scores(
        (0,),
        second_proposer._children_by_root_by_train_slice[(0,)]["root"],
        use_high_signal_gate=True,
    )

    assert cached == [(0.75, True), (1 / 3, True)]


def test_same_root_and_training_slice_reuses_children_and_screen(tmp_path) -> None:
    cache_file = str(tmp_path / "children.json")
    root = _candidate("root")

    class _State:
        i = -1
        program_candidates: ClassVar[list[dict[str, str]]] = [root.prompt_modules]
        total_num_evals = 0
        num_full_ds_evals = 1
        program_full_scores_val_set: ClassVar[list[float]] = [1.0]

        @staticmethod
        def get_pareto_front_mapping():
            return {0: {0}}

    first_adapter = _ProposerAdapter()
    first_proposer = _proposer(first_adapter, cache_file)
    first_proposer.trainset = _OneSliceLoader()
    first_proposer.propose(_State())

    second_adapter = _ProposerAdapter()
    second_proposer = _proposer(second_adapter, cache_file)
    second_proposer.trainset = _OneSliceLoader()
    second_proposer.propose(_State())

    assert first_adapter.reflection_calls == 1
    assert first_adapter.root_evaluation_calls == 1
    assert first_adapter.screen_evaluation_calls == 1
    assert second_adapter.reflection_calls == 0
    assert second_adapter.root_evaluation_calls == 0
    assert second_adapter.screen_evaluation_calls == 0


def test_high_signal_screen_uses_zero_baseline_before_full_validation(tmp_path) -> None:
    """A high-signal fix rate must not be compared to the parent's full score."""
    root = _candidate("root")

    class _State:
        i = -1
        program_candidates: ClassVar[list[dict[str, str]]] = [root.prompt_modules]
        total_num_evals = 0
        num_full_ds_evals = 1
        program_full_scores_val_set: ClassVar[list[float]] = [1.0]

        @staticmethod
        def get_pareto_front_mapping():
            return {0: {0}}

    proposer = _proposer(_HighSignalProposerAdapter(), str(tmp_path / "children.json"))
    proposer.trainset = _OneSliceLoader()

    proposals = proposer.propose(_State())

    assert proposals
    assert all(proposal.tag == "evolutionary_high_signal" for proposal in proposals)
    assert all(proposal.subsample_scores_before == [0.0] for proposal in proposals)
    assert all(proposal.subsample_scores_after == [7 / 17] for proposal in proposals)
    assert all(StrictImprovementAcceptance().should_accept(proposal, _State()) for proposal in proposals)


def test_display_iteration_advances_only_after_full_evaluation() -> None:
    class _State:
        i = 5
        num_full_ds_evals = 1

    assert EvolutionaryProposer.get_display_iteration(_State()) == 1
    _State.num_full_ds_evals += 1
    assert EvolutionaryProposer.get_display_iteration(_State()) == 2


def test_engine_keeps_the_stamped_full_eval_iteration_during_validation() -> None:
    class _State:
        i = 5
        num_full_ds_evals = 1
        full_program_trace: ClassVar[list[dict[str, int]]] = []

    engine = MagicMock()
    engine.reflective_proposer = EvolutionaryProposer

    assert GEPAEngine._next_display_iteration(engine, _State()) == 1
    _State.full_program_trace.append({"display_iteration": 1})
    _State.num_full_ds_evals += 1
    assert GEPAEngine._display_iteration(engine, _State()) == 1
