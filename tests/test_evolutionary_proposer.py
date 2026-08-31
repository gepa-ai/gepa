import json
from unittest.mock import MagicMock

from glean_gepa.al_adapter import Candidate, ModuleSpec
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
