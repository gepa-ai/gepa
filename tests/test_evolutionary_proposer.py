from glean_gepa.al_adapter import Candidate, ModuleSpec
from glean_gepa.evolutionary_proposer import make_children_for_generation


class _ReflectionAdapter:
    def __init__(self) -> None:
        self.reflection_calls = 0

    def get_screening_score(self, _eval: object) -> float:
        return 1.0

    def make_reflective_dataset(self, **_kwargs: object) -> dict[str, list[dict[str, str]]]:
        return {"WRITING_CODE": [{"feedback": "fix it"}]}

    def propose_new_texts(self, **_kwargs: object) -> tuple[list[str], object]:
        self.reflection_calls += 1
        return ["cached rewrite 1", "cached rewrite 2"], None


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
