from turbo_gepa.cache import DiskCache
from turbo_gepa.interfaces import Candidate, EvalResult


def test_save_load_state_preserves_eval_results(tmp_path):
    cache = DiskCache(str(tmp_path))
    candidate = Candidate(text="prompt", meta={"foo": "bar"})
    result = EvalResult(objectives={"quality": 0.5}, traces=[], n_examples=1, shard_fraction=1.0)

    cache.save_state(
        round_num=1,
        evaluations=3,
        pareto_entries=[(candidate, result)],
        queue=[],
    )

    state = cache.load_state()
    assert state is not None
    assert "pareto" in state
    assert len(state["pareto"]) == 1
    entry = state["pareto"][0]
    assert entry["candidate"].text == "prompt"
    assert isinstance(entry["result"], EvalResult)
    assert entry["result"].objectives["quality"] == 0.5
