import pytest

from gepa.core.data_loader import ListDataLoader
from glean_gepa.evalset_policy import UnseenEvalSetPolicy


def test_unseen_evalset_policy_reveals_one_id_at_a_time():
    loader = ListDataLoader(["v1", "v2", "v3"])
    policy = UnseenEvalSetPolicy()

    assert policy.get_seed_eval_batch(loader) == [0]
    assert policy.take_unseen(loader, purpose="offspring full screen") == [1]

    class State:
        pass

    assert policy.get_eval_batch(loader, State()) == [2]


def test_unseen_evalset_policy_fails_instead_of_reusing_seen_data():
    loader = ListDataLoader(["v1"])
    policy = UnseenEvalSetPolicy()
    policy.get_seed_eval_batch(loader)

    with pytest.raises(RuntimeError, match="No unseen eval sets remain"):
        policy.take_unseen(loader, purpose="offspring full screen")
