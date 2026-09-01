import pytest

from gepa.core.data_loader import ListDataLoader
from glean_gepa.evalset_policy import UnseenEvalSetPolicy


def test_unseen_evalset_policy_reveals_one_training_id_at_a_time():
    loader = ListDataLoader(["v1", "v2", "v3"])
    policy = UnseenEvalSetPolicy()

    assert policy.take_unseen(loader, purpose="reflection and offspring screening") == [0]
    assert policy.take_unseen(loader, purpose="reflection and offspring screening") == [1]
    assert policy.take_unseen(loader, purpose="reflection and offspring screening") == [2]


def test_unseen_evalset_policy_fails_instead_of_reusing_seen_data():
    loader = ListDataLoader(["v1"])
    policy = UnseenEvalSetPolicy()
    policy.take_unseen(loader, purpose="reflection and offspring screening")

    with pytest.raises(RuntimeError, match="No unseen eval sets remain"):
        policy.take_unseen(loader, purpose="reflection and offspring screening")
