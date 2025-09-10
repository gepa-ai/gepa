from unittest.mock import Mock, patch

import pytest

from gepa import optimize
from gepa.strategies.component_selector import (
    AllReflectionComponentSelector,
    RoundRobinReflectionComponentSelector,
)


@pytest.fixture
def common_mocks():
    """Common mock setup for all module selector tests."""
    mock_run_return = Mock(
        program_candidates=[{"test": "value"}],
        parent_program_for_candidate=[None],
        program_full_scores_val_set=[0.5],
        prog_candidate_val_subscores=[[]],
        program_at_pareto_front_valset=[set()],
        num_metric_calls_by_discovery=[1],
    )

    mock_adapter = Mock()
    mock_adapter.evaluate.return_value = Mock(outputs=[], scores=[])

    return mock_run_return, mock_adapter


@patch("gepa.api.GEPAEngine.run")
@patch("gepa.api.ReflectiveMutationProposer")
def test_module_selector_default_round_robin(mock_proposer, mock_run, common_mocks):
    """Test that module_selector defaults to round robin."""
    mock_run_return, mock_adapter = common_mocks
    mock_run.return_value = mock_run_return

    # Create mock data instances
    mock_data = [Mock() for _ in range(3)]

    result = optimize(
        seed_candidate={"test": "value"},
        trainset=mock_data,
        adapter=mock_adapter,
        reflection_lm=lambda x: "test response",
        # Use default module_selector
        max_metric_calls=1,
    )

    # Verify that ReflectiveMutationProposer was called with RoundRobinReflectionComponentSelector
    mock_proposer.assert_called_once()
    call_args = mock_proposer.call_args
    module_selector = call_args.kwargs["module_selector"]
    assert isinstance(module_selector, RoundRobinReflectionComponentSelector)
    assert result is not None


@patch("gepa.api.GEPAEngine.run")
@patch("gepa.api.ReflectiveMutationProposer")
def test_module_selector_string_round_robin(mock_proposer, mock_run, common_mocks):
    """Test that module_selector='round_robin' works with optimize()."""
    mock_run_return, mock_adapter = common_mocks
    mock_run.return_value = mock_run_return

    # Create mock data instances
    mock_data = [Mock() for _ in range(3)]

    result = optimize(
        seed_candidate={"test": "value"},
        trainset=mock_data,
        adapter=mock_adapter,
        reflection_lm=lambda x: "test response",
        module_selector="round_robin",
        max_metric_calls=1,
    )

    # Verify that ReflectiveMutationProposer was called with RoundRobinReflectionComponentSelector
    mock_proposer.assert_called_once()
    call_args = mock_proposer.call_args
    module_selector = call_args.kwargs["module_selector"]
    assert isinstance(module_selector, RoundRobinReflectionComponentSelector)
    assert result is not None


@patch("gepa.api.GEPAEngine.run")
@patch("gepa.api.ReflectiveMutationProposer")
def test_module_selector_string_all(mock_proposer, mock_run, common_mocks):
    """Test that module_selector='all' works with optimize()."""
    mock_run_return, mock_adapter = common_mocks
    mock_run.return_value = mock_run_return

    # Create mock data instances to avoid empty trainset concern
    mock_data = [Mock() for _ in range(3)]

    result = optimize(
        seed_candidate={"test": "value"},
        trainset=mock_data,
        adapter=mock_adapter,
        reflection_lm=lambda x: "test response",
        module_selector="all",
        max_metric_calls=1,
    )

    # Verify that ReflectiveMutationProposer was called with AllReflectionComponentSelector
    mock_proposer.assert_called_once()
    call_args = mock_proposer.call_args
    module_selector = call_args.kwargs["module_selector"]
    assert isinstance(module_selector, AllReflectionComponentSelector)
    assert result is not None


@patch("gepa.api.GEPAEngine.run")
@patch("gepa.api.ReflectiveMutationProposer")
def test_module_selector_custom_instance(mock_proposer, mock_run, common_mocks):
    """Test that module_selector accepts custom instances with optimize()."""
    mock_run_return, mock_adapter = common_mocks
    mock_run.return_value = mock_run_return

    def custom_component_selector(state, trajectories, subsample_scores, candidate_idx, candidate):
        return ["test_component"]

    custom_selector = custom_component_selector

    # Create mock data instances
    mock_data = [Mock() for _ in range(3)]

    result = optimize(
        seed_candidate={"test": "value"},
        trainset=mock_data,
        adapter=mock_adapter,
        reflection_lm=lambda x: "test response",
        module_selector=custom_selector,
        max_metric_calls=1,
    )

    # Verify that ReflectiveMutationProposer was called with our custom selector
    mock_proposer.assert_called_once()
    call_args = mock_proposer.call_args
    module_selector = call_args.kwargs["module_selector"]
    assert module_selector is custom_selector
    assert result is not None


def test_all_reflection_component_selector_behavior():
    """Test that AllReflectionComponentSelector returns all component names from candidate."""

    # Create a mock state (not used in the new implementation)
    mock_state = Mock()

    # Call selector class instance directly - should return all components from candidate
    selector = AllReflectionComponentSelector()
    candidate = {"component1": "value1", "component2": "value2", "component3": "value3"}
    result = selector(
        state=mock_state,
        trajectories=[],
        subsample_scores=[],
        candidate_idx=0,
        candidate=candidate,
    )

    assert result == ["component1", "component2", "component3"]
    assert len(result) == 3


def test_module_selector_invalid_string_raises_error(common_mocks):
    """Test that invalid module_selector string raises AssertionError."""
    _, mock_adapter = common_mocks

    # Create mock data instances
    mock_data = [Mock() for _ in range(3)]

    with pytest.raises(AssertionError, match="Unknown module_selector strategy"):
        optimize(
            seed_candidate={"test": "value"},
            trainset=mock_data,
            adapter=mock_adapter,
            reflection_lm=lambda x: "test response",
            module_selector="invalid_strategy",
            max_metric_calls=1,
        )
