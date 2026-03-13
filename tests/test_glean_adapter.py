# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Tests for Glean adapter.

Tests cover:
- Adapter initialization
- Fake client simulation
- Evaluation and scoring
- Reflective dataset generation
"""

import pytest

from gepa.adapters.glean_adapter import GleanAdapter, GleanDataInst, GleanOutput, GleanTrajectory
from gepa.core.adapter import EvaluationBatch

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return [
        {
            "query": "What is our current PTO policy?",
            "expected_tool": "glean_search",
            "reference_answer": "20 days of PTO per year",
            "additional_context": {},
        },
        {
            "query": "Explain what AAQM is at a high level.",
            "expected_tool": None,
            "reference_answer": "AAQM is our internal framework",
            "additional_context": {},
        },
    ]


@pytest.fixture
def seed_candidate():
    """Sample seed candidate."""
    return {"system_prompt": "You are a helpful assistant with access to company knowledge."}


@pytest.fixture
def simple_metric():
    """Simple metric function that checks if expected tool was used."""

    def metric(item: GleanDataInst, output: GleanOutput) -> float:
        # If expected_tool is None, score based on reference answer
        if item["expected_tool"] is None:
            if item["reference_answer"] and item["reference_answer"] in output["final_answer"]:
                return 1.0
            return 0.0

        # Otherwise, check if the correct tool was selected
        if output["primary_tool"] == item["expected_tool"]:
            return 1.0
        return 0.0

    return metric


# ============================================================================
# Test Adapter Initialization
# ============================================================================


class TestGleanAdapterInitialization:
    """Tests for GleanAdapter initialization."""

    def test_adapter_initialization(self, simple_metric):
        """Test creating adapter with basic configuration."""
        adapter = GleanAdapter(
            metric_fn=simple_metric,
            failure_score=0.0,
        )

        assert adapter.metric_fn is not None
        assert adapter.failure_score == 0.0

    def test_adapter_custom_failure_score(self, simple_metric):
        """Test adapter with custom failure score."""
        adapter = GleanAdapter(
            metric_fn=simple_metric,
            failure_score=0.25,
        )

        assert adapter.failure_score == 0.25


# ============================================================================
# Test Evaluation
# ============================================================================


class TestGleanAdapterEvaluation:
    """Tests for GleanAdapter evaluation."""

    def test_evaluate_structure(self, sample_dataset, seed_candidate, simple_metric):
        """Test evaluation batch structure."""
        adapter = GleanAdapter(
            metric_fn=simple_metric,
            failure_score=0.0,
        )

        # Evaluate with capture_traces=False
        result = adapter.evaluate(
            batch=sample_dataset,
            candidate=seed_candidate,
            capture_traces=False,
        )

        # Verify structure
        assert isinstance(result, EvaluationBatch)
        assert len(result.outputs) == len(sample_dataset)
        assert len(result.scores) == len(sample_dataset)
        assert result.trajectories is None

    def test_evaluate_with_traces(self, sample_dataset, seed_candidate, simple_metric):
        """Test evaluation with trajectory capture."""
        adapter = GleanAdapter(
            metric_fn=simple_metric,
            failure_score=0.0,
        )

        # Evaluate with capture_traces=True
        result = adapter.evaluate(
            batch=sample_dataset,
            candidate=seed_candidate,
            capture_traces=True,
        )

        # Verify structure
        assert isinstance(result, EvaluationBatch)
        assert result.trajectories is not None
        assert len(result.trajectories) == len(sample_dataset)

        # Verify trajectory structure
        traj = result.trajectories[0]
        assert "trace_id" in traj
        assert "query" in traj
        assert "model" in traj
        assert "system_prompt_version" in traj
        assert "steps" in traj
        assert "tool_events" in traj
        assert "final_step" in traj
        assert "metrics" in traj
        assert "score" in traj

    def test_evaluate_output_structure(self, sample_dataset, seed_candidate, simple_metric):
        """Test output structure."""
        adapter = GleanAdapter(
            metric_fn=simple_metric,
            failure_score=0.0,
        )

        result = adapter.evaluate(
            batch=sample_dataset,
            candidate=seed_candidate,
            capture_traces=False,
        )

        # Verify output structure
        output = result.outputs[0]
        assert "final_answer" in output
        assert "tool_sequence" in output
        assert "primary_tool" in output
        assert "num_loops" in output


# ============================================================================
# Test Reflective Dataset
# ============================================================================


class TestGleanAdapterReflectiveDataset:
    """Tests for reflective dataset generation."""

    def test_make_reflective_dataset_system_prompt(self, sample_dataset, seed_candidate, simple_metric):
        """Test reflective dataset for system_prompt."""
        adapter = GleanAdapter(
            metric_fn=simple_metric,
            failure_score=0.0,
        )

        # First evaluate to get trajectories
        eval_batch = adapter.evaluate(
            batch=sample_dataset,
            candidate=seed_candidate,
            capture_traces=True,
        )

        # Generate reflective dataset
        reflective_data = adapter.make_reflective_dataset(
            candidate=seed_candidate,
            eval_batch=eval_batch,
            components_to_update=["system_prompt"],
        )

        # Verify structure
        assert "system_prompt" in reflective_data
        assert len(reflective_data["system_prompt"]) == len(sample_dataset)

        # Verify example structure
        example = reflective_data["system_prompt"][0]
        assert "Inputs" in example
        assert "Generated Outputs" in example
        assert "Feedback" in example

        # Verify inputs
        assert "query" in example["Inputs"]
        assert "system_prompt_version" in example["Inputs"]

        # Verify outputs
        assert "tool_sequence" in example["Generated Outputs"]
        assert "primary_tool" in example["Generated Outputs"]
        assert "final_answer" in example["Generated Outputs"]
        assert "num_loops" in example["Generated Outputs"]

    def test_feedback_generation(self, sample_dataset, seed_candidate, simple_metric):
        """Test that feedback is generated for examples."""
        adapter = GleanAdapter(
            metric_fn=simple_metric,
            failure_score=0.0,
        )

        # Evaluate with traces
        eval_batch = adapter.evaluate(
            batch=sample_dataset,
            candidate=seed_candidate,
            capture_traces=True,
        )

        # Generate reflective dataset
        reflective_data = adapter.make_reflective_dataset(
            candidate=seed_candidate,
            eval_batch=eval_batch,
            components_to_update=["system_prompt"],
        )

        # Check that all examples have feedback
        for example in reflective_data["system_prompt"]:
            assert "Feedback" in example
            assert len(example["Feedback"]) > 0


# ============================================================================
# Test Type Definitions
# ============================================================================


def test_glean_types_import():
    """Test that Glean types can be imported."""
    assert GleanDataInst is not None
    assert GleanOutput is not None
    assert GleanTrajectory is not None


def test_glean_adapter_import():
    """Test that GleanAdapter can be imported."""
    from gepa.adapters.glean_adapter import GleanAdapter

    assert GleanAdapter is not None
