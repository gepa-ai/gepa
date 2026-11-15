# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the candidate tree visualization module."""

from gepa.visualization import candidate_tree_dot_from_data, candidate_tree_html_from_data


def _sample_data():
    """Create minimal optimization data for testing."""
    candidates = [
        {"system_prompt": "You are a helpful assistant."},
        {"system_prompt": "You are an expert math tutor. Show step-by-step solutions."},
        {"system_prompt": "You are a precise math solver. Always verify your answer."},
    ]
    parents = [[None], [0], [0]]
    val_scores = [0.5, 0.7, 0.65]
    pareto_front = {
        "ex_0": {1},
        "ex_1": {1, 2},
        "ex_2": {2},
    }
    return candidates, parents, val_scores, pareto_front


class TestCandidateTreeDot:
    def test_returns_valid_dot(self):
        candidates, parents, val_scores, pareto_front = _sample_data()
        dot = candidate_tree_dot_from_data(candidates, parents, val_scores, pareto_front)
        assert dot.startswith("digraph G {")
        assert dot.endswith("}")

    def test_contains_all_nodes(self):
        candidates, parents, val_scores, pareto_front = _sample_data()
        dot = candidate_tree_dot_from_data(candidates, parents, val_scores, pareto_front)
        for idx in range(len(candidates)):
            assert f'    {idx} [label="' in dot

    def test_contains_edges(self):
        candidates, parents, val_scores, pareto_front = _sample_data()
        dot = candidate_tree_dot_from_data(candidates, parents, val_scores, pareto_front)
        assert "0 -> 1;" in dot
        assert "0 -> 2;" in dot

    def test_best_node_is_cyan(self):
        candidates, parents, val_scores, pareto_front = _sample_data()
        dot = candidate_tree_dot_from_data(candidates, parents, val_scores, pareto_front)
        # Candidate 1 has the highest score (0.7)
        assert "fillcolor=cyan" in dot

    def test_tooltip_suppressed_in_dot(self):
        """DOT tooltip is a space (native SVG tooltip suppressed); candidate text lives in HTML JS."""
        candidates, parents, val_scores, pareto_front = _sample_data()
        dot = candidate_tree_dot_from_data(candidates, parents, val_scores, pareto_front)
        # Candidate text should NOT be in DOT (moved to HTML JS tooltip)
        assert "helpful assistant" not in dot
        assert 'tooltip=" "' in dot

    def test_single_candidate(self):
        dot = candidate_tree_dot_from_data(
            candidates=[{"prompt": "test"}],
            parents=[[None]],
            val_scores=[0.5],
            pareto_front_programs={"ex_0": {0}},
        )
        assert "digraph G {" in dot
        assert '0 [label="' in dot


class TestCandidateTreeHtml:
    def test_returns_valid_html(self):
        candidates, parents, val_scores, pareto_front = _sample_data()
        html = candidate_tree_html_from_data(candidates, parents, val_scores, pareto_front)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_contains_dot_string(self):
        candidates, parents, val_scores, pareto_front = _sample_data()
        html = candidate_tree_html_from_data(candidates, parents, val_scores, pareto_front)
        assert "digraph G" in html

    def test_contains_node_metadata(self):
        candidates, parents, val_scores, pareto_front = _sample_data()
        html = candidate_tree_html_from_data(candidates, parents, val_scores, pareto_front)
        # Node data should be embedded as JSON
        assert '"score"' in html
        assert '"components"' in html
        assert "helpful assistant" in html

    def test_contains_viz_js_cdn(self):
        candidates, parents, val_scores, pareto_front = _sample_data()
        html = candidate_tree_html_from_data(candidates, parents, val_scores, pareto_front)
        assert "viz-standalone.mjs" in html

    def test_contains_tooltip_elements(self):
        candidates, parents, val_scores, pareto_front = _sample_data()
        html = candidate_tree_html_from_data(candidates, parents, val_scores, pareto_front)
        assert 'id="tooltip"' in html
        assert "showTooltip" in html


class TestGEPAResultVisualization:
    def test_result_candidate_tree_dot(self):
        """Test that GEPAResult.candidate_tree_dot() works."""
        from gepa.core.result import GEPAResult

        candidates, parents, val_scores, pareto_front = _sample_data()
        result = GEPAResult(
            candidates=candidates,
            parents=parents,
            val_aggregate_scores=val_scores,
            val_subscores=[{} for _ in candidates],
            per_val_instance_best_candidates=pareto_front,
            discovery_eval_counts=[0, 5, 5],
        )
        dot = result.candidate_tree_dot()
        assert "digraph G {" in dot
        assert "0 -> 1;" in dot

    def test_result_candidate_tree_html(self):
        """Test that GEPAResult.candidate_tree_html() works."""
        from gepa.core.result import GEPAResult

        candidates, parents, val_scores, pareto_front = _sample_data()
        result = GEPAResult(
            candidates=candidates,
            parents=parents,
            val_aggregate_scores=val_scores,
            val_subscores=[{} for _ in candidates],
            per_val_instance_best_candidates=pareto_front,
            discovery_eval_counts=[0, 5, 5],
        )
        html = result.candidate_tree_html()
        assert "<!DOCTYPE html>" in html
        assert "math tutor" in html
