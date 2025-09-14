# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest
from typing import List, Dict, Any

from gepa.adapters.generic_rag_adapter.evaluation_metrics import RAGEvaluationMetrics


class TestRAGEvaluationMetrics:
    """Test suite for RAG evaluation metrics."""
    
    @pytest.fixture
    def metrics(self):
        """Create a RAGEvaluationMetrics instance for testing."""
        return RAGEvaluationMetrics()
    
    def test_initialization(self, metrics):
        """Test RAGEvaluationMetrics initialization."""
        assert metrics is not None
        assert isinstance(metrics, RAGEvaluationMetrics)
    
    def test_evaluate_retrieval_perfect(self, metrics):
        """Test retrieval evaluation with perfect retrieval."""
        retrieved_docs = [
            {"metadata": {"doc_id": "doc1"}},
            {"metadata": {"doc_id": "doc2"}},
            {"metadata": {"doc_id": "doc3"}}
        ]
        relevant_docs = ["doc1", "doc2", "doc3"]
        
        result = metrics.evaluate_retrieval(retrieved_docs, relevant_docs)
        assert result["retrieval_precision"] == 1.0
        assert result["retrieval_recall"] == 1.0
        assert result["retrieval_f1"] == 1.0
        assert result["retrieval_mrr"] == 1.0
    
    def test_evaluate_retrieval_partial(self, metrics):
        """Test retrieval evaluation with partial relevance."""
        retrieved_docs = [
            {"metadata": {"doc_id": "doc1"}},
            {"metadata": {"doc_id": "doc2"}},
            {"metadata": {"doc_id": "doc3"}},
            {"metadata": {"doc_id": "doc4"}}
        ]
        relevant_docs = ["doc1", "doc3"]
        
        result = metrics.evaluate_retrieval(retrieved_docs, relevant_docs)
        assert result["retrieval_precision"] == 0.5  # 2 relevant out of 4 retrieved
        assert result["retrieval_recall"] == 1.0     # 2 retrieved out of 2 relevant
        assert result["retrieval_mrr"] == 1.0        # First doc is relevant
    
    def test_evaluate_retrieval_no_relevant(self, metrics):
        """Test retrieval evaluation with no relevant documents."""
        retrieved_docs = [
            {"metadata": {"doc_id": "doc1"}},
            {"metadata": {"doc_id": "doc2"}}
        ]
        relevant_docs = ["doc3", "doc4"]
        
        result = metrics.evaluate_retrieval(retrieved_docs, relevant_docs)
        assert result["retrieval_precision"] == 0.0
        assert result["retrieval_recall"] == 0.0
        assert result["retrieval_f1"] == 0.0
        assert result["retrieval_mrr"] == 0.0
    
    def test_evaluate_retrieval_empty_retrieved(self, metrics):
        """Test retrieval evaluation with empty retrieved list."""
        retrieved_docs = []
        relevant_docs = ["doc1", "doc2"]
        
        result = metrics.evaluate_retrieval(retrieved_docs, relevant_docs)
        assert result["retrieval_precision"] == 0.0
        assert result["retrieval_recall"] == 0.0
        assert result["retrieval_f1"] == 0.0
        assert result["retrieval_mrr"] == 0.0
    
    def test_evaluate_retrieval_empty_relevant(self, metrics):
        """Test retrieval evaluation with empty relevant list."""
        retrieved_docs = [
            {"metadata": {"doc_id": "doc1"}},
            {"metadata": {"doc_id": "doc2"}}
        ]
        relevant_docs = []
        
        result = metrics.evaluate_retrieval(retrieved_docs, relevant_docs)
        assert result["retrieval_precision"] == 0.0
        assert result["retrieval_recall"] == 0.0
        assert result["retrieval_f1"] == 0.0
        assert result["retrieval_mrr"] == 0.0
    
    def test_evaluate_retrieval_id_field_variation(self, metrics):
        """Test retrieval evaluation with different ID field names."""
        # Test with 'id' field in metadata
        retrieved_docs = [
            {"metadata": {"id": "doc1"}},
            {"metadata": {"id": "doc2"}}
        ]
        relevant_docs = ["doc1"]
        
        result = metrics.evaluate_retrieval(retrieved_docs, relevant_docs)
        assert result["retrieval_precision"] == 0.5
        assert result["retrieval_recall"] == 1.0
        assert result["retrieval_mrr"] == 1.0
    
    def test_evaluate_generation_perfect_match(self, metrics):
        """Test generation evaluation with perfect exact match."""
        generated_answer = "Machine learning is a subset of AI."
        ground_truth = "Machine learning is a subset of AI."
        context = "Machine learning is a subset of artificial intelligence."
        
        result = metrics.evaluate_generation(generated_answer, ground_truth, context)
        
        assert result["exact_match"] == 1.0
        assert result["token_f1"] == 1.0
        assert "answer_relevance" in result
        assert "faithfulness" in result
        assert "answer_confidence" in result
    
    def test_evaluate_generation_partial_match(self, metrics):
        """Test generation evaluation with partial match."""
        generated_answer = "Machine learning uses algorithms to learn patterns."
        ground_truth = "Machine learning is a subset of AI that employs algorithms."
        context = "Machine learning algorithms learn patterns from data."
        
        result = metrics.evaluate_generation(generated_answer, ground_truth, context)
        
        assert result["exact_match"] == 0.0  # Not exact match
        assert 0 < result["token_f1"] < 1     # Some token overlap
        assert 0 <= result["answer_relevance"] <= 1
        assert 0 <= result["faithfulness"] <= 1
        assert 0 <= result["answer_confidence"] <= 1
    
    def test_evaluate_generation_empty_answer(self, metrics):
        """Test generation evaluation with empty answer."""
        generated_answer = ""
        ground_truth = "Machine learning is AI."
        context = "Context about machine learning."
        
        result = metrics.evaluate_generation(generated_answer, ground_truth, context)
        
        assert result["exact_match"] == 0.0
        assert result["token_f1"] == 0.0
        assert result["answer_relevance"] == 0.0
        assert result["faithfulness"] == 1.0  # Empty answer is technically faithful
    
    def test_combined_rag_score_balanced(self, metrics):
        """Test combined RAG score calculation."""
        retrieval_metrics = {
            "retrieval_precision": 0.8,
            "retrieval_recall": 0.6,
            "retrieval_f1": 0.69,  # ~2 * 0.8 * 0.6 / (0.8 + 0.6)
            "retrieval_mrr": 0.5
        }
        
        generation_metrics = {
            "exact_match": 0.0,
            "token_f1": 0.8,
            "bleu_score": 0.7,
            "answer_relevance": 0.9,
            "faithfulness": 0.85,
            "answer_confidence": 0.85
        }
        
        score = metrics.combined_rag_score(
            retrieval_metrics, 
            generation_metrics,
            retrieval_weight=0.3,
            generation_weight=0.7
        )
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_combined_rag_score_default_weights(self, metrics):
        """Test combined RAG score with default weights."""
        retrieval_metrics = {"retrieval_f1": 0.8}
        generation_metrics = {
            "token_f1": 0.7,
            "answer_relevance": 0.6,
            "faithfulness": 0.9
        }
        
        score = metrics.combined_rag_score(retrieval_metrics, generation_metrics)
        
        # Should use default weights (0.3 retrieval, 0.7 generation)
        expected_generation_score = (0.7 * 0.4) + (0.6 * 0.3) + (0.9 * 0.3)
        expected_score = 0.3 * 0.8 + 0.7 * expected_generation_score
        
        assert abs(score - expected_score) < 1e-6
    
    def test_exact_match_case_insensitive(self, metrics):
        """Test that exact match is case insensitive."""
        assert metrics._exact_match("Hello World", "hello world") == True
        assert metrics._exact_match("HELLO WORLD", "hello world") == True
        assert metrics._exact_match("Hello World", "Hello Universe") == False
    
    def test_exact_match_whitespace_handling(self, metrics):
        """Test that exact match handles whitespace correctly."""
        assert metrics._exact_match("  hello world  ", "hello world") == True
        assert metrics._exact_match("hello\tworld", "hello world") == False  # Different whitespace
    
    def test_token_f1_perfect_match(self, metrics):
        """Test token F1 with perfect match."""
        f1 = metrics._token_f1("machine learning is great", "machine learning is great")
        assert f1 == 1.0
    
    def test_token_f1_partial_overlap(self, metrics):
        """Test token F1 with partial overlap."""
        f1 = metrics._token_f1("machine learning algorithms", "machine learning techniques")
        
        # Common tokens: "machine", "learning" (2 out of 3 each)
        # Precision: 2/3, Recall: 2/3, F1: 2/3
        expected_f1 = 2/3
        assert abs(f1 - expected_f1) < 1e-6
    
    def test_token_f1_no_overlap(self, metrics):
        """Test token F1 with no token overlap."""
        f1 = metrics._token_f1("python programming", "java development")
        assert f1 == 0.0
    
    def test_token_f1_empty_strings(self, metrics):
        """Test token F1 with empty strings."""
        # Both empty
        assert metrics._token_f1("", "") == 1.0
        
        # One empty
        assert metrics._token_f1("hello", "") == 0.0
        assert metrics._token_f1("", "hello") == 0.0
    
    def test_simple_bleu_score(self, metrics):
        """Test simple BLEU score calculation."""
        # Perfect match
        bleu = metrics._simple_bleu("machine learning is amazing", "machine learning is amazing")
        assert bleu == 1.0
        
        # Partial match
        bleu = metrics._simple_bleu("machine learning algorithms", "machine learning techniques")
        assert 0 <= bleu <= 1
        
        # No match
        bleu = metrics._simple_bleu("completely different", "totally unrelated")
        assert bleu == 0.0
    
    def test_answer_relevance_calculation(self, metrics):
        """Test answer relevance calculation."""
        answer = "machine learning algorithms are powerful"
        context = "machine learning uses algorithms to solve problems"
        
        relevance = metrics._answer_relevance(answer, context)
        
        # Should have some overlap between answer and context
        assert 0 < relevance <= 1
        
        # Test with no overlap
        relevance_no_overlap = metrics._answer_relevance("python programming", "java development")
        assert relevance_no_overlap == 0.0
    
    def test_faithfulness_score_calculation(self, metrics):
        """Test faithfulness score calculation."""
        answer = "machine learning uses algorithms"
        context = "machine learning algorithms are used to learn patterns"
        
        faithfulness = metrics._faithfulness_score(answer, context)
        assert 0 <= faithfulness <= 1
        
        # Empty answer should be faithful
        empty_faithfulness = metrics._faithfulness_score("", context)
        assert empty_faithfulness == 1.0
    
    def test_extract_phrases_functionality(self, metrics):
        """Test phrase extraction functionality."""
        text = "machine learning algorithms are very powerful tools"
        phrases = metrics._extract_phrases(text)
        
        assert isinstance(phrases, set)
        assert len(phrases) > 0
        
        # Should contain some meaningful phrases
        # Exact content depends on implementation, but should not be empty for meaningful text
    
    def test_normalize_text_functionality(self, metrics):
        """Test text normalization."""
        text = "  Hello, World!  This is a TEST.  "
        normalized = metrics._normalize_text(text)
        
        # Should be lowercase, no punctuation, normalized whitespace
        expected = "hello world this is a test"
        assert normalized.strip() == expected
    
    def test_normalize_text_punctuation_removal(self, metrics):
        """Test that punctuation is properly removed."""
        text = "Hello! How are you? I'm fine, thanks."
        normalized = metrics._normalize_text(text)
        
        # Should have no punctuation
        assert "!" not in normalized
        assert "?" not in normalized
        assert "," not in normalized
        assert "." not in normalized
        assert "'" not in normalized


class TestRAGEvaluationMetricsIntegration:
    """Integration tests for RAGEvaluationMetrics with realistic scenarios."""
    
    @pytest.fixture
    def metrics(self):
        return RAGEvaluationMetrics()
    
    def test_typical_rag_evaluation_scenario(self, metrics):
        """Test a typical RAG evaluation scenario with all metrics."""
        # Retrieved documents
        retrieved_docs = [
            {"metadata": {"doc_id": "doc1"}},
            {"metadata": {"doc_id": "doc2"}},
            {"metadata": {"doc_id": "doc3"}},
            {"metadata": {"doc_id": "doc4"}},
            {"metadata": {"doc_id": "doc5"}}
        ]
        relevant_docs = ["doc1", "doc3", "doc6"]  # doc6 not retrieved
        
        # Generated answer vs ground truth
        predicted_answer = "Machine learning is a subset of artificial intelligence that uses algorithms."
        ground_truth_answer = "Machine learning is a subset of AI that employs algorithms to learn patterns."
        context = "Machine learning, a subset of artificial intelligence, employs various algorithms."
        
        # Compute all retrieval metrics
        retrieval_result = metrics.evaluate_retrieval(retrieved_docs, relevant_docs)
        
        # Compute all generation metrics
        generation_result = metrics.evaluate_generation(predicted_answer, ground_truth_answer, context)
        
        # Compute combined score
        combined_score = metrics.combined_rag_score(retrieval_result, generation_result)
        
        # Verify reasonable values
        assert 0 <= retrieval_result["retrieval_precision"] <= 1
        assert 0 <= retrieval_result["retrieval_recall"] <= 1
        assert 0 <= retrieval_result["retrieval_f1"] <= 1
        assert 0 <= retrieval_result["retrieval_mrr"] <= 1
        
        assert 0 <= generation_result["exact_match"] <= 1
        assert 0 <= generation_result["token_f1"] <= 1
        assert 0 <= generation_result["bleu_score"] <= 1
        assert 0 <= generation_result["answer_relevance"] <= 1
        assert 0 <= generation_result["faithfulness"] <= 1
        
        assert 0 <= combined_score <= 1
        
        # Specific checks for this scenario
        assert retrieval_result["retrieval_precision"] == 0.4  # 2/5 retrieved docs are relevant
        assert retrieval_result["retrieval_recall"] == 2/3     # 2/3 relevant docs were retrieved
        assert retrieval_result["retrieval_mrr"] == 1.0        # First retrieved doc is relevant