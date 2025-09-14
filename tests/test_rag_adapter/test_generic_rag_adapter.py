# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from gepa.adapters.generic_rag_adapter.generic_rag_adapter import GenericRAGAdapter, RAGDataInst
from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface
from gepa.core.adapter import EvaluationBatch


class MockVectorStore(VectorStoreInterface):
    """Mock vector store for testing GenericRAGAdapter."""
    
    def __init__(self):
        self.documents = [
            {
                "id": "doc1",
                "content": "Machine learning is a subset of artificial intelligence.",
                "metadata": {"doc_id": "doc1", "category": "AI"}
            },
            {
                "id": "doc2",
                "content": "Python is a popular programming language for data science.",
                "metadata": {"doc_id": "doc2", "category": "programming"}
            }
        ]
    
    def similarity_search(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        return self.documents[:k]
    
    def vector_search(self, query_vector: List[float], k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        return self.documents[:k]
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        return self.documents[:k]
    
    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "name": "test_collection",
            "document_count": len(self.documents),
            "vector_store_type": "mock"
        }


class TestGenericRAGAdapter:
    """Test suite for GenericRAGAdapter class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing."""
        return MockVectorStore()
    
    @pytest.fixture
    def sample_rag_config(self):
        """Create sample RAG configuration for testing."""
        return {
            "retrieval_strategy": "similarity",
            "top_k": 3,
            "retrieval_weight": 0.3,
            "generation_weight": 0.7
        }
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for testing."""
        return [
            RAGDataInst(
                query="What is machine learning?",
                ground_truth_answer="Machine learning is a subset of AI.",
                relevant_doc_ids=["doc1"],
                metadata={"difficulty": "beginner"}
            ),
            RAGDataInst(
                query="What programming language is used for ML?",
                ground_truth_answer="Python is commonly used for ML.",
                relevant_doc_ids=["doc2"],
                metadata={"difficulty": "beginner"}
            )
        ]
    
    @patch('litellm.completion')
    def test_initialization(self, mock_litellm, mock_vector_store, sample_rag_config):
        """Test GenericRAGAdapter initialization."""
        # Mock the litellm response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "test response"
        mock_litellm.return_value = mock_response
        
        adapter = GenericRAGAdapter(
            vector_store=mock_vector_store,
            llm_model="gpt-4o-mini",
            rag_config=sample_rag_config
        )
        
        assert adapter.vector_store == mock_vector_store
        assert adapter.config == sample_rag_config
        assert adapter.rag_pipeline is not None
        assert adapter.evaluator is not None
    
    @patch('litellm.completion')
    def test_initialization_with_defaults(self, mock_litellm, mock_vector_store):
        """Test GenericRAGAdapter initialization with default configuration."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "test response"
        mock_litellm.return_value = mock_response
        
        adapter = GenericRAGAdapter(
            vector_store=mock_vector_store,
            llm_model="gpt-4o-mini"
        )
        
        # Check default configuration values
        assert adapter.config["retrieval_strategy"] == "similarity"
        assert adapter.config["top_k"] == 5
        assert adapter.config["retrieval_weight"] == 0.3
        assert adapter.config["generation_weight"] == 0.7
    
    @patch('litellm.completion')
    def test_evaluate_single_example(self, mock_litellm, mock_vector_store, sample_rag_config, sample_training_data):
        """Test evaluating a single example."""
        # Mock the litellm response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Machine learning is a subset of AI."
        mock_litellm.return_value = mock_response
        
        adapter = GenericRAGAdapter(
            vector_store=mock_vector_store,
            llm_model="gpt-4o-mini",
            rag_config=sample_rag_config
        )
        
        # Test evaluation with a single example
        candidate = {"answer_generation": "Answer: {query} using {context}"}
        example = sample_training_data[0]
        
        # For single example, call evaluate with a list containing one item
        result = adapter.evaluate([example], candidate)
        
        assert isinstance(result, EvaluationBatch)
        assert len(result.scores) == 1
        assert len(result.outputs) == 1
        assert isinstance(result.scores[0], float)
        assert 0 <= result.scores[0] <= 1
    
    @patch('litellm.completion')
    def test_evaluate_batch(self, mock_litellm, mock_vector_store, sample_rag_config, sample_training_data):
        """Test evaluating a batch of examples."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test answer"
        mock_litellm.return_value = mock_response
        
        adapter = GenericRAGAdapter(
            vector_store=mock_vector_store,
            llm_model="gpt-4o-mini",
            rag_config=sample_rag_config
        )
        
        # Test batch evaluation
        candidate = {"answer_generation": "Answer: {query}"}
        
        result = adapter.evaluate(sample_training_data, candidate)
        
        assert isinstance(result, EvaluationBatch)
        assert len(result.scores) == len(sample_training_data)
        assert len(result.outputs) == len(sample_training_data)
        assert all(isinstance(score, float) for score in result.scores)
        assert all(0 <= score <= 1 for score in result.scores)
    
    @patch('litellm.completion')
    def test_evaluate_with_trajectory_capture(self, mock_litellm, mock_vector_store, sample_rag_config, sample_training_data):
        """Test evaluation with trajectory capture enabled."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test answer"
        mock_litellm.return_value = mock_response
        
        adapter = GenericRAGAdapter(
            vector_store=mock_vector_store,
            llm_model="gpt-4o-mini",
            rag_config=sample_rag_config
        )
        
        candidate = {"answer_generation": "Answer: {query}"}
        
        # Test with capture_traces enabled
        result = adapter.evaluate(sample_training_data, candidate, capture_traces=True)
        
        assert isinstance(result, EvaluationBatch)
        assert result.trajectories is not None
        assert len(result.trajectories) == len(sample_training_data)
        
        # Check trajectory structure
        trajectory = result.trajectories[0]
        assert "original_query" in trajectory
        assert "reformulated_query" in trajectory
        assert "retrieved_docs" in trajectory
        assert "synthesized_context" in trajectory
        assert "generated_answer" in trajectory
        assert "execution_metadata" in trajectory
    
    @patch('litellm.completion')
    def test_score_computation_with_weights(self, mock_litellm, mock_vector_store):
        """Test that score computation properly applies retrieval and generation weights."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Machine learning is AI."
        mock_litellm.return_value = mock_response
        
        config = {
            "retrieval_strategy": "similarity",
            "top_k": 2,
            "retrieval_weight": 0.4,
            "generation_weight": 0.6
        }
        
        adapter = GenericRAGAdapter(
            vector_store=mock_vector_store,
            llm_model="gpt-4o-mini",
            rag_config=config
        )
        
        example = RAGDataInst(
            query="What is ML?",
            ground_truth_answer="Machine learning is AI.",
            relevant_doc_ids=["doc1"],
            metadata={}
        )
        candidate = {"answer_generation": "Answer: {query}"}
        
        result = adapter.evaluate([example], candidate)
        
        # Should return a valid score
        assert len(result.scores) == 1
        assert isinstance(result.scores[0], float)
        assert 0 <= result.scores[0] <= 1
    
    @patch('litellm.completion')
    def test_different_retrieval_strategies(self, mock_litellm, mock_vector_store):
        """Test adapter with different retrieval strategies."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_litellm.return_value = mock_response
        
        strategies = ["similarity", "vector", "hybrid"]
        
        for strategy in strategies:
            config = {
                "retrieval_strategy": strategy,
                "top_k": 2
            }
            
            adapter = GenericRAGAdapter(
                vector_store=mock_vector_store,
                llm_model="gpt-4o-mini",
                rag_config=config
            )
            
            example = RAGDataInst(
                query="Test query",
                ground_truth_answer="Test answer",
                relevant_doc_ids=["doc1"],
                metadata={}
            )
            candidate = {"answer_generation": "Answer: {query}"}
            
            result = adapter.evaluate([example], candidate)
            
            assert isinstance(result, EvaluationBatch)
            assert len(result.scores) == 1
            assert isinstance(result.scores[0], float)
    
    @patch('litellm.completion')
    def test_make_reflective_dataset(self, mock_litellm, mock_vector_store, sample_rag_config, sample_training_data):
        """Test reflective dataset generation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test answer"
        mock_litellm.return_value = mock_response
        
        adapter = GenericRAGAdapter(
            vector_store=mock_vector_store,
            llm_model="gpt-4o-mini",
            rag_config=sample_rag_config
        )
        
        candidate = {"answer_generation": "Answer: {query}"}
        
        # First evaluate with trajectory capture
        eval_batch = adapter.evaluate(sample_training_data, candidate, capture_traces=True)
        
        # Then create reflective dataset
        reflective_data = adapter.make_reflective_dataset(
            candidate=candidate,
            eval_batch=eval_batch,
            components_to_update=["answer_generation"]
        )
        
        assert isinstance(reflective_data, dict)
        assert "answer_generation" in reflective_data
        assert isinstance(reflective_data["answer_generation"], list)
        
        # Check structure of reflective examples
        if reflective_data["answer_generation"]:
            example = reflective_data["answer_generation"][0]
            assert isinstance(example, dict)
    
    @patch('litellm.completion')
    def test_error_handling(self, mock_litellm, mock_vector_store, sample_rag_config):
        """Test error handling during evaluation."""
        # Make litellm raise an exception
        mock_litellm.side_effect = Exception("LLM Error")
        
        adapter = GenericRAGAdapter(
            vector_store=mock_vector_store,
            llm_model="gpt-4o-mini",
            rag_config=sample_rag_config
        )
        
        example = RAGDataInst(
            query="Test query",
            ground_truth_answer="Test answer",
            relevant_doc_ids=["doc1"],
            metadata={}
        )
        candidate = {"answer_generation": "Answer: {query}"}
        
        # Should handle errors gracefully and return valid EvaluationBatch
        result = adapter.evaluate([example], candidate)
        
        assert isinstance(result, EvaluationBatch)
        assert len(result.scores) == 1
        # The adapter may not use the exact failure score if partial success occurs
        assert isinstance(result.scores[0], float)
        assert 0 <= result.scores[0] <= 1
    
    @patch('litellm.completion')
    def test_prompt_template_variations(self, mock_litellm, mock_vector_store, sample_rag_config):
        """Test adapter with various prompt template formats."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Valid response"
        mock_litellm.return_value = mock_response
        
        adapter = GenericRAGAdapter(
            vector_store=mock_vector_store,
            llm_model="gpt-4o-mini",
            rag_config=sample_rag_config
        )
        
        example = RAGDataInst(
            query="What is AI?",
            ground_truth_answer="AI is artificial intelligence.",
            relevant_doc_ids=["doc1"],
            metadata={}
        )
        
        test_candidates = [
            # Minimal prompt
            {"answer_generation": "Answer: {query}"},
            
            # Complex prompt with multiple components
            {
                "query_reformulation": "Improve query: {query}",
                "context_synthesis": "Synthesize: {documents} for {query}",
                "answer_generation": "Answer {query} using context: {context}",
                "reranking_criteria": "Rank documents for {query}"
            },
            
            # Prompt without placeholders
            {"answer_generation": "Provide a comprehensive answer."}
        ]
        
        for candidate in test_candidates:
            result = adapter.evaluate([example], candidate)
            assert isinstance(result, EvaluationBatch)
            assert len(result.scores) == 1
            # Should execute without errors