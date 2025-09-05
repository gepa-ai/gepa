# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from gepa.adapters.generic_rag_adapter.rag_pipeline import RAGPipeline
from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface


class MockVectorStore(VectorStoreInterface):
    """Mock vector store for testing RAGPipeline."""
    
    def __init__(self):
        self.documents = [
            {
                "id": "doc1",
                "content": "Machine learning is a subset of artificial intelligence.",
                "metadata": {"category": "AI", "score": 0.95}
            },
            {
                "id": "doc2", 
                "content": "Python is a popular programming language for data science.",
                "metadata": {"category": "programming", "score": 0.89}
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
            "name": "mock_collection",
            "document_count": len(self.documents),
            "vector_store_type": "mock"
        }


class TestRAGPipeline:
    """Test suite for RAGPipeline class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing."""
        return MockVectorStore()
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client for testing."""
        def mock_callable_client(messages):
            """Mock callable LLM client that returns string responses."""
            return "This is a test response from the LLM."
        
        return mock_callable_client
    
    @pytest.fixture
    def mock_embedding_function(self):
        """Create a mock embedding function for testing."""
        def embedding_fn(text: str) -> List[float]:
            return [0.1] * 384
        return embedding_fn
    
    @pytest.fixture
    def rag_pipeline(self, mock_vector_store, mock_llm_client, mock_embedding_function):
        """Create a RAGPipeline instance for testing."""
        return RAGPipeline(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            embedding_model="text-embedding-3-small",
            embedding_function=mock_embedding_function
        )
    
    def test_initialization(self, mock_vector_store, mock_llm_client):
        """Test RAGPipeline initialization."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client
        )
        
        assert pipeline.vector_store == mock_vector_store
        assert pipeline.llm_client == mock_llm_client
        assert pipeline.embedding_model == "text-embedding-3-small"
        assert pipeline.embedding_function is not None
    
    def test_initialization_with_embedding_function(self, mock_vector_store, mock_llm_client, mock_embedding_function):
        """Test RAGPipeline initialization with custom embedding function."""
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            embedding_function=mock_embedding_function
        )
        
        assert pipeline.embedding_function == mock_embedding_function
    
    def test_execute_rag_basic(self, rag_pipeline):
        """Test basic RAG execution."""
        query = "What is machine learning?"
        prompts = {
            "answer_generation": "Answer: {query} using context: {context}"
        }
        config = {
            "retrieval_strategy": "similarity",
            "top_k": 2
        }
        
        result = rag_pipeline.execute_rag(query, prompts, config)
        
        # Check result structure
        assert isinstance(result, dict)
        assert "original_query" in result
        assert "reformulated_query" in result
        assert "retrieved_docs" in result
        assert "synthesized_context" in result
        assert "generated_answer" in result
        assert "metadata" in result
        
        # Check that answer was generated
        assert isinstance(result["generated_answer"], str)
        assert result["generated_answer"]  # Should not be empty
        
        # LLM client is now a simple callable, can't easily check if called
        # But we can verify the result structure is correct
    
    def test_execute_rag_with_query_reformulation(self, rag_pipeline):
        """Test RAG execution with query reformulation."""
        query = "What is ML?"
        prompts = {
            "query_reformulation": "Reformulate this query: {query}",
            "answer_generation": "Answer: {query}"
        }
        config = {
            "retrieval_strategy": "similarity",
            "top_k": 3
        }
        
        result = rag_pipeline.execute_rag(query, prompts, config)
        
        assert result["original_query"] == query
        assert isinstance(result["reformulated_query"], str)
        # Can't check call count with simple callable, but verify we got responses
    
    def test_execute_rag_with_reranking(self, rag_pipeline):
        """Test RAG execution with document reranking."""
        query = "machine learning"
        prompts = {
            "reranking_criteria": "Rank documents by relevance to: {query}",
            "answer_generation": "Answer using context: {context}"
        }
        config = {
            "retrieval_strategy": "similarity",
            "top_k": 2
        }
        
        result = rag_pipeline.execute_rag(query, prompts, config)
        
        assert isinstance(result["retrieved_docs"], list)
        assert len(result["retrieved_docs"]) <= config["top_k"]
    
    def test_execute_rag_minimal_config(self, rag_pipeline):
        """Test RAG execution with minimal configuration."""
        query = "What is AI?"
        prompts = {
            "answer_generation": "Answer: {query}"
        }
        config = {
            "retrieval_strategy": "similarity",
            "top_k": 1
        }
        
        result = rag_pipeline.execute_rag(query, prompts, config)
        
        assert isinstance(result, dict)
        assert "generated_answer" in result
        assert result["generated_answer"]
    
    def test_execute_rag_different_strategies(self, rag_pipeline):
        """Test RAG execution with different retrieval strategies."""
        query = "test query"
        prompts = {"answer_generation": "Answer: {query}"}
        
        strategies = ["similarity", "vector", "hybrid"]
        
        for strategy in strategies:
            config = {
                "retrieval_strategy": strategy,
                "top_k": 2
            }
            
            result = rag_pipeline.execute_rag(query, prompts, config)
            
            assert isinstance(result, dict)
            assert "generated_answer" in result
            assert isinstance(result["retrieved_docs"], list)
    
    def test_execute_rag_metadata(self, rag_pipeline):
        """Test that RAG execution includes proper metadata."""
        query = "test"
        prompts = {"answer_generation": "Answer: {query}"}
        config = {"retrieval_strategy": "similarity", "top_k": 1}
        
        result = rag_pipeline.execute_rag(query, prompts, config)
        
        metadata = result["metadata"]
        assert "retrieval_count" in metadata
        assert "total_tokens" in metadata
        assert "vector_store_type" in metadata
        assert isinstance(metadata["retrieval_count"], int)
        assert isinstance(metadata["total_tokens"], int)
    
    def test_private_method_reformulate_query(self, rag_pipeline):
        """Test the private _reformulate_query method."""
        query = "What is ML?"
        prompt = "Reformulate this query for better search: {query}"
        
        reformulated = rag_pipeline._reformulate_query(query, prompt)
        
        assert isinstance(reformulated, str)
        assert reformulated  # Should not be empty
    
    def test_private_method_retrieve_documents(self, rag_pipeline):
        """Test the private _retrieve_documents method."""
        query = "machine learning"
        config = {
            "retrieval_strategy": "similarity",
            "top_k": 2
        }
        
        docs = rag_pipeline._retrieve_documents(query, config)
        
        assert isinstance(docs, list)
        assert len(docs) <= config["top_k"]
        assert all(isinstance(doc, dict) for doc in docs)
    
    def test_private_method_generate_answer(self, rag_pipeline):
        """Test the private _generate_answer method."""
        query = "What is AI?"
        context = "Artificial intelligence is machine intelligence."
        prompt = "Answer the question: {query} using context: {context}"
        
        answer = rag_pipeline._generate_answer(query, context, prompt)
        
        assert isinstance(answer, str)
        assert answer  # Should not be empty
    
    def test_error_handling_llm_failure(self, rag_pipeline):
        """Test error handling when LLM calls fail."""
        # Replace the LLM client with one that raises an exception
        def failing_client(messages):
            raise Exception("LLM Error")
        rag_pipeline.llm_client = failing_client
        
        query = "test query"
        prompts = {"answer_generation": "Answer: {query}"}
        config = {"retrieval_strategy": "similarity", "top_k": 1}
        
        # The pipeline should handle LLM errors gracefully, not raise exceptions
        result = rag_pipeline.execute_rag(query, prompts, config)
        
        # Should still return a valid result structure
        assert isinstance(result, dict)
        assert "generated_answer" in result
    
    def test_empty_prompts_handling(self, rag_pipeline):
        """Test handling of empty prompts."""
        query = "test query"
        prompts = {}  # Empty prompts
        config = {"retrieval_strategy": "similarity", "top_k": 1}
        
        result = rag_pipeline.execute_rag(query, prompts, config)
        
        # Should still work with minimal functionality
        assert isinstance(result, dict)
        assert "original_query" in result
        assert result["original_query"] == query