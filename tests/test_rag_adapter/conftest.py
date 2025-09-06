# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Pytest configuration and shared fixtures for RAG adapter tests.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

from gepa.adapters.generic_rag_adapter.generic_rag_adapter import RAGDataInst
from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface


class TestVectorStore(VectorStoreInterface):
    """Test vector store implementation for pytest fixtures."""
    
    def __init__(self, collection_name: str = "test_collection"):
        self.collection_name = collection_name
        self.documents = [
            {
                "id": "doc1",
                "content": "Machine learning is a subset of artificial intelligence that uses statistical techniques.",
                "metadata": {"category": "AI", "difficulty": "intermediate", "relevance": "high"}
            },
            {
                "id": "doc2",
                "content": "Python is a high-level programming language widely used for data science and machine learning.",
                "metadata": {"category": "programming", "difficulty": "beginner", "relevance": "medium"}
            },
            {
                "id": "doc3",
                "content": "Neural networks are computing systems inspired by biological neural networks.",
                "metadata": {"category": "AI", "difficulty": "advanced", "relevance": "high"}
            },
            {
                "id": "doc4",
                "content": "Data preprocessing is a crucial step in machine learning pipelines.",
                "metadata": {"category": "data-science", "difficulty": "intermediate", "relevance": "medium"}
            },
            {
                "id": "doc5",
                "content": "Deep learning is a subset of machine learning based on artificial neural networks.",
                "metadata": {"category": "AI", "difficulty": "advanced", "relevance": "high"}
            }
        ]
    
    def similarity_search(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Mock similarity search with basic keyword matching."""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            # Simple keyword matching for realistic testing
            content_lower = doc["content"].lower()
            if any(word in content_lower for word in query_lower.split()):
                if not filters or self._matches_filters(doc["metadata"], filters):
                    results.append(doc.copy())
        
        # Sort by simple relevance score (number of matching words)
        def relevance_score(doc):
            matches = sum(1 for word in query_lower.split() 
                         if word in doc["content"].lower())
            return matches
        
        results.sort(key=relevance_score, reverse=True)
        return results[:k]
    
    def vector_search(self, query_vector: List[float], k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Mock vector search returning documents based on filters."""
        results = []
        for doc in self.documents:
            if not filters or self._matches_filters(doc["metadata"], filters):
                results.append(doc.copy())
        return results[:k]
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Mock hybrid search combining similarity and keyword search."""
        # For testing, just return similarity search results
        return self.similarity_search(query, k)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Return test collection information."""
        return {
            "name": self.collection_name,
            "document_count": len(self.documents),
            "vector_store_type": "test",
            "embedding_dimension": 384,
            "created_at": "2025-01-01T00:00:00Z"
        }
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches the given filters."""
        for key, expected_value in filters.items():
            if key not in metadata:
                return False
            
            actual_value = metadata[key]
            
            # Handle different filter types
            if isinstance(expected_value, dict):
                # Range filters like {"$gt": 0.5}
                for op, val in expected_value.items():
                    if op == "$gt" and actual_value <= val:
                        return False
                    elif op == "$lt" and actual_value >= val:
                        return False
                    elif op == "$gte" and actual_value < val:
                        return False
                    elif op == "$lte" and actual_value > val:
                        return False
                    elif op == "$ne" and actual_value == val:
                        return False
                    elif op == "$in" and actual_value not in val:
                        return False
            else:
                # Exact match
                if actual_value != expected_value:
                    return False
        
        return True


@pytest.fixture
def test_vector_store():
    """Provide a test vector store instance."""
    return TestVectorStore("pytest_test_collection")


@pytest.fixture
def sample_rag_training_data():
    """Provide sample RAG training data for tests."""
    return [
        RAGDataInst(
            query="What is machine learning?",
            ground_truth_answer="Machine learning is a subset of artificial intelligence that uses statistical techniques to enable computers to learn and make decisions from data without being explicitly programmed for every task.",
            relevant_doc_ids=["doc1", "doc5"],
            metadata={"category": "AI", "difficulty": "beginner", "expected_docs": 2}
        ),
        RAGDataInst(
            query="Which programming language is best for data science?",
            ground_truth_answer="Python is widely considered the best programming language for data science due to its extensive libraries, ease of use, and strong community support.",
            relevant_doc_ids=["doc2"],
            metadata={"category": "programming", "difficulty": "beginner", "expected_docs": 1}
        ),
        RAGDataInst(
            query="How do neural networks work?",
            ground_truth_answer="Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections and activation functions.",
            relevant_doc_ids=["doc3", "doc5"],
            metadata={"category": "AI", "difficulty": "advanced", "expected_docs": 2}
        ),
        RAGDataInst(
            query="What is data preprocessing in machine learning?",
            ground_truth_answer="Data preprocessing is the process of cleaning, transforming, and preparing raw data for machine learning algorithms. It includes tasks like handling missing values, normalization, and feature selection.",
            relevant_doc_ids=["doc4"],
            metadata={"category": "data-science", "difficulty": "intermediate", "expected_docs": 1}
        )
    ]


@pytest.fixture
def mock_llm_client():
    """Provide a mock LLM client for testing."""
    mock_client = Mock()
    
    # Mock successful completion response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "This is a test response from the mock LLM client."
    
    # Mock token usage
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_rag_config():
    """Provide sample RAG configuration for testing."""
    return {
        "retrieval_strategy": "similarity",
        "top_k": 3,
        "retrieval_weight": 0.4,
        "generation_weight": 0.6,
        "use_query_reformulation": False,
        "use_reranking": False,
        "filters": {},
        "hybrid_alpha": 0.5
    }


@pytest.fixture
def sample_prompts():
    """Provide sample prompt templates for testing."""
    return {
        "query_reformulation": """
        You are an expert at reformulating queries for better information retrieval.
        
        Original query: {query}
        
        Please reformulate this query to be more specific and likely to retrieve relevant documents:
        """.strip(),
        
        "context_synthesis": """
        You are an expert at synthesizing information from multiple documents.
        
        Query: {query}
        
        Documents:
        {documents}
        
        Please synthesize these documents into a coherent context that addresses the query:
        """.strip(),
        
        "answer_generation": """
        You are a helpful AI assistant providing accurate answers based on given context.
        
        Question: {query}
        
        Context:
        {context}
        
        Please provide a comprehensive and accurate answer to the question using the provided context:
        """.strip(),
        
        "reranking_criteria": """
        You are an expert at ranking documents by relevance to a specific query.
        
        Query: {query}
        
        Documents to rank:
        {documents}
        
        Please rank these documents from most relevant to least relevant for answering the query:
        """.strip()
    }


@pytest.fixture
def mock_embedding_function():
    """Provide a mock embedding function for testing."""
    def embedding_fn(text: str) -> List[float]:
        # Return a deterministic mock embedding based on text length
        # This makes tests predictable while still being somewhat realistic
        base_embedding = [0.1] * 384
        text_hash = hash(text) % 1000 / 1000.0  # Normalize to [0, 1]
        
        # Modify a few dimensions based on text hash for uniqueness
        for i in range(0, min(10, len(base_embedding))):
            base_embedding[i] = (base_embedding[i] + text_hash + i * 0.01) % 1.0
        
        return base_embedding
    
    return embedding_fn


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Provide a temporary directory for test data."""
    return tmp_path_factory.mktemp("rag_test_data")


@pytest.fixture
def mock_metrics_perfect():
    """Provide a mock evaluation metrics instance with perfect scores."""
    mock_metrics = Mock()
    
    # Perfect retrieval metrics
    mock_metrics.compute_retrieval_precision.return_value = 1.0
    mock_metrics.compute_retrieval_recall.return_value = 1.0
    mock_metrics.compute_retrieval_f1.return_value = 1.0
    mock_metrics.compute_mrr.return_value = 1.0
    
    # Perfect generation metrics
    mock_metrics.compute_exact_match.return_value = 1.0
    mock_metrics.compute_token_f1.return_value = 1.0
    mock_metrics.compute_bleu_score.return_value = 1.0
    
    return mock_metrics


@pytest.fixture
def mock_metrics_realistic():
    """Provide a mock evaluation metrics instance with realistic scores."""
    mock_metrics = Mock()
    
    # Realistic retrieval metrics
    mock_metrics.compute_retrieval_precision.return_value = 0.67
    mock_metrics.compute_retrieval_recall.return_value = 0.80
    mock_metrics.compute_retrieval_f1.return_value = 0.73
    mock_metrics.compute_mrr.return_value = 0.75
    
    # Realistic generation metrics
    mock_metrics.compute_exact_match.return_value = 0.0
    mock_metrics.compute_token_f1.return_value = 0.85
    mock_metrics.compute_bleu_score.return_value = 0.72
    
    return mock_metrics


# Utility functions for tests
def create_test_documents(count: int = 5) -> List[Dict[str, Any]]:
    """Create test documents for use in tests."""
    documents = []
    categories = ["AI", "programming", "data-science", "technology", "research"]
    difficulties = ["beginner", "intermediate", "advanced"]
    
    for i in range(count):
        doc = {
            "id": f"test_doc_{i+1}",
            "content": f"Test document {i+1} content about {categories[i % len(categories)]}.",
            "metadata": {
                "category": categories[i % len(categories)],
                "difficulty": difficulties[i % len(difficulties)],
                "test_id": i+1,
                "relevance": 0.5 + (i % 3) * 0.25  # 0.5, 0.75, 1.0
            }
        }
        documents.append(doc)
    
    return documents


def assert_valid_rag_result(result: Dict[str, Any]):
    """Assert that a RAG pipeline result has the expected structure."""
    assert isinstance(result, dict)
    assert "final_answer" in result
    assert "retrieved_documents" in result
    assert "execution_metadata" in result
    
    # Check answer
    assert isinstance(result["final_answer"], str)
    assert len(result["final_answer"]) > 0
    
    # Check documents
    assert isinstance(result["retrieved_documents"], list)
    for doc in result["retrieved_documents"]:
        assert isinstance(doc, dict)
        assert "id" in doc or "content" in doc
    
    # Check metadata
    metadata = result["execution_metadata"]
    assert isinstance(metadata, dict)
    assert "total_tokens" in metadata or "retrieval_time" in metadata


def assert_valid_evaluation_score(score: float):
    """Assert that an evaluation score is valid."""
    assert isinstance(score, (int, float))
    assert 0.0 <= score <= 1.0


# Test markers for different types of tests
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.requires_llm = pytest.mark.requires_llm