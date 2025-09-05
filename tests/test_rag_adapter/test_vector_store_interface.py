# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest
from abc import ABC
from typing import Any, Dict, List

from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface


class MockVectorStore(VectorStoreInterface):
    """Mock implementation of VectorStoreInterface for testing."""
    
    def __init__(self, collection_name: str = "test_collection"):
        self.collection_name = collection_name
        self.documents = [
            {
                "id": "doc1",
                "content": "Machine learning is a subset of artificial intelligence.",
                "metadata": {"category": "AI", "difficulty": "beginner"}
            },
            {
                "id": "doc2", 
                "content": "Neural networks are inspired by biological neural networks.",
                "metadata": {"category": "AI", "difficulty": "intermediate"}
            },
            {
                "id": "doc3",
                "content": "Python is a popular programming language for data science.",
                "metadata": {"category": "programming", "difficulty": "beginner"}
            }
        ]
    
    def similarity_search(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Mock similarity search that returns documents based on simple keyword matching."""
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            # Simple keyword matching for testing
            if any(word in doc["content"].lower() for word in query_lower.split()):
                if not filters or self._matches_filters(doc["metadata"], filters):
                    results.append(doc)
        
        return results[:k]
    
    def vector_search(self, query_vector: List[float], k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Mock vector search that returns first k documents."""
        results = []
        for doc in self.documents:
            if not filters or self._matches_filters(doc["metadata"], filters):
                results.append(doc)
        return results[:k]
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Mock hybrid search that combines similarity and keyword search."""
        # For testing, just return similarity search results
        return self.similarity_search(query, k)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Return collection metadata."""
        return {
            "name": self.collection_name,
            "document_count": len(self.documents),
            "vector_store_type": "mock"
        }
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches the given filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True


class TestVectorStoreInterface:
    """Test suite for VectorStoreInterface abstract class and mock implementation."""
    
    def test_abstract_base_class(self):
        """Test that VectorStoreInterface is an abstract base class."""
        assert issubclass(VectorStoreInterface, ABC)
        
        # Should not be able to instantiate abstract class directly
        with pytest.raises(TypeError):
            VectorStoreInterface()
    
    def test_mock_vector_store_initialization(self):
        """Test mock vector store initialization."""
        store = MockVectorStore("test_collection")
        assert store.collection_name == "test_collection"
        assert len(store.documents) == 3
    
    def test_similarity_search_basic(self):
        """Test basic similarity search functionality."""
        store = MockVectorStore()
        
        # Search for machine learning content
        results = store.similarity_search("machine learning", k=2)
        assert len(results) <= 2
        assert any("machine learning" in doc["content"].lower() for doc in results)
    
    def test_similarity_search_with_filters(self):
        """Test similarity search with metadata filters."""
        store = MockVectorStore()
        
        # Search with category filter
        results = store.similarity_search(
            "learning", 
            k=5, 
            filters={"category": "AI"}
        )
        assert len(results) > 0
        assert all(doc["metadata"]["category"] == "AI" for doc in results)
    
    def test_similarity_search_empty_results(self):
        """Test similarity search with no matching results."""
        store = MockVectorStore()
        
        # Search for content that doesn't exist
        results = store.similarity_search("quantum computing", k=5)
        assert len(results) == 0
    
    def test_vector_search_basic(self):
        """Test basic vector search functionality."""
        store = MockVectorStore()
        
        # Mock vector search with dummy embeddings
        dummy_vector = [0.1] * 384  # Common embedding dimension
        results = store.vector_search(dummy_vector, k=2)
        assert len(results) == 2
        assert all("id" in doc for doc in results)
    
    def test_vector_search_with_filters(self):
        """Test vector search with metadata filters."""
        store = MockVectorStore()
        
        dummy_vector = [0.1] * 384
        results = store.vector_search(
            dummy_vector, 
            k=5, 
            filters={"difficulty": "beginner"}
        )
        assert len(results) > 0
        assert all(doc["metadata"]["difficulty"] == "beginner" for doc in results)
    
    def test_hybrid_search_basic(self):
        """Test hybrid search functionality."""
        store = MockVectorStore()
        
        results = store.hybrid_search("machine learning", k=3, alpha=0.5)
        assert len(results) <= 3
        # Should return similar results to similarity search for this mock
        similarity_results = store.similarity_search("machine learning", k=3)
        assert len(results) == len(similarity_results)
    
    def test_get_collection_info(self):
        """Test collection info retrieval."""
        store = MockVectorStore("my_collection")
        
        info = store.get_collection_info()
        assert info["name"] == "my_collection"
        assert info["document_count"] == 3
        assert info["vector_store_type"] == "mock"
    
    def test_k_parameter_limits_results(self):
        """Test that k parameter properly limits number of results."""
        store = MockVectorStore()
        
        # Test with similarity search
        results = store.similarity_search("learning", k=1)
        assert len(results) <= 1
        
        # Test with vector search  
        dummy_vector = [0.1] * 384
        results = store.vector_search(dummy_vector, k=2)
        assert len(results) <= 2
    
    def test_filter_matching_logic(self):
        """Test the internal filter matching logic."""
        store = MockVectorStore()
        
        # Test exact match
        assert store._matches_filters(
            {"category": "AI", "difficulty": "beginner"},
            {"category": "AI"}
        )
        
        # Test no match
        assert not store._matches_filters(
            {"category": "programming", "difficulty": "beginner"},
            {"category": "AI"}
        )
        
        # Test multiple filters
        assert store._matches_filters(
            {"category": "AI", "difficulty": "beginner"},
            {"category": "AI", "difficulty": "beginner"}
        )
        
        # Test missing key
        assert not store._matches_filters(
            {"category": "AI"},
            {"category": "AI", "difficulty": "beginner"}
        )


class TestVectorStoreInterfaceRequiredMethods:
    """Test that required abstract methods are properly defined."""
    
    def test_required_methods_exist(self):
        """Test that all required abstract methods are defined."""
        required_methods = [
            'similarity_search',
            'vector_search', 
            'get_collection_info'
        ]
        
        for method_name in required_methods:
            assert hasattr(VectorStoreInterface, method_name)
            method = getattr(VectorStoreInterface, method_name)
            assert getattr(method, '__isabstractmethod__', False)
    
    def test_optional_methods_exist(self):
        """Test that optional methods with default implementations exist."""
        optional_methods = ['hybrid_search']
        
        for method_name in optional_methods:
            assert hasattr(VectorStoreInterface, method_name)
            # These should not be abstract methods
            method = getattr(VectorStoreInterface, method_name)
            assert not getattr(method, '__isabstractmethod__', False)