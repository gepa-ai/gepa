# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Simple tests for the vector store interface that don't require complex mocking.
"""

import pytest
from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface
from gepa.adapters.generic_rag_adapter.generic_rag_adapter import RAGDataInst


def test_rag_data_inst_creation():
    """Test creating a RAGDataInst instance."""
    data_inst = RAGDataInst(
        query="What is machine learning?",
        ground_truth_answer="ML is a subset of AI.",
        relevant_doc_ids=["doc1", "doc2"],
        metadata={"category": "AI", "difficulty": "beginner"}
    )
    
    assert data_inst["query"] == "What is machine learning?"
    assert data_inst["ground_truth_answer"] == "ML is a subset of AI."
    assert data_inst["relevant_doc_ids"] == ["doc1", "doc2"]
    assert data_inst["metadata"]["category"] == "AI"


def test_rag_data_inst_required_fields():
    """Test that RAGDataInst requires all specified fields."""
    data_inst = RAGDataInst(
        query="Test query",
        ground_truth_answer="Test answer",
        relevant_doc_ids=[],
        metadata={}
    )
    
    # All fields should be accessible
    assert "query" in data_inst
    assert "ground_truth_answer" in data_inst
    assert "relevant_doc_ids" in data_inst
    assert "metadata" in data_inst
    assert isinstance(data_inst["relevant_doc_ids"], list)
    assert isinstance(data_inst["metadata"], dict)


def test_vector_store_interface_is_abstract():
    """Test that VectorStoreInterface cannot be instantiated directly."""
    from abc import ABC
    
    assert issubclass(VectorStoreInterface, ABC)
    
    # Should not be able to instantiate abstract class directly
    with pytest.raises(TypeError):
        VectorStoreInterface()


def test_vector_store_interface_has_required_methods():
    """Test that VectorStoreInterface defines the required abstract methods."""
    required_methods = [
        'similarity_search',
        'vector_search', 
        'get_collection_info'
    ]
    
    for method_name in required_methods:
        assert hasattr(VectorStoreInterface, method_name)
        method = getattr(VectorStoreInterface, method_name)
        assert getattr(method, '__isabstractmethod__', False)


def test_vector_store_interface_has_optional_methods():
    """Test that VectorStoreInterface defines optional methods with default implementations."""
    optional_methods = ['hybrid_search']
    
    for method_name in optional_methods:
        assert hasattr(VectorStoreInterface, method_name)
        # These should not be abstract methods
        method = getattr(VectorStoreInterface, method_name)
        assert not getattr(method, '__isabstractmethod__', False)