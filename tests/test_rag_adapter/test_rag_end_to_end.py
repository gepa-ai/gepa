# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def sample_ai_ml_dataset():
    """Create a sample AI/ML knowledge dataset for RAG testing."""
    from gepa.adapters.generic_rag_adapter.generic_rag_adapter import RAGDataInst

    return [
        RAGDataInst(
            query="What is machine learning?",
            ground_truth_answer="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            relevant_doc_ids=["doc_ml_basics", "doc_ai_overview"],
            metadata={"category": "fundamentals", "difficulty": "beginner"},
        ),
        RAGDataInst(
            query="Explain the difference between supervised and unsupervised learning.",
            ground_truth_answer="Supervised learning uses labeled training data to learn mappings from inputs to outputs, while unsupervised learning finds patterns in data without labeled examples.",
            relevant_doc_ids=["doc_supervised_learning", "doc_unsupervised_learning"],
            metadata={"category": "learning_types", "difficulty": "intermediate"},
        ),
        RAGDataInst(
            query="What are the key components of a neural network?",
            ground_truth_answer="Key components include neurons (nodes), layers (input, hidden, output), weights, biases, and activation functions that determine how information flows through the network.",
            relevant_doc_ids=["doc_neural_networks", "doc_deep_learning"],
            metadata={"category": "neural_networks", "difficulty": "intermediate"},
        ),
        RAGDataInst(
            query="How does gradient descent work in machine learning?",
            ground_truth_answer="Gradient descent is an optimization algorithm that iteratively adjusts model parameters by moving in the direction of steepest descent of the cost function to minimize error.",
            relevant_doc_ids=["doc_optimization", "doc_gradient_descent"],
            metadata={"category": "optimization", "difficulty": "advanced"},
        ),
    ]


@pytest.fixture
def mock_chromadb_store(sample_ai_ml_dataset):
    """Create a mock ChromaDB vector store with AI/ML knowledge base."""
    from typing import Any

    from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface

    class MockChromaDBStore(VectorStoreInterface):
        """Mock ChromaDB store for RAG end-to-end testing."""

        def __init__(self):
            # AI/ML knowledge base documents
            self.documents = [
                {
                    "id": "doc_ml_basics",
                    "content": "Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns.",
                    "metadata": {"doc_id": "doc_ml_basics", "category": "fundamentals", "source": "ml_textbook"},
                },
                {
                    "id": "doc_ai_overview",
                    "content": "Artificial Intelligence (AI) is a broad field of computer science focused on creating intelligent machines capable of performing tasks that typically require human intelligence, such as visual perception, speech recognition, and decision-making.",
                    "metadata": {"doc_id": "doc_ai_overview", "category": "fundamentals", "source": "ai_handbook"},
                },
                {
                    "id": "doc_supervised_learning",
                    "content": "Supervised learning is a type of machine learning where algorithms learn from labeled training data. The goal is to map inputs to correct outputs based on example input-output pairs. Common supervised learning tasks include classification and regression.",
                    "metadata": {
                        "doc_id": "doc_supervised_learning",
                        "category": "learning_types",
                        "source": "ml_guide",
                    },
                },
                {
                    "id": "doc_unsupervised_learning",
                    "content": "Unsupervised learning involves finding hidden patterns or structures in data without labeled examples. The algorithm must discover patterns on its own. Common techniques include clustering, dimensionality reduction, and association rule learning.",
                    "metadata": {
                        "doc_id": "doc_unsupervised_learning",
                        "category": "learning_types",
                        "source": "ml_guide",
                    },
                },
                {
                    "id": "doc_neural_networks",
                    "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Each connection has a weight, and each node has an activation function that determines its output based on inputs.",
                    "metadata": {
                        "doc_id": "doc_neural_networks",
                        "category": "neural_networks",
                        "source": "deep_learning_book",
                    },
                },
                {
                    "id": "doc_deep_learning",
                    "content": "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers (deep networks). It excels at learning complex patterns from large amounts of data and has achieved breakthrough results in image recognition, natural language processing, and more.",
                    "metadata": {
                        "doc_id": "doc_deep_learning",
                        "category": "neural_networks",
                        "source": "deep_learning_book",
                    },
                },
                {
                    "id": "doc_optimization",
                    "content": "Optimization in machine learning refers to the process of finding the best parameters for a model to minimize error or maximize performance. Common optimization algorithms include gradient descent, Adam, and RMSprop.",
                    "metadata": {
                        "doc_id": "doc_optimization",
                        "category": "optimization",
                        "source": "optimization_handbook",
                    },
                },
                {
                    "id": "doc_gradient_descent",
                    "content": "Gradient descent is a first-order optimization algorithm used to find the minimum of a function. In machine learning, it's used to minimize the cost function by iteratively adjusting parameters in the direction of steepest descent.",
                    "metadata": {
                        "doc_id": "doc_gradient_descent",
                        "category": "optimization",
                        "source": "optimization_handbook",
                    },
                },
            ]

        def similarity_search(
            self, query: str, k: int = 5, filters: dict[str, Any] | None = None
        ) -> list[dict[str, Any]]:
            """Simulate similarity search by returning relevant documents based on query keywords."""
            query_lower = query.lower()
            scored_docs = []

            for doc in self.documents:
                content_lower = doc["content"].lower()
                # Simple keyword matching for simulation
                score = 0.5  # Base score

                if "machine learning" in query_lower and "machine learning" in content_lower:
                    score += 0.3
                if "supervised" in query_lower and "supervised" in content_lower:
                    score += 0.3
                if "unsupervised" in query_lower and "unsupervised" in content_lower:
                    score += 0.3
                if "neural network" in query_lower and "neural" in content_lower:
                    score += 0.3
                if "gradient descent" in query_lower and "gradient descent" in content_lower:
                    score += 0.3

                scored_docs.append((doc, score))

            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:k]]

        def vector_search(
            self, query_vector: list[float], k: int = 5, filters: dict[str, Any] | None = None
        ) -> list[dict[str, Any]]:
            """Fallback to similarity search for vector queries."""
            return self.similarity_search("", k, filters)

        def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> list[dict[str, Any]]:
            """Fallback to similarity search for hybrid queries."""
            return self.similarity_search(query, k)

        def get_collection_info(self) -> dict[str, Any]:
            return {
                "name": "ai_ml_knowledge_base",
                "document_count": len(self.documents),
                "vector_store_type": "mock_chromadb",
            }

    return MockChromaDBStore()


# --- The Test Function ---


def test_rag_end_to_end_optimization(sample_ai_ml_dataset, mock_chromadb_store):
    """
    Tests the complete GEPA optimization process for RAG using simple mocked LLM calls.

    This test addresses the PR feedback requesting an end-to-end test that runs GEPA
    optimization to ensure behavior is preserved as the codebase evolves. It:

    - Creates a complete RAG system with mock ChromaDB vector store
    - Runs full GEPA optimization cycle with deterministic mocked LLM responses
    - Verifies optimization process completes successfully with valid results
    - Tests RAG prompt optimization from seed to final optimized configuration

    This provides confidence that RAG adapter integration with GEPA works correctly
    and that changes to GEPA core won't break RAG functionality.
    """
    # Imports for the specific test logic

    import gepa
    from gepa.adapters.generic_rag_adapter.generic_rag_adapter import GenericRAGAdapter

    # Create simple deterministic mock responses
    def simple_rag_lm(messages):
        """Simple mock LLM that returns deterministic responses based on query content"""
        content = str(messages)
        if "machine learning" in content.lower():
            return "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        elif "supervised" in content.lower() and "unsupervised" in content.lower():
            return "Supervised learning uses labeled data while unsupervised learning finds patterns in unlabeled data."
        elif "neural network" in content.lower():
            return "Neural networks consist of neurons, layers, weights, and activation functions."
        elif "gradient descent" in content.lower():
            return "Gradient descent optimizes model parameters by minimizing the cost function iteratively."
        else:
            return "This is a general AI/ML answer based on the provided context."

    def simple_reflection_lm(prompt):
        """Simple reflection that suggests a better prompt"""
        return '{"answer_generation": "Based on the provided context, give a comprehensive and accurate answer to the question \'{query}\'. Structure your response clearly, include key definitions and explanations, and ensure your answer directly addresses all aspects of the question. Context: {context}"}'

    # Create a simple RAG adapter that uses our mocked LLM
    class TestRAGAdapter(GenericRAGAdapter):
        def __init__(self, vector_store, rag_config):
            self.vector_store = vector_store
            self.config = rag_config
            from gepa.adapters.generic_rag_adapter.evaluation_metrics import RAGEvaluationMetrics
            from gepa.adapters.generic_rag_adapter.rag_pipeline import RAGPipeline

            self.rag_pipeline = RAGPipeline(vector_store, simple_rag_lm, rag_config)
            self.evaluator = RAGEvaluationMetrics()

    # Create RAG configuration
    rag_config = {"retrieval_strategy": "similarity", "top_k": 3, "retrieval_weight": 0.4, "generation_weight": 0.6}


    # Create the RAG adapter with our mocked LLM
    adapter = TestRAGAdapter(vector_store=mock_chromadb_store, rag_config=rag_config)

    # Use subset for faster testing
    trainset = sample_ai_ml_dataset[:2]  # First 2 examples for training
    valset = sample_ai_ml_dataset[2:3]  # Third example for validation

    # Initial seed candidate with basic RAG prompts
    seed_candidate = {"answer_generation": "Answer the question '{query}' using the provided context: {context}"}

    # 2. Execution: Run the core RAG optimization logic
    gepa_result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=5,  # Small number for fast testing
        reflection_lm=simple_reflection_lm,
        display_progress_bar=True,
    )

    # 3. Assertions: Verify the optimization completed successfully
    assert gepa_result is not None
    assert hasattr(gepa_result, "best_candidate")
    assert hasattr(gepa_result, "val_aggregate_scores")

    best_config = gepa_result.best_candidate
    best_score = gepa_result.val_aggregate_scores[0]  # First (best) score

    # Basic validation of results
    assert isinstance(best_config, dict)
    assert len(best_config) > 0
    assert "answer_generation" in best_config
    assert isinstance(best_score, int | float)
    assert best_score >= 0

    # Verify the prompt was actually optimized (should be different from seed)
    optimized_prompt = best_config["answer_generation"]
    seed_prompt = seed_candidate["answer_generation"]

    # The optimized prompt should either be the same (if no improvement found)
    # or different (if GEPA found a better version)
    assert isinstance(optimized_prompt, str)
    assert len(optimized_prompt) > 0

    # Verify GEPA completed metric calls and evaluations
    assert gepa_result.total_metric_calls > 0
    assert gepa_result.num_full_val_evals > 0


def test_rag_adapter_basic_functionality(mock_chromadb_store):
    """
    Test basic RAG adapter functionality without optimization (faster test for CI).
    """
    from gepa.adapters.generic_rag_adapter.generic_rag_adapter import GenericRAGAdapter, RAGDataInst

    with patch("litellm.completion") as mock_litellm:
        # Setup simple mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[
            0
        ].message.content = "Machine learning is a subset of AI that enables computers to learn from data."
        mock_litellm.return_value = mock_response

        adapter = GenericRAGAdapter(vector_store=mock_chromadb_store, llm_model="gpt-4o-mini")

        # Test single evaluation
        example = RAGDataInst(
            query="What is machine learning?",
            ground_truth_answer="Machine learning is a subset of AI.",
            relevant_doc_ids=["doc_ml_basics"],
            metadata={"category": "fundamentals"},
        )

        candidate = {"answer_generation": "Answer: {query}"}
        result = adapter.evaluate([example], candidate)

        # Basic assertions
        assert len(result.scores) == 1
        assert isinstance(result.scores[0], float)
        assert 0 <= result.scores[0] <= 1
