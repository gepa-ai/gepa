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
    
    # ==== Enhanced GEPA workflow validation (addresses PR feedback) ====
    
    # 1. Verify expected GEPA workflow structure - these values encode current behavior
    assert gepa_result.total_metric_calls == 7  # Actual observed value from test run
    assert gepa_result.num_full_val_evals == 1  # Only full validation evals, not all metric calls
    assert gepa_result.best_idx == 0  # First (base) program should be best in this deterministic test
    
    # 2. Verify score arrays have expected structure
    assert len(gepa_result.val_aggregate_scores) == 1  # Only one validation score recorded
    assert len(gepa_result.val_subscores) == 1  # Val subscores for each candidate
    assert len(gepa_result.val_subscores[0]) == 1  # One score per validation instance
    assert all(isinstance(score, (int, float)) for score in gepa_result.val_aggregate_scores)
    assert all(isinstance(subscores, list) for subscores in gepa_result.val_subscores)
    assert all(0 <= score <= 1 for score in gepa_result.val_aggregate_scores)  # Scores should be normalized
    
    # 3. Verify base program evaluation (iteration 0) happened
    base_val_score = gepa_result.val_aggregate_scores[0]  
    base_val_subscore = gepa_result.val_subscores[0][0]  # First candidate, first val instance
    assert isinstance(base_val_score, (int, float))
    assert isinstance(base_val_subscore, (int, float))
    assert base_val_score > 0  # Should have meaningful score from mock LLM
    assert base_val_score == base_val_subscore  # Should match since only one val instance
    
    # 4. Verify GEPA attempted optimization iterations (metric calls include subsampling)
    assert gepa_result.total_metric_calls > gepa_result.num_full_val_evals  # Should have done subsampling
    # With 7 total metric calls but only 1 full val eval, GEPA did internal optimization work
    
    # 5. Verify best candidate selection logic worked
    best_val_score = gepa_result.val_aggregate_scores[gepa_result.best_idx]
    assert best_val_score == max(gepa_result.val_aggregate_scores)  # Best idx should point to highest score
    
    # 6. Encode expected optimized prompt structure based on reflection_lm output
    # Our simple_reflection_lm returns a specific JSON structure - verify GEPA used it correctly
    expected_optimized_prompt_contains = [
        "comprehensive",  # From our reflection LLM output
        "accurate", 
        "Context:",  # Should preserve template variables
        "{context}",  # Template variable should be maintained
        "{query}"    # Template variable should be maintained  
    ]
    
    # The reflection may or may not be accepted, but if a new prompt was generated, it should have these characteristics
    final_prompt = gepa_result.best_candidate["answer_generation"]
    if final_prompt != seed_candidate["answer_generation"]:
        # If optimization occurred, verify the expected structure
        for expected_text in expected_optimized_prompt_contains[:2]:  # Check first 2 elements
            assert expected_text.lower() in final_prompt.lower(), f"Expected '{expected_text}' in optimized prompt"
        
        # Verify template variables are preserved
        assert "{context}" in final_prompt, "Template variable {context} should be preserved"
        assert "{query}" in final_prompt, "Template variable {query} should be preserved"
    
    # 7. Verify reproducibility - with same mock functions, results should be deterministic
    # Run a second optimization with identical setup
    adapter2 = TestRAGAdapter(vector_store=mock_chromadb_store, rag_config=rag_config)
    gepa_result2 = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter2,
        max_metric_calls=5,
        reflection_lm=simple_reflection_lm,
        display_progress_bar=False,  # Disable progress bar for cleaner test output
    )
    
    # Results should be identical due to deterministic mocks
    assert gepa_result2.total_metric_calls == gepa_result.total_metric_calls
    assert gepa_result2.num_full_val_evals == gepa_result.num_full_val_evals
    assert gepa_result2.best_idx == gepa_result.best_idx
    assert gepa_result2.val_aggregate_scores == gepa_result.val_aggregate_scores
    assert gepa_result2.best_candidate == gepa_result.best_candidate
    
    # 8. Record and assert exact expected results (regression test for GEPA workflow)
    # These values are captured from actual runs and should remain stable unless GEPA logic changes
    EXPECTED_EXACT_RESULTS = {
        "total_metric_calls": 7,  # Updated from actual test run
        "num_full_val_evals": 1,   # Updated from actual test run
        "best_idx": 0,  # Base program should be best with our deterministic setup
        # Exact scores depend on the mock LLM responses and evaluation logic
        # These values were captured from an actual test run  
        "expected_val_score": 0.6637837837837838,  # Exact score from deterministic run
        # The final prompt should match exactly - recorded from actual run
        "seed_prompt": "Answer the question '{query}' using the provided context: {context}",
        "expected_final_prompt": "Answer the question '{query}' using the provided context: {context}",  # No change in this deterministic case
        "expected_prompt_changed": False,  # Reflection didn't improve score, so prompt wasn't changed
        # Record what the reflection LLM proposed (even though it wasn't accepted)
        "proposed_optimized_prompt": "Based on the provided context, give a comprehensive and accurate answer to the question '{query}'. Structure your response clearly, include key definitions and explanations, and ensure your answer directly addresses all aspects of the question. Context: {context}",
    }
    
    # Assert exact workflow values match expected
    assert gepa_result.total_metric_calls == EXPECTED_EXACT_RESULTS["total_metric_calls"]
    assert gepa_result.num_full_val_evals == EXPECTED_EXACT_RESULTS["num_full_val_evals"] 
    assert gepa_result.best_idx == EXPECTED_EXACT_RESULTS["best_idx"]
    
    # Assert exact score from recorded run
    expected_val_score = EXPECTED_EXACT_RESULTS["expected_val_score"]
    assert base_val_score == expected_val_score, f"Base validation score {base_val_score} != expected {expected_val_score}"
    
    # 9. Assert exact final prompt (the key assertion requested in PR feedback)
    # Record the ACTUAL optimized prompt from the deterministic run
    expected_final_prompt = EXPECTED_EXACT_RESULTS["expected_final_prompt"]
    actual_final_prompt = gepa_result.best_candidate["answer_generation"]
    prompt_changed = (seed_candidate["answer_generation"] != actual_final_prompt)
    
    # Assert the exact prompt matches what we recorded
    assert actual_final_prompt == expected_final_prompt, \
        f"Final prompt doesn't match recorded result:\nActual: {repr(actual_final_prompt)}\nExpected: {repr(expected_final_prompt)}"
    
    # Assert whether prompt changed matches expected behavior
    expected_changed = EXPECTED_EXACT_RESULTS["expected_prompt_changed"]
    assert prompt_changed == expected_changed, \
        f"Prompt change behavior {prompt_changed} != expected {expected_changed}"
    
    # Document what GEPA's reflection LLM proposed (for reference, even if not accepted)
    proposed_prompt = EXPECTED_EXACT_RESULTS["proposed_optimized_prompt"]
    print(f"\n💡 GEPA Workflow Record:")
    print(f"   Seed prompt: {repr(seed_candidate['answer_generation'])}")
    print(f"   Proposed by reflection: {repr(proposed_prompt)}")  
    print(f"   Final prompt: {repr(actual_final_prompt)}")
    print(f"   Prompt was changed: {prompt_changed}")
    print(f"   Why unchanged: Reflection proposal didn't improve validation score")



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
