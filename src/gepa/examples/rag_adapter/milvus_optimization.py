# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Example demonstrating GEPA optimization with Milvus vector database.

This example shows how to use GEPA to optimize RAG prompts with Milvus as the vector database.
Milvus is a cloud-native vector database designed for large-scale AI applications.

This example uses Milvus Lite by default - a lightweight, local SQLite-based version
that doesn't require Docker. Perfect for development and testing.

Prerequisites:
- Install Milvus dependencies: pip install pymilvus
- No Docker required! Uses Milvus Lite (creates local ./milvus_demo.db file)

For production deployments, you can optionally use full Milvus server:
- Docker: docker run -d -p 19530:19530 milvusdb/milvus:latest standalone
- Then modify create_local() to create_remote() with uri="http://localhost:19530"
"""

import warnings
import logging
import os

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")

# Set logging level to reduce noise
logging.getLogger("pymilvus").setLevel(logging.WARNING)
logging.getLogger("grpc").setLevel(logging.WARNING)

import gepa
from gepa.adapters.generic_rag_adapter import (
    GenericRAGAdapter,
    RAGDataInst,
    MilvusVectorStore,
    RAGEvaluationMetrics,
)


def create_embedding_function():
    """Create a simple embedding function using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return lambda text: model.encode(text)
    except ImportError:
        print("sentence-transformers not found. Install with: pip install sentence-transformers")
        raise


def create_sample_rag_data():
    """Create sample data for RAG optimization."""
    return [
        RAGDataInst(
            query="What is machine learning?",
            ground_truth_answer="Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            relevant_doc_ids=["ml_doc"],
            metadata={"category": "definition", "difficulty": "beginner"}
        ),
        RAGDataInst(
            query="What is deep learning?",
            ground_truth_answer="Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. It uses multiple layers to progressively extract higher-level features from the raw input, allowing models to automatically learn data representations with multiple levels of abstraction.",
            relevant_doc_ids=["dl_doc"],
            metadata={"category": "definition", "difficulty": "intermediate"}
        ),
        RAGDataInst(
            query="What is natural language processing?",
            ground_truth_answer="Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. It focuses on how to program computers to process and analyze large amounts of natural language data.",
            relevant_doc_ids=["nlp_doc"],
            metadata={"category": "definition", "difficulty": "intermediate"}
        ),
    ]


def clean_answer(answer):
    """Clean answer by removing <think> tokens and formatting nicely."""
    if not answer:
        return "No answer generated"
    
    # Handle error cases
    if answer.startswith("Error:"):
        return answer[:500]
    
    # Remove <think> sections - handle multiple patterns
    import re
    
    # Remove <think>...</think> sections
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    
    # Remove standalone <think> or </think> tags
    answer = re.sub(r'</?think>', '', answer)
    
    # Clean up extra whitespace
    answer = ' '.join(answer.split())
    
    # Return first 500 characters for display
    return answer[:500] + "..." if len(answer) > 500 else answer


def setup_milvus_store():
    """Set up Milvus vector store with sample data."""
    print("🗄️ Setting up Milvus vector store...")
    
    try:
        # Create embedding function
        embedding_function = create_embedding_function()
        
        # Create local Milvus vector store (using Milvus Lite)
        # This creates a local database file instead of requiring Docker
        vector_store = MilvusVectorStore.create_local(
            collection_name="rag_demo",
            embedding_function=embedding_function,
            vector_size=384,  # all-MiniLM-L6-v2 embedding size
            uri="./milvus_demo.db"  # Local SQLite-based Milvus Lite
        )
        
        # Prepare sample documents
        documents = [
            {"content": "Machine learning is a method of data analysis that automates analytical model building."},
            {"content": "It is a branch of artificial intelligence based on the idea that systems can learn from data."},
            {"content": "Machine learning algorithms build a model based on training data to make predictions."},
            {"content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."},
            {"content": "It uses multiple layers to progressively extract higher-level features from raw input."},
            {"content": "Deep learning models can automatically learn representations of data with multiple levels of abstraction."},
            {"content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence."},
            {"content": "It deals with the interaction between computers and human language."},
            {"content": "NLP techniques enable computers to process and analyze large amounts of natural language data."},
        ]
        
        # Generate embeddings for documents
        embeddings = [embedding_function(doc["content"]) for doc in documents]
        
        # Add documents to vector store
        ids = vector_store.add_documents(documents, embeddings)
        print(f"✅ Added {len(ids)} documents to Milvus collection")
        
        # Get collection info
        info = vector_store.get_collection_info()
        print(f"📊 Collection info: {info['document_count']} docs, dimension: {info['dimension']}")
        
        return vector_store
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install pymilvus sentence-transformers")
        raise
    except Exception as e:
        print(f"❌ Error setting up Milvus: {e}")
        raise


def main():
    """Main optimization example."""
    print("🚀 Starting GEPA optimization with Milvus")
    print("=" * 50)
    
    try:
        # Set up vector store
        vector_store = setup_milvus_store()
        
        # Create sample data
        print("\n📝 Creating sample data...")
        rag_data = create_sample_rag_data()
        print(f"✅ Created {len(rag_data)} sample questions")
        
        # Set up LLM client (using Ollama with llama3.1:8b model)
        print("\n🤖 Setting up LLM client...")
        from litellm import completion
        
        def llm_client(prompt, **kwargs):
            try:
                # Handle both string prompts and message lists
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                elif isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [{"role": "user", "content": str(prompt)}]
                
                response = completion(
                    model="ollama/llama3.1:8b",
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error: {str(e)}"
        
        print("✅ LLM client configured (ollama/llama3.1:8b)")
        
        # Create RAG adapter
        print("\n🔧 Creating RAG adapter...")
        rag_adapter = GenericRAGAdapter(
            vector_store=vector_store,
            llm_model=llm_client,
            embedding_function=vector_store.embedding_function
        )
        
        # Set up metrics
        evaluation_metrics = RAGEvaluationMetrics()
        
        # Test initial performance
        print("\n📊 Testing initial performance...")
        sample_question = rag_data[0]
        
        # Retrieve relevant context
        retrieved_docs = vector_store.similarity_search(sample_question["query"], k=3)
        context_texts = [doc['content'] for doc in retrieved_docs]
        
        # Generate initial answer
        initial_prompt = f"""Based on the following context, answer the question.

Context: {' '.join(context_texts)}

Question: {sample_question["query"]}

Answer:"""
        
        initial_answer = llm_client(initial_prompt)
        cleaned_initial = clean_answer(initial_answer)
        
        print(f"❓ Question: {sample_question["query"]}")
        print(f"💭 Initial answer: {cleaned_initial}")
        
        # Run GEPA optimization
        print(f"\n🧠 Running GEPA optimization...")
        print("This may take a few minutes...")
        
        # Split data into train/val sets
        train_data = rag_data[:2]  # First 2 examples for training
        val_data = rag_data[2:]    # Last 1 example for validation
        
        # Create initial prompts (seed candidate)
        initial_prompts = {
            "answer_generation": """Based on the provided context, answer the question clearly and accurately.

Context: {context}

Question: {query}

Answer:"""
        }
        
        optimization_result = gepa.optimize(
            seed_candidate=initial_prompts,
            trainset=train_data,
            valset=val_data,
            adapter=rag_adapter,
            reflection_lm=llm_client,
            max_metric_calls=3
        )
        
        # Get optimization results
        print("\n🎯 Optimization completed!")
        
        best_score = optimization_result.val_aggregate_scores[optimization_result.best_idx]
        print(f"\n📈 Optimization results:")
        print(f"   🏆 Best validation score: {best_score:.3f}")
        print(f"   🔄 Total metric calls: {len(optimization_result.val_aggregate_scores)}")
        print(f"   📊 Best program index: {optimization_result.best_idx}")
        
        if len(optimization_result.val_aggregate_scores) > 1:
            initial_val_score = optimization_result.val_aggregate_scores[0]
            improvement = best_score - initial_val_score
            print(f"   📈 Score improvement: {improvement:+.3f}")
            
            if improvement > 0:
                print("🎉 Optimization successful! The prompts have been improved.")
            else:
                print("🤔 Limited improvement. Try running with more iterations or data.")
                print("   Note: Small datasets may not show significant improvement.")
        else:
            print("ℹ️ Single iteration completed.")
        
        print("\n✅ Milvus optimization example completed!")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()