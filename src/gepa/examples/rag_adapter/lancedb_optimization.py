# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Example demonstrating GEPA optimization with LanceDB vector database.

This example shows how to use GEPA to optimize RAG prompts with LanceDB as the vector database.
LanceDB is a developer-friendly, serverless vector database built on Lance columnar format.

Prerequisites:
- Install LanceDB dependencies: pip install lancedb pyarrow
- LanceDB runs locally without Docker requirements
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
logging.getLogger("lancedb").setLevel(logging.WARNING)

import gepa
from gepa.adapters.generic_rag_adapter import (
    GenericRAGAdapter,
    RAGDataInst,
    LanceDBVectorStore,
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


def setup_lancedb_store():
    """Set up LanceDB vector store with sample data."""
    print("🗄️ Setting up LanceDB vector store...")
    
    try:
        # Create embedding function
        embedding_function = create_embedding_function()
        
        # Create local LanceDB vector store
        vector_store = LanceDBVectorStore.create_local(
            table_name="rag_demo",
            embedding_function=embedding_function,
            db_path="./lancedb_demo",  # Local directory for LanceDB
            vector_size=384  # all-MiniLM-L6-v2 embedding size
        )
        
        # Prepare sample documents
        documents = [
            {"content": "Machine learning is a method of data analysis that automates analytical model building.", "category": "ml"},
            {"content": "It is a branch of artificial intelligence based on the idea that systems can learn from data.", "category": "ai"},
            {"content": "Machine learning algorithms build a model based on training data to make predictions.", "category": "ml"},
            {"content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.", "category": "dl"},
            {"content": "It uses multiple layers to progressively extract higher-level features from raw input.", "category": "dl"},
            {"content": "Deep learning models can automatically learn representations of data with multiple levels of abstraction.", "category": "dl"},
            {"content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.", "category": "nlp"},
            {"content": "It deals with the interaction between computers and human language.", "category": "nlp"},
            {"content": "NLP techniques enable computers to process and analyze large amounts of natural language data.", "category": "nlp"},
        ]
        
        # Generate embeddings for documents
        embeddings = [embedding_function(doc["content"]) for doc in documents]
        
        # Add documents to vector store
        ids = vector_store.add_documents(documents, embeddings)
        print(f"✅ Added {len(ids)} documents to LanceDB table")
        
        # Get collection info
        info = vector_store.get_collection_info()
        print(f"📊 Table info: {info['document_count']} docs, dimension: {info['dimension']}")
        
        return vector_store
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install lancedb pyarrow sentence-transformers")
        raise
    except Exception as e:
        print(f"❌ Error setting up LanceDB: {e}")
        raise


def main():
    """Main optimization example."""
    print("🚀 Starting GEPA optimization with LanceDB")
    print("=" * 50)
    
    try:
        # Set up vector store
        vector_store = setup_lancedb_store()
        
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
        
        print(f"🔍 Retrieved context from {len(retrieved_docs)} documents")
        
        # Generate initial answer
        initial_prompt = f"""Based on the following context, answer the question.

Context: {' '.join(context_texts)}

Question: {sample_question["query"]}

Answer:"""
        
        initial_answer = llm_client(initial_prompt)
        cleaned_initial = clean_answer(initial_answer)
        
        print(f"❓ Question: {sample_question["query"]}")
        print(f"💭 Initial answer: {cleaned_initial}")
        
        # Test LanceDB's filtering capabilities
        print("\n🔎 Testing LanceDB filtering...")
        filtered_results = vector_store.similarity_search(
            "neural networks", 
            k=2, 
            filters={"category": "dl"}
        )
        print(f"✅ Found {len(filtered_results)} deep learning documents")
        
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
        
        # Test hybrid search if available
        if vector_store.supports_hybrid_search():
            print("\n🔀 Testing LanceDB hybrid search...")
            hybrid_results = vector_store.hybrid_search(
                "machine learning algorithms",
                k=2,
                alpha=0.7
            )
            print(f"✅ Hybrid search returned {len(hybrid_results)} results")
        
        print("\n✅ LanceDB optimization example completed!")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()