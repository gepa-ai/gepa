#!/usr/bin/env python3
"""
GEPA Weaviate RAG Optimization Example

This example demonstrates how to use GEPA to optimize a RAG system using Weaviate
as the vector store, showcasing Weaviate's powerful hybrid search capabilities.

Usage:
    # Start Weaviate first (if using Docker)
    docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1
    
    # Basic usage with Ollama
    python weaviate_optimization.py
    
    # With specific model
    python weaviate_optimization.py --model ollama/llama3.1:8b
    
    # With cloud model (requires API key)
    python weaviate_optimization.py --model gpt-4o-mini
    
    # Skip data creation (use existing Weaviate collection)
    python weaviate_optimization.py --skip-setup

Requirements:
    pip install gepa[rag] weaviate-client

Prerequisites:
    - Running Weaviate instance (local Docker or cloud)
    - Or let the script create sample data automatically
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List

# Suppress all warnings
warnings.filterwarnings("ignore")
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import gepa
from gepa.adapters.generic_rag_adapter import (
    GenericRAGAdapter,
    WeaviateVectorStore,
    RAGDataInst,
)


def create_llm_client(model_name: str):
    """Create LLM client supporting both Ollama and cloud models."""
    try:
        import litellm
        litellm.drop_params = True
        litellm.set_verbose = False
    except ImportError:
        raise ImportError("LiteLLM is required. Install with: pip install litellm")
    
    def llm_client(messages_or_prompt, **kwargs):
        try:
            # Handle both string prompts and message lists
            if isinstance(messages_or_prompt, str):
                # Convert string prompt to messages format
                messages = [{"role": "user", "content": messages_or_prompt}]
            else:
                # Use as-is if it's already in messages format
                messages = messages_or_prompt
            
            params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 400),
                "temperature": kwargs.get("temperature", 0.1),
            }
            
            if "ollama/" in model_name:
                params["request_timeout"] = 120
                
            response = litellm.completion(**params)
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error: Unable to generate response ({str(e)})"
    
    return llm_client


def create_embedding_function():
    """Create embedding function using LiteLLM."""
    try:
        import litellm
    except ImportError:
        raise ImportError("LiteLLM is required. Install with: pip install litellm")
    
    def embed_text(text: str):
        """Embed text using LiteLLM."""
        try:
            response = litellm.embedding(model="ollama/nomic-embed-text:latest", input=text)
            # Handle different response formats
            if hasattr(response, 'data') and response.data:
                if hasattr(response.data[0], 'embedding'):
                    return response.data[0].embedding
                elif isinstance(response.data[0], dict) and 'embedding' in response.data[0]:
                    return response.data[0]['embedding']
            # Try direct access if it's a dict
            elif isinstance(response, dict):
                if 'data' in response and response['data']:
                    return response['data'][0]['embedding']
                elif 'embedding' in response:
                    return response['embedding']
            raise ValueError(f"Unknown response format: {type(response)}")
        except Exception as e:
            # Fallback to a simple placeholder vector if embedding fails
            import random
            print(f"Warning: Embedding failed, using random vector: {e}")
            return [random.random() for _ in range(768)]  # Common embedding dimension
    
    return embed_text


def setup_weaviate_knowledge_base():
    """Create Weaviate collection with AI/ML knowledge base."""
    try:
        import weaviate
        import weaviate.classes as wvc
    except ImportError:
        raise ImportError("Weaviate client required. Install with: pip install weaviate-client")
    
    # Connect to local Weaviate
    print("🔗 Connecting to Weaviate...")
    try:
        client = weaviate.connect_to_local()
        print("   ✅ Connected to local Weaviate")
    except Exception as e:
        print(f"   ❌ Failed to connect to Weaviate: {e}")
        print("   💡 Make sure Weaviate is running:")
        print("      docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1")
        raise
    
    # Delete existing collection if it exists
    collection_name = "AIKnowledge"
    try:
        client.collections.delete(collection_name)
        print(f"   🗑️ Removed existing collection: {collection_name}")
    except:
        pass
    
    # Create collection with hybrid search capabilities
    print(f"   📁 Creating collection: {collection_name}")
    collection = client.collections.create(
        name=collection_name,
        properties=[
            wvc.config.Property(
                name="content",
                data_type=wvc.config.DataType.TEXT,
                description="Document content"
            ),
            wvc.config.Property(
                name="topic", 
                data_type=wvc.config.DataType.TEXT,
                description="Topic category"
            ),
            wvc.config.Property(
                name="difficulty",
                data_type=wvc.config.DataType.TEXT,
                description="Difficulty level"
            ),
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        # Enable hybrid search (BM25 + semantic)
        inverted_index_config=wvc.config.Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
        ),
    )
    
    # AI/ML knowledge base documents
    documents = [
        {
            "content": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
            "topic": "artificial_intelligence",
            "difficulty": "beginner"
        },
        {
            "content": "Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "topic": "machine_learning", 
            "difficulty": "beginner"
        },
        {
            "content": "Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks have been applied to computer vision, speech recognition, and natural language processing.",
            "topic": "deep_learning",
            "difficulty": "intermediate"
        },
        {
            "content": "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. The goal is to program computers to process and analyze large amounts of natural language data.",
            "topic": "nlp",
            "difficulty": "intermediate"
        },
        {
            "content": "Computer Vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From an engineering perspective, it seeks to understand and automate tasks that the human visual system can do.",
            "topic": "computer_vision",
            "difficulty": "intermediate"
        },
        {
            "content": "Transformers are a deep learning architecture that has revolutionized natural language processing. They rely entirely on self-attention mechanisms to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely.",
            "topic": "transformers",
            "difficulty": "advanced"
        }
    ]
    
    # Create embedding function for document insertion
    embedding_fn = create_embedding_function()
    
    # Insert documents using batch processing
    print(f"   📄 Adding {len(documents)} documents...")
    with collection.batch.dynamic() as batch:
        for doc in documents:
            # Compute embedding for the document content
            doc_vector = embedding_fn(doc["content"])
            batch.add_object(properties=doc, vector=doc_vector)
    
    client.close()
    print(f"   ✅ Created Weaviate knowledge base with {len(documents)} articles")
    
    # Create embedding function and vector store wrapper
    embedding_fn = create_embedding_function()
    
    # Reconnect to get a fresh client for the vector store
    client_for_store = weaviate.connect_to_local()
    vector_store = WeaviateVectorStore(client_for_store, collection_name, embedding_fn)
    return vector_store


def create_training_data() -> tuple[List[RAGDataInst], List[RAGDataInst]]:
    """Create training and validation datasets for RAG optimization."""
    
    # Training examples
    train_data = [
        RAGDataInst(
            query="What is artificial intelligence?",
            ground_truth_answer="Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
            relevant_doc_ids=["doc_1"],  # Weaviate uses UUIDs, this is conceptual
            metadata={"category": "definition", "difficulty": "beginner"}
        ),
        RAGDataInst(
            query="How does machine learning work?",
            ground_truth_answer="Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            relevant_doc_ids=["doc_2"],
            metadata={"category": "explanation", "difficulty": "beginner"}
        ),
        RAGDataInst(
            query="What are the applications of deep learning?",
            ground_truth_answer="Deep learning architectures such as deep neural networks have been applied to computer vision, speech recognition, and natural language processing. It is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            relevant_doc_ids=["doc_3"],
            metadata={"category": "application", "difficulty": "intermediate"}
        ),
    ]
    
    # Validation examples
    val_data = [
        RAGDataInst(
            query="What is natural language processing used for?",
            ground_truth_answer="Natural Language Processing (NLP) is concerned with the interactions between computers and human language. The goal is to program computers to process and analyze large amounts of natural language data.",
            relevant_doc_ids=["doc_4"],
            metadata={"category": "explanation", "difficulty": "intermediate"}
        ),
        RAGDataInst(
            query="How do transformers work in AI?",
            ground_truth_answer="Transformers are a deep learning architecture that has revolutionized natural language processing. They rely entirely on self-attention mechanisms to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely.",
            relevant_doc_ids=["doc_6"],
            metadata={"category": "technical", "difficulty": "advanced"}
        ),
    ]
    
    return train_data, val_data


def clean_answer(answer: str) -> str:
    """Clean up LLM answer by removing thinking tokens and truncating appropriately."""
    # Remove thinking tokens if present
    import re
    cleaned = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    cleaned = cleaned.strip()
    
    # If still empty or starts with <think> without closing tag, try to find content after
    if not cleaned or cleaned.startswith('<think>'):
        # More aggressive approach - find any content that's not thinking
        lines = answer.split('\n')
        content_lines = []
        found_thinking = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip lines that are clearly thinking
            if '<think>' in line or line.startswith('Okay,') or line.startswith('Let me'):
                found_thinking = True
                continue
            if '</think>' in line:
                continue
                
            # If we found thinking before and now have substantial content, keep it
            if found_thinking and len(line) > 20 and not any(phrase in line.lower() for phrase in ['let me', 'okay', 'i need to', 'i should']):
                content_lines.append(line)
            elif not found_thinking:
                content_lines.append(line)
        
        # If we still don't have content, just take everything after the first few thinking lines
        if not content_lines:
            for i, line in enumerate(lines):
                if i > 3 and line.strip() and len(line.strip()) > 20:  # Skip first few lines and find substantial content
                    content_lines.append(line.strip())
                    if len(content_lines) >= 3:  # Take a few lines of actual content
                        break
        
        cleaned = ' '.join(content_lines)
    
    # Show more of the answer - increase limit significantly
    if len(cleaned) > 500:
        return cleaned[:500] + "..."
    return cleaned or answer[:500] + ("..." if len(answer) > 500 else "")


def create_weaviate_prompts() -> dict[str, str]:
    """Create prompts optimized for Weaviate's hybrid search capabilities."""
    return {
        "answer_generation": """You are an AI expert providing technical explanations using Weaviate's hybrid search results.

The context below was retrieved using both semantic similarity and keyword matching for optimal relevance.

Guidelines:
- Use the retrieved context as your primary information source
- Synthesize information from multiple sources when available
- Maintain technical accuracy and completeness
- Structure your response clearly with key concepts highlighted

Context: {context}

Question: {query}

Provide a comprehensive, accurate answer:"""
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GEPA Weaviate RAG Optimization Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python weaviate_optimization.py
  python weaviate_optimization.py --model ollama/llama3.1:8b
  python weaviate_optimization.py --model gpt-4o-mini --max-iterations 10
  python weaviate_optimization.py --skip-setup
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/qwen3:8b",
        help="LLM model (default: ollama/qwen3:8b)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="ollama/nomic-embed-text:latest",
        help="Embedding model (default: ollama/nomic-embed-text:latest)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="GEPA optimization iterations (default: 5, use 0 to skip optimization)"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip creating sample data (use existing Weaviate collection)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main function demonstrating Weaviate RAG optimization."""
    args = parse_arguments()
    
    print("🚀 GEPA Weaviate RAG Optimization")
    print("=" * 50)
    print(f"📊 Model: {args.model}")
    print(f"🔗 Embeddings: {args.embedding_model}")
    print(f"🔄 Max Iterations: {args.max_iterations}")
    print(f"⚙️ Skip Setup: {args.skip_setup}")
    
    try:
        # Step 1: Setup Weaviate knowledge base
        if not args.skip_setup:
            print("\n1️⃣ Setting up Weaviate knowledge base...")
            vector_store = setup_weaviate_knowledge_base()
        else:
            print("\n1️⃣ Connecting to existing Weaviate collection...")
            import weaviate
            client = weaviate.connect_to_local()
            embedding_fn = create_embedding_function()
            vector_store = WeaviateVectorStore(client, "AIKnowledge", embedding_fn)
            info = vector_store.get_collection_info()
            print(f"   ✅ Connected to collection: {info['name']} ({info['document_count']} docs)")
        
        # Step 2: Create datasets
        print("\n2️⃣ Creating training and validation datasets...")
        train_data, val_data = create_training_data()
        print(f"   📚 Training examples: {len(train_data)}")
        print(f"   📝 Validation examples: {len(val_data)}")
        
        # Step 3: Initialize LLM client
        print(f"\n3️⃣ Initializing LLM client ({args.model})...")
        llm_client = create_llm_client(args.model)
        
        # Test LLM
        test_response = llm_client([{"role": "user", "content": "Say 'OK' only."}])
        if "Error:" not in test_response:
            print(f"   ✅ LLM connected: {test_response[:30]}...")
        else:
            print(f"   ⚠️ LLM issue: {test_response}")
        
        # Step 4: Initialize RAG adapter with Weaviate hybrid search
        print("\n4️⃣ Initializing GenericRAGAdapter with Weaviate...")
        rag_adapter = GenericRAGAdapter(
            vector_store=vector_store,
            llm_model=llm_client,
            embedding_model=args.embedding_model,
            rag_config={
                "retrieval_strategy": "hybrid",  # Use Weaviate's hybrid search
                "top_k": 3,
                "retrieval_weight": 0.3,
                "generation_weight": 0.7,
                "hybrid_alpha": 0.7,  # More semantic than keyword
            }
        )
        
        # Verify hybrid search capability
        info = vector_store.get_collection_info()
        hybrid_support = info.get("supports_hybrid_search", False)
        print(f"   🔍 Hybrid search enabled: {hybrid_support}")
        
        # Step 5: Create Weaviate-optimized prompts
        print("\n5️⃣ Creating Weaviate-optimized prompts...")
        initial_prompts = create_weaviate_prompts()
        
        # Step 6: Test initial performance
        print("\n6️⃣ Testing initial performance...")
        eval_result = rag_adapter.evaluate(
            batch=val_data[:1],
            candidate=initial_prompts,
            capture_traces=True
        )
        
        initial_score = eval_result.scores[0]
        print(f"   📊 Initial score: {initial_score:.3f}")
        print(f"   💬 Sample answer: {clean_answer(eval_result.outputs[0]['final_answer'])}")
        
        if eval_result.trajectories:
            traj = eval_result.trajectories[0]
            print(f"   📄 Retrieved docs: {len(traj['retrieved_docs'])}")
        
        # Step 7: Run GEPA optimization
        if args.max_iterations > 0:
            print(f"\n7️⃣ Running GEPA optimization ({args.max_iterations} iterations)...")
            
            result = gepa.optimize(
                seed_candidate=initial_prompts,
                trainset=train_data,
                valset=val_data,
                adapter=rag_adapter,
                reflection_lm=llm_client,
                max_metric_calls=args.max_iterations
            )
            
            best_score = result.val_aggregate_scores[result.best_idx]
            print(f"   🎉 Optimization complete!")
            print(f"   🏆 Best score: {best_score:.3f}")
            print(f"   📈 Improvement: {best_score - initial_score:+.3f}")
            print(f"   🔄 Total iterations: {result.total_metric_calls or 0}")
            
            # Test optimized prompts
            print("\n   Testing optimized prompts...")
            optimized_result = rag_adapter.evaluate(
                batch=val_data[:1],
                candidate=result.best_candidate,
                capture_traces=False
            )
            print(f"   💬 Optimized answer: {clean_answer(optimized_result.outputs[0]['final_answer'])}")
            
        else:
            print("\n7️⃣ Skipping optimization (use --max-iterations > 0 to enable)")
        
        print("\n✅ Weaviate RAG optimization completed successfully!")
        
        # Clean up connections
        try:
            if 'vector_store' in locals() and hasattr(vector_store, 'client'):
                vector_store.client.close()
        except:
            pass
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print("\n🔧 Troubleshooting tips:")
        print("  • Ensure Weaviate is running: curl http://localhost:8080/v1/meta")
        print("  • Start Weaviate: docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1")
        print("  • Check Ollama: ollama list")
        print("  • For cloud models: set API keys (OPENAI_API_KEY, etc.)")
        print("  • Try smaller model: --model ollama/llama3.2:1b")


if __name__ == "__main__":
    main()