#!/usr/bin/env python3
"""
GEPA Qdrant RAG Optimization Example

This example demonstrates how to use GEPA to optimize a RAG system using Qdrant
as the vector store, showcasing Qdrant's powerful filtering and search capabilities.

Usage:
    # Basic usage with in-memory Qdrant
    python qdrant_optimization.py
    
    # With persistent local storage
    python qdrant_optimization.py --path ./qdrant_db
    
    # With remote Qdrant server
    python qdrant_optimization.py --host localhost --port 6333
    
    # With specific model
    python qdrant_optimization.py --model ollama/llama3.1:8b
    
    # With cloud model (requires API key)
    python qdrant_optimization.py --model gpt-4o-mini

Requirements:
    pip install gepa[rag] qdrant-client

Prerequisites:
    - For remote Qdrant: Running Qdrant instance (Docker: qdrant/qdrant)
    - For local models: Ollama with embedding models
    - Or cloud API keys for cloud models
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
    QdrantVectorStore,
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


def setup_qdrant_knowledge_base(path: str = ":memory:", host: str = None, port: int = None, api_key: str = None):
    """Create Qdrant collection with AI/ML knowledge base."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError:
        raise ImportError("Qdrant client required. Install with: pip install qdrant-client")
    
    # Connect to Qdrant
    print("🔗 Connecting to Qdrant...")
    try:
        if host and port:
            # Remote connection
            client = QdrantClient(host=host, port=port, api_key=api_key)
            print(f"   ✅ Connected to Qdrant at {host}:{port}")
        else:
            # Local connection
            client = QdrantClient(path=path)
            location_desc = "in-memory" if path == ":memory:" else f"at {path}"
            print(f"   ✅ Connected to local Qdrant {location_desc}")
    except Exception as e:
        print(f"   ❌ Failed to connect to Qdrant: {e}")
        if host:
            print(f"   💡 Make sure Qdrant is running at {host}:{port}")
            print("      Docker: docker run -p 6333:6333 qdrant/qdrant")
        raise
    
    # Collection setup
    collection_name = "AIKnowledge"
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        print(f"   🗑️ Removed existing collection: {collection_name}")
    except:
        pass
    
    # Create embedding function for document insertion
    embedding_fn = create_embedding_function()
    
    # Create collection
    print(f"   📁 Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=768,  # Match embedding dimension
            distance=models.Distance.COSINE
        )
    )
    
    # AI/ML knowledge base documents
    documents = [
        {
            "content": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
            "topic": "artificial_intelligence",
            "difficulty": "beginner",
            "category": "definition"
        },
        {
            "content": "Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "topic": "machine_learning", 
            "difficulty": "beginner",
            "category": "definition"
        },
        {
            "content": "Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks have been applied to computer vision, speech recognition, and natural language processing.",
            "topic": "deep_learning",
            "difficulty": "intermediate",
            "category": "technical"
        },
        {
            "content": "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. The goal is to program computers to process and analyze large amounts of natural language data.",
            "topic": "nlp",
            "difficulty": "intermediate", 
            "category": "technical"
        },
        {
            "content": "Computer Vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos and other visual inputs. It uses machine learning models to analyze and interpret visual data.",
            "topic": "computer_vision",
            "difficulty": "intermediate",
            "category": "application"
        },
        {
            "content": "Transformers are a deep learning architecture that has revolutionized natural language processing. They rely entirely on self-attention mechanisms to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely.",
            "topic": "transformers",
            "difficulty": "advanced",
            "category": "architecture"
        }
    ]
    
    # Insert documents with embeddings
    print(f"   📄 Adding {len(documents)} documents...")
    points = []
    for i, doc in enumerate(documents):
        # Compute embedding for the document content
        doc_vector = embedding_fn(doc["content"])
        # Add original string ID to payload for retrieval compatibility
        payload = dict(doc)
        payload["original_id"] = f"doc_{i+1}"
        
        point = models.PointStruct(
            id=i+1,  # Use integer ID for Qdrant compatibility
            vector=doc_vector,
            payload=payload
        )
        points.append(point)
    
    # Batch upsert points
    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True
    )
    
    print(f"   ✅ Created Qdrant knowledge base with {len(documents)} articles")
    
    # Create vector store wrapper
    vector_store = QdrantVectorStore(client, collection_name, embedding_fn)
    return vector_store


def create_training_data() -> tuple[List[RAGDataInst], List[RAGDataInst]]:
    """Create training and validation datasets for RAG optimization."""
    
    # Training examples
    train_data = [
        RAGDataInst(
            query="What is artificial intelligence?",
            ground_truth_answer="Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
            relevant_doc_ids=["doc_1"],
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


def create_qdrant_prompts() -> dict[str, str]:
    """Create prompts optimized for Qdrant's search and filtering capabilities."""
    return {
        "answer_generation": """You are an AI expert providing technical explanations using Qdrant's vector search results.

The context below was retrieved using advanced vector similarity search with metadata filtering for optimal relevance.

Guidelines:
- Use the retrieved context as your primary information source
- Leverage the rich metadata (topic, difficulty, category) to provide targeted explanations
- Synthesize information from multiple sources when available  
- Maintain technical accuracy and provide clear, structured responses
- Include relevant technical details and examples when appropriate

Context: {context}

Question: {query}

Provide a comprehensive, accurate answer:"""
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GEPA Qdrant RAG Optimization Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qdrant_optimization.py
  python qdrant_optimization.py --path ./qdrant_db
  python qdrant_optimization.py --host localhost --port 6333
  python qdrant_optimization.py --model ollama/llama3.1:8b --max-iterations 10
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
        help="Maximum optimization iterations (default: 5)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=":memory:",
        help="Local Qdrant database path (default: :memory:)"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Qdrant server host (for remote connection)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant server port (default: 6333)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Qdrant API key (for cloud/secured instances)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose error output"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("🚀 GEPA Qdrant RAG Optimization")
    print("=" * 50)
    print(f"📊 Model: {args.model}")
    print(f"🔗 Embeddings: {args.embedding_model}")
    print(f"🔄 Max Iterations: {args.max_iterations}")
    
    connection_info = f"Host: {args.host}:{args.port}" if args.host else f"Path: {args.path}"
    print(f"🗄️ Qdrant: {connection_info}")
    
    try:
        # Step 1: Setup Qdrant knowledge base
        print("\n1️⃣ Setting up Qdrant knowledge base...")
        vector_store = setup_qdrant_knowledge_base(
            path=args.path,
            host=args.host,
            port=args.port,
            api_key=args.api_key
        )
        
        # Step 2: Create datasets
        print("\n2️⃣ Creating training and validation datasets...")
        train_data, val_data = create_training_data()
        print(f"   📚 Training examples: {len(train_data)}")
        print(f"   📝 Validation examples: {len(val_data)}")
        
        # Step 3: Initialize LLM client
        print(f"\n3️⃣ Initializing LLM client ({args.model})...")
        llm_client = create_llm_client(args.model)
        test_response = llm_client("Hello")
        print(f"   ✅ LLM connected: {test_response[:50]}...")
        
        # Step 4: Initialize GenericRAGAdapter
        print("\n4️⃣ Initializing GenericRAGAdapter with Qdrant...")
        rag_adapter = GenericRAGAdapter(
            vector_store=vector_store,
            llm_model=llm_client,
            embedding_function=vector_store.embedding_function,
        )
        
        print(f"   🔍 Vector store type: {vector_store.get_collection_info()['vector_store_type']}")
        print(f"   📊 Document count: {vector_store.get_collection_info()['document_count']}")
        
        # Step 5: Create Qdrant-optimized prompts
        print("\n5️⃣ Creating Qdrant-optimized prompts...")
        initial_prompts = create_qdrant_prompts()
        
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
            
            # Show filtering capabilities
            for i, doc in enumerate(traj['retrieved_docs'][:2]):
                metadata = doc.get('metadata', {})
                topic = metadata.get('topic', 'unknown')
                difficulty = metadata.get('difficulty', 'unknown')
                print(f"      Doc {i+1}: {topic} ({difficulty})")
        
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
        
        print("\n✅ Qdrant RAG optimization completed successfully!")
        
        # Clean up connections
        try:
            if hasattr(vector_store, 'client'):
                # Qdrant client cleanup is handled automatically
                pass
        except:
            pass
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print("\n🔧 Troubleshooting tips:")
        if args.host:
            print(f"  • Ensure Qdrant is running at {args.host}:{args.port}")
            print("  • Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("  • Check Ollama: ollama list")
        print("  • Try: ollama pull qwen3:8b")
        print("  • Try: ollama pull nomic-embed-text:latest")
        print("  • For cloud models: set API keys (OPENAI_API_KEY, etc.)")
        
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())