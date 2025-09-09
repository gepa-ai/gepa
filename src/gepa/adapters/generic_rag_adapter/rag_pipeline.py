# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any, Callable

from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface


class RAGPipeline:
    """
    Generic RAG pipeline that works with any vector store.

    This pipeline orchestrates the full RAG process:
    1. Query reformulation (optional)
    2. Document retrieval
    3. Document reranking (optional)
    4. Context synthesis
    5. Answer generation
    """

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        llm_client,
        embedding_model: str = "text-embedding-3-small",
        embedding_function: Callable[[str], list[float]] | None = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: Vector store interface implementation
            llm_client: LLM client for generation (should have a callable interface)
            embedding_model: Model name for embeddings (if using default embedding function)
            embedding_function: Optional custom embedding function
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.embedding_function = embedding_function

        # Initialize default embedding function if none provided
        if self.embedding_function is None:
            self.embedding_function = self._default_embedding_function

    def execute_rag(
        self,
        query: str,
        prompts: dict[str, str],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute the full RAG pipeline with given prompts and configuration.

        Args:
            query: User query
            prompts: Dictionary of prompt templates for different stages
            config: Configuration parameters for retrieval and generation

        Returns:
            Dictionary containing all pipeline outputs and metadata
        """
        # Step 1: Query reformulation (if enabled)
        reformulated_query = query
        if "query_reformulation" in prompts and prompts["query_reformulation"].strip():
            reformulated_query = self._reformulate_query(query, prompts["query_reformulation"])

        # Step 2: Retrieval
        retrieved_docs = self._retrieve_documents(reformulated_query, config)

        # Step 3: Reranking (if enabled)
        if "reranking_criteria" in prompts and prompts["reranking_criteria"].strip():
            retrieved_docs = self._rerank_documents(retrieved_docs, query, prompts["reranking_criteria"], config)

        # Step 4: Context synthesis
        context = self._synthesize_context(retrieved_docs, query, prompts.get("context_synthesis", ""))

        # Step 5: Answer generation
        answer = self._generate_answer(query, context, prompts.get("answer_generation", ""))

        return {
            "original_query": query,
            "reformulated_query": reformulated_query,
            "retrieved_docs": retrieved_docs,
            "synthesized_context": context,
            "generated_answer": answer,
            "metadata": {
                "retrieval_count": len(retrieved_docs),
                "total_tokens": self._estimate_token_count(context + answer),
                "vector_store_type": self.vector_store.get_collection_info().get("vector_store_type", "unknown"),
            },
        }

    def _reformulate_query(self, query: str, reformulation_prompt: str) -> str:
        """Reformulate the user query using the provided prompt."""
        messages = [
            {"role": "system", "content": reformulation_prompt},
            {"role": "user", "content": f"Original query: {query}"},
        ]

        try:
            if callable(self.llm_client):
                response = self.llm_client(messages)
            else:
                # Assume it's a litellm-style client
                response = self.llm_client.completion(messages=messages).choices[0].message.content

            return response.strip() if response else query
        except Exception:
            # Fallback to original query if reformulation fails
            return query

    def _retrieve_documents(self, query: str, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Retrieve documents using the configured strategy."""
        retrieval_strategy = config.get("retrieval_strategy", "similarity")
        k = config.get("top_k", 5)
        filters = config.get("filters", None)

        if retrieval_strategy == "similarity":
            return self.vector_store.similarity_search(query, k=k, filters=filters)
        elif retrieval_strategy == "hybrid":
            if self.vector_store.supports_hybrid_search():
                alpha = config.get("hybrid_alpha", 0.5)
                return self.vector_store.hybrid_search(query, k=k, alpha=alpha, filters=filters)
            else:
                # Fallback to similarity search
                return self.vector_store.similarity_search(query, k=k, filters=filters)
        elif retrieval_strategy == "vector":
            # Use pre-computed embedding
            query_vector = self.embedding_function(query)
            return self.vector_store.vector_search(query_vector, k=k, filters=filters)
        else:
            raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")

    def _rerank_documents(
        self, documents: list[dict[str, Any]], query: str, reranking_prompt: str, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Rerank documents based on relevance criteria."""
        if not documents:
            return documents

        # For simplicity, we'll use a prompt-based reranking approach
        # In production, you might use dedicated reranking models
        try:
            doc_texts = [f"Document {i + 1}: {doc['content']}" for i, doc in enumerate(documents)]
            doc_context = "\n\n".join(doc_texts)

            messages = [
                {"role": "system", "content": reranking_prompt},
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nDocuments:\n{doc_context}\n\nPlease rank these documents by relevance (return document numbers in order, e.g., '3,1,4,2,5'):",
                },
            ]

            if callable(self.llm_client):
                response = self.llm_client(messages)
            else:
                response = self.llm_client.completion(messages=messages).choices[0].message.content

            # Parse the ranking response
            ranking_str = response.strip()
            rankings = [int(x.strip()) - 1 for x in ranking_str.split(",") if x.strip().isdigit()]

            # Reorder documents based on ranking
            if len(rankings) == len(documents):
                return [documents[i] for i in rankings if 0 <= i < len(documents)]
        except Exception:
            pass

        # Return original order if reranking fails
        return documents

    def _synthesize_context(self, documents: list[dict[str, Any]], query: str, synthesis_prompt: str) -> str:
        """Synthesize retrieved documents into coherent context."""
        if not documents:
            return ""

        if not synthesis_prompt.strip():
            # Default: simple concatenation
            contexts = []
            for i, doc in enumerate(documents):
                contexts.append(f"[Document {i + 1}] {doc['content']}")
            return "\n\n".join(contexts)

        # Use LLM for context synthesis
        doc_texts = [doc["content"] for doc in documents]
        doc_context = "\n\n".join(f"Document {i + 1}: {text}" for i, text in enumerate(doc_texts))

        messages = [
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": f"Query: {query}\n\nRetrieved Documents:\n{doc_context}"},
        ]

        try:
            if callable(self.llm_client):
                response = self.llm_client(messages)
            else:
                response = self.llm_client.completion(messages=messages).choices[0].message.content

            return response.strip() if response else doc_context
        except Exception:
            # Fallback to simple concatenation
            return doc_context

    def _generate_answer(self, query: str, context: str, generation_prompt: str) -> str:
        """Generate the final answer using the query and synthesized context."""
        if not generation_prompt.strip():
            generation_prompt = "You are a helpful assistant. Answer the user's question based on the provided context."

        messages = [
            {"role": "system", "content": generation_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        try:
            if callable(self.llm_client):
                response = self.llm_client(messages)
            else:
                response = self.llm_client.completion(messages=messages).choices[0].message.content

            return response.strip() if response else "I couldn't generate an answer based on the provided context."
        except Exception as e:
            return f"Error generating answer: {e!s}"

    def _default_embedding_function(self, text: str) -> list[float]:
        """Default embedding function using litellm."""
        try:
            import litellm

            response = litellm.embedding(model=self.embedding_model, input=text)
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e!s}") from e

    def _estimate_token_count(self, text: str) -> int:
        """Rough estimate of token count (4 chars per token approximation)."""
        return len(text) // 4
