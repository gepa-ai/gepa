# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any

from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface


class QdrantVectorStore(VectorStoreInterface):
    """
    Qdrant implementation of the VectorStoreInterface.

    Qdrant is an open-source vector database with excellent filtering capabilities
    and support for both REST and gRPC APIs. It excels at handling complex metadata
    filtering alongside vector similarity search.
    """

    def __init__(self, client, collection_name: str, embedding_function=None):
        """
        Initialize QdrantVectorStore.

        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to use
            embedding_function: Optional function to compute embeddings for queries
        """
        import importlib.util

        if importlib.util.find_spec("qdrant_client") is None:
            raise ImportError(
                "Qdrant client is required for QdrantVectorStore. Install with: pip install litellm qdrant-client"
            )

        from qdrant_client.http import models

        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.models = models

        # Verify collection exists
        try:
            self.client.get_collection(collection_name)
        except Exception as e:
            raise ValueError(
                f"Collection '{collection_name}' not found. Please create the collection first. Error: {e!s}"
            ) from e

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for documents similar to the query text using embeddings."""
        if self.embedding_function is None:
            raise ValueError("No embedding function provided for similarity search")

        # Compute embeddings for the query
        try:
            query_vector = self.embedding_function(query)
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to compute embeddings for query: {e!s}") from e

        # Use vector search with computed embeddings
        return self.vector_search(query_vector, k, filters)

    def vector_search(
        self,
        query_vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using a pre-computed query vector."""
        qdrant_filter = self._convert_filters(filters) if filters else None

        try:
            # Use modern query_points API
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=k,
                with_payload=True,
                with_vectors=False,
                score_threshold=None,
            )

            return self._format_results(results)

        except Exception as e:
            raise RuntimeError(f"Qdrant vector search failed: {e!s}") from e

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents with their embeddings to the collection."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        # Generate IDs if not provided - use integers for Qdrant compatibility
        if ids is None:
            ids = list(range(len(documents)))
            string_ids = [f"doc_{i}" for i in range(len(documents))]
        else:
            if len(ids) != len(documents):
                raise ValueError("Number of IDs must match number of documents")
            # Convert string IDs to integers for Qdrant, but keep original strings in payload
            string_ids = ids
            ids = list(range(len(documents)))

        # Create Qdrant points
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings, strict=False)):
            # Add the original string ID to the payload for retrieval
            payload = dict(doc)
            payload["original_id"] = string_ids[i]

            point = self.models.PointStruct(
                id=ids[i],  # Use integer ID for Qdrant
                vector=embedding,
                payload=payload,
            )
            points.append(point)

        try:
            # Upsert points to collection
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,  # Wait for operation to complete
            )

            if result.status.name != "COMPLETED":
                raise RuntimeError(f"Upsert operation failed: {result.status}")

            return string_ids  # Return original string IDs

        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Qdrant: {e!s}") from e

    def delete_documents(self, ids: list[str]) -> bool:
        """Delete documents by their IDs."""
        try:
            # Create filter to match documents by original_id
            delete_filter = self.models.Filter(
                must=[self.models.FieldCondition(key="original_id", match=self.models.MatchAny(any=ids))]
            )

            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=self.models.FilterSelector(filter=delete_filter),
                wait=True,
            )

            return result.status.name == "COMPLETED"

        except Exception as e:
            raise RuntimeError(f"Failed to delete documents from Qdrant: {e!s}") from e

    def get_collection_info(self) -> dict[str, Any]:
        """Get metadata about the Qdrant collection."""
        try:
            # Get collection info
            info = self.client.get_collection(self.collection_name)

            # Get collection statistics
            points_count = info.points_count or 0
            vectors_config = info.config.params.vectors

            # Handle both named and unnamed vector configs
            if isinstance(vectors_config, dict):
                # Named vectors
                vector_info = next(iter(vectors_config.values())) if vectors_config else None
                dimension = vector_info.size if vector_info else 0
                distance = vector_info.distance.name if vector_info else "UNKNOWN"
            else:
                # Single vector config
                dimension = vectors_config.size if vectors_config else 0
                distance = vectors_config.distance.name if vectors_config else "UNKNOWN"

            return {
                "name": self.collection_name,
                "document_count": points_count,
                "dimension": dimension,
                "vector_store_type": "qdrant",
                "distance_metric": distance.lower(),
                "status": info.status.name,
            }

        except Exception as e:
            # Fallback info if detailed info fails
            return {
                "name": self.collection_name,
                "document_count": 0,
                "dimension": 0,
                "vector_store_type": "qdrant",
                "error": str(e),
            }

    def supports_hybrid_search(self) -> bool:
        """Qdrant supports hybrid search through payload filtering."""
        return True

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining vector similarity and keyword matching.

        Note: Qdrant doesn't have built-in hybrid search like some other databases,
        but we can combine vector search with payload filtering for similar functionality.
        """
        # For now, implement as filtered vector search
        # In the future, this could be enhanced with text search capabilities
        return self.similarity_search(query, k, filters)

    def _format_results(self, results) -> list[dict[str, Any]]:
        """Convert Qdrant results to standardized format."""
        documents = []

        # Handle both direct points list and QueryResponse
        points = results.points if hasattr(results, "points") else results

        if not points:
            return documents

        for point in points:
            # Extract content from payload
            payload = point.payload or {}

            # Get content - try different field names
            content = ""
            content_fields = ["content", "text", "document", "body", "description"]
            for field in content_fields:
                if field in payload:
                    content = payload[field]
                    break

            # If no content field found, use all text properties
            if not content:
                text_properties = []
                for key, value in payload.items():
                    if isinstance(value, str) and value.strip():
                        text_properties.append(f"{key}: {value}")
                content = " | ".join(text_properties)

            # Create metadata (all payload except content field)
            metadata = {k: v for k, v in payload.items() if k not in content_fields and k != "original_id"}
            metadata["doc_id"] = payload.get("original_id", str(point.id))

            # Convert score (Qdrant returns similarity score, higher is better)
            score = float(point.score) if hasattr(point, "score") and point.score is not None else 0.0

            documents.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "score": score,
                }
            )

        return documents

    def _convert_filters(self, filters: dict[str, Any]) -> Any:
        """Convert generic filters to Qdrant filter format."""
        if not filters:
            return None

        must_conditions = []

        for key, value in filters.items():
            if isinstance(value, str):
                # Exact string match
                condition = self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value))
            elif isinstance(value, int | float):
                # Exact numeric match
                condition = self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value))
            elif isinstance(value, list):
                # Match any of the values
                condition = self.models.FieldCondition(key=key, match=self.models.MatchAny(any=value))
            elif isinstance(value, dict):
                # Range queries or complex conditions
                if "gte" in value or "gt" in value or "lte" in value or "lt" in value:
                    condition = self.models.FieldCondition(
                        key=key,
                        range=self.models.Range(
                            gte=value.get("gte"), gt=value.get("gt"), lte=value.get("lte"), lt=value.get("lt")
                        ),
                    )
                else:
                    # Skip unsupported filter format
                    continue
            else:
                # Skip unsupported filter type
                continue

            must_conditions.append(condition)

        if must_conditions:
            return self.models.Filter(must=must_conditions)

        return None

    @classmethod
    def create_local(
        cls, collection_name: str, embedding_function=None, path: str = ":memory:", vector_size: int = 384
    ):
        """Create a local Qdrant vector store (useful for testing)."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as e:
            raise ImportError("Qdrant client is required. Install with: pip install litellm qdrant-client") from e

        client = QdrantClient(path=path)

        # Create collection if it doesn't exist
        try:
            client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist, create it
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

        return cls(client, collection_name, embedding_function)

    @classmethod
    def create_remote(
        cls,
        collection_name: str,
        embedding_function=None,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
        vector_size: int = 384,
    ):
        """Create a remote Qdrant vector store."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as e:
            raise ImportError("Qdrant client is required. Install with: pip install litellm qdrant-client") from e

        client = QdrantClient(host=host, port=port, api_key=api_key)

        # Create collection if it doesn't exist
        try:
            client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist, create it
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

        return cls(client, collection_name, embedding_function)
