# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any

from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface


class WeaviateVectorStore(VectorStoreInterface):
    """
    Weaviate implementation of the VectorStoreInterface.

    Weaviate is a cloud-native, modular, real-time vector database
    with powerful search capabilities including hybrid search.
    Supports both Weaviate Cloud Services and self-hosted deployments.
    """

    def __init__(self, client, collection_name: str, embedding_function=None):
        """
        Initialize WeaviateVectorStore.

        Args:
            client: Weaviate client instance (WeaviateClient from weaviate-client)
            collection_name: Name of the collection/class to use
            embedding_function: Optional function to compute embeddings for queries
        """
        import importlib.util

        if importlib.util.find_spec("weaviate") is None:
            raise ImportError(
                "Weaviate client is required for WeaviateVectorStore. Install with: pip install litellm weaviate-client"
            )

        import weaviate.classes as wvc

        self.client = client
        self.collection_name = collection_name
        self.wvc = wvc
        self.embedding_function = embedding_function

        # Get the collection
        try:
            self.collection = self.client.collections.get(collection_name)
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
        weaviate_filters = self._convert_filters(filters) if filters else None

        try:
            # Execute query
            if weaviate_filters:
                response = self.collection.query.near_vector(near_vector=query_vector, limit=k).where(weaviate_filters)
            else:
                response = self.collection.query.near_vector(near_vector=query_vector, limit=k)

            # Handle GenerativeReturn object - access .objects attribute
            if hasattr(response, "objects"):
                results = response.objects
            else:
                results = response

            return self._format_results(results)

        except Exception as e:
            raise RuntimeError(f"Weaviate vector search failed: {e!s}") from e

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid semantic + keyword search using Weaviate's native hybrid search.

        Args:
            query: Text query to search for
            k: Number of documents to return
            alpha: Weight for semantic vs keyword search (0.0 = pure keyword, 1.0 = pure semantic)
            filters: Optional metadata filters
        """
        weaviate_filters = self._convert_filters(filters) if filters else None

        try:
            results = self.collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=k,
                where=weaviate_filters,
                return_metadata=self.wvc.query.MetadataQuery(score=True, explain_score=True),
                return_properties=["content", "*"],
            )

            return self._format_results(results)

        except Exception as e:
            raise RuntimeError(f"Weaviate hybrid search failed: {e!s}") from e

    def get_collection_info(self) -> dict[str, Any]:
        """Get metadata about the Weaviate collection."""
        try:
            # Get collection configuration
            config = self.collection.config.get()

            # Count objects in collection
            count_result = self.collection.aggregate.over_all(total_count=True)
            total_count = count_result.total_count

            # Try to determine vector dimensions
            dimension = 0
            if hasattr(config, "vector_config") and config.vector_config:
                # For collections with vector config
                if hasattr(config.vector_config, "vector_index_config"):
                    vector_index = config.vector_index_config
                    if hasattr(vector_index, "distance"):
                        # This indicates vectors are configured
                        # Try to get dimension from a sample object
                        try:
                            sample = self.collection.query.fetch_objects(limit=1, include_vector=True)
                            if sample.objects and hasattr(sample.objects[0], "vector") and sample.objects[0].vector:
                                # Handle both named vectors and default vector
                                vector_data = sample.objects[0].vector
                                if isinstance(vector_data, dict):
                                    # Named vectors case
                                    for _vector_name, vector_values in vector_data.items():
                                        if vector_values:
                                            dimension = len(vector_values)
                                            break
                                elif isinstance(vector_data, list):
                                    # Default vector case
                                    dimension = len(vector_data)
                        except Exception:
                            # If we can't determine dimension from sample, that's ok
                            pass

            # Determine if hybrid search is supported
            supports_hybrid = True  # Weaviate generally supports hybrid search
            try:
                # Test if hybrid search works by doing a minimal query
                self.collection.query.hybrid(query="test", limit=1)
                supports_hybrid = True
            except Exception:
                supports_hybrid = False

            return {
                "name": self.collection_name,
                "document_count": total_count,
                "dimension": dimension,
                "vector_store_type": "weaviate",
                "supports_hybrid_search": supports_hybrid,
                "vectorizer": getattr(config, "vectorizer_config", None),
                "properties": [prop.name for prop in getattr(config, "properties", [])],
            }

        except Exception as e:
            # Fallback info if detailed info fails
            return {
                "name": self.collection_name,
                "document_count": 0,
                "dimension": 0,
                "vector_store_type": "weaviate",
                "error": str(e),
            }

    def supports_hybrid_search(self) -> bool:
        """Weaviate supports hybrid search."""
        try:
            # Test if hybrid search works
            self.collection.query.hybrid(query="test", limit=1)
            return True
        except Exception:
            return False

    def supports_metadata_filtering(self) -> bool:
        """Weaviate supports metadata filtering with where clauses."""
        return True

    def _convert_filters(self, filters: dict[str, Any]) -> Any:
        """
        Convert generic filters to Weaviate where clause format.

        Generic format: {"key": "value", "key2": {"$gt": 5}}
        Weaviate format uses Filter class
        """
        if not filters:
            return None

        try:
            from weaviate.collections.classes.filters import Filter
        except ImportError:
            # Fallback if Filter class not available
            return None

        filter_conditions = []

        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle operator-based filters
                for op, op_value in value.items():
                    if op == "$eq" or op == "equal":
                        filter_conditions.append(Filter.by_property(key).equal(op_value))
                    elif op == "$ne" or op == "not_equal":
                        filter_conditions.append(Filter.by_property(key).not_equal(op_value))
                    elif op == "$gt" or op == "greater_than":
                        filter_conditions.append(Filter.by_property(key).greater_than(op_value))
                    elif op == "$gte" or op == "greater_equal":
                        filter_conditions.append(Filter.by_property(key).greater_or_equal(op_value))
                    elif op == "$lt" or op == "less_than":
                        filter_conditions.append(Filter.by_property(key).less_than(op_value))
                    elif op == "$lte" or op == "less_equal":
                        filter_conditions.append(Filter.by_property(key).less_or_equal(op_value))
                    elif op == "$in" or op == "contains_any":
                        if isinstance(op_value, list):
                            filter_conditions.append(Filter.by_property(key).contains_any(op_value))
                    elif op == "$like" or op == "like":
                        filter_conditions.append(Filter.by_property(key).like(op_value))
            else:
                # Simple equality filter
                filter_conditions.append(Filter.by_property(key).equal(value))

        # Combine all conditions with AND
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        elif len(filter_conditions) > 1:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            return combined_filter

        return None

    def _format_results(self, results) -> list[dict[str, Any]]:
        """Convert Weaviate results to standardized format."""
        documents = []

        # Handle both GenerativeReturn objects and direct lists
        if hasattr(results, "objects"):
            objects_list = results.objects
        else:
            objects_list = results

        if not objects_list:
            return documents

        for obj in objects_list:
            # Get the content - try different property names
            content = ""
            properties = obj.properties

            # Common content field names
            content_fields = ["content", "text", "document", "body", "description"]
            for field in content_fields:
                if field in properties:
                    content = properties[field]
                    break

            # If no content field found, use all text properties
            if not content:
                text_properties = []
                for key, value in properties.items():
                    if isinstance(value, str) and value.strip():
                        text_properties.append(f"{key}: {value}")
                content = " | ".join(text_properties)

            # Extract metadata (all properties except content)
            metadata = {}
            for key, value in properties.items():
                if key not in ["content", "text", "document", "body"] or not content:
                    metadata[key] = value

            # Add UUID as doc_id in metadata
            metadata["doc_id"] = str(obj.uuid)

            # Calculate similarity score from distance or use provided score
            score = 0.0
            if hasattr(obj.metadata, "score") and obj.metadata.score is not None:
                score = float(obj.metadata.score)
            elif hasattr(obj.metadata, "distance") and obj.metadata.distance is not None:
                # Convert distance to similarity (assuming cosine distance)
                # Weaviate cosine distance is between 0 and 2, similarity = 1 - (distance/2)
                distance = float(obj.metadata.distance)
                score = max(0.0, 1.0 - (distance / 2.0))

            documents.append({"content": content, "metadata": metadata, "score": score})

        return documents

    @classmethod
    def create_local(
        cls,
        host: str = "localhost",
        port: int = 8080,
        grpc_port: int = 50051,
        collection_name: str = "Documents",
        headers: dict[str, str] | None = None,
    ) -> "WeaviateVectorStore":
        """
        Create a WeaviateVectorStore connected to local Weaviate instance.

        Args:
            host: Weaviate host
            port: HTTP port
            grpc_port: gRPC port
            collection_name: Name of the collection
            headers: Optional headers for authentication

        Returns:
            WeaviateVectorStore instance
        """
        try:
            import weaviate
        except ImportError as e:
            raise ImportError("Weaviate client is required. Install with: pip install litellm weaviate-client") from e

        client = weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port, headers=headers)

        return cls(client, collection_name)

    @classmethod
    def create_cloud(
        cls,
        cluster_url: str,
        auth_credentials,
        collection_name: str = "Documents",
        headers: dict[str, str] | None = None,
    ) -> "WeaviateVectorStore":
        """
        Create a WeaviateVectorStore connected to Weaviate Cloud Services.

        Args:
            cluster_url: Weaviate cluster URL
            auth_credentials: Authentication credentials (API key or other auth)
            collection_name: Name of the collection
            headers: Optional headers

        Returns:
            WeaviateVectorStore instance
        """
        try:
            import weaviate
        except ImportError as e:
            raise ImportError("Weaviate client is required. Install with: pip install litellm weaviate-client") from e

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url, auth_credentials=auth_credentials, headers=headers
        )

        return cls(client, collection_name)

    @classmethod
    def create_custom(
        cls,
        url: str,
        auth_credentials=None,
        collection_name: str = "Documents",
        headers: dict[str, str] | None = None,
        grpc_port: int | None = None,
    ) -> "WeaviateVectorStore":
        """
        Create a WeaviateVectorStore with custom connection parameters.

        Args:
            url: Weaviate instance URL
            auth_credentials: Authentication credentials
            collection_name: Name of the collection
            headers: Optional headers
            grpc_port: Optional gRPC port

        Returns:
            WeaviateVectorStore instance
        """
        try:
            import weaviate
        except ImportError as e:
            raise ImportError("Weaviate client is required. Install with: pip install litellm weaviate-client") from e

        client = weaviate.connect_to_custom(
            http_host=url,
            http_port=443 if "https" in url else 80,
            http_secure="https" in url,
            grpc_port=grpc_port,
            grpc_secure="https" in url,
            auth_credentials=auth_credentials,
            headers=headers,
        )

        return cls(client, collection_name)
