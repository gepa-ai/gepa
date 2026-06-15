# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import uuid
from typing import Any, ClassVar

from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface


class PGVectorStore(VectorStoreInterface):
    """
    PostgreSQL pgvector implementation of the VectorStoreInterface.

    pgvector is a PostgreSQL extension that adds vector similarity search to Postgres,
    making it an excellent choice for production RAG applications that already run
    PostgreSQL as their primary database.

    Requires the pgvector extension to be installed in the database:
        CREATE EXTENSION IF NOT EXISTS vector;

    Example:
        .. code-block:: python

            from gepa.adapters.generic_rag_adapter import PGVectorStore

            store = PGVectorStore.create_local(
                table_name="documents",
                embedding_function=my_embed_fn,
                vector_size=1536,
            )
            adapter = GenericRAGAdapter(vector_store=store)
    """

    _DISTANCE_OPS: ClassVar[dict[str, str]] = {
        "cosine": "<=>",
        "l2": "<->",
        "inner_product": "<#>",
    }
    _INDEX_OPS: ClassVar[dict[str, str]] = {
        "cosine": "vector_cosine_ops",
        "l2": "vector_l2_ops",
        "inner_product": "vector_ip_ops",
    }

    def __init__(
        self,
        conn,
        table_name: str,
        embedding_function=None,
        distance_metric: str = "cosine",
    ):
        """
        Initialize PGVectorStore.

        Args:
            conn: psycopg2 connection object (pgvector type registered automatically)
            table_name: Name of the table that stores the embeddings
            embedding_function: Optional callable(str) -> list[float] for text queries
            distance_metric: One of "cosine" (default), "l2", or "inner_product"
        """
        import importlib.util

        if importlib.util.find_spec("psycopg2") is None:
            raise ImportError(
                "psycopg2 is required for PGVectorStore. Install with: pip install psycopg2-binary pgvector"
            )
        if importlib.util.find_spec("pgvector") is None:
            raise ImportError(
                "pgvector Python package is required for PGVectorStore. Install with: pip install psycopg2-binary pgvector"
            )
        if distance_metric not in self._DISTANCE_OPS:
            raise ValueError(f"distance_metric must be one of {list(self._DISTANCE_OPS)}, got {distance_metric!r}")

        from pgvector.psycopg2 import register_vector

        self.conn = conn
        self.table_name = table_name
        self.embedding_function = embedding_function
        self.distance_metric = distance_metric
        self._distance_op = self._DISTANCE_OPS[distance_metric]

        register_vector(self.conn)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for documents semantically similar to the query text."""
        if self.embedding_function is None:
            raise ValueError("No embedding_function provided. Pass one to PGVectorStore.")

        try:
            query_vector = self.embedding_function(query)
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to compute embedding for query: {e!s}") from e

        return self.vector_search(query_vector, k, filters)

    def vector_search(
        self,
        query_vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using a pre-computed query embedding vector."""
        from psycopg2 import sql

        where_sql, filter_params = self._build_where_clause(filters)

        query_sql = sql.SQL(
            "SELECT id, content, metadata, {score} AS score "
            "FROM {table} "
            "{where} "
            "ORDER BY embedding {op} %s::vector ASC "
            "LIMIT %s"
        ).format(
            score=self._vector_score_sql(),
            table=sql.Identifier(self.table_name),
            where=where_sql,
            op=sql.SQL(self._distance_op),
        )

        # query_vector appears twice: once for score, once for ORDER BY
        params = [query_vector, *filter_params, query_vector, k]

        try:
            with self.conn.cursor() as cur:
                cur.execute(query_sql, params)
                rows = cur.fetchall()
            return self._format_rows(rows)
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"pgvector search failed: {e!s}") from e

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid semantic + full-text search combining pgvector and PostgreSQL tsvector.

        Score = alpha * vector_similarity + (1 - alpha) * ts_rank.
        """
        if self.embedding_function is None:
            raise ValueError("No embedding_function provided. Pass one to PGVectorStore.")

        try:
            query_vector = self.embedding_function(query)
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to compute embedding for query: {e!s}") from e

        from psycopg2 import sql

        where_sql, filter_params = self._build_where_clause(filters)

        query_sql = sql.SQL(
            "SELECT id, content, metadata, "
            "    ({alpha} * {vscore} "
            "     + {beta} * COALESCE(ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)), 0)) AS score "
            "FROM {table} "
            "{where} "
            "ORDER BY score DESC "
            "LIMIT %s"
        ).format(
            alpha=sql.Literal(alpha),
            vscore=self._vector_score_sql(),
            beta=sql.Literal(1.0 - alpha),
            table=sql.Identifier(self.table_name),
            where=where_sql,
        )

        params = [query_vector, query, *filter_params, k]

        try:
            with self.conn.cursor() as cur:
                cur.execute(query_sql, params)
                rows = cur.fetchall()
            return self._format_rows(rows)
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"pgvector hybrid search failed: {e!s}") from e

    def get_collection_info(self) -> dict[str, Any]:
        """Get metadata about the pgvector table."""
        from psycopg2 import sql

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(self.table_name)))
                doc_count = cur.fetchone()[0]

                # Filter by current schema to avoid ambiguity when multiple schemas
                # share the same table name.
                cur.execute(
                    "SELECT pg_attribute.atttypmod "
                    "FROM pg_attribute "
                    "JOIN pg_class ON attrelid = pg_class.oid "
                    "JOIN pg_namespace ON pg_class.relnamespace = pg_namespace.oid "
                    "WHERE pg_class.relname = %s AND pg_attribute.attname = 'embedding' "
                    "AND pg_namespace.nspname = current_schema()",
                    (self.table_name,),
                )
                row = cur.fetchone()
                dimension = row[0] if row else 0

            return {
                "name": self.table_name,
                "document_count": doc_count,
                "dimension": dimension,
                "vector_store_type": "pgvector",
                "distance_metric": self.distance_metric,
            }
        except Exception as e:
            self.conn.rollback()
            return {
                "name": self.table_name,
                "document_count": 0,
                "dimension": 0,
                "vector_store_type": "pgvector",
                "error": str(e),
            }

    def supports_hybrid_search(self) -> bool:
        return True

    def supports_metadata_filtering(self) -> bool:
        return True

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Insert or upsert documents with their embeddings."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        from psycopg2 import extras, sql

        insert_sql = sql.SQL(
            "INSERT INTO {table} (id, content, metadata, embedding) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET "
            "content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding"
        ).format(table=sql.Identifier(self.table_name))

        try:
            with self.conn.cursor() as cur:
                for doc_id, doc, embedding in zip(ids, documents, embeddings, strict=False):
                    content = doc.get("content", "")
                    metadata = {k: v for k, v in doc.items() if k != "content"}
                    cur.execute(insert_sql, (doc_id, content, extras.Json(metadata), embedding))
            self.conn.commit()
            return ids
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to add documents to pgvector: {e!s}") from e

    def delete_documents(self, ids: list[str]) -> bool:
        """Delete documents by ID."""
        from psycopg2 import sql

        delete_sql = sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(sql.Identifier(self.table_name))

        try:
            with self.conn.cursor() as cur:
                cur.execute(delete_sql, (ids,))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to delete documents from pgvector: {e!s}") from e

    def create_table(self, vector_size: int) -> None:
        """
        Create the pgvector table and an HNSW index if they do not already exist.

        Args:
            vector_size: Dimensionality of the embedding vectors
        """
        from psycopg2 import sql

        index_op = self._INDEX_OPS[self.distance_metric]

        create_table_sql = sql.SQL(
            "CREATE TABLE IF NOT EXISTS {table} ("
            "    id TEXT PRIMARY KEY, "
            "    content TEXT NOT NULL, "
            "    metadata JSONB DEFAULT '{{}}'::jsonb, "
            "    embedding vector({dim})"
            ")"
        ).format(table=sql.Identifier(self.table_name), dim=sql.Literal(vector_size))

        # HNSW index works on any data size (unlike ivfflat which requires training data)
        create_index_sql = sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} USING hnsw (embedding {op})").format(
            idx=sql.Identifier(f"{self.table_name}_embedding_hnsw_idx"),
            table=sql.Identifier(self.table_name),
            op=sql.SQL(index_op),
        )

        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(create_table_sql)
                cur.execute(create_index_sql)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to create pgvector table: {e!s}") from e

    def _vector_score_sql(self) -> "Any":
        """Return a psycopg2 sql.SQL fragment for the similarity score (one %s::vector placeholder).

        All scores are normalized to [0, 1]: cosine uses 1-distance (clamped),
        L2 uses 1/(1+distance), inner product clamps the dot product to [0, 1].
        """
        from psycopg2 import sql

        op = sql.SQL(self._distance_op)
        if self.distance_metric == "cosine":
            return sql.SQL("GREATEST(0.0, 1.0 - (embedding ") + op + sql.SQL(" %s::vector))")
        elif self.distance_metric == "l2":
            return sql.SQL("1.0 / (1.0 + (embedding ") + op + sql.SQL(" %s::vector))")
        else:  # inner_product: <#> returns -(a·b), so negate; clamp to [0,1] for normalized vectors
            return sql.SQL("GREATEST(0.0, LEAST(1.0, -(embedding ") + op + sql.SQL(" %s::vector)))")

    def _build_where_clause(self, filters: dict[str, Any] | None):
        """Build a parameterized WHERE clause from a filters dict.

        Returns:
            Tuple of (sql.SQL fragment, list of params). The SQL fragment is
            empty when there are no filters, or "WHERE cond1 AND cond2 ..." otherwise.
        """
        from psycopg2 import sql

        if not filters:
            return sql.SQL(""), []

        conditions = []
        params: list[Any] = []

        for key, value in filters.items():
            if isinstance(value, bool):
                conditions.append(sql.SQL("(metadata->>%s)::boolean = %s"))
                params.extend([key, value])
            elif isinstance(value, str):
                conditions.append(sql.SQL("metadata->>%s = %s"))
                params.extend([key, value])
            elif isinstance(value, int | float):
                conditions.append(sql.SQL("(metadata->>%s)::numeric = %s"))
                params.extend([key, value])
            elif isinstance(value, list):
                # Convert to strings for ANY(%s), matching JSONB's ->> text output.
                # Booleans must use lowercase ('true'/'false') to match JSONB's representation.
                str_values = []
                for v in value:
                    if isinstance(v, bool):
                        str_values.append("true" if v else "false")
                    else:
                        str_values.append(str(v))
                conditions.append(sql.SQL("metadata->>%s = ANY(%s)"))
                params.extend([key, str_values])
            elif isinstance(value, dict):
                op_map = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<"}
                for op_key, op_val in value.items():
                    pg_op = op_map.get(op_key)
                    if pg_op:
                        conditions.append(sql.SQL("(metadata->>%s)::numeric ") + sql.SQL(pg_op) + sql.SQL(" %s"))
                        params.extend([key, op_val])

        if not conditions:
            return sql.SQL(""), []

        where = sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions)
        return where, params

    def _format_rows(self, rows: list) -> list[dict[str, Any]]:
        """Convert raw psycopg2 rows to the standard document format."""
        documents = []
        for row in rows:
            doc_id, content, metadata, score = row
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            metadata = metadata or {}
            metadata["doc_id"] = doc_id
            documents.append(
                {
                    "content": content or "",
                    "metadata": metadata,
                    "score": float(score) if score is not None else 0.0,
                }
            )
        return documents

    @classmethod
    def from_connection_string(
        cls,
        dsn: str,
        table_name: str,
        embedding_function=None,
        distance_metric: str = "cosine",
        vector_size: int | None = None,
    ) -> "PGVectorStore":
        """
        Create a PGVectorStore from a PostgreSQL DSN.

        Args:
            dsn: PostgreSQL connection string, e.g. "postgresql://user:pass@host:5432/db"
            table_name: Table name to use for the vector store
            embedding_function: Optional callable(str) -> list[float]
            distance_metric: "cosine" (default), "l2", or "inner_product"
            vector_size: If provided, creates the table (and HNSW index) if it does not exist

        Returns:
            PGVectorStore instance
        """
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "psycopg2 is required for PGVectorStore. Install with: pip install psycopg2-binary pgvector"
            ) from e

        conn = psycopg2.connect(dsn)
        store = cls(conn, table_name, embedding_function, distance_metric)
        if vector_size is not None:
            store.create_table(vector_size)
        return store

    @classmethod
    def create_local(
        cls,
        table_name: str,
        embedding_function=None,
        distance_metric: str = "cosine",
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
        vector_size: int = 384,
    ) -> "PGVectorStore":
        """
        Connect to a local PostgreSQL instance and create the table if it does not exist.

        Args:
            table_name: Table name for the vector store
            embedding_function: Optional callable(str) -> list[float]
            distance_metric: "cosine" (default), "l2", or "inner_product"
            host: Database host (default: localhost)
            port: Database port (default: 5432)
            database: Database name (default: postgres)
            user: Database user (default: postgres)
            password: Database password
            vector_size: Embedding dimension (default: 384)

        Returns:
            PGVectorStore instance with the table created
        """
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "psycopg2 is required for PGVectorStore. Install with: pip install psycopg2-binary pgvector"
            ) from e

        # Use keyword arguments to avoid URL-parsing issues with special characters
        # in passwords or usernames (e.g. '@', '/', '?').
        conn = psycopg2.connect(host=host, port=port, dbname=database, user=user, password=password)
        store = cls(conn, table_name, embedding_function, distance_metric)
        store.create_table(vector_size)
        return store

    @classmethod
    def create_remote(
        cls,
        dsn: str,
        table_name: str,
        embedding_function=None,
        distance_metric: str = "cosine",
        vector_size: int = 384,
    ) -> "PGVectorStore":
        """
        Connect to a remote PostgreSQL instance and create the table if it does not exist.

        Args:
            dsn: Full PostgreSQL DSN including credentials and host
            table_name: Table name for the vector store
            embedding_function: Optional callable(str) -> list[float]
            distance_metric: "cosine" (default), "l2", or "inner_product"
            vector_size: Embedding dimension (default: 384)

        Returns:
            PGVectorStore instance with the table created
        """
        return cls.from_connection_string(dsn, table_name, embedding_function, distance_metric, vector_size)
