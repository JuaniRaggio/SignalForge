"""Vector storage and similarity search using pgvector.

This module provides vector storage and similarity search capabilities
for document embeddings using PostgreSQL's pgvector extension. It supports
efficient nearest neighbor search with multiple distance metrics and index types.

Key Features:
- Multiple distance metrics (cosine, L2, inner product)
- Multiple index types (HNSW, IVF-Flat)
- Batch operations for efficient storage
- Metadata filtering for contextual search
- Async/await support throughout
- Integration with existing database session management

Examples:
    Basic usage with default configuration:

    >>> from signalforge.nlp.vector_store import VectorStore
    >>> from signalforge.core.database import async_session_factory
    >>>
    >>> vector_store = VectorStore(async_session_factory)
    >>> await vector_store.create_index(dimension=384)
    >>>
    >>> # Store a document
    >>> await vector_store.store(
    ...     document_id="doc_001",
    ...     text="Apple reports strong earnings",
    ...     embedding=[0.1, 0.2, ...],  # 384-dim vector
    ...     metadata={"source": "reuters", "symbol": "AAPL"}
    ... )
    >>>
    >>> # Search for similar documents
    >>> results = await vector_store.search(
    ...     query_embedding=[0.1, 0.2, ...],
    ...     k=5
    ... )

    Advanced configuration with HNSW index:

    >>> config = VectorStoreConfig(
    ...     index_type="hnsw",
    ...     distance_metric="cosine",
    ...     m=16,
    ...     ef_construction=64
    ... )
    >>> vector_store = VectorStore(async_session_factory, config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import literal_column
from sqlalchemy.sql.elements import ColumnElement

from signalforge.core.logging import get_logger
from signalforge.models.document_embedding import DocumentEmbedding

logger = get_logger(__name__)


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search.

    Attributes:
        document_id: Unique identifier of the document.
        text: Original text content of the document.
        embedding: Vector embedding of the document.
        distance: Raw distance score (depends on metric).
        similarity: Normalized similarity score (1 - distance for cosine).
        metadata: Additional metadata associated with the document.

    Examples:
        >>> result = VectorSearchResult(
        ...     document_id="doc_001",
        ...     text="Apple reports earnings",
        ...     embedding=[0.1, 0.2, ...],
        ...     distance=0.15,
        ...     similarity=0.85,
        ...     metadata={"source": "reuters"}
        ... )
        >>> print(f"Found document {result.document_id} with similarity {result.similarity:.3f}")
    """

    document_id: str
    text: str
    embedding: list[float]
    distance: float
    similarity: float
    metadata: dict[str, Any]


@dataclass
class VectorStoreConfig:
    """Configuration for vector store and indexing.

    Attributes:
        index_type: Type of vector index ("hnsw" or "ivfflat").
        lists: Number of lists for IVF-Flat index (ignored for HNSW).
        ef_construction: Size of dynamic candidate list for HNSW construction.
        m: Number of connections per layer for HNSW.
        distance_metric: Distance metric for similarity search.

    Distance Metrics:
        - cosine: Cosine distance (1 - cosine similarity), range [0, 2]
        - l2: Euclidean distance, range [0, infinity)
        - inner_product: Negative inner product, range (-infinity, infinity)

    Index Types:
        - hnsw: Hierarchical Navigable Small World (faster queries, more memory)
        - ivfflat: Inverted File with Flat Compression (less memory, slower)

    Examples:
        Default configuration (HNSW with cosine similarity):

        >>> config = VectorStoreConfig()

        IVF-Flat with L2 distance:

        >>> config = VectorStoreConfig(
        ...     index_type="ivfflat",
        ...     lists=100,
        ...     distance_metric="l2"
        ... )

        HNSW with high precision:

        >>> config = VectorStoreConfig(
        ...     index_type="hnsw",
        ...     m=32,
        ...     ef_construction=128,
        ...     distance_metric="cosine"
        ... )
    """

    index_type: Literal["ivfflat", "hnsw"] = "hnsw"
    lists: int = 100
    ef_construction: int = 64
    m: int = 16
    distance_metric: Literal["cosine", "l2", "inner_product"] = "cosine"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.lists <= 0:
            raise ValueError(f"lists must be positive, got {self.lists}")

        if self.ef_construction <= 0:
            raise ValueError(f"ef_construction must be positive, got {self.ef_construction}")

        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")


class VectorStore:
    """Vector storage and similarity search using pgvector.

    This class provides a high-level interface for storing document embeddings
    and performing similarity searches using PostgreSQL's pgvector extension.

    The store supports:
    - Multiple distance metrics (cosine, L2, inner product)
    - Multiple index types (HNSW, IVF-Flat)
    - Metadata filtering for contextual search
    - Batch operations for efficiency
    - Async/await throughout

    Examples:
        Initialize and create index:

        >>> from signalforge.core.database import async_session_factory
        >>> store = VectorStore(async_session_factory)
        >>> await store.create_index(dimension=384)

        Store documents:

        >>> await store.store(
        ...     document_id="doc_001",
        ...     text="Market rallied today",
        ...     embedding=[...],
        ...     metadata={"source": "reuters"}
        ... )

        Search for similar documents:

        >>> results = await store.search(
        ...     query_embedding=[...],
        ...     k=10,
        ...     filter_metadata={"source": "reuters"}
        ... )
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        config: VectorStoreConfig | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            session_factory: SQLAlchemy async session factory for database access.
            config: Configuration for indexing and search. If None, uses defaults.
        """
        self._session_factory = session_factory
        self._config = config or VectorStoreConfig()

        logger.info(
            "vector_store_initialized",
            index_type=self._config.index_type,
            distance_metric=self._config.distance_metric,
        )

    def _get_distance_operator(self) -> str:
        """Get the pgvector distance operator for the configured metric.

        Returns:
            SQL operator string for distance calculation.
        """
        operators = {
            "cosine": "<=>",  # Cosine distance
            "l2": "<->",  # L2 distance (Euclidean)
            "inner_product": "<#>",  # Negative inner product
        }
        return operators[self._config.distance_metric]

    async def _check_pgvector_extension(self, session: AsyncSession) -> bool:
        """Check if pgvector extension is installed and enabled.

        Args:
            session: Active database session.

        Returns:
            True if pgvector extension is available, False otherwise.
        """
        try:
            result = await session.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            row = result.fetchone()
            return row is not None
        except Exception as e:
            logger.error("pgvector_check_failed", error=str(e))
            return False

    async def create_index(self, dimension: int) -> None:
        """Create a vector index on the embeddings column.

        This method creates an appropriate index based on the configured
        index type (HNSW or IVF-Flat). The index significantly improves
        query performance for similarity searches.

        Args:
            dimension: Dimensionality of the embeddings (e.g., 384).

        Raises:
            RuntimeError: If pgvector extension is not installed.
            ValueError: If dimension is invalid.

        Notes:
            - For HNSW: Higher m and ef_construction improve accuracy but increase
              build time and memory usage.
            - For IVF-Flat: More lists improve accuracy but increase query time.
              Generally, use lists = sqrt(total_rows).

        Examples:
            >>> await store.create_index(dimension=384)
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        async with self._session_factory() as session:
            # Check if pgvector is available
            if not await self._check_pgvector_extension(session):
                error_msg = (
                    "pgvector extension is not installed. Install it with: CREATE EXTENSION vector;"
                )
                logger.error("pgvector_not_available")
                raise RuntimeError(error_msg)

            try:
                # Drop existing index if it exists
                await session.execute(text("DROP INDEX IF EXISTS idx_document_embeddings_vector"))

                # Create index based on type
                if self._config.index_type == "hnsw":
                    # HNSW index - better for most use cases
                    index_sql = text(
                        f"""
                        CREATE INDEX idx_document_embeddings_vector
                        ON document_embeddings
                        USING hnsw (embedding {self._get_distance_operator()})
                        WITH (m = :m, ef_construction = :ef_construction)
                        """
                    )
                    await session.execute(
                        index_sql,
                        {
                            "m": self._config.m,
                            "ef_construction": self._config.ef_construction,
                        },
                    )
                    logger.info(
                        "hnsw_index_created",
                        m=self._config.m,
                        ef_construction=self._config.ef_construction,
                    )
                else:
                    # IVF-Flat index
                    index_sql = text(
                        f"""
                        CREATE INDEX idx_document_embeddings_vector
                        ON document_embeddings
                        USING ivfflat (embedding {self._get_distance_operator()})
                        WITH (lists = :lists)
                        """
                    )
                    await session.execute(index_sql, {"lists": self._config.lists})
                    logger.info("ivfflat_index_created", lists=self._config.lists)

                await session.commit()

            except Exception as e:
                await session.rollback()
                logger.error("index_creation_failed", error=str(e))
                raise RuntimeError(f"Failed to create vector index: {e}") from e

    async def store(
        self,
        document_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a document with its embedding.

        Args:
            document_id: Unique identifier for the document.
            text: Original text content.
            embedding: Vector embedding of the text.
            metadata: Optional metadata for filtering and context.

        Raises:
            ValueError: If document_id, text, or embedding is invalid.
            RuntimeError: If storage fails.

        Examples:
            >>> await store.store(
            ...     document_id="doc_001",
            ...     text="Apple reports earnings",
            ...     embedding=[0.1, 0.2, ...],
            ...     metadata={"source": "reuters", "symbol": "AAPL"}
            ... )
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id cannot be empty")

        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        if not embedding:
            raise ValueError("embedding cannot be empty")

        async with self._session_factory() as session:
            try:
                # Create or update document embedding
                doc = DocumentEmbedding(
                    id=document_id,
                    text=text,
                    embedding=embedding,
                    metadata_=metadata or {},
                )

                # Use merge to handle upsert
                await session.merge(doc)
                await session.commit()

                logger.debug(
                    "document_stored",
                    document_id=document_id,
                    text_length=len(text),
                    embedding_dim=len(embedding),
                )

            except Exception as e:
                await session.rollback()
                logger.error(
                    "document_store_failed",
                    document_id=document_id,
                    error=str(e),
                )
                raise RuntimeError(f"Failed to store document {document_id}: {e}") from e

    async def store_batch(
        self,
        documents: list[tuple[str, str, list[float], dict[str, Any] | None]],
    ) -> None:
        """Store multiple documents in batch.

        This method is more efficient than calling store() multiple times
        as it uses a single transaction for all documents.

        Args:
            documents: List of tuples containing (document_id, text, embedding, metadata).

        Raises:
            ValueError: If documents list is empty or contains invalid entries.
            RuntimeError: If batch storage fails.

        Examples:
            >>> documents = [
            ...     ("doc_001", "Text 1", [0.1, ...], {"source": "reuters"}),
            ...     ("doc_002", "Text 2", [0.2, ...], {"source": "bloomberg"}),
            ... ]
            >>> await store.store_batch(documents)
        """
        if not documents:
            raise ValueError("documents list cannot be empty")

        async with self._session_factory() as session:
            try:
                for doc_id, text, embedding, metadata in documents:
                    if not doc_id or not doc_id.strip():
                        raise ValueError(f"Invalid document_id: {doc_id}")

                    if not text or not text.strip():
                        raise ValueError(f"Invalid text for document {doc_id}")

                    if not embedding:
                        raise ValueError(f"Invalid embedding for document {doc_id}")

                    doc = DocumentEmbedding(
                        id=doc_id,
                        text=text,
                        embedding=embedding,
                        metadata_=metadata or {},
                    )
                    await session.merge(doc)

                await session.commit()

                logger.info(
                    "batch_documents_stored",
                    count=len(documents),
                )

            except ValueError:
                await session.rollback()
                raise
            except Exception as e:
                await session.rollback()
                logger.error(
                    "batch_store_failed",
                    count=len(documents),
                    error=str(e),
                )
                raise RuntimeError(f"Failed to store batch: {e}") from e

    async def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar documents using vector similarity.

        This method performs a nearest neighbor search using the configured
        distance metric and returns the k most similar documents.

        Args:
            query_embedding: Vector embedding to search for.
            k: Number of results to return.
            filter_metadata: Optional metadata filters (AND conditions).

        Returns:
            List of VectorSearchResult objects, ordered by similarity (highest first).

        Raises:
            ValueError: If query_embedding or k is invalid.
            RuntimeError: If search fails.

        Examples:
            Basic search:

            >>> results = await store.search(
            ...     query_embedding=[0.1, 0.2, ...],
            ...     k=10
            ... )

            Search with metadata filtering:

            >>> results = await store.search(
            ...     query_embedding=[0.1, 0.2, ...],
            ...     k=5,
            ...     filter_metadata={"source": "reuters", "symbol": "AAPL"}
            ... )
        """
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        async with self._session_factory() as session:
            try:
                # Build the distance expression
                distance_op = self._get_distance_operator()

                # Build query with distance calculation
                # Use literal_column to create a labelable expression
                query_embedding_str = str(query_embedding)
                distance_expr: ColumnElement[float] = literal_column(
                    f"embedding {distance_op} CAST('{query_embedding_str}' AS vector)"
                )

                # Build base query
                stmt = select(
                    DocumentEmbedding.id,
                    DocumentEmbedding.text,
                    DocumentEmbedding.embedding,
                    DocumentEmbedding.metadata_,
                    distance_expr.label("distance"),
                )

                # Add metadata filters if provided
                if filter_metadata:
                    for key, value in filter_metadata.items():
                        # Use JSONB containment operator for filtering
                        stmt = stmt.where(
                            func.jsonb_extract_path_text(DocumentEmbedding.metadata_, key)
                            == str(value)
                        )

                # Order by distance and limit
                stmt = stmt.order_by(literal_column("distance")).limit(k)

                # Execute query
                result = await session.execute(stmt)
                rows = result.fetchall()

                # Convert to VectorSearchResult objects
                results = []
                for row in rows:
                    distance = float(row.distance)

                    # Calculate similarity based on distance metric
                    if self._config.distance_metric == "cosine":
                        # Cosine distance is 1 - cosine_similarity
                        similarity = 1.0 - distance
                    elif self._config.distance_metric == "l2":
                        # For L2, we use negative distance as similarity
                        # (closer = higher similarity)
                        similarity = -distance
                    else:  # inner_product
                        # Inner product is already negative, so negate it
                        similarity = -distance

                    results.append(
                        VectorSearchResult(
                            document_id=row.id,
                            text=row.text,
                            embedding=list(row.embedding),
                            distance=distance,
                            similarity=similarity,
                            metadata=row.metadata_,
                        )
                    )

                logger.debug(
                    "vector_search_completed",
                    k=k,
                    results_found=len(results),
                    has_filters=filter_metadata is not None,
                )

                return results

            except ValueError:
                raise
            except Exception as e:
                logger.error("vector_search_failed", error=str(e), k=k)
                raise RuntimeError(f"Vector search failed: {e}") from e

    async def delete(self, document_id: str) -> bool:
        """Delete a document by ID.

        Args:
            document_id: ID of the document to delete.

        Returns:
            True if document was deleted, False if not found.

        Raises:
            ValueError: If document_id is invalid.
            RuntimeError: If deletion fails.

        Examples:
            >>> deleted = await store.delete("doc_001")
            >>> print(f"Document deleted: {deleted}")
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id cannot be empty")

        async with self._session_factory() as session:
            try:
                stmt = delete(DocumentEmbedding).where(DocumentEmbedding.id == document_id)
                result = await session.execute(stmt)
                await session.commit()

                deleted: bool = result.rowcount > 0  # type: ignore[attr-defined]

                if deleted:
                    logger.debug("document_deleted", document_id=document_id)
                else:
                    logger.debug("document_not_found", document_id=document_id)

                return deleted

            except Exception as e:
                await session.rollback()
                logger.error(
                    "document_delete_failed",
                    document_id=document_id,
                    error=str(e),
                )
                raise RuntimeError(f"Failed to delete document {document_id}: {e}") from e

    async def get(self, document_id: str) -> VectorSearchResult | None:
        """Retrieve a document by ID.

        Args:
            document_id: ID of the document to retrieve.

        Returns:
            VectorSearchResult if found, None otherwise.

        Raises:
            ValueError: If document_id is invalid.
            RuntimeError: If retrieval fails.

        Examples:
            >>> doc = await store.get("doc_001")
            >>> if doc:
            ...     print(f"Found: {doc.text}")
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id cannot be empty")

        async with self._session_factory() as session:
            try:
                stmt = select(DocumentEmbedding).where(DocumentEmbedding.id == document_id)
                result = await session.execute(stmt)
                doc = result.scalar_one_or_none()

                if doc is None:
                    logger.debug("document_not_found", document_id=document_id)
                    return None

                # Create VectorSearchResult with distance=0 and similarity=1
                # since this is an exact match
                return VectorSearchResult(
                    document_id=doc.id,
                    text=doc.text,
                    embedding=list(doc.embedding),
                    distance=0.0,
                    similarity=1.0,
                    metadata=doc.metadata_,
                )

            except Exception as e:
                logger.error(
                    "document_get_failed",
                    document_id=document_id,
                    error=str(e),
                )
                raise RuntimeError(f"Failed to get document {document_id}: {e}") from e
