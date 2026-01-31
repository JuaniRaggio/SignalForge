"""Tests for the vector store module.

This module tests the VectorStore class with mocked database sessions
to avoid requiring actual PostgreSQL and pgvector setup during testing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from signalforge.nlp.vector_store import (
    VectorSearchResult,
    VectorStore,
    VectorStoreConfig,
)


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VectorStoreConfig()

        assert config.index_type == "hnsw"
        assert config.lists == 100
        assert config.ef_construction == 64
        assert config.m == 16
        assert config.distance_metric == "cosine"

    def test_custom_config_hnsw(self) -> None:
        """Test custom HNSW configuration."""
        config = VectorStoreConfig(
            index_type="hnsw",
            ef_construction=128,
            m=32,
            distance_metric="l2",
        )

        assert config.index_type == "hnsw"
        assert config.ef_construction == 128
        assert config.m == 32
        assert config.distance_metric == "l2"

    def test_custom_config_ivfflat(self) -> None:
        """Test custom IVF-Flat configuration."""
        config = VectorStoreConfig(
            index_type="ivfflat",
            lists=200,
            distance_metric="inner_product",
        )

        assert config.index_type == "ivfflat"
        assert config.lists == 200
        assert config.distance_metric == "inner_product"

    def test_invalid_lists_raises_error(self) -> None:
        """Test that invalid lists parameter raises ValueError."""
        with pytest.raises(ValueError, match="lists must be positive"):
            VectorStoreConfig(lists=0)

        with pytest.raises(ValueError, match="lists must be positive"):
            VectorStoreConfig(lists=-10)

    def test_invalid_ef_construction_raises_error(self) -> None:
        """Test that invalid ef_construction raises ValueError."""
        with pytest.raises(ValueError, match="ef_construction must be positive"):
            VectorStoreConfig(ef_construction=0)

        with pytest.raises(ValueError, match="ef_construction must be positive"):
            VectorStoreConfig(ef_construction=-5)

    def test_invalid_m_raises_error(self) -> None:
        """Test that invalid m parameter raises ValueError."""
        with pytest.raises(ValueError, match="m must be positive"):
            VectorStoreConfig(m=0)

        with pytest.raises(ValueError, match="m must be positive"):
            VectorStoreConfig(m=-1)


class TestVectorSearchResult:
    """Tests for VectorSearchResult dataclass."""

    def test_valid_search_result(self) -> None:
        """Test creating a valid search result."""
        result = VectorSearchResult(
            document_id="doc_001",
            text="Test document",
            embedding=[0.1, 0.2, 0.3],
            distance=0.15,
            similarity=0.85,
            metadata={"source": "test"},
        )

        assert result.document_id == "doc_001"
        assert result.text == "Test document"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.distance == 0.15
        assert result.similarity == 0.85
        assert result.metadata == {"source": "test"}


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture
    def mock_session_factory(self) -> MagicMock:
        """Create a mock async session factory."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.merge = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_factory = MagicMock()
        mock_factory.return_value = mock_session

        return mock_factory

    @pytest.fixture
    def vector_store(self, mock_session_factory: MagicMock) -> VectorStore:
        """Create a VectorStore instance with mock session factory."""
        return VectorStore(mock_session_factory)

    def test_initialization_default_config(self, mock_session_factory: MagicMock) -> None:
        """Test VectorStore initialization with default config."""
        store = VectorStore(mock_session_factory)

        assert store._config.index_type == "hnsw"
        assert store._config.distance_metric == "cosine"

    def test_initialization_custom_config(self, mock_session_factory: MagicMock) -> None:
        """Test VectorStore initialization with custom config."""
        config = VectorStoreConfig(
            index_type="ivfflat",
            lists=200,
            distance_metric="l2",
        )
        store = VectorStore(mock_session_factory, config)

        assert store._config.index_type == "ivfflat"
        assert store._config.lists == 200
        assert store._config.distance_metric == "l2"

    def test_get_distance_operator_cosine(self, vector_store: VectorStore) -> None:
        """Test distance operator for cosine metric."""
        operator = vector_store._get_distance_operator()
        assert operator == "<=>"

    def test_get_distance_operator_l2(self, mock_session_factory: MagicMock) -> None:
        """Test distance operator for L2 metric."""
        config = VectorStoreConfig(distance_metric="l2")
        store = VectorStore(mock_session_factory, config)
        operator = store._get_distance_operator()
        assert operator == "<->"

    def test_get_distance_operator_inner_product(self, mock_session_factory: MagicMock) -> None:
        """Test distance operator for inner product metric."""
        config = VectorStoreConfig(distance_metric="inner_product")
        store = VectorStore(mock_session_factory, config)
        operator = store._get_distance_operator()
        assert operator == "<#>"

    @pytest.mark.asyncio
    async def test_check_pgvector_extension_available(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test pgvector extension check when available."""
        mock_session = await mock_session_factory().__aenter__()

        # Mock result with pgvector extension
        mock_result = MagicMock()
        mock_result.fetchone = Mock(return_value=("vector",))
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await vector_store._check_pgvector_extension(mock_session)

        assert result is True

    @pytest.mark.asyncio
    async def test_check_pgvector_extension_not_available(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test pgvector extension check when not available."""
        mock_session = await mock_session_factory().__aenter__()

        # Mock result without pgvector extension
        mock_result = MagicMock()
        mock_result.fetchone = Mock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await vector_store._check_pgvector_extension(mock_session)

        assert result is False

    @pytest.mark.asyncio
    async def test_create_index_hnsw(self, mock_session_factory: MagicMock) -> None:
        """Test creating HNSW index."""
        config = VectorStoreConfig(index_type="hnsw", m=16, ef_construction=64)
        store = VectorStore(mock_session_factory, config)

        mock_session = await mock_session_factory().__aenter__()

        # Mock pgvector extension as available
        with patch.object(store, "_check_pgvector_extension", return_value=True):
            await store.create_index(dimension=384)

        # Verify session methods were called
        mock_session.execute.assert_called()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_index_ivfflat(self, mock_session_factory: MagicMock) -> None:
        """Test creating IVF-Flat index."""
        config = VectorStoreConfig(index_type="ivfflat", lists=100)
        store = VectorStore(mock_session_factory, config)

        mock_session = await mock_session_factory().__aenter__()

        # Mock pgvector extension as available
        with patch.object(store, "_check_pgvector_extension", return_value=True):
            await store.create_index(dimension=384)

        # Verify session methods were called
        mock_session.execute.assert_called()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_index_pgvector_not_available(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test creating index when pgvector is not available."""
        _ = await mock_session_factory().__aenter__()  # Setup session

        # Mock pgvector extension as not available
        with (
            patch.object(vector_store, "_check_pgvector_extension", return_value=False),
            pytest.raises(RuntimeError, match="pgvector extension is not installed"),
        ):
            await vector_store.create_index(dimension=384)

    @pytest.mark.asyncio
    async def test_create_index_invalid_dimension(self, vector_store: VectorStore) -> None:
        """Test creating index with invalid dimension."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            await vector_store.create_index(dimension=0)

        with pytest.raises(ValueError, match="dimension must be positive"):
            await vector_store.create_index(dimension=-10)

    @pytest.mark.asyncio
    async def test_store_document(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test storing a document."""
        mock_session = await mock_session_factory().__aenter__()

        await vector_store.store(
            document_id="doc_001",
            text="Test document",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
        )

        # Verify merge and commit were called
        mock_session.merge.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_document_invalid_id(self, vector_store: VectorStore) -> None:
        """Test storing document with invalid ID."""
        with pytest.raises(ValueError, match="document_id cannot be empty"):
            await vector_store.store(
                document_id="",
                text="Test",
                embedding=[0.1, 0.2],
                metadata=None,
            )

        with pytest.raises(ValueError, match="document_id cannot be empty"):
            await vector_store.store(
                document_id="   ",
                text="Test",
                embedding=[0.1, 0.2],
                metadata=None,
            )

    @pytest.mark.asyncio
    async def test_store_document_invalid_text(self, vector_store: VectorStore) -> None:
        """Test storing document with invalid text."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            await vector_store.store(
                document_id="doc_001",
                text="",
                embedding=[0.1, 0.2],
                metadata=None,
            )

        with pytest.raises(ValueError, match="text cannot be empty"):
            await vector_store.store(
                document_id="doc_001",
                text="   ",
                embedding=[0.1, 0.2],
                metadata=None,
            )

    @pytest.mark.asyncio
    async def test_store_document_invalid_embedding(self, vector_store: VectorStore) -> None:
        """Test storing document with invalid embedding."""
        with pytest.raises(ValueError, match="embedding cannot be empty"):
            await vector_store.store(
                document_id="doc_001",
                text="Test",
                embedding=[],
                metadata=None,
            )

    @pytest.mark.asyncio
    async def test_store_batch(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test storing multiple documents in batch."""
        mock_session = await mock_session_factory().__aenter__()

        documents = [
            ("doc_001", "Text 1", [0.1, 0.2], {"source": "test1"}),
            ("doc_002", "Text 2", [0.3, 0.4], {"source": "test2"}),
            ("doc_003", "Text 3", [0.5, 0.6], None),
        ]

        await vector_store.store_batch(documents)

        # Verify merge was called for each document
        assert mock_session.merge.call_count == 3
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_batch_empty_list(self, vector_store: VectorStore) -> None:
        """Test storing empty batch raises error."""
        with pytest.raises(ValueError, match="documents list cannot be empty"):
            await vector_store.store_batch([])

    @pytest.mark.asyncio
    async def test_store_batch_invalid_document(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test storing batch with invalid document."""
        mock_session = await mock_session_factory().__aenter__()

        documents = [
            ("doc_001", "Text 1", [0.1, 0.2], None),
            ("", "Text 2", [0.3, 0.4], None),  # Invalid ID
        ]

        with pytest.raises(ValueError, match="Invalid document_id"):
            await vector_store.store_batch(documents)

        # Verify rollback was called
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_basic(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test basic vector search."""
        mock_session = await mock_session_factory().__aenter__()

        # Mock search results
        mock_row1 = Mock()
        mock_row1.id = "doc_001"
        mock_row1.text = "Test document 1"
        mock_row1.embedding = [0.1, 0.2, 0.3]
        mock_row1.metadata_ = {"source": "test1"}
        mock_row1.distance = 0.15

        mock_row2 = Mock()
        mock_row2.id = "doc_002"
        mock_row2.text = "Test document 2"
        mock_row2.embedding = [0.4, 0.5, 0.6]
        mock_row2.metadata_ = {"source": "test2"}
        mock_row2.distance = 0.25

        mock_result = MagicMock()
        mock_result.fetchall = Mock(return_value=[mock_row1, mock_row2])
        mock_session.execute = AsyncMock(return_value=mock_result)

        query_embedding = [0.1, 0.2, 0.3]
        results = await vector_store.search(query_embedding, k=2)

        assert len(results) == 2
        assert results[0].document_id == "doc_001"
        assert results[0].text == "Test document 1"
        assert results[0].distance == 0.15
        assert results[0].similarity == pytest.approx(0.85)  # 1 - 0.15 for cosine
        assert results[1].document_id == "doc_002"

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test search with metadata filtering."""
        mock_session = await mock_session_factory().__aenter__()

        # Mock empty results
        mock_result = MagicMock()
        mock_result.fetchall = Mock(return_value=[])
        mock_session.execute = AsyncMock(return_value=mock_result)

        query_embedding = [0.1, 0.2, 0.3]
        results = await vector_store.search(
            query_embedding,
            k=5,
            filter_metadata={"source": "reuters", "symbol": "AAPL"},
        )

        assert len(results) == 0
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_invalid_embedding(self, vector_store: VectorStore) -> None:
        """Test search with invalid query embedding."""
        with pytest.raises(ValueError, match="query_embedding cannot be empty"):
            await vector_store.search([], k=5)

    @pytest.mark.asyncio
    async def test_search_invalid_k(self, vector_store: VectorStore) -> None:
        """Test search with invalid k parameter."""
        with pytest.raises(ValueError, match="k must be positive"):
            await vector_store.search([0.1, 0.2], k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            await vector_store.search([0.1, 0.2], k=-5)

    @pytest.mark.asyncio
    async def test_delete_document(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test deleting a document."""
        mock_session = await mock_session_factory().__aenter__()

        # Mock successful deletion (1 row affected)
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        deleted = await vector_store.delete("doc_001")

        assert deleted is True
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_not_found(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test deleting a non-existent document."""
        mock_session = await mock_session_factory().__aenter__()

        # Mock unsuccessful deletion (0 rows affected)
        mock_result = AsyncMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        deleted = await vector_store.delete("nonexistent")

        assert deleted is False
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_invalid_id(self, vector_store: VectorStore) -> None:
        """Test deleting with invalid document ID."""
        with pytest.raises(ValueError, match="document_id cannot be empty"):
            await vector_store.delete("")

        with pytest.raises(ValueError, match="document_id cannot be empty"):
            await vector_store.delete("   ")

    @pytest.mark.asyncio
    async def test_get_document(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test retrieving a document by ID."""
        mock_session = await mock_session_factory().__aenter__()

        # Mock document
        mock_doc = Mock()
        mock_doc.id = "doc_001"
        mock_doc.text = "Test document"
        mock_doc.embedding = [0.1, 0.2, 0.3]
        mock_doc.metadata_ = {"source": "test"}

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = Mock(return_value=mock_doc)
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await vector_store.get("doc_001")

        assert result is not None
        assert result.document_id == "doc_001"
        assert result.text == "Test document"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.distance == 0.0
        assert result.similarity == 1.0
        assert result.metadata == {"source": "test"}

    @pytest.mark.asyncio
    async def test_get_document_not_found(
        self, vector_store: VectorStore, mock_session_factory: MagicMock
    ) -> None:
        """Test retrieving a non-existent document."""
        mock_session = await mock_session_factory().__aenter__()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = Mock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await vector_store.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_invalid_id(self, vector_store: VectorStore) -> None:
        """Test retrieving with invalid document ID."""
        with pytest.raises(ValueError, match="document_id cannot be empty"):
            await vector_store.get("")

        with pytest.raises(ValueError, match="document_id cannot be empty"):
            await vector_store.get("   ")

    @pytest.mark.asyncio
    async def test_search_l2_distance_similarity(self, mock_session_factory: MagicMock) -> None:
        """Test similarity calculation for L2 distance metric."""
        config = VectorStoreConfig(distance_metric="l2")
        store = VectorStore(mock_session_factory, config)

        mock_session = await mock_session_factory().__aenter__()

        # Mock search result with L2 distance
        mock_row = Mock()
        mock_row.id = "doc_001"
        mock_row.text = "Test"
        mock_row.embedding = [0.1, 0.2]
        mock_row.metadata_ = {}
        mock_row.distance = 0.5

        mock_result = MagicMock()
        mock_result.fetchall = Mock(return_value=[mock_row])
        mock_session.execute = AsyncMock(return_value=mock_result)

        results = await store.search([0.1, 0.2], k=1)

        # For L2, similarity is negative distance
        assert results[0].similarity == -0.5

    @pytest.mark.asyncio
    async def test_search_inner_product_similarity(self, mock_session_factory: MagicMock) -> None:
        """Test similarity calculation for inner product metric."""
        config = VectorStoreConfig(distance_metric="inner_product")
        store = VectorStore(mock_session_factory, config)

        mock_session = await mock_session_factory().__aenter__()

        # Mock search result with inner product distance
        mock_row = Mock()
        mock_row.id = "doc_001"
        mock_row.text = "Test"
        mock_row.embedding = [0.1, 0.2]
        mock_row.metadata_ = {}
        mock_row.distance = -0.8  # Negative inner product

        mock_result = MagicMock()
        mock_result.fetchall = Mock(return_value=[mock_row])
        mock_session.execute = AsyncMock(return_value=mock_result)

        results = await store.search([0.1, 0.2], k=1)

        # For inner product, negate the negative distance
        assert results[0].similarity == 0.8
