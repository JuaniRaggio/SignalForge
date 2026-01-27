"""SQLAlchemy model for document embeddings with pgvector support.

This module defines the DocumentEmbedding model for storing text documents
and their vector embeddings for similarity search using pgvector.

The model supports:
- Storage of text documents with unique IDs
- Vector embeddings with configurable dimensions
- Metadata as JSONB for flexible filtering
- Automatic timestamp tracking
- Vector similarity search through pgvector extension
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

try:
    from pgvector.sqlalchemy import Vector

    VECTOR_AVAILABLE = True
except ImportError:
    # Fallback if pgvector is not installed
    # This allows the model to be imported without pgvector
    VECTOR_AVAILABLE = False
    Vector = None

from signalforge.models.base import Base


class DocumentEmbedding(Base):
    """Document embedding storage with pgvector support.

    This model stores text documents along with their vector embeddings
    for efficient similarity search. It uses PostgreSQL's pgvector extension
    for native vector operations.

    Attributes:
        id: Unique document identifier (primary key).
        text: The original text content of the document.
        embedding: Dense vector representation of the text (384 dimensions).
        metadata: Additional metadata stored as JSONB for filtering.
        created_at: Timestamp when the document was created.
        updated_at: Timestamp when the document was last updated.

    Examples:
        Create a new document embedding:

        >>> doc = DocumentEmbedding(
        ...     id="doc_001",
        ...     text="Apple reports strong earnings",
        ...     embedding=[0.1, 0.2, ...],  # 384-dim vector
        ...     metadata={"source": "reuters", "symbol": "AAPL"}
        ... )

    Notes:
        - The embedding dimension is set to 384 to match the default
          sentence-transformer model (all-MiniLM-L6-v2).
        - For different models, the dimension can be changed in the migration.
        - The metadata field supports JSONB queries for efficient filtering.
        - If pgvector is not installed, the model uses ARRAY(Float) type.
    """

    __tablename__ = "document_embeddings"

    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        comment="Unique document identifier"
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Original text content of the document"
    )

    # Vector column for embeddings
    # Use pgvector Vector type if available, otherwise fall back to ARRAY
    # NOTE: This conditional is evaluated at class definition time
    _embedding_type = Vector(384) if (VECTOR_AVAILABLE and Vector is not None) else ARRAY(JSONB)
    embedding: Mapped[Any] = mapped_column(
        _embedding_type,
        nullable=False,
        comment="Dense vector embedding of the text"
    )

    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",  # Column name in database
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Additional metadata for filtering and context"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="now()",
        comment="Timestamp when document was created"
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default="now()",
        comment="Timestamp when document was last updated"
    )

    def __repr__(self) -> str:
        """String representation of the DocumentEmbedding."""
        return (
            f"DocumentEmbedding(id={self.id!r}, "
            f"text={self.text[:50]!r}..., "
            f"metadata={self.metadata_!r})"
        )
