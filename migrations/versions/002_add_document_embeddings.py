"""Add document_embeddings table with pgvector support.

This migration:
1. Enables the pgvector extension for vector operations
2. Creates the document_embeddings table for storing text and embeddings
3. Creates a HNSW index for efficient similarity search

Revision ID: 002_add_document_embeddings
Revises: 001_initial
Create Date: 2026-01-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002_add_document_embeddings"
down_revision: str = "001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Default embedding dimension for all-MiniLM-L6-v2 model
EMBEDDING_DIMENSION = 384


def upgrade() -> None:
    """Create document_embeddings table and vector index."""
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create document_embeddings table
    op.create_table(
        "document_embeddings",
        sa.Column(
            "id",
            sa.String(255),
            nullable=False,
            comment="Unique document identifier"
        ),
        sa.Column(
            "text",
            sa.Text(),
            nullable=False,
            comment="Original text content of the document"
        ),
        sa.Column(
            "embedding",
            postgresql.ARRAY(sa.Float),  # Will be cast to vector type
            nullable=False,
            comment="Dense vector embedding of the text"
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default="{}",
            comment="Additional metadata for filtering and context"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="Timestamp when document was created"
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="Timestamp when document was last updated"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_embeddings"),
    )

    # Cast the embedding column to vector type with the correct dimension
    op.execute(
        f"""
        ALTER TABLE document_embeddings
        ALTER COLUMN embedding TYPE vector({EMBEDDING_DIMENSION})
        USING embedding::vector({EMBEDDING_DIMENSION})
        """
    )

    # Create indexes
    # Index on created_at for time-based queries
    op.create_index(
        "ix_document_embeddings_created_at",
        "document_embeddings",
        ["created_at"]
    )

    # GIN index on metadata for efficient JSONB queries
    op.execute(
        """
        CREATE INDEX ix_document_embeddings_metadata
        ON document_embeddings
        USING GIN (metadata)
        """
    )

    # Create HNSW index for vector similarity search
    # Using cosine distance (<=>), m=16, ef_construction=64 (good defaults)
    op.execute(
        """
        CREATE INDEX idx_document_embeddings_vector
        ON document_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    """Drop document_embeddings table and disable pgvector extension."""
    # Drop table (indexes will be dropped automatically)
    op.drop_table("document_embeddings")

    # Note: We don't drop the vector extension as other tables might use it
    # If you want to drop it, uncomment the following line:
    # op.execute("DROP EXTENSION IF EXISTS vector")
