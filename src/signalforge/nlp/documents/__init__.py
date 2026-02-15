"""Document processing module for financial documents."""

from .classifier import DocumentClassifier
from .deduplication import DocumentDeduplicator
from .ingestion import DocumentIngestionPipeline
from .schemas import (
    ClassificationResult,
    DocumentSource,
    DocumentType,
    FinancialDocument,
    IngestionResult,
)
from .versioning import DocumentVersionManager

__all__ = [
    "ClassificationResult",
    "DocumentClassifier",
    "DocumentDeduplicator",
    "DocumentIngestionPipeline",
    "DocumentSource",
    "DocumentType",
    "DocumentVersionManager",
    "FinancialDocument",
    "IngestionResult",
]
