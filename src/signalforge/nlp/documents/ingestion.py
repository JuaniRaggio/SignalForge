"""Document ingestion pipeline."""

import uuid
from datetime import UTC, datetime

import httpx
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .classifier import DocumentClassifier
from .deduplication import DocumentDeduplicator
from .schemas import DocumentSource, FinancialDocument, IngestionResult
from .versioning import DocumentVersionManager

logger = structlog.get_logger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for ingesting financial documents."""

    def __init__(
        self,
        classifier: DocumentClassifier,
        deduplicator: DocumentDeduplicator,
        session: AsyncSession | None = None,
    ) -> None:
        """
        Initialize ingestion pipeline.

        Args:
            classifier: Document classifier instance
            deduplicator: Document deduplicator instance
            session: Optional database session for persistence
        """
        self.classifier = classifier
        self.deduplicator = deduplicator
        self.session = session
        self.version_manager = DocumentVersionManager(session)

        # In-memory storage for demo (when no session provided)
        self._documents: dict[str, FinancialDocument] = {}
        self._content_hashes: set[str] = set()

    async def ingest(
        self,
        content: str,
        source: DocumentSource,
        metadata: dict[str, str] | None = None,
    ) -> IngestionResult:
        """
        Ingest a document.

        Steps:
        1. Classify document type
        2. Check for duplicates
        3. Extract symbols and sectors
        4. Store in database
        5. Return ingestion result

        Args:
            content: Document content
            source: Document source
            metadata: Optional metadata

        Returns:
            IngestionResult with status and details
        """
        metadata = metadata or {}

        try:
            # Step 1: Classify document
            classification = self.classifier.classify(content, metadata)

            logger.info(
                "classified_document",
                document_type=classification.document_type.value,
                confidence=classification.confidence,
            )

            # Step 2: Check for duplicates
            content_hash = self.deduplicator.compute_hash(content)

            if self.deduplicator.is_duplicate(content_hash, self._content_hashes):
                logger.info("duplicate_document_detected", content_hash=content_hash)
                return IngestionResult(
                    document_id="",
                    status="duplicate",
                    is_new=False,
                    version=0,
                    message="Exact duplicate detected",
                )

            # Check for near-duplicates
            candidates = [
                (doc_id, doc.content) for doc_id, doc in self._documents.items()
            ]
            similar_docs = self.deduplicator.find_similar(content, candidates)

            if similar_docs:
                # Found similar document - create new version
                doc_id, similarity = similar_docs[0]
                existing_doc = self._documents[doc_id]

                new_version = await self.version_manager.create_version(
                    document_id=doc_id,
                    new_content=content,
                    reason=f"Similar content update (similarity: {similarity:.2f})",
                )

                # Update document
                updated_doc = FinancialDocument(
                    document_id=doc_id,
                    document_type=classification.document_type,
                    source=source,
                    title=metadata.get("title", existing_doc.title),
                    content=content,
                    symbols=classification.detected_symbols,
                    sectors=classification.detected_sectors,
                    published_at=existing_doc.published_at,
                    ingested_at=datetime.now(UTC),
                    version=new_version,
                    content_hash=content_hash,
                    metadata=metadata,
                )

                self._documents[doc_id] = updated_doc
                self._content_hashes.add(content_hash)

                logger.info(
                    "document_updated",
                    document_id=doc_id,
                    version=new_version,
                    similarity=similarity,
                )

                return IngestionResult(
                    document_id=doc_id,
                    status="ingested",
                    is_new=False,
                    version=new_version,
                    message=f"Updated version (similarity: {similarity:.2f})",
                )

            # Step 3: Create new document
            doc_id = str(uuid.uuid4())
            published_at = self._parse_published_date(metadata)

            document = FinancialDocument(
                document_id=doc_id,
                document_type=classification.document_type,
                source=source,
                title=metadata.get("title", "Untitled"),
                content=content,
                symbols=classification.detected_symbols,
                sectors=classification.detected_sectors,
                published_at=published_at,
                ingested_at=datetime.now(UTC),
                version=1,
                content_hash=content_hash,
                metadata=metadata,
            )

            # Step 4: Store document
            self._documents[doc_id] = document
            self._content_hashes.add(content_hash)

            # Create initial version
            await self.version_manager.create_version(
                document_id=doc_id,
                new_content=content,
                reason="initial_version",
            )

            logger.info(
                "document_ingested",
                document_id=doc_id,
                document_type=classification.document_type.value,
                symbols_count=len(classification.detected_symbols),
                sectors_count=len(classification.detected_sectors),
            )

            return IngestionResult(
                document_id=doc_id,
                status="ingested",
                is_new=True,
                version=1,
                message="Successfully ingested new document",
            )

        except Exception as e:
            logger.error("ingestion_error", error=str(e), exc_info=True)
            return IngestionResult(
                document_id="",
                status="error",
                is_new=False,
                version=0,
                message=f"Ingestion failed: {str(e)}",
            )

    async def ingest_batch(
        self,
        documents: list[tuple[str, DocumentSource, dict[str, str]]],
    ) -> list[IngestionResult]:
        """
        Ingest multiple documents.

        Args:
            documents: List of (content, source, metadata) tuples

        Returns:
            List of IngestionResults
        """
        results = []

        for content, source, metadata in documents:
            result = await self.ingest(content, source, metadata)
            results.append(result)

        logger.info(
            "batch_ingestion_complete",
            total=len(documents),
            ingested=sum(1 for r in results if r.status == "ingested"),
            duplicates=sum(1 for r in results if r.status == "duplicate"),
            errors=sum(1 for r in results if r.status == "error"),
        )

        return results

    async def ingest_from_url(
        self, url: str, source: DocumentSource
    ) -> IngestionResult:
        """
        Fetch and ingest document from URL.

        Args:
            url: URL to fetch document from
            source: Document source

        Returns:
            IngestionResult
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                content = response.text

                # Extract metadata from response
                metadata = {
                    "url": url,
                    "content_type": response.headers.get("content-type", ""),
                }

                # Try to extract title from HTML if present
                if "html" in metadata["content_type"]:
                    import re

                    title_match = re.search(
                        r"<title>(.*?)</title>", content, re.IGNORECASE
                    )
                    if title_match:
                        metadata["title"] = title_match.group(1)

                logger.info("fetched_document_from_url", url=url, size=len(content))

                return await self.ingest(content, source, metadata)

        except httpx.HTTPError as e:
            logger.error("url_fetch_error", url=url, error=str(e))
            return IngestionResult(
                document_id="",
                status="error",
                is_new=False,
                version=0,
                message=f"Failed to fetch from URL: {str(e)}",
            )

    def _parse_published_date(self, metadata: dict[str, str]) -> datetime:
        """
        Parse published date from metadata.

        Args:
            metadata: Document metadata

        Returns:
            Parsed datetime or current time if not found
        """
        published_str = metadata.get("published_at") or metadata.get("date")

        if published_str:
            try:
                # Try parsing ISO format
                return datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Default to current time
        return datetime.now(UTC)

    def get_document(self, document_id: str) -> FinancialDocument | None:
        """
        Retrieve document by ID.

        Args:
            document_id: Document identifier

        Returns:
            FinancialDocument if found, None otherwise
        """
        return self._documents.get(document_id)

    def get_all_documents(self) -> list[FinancialDocument]:
        """
        Retrieve all documents.

        Returns:
            List of all documents
        """
        return list(self._documents.values())
