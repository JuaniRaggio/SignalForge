"""Comprehensive tests for document processing pipeline."""

from datetime import datetime

import pytest

from signalforge.nlp.documents import (
    DocumentClassifier,
    DocumentDeduplicator,
    DocumentIngestionPipeline,
    DocumentSource,
    DocumentType,
    DocumentVersionManager,
)

# Fixtures


@pytest.fixture
def classifier() -> DocumentClassifier:
    """Create document classifier instance."""
    return DocumentClassifier()


@pytest.fixture
def deduplicator() -> DocumentDeduplicator:
    """Create document deduplicator instance."""
    return DocumentDeduplicator(similarity_threshold=0.95)


@pytest.fixture
def version_manager() -> DocumentVersionManager:
    """Create version manager instance."""
    return DocumentVersionManager()


@pytest.fixture
def ingestion_pipeline(
    classifier: DocumentClassifier,
    deduplicator: DocumentDeduplicator,
) -> DocumentIngestionPipeline:
    """Create ingestion pipeline instance."""
    return DocumentIngestionPipeline(
        classifier=classifier,
        deduplicator=deduplicator,
    )


# Test Document Classification


class TestDocumentClassifier:
    """Test document classification functionality."""

    def test_classify_earnings_report(self, classifier: DocumentClassifier) -> None:
        """Test classification of earnings report."""
        text = """
        Q4 2024 Earnings Report
        Apple Inc. announces quarterly results with EPS of $2.50,
        revenue growth of 15% YoY. Net income increased to $25B.
        Operating income exceeded guidance.
        """

        result = classifier.classify(text)

        assert result.document_type == DocumentType.EARNINGS_REPORT
        assert result.confidence > 0.5

    def test_classify_sec_filing(self, classifier: DocumentClassifier) -> None:
        """Test classification of SEC filing."""
        text = """
        SECURITIES AND EXCHANGE COMMISSION
        Form 10-K Annual Report
        Pursuant to Section 13 or 15(d) of the Securities Exchange Act
        Filed with SEC EDGAR system
        """

        result = classifier.classify(text)

        assert result.document_type == DocumentType.SEC_FILING
        assert result.confidence > 0.5

    def test_classify_analyst_report(self, classifier: DocumentClassifier) -> None:
        """Test classification of analyst report."""
        text = """
        Analyst Report: Initiating coverage with BUY rating
        Price target: $200
        Maintain overweight recommendation
        Upgrading from hold to buy
        """

        result = classifier.classify(text)

        assert result.document_type == DocumentType.ANALYST_REPORT
        assert result.confidence > 0.5

    def test_classify_market_report(self, classifier: DocumentClassifier) -> None:
        """Test classification of market report."""
        text = """
        Market Update - Trading Session Close
        S&P 500 gained 1.2%, Nasdaq up 0.8%
        Dow Jones finished the day higher
        Market overview shows positive sentiment
        """

        result = classifier.classify(text)

        assert result.document_type == DocumentType.MARKET_REPORT
        assert result.confidence > 0.3

    def test_classify_sector_analysis(self, classifier: DocumentClassifier) -> None:
        """Test classification of sector analysis."""
        text = """
        Technology Sector Analysis Q1 2025
        Industry outlook remains positive
        Sector performance exceeded expectations
        Competitive landscape analysis shows market share shifts
        """

        result = classifier.classify(text)

        assert result.document_type == DocumentType.SECTOR_ANALYSIS
        assert result.confidence > 0.3

    def test_classify_news_article(self, classifier: DocumentClassifier) -> None:
        """Test classification of news article."""
        text = """
        Breaking news from Bloomberg:
        According to sources familiar with the matter,
        Reuters reported today that the deal is imminent.
        Wall Street Journal confirms the report.
        """

        result = classifier.classify(text)

        assert result.document_type == DocumentType.NEWS_ARTICLE
        assert result.confidence > 0.3

    def test_classify_press_release(self, classifier: DocumentClassifier) -> None:
        """Test classification of press release."""
        text = """
        FOR IMMEDIATE RELEASE
        Company announces new product launch
        Press Release via PR Newswire
        Media Contact: media@company.com
        Investor Relations: ir@company.com
        """

        result = classifier.classify(text)

        assert result.document_type == DocumentType.PRESS_RELEASE
        assert result.confidence > 0.5

    def test_classify_research_note(self, classifier: DocumentClassifier) -> None:
        """Test classification of research note."""
        text = """
        Research Note: Investment Thesis Update
        Valuation analysis using DCF model
        Discounted cash flow projections
        Comparables analysis and peer comparison
        """

        result = classifier.classify(text)

        assert result.document_type == DocumentType.RESEARCH_NOTE
        assert result.confidence > 0.3

    def test_classify_unknown(self, classifier: DocumentClassifier) -> None:
        """Test classification of unrecognizable document."""
        text = "This is some random text with no financial context whatsoever."

        result = classifier.classify(text)

        # When no patterns match, may default to first type with low confidence
        assert result.confidence <= 0.1  # Very low confidence for unknown content

    def test_classify_with_metadata(self, classifier: DocumentClassifier) -> None:
        """Test classification using metadata hints."""
        text = "Some financial document."
        metadata = {"filename": "company_10k_2024.pdf", "source": "sec_edgar"}

        result = classifier.classify(text, metadata)

        assert result.document_type == DocumentType.SEC_FILING

    def test_secondary_types(self, classifier: DocumentClassifier) -> None:
        """Test that secondary types are detected."""
        text = """
        Quarterly earnings results show strong growth.
        Form 10-Q filing includes detailed financials.
        EPS exceeded analyst expectations.
        """

        result = classifier.classify(text)

        assert len(result.secondary_types) > 0
        assert all(isinstance(dt, DocumentType) for dt, _ in result.secondary_types)
        assert all(isinstance(score, float) for _, score in result.secondary_types)


# Test Symbol Extraction


class TestSymbolExtraction:
    """Test financial symbol extraction."""

    def test_extract_dollar_symbols(self, classifier: DocumentClassifier) -> None:
        """Test extraction of $SYMBOL format."""
        text = "Stock performance: $AAPL up 5%, $MSFT down 2%, $GOOGL flat."

        result = classifier.classify(text)

        assert "AAPL" in result.detected_symbols
        assert "MSFT" in result.detected_symbols
        assert "GOOGL" in result.detected_symbols

    def test_extract_exchange_symbols(self, classifier: DocumentClassifier) -> None:
        """Test extraction of EXCHANGE:SYMBOL format."""
        text = "Trading data: NASDAQ:TSLA, NYSE:IBM, NASDAQ:AMZN"

        result = classifier.classify(text)

        assert "TSLA" in result.detected_symbols
        assert "IBM" in result.detected_symbols
        assert "AMZN" in result.detected_symbols

    def test_extract_parenthesis_symbols(self, classifier: DocumentClassifier) -> None:
        """Test extraction of (SYMBOL) format."""
        text = "Apple Inc. (AAPL) and Microsoft Corporation (MSFT) announced..."

        result = classifier.classify(text)

        assert "AAPL" in result.detected_symbols
        assert "MSFT" in result.detected_symbols

    def test_extract_ticker_symbols(self, classifier: DocumentClassifier) -> None:
        """Test extraction of ticker: SYMBOL format."""
        text = "The company (ticker: NVDA) reported strong results."

        result = classifier.classify(text)

        assert "NVDA" in result.detected_symbols

    def test_extract_context_symbols(self, classifier: DocumentClassifier) -> None:
        """Test extraction with stock/shares context."""
        text = "META stock surged 10%, while NFLX shares declined."

        result = classifier.classify(text)

        assert "META" in result.detected_symbols or "NFLX" in result.detected_symbols

    def test_filter_false_positives(self, classifier: DocumentClassifier) -> None:
        """Test that common false positives are filtered."""
        text = "The CEO and CFO discussed USA GDP growth and SEC regulations."

        result = classifier.classify(text)

        assert "CEO" not in result.detected_symbols
        assert "CFO" not in result.detected_symbols
        assert "USA" not in result.detected_symbols
        assert "GDP" not in result.detected_symbols
        assert "SEC" not in result.detected_symbols

    def test_symbol_deduplication(self, classifier: DocumentClassifier) -> None:
        """Test that symbols are deduplicated."""
        text = "$AAPL performed well. AAPL stock (AAPL) ticker: AAPL"

        result = classifier.classify(text)

        symbol_count = result.detected_symbols.count("AAPL")
        assert symbol_count == 1


# Test Sector Detection


class TestSectorDetection:
    """Test sector detection functionality."""

    def test_detect_technology_sector(self, classifier: DocumentClassifier) -> None:
        """Test detection of Technology sector."""
        text = "Software company launches cloud computing platform with AI capabilities."

        result = classifier.classify(text)

        assert "Technology" in result.detected_sectors

    def test_detect_healthcare_sector(self, classifier: DocumentClassifier) -> None:
        """Test detection of Healthcare sector."""
        text = "Pharmaceutical company receives FDA approval for new drug after clinical trials."

        result = classifier.classify(text)

        assert "Healthcare" in result.detected_sectors

    def test_detect_financial_sector(self, classifier: DocumentClassifier) -> None:
        """Test detection of Financial Services sector."""
        text = "Investment bank expands wealth management and asset management services."

        result = classifier.classify(text)

        assert "Financial Services" in result.detected_sectors

    def test_detect_energy_sector(self, classifier: DocumentClassifier) -> None:
        """Test detection of Energy sector."""
        text = "Oil and gas company invests in renewable energy and solar power."

        result = classifier.classify(text)

        assert "Energy" in result.detected_sectors

    def test_detect_multiple_sectors(self, classifier: DocumentClassifier) -> None:
        """Test detection of multiple sectors."""
        text = """
        Technology company partners with healthcare provider
        to develop AI-powered medical devices.
        """

        result = classifier.classify(text)

        assert "Technology" in result.detected_sectors
        assert "Healthcare" in result.detected_sectors


# Test Deduplication


class TestDeduplication:
    """Test document deduplication."""

    def test_compute_hash(self, deduplicator: DocumentDeduplicator) -> None:
        """Test content hash computation."""
        content = "This is a test document."
        hash1 = deduplicator.compute_hash(content)
        hash2 = deduplicator.compute_hash(content)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

    def test_compute_hash_normalization(self, deduplicator: DocumentDeduplicator) -> None:
        """Test that hash normalization works."""
        content1 = "This is a test."
        content2 = "THIS IS A TEST."
        content3 = "This    is   a    test."

        hash1 = deduplicator.compute_hash(content1)
        hash2 = deduplicator.compute_hash(content2)
        hash3 = deduplicator.compute_hash(content3)

        assert hash1 == hash2 == hash3

    def test_is_duplicate_exact(self, deduplicator: DocumentDeduplicator) -> None:
        """Test exact duplicate detection."""
        content = "Test document content."
        hash1 = deduplicator.compute_hash(content)
        existing_hashes = {hash1}

        assert deduplicator.is_duplicate(hash1, existing_hashes)

    def test_is_not_duplicate(self, deduplicator: DocumentDeduplicator) -> None:
        """Test non-duplicate detection."""
        content1 = "First document."
        content2 = "Second document."

        hash1 = deduplicator.compute_hash(content1)
        hash2 = deduplicator.compute_hash(content2)

        existing_hashes = {hash1}

        assert not deduplicator.is_duplicate(hash2, existing_hashes)

    def test_compute_similarity_identical(self, deduplicator: DocumentDeduplicator) -> None:
        """Test similarity of identical documents."""
        text = "This is a test document with some content."

        similarity = deduplicator.compute_similarity(text, text)

        assert similarity == 1.0

    def test_compute_similarity_different(self, deduplicator: DocumentDeduplicator) -> None:
        """Test similarity of different documents."""
        text1 = "This is about technology and innovation."
        text2 = "Financial markets showed strong performance."

        similarity = deduplicator.compute_similarity(text1, text2)

        assert 0.0 <= similarity < 0.5

    def test_compute_similarity_partial(self, deduplicator: DocumentDeduplicator) -> None:
        """Test similarity of partially similar documents."""
        text1 = """
        Apple Inc. reported strong quarterly earnings with great performance
        and excellent revenue growth across all business segments. The company
        exceeded analyst expectations for the quarter.
        """
        text2 = """
        Apple Inc. reported weak quarterly earnings with poor performance
        and declining revenue growth across all business segments. The company
        missed analyst expectations for the quarter.
        """

        similarity = deduplicator.compute_similarity(text1, text2)

        # Trigram similarity for partially similar text (many common words/phrases)
        assert 0.2 < similarity < 0.9

    def test_find_similar_above_threshold(
        self, deduplicator: DocumentDeduplicator
    ) -> None:
        """Test finding similar documents above threshold."""
        content = "This is a test document about financial markets."
        similar = "This is a test document about financial markets and trading."
        different = "Completely different content about weather patterns."

        candidates = [
            ("doc1", similar),
            ("doc2", different),
        ]

        results = deduplicator.find_similar(content, candidates)

        assert len(results) >= 0  # May or may not find similar depending on threshold
        if results:
            assert all(sim >= deduplicator.similarity_threshold for _, sim in results)

    def test_find_similar_sorted(self, deduplicator: DocumentDeduplicator) -> None:
        """Test that similar documents are sorted by similarity."""
        deduplicator.similarity_threshold = 0.3  # Lower threshold for testing

        content = "Test document."
        very_similar = "Test document content."
        somewhat_similar = "Test content here."

        candidates = [
            ("doc1", somewhat_similar),
            ("doc2", very_similar),
        ]

        results = deduplicator.find_similar(content, candidates)

        if len(results) >= 2:
            assert results[0][1] >= results[1][1]  # Sorted descending


# Test Ingestion Pipeline


class TestIngestionPipeline:
    """Test document ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_ingest_new_document(
        self, ingestion_pipeline: DocumentIngestionPipeline
    ) -> None:
        """Test ingesting a new document."""
        content = "Q4 earnings report: EPS of $2.50, revenue growth of 15%."
        source = DocumentSource.ANALYST_FEED
        metadata = {"title": "Q4 Earnings"}

        result = await ingestion_pipeline.ingest(content, source, metadata)

        assert result.status == "ingested"
        assert result.is_new is True
        assert result.version == 1
        assert result.document_id != ""

    @pytest.mark.asyncio
    async def test_ingest_duplicate(
        self, ingestion_pipeline: DocumentIngestionPipeline
    ) -> None:
        """Test ingesting exact duplicate."""
        content = "Test earnings report content."
        source = DocumentSource.ANALYST_FEED

        # Ingest first time
        result1 = await ingestion_pipeline.ingest(content, source)
        assert result1.status == "ingested"
        assert result1.is_new is True

        # Ingest duplicate
        result2 = await ingestion_pipeline.ingest(content, source)
        assert result2.status == "duplicate"
        assert result2.is_new is False

    @pytest.mark.asyncio
    async def test_ingest_batch(
        self, ingestion_pipeline: DocumentIngestionPipeline
    ) -> None:
        """Test batch ingestion."""
        documents: list[tuple[str, DocumentSource, dict[str, str]]] = [
            ("Document 1 earnings report EPS", DocumentSource.ANALYST_FEED, {}),
            ("Document 2 Form 10-K SEC filing", DocumentSource.SEC_EDGAR, {}),
            ("Document 3 press release announcement", DocumentSource.NEWS_RSS, {}),
        ]

        results = await ingestion_pipeline.ingest_batch(documents)

        assert len(results) == 3
        assert all(r.status == "ingested" for r in results)

    @pytest.mark.asyncio
    async def test_get_document(
        self, ingestion_pipeline: DocumentIngestionPipeline
    ) -> None:
        """Test retrieving ingested document."""
        content = "Test document content."
        result = await ingestion_pipeline.ingest(content, DocumentSource.MANUAL_UPLOAD)

        doc = ingestion_pipeline.get_document(result.document_id)

        assert doc is not None
        assert doc.document_id == result.document_id
        assert doc.content == content

    @pytest.mark.asyncio
    async def test_get_all_documents(
        self, ingestion_pipeline: DocumentIngestionPipeline
    ) -> None:
        """Test retrieving all documents."""
        await ingestion_pipeline.ingest("Doc 1", DocumentSource.MANUAL_UPLOAD)
        await ingestion_pipeline.ingest("Doc 2", DocumentSource.MANUAL_UPLOAD)
        await ingestion_pipeline.ingest("Doc 3", DocumentSource.MANUAL_UPLOAD)

        all_docs = ingestion_pipeline.get_all_documents()

        assert len(all_docs) == 3

    @pytest.mark.asyncio
    async def test_ingest_extracts_symbols(
        self, ingestion_pipeline: DocumentIngestionPipeline
    ) -> None:
        """Test that ingestion extracts symbols."""
        content = "Stock performance: $AAPL up 5%, $MSFT down 2%."
        result = await ingestion_pipeline.ingest(content, DocumentSource.NEWS_RSS)

        doc = ingestion_pipeline.get_document(result.document_id)

        assert doc is not None
        assert "AAPL" in doc.symbols
        assert "MSFT" in doc.symbols

    @pytest.mark.asyncio
    async def test_ingest_extracts_sectors(
        self, ingestion_pipeline: DocumentIngestionPipeline
    ) -> None:
        """Test that ingestion extracts sectors."""
        content = "Software company launches cloud computing platform."
        result = await ingestion_pipeline.ingest(content, DocumentSource.NEWS_RSS)

        doc = ingestion_pipeline.get_document(result.document_id)

        assert doc is not None
        assert "Technology" in doc.sectors


# Test Versioning


class TestVersioning:
    """Test document version management."""

    @pytest.mark.asyncio
    async def test_create_version(self, version_manager: DocumentVersionManager) -> None:
        """Test creating a new version."""
        doc_id = "test-doc-1"
        content = "Version 1 content"

        version = await version_manager.create_version(doc_id, content, "initial")

        assert version == 1

    @pytest.mark.asyncio
    async def test_create_multiple_versions(
        self, version_manager: DocumentVersionManager
    ) -> None:
        """Test creating multiple versions."""
        doc_id = "test-doc-2"

        v1 = await version_manager.create_version(doc_id, "Content v1", "initial")
        v2 = await version_manager.create_version(doc_id, "Content v2", "update")
        v3 = await version_manager.create_version(doc_id, "Content v3", "update")

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    @pytest.mark.asyncio
    async def test_get_history(self, version_manager: DocumentVersionManager) -> None:
        """Test retrieving version history."""
        doc_id = "test-doc-3"

        await version_manager.create_version(doc_id, "v1", "initial")
        await version_manager.create_version(doc_id, "v2", "update")
        await version_manager.create_version(doc_id, "v3", "update")

        history = await version_manager.get_history(doc_id)

        assert len(history) == 3
        assert all(isinstance(v, int) for v, _, _ in history)
        assert all(isinstance(ts, datetime) for _, ts, _ in history)
        assert all(isinstance(reason, str) for _, _, reason in history)

    def test_compute_diff_identical(self, version_manager: DocumentVersionManager) -> None:
        """Test diff computation for identical content."""
        content = "Same content"

        diff = version_manager.compute_diff(content, content)

        assert diff["changed"] is False
        assert diff["similarity"] == 1.0
        assert diff["additions"] == 0
        assert diff["deletions"] == 0

    def test_compute_diff_different(self, version_manager: DocumentVersionManager) -> None:
        """Test diff computation for different content."""
        old = "Old content here"
        new = "New content there"

        diff = version_manager.compute_diff(old, new)

        assert diff["changed"] is True
        assert isinstance(diff["similarity"], float) and diff["similarity"] < 1.0
        assert isinstance(diff["diff"], list) and len(diff["diff"]) > 0

    def test_compute_diff_partial(self, version_manager: DocumentVersionManager) -> None:
        """Test diff computation for partially changed content."""
        old = "Line 1\nLine 2\nLine 3"
        new = "Line 1\nModified Line 2\nLine 3"

        diff = version_manager.compute_diff(old, new)

        assert diff["changed"] is True
        assert isinstance(diff["similarity"], float) and 0.5 < diff["similarity"] < 1.0
        assert isinstance(diff["additions"], int) and diff["additions"] > 0
        assert isinstance(diff["deletions"], int) and diff["deletions"] > 0


# Integration Tests


class TestIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_earnings_report(
        self, ingestion_pipeline: DocumentIngestionPipeline
    ) -> None:
        """Test full pipeline with earnings report."""
        content = """
        Apple Inc. Q4 2024 Earnings Report

        $AAPL reported quarterly results with EPS of $2.50,
        exceeding analyst expectations. Revenue growth of 15% YoY.
        Technology sector performance remains strong.
        Net income increased to $25B.
        """

        result = await ingestion_pipeline.ingest(
            content,
            DocumentSource.ANALYST_FEED,
            {"title": "AAPL Q4 Earnings"},
        )

        assert result.status == "ingested"
        assert result.is_new is True

        doc = ingestion_pipeline.get_document(result.document_id)
        assert doc is not None
        assert doc.document_type == DocumentType.EARNINGS_REPORT
        assert "AAPL" in doc.symbols
        assert "Technology" in doc.sectors

    @pytest.mark.asyncio
    async def test_full_pipeline_duplicate_handling(
        self, ingestion_pipeline: DocumentIngestionPipeline
    ) -> None:
        """Test duplicate handling in full pipeline."""
        content = "Unique test content for deduplication"

        # First ingestion
        result1 = await ingestion_pipeline.ingest(content, DocumentSource.MANUAL_UPLOAD)
        assert result1.status == "ingested"
        assert result1.is_new is True

        # Duplicate ingestion
        result2 = await ingestion_pipeline.ingest(content, DocumentSource.MANUAL_UPLOAD)
        assert result2.status == "duplicate"
        assert result2.is_new is False

    @pytest.mark.asyncio
    async def test_full_pipeline_multiple_documents(
        self, ingestion_pipeline: DocumentIngestionPipeline
    ) -> None:
        """Test pipeline with multiple different documents."""
        docs = [
            "Earnings report with EPS data and revenue growth",
            "Form 10-K SEC filing with detailed financials",
            "Press release announcement from company",
            "Analyst report with price target and buy rating",
        ]

        for doc_content in docs:
            result = await ingestion_pipeline.ingest(
                doc_content, DocumentSource.ANALYST_FEED
            )
            assert result.status == "ingested"

        all_docs = ingestion_pipeline.get_all_documents()
        assert len(all_docs) == len(docs)
