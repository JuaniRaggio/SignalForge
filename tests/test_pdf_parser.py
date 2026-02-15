"""Tests for PDF parser module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from signalforge.nlp.pdf_parser import (
    DocumentQuality,
    DocumentQualityScorer,
    DocumentSection,
    DocumentStructure,
    ExtractedTable,
    LayoutAnalyzer,
    OCRFallback,
    TableExtractor,
)


class TestDocumentSection:
    """Tests for DocumentSection schema."""

    def test_create_document_section(self) -> None:
        """Test creating a document section."""
        section = DocumentSection(
            section_type="paragraph",
            content="Test content",
            page_number=1,
        )

        assert section.section_type == "paragraph"
        assert section.content == "Test content"
        assert section.page_number == 1
        assert section.bounding_box is None
        assert section.confidence == 1.0

    def test_document_section_with_bbox(self) -> None:
        """Test document section with bounding box."""
        section = DocumentSection(
            section_type="header",
            content="Test Header",
            page_number=1,
            bounding_box=(10.0, 20.0, 100.0, 50.0),
            confidence=0.95,
        )

        assert section.bounding_box == (10.0, 20.0, 100.0, 50.0)
        assert section.confidence == 0.95

    def test_document_section_validation(self) -> None:
        """Test document section validation."""
        with pytest.raises(ValueError):
            DocumentSection(
                section_type="header",
                content="Test",
                page_number=1,
                confidence=1.5,  # Invalid confidence > 1
            )


class TestExtractedTable:
    """Tests for ExtractedTable schema."""

    def test_create_extracted_table(self) -> None:
        """Test creating an extracted table."""
        table = ExtractedTable(
            headers=["Column 1", "Column 2"],
            rows=[["A", "B"], ["C", "D"]],
            page_number=1,
        )

        assert table.headers == ["Column 1", "Column 2"]
        assert len(table.rows) == 2
        assert table.page_number == 1
        assert table.table_title is None

    def test_extracted_table_with_title(self) -> None:
        """Test extracted table with title."""
        table = ExtractedTable(
            headers=["Q1", "Q2"],
            rows=[["100", "200"]],
            page_number=2,
            table_title="Quarterly Revenue",
        )

        assert table.table_title == "Quarterly Revenue"


class TestDocumentStructure:
    """Tests for DocumentStructure schema."""

    def test_create_document_structure(self) -> None:
        """Test creating a document structure."""
        structure = DocumentStructure(
            title="Test Document",
            sections=[],
            tables=[],
            page_count=5,
            metadata={"author": "Test Author"},
        )

        assert structure.title == "Test Document"
        assert len(structure.sections) == 0
        assert len(structure.tables) == 0
        assert structure.page_count == 5
        assert structure.metadata["author"] == "Test Author"

    def test_document_structure_defaults(self) -> None:
        """Test document structure with defaults."""
        structure = DocumentStructure(page_count=1)

        assert structure.title is None
        assert structure.sections == []
        assert structure.tables == []
        assert structure.metadata == {}


class TestDocumentQuality:
    """Tests for DocumentQuality schema."""

    def test_create_document_quality(self) -> None:
        """Test creating a document quality score."""
        quality = DocumentQuality(
            overall_score=85.5,
            text_quality=90.0,
            table_quality=80.0,
            ocr_needed=False,
            issues=["Minor formatting issue"],
        )

        assert quality.overall_score == 85.5
        assert quality.text_quality == 90.0
        assert quality.table_quality == 80.0
        assert quality.ocr_needed is False
        assert len(quality.issues) == 1

    def test_document_quality_validation(self) -> None:
        """Test document quality score validation."""
        with pytest.raises(ValueError):
            DocumentQuality(
                overall_score=150.0,  # Invalid > 100
                text_quality=90.0,
                table_quality=80.0,
                ocr_needed=False,
            )


class TestLayoutAnalyzer:
    """Tests for LayoutAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> LayoutAnalyzer:
        """Create layout analyzer instance."""
        return LayoutAnalyzer()

    @pytest.fixture
    def mock_pdf_doc(self) -> Mock:
        """Create mock PDF document."""
        mock_doc = Mock()
        mock_doc.page_count = 3
        mock_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author",
        }

        mock_page = Mock()
        mock_page.get_text.return_value = "Test content on page"

        mock_doc.__iter__ = Mock(return_value=iter([mock_page] * 3))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=False)
        mock_doc.close = Mock()

        return mock_doc

    def test_analyzer_initialization(self, analyzer: LayoutAnalyzer) -> None:
        """Test layout analyzer initialization."""
        assert analyzer.min_header_size == 12.0
        assert analyzer.header_size_threshold == 1.2

    def test_analyze_nonexistent_file(self, analyzer: LayoutAnalyzer) -> None:
        """Test analyzing non-existent PDF file."""
        with pytest.raises(FileNotFoundError):
            analyzer.analyze("/nonexistent/file.pdf")

    @patch("signalforge.nlp.pdf_parser.layout_analyzer.fitz")
    def test_analyze_pdf(
        self, mock_fitz: Mock, analyzer: LayoutAnalyzer, mock_pdf_doc: Mock
    ) -> None:
        """Test analyzing a PDF document."""
        mock_fitz.open.return_value = mock_pdf_doc

        mock_page = Mock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (10, 10, 100, 30),
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Test content",
                                    "size": 12.0,
                                    "font": "Times",
                                }
                            ]
                        }
                    ],
                }
            ]
        }

        mock_pdf_doc.__iter__ = Mock(return_value=iter([mock_page]))

        with patch.object(Path, "exists", return_value=True):
            structure = analyzer.analyze("/test/document.pdf")

        assert structure.page_count == 3
        assert structure.title == "Test Document"
        assert structure.metadata["author"] == "Test Author"
        # Note: close is called twice - once in analyze() and once in extract_text_blocks()
        assert mock_pdf_doc.close.call_count == 2

    def test_extract_text_blocks(self, analyzer: LayoutAnalyzer) -> None:
        """Test extracting text blocks from PDF."""
        mock_doc = Mock()
        mock_page = Mock()

        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (10, 10, 100, 30),
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "First paragraph",
                                    "size": 11.0,
                                    "font": "Arial",
                                }
                            ]
                        }
                    ],
                },
                {
                    "type": 0,
                    "bbox": (10, 40, 100, 60),
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Second paragraph",
                                    "size": 11.0,
                                    "font": "Arial",
                                }
                            ]
                        }
                    ],
                },
            ]
        }

        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.close = Mock()

        with patch("signalforge.nlp.pdf_parser.layout_analyzer.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc

            with patch.object(Path, "exists", return_value=True):
                sections = analyzer.extract_text_blocks("/test/doc.pdf")

        assert len(sections) == 2
        assert sections[0].content == "First paragraph"
        assert sections[1].content == "Second paragraph"
        assert all(s.page_number == 1 for s in sections)

    def test_identify_headers(self, analyzer: LayoutAnalyzer) -> None:
        """Test identifying headers by font size."""
        blocks = [
            {
                "bbox": (10, 10, 100, 40),
                "lines": [
                    {
                        "spans": [
                            {"text": "Large Header", "size": 18.0, "font": "Arial-Bold"}
                        ]
                    }
                ],
            },
            {
                "bbox": (10, 50, 100, 70),
                "lines": [
                    {
                        "spans": [
                            {"text": "Normal text", "size": 11.0, "font": "Arial"}
                        ]
                    }
                ],
            },
        ]

        headers = analyzer.identify_headers(blocks)

        assert len(headers) == 1
        assert headers[0].content == "Large Header"
        assert headers[0].section_type == "header"

    def test_get_reading_order(self, analyzer: LayoutAnalyzer) -> None:
        """Test sorting sections in reading order."""
        sections = [
            DocumentSection(
                section_type="paragraph",
                content="Bottom text",
                page_number=1,
                bounding_box=(10, 100, 100, 120),
            ),
            DocumentSection(
                section_type="paragraph",
                content="Top text",
                page_number=1,
                bounding_box=(10, 10, 100, 30),
            ),
            DocumentSection(
                section_type="paragraph",
                content="Page 2 text",
                page_number=2,
                bounding_box=(10, 10, 100, 30),
            ),
        ]

        sorted_sections = analyzer.get_reading_order(sections)

        assert sorted_sections[0].content == "Top text"
        assert sorted_sections[1].content == "Bottom text"
        assert sorted_sections[2].content == "Page 2 text"

    def test_is_list_item(self, analyzer: LayoutAnalyzer) -> None:
        """Test list item detection."""
        assert analyzer._is_list_item("• First item") is True
        assert analyzer._is_list_item("- Second item") is True
        assert analyzer._is_list_item("1. Numbered item") is True
        assert analyzer._is_list_item("a) Letter item") is True
        assert analyzer._is_list_item("Regular paragraph") is False

    def test_extract_block_text(self, analyzer: LayoutAnalyzer) -> None:
        """Test extracting text from block dictionary."""
        block = {
            "lines": [
                {"spans": [{"text": "First "}, {"text": "part"}]},
                {"spans": [{"text": "Second part"}]},
            ]
        }

        text = analyzer._extract_block_text(block)

        assert text == "First  part Second part"

    def test_get_block_font_info(self, analyzer: LayoutAnalyzer) -> None:
        """Test getting font information from block."""
        block = {
            "lines": [
                {"spans": [{"size": 12.0, "font": "Arial"}]},
                {"spans": [{"size": 12.0, "font": "Arial"}]},
            ]
        }

        font_info = analyzer._get_block_font_info(block)

        assert font_info["avg_size"] == 12.0
        assert font_info["font"] == "Arial"


class TestTableExtractor:
    """Tests for TableExtractor."""

    @pytest.fixture
    def extractor(self) -> TableExtractor:
        """Create table extractor instance."""
        return TableExtractor(use_pdfplumber=False)

    def test_extractor_initialization(self, extractor: TableExtractor) -> None:
        """Test table extractor initialization."""
        assert extractor.use_pdfplumber is False
        assert extractor._pdfplumber is None

    @patch("signalforge.nlp.pdf_parser.table_extractor.fitz")
    def test_extract_nonexistent_file(
        self, mock_fitz: Mock, extractor: TableExtractor
    ) -> None:
        """Test extracting from non-existent file."""
        with pytest.raises(FileNotFoundError):
            extractor.extract_tables("/nonexistent/file.pdf")

    @patch("signalforge.nlp.pdf_parser.table_extractor.fitz")
    def test_extract_tables(
        self, mock_fitz: Mock, extractor: TableExtractor
    ) -> None:
        """Test extracting tables from PDF."""
        mock_doc = Mock()
        mock_page = Mock()
        mock_table_finder = Mock()
        mock_table = Mock()

        mock_table.extract.return_value = [
            ["Header1", "Header2"],
            ["Data1", "Data2"],
            ["Data3", "Data4"],
        ]
        mock_table.bbox = (10, 10, 100, 100)

        mock_table_finder.tables = [mock_table]
        mock_page.find_tables.return_value = mock_table_finder

        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.close = Mock()

        mock_fitz.open.return_value = mock_doc

        with patch.object(Path, "exists", return_value=True):
            tables = extractor.extract_tables("/test/doc.pdf")

        assert len(tables) == 1
        assert tables[0].headers == ["Header1", "Header2"]
        assert len(tables[0].rows) == 2
        assert tables[0].page_number == 1

    def test_clean_table(self, extractor: TableExtractor) -> None:
        """Test cleaning extracted table data."""
        table = ExtractedTable(
            headers=["  Header1  ", "Header2"],
            rows=[
                ["  Data1  ", "Data2"],
                ["", ""],  # Empty row
                ["Data3", "  Data4  "],
            ],
            page_number=1,
        )

        cleaned = extractor.clean_table(table)

        assert cleaned.headers == ["Header1", "Header2"]
        assert len(cleaned.rows) == 2  # Empty row removed
        assert cleaned.rows[0] == ["Data1", "Data2"]
        assert cleaned.rows[1] == ["Data3", "Data4"]

    def test_clean_table_removes_empty_columns(
        self, extractor: TableExtractor
    ) -> None:
        """Test cleaning removes trailing empty columns."""
        table = ExtractedTable(
            headers=["H1", "H2", ""],
            rows=[["A", "B", ""], ["C", "D", ""]],
            page_number=1,
        )

        cleaned = extractor.clean_table(table)

        assert len(cleaned.headers) == 2
        assert all(len(row) == 2 for row in cleaned.rows)

    def test_detect_financial_tables(self, extractor: TableExtractor) -> None:
        """Test detecting tables with financial data."""
        financial_table = ExtractedTable(
            headers=["Quarter", "Revenue", "Profit"],
            rows=[
                ["Q1 2024", "$1,000,000", "$200,000"],
                ["Q2 2024", "$1,200,000", "$250,000"],
            ],
            page_number=1,
        )

        non_financial_table = ExtractedTable(
            headers=["Name", "Color"],
            rows=[["Apple", "Red"], ["Banana", "Yellow"]],
            page_number=1,
        )

        tables = [financial_table, non_financial_table]
        financial = extractor.detect_financial_tables(tables)

        assert len(financial) == 1
        assert financial[0].headers == ["Quarter", "Revenue", "Profit"]

    def test_is_financial_table_by_keywords(
        self, extractor: TableExtractor
    ) -> None:
        """Test financial table detection by keywords."""
        table = ExtractedTable(
            headers=["Item", "Revenue", "Cost"],
            rows=[["A", "100", "50"]],
            page_number=1,
        )

        assert extractor._is_financial_table(table) is True

    def test_is_financial_table_by_numeric_data(
        self, extractor: TableExtractor
    ) -> None:
        """Test financial table detection by numeric data."""
        table = ExtractedTable(
            headers=["Date", "Value1", "Value2"],
            rows=[
                ["2024-01-01", "$1,000", "50%"],
                ["2024-02-01", "$2,000", "60%"],
            ],
            page_number=1,
        )

        assert extractor._is_financial_table(table) is True


class TestOCRFallback:
    """Tests for OCRFallback."""

    @pytest.fixture
    def ocr_fallback(self) -> OCRFallback:
        """Create OCR fallback instance."""
        return OCRFallback()

    def test_ocr_initialization(self, ocr_fallback: OCRFallback) -> None:
        """Test OCR fallback initialization."""
        assert ocr_fallback.tesseract_path is None

    def test_ocr_initialization_with_path(self) -> None:
        """Test OCR initialization with Tesseract path."""
        ocr = OCRFallback(tesseract_path="/usr/bin/tesseract")
        assert ocr.tesseract_path == "/usr/bin/tesseract"

    @patch("signalforge.nlp.pdf_parser.ocr_fallback.fitz")
    def test_needs_ocr_nonexistent_file(
        self, mock_fitz: Mock, ocr_fallback: OCRFallback
    ) -> None:
        """Test OCR check on non-existent file."""
        with pytest.raises(FileNotFoundError):
            ocr_fallback.needs_ocr("/nonexistent/file.pdf")

    @patch("signalforge.nlp.pdf_parser.ocr_fallback.fitz")
    def test_needs_ocr_scanned_document(
        self, mock_fitz: Mock, ocr_fallback: OCRFallback
    ) -> None:
        """Test detecting scanned document needing OCR."""
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = ""  # No text extracted

        mock_doc.__iter__ = Mock(return_value=iter([mock_page] * 3))
        mock_doc.close = Mock()

        mock_fitz.open.return_value = mock_doc

        with patch.object(Path, "exists", return_value=True):
            needs_ocr = ocr_fallback.needs_ocr("/test/scanned.pdf")

        assert needs_ocr is True

    @patch("signalforge.nlp.pdf_parser.ocr_fallback.fitz")
    def test_needs_ocr_text_document(
        self, mock_fitz: Mock, ocr_fallback: OCRFallback
    ) -> None:
        """Test detecting text-based document not needing OCR."""
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "A" * 500  # Sufficient text

        mock_doc.__iter__ = Mock(return_value=iter([mock_page] * 3))
        mock_doc.close = Mock()

        mock_fitz.open.return_value = mock_doc

        with patch.object(Path, "exists", return_value=True):
            needs_ocr = ocr_fallback.needs_ocr("/test/document.pdf")

        assert needs_ocr is False

    def test_extract_text_ocr_no_libraries(
        self, ocr_fallback: OCRFallback
    ) -> None:
        """Test OCR extraction without libraries installed."""
        ocr_fallback._pytesseract = None
        ocr_fallback._pil = None

        with (
            patch.object(Path, "exists", return_value=True),
            pytest.raises(RuntimeError, match="OCR libraries not available"),
        ):
            ocr_fallback.extract_text_ocr("/test/doc.pdf")

    def test_extract_from_image_no_libraries(
        self, ocr_fallback: OCRFallback
    ) -> None:
        """Test image OCR without libraries installed."""
        ocr_fallback._pytesseract = None
        ocr_fallback._pil = None

        with (
            patch.object(Path, "exists", return_value=True),
            pytest.raises(RuntimeError, match="OCR libraries not available"),
        ):
            ocr_fallback.extract_from_image("/test/image.png")


class TestDocumentQualityScorer:
    """Tests for DocumentQualityScorer."""

    @pytest.fixture
    def scorer(self) -> DocumentQualityScorer:
        """Create quality scorer instance."""
        return DocumentQualityScorer()

    @pytest.fixture
    def sample_structure(self) -> DocumentStructure:
        """Create sample document structure."""
        sections = [
            DocumentSection(
                section_type="header",
                content="Introduction",
                page_number=1,
                confidence=1.0,
            ),
            DocumentSection(
                section_type="paragraph",
                content="This is a test paragraph with sufficient content to be meaningful.",
                page_number=1,
                confidence=0.95,
            ),
        ]

        tables = [
            ExtractedTable(
                headers=["Q1", "Q2"],
                rows=[["100", "200"]],
                page_number=1,
            )
        ]

        return DocumentStructure(
            title="Test Document",
            sections=sections,
            tables=tables,
            page_count=1,
            metadata={"author": "Test"},
        )

    def test_scorer_initialization(self, scorer: DocumentQualityScorer) -> None:
        """Test quality scorer initialization."""
        assert scorer.ocr_fallback is not None

    def test_score_nonexistent_file(
        self, scorer: DocumentQualityScorer, sample_structure: DocumentStructure
    ) -> None:
        """Test scoring non-existent file."""
        with pytest.raises(FileNotFoundError):
            scorer.score("/nonexistent/file.pdf", sample_structure)

    @patch.object(OCRFallback, "needs_ocr", return_value=False)
    def test_score_document(
        self,
        mock_needs_ocr: Mock,
        scorer: DocumentQualityScorer,
        sample_structure: DocumentStructure,
    ) -> None:
        """Test scoring a document."""
        with patch.object(Path, "exists", return_value=True):
            quality = scorer.score("/test/doc.pdf", sample_structure)

        assert isinstance(quality, DocumentQuality)
        assert 0 <= quality.overall_score <= 100
        assert 0 <= quality.text_quality <= 100
        assert 0 <= quality.table_quality <= 100
        assert quality.ocr_needed is False
        assert isinstance(quality.issues, list)

    def test_score_text_quality_no_sections(
        self, scorer: DocumentQualityScorer
    ) -> None:
        """Test text quality scoring with no sections."""
        score = scorer._score_text_quality([])
        assert score == 0.0

    def test_score_text_quality_good_content(
        self, scorer: DocumentQualityScorer
    ) -> None:
        """Test text quality scoring with good content."""
        sections = [
            DocumentSection(
                section_type="paragraph",
                content="This is a well-formed paragraph with sufficient content.",
                page_number=1,
                confidence=1.0,
            ),
            DocumentSection(
                section_type="paragraph",
                content="Another paragraph with good quality text content here.",
                page_number=1,
                confidence=0.95,
            ),
        ]

        score = scorer._score_text_quality(sections)
        assert score >= 70.0  # Should be high quality

    def test_score_text_quality_poor_content(
        self, scorer: DocumentQualityScorer
    ) -> None:
        """Test text quality scoring with poor content."""
        sections = [
            DocumentSection(
                section_type="paragraph",
                content="abc",  # Very short
                page_number=1,
                confidence=0.5,
            )
        ]

        score = scorer._score_text_quality(sections)
        assert score < 70.0  # Should be lower quality

    def test_score_table_quality_no_tables(
        self, scorer: DocumentQualityScorer
    ) -> None:
        """Test table quality scoring with no tables."""
        score = scorer._score_table_quality([])
        assert score == 100.0  # No tables is neutral

    def test_score_table_quality_good_tables(
        self, scorer: DocumentQualityScorer
    ) -> None:
        """Test table quality scoring with good tables."""
        tables = [
            ExtractedTable(
                headers=["A", "B", "C"],
                rows=[["1", "2", "3"], ["4", "5", "6"]],
                page_number=1,
            )
        ]

        score = scorer._score_table_quality(tables)
        assert score >= 90.0

    def test_score_table_quality_ragged_tables(
        self, scorer: DocumentQualityScorer
    ) -> None:
        """Test table quality scoring with ragged tables."""
        tables = [
            ExtractedTable(
                headers=["A", "B", "C"],
                rows=[["1", "2"], ["4", "5", "6", "7"]],  # Inconsistent columns
                page_number=1,
            )
        ]

        score = scorer._score_table_quality(tables)
        assert score < 100.0

    def test_detect_issues_empty_structure(
        self, scorer: DocumentQualityScorer
    ) -> None:
        """Test detecting issues in empty structure."""
        structure = DocumentStructure(page_count=1)

        issues = scorer._detect_issues(structure, ocr_needed=False)

        assert "No text sections extracted from document" in issues
        assert "Document title not found" in issues

    def test_detect_issues_ocr_needed(
        self, scorer: DocumentQualityScorer, sample_structure: DocumentStructure
    ) -> None:
        """Test detecting OCR requirement issue."""
        issues = scorer._detect_issues(sample_structure, ocr_needed=True)

        assert any("OCR required" in issue for issue in issues)

    def test_detect_issues_low_text_density(
        self, scorer: DocumentQualityScorer
    ) -> None:
        """Test detecting low text density issue."""
        structure = DocumentStructure(
            sections=[
                DocumentSection(
                    section_type="paragraph", content="A", page_number=1
                )
            ],
            page_count=10,
        )

        issues = scorer._detect_issues(structure, ocr_needed=False)

        assert any("Low text density" in issue for issue in issues)

    def test_calculate_overall_score(
        self, scorer: DocumentQualityScorer
    ) -> None:
        """Test overall score calculation."""
        score = scorer._calculate_overall_score(
            text_quality=90.0,
            table_quality=80.0,
            ocr_needed=False,
            issue_count=1,
        )

        assert 0 <= score <= 100
        assert score < 90.0  # Should be lower than max component

    def test_calculate_overall_score_with_ocr(
        self, scorer: DocumentQualityScorer
    ) -> None:
        """Test overall score with OCR penalty."""
        score_without_ocr = scorer._calculate_overall_score(
            text_quality=90.0,
            table_quality=90.0,
            ocr_needed=False,
            issue_count=0,
        )

        score_with_ocr = scorer._calculate_overall_score(
            text_quality=90.0,
            table_quality=90.0,
            ocr_needed=True,
            issue_count=0,
        )

        assert score_with_ocr < score_without_ocr
