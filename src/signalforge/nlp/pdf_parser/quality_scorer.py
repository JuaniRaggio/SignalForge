"""Document quality scoring for parsing reliability assessment."""

from pathlib import Path

import structlog

from signalforge.nlp.pdf_parser.ocr_fallback import OCRFallback
from signalforge.nlp.pdf_parser.schemas import (
    DocumentQuality,
    DocumentSection,
    DocumentStructure,
    ExtractedTable,
)

logger = structlog.get_logger(__name__)


class DocumentQualityScorer:
    """Score document quality for parsing reliability."""

    def __init__(self) -> None:
        """Initialize quality scorer."""
        self.ocr_fallback = OCRFallback()

    def score(
        self, pdf_path: str | Path, structure: DocumentStructure
    ) -> DocumentQuality:
        """
        Calculate comprehensive quality score for parsed document.

        Args:
            pdf_path: Path to original PDF file
            structure: Parsed document structure

        Returns:
            DocumentQuality with scores and issues

        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            msg = f"PDF file not found: {pdf_path}"
            raise FileNotFoundError(msg)

        logger.info("scoring_document_quality", path=str(pdf_path))

        # Score individual components
        text_quality = self._score_text_quality(structure.sections)
        table_quality = self._score_table_quality(structure.tables)

        # Check if OCR is needed
        ocr_needed = self.ocr_fallback.needs_ocr(pdf_path)

        # Detect issues
        issues = self._detect_issues(structure, ocr_needed)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            text_quality=text_quality,
            table_quality=table_quality,
            ocr_needed=ocr_needed,
            issue_count=len(issues),
        )

        quality = DocumentQuality(
            overall_score=overall_score,
            text_quality=text_quality,
            table_quality=table_quality,
            ocr_needed=ocr_needed,
            issues=issues,
        )

        logger.info(
            "quality_scoring_complete",
            overall_score=overall_score,
            text_quality=text_quality,
            table_quality=table_quality,
            ocr_needed=ocr_needed,
            issue_count=len(issues),
        )

        return quality

    def _score_text_quality(self, sections: list[DocumentSection]) -> float:
        """
        Score text extraction quality.

        Args:
            sections: List of document sections

        Returns:
            Quality score from 0-100
        """
        if not sections:
            return 0.0

        score = 100.0

        # Calculate metrics
        total_chars = 0
        total_words = 0
        sections_with_content = 0

        for section in sections:
            content = section.content.strip()
            if content:
                sections_with_content += 1
                total_chars += len(content)
                total_words += len(content.split())

        # Check for sufficient content
        if sections_with_content == 0:
            return 0.0

        avg_section_length = total_chars / sections_with_content

        # Penalize very short sections (likely extraction issues)
        if avg_section_length < 50:
            score -= 30

        # Check word to character ratio (should be ~15-20% for English)
        if total_chars > 0:
            word_char_ratio = total_words / total_chars
            if word_char_ratio < 0.10:  # Too few words per character
                score -= 20
            elif word_char_ratio > 0.30:  # Too many (likely junk)
                score -= 15

        # Check for garbled text indicators
        garbled_indicators = 0
        sample_size = min(5, len(sections))

        for section in sections[:sample_size]:
            content = section.content
            # Check for excessive special characters
            special_chars = sum(
                1 for c in content if not c.isalnum() and not c.isspace()
            )
            if len(content) > 0 and (special_chars / len(content)) > 0.3:
                garbled_indicators += 1

        if garbled_indicators > sample_size / 2:
            score -= 25

        # Use confidence scores if available
        confidence_scores = [
            section.confidence for section in sections if hasattr(section, "confidence")
        ]
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            # Scale confidence from 0-1 to 0-20 point contribution
            score = score * 0.8 + (avg_confidence * 20)

        return max(0.0, min(100.0, score))

    def _score_table_quality(self, tables: list[ExtractedTable]) -> float:
        """
        Score table extraction quality.

        Args:
            tables: List of extracted tables

        Returns:
            Quality score from 0-100
        """
        if not tables:
            # No tables is not necessarily bad, just neutral
            return 100.0

        score = 100.0
        issues = 0

        for table in tables:
            # Check for empty headers
            if not table.headers or all(not h.strip() for h in table.headers):
                issues += 1

            # Check for empty rows
            if not table.rows:
                issues += 1
                continue

            # Check for ragged rows (inconsistent column counts)
            header_cols = len(table.headers)
            for row in table.rows:
                if len(row) != header_cols:
                    issues += 1
                    break

            # Check for mostly empty cells
            total_cells = sum(len(row) for row in table.rows)
            empty_cells = sum(
                1 for row in table.rows for cell in row if not cell.strip()
            )

            if total_cells > 0 and (empty_cells / total_cells) > 0.5:
                issues += 1

        # Penalize based on issue ratio
        if tables:
            issue_ratio = issues / len(tables)
            score -= issue_ratio * 50

        return max(0.0, min(100.0, score))

    def _detect_issues(
        self, structure: DocumentStructure, ocr_needed: bool
    ) -> list[str]:
        """
        Detect potential parsing issues.

        Args:
            structure: Parsed document structure
            ocr_needed: Whether OCR is needed

        Returns:
            List of issue descriptions
        """
        issues: list[str] = []

        # Check for OCR requirement
        if ocr_needed:
            issues.append(
                "Document appears to be scanned - OCR required for text extraction"
            )

        # Check for missing content
        if not structure.sections:
            issues.append("No text sections extracted from document")

        # Check for very short content
        total_content = sum(len(s.content) for s in structure.sections)
        avg_per_page = (
            total_content / structure.page_count if structure.page_count > 0 else 0
        )

        if avg_per_page < 100:
            issues.append(
                f"Low text density ({avg_per_page:.0f} chars/page) - possible extraction issues"
            )

        # Check for missing title
        if not structure.title:
            issues.append("Document title not found")

        # Check table issues
        if structure.tables:
            empty_tables = sum(1 for t in structure.tables if not t.rows)
            if empty_tables > 0:
                issues.append(f"{empty_tables} tables with no data extracted")

            ragged_tables = sum(
                1
                for t in structure.tables
                if any(len(row) != len(t.headers) for row in t.rows)
            )
            if ragged_tables > 0:
                issues.append(
                    f"{ragged_tables} tables with inconsistent column counts"
                )

        # Check for duplicate content (copy-paste issues)
        if len(structure.sections) > 1:
            contents = [s.content for s in structure.sections[:20]]
            unique_contents = set(contents)
            if len(unique_contents) < len(contents) * 0.8:
                issues.append("Significant duplicate content detected")

        # Check metadata quality
        if not structure.metadata:
            issues.append("No document metadata available")

        return issues

    def _calculate_overall_score(
        self,
        text_quality: float,
        table_quality: float,
        ocr_needed: bool,
        issue_count: int,
    ) -> float:
        """
        Calculate overall quality score.

        Args:
            text_quality: Text quality score
            table_quality: Table quality score
            ocr_needed: Whether OCR is needed
            issue_count: Number of detected issues

        Returns:
            Overall quality score from 0-100
        """
        # Weight text quality more heavily (70%) than table quality (30%)
        base_score = (text_quality * 0.7) + (table_quality * 0.3)

        # Penalize for OCR requirement
        if ocr_needed:
            base_score *= 0.7

        # Penalize for issues (up to -5 points per issue)
        issue_penalty = min(issue_count * 5, 25)
        final_score = base_score - issue_penalty

        return max(0.0, min(100.0, final_score))
