"""PDF layout analysis using PyMuPDF."""

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF  # type: ignore[import-not-found]
import structlog

from signalforge.nlp.pdf_parser.schemas import DocumentSection, DocumentStructure

logger = structlog.get_logger(__name__)


class LayoutAnalyzer:
    """Analyze PDF layout and structure using PyMuPDF."""

    def __init__(self) -> None:
        """Initialize layout analyzer."""
        self.min_header_size = 12.0  # Minimum font size for headers
        self.header_size_threshold = 1.2  # Multiplier vs average font size

    def analyze(self, pdf_path: str | Path) -> DocumentStructure:
        """
        Analyze PDF structure and extract content with layout information.

        Args:
            pdf_path: Path to PDF file

        Returns:
            DocumentStructure containing all extracted content

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            fitz.FitzError: If PDF cannot be opened
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            msg = f"PDF file not found: {pdf_path}"
            raise FileNotFoundError(msg)

        logger.info("analyzing_pdf_layout", path=str(pdf_path))

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error("failed_to_open_pdf", path=str(pdf_path), error=str(e))
            raise

        try:
            # Extract metadata
            metadata = self._extract_metadata(doc)

            # Extract text blocks from all pages
            text_blocks = self.extract_text_blocks(pdf_path)

            # Identify headers
            sections = self._classify_sections(text_blocks)

            # Get reading order
            sections = self.get_reading_order(sections)

            # Extract title (first header or from metadata)
            title = self._extract_title(sections, metadata)

            structure = DocumentStructure(
                title=title,
                sections=sections,
                tables=[],  # Tables extracted separately
                page_count=doc.page_count,
                metadata=metadata,
            )

            logger.info(
                "layout_analysis_complete",
                pages=doc.page_count,
                sections=len(sections),
            )

            return structure

        finally:
            doc.close()

    def extract_text_blocks(self, pdf_path: str | Path) -> list[DocumentSection]:
        """
        Extract text blocks with bounding boxes from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of document sections with position information
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        sections: list[DocumentSection] = []

        try:
            for page_num, page in enumerate(doc, start=1):
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if block.get("type") != 0:  # Skip non-text blocks (images)
                        continue

                    # Extract text from lines in block
                    text_content = self._extract_block_text(block)

                    if not text_content.strip():
                        continue

                    # Get bounding box
                    bbox = block.get("bbox")
                    bounding_box = tuple(bbox) if bbox else None

                    # Get font information for classification
                    font_info = self._get_block_font_info(block)

                    section = DocumentSection(
                        section_type="paragraph",  # Will be classified later
                        content=text_content,
                        page_number=page_num,
                        bounding_box=bounding_box,
                        confidence=1.0,
                    )

                    # Store font info temporarily for classification
                    section.__dict__["_font_info"] = font_info

                    sections.append(section)

            logger.info("text_blocks_extracted", count=len(sections))

        finally:
            doc.close()

        return sections

    def identify_headers(self, blocks: list[dict[str, Any]]) -> list[DocumentSection]:
        """
        Identify section headers by font size and style.

        Args:
            blocks: List of text block dictionaries from PyMuPDF

        Returns:
            List of header sections
        """
        sections: list[DocumentSection] = []

        # Calculate average font size for comparison
        all_sizes: list[float] = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        size = span.get("size", 0)
                        if size > 0:
                            all_sizes.append(size)

        avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 12.0

        for block in blocks:
            if "lines" not in block:
                continue

            text = self._extract_block_text(block)
            if not text.strip():
                continue

            # Get first span to determine formatting
            first_span = self._get_first_span(block)
            if not first_span:
                continue

            font_size = first_span.get("size", 0)
            is_bold = "bold" in first_span.get("font", "").lower()

            # Determine if this is a header
            is_header = (
                font_size >= self.min_header_size
                and font_size >= avg_size * self.header_size_threshold
            ) or (is_bold and font_size >= avg_size)

            if is_header:
                section = DocumentSection(
                    section_type="header",
                    content=text,
                    page_number=1,  # Will be set correctly later
                    bounding_box=tuple(block.get("bbox", (0, 0, 0, 0))),
                    confidence=1.0,
                )
                sections.append(section)

        return sections

    def get_reading_order(
        self, sections: list[DocumentSection]
    ) -> list[DocumentSection]:
        """
        Sort sections in proper reading order (top to bottom, left to right).

        Args:
            sections: List of document sections

        Returns:
            Sorted list of sections
        """
        if not sections:
            return sections

        # Sort by page number first, then by vertical position, then horizontal
        def sort_key(section: DocumentSection) -> tuple[int, float, float]:
            page = section.page_number
            if section.bounding_box:
                y0 = section.bounding_box[1]  # Top coordinate
                x0 = section.bounding_box[0]  # Left coordinate
                return (page, y0, x0)
            return (page, 0.0, 0.0)

        sorted_sections = sorted(sections, key=sort_key)

        logger.info("sections_sorted", count=len(sorted_sections))

        return sorted_sections

    def _extract_metadata(self, doc: fitz.Document) -> dict[str, str]:
        """Extract metadata from PDF document."""
        metadata: dict[str, str] = {}

        try:
            pdf_metadata = doc.metadata
            if pdf_metadata:
                for key, value in pdf_metadata.items():
                    if value:
                        metadata[key] = str(value)
        except Exception as e:
            logger.warning("failed_to_extract_metadata", error=str(e))

        return metadata

    def _extract_block_text(self, block: dict[str, Any]) -> str:
        """Extract text from a block dictionary."""
        lines = block.get("lines", [])
        text_parts: list[str] = []

        for line in lines:
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text:
                    text_parts.append(text)

        return " ".join(text_parts)

    def _get_block_font_info(self, block: dict[str, Any]) -> dict[str, float | str]:
        """Get font information from a block."""
        sizes: list[float] = []
        fonts: list[str] = []

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size", 0)
                font = span.get("font", "")
                if size > 0:
                    sizes.append(size)
                if font:
                    fonts.append(font)

        avg_size = sum(sizes) / len(sizes) if sizes else 12.0
        most_common_font = max(set(fonts), key=fonts.count) if fonts else ""

        return {
            "avg_size": avg_size,
            "font": most_common_font,
        }

    def _get_first_span(self, block: dict[str, Any]) -> dict[str, Any] | None:
        """Get the first text span from a block."""
        lines = block.get("lines", [])
        if lines:
            spans = lines[0].get("spans", [])
            if spans:
                return dict(spans[0])
        return None

    def _classify_sections(
        self, sections: list[DocumentSection]
    ) -> list[DocumentSection]:
        """Classify sections as headers, paragraphs, lists, etc."""
        if not sections:
            return sections

        # Calculate average font size
        font_sizes: list[float] = []
        for section in sections:
            font_info = section.__dict__.get("_font_info", {})
            avg_size = font_info.get("avg_size", 0)
            if avg_size > 0:
                font_sizes.append(avg_size)

        avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0

        # Classify each section
        for section in sections:
            font_info = section.__dict__.get("_font_info", {})
            size = font_info.get("avg_size", avg_size)
            font = font_info.get("font", "")

            is_bold = "bold" in font.lower()
            is_large = size >= avg_size * self.header_size_threshold

            # Classify by characteristics
            if (is_large and size >= self.min_header_size) or (is_bold and is_large):
                section.section_type = "header"
            elif self._is_list_item(section.content):
                section.section_type = "list"
            else:
                section.section_type = "paragraph"

            # Clean up temporary font info
            if "_font_info" in section.__dict__:
                del section.__dict__["_font_info"]

        return sections

    def _is_list_item(self, text: str) -> bool:
        """Check if text appears to be a list item."""
        text = text.strip()
        if not text:
            return False

        # Check for bullet points or numbered lists
        list_markers = ["•", "-", "*", "◦", "▪"]
        if any(text.startswith(marker) for marker in list_markers):
            return True

        # Check for numbered lists (1., 2., a., b., etc.)
        import re

        numbered_pattern = r"^(\d+|[a-z])[.)]\s"
        return bool(re.match(numbered_pattern, text))

    def _extract_title(
        self, sections: list[DocumentSection], metadata: dict[str, str]
    ) -> str | None:
        """Extract document title from sections or metadata."""
        # Try metadata first
        title = metadata.get("title")
        if title and title.strip():
            return title.strip()

        # Try first header
        for section in sections:
            if section.section_type == "header" and section.page_number == 1:
                return section.content.strip()

        return None
