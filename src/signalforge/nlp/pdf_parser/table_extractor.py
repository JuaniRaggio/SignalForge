"""Table extraction from PDFs using multiple strategies."""

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF  # type: ignore[import-not-found]
import structlog

from signalforge.nlp.pdf_parser.schemas import ExtractedTable

logger = structlog.get_logger(__name__)


class TableExtractor:
    """Extract tables from PDFs using multiple strategies."""

    def __init__(self, use_pdfplumber: bool = True) -> None:
        """
        Initialize table extractor.

        Args:
            use_pdfplumber: Whether to use pdfplumber as fallback
        """
        self.use_pdfplumber = use_pdfplumber
        self._pdfplumber = None

        if use_pdfplumber:
            try:
                import pdfplumber  # type: ignore[import-not-found]

                self._pdfplumber = pdfplumber
                logger.info("pdfplumber_available")
            except ImportError:
                logger.warning("pdfplumber_not_installed")
                self.use_pdfplumber = False

    def extract_tables(self, pdf_path: str | Path) -> list[ExtractedTable]:
        """
        Extract all tables from PDF using multiple strategies.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of extracted tables

        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            msg = f"PDF file not found: {pdf_path}"
            raise FileNotFoundError(msg)

        logger.info("extracting_tables", path=str(pdf_path))

        all_tables: list[ExtractedTable] = []

        # Try PyMuPDF first
        pymupdf_tables = self._extract_with_pymupdf(pdf_path)
        all_tables.extend(pymupdf_tables)

        # Fall back to pdfplumber for pages with no tables found
        if self.use_pdfplumber and self._pdfplumber:
            pages_with_tables = {table.page_number for table in all_tables}
            pdfplumber_tables = self._extract_with_pdfplumber(
                pdf_path, exclude_pages=pages_with_tables
            )
            all_tables.extend(pdfplumber_tables)

        # Clean all extracted tables
        cleaned_tables = [self.clean_table(table) for table in all_tables]

        # Remove empty tables
        cleaned_tables = [
            table for table in cleaned_tables if table.rows and table.headers
        ]

        logger.info("tables_extracted", count=len(cleaned_tables))

        return cleaned_tables

    def extract_from_page(
        self, page: fitz.Page, page_num: int
    ) -> list[ExtractedTable]:
        """
        Extract tables from a single page using PyMuPDF.

        Args:
            page: PyMuPDF page object
            page_num: Page number

        Returns:
            List of extracted tables from the page
        """
        tables: list[ExtractedTable] = []

        try:
            # Find tables using PyMuPDF
            tab_rects = page.find_tables()

            if not tab_rects or not tab_rects.tables:
                return tables

            for table_idx, table in enumerate(tab_rects.tables):
                try:
                    # Extract table data
                    extracted = table.extract()

                    if not extracted or len(extracted) < 2:
                        continue

                    # First row as headers
                    headers = [str(cell) if cell else "" for cell in extracted[0]]

                    # Remaining rows as data
                    rows = [
                        [str(cell) if cell else "" for cell in row]
                        for row in extracted[1:]
                    ]

                    # Try to find table title (text above table)
                    title = self._find_table_title(page, table.bbox, page_num)

                    extracted_table = ExtractedTable(
                        headers=headers,
                        rows=rows,
                        page_number=page_num,
                        table_title=title,
                    )

                    tables.append(extracted_table)

                except Exception as e:
                    logger.warning(
                        "failed_to_extract_table",
                        page=page_num,
                        table_idx=table_idx,
                        error=str(e),
                    )

        except Exception as e:
            logger.warning("failed_to_find_tables", page=page_num, error=str(e))

        return tables

    def clean_table(self, table: ExtractedTable) -> ExtractedTable:
        """
        Clean extracted table data.

        Args:
            table: Extracted table to clean

        Returns:
            Cleaned table
        """
        # Remove completely empty rows
        cleaned_rows: list[list[str]] = []
        for row in table.rows:
            if any(cell.strip() for cell in row):
                # Normalize whitespace in cells
                cleaned_row = [" ".join(cell.split()) for cell in row]
                cleaned_rows.append(cleaned_row)

        # Clean headers
        cleaned_headers = [" ".join(header.split()) for header in table.headers]

        # Remove trailing empty columns
        if cleaned_rows:
            max_col = len(cleaned_headers)
            for i in range(len(cleaned_headers) - 1, -1, -1):
                if all(
                    i >= len(row) or not row[i].strip() for row in cleaned_rows
                ) and not cleaned_headers[i].strip():
                    max_col = i
                else:
                    break

            if max_col < len(cleaned_headers):
                cleaned_headers = cleaned_headers[:max_col]
                cleaned_rows = [row[:max_col] for row in cleaned_rows]

        return ExtractedTable(
            headers=cleaned_headers,
            rows=cleaned_rows,
            page_number=table.page_number,
            table_title=table.table_title,
        )

    def detect_financial_tables(
        self, tables: list[ExtractedTable]
    ) -> list[ExtractedTable]:
        """
        Identify tables with financial data.

        Args:
            tables: List of extracted tables

        Returns:
            List of tables containing financial data
        """
        financial_tables: list[ExtractedTable] = []

        for table in tables:
            if self._is_financial_table(table):
                financial_tables.append(table)

        logger.info("financial_tables_detected", count=len(financial_tables))

        return financial_tables

    def _extract_with_pymupdf(self, pdf_path: Path) -> list[ExtractedTable]:
        """Extract tables using PyMuPDF."""
        tables: list[ExtractedTable] = []

        try:
            doc = fitz.open(pdf_path)

            try:
                for page_num, page in enumerate(doc, start=1):
                    page_tables = self.extract_from_page(page, page_num)
                    tables.extend(page_tables)

            finally:
                doc.close()

        except Exception as e:
            logger.error("pymupdf_extraction_failed", error=str(e))

        return tables

    def _extract_with_pdfplumber(
        self, pdf_path: Path, exclude_pages: set[int] | None = None
    ) -> list[ExtractedTable]:
        """Extract tables using pdfplumber fallback."""
        if not self._pdfplumber:
            return []

        tables: list[ExtractedTable] = []
        exclude_pages = exclude_pages or set()

        try:
            with self._pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    if page_num in exclude_pages:
                        continue

                    try:
                        page_tables = page.extract_tables()

                        for table_data in page_tables:
                            if not table_data or len(table_data) < 2:
                                continue

                            headers = [
                                str(cell) if cell else "" for cell in table_data[0]
                            ]
                            rows = [
                                [str(cell) if cell else "" for cell in row]
                                for row in table_data[1:]
                            ]

                            table = ExtractedTable(
                                headers=headers,
                                rows=rows,
                                page_number=page_num,
                                table_title=None,
                            )

                            tables.append(table)

                    except Exception as e:
                        logger.warning(
                            "pdfplumber_page_extraction_failed",
                            page=page_num,
                            error=str(e),
                        )

        except Exception as e:
            logger.error("pdfplumber_extraction_failed", error=str(e))

        return tables

    def _find_table_title(
        self, page: fitz.Page, table_bbox: tuple[Any, ...], page_num: int
    ) -> str | None:
        """Find text above table that might be its title."""
        try:
            # Look for text above table
            search_rect = fitz.Rect(
                table_bbox[0],
                max(0, table_bbox[1] - 50),  # 50 points above
                table_bbox[2],
                table_bbox[1],
            )

            text_blocks = page.get_text("dict", clip=search_rect)["blocks"]

            # Get text from blocks
            texts: list[str] = []
            for block in text_blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                texts.append(text)

            if texts:
                title = " ".join(texts)
                # Check if it's reasonable title length
                if len(title) < 200:
                    return title

        except Exception as e:
            logger.debug(
                "failed_to_find_table_title", page=page_num, error=str(e)
            )

        return None

    def _is_financial_table(self, table: ExtractedTable) -> bool:
        """Check if table contains financial data."""
        import re

        # Check for financial keywords in headers
        financial_keywords = [
            "revenue",
            "income",
            "earnings",
            "ebitda",
            "profit",
            "loss",
            "expense",
            "sales",
            "price",
            "value",
            "cost",
            "margin",
            "equity",
            "assets",
            "liabilities",
            "cash",
            "debt",
            "shares",
        ]

        headers_text = " ".join(table.headers).lower()
        has_financial_header = any(
            keyword in headers_text for keyword in financial_keywords
        )

        # Check for numeric/currency data in cells
        numeric_pattern = r"[\$€£¥]?\s*-?\d+[,\d]*\.?\d*[%MBK]?"
        date_pattern = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}"

        numeric_cells = 0
        total_cells = 0

        for row in table.rows[:10]:  # Check first 10 rows
            for cell in row:
                total_cells += 1
                cell_text = cell.strip()
                if re.search(numeric_pattern, cell_text) or re.search(
                    date_pattern, cell_text
                ):
                    numeric_cells += 1

        # Consider it financial if has financial keywords and >30% numeric data
        has_numeric_data = (
            total_cells > 0 and (numeric_cells / total_cells) > 0.3
        )

        return has_financial_header or has_numeric_data
