"""OCR fallback for scanned or image-based PDFs."""

from pathlib import Path

import fitz  # PyMuPDF  # type: ignore[import-not-found]
import structlog

logger = structlog.get_logger(__name__)


class OCRFallback:
    """OCR fallback for scanned or image-based PDFs."""

    def __init__(self, tesseract_path: str | None = None) -> None:
        """
        Initialize OCR fallback.

        Args:
            tesseract_path: Optional path to Tesseract executable
        """
        self.tesseract_path = tesseract_path
        self._pytesseract = None
        self._pil = None

        # Try to import OCR libraries
        try:
            import pytesseract  # type: ignore[import-not-found,unused-ignore]
            from PIL import Image  # type: ignore[import-not-found,unused-ignore]

            self._pytesseract = pytesseract
            self._pil = Image

            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

            logger.info("ocr_libraries_available")

        except ImportError:
            logger.warning(
                "ocr_libraries_not_installed",
                msg="pytesseract and/or Pillow not installed",
            )

    def needs_ocr(self, pdf_path: str | Path) -> bool:
        """
        Determine if PDF needs OCR based on text content analysis.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if document appears to be scanned and needs OCR

        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            msg = f"PDF file not found: {pdf_path}"
            raise FileNotFoundError(msg)

        logger.info("checking_ocr_requirement", path=str(pdf_path))

        try:
            doc = fitz.open(pdf_path)

            try:
                total_chars = 0
                pages_checked = 0
                max_pages_to_check = 5

                for _page_num, page in enumerate(doc):
                    if pages_checked >= max_pages_to_check:
                        break

                    # Get text from page
                    text = page.get_text()
                    total_chars += len(text.strip())
                    pages_checked += 1

                # If very little text extracted, likely needs OCR
                avg_chars_per_page = (
                    total_chars / pages_checked if pages_checked > 0 else 0
                )

                # Threshold: less than 100 chars per page suggests scanned document
                needs_ocr = avg_chars_per_page < 100

                logger.info(
                    "ocr_check_complete",
                    avg_chars_per_page=avg_chars_per_page,
                    needs_ocr=needs_ocr,
                )

                return needs_ocr

            finally:
                doc.close()

        except Exception as e:
            logger.error("ocr_check_failed", error=str(e))
            return False

    def extract_text_ocr(self, pdf_path: str | Path) -> str:
        """
        Extract text from PDF using OCR.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text from all pages

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            RuntimeError: If OCR libraries not available
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            msg = f"PDF file not found: {pdf_path}"
            raise FileNotFoundError(msg)

        if not self._pytesseract or not self._pil:
            msg = "OCR libraries not available. Install pytesseract and Pillow."
            raise RuntimeError(msg)

        logger.info("extracting_text_with_ocr", path=str(pdf_path))

        try:
            doc = fitz.open(pdf_path)
            all_text: list[str] = []

            try:
                for page_num, page in enumerate(doc, start=1):
                    logger.debug("processing_page_with_ocr", page=page_num)

                    # Render page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
                    img_data = pix.tobytes("png")

                    # Convert to PIL Image
                    import io

                    img = self._pil.open(io.BytesIO(img_data))

                    # Extract text using OCR
                    text: str = self._pytesseract.image_to_string(img)

                    if text.strip():
                        all_text.append(f"--- Page {page_num} ---\n{text}")

                combined_text = "\n\n".join(all_text)

                logger.info(
                    "ocr_extraction_complete",
                    pages=doc.page_count,
                    chars_extracted=len(combined_text),
                )

                return combined_text

            finally:
                doc.close()

        except Exception as e:
            logger.error("ocr_extraction_failed", error=str(e))
            raise

    def extract_from_image(self, image_path: str | Path) -> str:
        """
        Extract text from a single image using OCR.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If OCR libraries not available
        """
        image_path = Path(image_path)
        if not image_path.exists():
            msg = f"Image file not found: {image_path}"
            raise FileNotFoundError(msg)

        if not self._pytesseract or not self._pil:
            msg = "OCR libraries not available. Install pytesseract and Pillow."
            raise RuntimeError(msg)

        logger.info("extracting_text_from_image", path=str(image_path))

        try:
            img = self._pil.open(image_path)
            text: str = self._pytesseract.image_to_string(img)

            logger.info(
                "image_ocr_complete",
                chars_extracted=len(text),
            )

            return text

        except Exception as e:
            logger.error("image_ocr_failed", error=str(e))
            raise
