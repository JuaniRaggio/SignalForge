"""Data schemas for PDF parsing structures."""

from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    """Represents a section of a document with its metadata."""

    section_type: str = Field(
        ...,
        description="Type of section: header, paragraph, table, figure, list",
    )
    content: str = Field(..., description="Text content of the section")
    page_number: int = Field(..., description="Page number where section appears")
    bounding_box: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Bounding box coordinates (x0, y0, x1, y1)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the extraction",
    )


class ExtractedTable(BaseModel):
    """Represents an extracted table from a PDF."""

    headers: list[str] = Field(..., description="Column headers")
    rows: list[list[str]] = Field(..., description="Table data rows")
    page_number: int = Field(..., description="Page number where table appears")
    table_title: str | None = Field(
        default=None,
        description="Title or caption of the table",
    )


class DocumentStructure(BaseModel):
    """Complete structure of a parsed PDF document."""

    title: str | None = Field(default=None, description="Document title")
    sections: list[DocumentSection] = Field(
        default_factory=list,
        description="All document sections",
    )
    tables: list[ExtractedTable] = Field(
        default_factory=list,
        description="All extracted tables",
    )
    page_count: int = Field(..., description="Total number of pages")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Document metadata (author, date, etc.)",
    )


class DocumentQuality(BaseModel):
    """Quality metrics for a parsed document."""

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall quality score 0-100",
    )
    text_quality: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Text extraction quality score",
    )
    table_quality: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Table extraction quality score",
    )
    ocr_needed: bool = Field(
        ...,
        description="Whether OCR is needed for this document",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of detected issues",
    )
