"""Base schemas for standardized API responses.

This module provides a consistent response structure with:
- BaseResponse pattern for all API responses
- Standardized pagination
- Error response schemas
- Metadata support
- Type-safe generic responses
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Type variable for generic data payload
T = TypeVar("T")


class ResponseMetadata(BaseModel):
    """Metadata included in all API responses."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp in UTC",
    )
    request_id: str | None = Field(
        default=None,
        description="Unique request identifier",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for request tracing",
    )
    api_version: str = Field(
        default="v1",
        description="API version",
    )


class ErrorDetail(BaseModel):
    """Detailed error information."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description="Machine-readable error code (e.g., SF4001)",
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
    )
    field: str | None = Field(
        default=None,
        description="Field that caused the error (for validation errors)",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error context",
    )


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(
        default=False,
        description="Always false for error responses",
    )
    error: ErrorDetail = Field(
        ...,
        description="Error details",
    )
    meta: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="Response metadata",
    )


class PaginationInfo(BaseModel):
    """Pagination information for list responses."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(
        ...,
        ge=0,
        description="Total number of items available",
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Current page number (1-indexed)",
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of items per page",
    )
    total_pages: int = Field(
        default=1,
        ge=1,
        description="Total number of pages",
    )
    has_next: bool = Field(
        default=False,
        description="Whether there is a next page",
    )
    has_previous: bool = Field(
        default=False,
        description="Whether there is a previous page",
    )

    @classmethod
    def calculate(
        cls,
        total: int,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginationInfo:
        """Calculate pagination info from total and current page.

        Args:
            total: Total number of items.
            page: Current page number (1-indexed).
            page_size: Items per page.

        Returns:
            Calculated pagination info.
        """
        total_pages = max(1, (total + page_size - 1) // page_size)

        return cls(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )


class BaseResponse(BaseModel, Generic[T]):
    """Base response schema for all successful API responses.

    Usage:
        class UserResponse(BaseModel):
            id: str
            name: str

        # Single item response
        response = BaseResponse[UserResponse](
            data=UserResponse(id="123", name="John"),
        )

        # List response with pagination
        response = BaseResponse[list[UserResponse]](
            data=[...],
            pagination=PaginationInfo.calculate(total=100, page=1),
        )
    """

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(
        default=True,
        description="Indicates successful response",
    )
    data: T = Field(
        ...,
        description="Response payload",
    )
    message: str | None = Field(
        default=None,
        description="Optional human-readable message",
    )
    pagination: PaginationInfo | None = Field(
        default=None,
        description="Pagination info for list responses",
    )
    meta: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="Response metadata",
    )


class EmptyResponse(BaseModel):
    """Response for operations that don't return data (e.g., DELETE)."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(
        default=True,
        description="Indicates successful operation",
    )
    message: str = Field(
        default="Operation completed successfully",
        description="Human-readable success message",
    )
    meta: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="Response metadata",
    )


# ============================================================================
# Pagination Request Schema
# ============================================================================


class PaginationParams(BaseModel):
    """Query parameters for paginated requests."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-indexed)",
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        alias="limit",
        description="Number of items per page",
    )

    @property
    def offset(self) -> int:
        """Calculate the offset for database queries."""
        return (self.page - 1) * self.page_size


# ============================================================================
# Common Response Types
# ============================================================================


class HealthCheckResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Service status (healthy, degraded, unhealthy)",
    )
    version: str = Field(
        ...,
        description="Application version",
    )
    checks: dict[str, bool] | None = Field(
        default=None,
        description="Individual health check results",
    )


class CountResponse(BaseModel):
    """Response for count operations."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(
        ...,
        ge=0,
        description="The count value",
    )


class IdResponse(BaseModel):
    """Response for create operations returning an ID."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Created resource ID",
    )


# ============================================================================
# Helper Functions
# ============================================================================


def create_success_response(
    data: T,
    message: str | None = None,
    pagination: PaginationInfo | None = None,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Create a standardized success response dictionary.

    Args:
        data: The response payload.
        message: Optional success message.
        pagination: Pagination info for list responses.
        request_id: Request identifier.
        correlation_id: Correlation ID for tracing.

    Returns:
        Response dictionary ready for JSONResponse.
    """
    response: dict[str, Any] = {
        "success": True,
        "data": data if isinstance(data, dict) else data,
        "meta": {
            "timestamp": datetime.now(UTC).isoformat(),
            "api_version": "v1",
        },
    }

    if message:
        response["message"] = message

    if pagination:
        response["pagination"] = pagination.model_dump()

    if request_id:
        response["meta"]["request_id"] = request_id

    if correlation_id:
        response["meta"]["correlation_id"] = correlation_id

    return response


def create_error_response(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    field: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Create a standardized error response dictionary.

    Args:
        code: Machine-readable error code.
        message: Human-readable error message.
        details: Additional error context.
        field: Field that caused the error.
        correlation_id: Correlation ID for tracing.

    Returns:
        Error response dictionary ready for JSONResponse.
    """
    error: dict[str, Any] = {
        "code": code,
        "message": message,
    }

    if field:
        error["field"] = field

    if details:
        error["details"] = details

    response: dict[str, Any] = {
        "success": False,
        "error": error,
        "meta": {
            "timestamp": datetime.now(UTC).isoformat(),
            "api_version": "v1",
        },
    }

    if correlation_id:
        response["meta"]["correlation_id"] = correlation_id

    return response
