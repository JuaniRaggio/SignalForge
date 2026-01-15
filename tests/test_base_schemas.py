"""Tests for base response schemas."""

import pytest
from pydantic import BaseModel

from signalforge.schemas.base import (
    BaseResponse,
    CountResponse,
    EmptyResponse,
    ErrorDetail,
    ErrorResponse,
    PaginationInfo,
    PaginationParams,
    ResponseMetadata,
    create_error_response,
    create_success_response,
)


class TestResponseMetadata:
    """Tests for ResponseMetadata."""

    def test_default_values(self) -> None:
        """Test metadata has sensible defaults."""
        meta = ResponseMetadata()

        assert meta.timestamp is not None
        assert meta.api_version == "v1"
        assert meta.request_id is None
        assert meta.correlation_id is None

    def test_custom_values(self) -> None:
        """Test metadata accepts custom values."""
        meta = ResponseMetadata(
            request_id="req-123",
            correlation_id="corr-456",
        )

        assert meta.request_id == "req-123"
        assert meta.correlation_id == "corr-456"


class TestPaginationInfo:
    """Tests for PaginationInfo."""

    def test_calculate_first_page(self) -> None:
        """Test pagination calculation for first page."""
        pagination = PaginationInfo.calculate(total=100, page=1, page_size=20)

        assert pagination.total == 100
        assert pagination.page == 1
        assert pagination.page_size == 20
        assert pagination.total_pages == 5
        assert pagination.has_next is True
        assert pagination.has_previous is False

    def test_calculate_middle_page(self) -> None:
        """Test pagination calculation for middle page."""
        pagination = PaginationInfo.calculate(total=100, page=3, page_size=20)

        assert pagination.page == 3
        assert pagination.has_next is True
        assert pagination.has_previous is True

    def test_calculate_last_page(self) -> None:
        """Test pagination calculation for last page."""
        pagination = PaginationInfo.calculate(total=100, page=5, page_size=20)

        assert pagination.page == 5
        assert pagination.has_next is False
        assert pagination.has_previous is True

    def test_calculate_single_page(self) -> None:
        """Test pagination when all items fit on one page."""
        pagination = PaginationInfo.calculate(total=10, page=1, page_size=20)

        assert pagination.total_pages == 1
        assert pagination.has_next is False
        assert pagination.has_previous is False

    def test_calculate_empty_result(self) -> None:
        """Test pagination with no results."""
        pagination = PaginationInfo.calculate(total=0, page=1, page_size=20)

        assert pagination.total == 0
        assert pagination.total_pages == 1  # Always at least 1 page
        assert pagination.has_next is False
        assert pagination.has_previous is False

    def test_calculate_partial_last_page(self) -> None:
        """Test pagination when last page is partial."""
        pagination = PaginationInfo.calculate(total=95, page=1, page_size=20)

        assert pagination.total_pages == 5  # Ceiling division


class TestPaginationParams:
    """Tests for PaginationParams."""

    def test_default_values(self) -> None:
        """Test default pagination parameters."""
        params = PaginationParams()

        assert params.page == 1
        assert params.page_size == 20
        assert params.offset == 0

    def test_offset_calculation(self) -> None:
        """Test offset calculation from page number."""
        params = PaginationParams(page=3, limit=20)

        assert params.offset == 40  # (3-1) * 20

    def test_limit_alias(self) -> None:
        """Test that 'limit' works as alias for page_size."""
        params = PaginationParams(page=1, limit=50)

        assert params.page_size == 50


class TestBaseResponse:
    """Tests for BaseResponse generic type."""

    def test_simple_response(self) -> None:
        """Test BaseResponse with simple data."""

        class UserData(BaseModel):
            id: str
            name: str

        response = BaseResponse[UserData](
            data=UserData(id="123", name="John"),
            message="User found",
        )

        assert response.success is True
        assert response.data.id == "123"
        assert response.message == "User found"

    def test_list_response_with_pagination(self) -> None:
        """Test BaseResponse with list data and pagination."""

        class Item(BaseModel):
            id: str

        items = [Item(id=str(i)) for i in range(10)]
        pagination = PaginationInfo.calculate(total=100, page=1)

        response = BaseResponse[list[Item]](
            data=items,
            pagination=pagination,
        )

        assert response.success is True
        assert len(response.data) == 10
        assert response.pagination is not None
        assert response.pagination.total == 100


class TestErrorResponse:
    """Tests for ErrorResponse."""

    def test_error_response_structure(self) -> None:
        """Test error response has correct structure."""
        error = ErrorDetail(
            code="SF4001",
            message="Invalid input",
            field="email",
        )

        response = ErrorResponse(error=error)

        assert response.success is False
        assert response.error.code == "SF4001"
        assert response.error.message == "Invalid input"

    def test_error_with_details(self) -> None:
        """Test error response with additional details."""
        error = ErrorDetail(
            code="SF4000",
            message="Validation failed",
            details={"validation_errors": [{"field": "name", "message": "required"}]},
        )

        response = ErrorResponse(error=error)

        assert response.error.details is not None
        assert "validation_errors" in response.error.details


class TestEmptyResponse:
    """Tests for EmptyResponse."""

    def test_default_message(self) -> None:
        """Test empty response has default message."""
        response = EmptyResponse()

        assert response.success is True
        assert response.message == "Operation completed successfully"

    def test_custom_message(self) -> None:
        """Test empty response with custom message."""
        response = EmptyResponse(message="Resource deleted")

        assert response.message == "Resource deleted"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_success_response(self) -> None:
        """Test create_success_response helper."""
        response = create_success_response(
            data={"id": "123", "name": "Test"},
            message="Created successfully",
        )

        assert response["success"] is True
        assert response["data"]["id"] == "123"
        assert response["message"] == "Created successfully"
        assert "meta" in response
        assert "timestamp" in response["meta"]

    def test_create_success_response_with_pagination(self) -> None:
        """Test create_success_response with pagination."""
        pagination = PaginationInfo.calculate(total=100, page=1)

        response = create_success_response(
            data=[{"id": str(i)} for i in range(10)],
            pagination=pagination,
        )

        assert response["success"] is True
        assert "pagination" in response
        assert response["pagination"]["total"] == 100

    def test_create_error_response(self) -> None:
        """Test create_error_response helper."""
        response = create_error_response(
            code="SF4001",
            message="Invalid input",
            field="email",
            details={"reason": "invalid format"},
        )

        assert response["success"] is False
        assert response["error"]["code"] == "SF4001"
        assert response["error"]["message"] == "Invalid input"
        assert response["error"]["field"] == "email"
        assert response["error"]["details"]["reason"] == "invalid format"

    def test_create_error_response_with_correlation_id(self) -> None:
        """Test create_error_response includes correlation ID."""
        response = create_error_response(
            code="SF1000",
            message="Internal error",
            correlation_id="corr-123",
        )

        assert response["meta"]["correlation_id"] == "corr-123"


class TestCountResponse:
    """Tests for CountResponse."""

    def test_count_response(self) -> None:
        """Test count response structure."""
        response = CountResponse(count=42)

        assert response.count == 42

    def test_count_response_validation(self) -> None:
        """Test count cannot be negative."""
        with pytest.raises(ValueError):
            CountResponse(count=-1)
