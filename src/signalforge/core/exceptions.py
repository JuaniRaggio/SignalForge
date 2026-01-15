"""Custom exceptions for SignalForge.

This module provides a comprehensive exception hierarchy with:
- Structured error information
- HTTP status code mapping
- User-friendly error messages
- Machine-readable error codes
- Contextual details for debugging
"""

from __future__ import annotations

from enum import Enum
from http import HTTPStatus
from typing import Any


class ErrorCode(str, Enum):
    """Machine-readable error codes for API responses."""

    # General errors (1xxx)
    INTERNAL_ERROR = "SF1000"
    UNKNOWN_ERROR = "SF1001"
    CONFIGURATION_ERROR = "SF1002"

    # Authentication errors (2xxx)
    AUTHENTICATION_REQUIRED = "SF2000"
    INVALID_CREDENTIALS = "SF2001"
    TOKEN_EXPIRED = "SF2002"
    TOKEN_INVALID = "SF2003"
    REFRESH_TOKEN_EXPIRED = "SF2004"

    # Authorization errors (3xxx)
    PERMISSION_DENIED = "SF3000"
    INSUFFICIENT_PERMISSIONS = "SF3001"
    RESOURCE_ACCESS_DENIED = "SF3002"

    # Validation errors (4xxx)
    VALIDATION_ERROR = "SF4000"
    INVALID_INPUT = "SF4001"
    MISSING_REQUIRED_FIELD = "SF4002"
    INVALID_FORMAT = "SF4003"
    VALUE_OUT_OF_RANGE = "SF4004"

    # Resource errors (5xxx)
    RESOURCE_NOT_FOUND = "SF5000"
    USER_NOT_FOUND = "SF5001"
    SYMBOL_NOT_FOUND = "SF5002"
    ARTICLE_NOT_FOUND = "SF5003"

    # Database errors (6xxx)
    DATABASE_ERROR = "SF6000"
    CONNECTION_FAILED = "SF6001"
    QUERY_FAILED = "SF6002"
    TRANSACTION_FAILED = "SF6003"
    INTEGRITY_ERROR = "SF6004"

    # External API errors (7xxx)
    EXTERNAL_API_ERROR = "SF7000"
    YAHOO_FINANCE_ERROR = "SF7001"
    RSS_FEED_ERROR = "SF7002"
    API_TIMEOUT = "SF7003"
    API_RATE_LIMITED = "SF7004"
    CIRCUIT_BREAKER_OPEN = "SF7005"

    # Rate limiting errors (8xxx)
    RATE_LIMIT_EXCEEDED = "SF8000"
    QUOTA_EXCEEDED = "SF8001"

    # Data ingestion errors (9xxx)
    INGESTION_ERROR = "SF9000"
    INVALID_DATA_FORMAT = "SF9001"
    DATA_QUALITY_ERROR = "SF9002"
    SCRAPING_ERROR = "SF9003"


class SignalForgeException(Exception):
    """Base exception for all SignalForge errors.

    Provides structured error information suitable for API responses
    and logging.

    Attributes:
        message: Human-readable error message.
        error_code: Machine-readable error code.
        http_status: HTTP status code for API responses.
        details: Additional context for debugging.
        user_message: User-friendly message (may differ from message).
    """

    message: str = "An unexpected error occurred"
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR
    http_status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: ErrorCode | None = None,
        http_status: HTTPStatus | None = None,
        details: dict[str, Any] | None = None,
        user_message: str | None = None,
    ) -> None:
        """Initialize exception.

        Args:
            message: Human-readable error message.
            error_code: Machine-readable error code.
            http_status: HTTP status code for API responses.
            details: Additional context for debugging.
            user_message: User-friendly message for end users.
        """
        self.message = message or self.__class__.message
        self.error_code = error_code or self.__class__.error_code
        self.http_status = http_status or self.__class__.http_status
        self.details = details or {}
        self.user_message = user_message or self.message
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses.

        Returns:
            Dictionary with error information.
        """
        return {
            "error": {
                "code": self.error_code.value,
                "message": self.user_message,
                "details": self.details if self.details else None,
            }
        }

    def __str__(self) -> str:
        """String representation including error code."""
        return f"[{self.error_code.value}] {self.message}"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code.value}, "
            f"http_status={self.http_status.value}, "
            f"details={self.details!r}"
            f")"
        )


# ============================================================================
# Authentication Exceptions
# ============================================================================


class AuthenticationError(SignalForgeException):
    """Authentication-related errors."""

    message = "Authentication required"
    error_code = ErrorCode.AUTHENTICATION_REQUIRED
    http_status = HTTPStatus.UNAUTHORIZED


class InvalidCredentialsError(AuthenticationError):
    """Invalid username or password."""

    message = "Invalid username or password"
    error_code = ErrorCode.INVALID_CREDENTIALS
    user_message = "The credentials you provided are incorrect"


class TokenExpiredError(AuthenticationError):
    """Access token has expired."""

    message = "Access token has expired"
    error_code = ErrorCode.TOKEN_EXPIRED
    user_message = "Your session has expired. Please log in again."


class InvalidTokenError(AuthenticationError):
    """Invalid or malformed token."""

    message = "Invalid or malformed token"
    error_code = ErrorCode.TOKEN_INVALID


class RefreshTokenExpiredError(AuthenticationError):
    """Refresh token has expired."""

    message = "Refresh token has expired"
    error_code = ErrorCode.REFRESH_TOKEN_EXPIRED
    user_message = "Your session has expired. Please log in again."


# ============================================================================
# Authorization Exceptions
# ============================================================================


class AuthorizationError(SignalForgeException):
    """Authorization-related errors."""

    message = "Permission denied"
    error_code = ErrorCode.PERMISSION_DENIED
    http_status = HTTPStatus.FORBIDDEN


class InsufficientPermissionsError(AuthorizationError):
    """User lacks required permissions."""

    message = "Insufficient permissions for this action"
    error_code = ErrorCode.INSUFFICIENT_PERMISSIONS
    user_message = "You don't have permission to perform this action"


class ResourceAccessDeniedError(AuthorizationError):
    """Access to specific resource denied."""

    message = "Access to this resource is denied"
    error_code = ErrorCode.RESOURCE_ACCESS_DENIED


# ============================================================================
# Validation Exceptions
# ============================================================================


class ValidationError(SignalForgeException):
    """Validation-related errors."""

    message = "Validation error"
    error_code = ErrorCode.VALIDATION_ERROR
    http_status = HTTPStatus.BAD_REQUEST

    def __init__(
        self,
        message: str | None = None,
        *,
        field: str | None = None,
        value: Any = None,
        constraint: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Error message.
            field: Name of the field that failed validation.
            value: The invalid value.
            constraint: Description of the constraint that was violated.
            **kwargs: Additional arguments passed to parent.
        """
        details = kwargs.pop("details", {}) or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if constraint:
            details["constraint"] = constraint

        super().__init__(message, details=details, **kwargs)


class InvalidInputError(ValidationError):
    """Invalid input data."""

    message = "Invalid input data"
    error_code = ErrorCode.INVALID_INPUT


class MissingRequiredFieldError(ValidationError):
    """Required field is missing."""

    message = "Required field is missing"
    error_code = ErrorCode.MISSING_REQUIRED_FIELD


class InvalidFormatError(ValidationError):
    """Data format is invalid."""

    message = "Invalid data format"
    error_code = ErrorCode.INVALID_FORMAT


class ValueOutOfRangeError(ValidationError):
    """Value is outside acceptable range."""

    message = "Value is outside acceptable range"
    error_code = ErrorCode.VALUE_OUT_OF_RANGE


# ============================================================================
# Resource Exceptions
# ============================================================================


class NotFoundError(SignalForgeException):
    """Resource not found errors."""

    message = "Resource not found"
    error_code = ErrorCode.RESOURCE_NOT_FOUND
    http_status = HTTPStatus.NOT_FOUND

    def __init__(
        self,
        message: str | None = None,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize not found error.

        Args:
            message: Error message.
            resource_type: Type of resource not found.
            resource_id: ID of the resource.
            **kwargs: Additional arguments passed to parent.
        """
        details = kwargs.pop("details", {}) or {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        if not message and resource_type:
            message = f"{resource_type} not found"

        super().__init__(message, details=details, **kwargs)


class UserNotFoundError(NotFoundError):
    """User not found."""

    message = "User not found"
    error_code = ErrorCode.USER_NOT_FOUND


class SymbolNotFoundError(NotFoundError):
    """Symbol/ticker not found."""

    message = "Symbol not found"
    error_code = ErrorCode.SYMBOL_NOT_FOUND


class ArticleNotFoundError(NotFoundError):
    """News article not found."""

    message = "Article not found"
    error_code = ErrorCode.ARTICLE_NOT_FOUND


# ============================================================================
# Database Exceptions
# ============================================================================


class DatabaseError(SignalForgeException):
    """Database-related errors."""

    message = "Database error"
    error_code = ErrorCode.DATABASE_ERROR
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    user_message = "A database error occurred. Please try again later."


class ConnectionError(DatabaseError):
    """Database connection failed."""

    message = "Database connection failed"
    error_code = ErrorCode.CONNECTION_FAILED


class QueryError(DatabaseError):
    """Database query failed."""

    message = "Database query failed"
    error_code = ErrorCode.QUERY_FAILED


class TransactionError(DatabaseError):
    """Database transaction failed."""

    message = "Database transaction failed"
    error_code = ErrorCode.TRANSACTION_FAILED


class IntegrityError(DatabaseError):
    """Database integrity constraint violated."""

    message = "Data integrity constraint violated"
    error_code = ErrorCode.INTEGRITY_ERROR
    http_status = HTTPStatus.CONFLICT


# ============================================================================
# External API Exceptions
# ============================================================================


class ExternalAPIError(SignalForgeException):
    """External API-related errors."""

    message = "External API error"
    error_code = ErrorCode.EXTERNAL_API_ERROR
    http_status = HTTPStatus.BAD_GATEWAY
    user_message = "An external service is temporarily unavailable"

    def __init__(
        self,
        message: str,
        source: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize external API error.

        Args:
            message: Error message.
            source: Name of the external API.
            status_code: HTTP status code from the API.
            response_body: Response body from the API (truncated).
            **kwargs: Additional arguments passed to parent.
        """
        self.source = source
        self.api_status_code = status_code

        details = kwargs.pop("details", {}) or {}
        details["source"] = source
        if status_code:
            details["api_status_code"] = status_code
        if response_body:
            details["response_preview"] = response_body[:500]

        super().__init__(message, details=details, **kwargs)


class YahooFinanceError(ExternalAPIError):
    """Yahoo Finance API error."""

    error_code = ErrorCode.YAHOO_FINANCE_ERROR

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, source="yahoo_finance", **kwargs)


class RSSFeedError(ExternalAPIError):
    """RSS feed error."""

    error_code = ErrorCode.RSS_FEED_ERROR

    def __init__(self, message: str, feed_name: str, **kwargs: Any) -> None:
        super().__init__(message, source=f"rss:{feed_name}", **kwargs)


class APITimeoutError(ExternalAPIError):
    """External API timeout."""

    message = "External API request timed out"
    error_code = ErrorCode.API_TIMEOUT
    http_status = HTTPStatus.GATEWAY_TIMEOUT


class APIRateLimitedError(ExternalAPIError):
    """External API rate limited."""

    message = "External API rate limit exceeded"
    error_code = ErrorCode.API_RATE_LIMITED
    http_status = HTTPStatus.TOO_MANY_REQUESTS


class CircuitBreakerOpenError(ExternalAPIError):
    """Circuit breaker is open."""

    message = "Service temporarily unavailable due to repeated failures"
    error_code = ErrorCode.CIRCUIT_BREAKER_OPEN
    http_status = HTTPStatus.SERVICE_UNAVAILABLE


# ============================================================================
# Rate Limiting Exceptions
# ============================================================================


class RateLimitError(SignalForgeException):
    """Rate limit exceeded errors."""

    message = "Rate limit exceeded"
    error_code = ErrorCode.RATE_LIMIT_EXCEEDED
    http_status = HTTPStatus.TOO_MANY_REQUESTS
    user_message = "You have made too many requests. Please wait before trying again."

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after: int | None = None,
        limit: int | None = None,
        window_seconds: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize rate limit error.

        Args:
            message: Error message.
            retry_after: Seconds until rate limit resets.
            limit: The rate limit that was exceeded.
            window_seconds: The time window for the rate limit.
            **kwargs: Additional arguments passed to parent.
        """
        details = kwargs.pop("details", {}) or {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        if limit:
            details["limit"] = limit
        if window_seconds:
            details["window_seconds"] = window_seconds

        super().__init__(message, details=details, **kwargs)


class QuotaExceededError(RateLimitError):
    """API quota exceeded."""

    message = "API quota exceeded"
    error_code = ErrorCode.QUOTA_EXCEEDED


# ============================================================================
# Data Ingestion Exceptions
# ============================================================================


class DataIngestionError(SignalForgeException):
    """Data ingestion-related errors."""

    message = "Data ingestion error"
    error_code = ErrorCode.INGESTION_ERROR
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR


class InvalidDataFormatError(DataIngestionError):
    """Invalid data format during ingestion."""

    message = "Invalid data format"
    error_code = ErrorCode.INVALID_DATA_FORMAT
    http_status = HTTPStatus.BAD_REQUEST


class DataQualityError(DataIngestionError):
    """Data quality issues detected."""

    message = "Data quality error"
    error_code = ErrorCode.DATA_QUALITY_ERROR

    def __init__(
        self,
        message: str | None = None,
        *,
        valid_count: int | None = None,
        invalid_count: int | None = None,
        errors: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize data quality error.

        Args:
            message: Error message.
            valid_count: Number of valid records.
            invalid_count: Number of invalid records.
            errors: List of validation errors.
            **kwargs: Additional arguments passed to parent.
        """
        details = kwargs.pop("details", {}) or {}
        if valid_count is not None:
            details["valid_count"] = valid_count
        if invalid_count is not None:
            details["invalid_count"] = invalid_count
        if errors:
            details["errors"] = errors[:10]  # Limit error list

        super().__init__(message, details=details, **kwargs)


class ScrapingError(DataIngestionError):
    """Web scraping error."""

    message = "Scraping error"
    error_code = ErrorCode.SCRAPING_ERROR


# ============================================================================
# Exception to HTTP Status Mapping
# ============================================================================


def get_http_status_for_exception(exc: Exception) -> HTTPStatus:
    """Get the appropriate HTTP status code for an exception.

    Args:
        exc: The exception to map.

    Returns:
        The appropriate HTTP status code.
    """
    if isinstance(exc, SignalForgeException):
        return exc.http_status

    # Map common built-in exceptions
    exception_status_map: dict[type, HTTPStatus] = {
        ValueError: HTTPStatus.BAD_REQUEST,
        TypeError: HTTPStatus.BAD_REQUEST,
        KeyError: HTTPStatus.NOT_FOUND,
        PermissionError: HTTPStatus.FORBIDDEN,
        TimeoutError: HTTPStatus.GATEWAY_TIMEOUT,
        ConnectionError: HTTPStatus.BAD_GATEWAY,
    }

    for exc_type, status in exception_status_map.items():
        if isinstance(exc, exc_type):
            return status

    return HTTPStatus.INTERNAL_SERVER_ERROR
