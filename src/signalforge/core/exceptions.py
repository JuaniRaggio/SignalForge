"""Custom exceptions for SignalForge."""

from typing import Any


class SignalForgeException(Exception):
    """Base exception for SignalForge."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DatabaseError(SignalForgeException):
    """Database-related errors."""

    pass


class AuthenticationError(SignalForgeException):
    """Authentication-related errors."""

    pass


class AuthorizationError(SignalForgeException):
    """Authorization-related errors."""

    pass


class ValidationError(SignalForgeException):
    """Validation-related errors."""

    pass


class NotFoundError(SignalForgeException):
    """Resource not found errors."""

    pass


class RateLimitError(SignalForgeException):
    """Rate limit exceeded errors."""

    pass


class ExternalAPIError(SignalForgeException):
    """External API-related errors."""

    def __init__(
        self,
        message: str,
        source: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.status_code = status_code
        super().__init__(message, details)


class DataIngestionError(SignalForgeException):
    """Data ingestion-related errors."""

    pass
