"""Exception handlers for FastAPI.

This module provides centralized exception handling with:
- Structured error responses
- Correlation ID tracking
- Appropriate HTTP status codes
- Logging integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from signalforge.core.exceptions import (
    ErrorCode,
    SignalForgeException,
    get_http_status_for_exception,
)
from signalforge.core.logging import get_correlation_id, get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger(__name__)


def create_error_response(
    error_code: str,
    message: str,
    details: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Create a standardized error response.

    Args:
        error_code: Machine-readable error code.
        message: Human-readable error message.
        details: Additional error details.
        correlation_id: Request correlation ID.

    Returns:
        Standardized error response dictionary.
    """
    response: dict[str, Any] = {
        "success": False,
        "error": {
            "code": error_code,
            "message": message,
        },
    }

    if details:
        response["error"]["details"] = details

    if correlation_id:
        response["correlation_id"] = correlation_id

    return response


async def signalforge_exception_handler(
    request: Request,
    exc: SignalForgeException,
) -> JSONResponse:
    """Handle SignalForgeException and subclasses.

    Returns a structured JSON response with appropriate status code.
    """
    correlation_id = get_correlation_id()

    logger.warning(
        "signalforge_exception",
        error_code=exc.error_code.value,
        error_message=exc.message,
        http_status=exc.http_status.value,
        path=request.url.path,
        details=exc.details,
    )

    return JSONResponse(
        status_code=exc.http_status.value,
        content=create_error_response(
            error_code=exc.error_code.value,
            message=exc.user_message,
            details=exc.details if exc.details else None,
            correlation_id=correlation_id,
        ),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle FastAPI request validation errors.

    Transforms Pydantic validation errors into user-friendly responses.
    """
    correlation_id = get_correlation_id()

    # Extract validation error details
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        errors.append(
            {
                "field": field_path,
                "message": error["msg"],
                "type": error["type"],
            }
        )

    logger.warning(
        "validation_error",
        path=request.url.path,
        error_count=len(errors),
        errors=errors[:5],  # Log first 5 errors
    )

    return JSONResponse(
        status_code=400,
        content=create_error_response(
            error_code=ErrorCode.VALIDATION_ERROR.value,
            message="Request validation failed",
            details={"validation_errors": errors},
            correlation_id=correlation_id,
        ),
    )


async def pydantic_exception_handler(
    request: Request,
    exc: PydanticValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    correlation_id = get_correlation_id()

    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        errors.append(
            {
                "field": field_path,
                "message": error["msg"],
                "type": error["type"],
            }
        )

    logger.warning(
        "pydantic_validation_error",
        path=request.url.path,
        error_count=len(errors),
    )

    return JSONResponse(
        status_code=400,
        content=create_error_response(
            error_code=ErrorCode.VALIDATION_ERROR.value,
            message="Data validation failed",
            details={"validation_errors": errors},
            correlation_id=correlation_id,
        ),
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle Starlette HTTP exceptions.

    Maps standard HTTP exceptions to our error response format.
    """
    correlation_id = get_correlation_id()

    # Map status codes to error codes
    status_to_error_code = {
        400: ErrorCode.INVALID_INPUT,
        401: ErrorCode.AUTHENTICATION_REQUIRED,
        403: ErrorCode.PERMISSION_DENIED,
        404: ErrorCode.RESOURCE_NOT_FOUND,
        405: ErrorCode.INVALID_INPUT,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
        502: ErrorCode.EXTERNAL_API_ERROR,
        503: ErrorCode.EXTERNAL_API_ERROR,
        504: ErrorCode.API_TIMEOUT,
    }

    error_code = status_to_error_code.get(exc.status_code, ErrorCode.UNKNOWN_ERROR)

    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            error_code=error_code.value,
            message=str(exc.detail) if exc.detail else "An error occurred",
            correlation_id=correlation_id,
        ),
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions.

    Logs the full exception and returns a generic error response
    to avoid leaking implementation details.
    """
    correlation_id = get_correlation_id()
    http_status = get_http_status_for_exception(exc)

    logger.exception(
        "unhandled_exception",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
    )

    return JSONResponse(
        status_code=http_status.value,
        content=create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR.value,
            message="An unexpected error occurred. Please try again later.",
            correlation_id=correlation_id,
        ),
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """
    app.add_exception_handler(SignalForgeException, signalforge_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(PydanticValidationError, pydantic_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, generic_exception_handler)
