"""Request logging middleware for FastAPI.

Provides structured logging for all HTTP requests with:
- Correlation ID injection and propagation
- Request/response timing
- Error tracking
- Contextual information binding
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from signalforge.core.logging import (
    bind_contextvars,
    clear_contextvars,
    get_logger,
    set_correlation_id,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger(__name__)

# Header name for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds structured logging to all requests.

    Features:
    - Extracts or generates correlation ID
    - Logs request start and completion
    - Tracks request duration
    - Binds contextual information for all logs during request lifecycle
    - Adds correlation ID to response headers
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request with structured logging."""
        # Clear any existing context from previous requests
        clear_contextvars()

        # Extract or generate correlation ID
        correlation_id = request.headers.get(CORRELATION_ID_HEADER)
        if not correlation_id:
            correlation_id = str(uuid4())

        # Set correlation ID in context
        set_correlation_id(correlation_id)

        # Generate unique request ID
        request_id = str(uuid4())

        # Bind request context for all subsequent logs
        bind_contextvars(
            correlation_id=correlation_id,
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params) if request.query_params else None,
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )

        # Track timing
        start_time = time.perf_counter()

        # Log request start
        logger.info(
            "request_started",
            url=str(request.url),
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log successful completion
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            # Add correlation ID to response headers
            response.headers[CORRELATION_ID_HEADER] = correlation_id
            response.headers[REQUEST_ID_HEADER] = request_id

            return response

        except Exception as exc:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log exception
            logger.exception(
                "request_failed",
                duration_ms=round(duration_ms, 2),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise

        finally:
            # Clean up context
            clear_contextvars()

    def _get_client_ip(self, request: Request) -> str | None:
        """Extract client IP from request, considering proxies."""
        # Check X-Forwarded-For header (common with reverse proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header (nginx)
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client
        if request.client:
            return request.client.host

        return None


class RequestLoggingMiddleware(LoggingMiddleware):
    """Alias for LoggingMiddleware for backwards compatibility."""

    pass


def setup_logging_middleware(app: FastAPI) -> None:
    """Add logging middleware to FastAPI application.

    Args:
        app: The FastAPI application instance.
    """
    app.add_middleware(LoggingMiddleware)
