"""Structured logging configuration using structlog.

This module provides a comprehensive structured logging setup with:
- JSON output for production environments
- Human-readable console output for development
- Correlation ID tracking for request tracing
- Automatic context binding for all log entries
- Integration with standard library logging
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import structlog
from structlog.types import EventDict, Processor

if TYPE_CHECKING:
    pass

# Context variable for correlation ID tracking across async boundaries
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return correlation_id_ctx.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set correlation ID in context. Generates a new one if not provided."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    correlation_id_ctx.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID from context."""
    correlation_id_ctx.set(None)


def add_correlation_id(
    _logger: logging.Logger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add correlation ID to log event if present in context."""
    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_service_context(
    _logger: logging.Logger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add service-level context to all log events."""
    event_dict["service"] = "signalforge"
    return event_dict


def drop_color_message_key(
    _logger: logging.Logger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Drop the color_message key from the event dict.

    This key is added by uvicorn and is not needed in structured logs.
    """
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging(
    *,
    json_logs: bool = False,
    log_level: str = "INFO",
) -> None:
    """Configure structured logging for the application.

    Args:
        json_logs: If True, output logs in JSON format (recommended for production).
                   If False, use colored console output (recommended for development).
        log_level: The minimum log level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Shared processors for all log entries
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        add_correlation_id,
        add_service_context,
        drop_color_message_key,
    ]

    if json_logs:
        # Production: JSON output
        shared_processors.append(structlog.processors.format_exc_info)

        structlog.configure(
            processors=shared_processors
            + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    else:
        # Development: colored console output
        structlog.configure(
            processors=shared_processors
            + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        )

    # Configure root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    # Configure specific loggers
    # Reduce noise from third-party libraries
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logging.getLogger(logger_name).handlers.clear()
        logging.getLogger(logger_name).propagate = True

    # Suppress overly verbose loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: The logger name. If None, uses the calling module's name.

    Returns:
        A bound structlog logger instance.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


class LoggerMixin:
    """Mixin class that provides a logger property for classes.

    Usage:
        class MyService(LoggerMixin):
            def do_something(self):
                self.logger.info("doing_something", key="value")
    """

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get a logger bound to this class name."""
        return get_logger(self.__class__.__name__)


def bind_contextvars(**kwargs: Any) -> None:
    """Bind key-value pairs to the structlog context.

    These values will be included in all subsequent log entries
    until explicitly cleared.

    Args:
        **kwargs: Key-value pairs to bind to the logging context.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_contextvars(*keys: str) -> None:
    """Remove keys from the structlog context.

    Args:
        *keys: Keys to remove from the logging context.
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_contextvars() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
