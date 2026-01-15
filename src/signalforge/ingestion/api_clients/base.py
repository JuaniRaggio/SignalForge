"""Base class for API clients with retry and circuit breaker support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

from signalforge.core.logging import LoggerMixin, get_logger
from signalforge.core.retry import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryConfig,
    get_circuit_breaker,
)

logger = get_logger(__name__)


class BaseAPIClient(ABC, LoggerMixin):
    """Abstract base class for external API clients.

    Provides:
    - Structured logging via LoggerMixin
    - Configurable retry with exponential backoff
    - Optional circuit breaker protection
    """

    def __init__(
        self,
        *,
        retry_config: RetryConfig | None = None,
        circuit_breaker_name: str | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize API client.

        Args:
            retry_config: Configuration for retry behavior.
            circuit_breaker_name: Name for circuit breaker (enables if provided).
            circuit_breaker_config: Configuration for circuit breaker.
        """
        self.retry_config = retry_config or RetryConfig()
        self._circuit_breaker: CircuitBreaker | None = None

        if circuit_breaker_name:
            self._circuit_breaker = get_circuit_breaker(
                circuit_breaker_name,
                circuit_breaker_config,
            )

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        """Get the circuit breaker instance if configured."""
        return self._circuit_breaker

    @abstractmethod
    async def fetch_data(self, symbol: str, **kwargs: Any) -> pl.DataFrame:
        """Fetch data for a symbol. Must be implemented by subclasses.

        Args:
            symbol: The symbol to fetch data for.
            **kwargs: Additional parameters for the fetch operation.

        Returns:
            A Polars DataFrame containing the fetched data.

        Raises:
            ExternalAPIError: If the fetch operation fails.
        """
        pass
