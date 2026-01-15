"""Retry utilities with exponential backoff and circuit breaker.

This module provides robust retry mechanisms for external API calls with:
- Exponential backoff with jitter
- Circuit breaker pattern for failing services
- Configurable retry policies
- Structured logging integration
- Support for both sync and async functions
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from functools import wraps
from typing import ParamSpec, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from signalforge.core.logging import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (including initial).
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff calculation.
        jitter_max: Maximum jitter in seconds to add to delays.
        retry_exceptions: Exception types that trigger retries.
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_max: float = 1.0
    retry_exceptions: tuple[type[Exception], ...] = (Exception,)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Time in seconds before attempting recovery.
        half_open_max_calls: Max calls allowed in half-open state.
        success_threshold: Successes needed in half-open to close circuit.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


@dataclass
class CircuitBreakerState:
    """Internal state for a circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    half_open_calls: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, recovery_time: datetime) -> None:
        self.name = name
        self.recovery_time = recovery_time
        super().__init__(
            f"Circuit breaker '{name}' is open. Recovery at: {recovery_time.isoformat()}"
        )


class CircuitBreaker:
    """Circuit breaker implementation for external service calls.

    Prevents cascading failures by failing fast when a service is unhealthy.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker.
            config: Circuit breaker configuration.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state.state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state.state == CircuitState.OPEN

    async def _check_state_transition(self) -> None:
        """Check and handle state transitions."""
        if self._state.state == CircuitState.OPEN and self._state.last_failure_time:
            recovery_time = self._state.last_failure_time + timedelta(
                seconds=self.config.recovery_timeout
            )
            if datetime.now(UTC) >= recovery_time:
                logger.info(
                    "circuit_breaker_half_open",
                    name=self.name,
                    reason="recovery_timeout_elapsed",
                )
                self._state.state = CircuitState.HALF_OPEN
                self._state.half_open_calls = 0
                self._state.success_count = 0

    async def _should_allow_request(self) -> bool:
        """Determine if request should be allowed."""
        await self._check_state_transition()

        if self._state.state == CircuitState.CLOSED:
            return True

        if self._state.state == CircuitState.OPEN:
            recovery_time = (
                self._state.last_failure_time + timedelta(seconds=self.config.recovery_timeout)
                if self._state.last_failure_time
                else datetime.now(UTC)
            )
            raise CircuitBreakerError(self.name, recovery_time)

        if self._state.state == CircuitState.HALF_OPEN:
            if self._state.half_open_calls < self.config.half_open_max_calls:
                self._state.half_open_calls += 1
                return True
            recovery_time = datetime.now(UTC) + timedelta(seconds=self.config.recovery_timeout)
            raise CircuitBreakerError(self.name, recovery_time)

        return False

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            if self._state.state == CircuitState.HALF_OPEN:
                self._state.success_count += 1
                if self._state.success_count >= self.config.success_threshold:
                    logger.info(
                        "circuit_breaker_closed",
                        name=self.name,
                        reason="success_threshold_reached",
                    )
                    self._state.state = CircuitState.CLOSED
                    self._state.failure_count = 0
            elif self._state.state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._state.failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = datetime.now(UTC)

            if self._state.state == CircuitState.HALF_OPEN:
                logger.warning(
                    "circuit_breaker_reopened",
                    name=self.name,
                    reason="failure_in_half_open",
                )
                self._state.state = CircuitState.OPEN
                self._state.half_open_calls = 0
            elif self._state.state == CircuitState.CLOSED:
                if self._state.failure_count >= self.config.failure_threshold:
                    logger.warning(
                        "circuit_breaker_opened",
                        name=self.name,
                        failure_count=self._state.failure_count,
                        threshold=self.config.failure_threshold,
                    )
                    self._state.state = CircuitState.OPEN

    async def call(
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: The function to call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The function's return value.

        Raises:
            CircuitBreakerError: If circuit is open.
            Exception: If the function raises an exception.
        """
        async with self._lock:
            await self._should_allow_request()

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            await self.record_success()
            return result  # type: ignore[no-any-return]
        except Exception:
            await self.record_failure()
            raise

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitBreakerState()
        logger.info("circuit_breaker_reset", name=self.name)


# Global registry of circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name.

    Args:
        name: Unique identifier for the circuit breaker.
        config: Configuration for new circuit breaker.

    Returns:
        The circuit breaker instance.
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def retry_with_backoff(
    config: RetryConfig | None = None,
    circuit_breaker_name: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for adding retry with exponential backoff.

    Args:
        config: Retry configuration.
        circuit_breaker_name: Optional circuit breaker to use.

    Returns:
        Decorated function with retry behavior.
    """
    config = config or RetryConfig()

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            circuit_breaker = (
                get_circuit_breaker(circuit_breaker_name) if circuit_breaker_name else None
            )

            async def _execute() -> T:
                if circuit_breaker:
                    return await circuit_breaker.call(func, *args, **kwargs)
                return await func(*args, **kwargs)  # type: ignore[misc,no-any-return]

            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(config.max_attempts),
                    wait=wait_exponential_jitter(
                        initial=config.initial_delay,
                        max=config.max_delay,
                        jitter=config.jitter_max,
                    ),
                    retry=retry_if_exception_type(config.retry_exceptions),
                    reraise=True,
                ):
                    with attempt:
                        logger.debug(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt.retry_state.attempt_number,
                            max_attempts=config.max_attempts,
                        )
                        return await _execute()
            except RetryError as e:
                logger.error(
                    "retry_exhausted",
                    function=func.__name__,
                    attempts=config.max_attempts,
                    last_exception=str(e.last_attempt.exception()),
                )
                raise

            # This should never be reached, but satisfies type checker
            raise RuntimeError("Retry loop exited unexpectedly")

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                for attempt in Retrying(
                    stop=stop_after_attempt(config.max_attempts),
                    wait=wait_exponential_jitter(
                        initial=config.initial_delay,
                        max=config.max_delay,
                        jitter=config.jitter_max,
                    ),
                    retry=retry_if_exception_type(config.retry_exceptions),
                    reraise=True,
                ):
                    with attempt:
                        logger.debug(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt.retry_state.attempt_number,
                            max_attempts=config.max_attempts,
                        )
                        return func(*args, **kwargs)
            except RetryError as e:
                logger.error(
                    "retry_exhausted",
                    function=func.__name__,
                    attempts=config.max_attempts,
                    last_exception=str(e.last_attempt.exception()),
                )
                raise

            # This should never be reached
            raise RuntimeError("Retry loop exited unexpectedly")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper

    return decorator


# Pre-configured retry policies for common use cases
YAHOO_FINANCE_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=30.0,
    jitter_max=2.0,
    retry_exceptions=(ConnectionError, TimeoutError, Exception),
)

RSS_SCRAPER_RETRY = RetryConfig(
    max_attempts=2,
    initial_delay=0.5,
    max_delay=10.0,
    jitter_max=1.0,
    retry_exceptions=(ConnectionError, TimeoutError),
)

EXTERNAL_API_RETRY = RetryConfig(
    max_attempts=4,
    initial_delay=1.0,
    max_delay=60.0,
    jitter_max=3.0,
    retry_exceptions=(Exception,),
)

# Pre-configured circuit breaker configs
YAHOO_FINANCE_CIRCUIT = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    half_open_max_calls=2,
    success_threshold=2,
)

RSS_SCRAPER_CIRCUIT = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    half_open_max_calls=1,
    success_threshold=1,
)
