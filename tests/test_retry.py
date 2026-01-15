"""Tests for retry and circuit breaker functionality."""

import asyncio
import contextlib

import pytest

from signalforge.core.retry import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    RetryConfig,
    get_circuit_breaker,
    retry_with_backoff,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self) -> None:
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0

    def test_custom_config(self) -> None:
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            retry_exceptions=(ValueError, TypeError),
        )
        assert config.max_attempts == 5
        assert config.retry_exceptions == (ValueError, TypeError)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.fixture
    def circuit_breaker(self) -> CircuitBreaker:
        """Create a circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            half_open_max_calls=2,
            success_threshold=2,
        )
        return CircuitBreaker("test_circuit", config)

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, circuit_breaker: CircuitBreaker) -> None:
        """Test that circuit starts in closed state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.is_closed
        assert not circuit_breaker.is_open

    @pytest.mark.asyncio
    async def test_success_keeps_circuit_closed(self, circuit_breaker: CircuitBreaker) -> None:
        """Test that successful calls keep circuit closed."""

        async def success_func() -> str:
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.is_closed

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self, circuit_breaker: CircuitBreaker) -> None:
        """Test that enough failures open the circuit."""

        async def failing_func() -> None:
            raise ValueError("test error")

        # Cause failures up to threshold
        for _ in range(3):  # failure_threshold = 3
            with contextlib.suppress(ValueError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.is_open

    @pytest.mark.asyncio
    async def test_open_circuit_raises_error(self, circuit_breaker: CircuitBreaker) -> None:
        """Test that open circuit raises CircuitBreakerError."""

        async def failing_func() -> None:
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with contextlib.suppress(ValueError):
                await circuit_breaker.call(failing_func)

        # Now circuit should be open
        async def success_func() -> str:
            return "success"

        with pytest.raises(CircuitBreakerError) as exc_info:
            await circuit_breaker.call(success_func)

        assert "test_circuit" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_recovers_after_timeout(self, circuit_breaker: CircuitBreaker) -> None:
        """Test that circuit transitions to half-open after timeout."""

        async def failing_func() -> None:
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with contextlib.suppress(ValueError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.is_open

        # Wait for recovery timeout
        await asyncio.sleep(1.5)

        # Next call should transition to half-open
        async def success_func() -> str:
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_closes_on_success(self, circuit_breaker: CircuitBreaker) -> None:
        """Test that half-open circuit closes after enough successes."""

        async def failing_func() -> None:
            raise ValueError("test error")

        async def success_func() -> str:
            return "success"

        # Open the circuit
        for _ in range(3):
            with contextlib.suppress(ValueError):
                await circuit_breaker.call(failing_func)

        # Wait for recovery
        await asyncio.sleep(1.5)

        # Succeed enough times to close circuit
        for _ in range(2):  # success_threshold = 2
            await circuit_breaker.call(success_func)

        assert circuit_breaker.is_closed

    @pytest.mark.asyncio
    async def test_reset_returns_to_closed(self, circuit_breaker: CircuitBreaker) -> None:
        """Test that reset returns circuit to closed state."""

        async def failing_func() -> None:
            raise ValueError("test error")

        # Open the circuit
        for _ in range(3):
            with contextlib.suppress(ValueError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.is_open

        # Reset
        circuit_breaker.reset()
        assert circuit_breaker.is_closed


class TestRetryDecorator:
    """Tests for retry_with_backoff decorator."""

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self) -> None:
        """Test that successful calls don't trigger retries."""
        call_count = 0

        @retry_with_backoff(config=RetryConfig(max_attempts=3))
        async def success_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        """Test that failures trigger retries."""
        call_count = 0

        @retry_with_backoff(
            config=RetryConfig(
                max_attempts=3,
                initial_delay=0.01,
                max_delay=0.1,
            )
        )
        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        """Test that retries stop after max attempts."""
        call_count = 0

        @retry_with_backoff(
            config=RetryConfig(
                max_attempts=3,
                initial_delay=0.01,
                max_delay=0.1,
            )
        )
        async def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("persistent error")

        with pytest.raises(ValueError):
            await always_fails()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_only_retries_specified_exceptions(self) -> None:
        """Test that only specified exceptions trigger retries."""
        call_count = 0

        @retry_with_backoff(
            config=RetryConfig(
                max_attempts=3,
                initial_delay=0.01,
                retry_exceptions=(ValueError,),
            )
        )
        async def specific_error() -> None:
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong error type")

        with pytest.raises(TypeError):
            await specific_error()

        assert call_count == 1  # Should not retry for TypeError


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_get_creates_new_circuit_breaker(self) -> None:
        """Test that get_circuit_breaker creates new instances."""
        cb = get_circuit_breaker("new_test_circuit")
        assert cb is not None
        assert cb.name == "new_test_circuit"

    def test_get_returns_same_instance(self) -> None:
        """Test that get_circuit_breaker returns same instance for same name."""
        cb1 = get_circuit_breaker("singleton_test")
        cb2 = get_circuit_breaker("singleton_test")
        assert cb1 is cb2

    def test_different_names_different_instances(self) -> None:
        """Test that different names create different instances."""
        cb1 = get_circuit_breaker("circuit_a")
        cb2 = get_circuit_breaker("circuit_b")
        assert cb1 is not cb2
