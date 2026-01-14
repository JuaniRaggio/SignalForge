"""Base class for API clients."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class BaseAPIClient(ABC):
    """Abstract base class for external API clients."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def _retry_with_backoff(
        self,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.base_delay * (2**attempt),
                        self.max_delay,
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in retry logic")

    @abstractmethod
    async def fetch_data(self, symbol: str, **kwargs: Any) -> pl.DataFrame:
        """Fetch data for a symbol. Must be implemented by subclasses."""
        pass
