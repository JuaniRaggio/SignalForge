"""Yahoo Finance API client using yfinance with retry and circuit breaker."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import polars as pl
import yfinance as yf

from signalforge.core.exceptions import ExternalAPIError
from signalforge.core.logging import get_logger
from signalforge.core.retry import (
    YAHOO_FINANCE_CIRCUIT,
    YAHOO_FINANCE_RETRY,
    CircuitBreakerConfig,
    CircuitBreakerError,
    RetryConfig,
    retry_with_backoff,
)
from signalforge.ingestion.api_clients.base import BaseAPIClient

logger = get_logger(__name__)

PeriodType = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
IntervalType = Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
]


class YahooFinanceClient(BaseAPIClient):
    """Client for fetching data from Yahoo Finance using yfinance.

    Features:
    - Exponential backoff retry with jitter
    - Circuit breaker for API protection
    - Async-friendly with thread pool executor
    - Structured logging
    """

    def __init__(
        self,
        *,
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        enable_circuit_breaker: bool = True,
        max_workers: int = 4,
    ) -> None:
        """Initialize Yahoo Finance client.

        Args:
            retry_config: Configuration for retry behavior.
            circuit_breaker_config: Configuration for circuit breaker.
            enable_circuit_breaker: Whether to enable circuit breaker protection.
            max_workers: Number of thread pool workers for concurrent requests.
        """
        super().__init__(
            retry_config=retry_config or YAHOO_FINANCE_RETRY,
            circuit_breaker_name="yahoo_finance" if enable_circuit_breaker else None,
            circuit_breaker_config=circuit_breaker_config or YAHOO_FINANCE_CIRCUIT,
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _fetch_sync(
        self,
        symbol: str,
        period: PeriodType = "1mo",
        interval: IntervalType = "1d",
    ) -> pl.DataFrame:
        """Synchronous fetch using yfinance (runs in thread pool)."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise ExternalAPIError(
                    message=f"No data returned for symbol: {symbol}",
                    source="yahoo_finance",
                )

            df = df.reset_index()

            column_mapping = {
                "Date": "timestamp",
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adj_close",
            }

            rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=rename_cols)

            pl_df = pl.from_pandas(df)

            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in pl_df.columns:
                    raise ExternalAPIError(
                        message=f"Missing required column: {col}",
                        source="yahoo_finance",
                    )

            pl_df = pl_df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

            # Check if timestamp needs timezone
            timestamp_dtype = pl_df["timestamp"].dtype
            if isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone is None:
                pl_df = pl_df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

            select_cols = [
                "symbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            if "adj_close" in pl_df.columns:
                select_cols.append("adj_close")

            pl_df = pl_df.select(select_cols)

            pl_df = pl_df.with_columns(
                [
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Int64),
                ]
            )

            if "adj_close" in pl_df.columns:
                pl_df = pl_df.with_columns(pl.col("adj_close").cast(pl.Float64))

            logger.info(
                "yahoo_finance_fetch_success",
                symbol=symbol,
                period=period,
                interval=interval,
                rows=len(pl_df),
            )

            return pl_df

        except ExternalAPIError:
            raise
        except Exception as e:
            logger.error(
                "yahoo_finance_fetch_error",
                symbol=symbol,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise ExternalAPIError(
                message=f"Failed to fetch data for {symbol}: {e!s}",
                source="yahoo_finance",
            ) from e

    @retry_with_backoff(config=YAHOO_FINANCE_RETRY, circuit_breaker_name="yahoo_finance")
    async def _fetch_with_retry(
        self,
        symbol: str,
        period: PeriodType,
        interval: IntervalType,
    ) -> pl.DataFrame:
        """Fetch data with retry and circuit breaker protection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_sync,
            symbol,
            period,
            interval,
        )

    async def fetch_data(
        self,
        symbol: str,
        period: PeriodType = "1mo",
        interval: IntervalType = "1d",
        **_kwargs: Any,
    ) -> pl.DataFrame:
        """Fetch historical price data for a symbol.

        Args:
            symbol: The ticker symbol (e.g., "AAPL", "MSFT").
            period: Time period for data (e.g., "1mo", "1y").
            interval: Data interval (e.g., "1d", "1h").
            **_kwargs: Additional arguments (ignored).

        Returns:
            Polars DataFrame with OHLCV data.

        Raises:
            ExternalAPIError: If fetch fails after all retries.
            CircuitBreakerError: If circuit breaker is open.
        """
        try:
            return await self._fetch_with_retry(symbol, period, interval)
        except CircuitBreakerError as e:
            logger.warning(
                "yahoo_finance_circuit_open",
                symbol=symbol,
                recovery_time=e.recovery_time.isoformat(),
            )
            raise

    async def fetch_multiple(
        self,
        symbols: list[str],
        period: PeriodType = "1mo",
        interval: IntervalType = "1d",
    ) -> dict[str, pl.DataFrame]:
        """Fetch data for multiple symbols concurrently.

        Args:
            symbols: List of ticker symbols.
            period: Time period for data.
            interval: Data interval.

        Returns:
            Dictionary mapping symbols to their DataFrames.
            Failed fetches are logged but not included in results.
        """
        tasks = [self.fetch_data(symbol, period=period, interval=interval) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data: dict[str, pl.DataFrame] = {}
        for symbol, result in zip(symbols, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "yahoo_finance_batch_fetch_error",
                    symbol=symbol,
                    error_type=type(result).__name__,
                    error_message=str(result),
                )
            else:
                data[symbol] = result

        logger.info(
            "yahoo_finance_batch_fetch_complete",
            total_symbols=len(symbols),
            successful=len(data),
            failed=len(symbols) - len(data),
        )

        return data

    def close(self) -> None:
        """Cleanup resources."""
        self._executor.shutdown(wait=False)
        logger.debug("yahoo_finance_client_closed")

    async def __aenter__(self) -> YahooFinanceClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        self.close()
