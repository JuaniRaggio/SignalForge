"""Yahoo Finance API client using yfinance."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, cast

import polars as pl
import yfinance as yf

from signalforge.core.exceptions import ExternalAPIError
from signalforge.ingestion.api_clients.base import BaseAPIClient

logger = logging.getLogger(__name__)

PeriodType = Literal[
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
]
IntervalType = Literal[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
]


class YahooFinanceClient(BaseAPIClient):
    """Client for fetching data from Yahoo Finance using yfinance."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> None:
        super().__init__(max_retries, base_delay, max_delay)
        self._executor = ThreadPoolExecutor(max_workers=4)

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
            if isinstance(timestamp_dtype, pl.Datetime):
                if timestamp_dtype.time_zone is None:
                    pl_df = pl_df.with_columns(
                        pl.col("timestamp").dt.replace_time_zone("UTC")
                    )

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

            return pl_df

        except ExternalAPIError:
            raise
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise ExternalAPIError(
                message=f"Failed to fetch data for {symbol}: {e!s}",
                source="yahoo_finance",
            )

    async def fetch_data(
        self,
        symbol: str,
        period: PeriodType = "1mo",
        interval: IntervalType = "1d",
        **_kwargs: Any,
    ) -> pl.DataFrame:
        """Fetch historical price data for a symbol."""
        loop = asyncio.get_event_loop()
        result = await self._retry_with_backoff(
            loop.run_in_executor,
            self._executor,
            self._fetch_sync,
            symbol,
            period,
            interval,
        )
        return cast(pl.DataFrame, result)

    async def fetch_multiple(
        self,
        symbols: list[str],
        period: PeriodType = "1mo",
        interval: IntervalType = "1d",
    ) -> dict[str, pl.DataFrame]:
        """Fetch data for multiple symbols concurrently."""
        tasks = [
            self.fetch_data(symbol, period=period, interval=interval)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data: dict[str, pl.DataFrame] = {}
        for symbol, result in zip(symbols, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Failed to fetch {symbol}: {result}")
            else:
                data[symbol] = result

        return data

    def close(self) -> None:
        """Cleanup resources."""
        self._executor.shutdown(wait=False)
