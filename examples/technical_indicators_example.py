"""Example usage of technical indicators feature engineering module.

This script demonstrates how to:
1. Fetch price data from TimescaleDB
2. Compute technical indicators using Polars
3. Save results for downstream ML pipelines
"""

import asyncio
from datetime import datetime, timedelta

import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.config import get_settings
from signalforge.core.database import get_async_session_maker
from signalforge.core.logging import configure_logging, get_logger
from signalforge.ml.features.technical import (
    IndicatorConfig,
    compute_technical_indicators,
)
from signalforge.models.price import Price

# Configure logging
configure_logging(json_logs=False, log_level="INFO")
logger = get_logger(__name__)


async def fetch_price_data(
    session: AsyncSession,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> pl.DataFrame:
    """Fetch OHLCV price data from TimescaleDB.

    Args:
        session: Database session.
        symbol: Stock symbol to fetch.
        start_date: Start date for historical data.
        end_date: End date for historical data.

    Returns:
        Polars DataFrame with OHLCV data.
    """
    logger.info(
        "fetching_price_data",
        symbol=symbol,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    # Query TimescaleDB for price data
    stmt = (
        select(Price)
        .where(Price.symbol == symbol.upper())
        .where(Price.timestamp >= start_date)
        .where(Price.timestamp <= end_date)
        .order_by(Price.timestamp)
    )

    result = await session.execute(stmt)
    prices = result.scalars().all()

    if not prices:
        logger.warning("no_price_data_found", symbol=symbol)
        return pl.DataFrame()

    # Convert to Polars DataFrame
    data = {
        "symbol": [p.symbol for p in prices],
        "timestamp": [p.timestamp for p in prices],
        "open": [float(p.open) for p in prices],
        "high": [float(p.high) for p in prices],
        "low": [float(p.low) for p in prices],
        "close": [float(p.close) for p in prices],
        "volume": [p.volume for p in prices],
    }

    df = pl.DataFrame(data)
    logger.info("price_data_fetched", rows=df.height, symbol=symbol)

    return df


async def main() -> None:
    """Main function demonstrating technical indicators computation."""
    settings = get_settings()

    # Create async session
    async_session_maker = get_async_session_maker(settings.database_url)

    async with async_session_maker() as session:
        # Example 1: Fetch data for a single symbol
        symbol = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data

        df = await fetch_price_data(session, symbol, start_date, end_date)

        if df.height == 0:
            logger.error("no_data_available", symbol=symbol)
            return

        # Example 2: Compute indicators with default configuration
        logger.info("computing_default_indicators", symbol=symbol)
        df_with_indicators = compute_technical_indicators(df)

        # Display sample results
        logger.info("indicator_computation_complete")
        print("\n" + "=" * 80)
        print(f"Technical Indicators for {symbol}")
        print("=" * 80)
        print(f"\nDataFrame shape: {df_with_indicators.shape}")
        print(f"Total columns: {len(df_with_indicators.columns)}")
        print(f"\nColumns: {df_with_indicators.columns}")

        # Show last 5 rows with selected indicators
        selected_cols = [
            "timestamp",
            "close",
            "sma_20",
            "sma_50",
            "ema_12",
            "rsi_14",
            "macd",
            "bb_upper",
            "bb_lower",
            "atr_14",
        ]
        print("\nLast 5 rows (selected indicators):")
        print(df_with_indicators.select(selected_cols).tail(5))

        # Example 3: Custom configuration for day trading
        logger.info("computing_custom_indicators_for_day_trading")
        day_trading_config = IndicatorConfig(
            sma_periods=(5, 10, 20),
            ema_periods=(8, 13, 21),
            rsi_period=9,
            macd_fast=8,
            macd_slow=17,
            macd_signal=9,
            bb_period=15,
            atr_period=10,
        )

        df_day_trading = compute_technical_indicators(df, day_trading_config)
        print("\n" + "=" * 80)
        print("Day Trading Configuration Indicators")
        print("=" * 80)
        print(f"Columns: {df_day_trading.columns}")

        # Example 4: Analyze recent market conditions
        print("\n" + "=" * 80)
        print("Recent Market Analysis")
        print("=" * 80)

        latest = df_with_indicators.tail(1)
        if latest.height > 0:
            close_price = latest["close"][0]
            rsi = latest["rsi_14"][0]
            macd = latest["macd"][0]
            macd_signal = latest["macd_signal"][0]
            bb_upper = latest["bb_upper"][0]
            bb_lower = latest["bb_lower"][0]

            print(f"\nCurrent Price: ${close_price:.2f}")

            if rsi is not None:
                print(f"RSI(14): {rsi:.2f}", end=" - ")
                if rsi > 70:
                    print("OVERBOUGHT")
                elif rsi < 30:
                    print("OVERSOLD")
                else:
                    print("NEUTRAL")

            if macd is not None and macd_signal is not None:
                print(f"MACD: {macd:.4f}, Signal: {macd_signal:.4f}", end=" - ")
                if macd > macd_signal:
                    print("BULLISH MOMENTUM")
                else:
                    print("BEARISH MOMENTUM")

            if bb_upper is not None and bb_lower is not None:
                bb_position = (close_price - bb_lower) / (bb_upper - bb_lower) * 100
                print(f"Bollinger Band Position: {bb_position:.1f}%", end=" - ")
                if bb_position > 80:
                    print("NEAR UPPER BAND")
                elif bb_position < 20:
                    print("NEAR LOWER BAND")
                else:
                    print("WITHIN BANDS")

        # Example 5: Save results for ML pipeline
        output_file = f"/tmp/{symbol}_indicators.parquet"
        df_with_indicators.write_parquet(output_file)
        logger.info("indicators_saved", file=output_file)
        print(f"\nIndicators saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
