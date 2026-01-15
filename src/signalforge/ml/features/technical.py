"""Technical indicators computation using Polars.

This module provides efficient computation of common technical indicators
for financial time series data using the Polars DataFrame library.

All indicators handle missing values appropriately and maintain type safety
with strict mypy compliance.

Examples:
    Basic usage with default configuration:

    >>> import polars as pl
    >>> from signalforge.ml.features.technical import compute_technical_indicators
    >>>
    >>> df = pl.DataFrame({
    ...     "symbol": ["AAPL"] * 100,
    ...     "timestamp": pl.date_range(start="2024-01-01", periods=100, interval="1d"),
    ...     "open": [150.0] * 100,
    ...     "high": [155.0] * 100,
    ...     "low": [148.0] * 100,
    ...     "close": [152.0] * 100,
    ...     "volume": [1000000] * 100,
    ... })
    >>> result = compute_technical_indicators(df)

    Custom configuration:

    >>> from signalforge.ml.features.technical import IndicatorConfig
    >>> config = IndicatorConfig(
    ...     sma_periods=[5, 10],
    ...     ema_periods=[8, 21],
    ...     rsi_period=10,
    ... )
    >>> result = compute_technical_indicators(df, config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from signalforge.core.logging import get_logger

logger = get_logger(__name__)

# Default configuration constants
DEFAULT_SMA_PERIODS: Final[tuple[int, ...]] = (10, 20, 50, 200)
DEFAULT_EMA_PERIODS: Final[tuple[int, ...]] = (12, 26)
DEFAULT_RSI_PERIOD: Final[int] = 14
DEFAULT_MACD_FAST: Final[int] = 12
DEFAULT_MACD_SLOW: Final[int] = 26
DEFAULT_MACD_SIGNAL: Final[int] = 9
DEFAULT_BB_PERIOD: Final[int] = 20
DEFAULT_BB_STD: Final[float] = 2.0
DEFAULT_ATR_PERIOD: Final[int] = 14
DEFAULT_VOLUME_SMA_PERIOD: Final[int] = 20


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators computation.

    Attributes:
        sma_periods: Periods for Simple Moving Average calculation.
        ema_periods: Periods for Exponential Moving Average calculation.
        rsi_period: Period for RSI calculation.
        macd_fast: Fast period for MACD calculation.
        macd_slow: Slow period for MACD calculation.
        macd_signal: Signal period for MACD calculation.
        bb_period: Period for Bollinger Bands calculation.
        bb_std: Standard deviation multiplier for Bollinger Bands.
        atr_period: Period for Average True Range calculation.
        volume_sma_period: Period for Volume SMA calculation.
    """

    sma_periods: tuple[int, ...] = DEFAULT_SMA_PERIODS
    ema_periods: tuple[int, ...] = DEFAULT_EMA_PERIODS
    rsi_period: int = DEFAULT_RSI_PERIOD
    macd_fast: int = DEFAULT_MACD_FAST
    macd_slow: int = DEFAULT_MACD_SLOW
    macd_signal: int = DEFAULT_MACD_SIGNAL
    bb_period: int = DEFAULT_BB_PERIOD
    bb_std: float = DEFAULT_BB_STD
    atr_period: int = DEFAULT_ATR_PERIOD
    volume_sma_period: int = DEFAULT_VOLUME_SMA_PERIOD

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if any(p <= 0 for p in self.sma_periods):
            raise ValueError("All SMA periods must be positive")
        if any(p <= 0 for p in self.ema_periods):
            raise ValueError("All EMA periods must be positive")
        if self.rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        if self.macd_fast <= 0 or self.macd_slow <= 0 or self.macd_signal <= 0:
            raise ValueError("All MACD periods must be positive")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("MACD fast period must be less than slow period")
        if self.bb_period <= 0:
            raise ValueError("Bollinger Bands period must be positive")
        if self.bb_std <= 0:
            raise ValueError("Bollinger Bands std must be positive")
        if self.atr_period <= 0:
            raise ValueError("ATR period must be positive")
        if self.volume_sma_period <= 0:
            raise ValueError("Volume SMA period must be positive")


def compute_sma(df: pl.DataFrame, period: int, column: str = "close") -> pl.DataFrame:
    """Compute Simple Moving Average.

    Args:
        df: Input DataFrame with price data.
        period: Number of periods for the moving average.
        column: Column name to compute SMA on.

    Returns:
        DataFrame with added SMA column.

    Raises:
        ValueError: If the specified column does not exist.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    col_name = f"sma_{period}" if column == "close" else f"{column}_sma_{period}"

    return df.with_columns(pl.col(column).rolling_mean(window_size=period).alias(col_name))


def compute_ema(df: pl.DataFrame, period: int, column: str = "close") -> pl.DataFrame:
    """Compute Exponential Moving Average.

    Args:
        df: Input DataFrame with price data.
        period: Number of periods for the exponential moving average.
        column: Column name to compute EMA on.

    Returns:
        DataFrame with added EMA column.

    Raises:
        ValueError: If the specified column does not exist.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    col_name = f"ema_{period}" if column == "close" else f"{column}_ema_{period}"

    # EMA uses exponential smoothing with alpha = 2 / (period + 1)
    return df.with_columns(pl.col(column).ewm_mean(span=period, adjust=False).alias(col_name))


def compute_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Compute Relative Strength Index.

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Args:
        df: Input DataFrame with price data.
        period: Number of periods for RSI calculation.

    Returns:
        DataFrame with added RSI column.

    Raises:
        ValueError: If 'close' column does not exist.
    """
    if "close" not in df.columns:
        raise ValueError("Column 'close' not found in DataFrame")

    # Calculate price changes
    df_with_delta = df.with_columns((pl.col("close") - pl.col("close").shift(1)).alias("delta"))

    # Separate gains and losses
    df_with_gains = df_with_delta.with_columns(
        pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain"),
        pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0.0).alias("loss"),
    )

    # Calculate average gains and losses using EMA
    df_with_avg = df_with_gains.with_columns(
        pl.col("gain").ewm_mean(span=period, adjust=False).alias("avg_gain"),
        pl.col("loss").ewm_mean(span=period, adjust=False).alias("avg_loss"),
    )

    # Calculate RSI
    result = df_with_avg.with_columns(
        pl.when(pl.col("avg_loss") == 0)
        .then(100.0)
        .otherwise(100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / pl.col("avg_loss")))))
        .alias(f"rsi_{period}")
    )

    # Drop intermediate columns
    return result.drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])


def compute_macd(
    df: pl.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pl.DataFrame:
    """Compute Moving Average Convergence Divergence.

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.

    Args:
        df: Input DataFrame with price data.
        fast_period: Period for fast EMA.
        slow_period: Period for slow EMA.
        signal_period: Period for signal line EMA.

    Returns:
        DataFrame with added MACD, MACD signal, and MACD histogram columns.

    Raises:
        ValueError: If 'close' column does not exist or periods are invalid.
    """
    if "close" not in df.columns:
        raise ValueError("Column 'close' not found in DataFrame")
    if fast_period >= slow_period:
        raise ValueError("Fast period must be less than slow period")

    # Calculate fast and slow EMAs
    df_with_ema = df.with_columns(
        pl.col("close").ewm_mean(span=fast_period, adjust=False).alias("ema_fast"),
        pl.col("close").ewm_mean(span=slow_period, adjust=False).alias("ema_slow"),
    )

    # Calculate MACD line
    df_with_macd = df_with_ema.with_columns((pl.col("ema_fast") - pl.col("ema_slow")).alias("macd"))

    # Calculate signal line
    df_with_signal = df_with_macd.with_columns(
        pl.col("macd").ewm_mean(span=signal_period, adjust=False).alias("macd_signal")
    )

    # Calculate histogram
    result = df_with_signal.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram")
    )

    # Drop intermediate columns
    return result.drop(["ema_fast", "ema_slow"])


def compute_bollinger_bands(
    df: pl.DataFrame,
    period: int = 20,
    std_multiplier: float = 2.0,
) -> pl.DataFrame:
    """Compute Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) and two outer bands
    that are standard deviations away from the middle band.

    Args:
        df: Input DataFrame with price data.
        period: Number of periods for the middle band (SMA).
        std_multiplier: Number of standard deviations for the outer bands.

    Returns:
        DataFrame with added Bollinger Bands columns.

    Raises:
        ValueError: If 'close' column does not exist.
    """
    if "close" not in df.columns:
        raise ValueError("Column 'close' not found in DataFrame")

    # Calculate middle band (SMA)
    df_with_sma = df.with_columns(
        pl.col("close").rolling_mean(window_size=period).alias("bb_middle")
    )

    # Calculate standard deviation
    df_with_std = df_with_sma.with_columns(
        pl.col("close").rolling_std(window_size=period).alias("rolling_std")
    )

    # Calculate upper and lower bands
    result = df_with_std.with_columns(
        (pl.col("bb_middle") + (pl.col("rolling_std") * std_multiplier)).alias("bb_upper"),
        (pl.col("bb_middle") - (pl.col("rolling_std") * std_multiplier)).alias("bb_lower"),
    )

    # Drop intermediate columns
    return result.drop(["rolling_std"])


def compute_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Compute Average True Range.

    ATR measures market volatility by calculating the average of true ranges
    over a specified period.

    Args:
        df: Input DataFrame with OHLC data.
        period: Number of periods for ATR calculation.

    Returns:
        DataFrame with added ATR column.

    Raises:
        ValueError: If required columns do not exist.
    """
    required_cols = ["high", "low", "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Calculate True Range components
    df_with_tr = df.with_columns(
        # High - Low
        (pl.col("high") - pl.col("low")).alias("hl"),
        # |High - Previous Close|
        (pl.col("high") - pl.col("close").shift(1)).abs().alias("hc"),
        # |Low - Previous Close|
        (pl.col("low") - pl.col("close").shift(1)).abs().alias("lc"),
    )

    # True Range is the maximum of the three components
    df_with_tr_max = df_with_tr.with_columns(
        pl.max_horizontal("hl", "hc", "lc").alias("true_range")
    )

    # Calculate ATR using EMA of True Range
    result = df_with_tr_max.with_columns(
        pl.col("true_range").ewm_mean(span=period, adjust=False).alias(f"atr_{period}")
    )

    # Drop intermediate columns
    return result.drop(["hl", "hc", "lc", "true_range"])


def compute_volume_sma(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """Compute Volume Simple Moving Average.

    Args:
        df: Input DataFrame with volume data.
        period: Number of periods for volume SMA.

    Returns:
        DataFrame with added volume SMA column.

    Raises:
        ValueError: If 'volume' column does not exist.
    """
    if "volume" not in df.columns:
        raise ValueError("Column 'volume' not found in DataFrame")

    return df.with_columns(
        pl.col("volume").rolling_mean(window_size=period).alias(f"volume_sma_{period}")
    )


def compute_technical_indicators(
    df: pl.DataFrame,
    config: IndicatorConfig | None = None,
) -> pl.DataFrame:
    """Compute all technical indicators for the given DataFrame.

    This function computes multiple technical indicators in a single pass,
    efficiently handling missing values and maintaining data integrity.

    Args:
        df: Input DataFrame with OHLCV data. Must contain columns:
            - symbol: str
            - timestamp: datetime
            - open: float
            - high: float
            - low: float
            - close: float
            - volume: int
        config: Configuration for indicator parameters. If None, uses defaults.

    Returns:
        DataFrame with all original columns plus computed indicator columns.

    Raises:
        ValueError: If required columns are missing or data is invalid.

    Examples:
        >>> import polars as pl
        >>> from datetime import datetime, timedelta
        >>>
        >>> # Create sample data
        >>> dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        >>> df = pl.DataFrame({
        ...     "symbol": ["AAPL"] * 100,
        ...     "timestamp": dates,
        ...     "open": [150.0 + i * 0.5 for i in range(100)],
        ...     "high": [155.0 + i * 0.5 for i in range(100)],
        ...     "low": [148.0 + i * 0.5 for i in range(100)],
        ...     "close": [152.0 + i * 0.5 for i in range(100)],
        ...     "volume": [1000000] * 100,
        ... })
        >>>
        >>> # Compute indicators
        >>> result = compute_technical_indicators(df)
        >>> print(result.columns)
    """
    if config is None:
        config = IndicatorConfig()

    # Validate required columns
    required_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate data is not empty
    if df.height == 0:
        logger.warning("compute_technical_indicators called with empty DataFrame")
        return df

    logger.info(
        "computing_technical_indicators",
        rows=df.height,
        symbols=df.select(pl.col("symbol").n_unique()).item(),
    )

    result = df

    # Compute Simple Moving Averages
    for period in config.sma_periods:
        result = compute_sma(result, period)

    # Compute Exponential Moving Averages
    for period in config.ema_periods:
        result = compute_ema(result, period)

    # Compute RSI
    result = compute_rsi(result, config.rsi_period)

    # Compute MACD
    result = compute_macd(
        result,
        config.macd_fast,
        config.macd_slow,
        config.macd_signal,
    )

    # Compute Bollinger Bands
    result = compute_bollinger_bands(result, config.bb_period, config.bb_std)

    # Compute ATR
    result = compute_atr(result, config.atr_period)

    # Compute Volume SMA
    result = compute_volume_sma(result, config.volume_sma_period)

    logger.info(
        "technical_indicators_computed",
        total_columns=len(result.columns),
        indicator_columns=len(result.columns) - len(required_cols),
    )

    return result
