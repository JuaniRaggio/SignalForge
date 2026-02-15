"""Technical indicators computation using Polars.

This module provides efficient computation of comprehensive technical indicators
for financial time series data using the Polars DataFrame library.

All indicators handle missing values appropriately and maintain type safety
with strict mypy compliance.

Examples:
    Basic usage with default configuration:

    >>> import polars as pl
    >>> from signalforge.ml.features.technical import TechnicalFeatureEngine
    >>>
    >>> df = pl.DataFrame({
    ...     "timestamp": pl.date_range(start="2024-01-01", periods=100, interval="1d"),
    ...     "open": [150.0] * 100,
    ...     "high": [155.0] * 100,
    ...     "low": [148.0] * 100,
    ...     "close": [152.0] * 100,
    ...     "volume": [1000000] * 100,
    ... })
    >>> engine = TechnicalFeatureEngine()
    >>> result = engine.compute_all(df)

    Custom configuration:

    >>> from signalforge.ml.features.technical import FeatureConfig
    >>> config = FeatureConfig(
    ...     sma_periods=[5, 10],
    ...     ema_periods=[8, 21],
    ...     rsi_periods=[10, 14],
    ... )
    >>> engine = TechnicalFeatureEngine(config)
    >>> result = engine.compute_all(df)
"""

from __future__ import annotations

from typing import Final

import polars as pl
from pydantic import BaseModel, Field

from signalforge.core.logging import get_logger

logger = get_logger(__name__)

# Default configuration constants
DEFAULT_SMA_PERIODS: Final[list[int]] = [5, 10, 20, 50, 200]
DEFAULT_EMA_PERIODS: Final[list[int]] = [5, 10, 20, 50]
DEFAULT_RSI_PERIODS: Final[list[int]] = [14, 21]
DEFAULT_ROC_PERIODS: Final[list[int]] = [5, 10, 20]
DEFAULT_MOM_PERIODS: Final[list[int]] = [10, 20]
DEFAULT_HISTO_VOL_PERIODS: Final[list[int]] = [10, 20, 30]
DEFAULT_VOLUME_SMA_PERIODS: Final[list[int]] = [5, 20]
DEFAULT_RETURN_PERIODS: Final[list[int]] = [1, 5, 10, 20]


class FeatureConfig(BaseModel):
    """Configuration for technical indicators computation.

    All periods must be positive integers. Standard deviations must be positive floats.
    """

    # Trend indicators
    sma_periods: list[int] = Field(default_factory=lambda: DEFAULT_SMA_PERIODS.copy())
    ema_periods: list[int] = Field(default_factory=lambda: DEFAULT_EMA_PERIODS.copy())
    macd_fast: int = Field(default=12, gt=0)
    macd_slow: int = Field(default=26, gt=0)
    macd_signal: int = Field(default=9, gt=0)

    # Momentum indicators
    rsi_periods: list[int] = Field(default_factory=lambda: DEFAULT_RSI_PERIODS.copy())
    stoch_k_period: int = Field(default=14, gt=0)
    stoch_d_period: int = Field(default=3, gt=0)
    roc_periods: list[int] = Field(default_factory=lambda: DEFAULT_ROC_PERIODS.copy())
    mom_periods: list[int] = Field(default_factory=lambda: DEFAULT_MOM_PERIODS.copy())

    # Volatility indicators
    bb_period: int = Field(default=20, gt=0)
    bb_std: float = Field(default=2.0, gt=0)
    atr_period: int = Field(default=14, gt=0)
    histo_vol_periods: list[int] = Field(default_factory=lambda: DEFAULT_HISTO_VOL_PERIODS.copy())

    # Volume indicators
    volume_sma_periods: list[int] = Field(default_factory=lambda: DEFAULT_VOLUME_SMA_PERIODS.copy())

    # Price features
    return_periods: list[int] = Field(default_factory=lambda: DEFAULT_RETURN_PERIODS.copy())
    price_ratio_sma_periods: list[int] = Field(default_factory=lambda: [20, 50, 200])

    def model_post_init(self, __context: object) -> None:
        """Validate configuration after initialization."""
        if self.macd_fast >= self.macd_slow:
            raise ValueError("MACD fast period must be less than slow period")


class TechnicalFeatureEngine:
    """Engine for computing comprehensive technical indicators on OHLCV data.

    This class provides vectorized computation of technical indicators using Polars,
    ensuring high performance on large datasets.

    Attributes:
        config: Configuration object specifying indicator parameters.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """Initialize the feature engine.

        Args:
            config: Configuration for indicator parameters. If None, uses defaults.
        """
        self.config = config or FeatureConfig()
        logger.info("technical_feature_engine_initialized", config=self.config.model_dump())

    def compute_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all technical features.

        Args:
            df: Input DataFrame with OHLCV data. Must contain columns:
                - timestamp: datetime
                - open: float
                - high: float
                - low: float
                - close: float
                - volume: int or float

        Returns:
            DataFrame with all original columns plus computed indicator columns.

        Raises:
            ValueError: If required columns are missing or data is invalid.
        """
        self._validate_input(df)

        if df.height == 0:
            logger.warning("compute_all called with empty DataFrame")
            return df

        logger.info("computing_all_features", rows=df.height)

        result = df

        # Trend indicators
        result = self._compute_trend_indicators(result)

        # Momentum indicators
        result = self._compute_momentum_indicators(result)

        # Volatility indicators
        result = self._compute_volatility_indicators(result)

        # Volume indicators
        result = self._compute_volume_indicators(result)

        # Price features
        result = self._compute_price_features(result)

        logger.info(
            "all_features_computed",
            total_columns=len(result.columns),
            feature_columns=len(result.columns) - 6,  # Subtract OHLCV + timestamp
        )

        return result

    def compute_indicators(self, df: pl.DataFrame, indicators: list[str]) -> pl.DataFrame:
        """Compute specific indicators only.

        Args:
            df: Input DataFrame with OHLCV data.
            indicators: List of indicator names to compute. Valid values:
                - "sma": Simple Moving Averages
                - "ema": Exponential Moving Averages
                - "macd": MACD indicator
                - "rsi": Relative Strength Index
                - "stochastic": Stochastic Oscillator
                - "roc": Rate of Change
                - "momentum": Momentum
                - "bollinger": Bollinger Bands
                - "atr": Average True Range
                - "volatility": Historical and advanced volatility metrics
                - "obv": On-Balance Volume
                - "vwap": Volume Weighted Average Price
                - "volume_sma": Volume moving averages
                - "volume_roc": Volume Rate of Change
                - "returns": Price returns
                - "price_ratios": Price relative to SMAs
                - "price_features": High-Low range and gaps

        Returns:
            DataFrame with specified indicators computed.

        Raises:
            ValueError: If invalid indicator names are provided.
        """
        self._validate_input(df)

        valid_indicators = {
            "sma",
            "ema",
            "macd",
            "rsi",
            "stochastic",
            "roc",
            "momentum",
            "bollinger",
            "atr",
            "volatility",
            "obv",
            "vwap",
            "volume_sma",
            "volume_roc",
            "returns",
            "price_ratios",
            "price_features",
        }

        invalid = set(indicators) - valid_indicators
        if invalid:
            raise ValueError(f"Invalid indicator names: {invalid}")

        result = df

        # Compute requested indicators
        if "sma" in indicators:
            result = self._compute_sma(result)
        if "ema" in indicators:
            result = self._compute_ema(result)
        if "macd" in indicators:
            result = self._compute_macd(result)
        if "rsi" in indicators:
            result = self._compute_rsi(result)
        if "stochastic" in indicators:
            result = self._compute_stochastic(result)
        if "roc" in indicators:
            result = self._compute_roc(result)
        if "momentum" in indicators:
            result = self._compute_momentum(result)
        if "bollinger" in indicators:
            result = self._compute_bollinger_bands(result)
        if "atr" in indicators:
            result = self._compute_atr(result)
        if "volatility" in indicators:
            result = self._compute_volatility_metrics(result)
        if "obv" in indicators:
            result = self._compute_obv(result)
        if "vwap" in indicators:
            result = self._compute_vwap(result)
        if "volume_sma" in indicators:
            result = self._compute_volume_sma(result)
        if "volume_roc" in indicators:
            result = self._compute_volume_roc(result)
        if "returns" in indicators:
            result = self._compute_returns(result)
        if "price_ratios" in indicators:
            result = self._compute_price_ratios(result)
        if "price_features" in indicators:
            result = self._compute_price_features_misc(result)

        return result

    def get_feature_names(self) -> list[str]:
        """Return list of all feature column names that will be generated.

        Returns:
            List of feature column names.
        """
        features: list[str] = []

        # Trend indicators
        for period in self.config.sma_periods:
            features.append(f"sma_{period}")
        for period in self.config.ema_periods:
            features.append(f"ema_{period}")
        features.extend(["macd_line", "macd_signal", "macd_histogram"])

        # Momentum indicators
        for period in self.config.rsi_periods:
            features.append(f"rsi_{period}")
        features.extend(["stoch_k", "stoch_d"])
        for period in self.config.roc_periods:
            features.append(f"roc_{period}")
        for period in self.config.mom_periods:
            features.append(f"mom_{period}")

        # Volatility indicators
        features.extend(
            ["bb_upper", "bb_middle", "bb_lower", "bb_bandwidth", "bb_percent_b", f"atr_{self.config.atr_period}"]
        )
        for period in self.config.histo_vol_periods:
            features.append(f"histo_vol_{period}")
        features.extend(["parkinson_vol", "garman_klass_vol"])

        # Volume indicators
        features.extend(["obv", "vwap"])
        for period in self.config.volume_sma_periods:
            features.append(f"volume_sma_{period}")
        features.append("volume_roc")

        # Price features
        for period in self.config.return_periods:
            features.append(f"return_{period}d")
        features.append("log_return_1d")
        for period in self.config.price_ratio_sma_periods:
            features.append(f"price_to_sma_{period}")
        features.extend(["high_low_range", "gap"])

        return features

    def _validate_input(self, df: pl.DataFrame) -> None:
        """Validate that input DataFrame has required columns."""
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _compute_trend_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all trend indicators."""
        result = df
        result = self._compute_sma(result)
        result = self._compute_ema(result)
        result = self._compute_macd(result)
        return result

    def _compute_momentum_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all momentum indicators."""
        result = df
        result = self._compute_rsi(result)
        result = self._compute_stochastic(result)
        result = self._compute_roc(result)
        result = self._compute_momentum(result)
        return result

    def _compute_volatility_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all volatility indicators."""
        result = df
        result = self._compute_bollinger_bands(result)
        result = self._compute_atr(result)
        result = self._compute_volatility_metrics(result)
        return result

    def _compute_volume_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all volume indicators."""
        result = df
        result = self._compute_obv(result)
        result = self._compute_vwap(result)
        result = self._compute_volume_sma(result)
        result = self._compute_volume_roc(result)
        return result

    def _compute_price_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all price features."""
        result = df
        result = self._compute_returns(result)
        result = self._compute_price_ratios(result)
        result = self._compute_price_features_misc(result)
        return result

    def _compute_sma(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Simple Moving Averages."""
        result = df
        for period in self.config.sma_periods:
            result = result.with_columns(
                pl.col("close").rolling_mean(window_size=period).alias(f"sma_{period}")
            )
        return result

    def _compute_ema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Exponential Moving Averages."""
        result = df
        for period in self.config.ema_periods:
            result = result.with_columns(
                pl.col("close").ewm_mean(span=period, adjust=False).alias(f"ema_{period}")
            )
        return result

    def _compute_macd(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute MACD indicator."""
        result = df.with_columns(
            pl.col("close")
            .ewm_mean(span=self.config.macd_fast, adjust=False)
            .alias("ema_fast"),
            pl.col("close")
            .ewm_mean(span=self.config.macd_slow, adjust=False)
            .alias("ema_slow"),
        )

        result = result.with_columns(
            (pl.col("ema_fast") - pl.col("ema_slow")).alias("macd_line")
        )

        result = result.with_columns(
            pl.col("macd_line")
            .ewm_mean(span=self.config.macd_signal, adjust=False)
            .alias("macd_signal")
        )

        result = result.with_columns(
            (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram")
        )

        return result.drop(["ema_fast", "ema_slow"])

    def _compute_rsi(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Relative Strength Index for multiple periods."""
        result = df

        for period in self.config.rsi_periods:
            # Calculate price changes
            temp = result.with_columns(
                (pl.col("close") - pl.col("close").shift(1)).alias("delta")
            )

            # Separate gains and losses
            temp = temp.with_columns(
                pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain"),
                pl.when(pl.col("delta") < 0)
                .then(-pl.col("delta"))
                .otherwise(0.0)
                .alias("loss"),
            )

            # Calculate average gains and losses using EMA
            temp = temp.with_columns(
                pl.col("gain").ewm_mean(span=period, adjust=False).alias("avg_gain"),
                pl.col("loss").ewm_mean(span=period, adjust=False).alias("avg_loss"),
            )

            # Calculate RSI
            result = temp.with_columns(
                pl.when(pl.col("avg_loss") == 0)
                .then(100.0)
                .otherwise(
                    100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / pl.col("avg_loss"))))
                )
                .alias(f"rsi_{period}")
            ).drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])

        return result

    def _compute_stochastic(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Stochastic Oscillator."""
        # Calculate %K
        result = df.with_columns(
            pl.col("low")
            .rolling_min(window_size=self.config.stoch_k_period)
            .alias("lowest_low"),
            pl.col("high")
            .rolling_max(window_size=self.config.stoch_k_period)
            .alias("highest_high"),
        )

        result = result.with_columns(
            (
                100.0
                * (pl.col("close") - pl.col("lowest_low"))
                / (pl.col("highest_high") - pl.col("lowest_low"))
            ).alias("stoch_k")
        )

        # Calculate %D (SMA of %K)
        result = result.with_columns(
            pl.col("stoch_k")
            .rolling_mean(window_size=self.config.stoch_d_period)
            .alias("stoch_d")
        )

        return result.drop(["lowest_low", "highest_high"])

    def _compute_roc(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Rate of Change."""
        result = df
        for period in self.config.roc_periods:
            result = result.with_columns(
                (
                    100.0
                    * (pl.col("close") - pl.col("close").shift(period))
                    / pl.col("close").shift(period)
                ).alias(f"roc_{period}")
            )
        return result

    def _compute_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Momentum."""
        result = df
        for period in self.config.mom_periods:
            result = result.with_columns(
                (pl.col("close") - pl.col("close").shift(period)).alias(f"mom_{period}")
            )
        return result

    def _compute_bollinger_bands(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Bollinger Bands."""
        result = df.with_columns(
            pl.col("close")
            .rolling_mean(window_size=self.config.bb_period)
            .alias("bb_middle"),
            pl.col("close")
            .rolling_std(window_size=self.config.bb_period)
            .alias("rolling_std"),
        )

        result = result.with_columns(
            (pl.col("bb_middle") + (pl.col("rolling_std") * self.config.bb_std)).alias(
                "bb_upper"
            ),
            (pl.col("bb_middle") - (pl.col("rolling_std") * self.config.bb_std)).alias(
                "bb_lower"
            ),
        )

        # Bandwidth and %B
        result = result.with_columns(
            (
                (pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")
            ).alias("bb_bandwidth"),
            (
                (pl.col("close") - pl.col("bb_lower"))
                / (pl.col("bb_upper") - pl.col("bb_lower"))
            ).alias("bb_percent_b"),
        )

        return result.drop(["rolling_std"])

    def _compute_atr(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Average True Range."""
        result = df.with_columns(
            (pl.col("high") - pl.col("low")).alias("hl"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("hc"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("lc"),
        )

        result = result.with_columns(
            pl.max_horizontal("hl", "hc", "lc").alias("true_range")
        )

        result = result.with_columns(
            pl.col("true_range")
            .ewm_mean(span=self.config.atr_period, adjust=False)
            .alias(f"atr_{self.config.atr_period}")
        )

        return result.drop(["hl", "hc", "lc", "true_range"])

    def _compute_volatility_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute historical and advanced volatility metrics."""
        result = df

        # Historical volatility (annualized standard deviation of returns)
        for period in self.config.histo_vol_periods:
            result = result.with_columns(
                (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return_temp")
            )
            result = result.with_columns(
                (
                    pl.col("log_return_temp").rolling_std(window_size=period) * (252**0.5)
                ).alias(f"histo_vol_{period}")
            )
            result = result.drop("log_return_temp")

        # Parkinson volatility (uses high-low range)
        result = result.with_columns(
            (
                (1.0 / (4.0 * pl.lit(0.6931471805599453)))  # 1/(4*ln(2))
                * (pl.col("high") / pl.col("low")).log().pow(2)
            ).alias("parkinson_vol")
        )

        # Garman-Klass volatility
        result = result.with_columns(
            (
                0.5 * (pl.col("high") / pl.col("low")).log().pow(2)
                - (2.0 * pl.lit(0.6931471805599453) - 1.0)  # 2*ln(2) - 1
                * (pl.col("close") / pl.col("open")).log().pow(2)
            ).alias("garman_klass_vol")
        )

        return result

    def _compute_obv(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute On-Balance Volume."""
        result = df.with_columns(
            pl.when(pl.col("close") > pl.col("close").shift(1))
            .then(pl.col("volume"))
            .when(pl.col("close") < pl.col("close").shift(1))
            .then(-pl.col("volume"))
            .otherwise(0)
            .alias("obv_change")
        )

        result = result.with_columns(pl.col("obv_change").cum_sum().alias("obv"))

        return result.drop("obv_change")

    def _compute_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Volume Weighted Average Price."""
        result = df.with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("typical_price")
        )

        result = result.with_columns(
            (pl.col("typical_price") * pl.col("volume")).alias("tp_volume")
        )

        # For intraday VWAP, typically reset daily. Here we compute cumulative.
        # For proper daily VWAP, group by date.
        result = result.with_columns(
            (pl.col("tp_volume").cum_sum() / pl.col("volume").cum_sum()).alias("vwap")
        )

        return result.drop(["typical_price", "tp_volume"])

    def _compute_volume_sma(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Volume Simple Moving Averages."""
        result = df
        for period in self.config.volume_sma_periods:
            result = result.with_columns(
                pl.col("volume")
                .rolling_mean(window_size=period)
                .alias(f"volume_sma_{period}")
            )
        return result

    def _compute_volume_roc(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute Volume Rate of Change."""
        return df.with_columns(
            (
                100.0
                * (pl.col("volume") - pl.col("volume").shift(1))
                / pl.col("volume").shift(1)
            ).alias("volume_roc")
        )

    def _compute_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute price returns."""
        result = df

        # Simple returns for various periods
        for period in self.config.return_periods:
            result = result.with_columns(
                (
                    100.0
                    * (pl.col("close") - pl.col("close").shift(period))
                    / pl.col("close").shift(period)
                ).alias(f"return_{period}d")
            )

        # Log return (1-day)
        result = result.with_columns(
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return_1d")
        )

        return result

    def _compute_price_ratios(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute price relative to SMAs."""
        result = df

        for period in self.config.price_ratio_sma_periods:
            # Compute SMA if not already present
            sma_col = f"sma_{period}"
            if sma_col not in result.columns:
                result = result.with_columns(
                    pl.col("close").rolling_mean(window_size=period).alias(sma_col)
                )

            result = result.with_columns(
                (pl.col("close") / pl.col(sma_col)).alias(f"price_to_sma_{period}")
            )

        return result

    def _compute_price_features_misc(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute miscellaneous price features."""
        result = df.with_columns(
            (pl.col("high") - pl.col("low")).alias("high_low_range"),
            (pl.col("open") - pl.col("close").shift(1)).alias("gap"),
        )

        return result


# Legacy function for backward compatibility
def compute_technical_indicators(
    df: pl.DataFrame,
    config: FeatureConfig | None = None,
) -> pl.DataFrame:
    """Compute all technical indicators for the given DataFrame.

    This is a convenience function that wraps TechnicalFeatureEngine.

    Args:
        df: Input DataFrame with OHLCV data.
        config: Configuration for indicator parameters. If None, uses defaults.

    Returns:
        DataFrame with all original columns plus computed indicator columns.

    Raises:
        ValueError: If required columns are missing or data is invalid.
    """
    engine = TechnicalFeatureEngine(config)
    return engine.compute_all(df)


# For backward compatibility
IndicatorConfig = FeatureConfig


def compute_sma(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """Compute Simple Moving Average.

    Args:
        df: DataFrame with 'close' column.
        period: SMA period.

    Returns:
        DataFrame with SMA column added.
    """
    return df.with_columns(
        pl.col("close").rolling_mean(window_size=period).alias(f"sma_{period}")
    )


def compute_ema(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """Compute Exponential Moving Average.

    Args:
        df: DataFrame with 'close' column.
        period: EMA period.

    Returns:
        DataFrame with EMA column added.
    """
    return df.with_columns(
        pl.col("close").ewm_mean(span=period, adjust=False).alias(f"ema_{period}")
    )


def compute_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Compute Relative Strength Index.

    Args:
        df: DataFrame with 'close' column.
        period: RSI period.

    Returns:
        DataFrame with RSI column added.
    """
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0).rolling_mean(window_size=period)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(window_size=period)

    return df.with_columns(
        (100 - (100 / (1 + gain / loss))).alias(f"rsi_{period}")
    )


def compute_macd(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    """Compute MACD indicator.

    Args:
        df: DataFrame with 'close' column.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.

    Returns:
        DataFrame with MACD columns added.
    """
    result = df.with_columns(
        pl.col("close").ewm_mean(span=fast, adjust=False).alias("_ema_fast"),
        pl.col("close").ewm_mean(span=slow, adjust=False).alias("_ema_slow"),
    )
    result = result.with_columns(
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd_line")
    )
    result = result.with_columns(
        pl.col("macd_line").ewm_mean(span=signal, adjust=False).alias("macd_signal")
    )
    result = result.with_columns(
        (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram")
    )
    return result.drop(["_ema_fast", "_ema_slow"])


def compute_bollinger_bands(
    df: pl.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pl.DataFrame:
    """Compute Bollinger Bands.

    Args:
        df: DataFrame with 'close' column.
        period: SMA period for middle band.
        std_dev: Number of standard deviations for bands.

    Returns:
        DataFrame with Bollinger Band columns added.
    """
    result = df.with_columns(
        pl.col("close").rolling_mean(window_size=period).alias("bb_middle"),
        pl.col("close").rolling_std(window_size=period).alias("_bb_std"),
    )
    result = result.with_columns(
        (pl.col("bb_middle") + std_dev * pl.col("_bb_std")).alias("bb_upper"),
        (pl.col("bb_middle") - std_dev * pl.col("_bb_std")).alias("bb_lower"),
    )
    return result.drop("_bb_std")


def compute_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Compute Average True Range.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        period: ATR period.

    Returns:
        DataFrame with ATR column added.
    """
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return df.with_columns(
        tr.rolling_mean(window_size=period).alias(f"atr_{period}")
    )


def compute_volume_sma(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """Compute Volume Simple Moving Average.

    Args:
        df: DataFrame with 'volume' column.
        period: SMA period.

    Returns:
        DataFrame with volume SMA column added.
    """
    return df.with_columns(
        pl.col("volume").rolling_mean(window_size=period).alias(f"volume_sma_{period}")
    )


__all__ = [
    "TechnicalFeatureEngine",
    "FeatureConfig",
    "IndicatorConfig",
    "compute_technical_indicators",
    "compute_sma",
    "compute_ema",
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_volume_sma",
]
