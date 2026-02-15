"""Tests for technical indicators computation."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from pydantic import ValidationError

from signalforge.ml.features.technical import (
    FeatureConfig,
    IndicatorConfig,
    TechnicalFeatureEngine,
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_sma,
    compute_technical_indicators,
    compute_volume_sma,
)


@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    n_rows = 250  # Enough for all indicators including SMA 200
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_rows)]

    # Create realistic price data with some volatility
    base_price = 100.0
    prices = []
    for i in range(n_rows):
        # Add some trend and noise
        trend = i * 0.1
        noise = (i % 7) * 0.5 - 1.5
        prices.append(base_price + trend + noise)

    return pl.DataFrame(
        {
            "symbol": ["AAPL"] * n_rows,
            "timestamp": dates,
            "open": [p - 0.5 for p in prices],
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "volume": [1000000 + i * 10000 for i in range(n_rows)],
        }
    )


@pytest.fixture
def minimal_ohlcv_data() -> pl.DataFrame:
    """Create minimal OHLCV data for edge case testing."""
    return pl.DataFrame(
        {
            "symbol": ["AAPL"] * 50,
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)],
            "open": [100.0] * 50,
            "high": [105.0] * 50,
            "low": [95.0] * 50,
            "close": [100.0] * 50,
            "volume": [1000000] * 50,
        }
    )


@pytest.fixture
def empty_ohlcv_data() -> pl.DataFrame:
    """Create empty OHLCV DataFrame with correct schema."""
    return pl.DataFrame(
        {
            "symbol": [],
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        },
        schema={
            "symbol": pl.Utf8,
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        },
    )


class TestIndicatorConfig:
    """Tests for IndicatorConfig (FeatureConfig) class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FeatureConfig()
        assert config.sma_periods == [5, 10, 20, 50, 200]
        assert config.ema_periods == [5, 10, 20, 50]
        assert config.rsi_periods == [14, 21]
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9
        assert config.bb_period == 20
        assert config.bb_std == 2.0
        assert config.atr_period == 14
        assert config.volume_sma_periods == [5, 20]

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = FeatureConfig(
            sma_periods=[5, 10],
            ema_periods=[8, 21],
            rsi_periods=[10],
            macd_fast=10,
            macd_slow=20,
            macd_signal=5,
        )
        assert config.sma_periods == [5, 10]
        assert config.ema_periods == [8, 21]
        assert config.rsi_periods == [10]
        assert config.macd_fast == 10

    def test_indicator_config_alias(self) -> None:
        """Test that IndicatorConfig is an alias for FeatureConfig."""
        assert IndicatorConfig is FeatureConfig

    def test_invalid_macd_fast_raises_error(self) -> None:
        """Test that invalid MACD fast raises ValidationError."""
        with pytest.raises(ValidationError):
            FeatureConfig(macd_fast=-12)

    def test_invalid_macd_slow_raises_error(self) -> None:
        """Test that invalid MACD slow raises ValidationError."""
        with pytest.raises(ValidationError):
            FeatureConfig(macd_slow=-26)

    def test_invalid_macd_periods_raise_error(self) -> None:
        """Test that fast >= slow raises ValueError."""
        with pytest.raises(ValueError, match="MACD fast period must be less than slow period"):
            FeatureConfig(macd_fast=26, macd_slow=12)

    def test_invalid_bb_period_raises_error(self) -> None:
        """Test that invalid Bollinger Bands period raises ValidationError."""
        with pytest.raises(ValidationError):
            FeatureConfig(bb_period=0)

    def test_invalid_bb_std_raises_error(self) -> None:
        """Test that invalid Bollinger Bands std raises ValidationError."""
        with pytest.raises(ValidationError):
            FeatureConfig(bb_std=-1.0)

    def test_invalid_atr_period_raises_error(self) -> None:
        """Test that invalid ATR period raises ValidationError."""
        with pytest.raises(ValidationError):
            FeatureConfig(atr_period=-14)


class TestComputeSMA:
    """Tests for compute_sma function."""

    def test_compute_sma_basic(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test basic SMA computation."""
        result = compute_sma(sample_ohlcv_data, period=10)
        assert "sma_10" in result.columns
        assert result.height == sample_ohlcv_data.height

    def test_sma_values_correct(self) -> None:
        """Test that SMA values are computed correctly."""
        df = pl.DataFrame(
            {
                "symbol": ["TEST"] * 10,
                "timestamp": [datetime(2024, 1, i) for i in range(1, 11)],
                "close": [float(i) for i in range(1, 11)],
            }
        )
        result = compute_sma(df, period=3)

        # First two values should be null (not enough data)
        assert result["sma_3"][0] is None
        assert result["sma_3"][1] is None

        # Third value should be (1+2+3)/3 = 2.0
        assert result["sma_3"][2] == 2.0

        # Fourth value should be (2+3+4)/3 = 3.0
        assert result["sma_3"][3] == 3.0

    def test_sma_handles_nulls(self) -> None:
        """Test that SMA handles null values correctly."""
        df = pl.DataFrame(
            {
                "symbol": ["TEST"] * 5,
                "close": [1.0, 2.0, None, 4.0, 5.0],
            }
        )
        result = compute_sma(df, period=3)
        assert "sma_3" in result.columns


class TestComputeEMA:
    """Tests for compute_ema function."""

    def test_compute_ema_basic(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test basic EMA computation."""
        result = compute_ema(sample_ohlcv_data, period=12)
        assert "ema_12" in result.columns
        assert result.height == sample_ohlcv_data.height

    def test_ema_reacts_faster_than_sma(self) -> None:
        """Test that EMA reacts faster than SMA to price changes."""
        # Create data with a sharp price jump in the middle
        prices_before = [100.0] * 30
        prices_after = [150.0] * 20
        all_prices = prices_before + prices_after

        df = pl.DataFrame(
            {
                "symbol": ["TEST"] * len(all_prices),
                "timestamp": [
                    datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(all_prices))
                ],
                "close": all_prices,
            }
        )

        result = compute_sma(df, period=10)
        result = compute_ema(result, period=10)

        # After the price jump, check the rate of change
        # EMA should adjust faster than SMA
        # Look at a point shortly after the jump (position 35)
        row_after_jump = result[35]
        ema_value = row_after_jump["ema_10"][0]
        sma_value = row_after_jump["sma_10"][0]

        # Both should be between 100 and 150, but EMA should be closer to 150
        assert ema_value is not None
        assert sma_value is not None
        assert 100.0 < ema_value < 150.0
        assert 100.0 < sma_value < 150.0
        # EMA reacts faster, so it should be higher after a positive price jump
        assert ema_value > sma_value


class TestComputeRSI:
    """Tests for compute_rsi function."""

    def test_compute_rsi_basic(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test basic RSI computation."""
        result = compute_rsi(sample_ohlcv_data, period=14)
        assert "rsi_14" in result.columns
        assert result.height == sample_ohlcv_data.height

    def test_rsi_range_is_valid(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test that RSI values are between 0 and 100."""
        result = compute_rsi(sample_ohlcv_data, period=14)

        # Filter out null values
        rsi_values = result.select(pl.col("rsi_14")).drop_nulls()

        assert rsi_values["rsi_14"].min() >= 0.0
        assert rsi_values["rsi_14"].max() <= 100.0

    def test_rsi_trending_up(self) -> None:
        """Test RSI for consistently rising prices."""
        df = pl.DataFrame(
            {
                "symbol": ["TEST"] * 50,
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)],
                "close": [float(100 + i) for i in range(50)],
            }
        )
        result = compute_rsi(df, period=14)

        # For rising prices, RSI should be high (> 50)
        last_rsi = result.tail(1)["rsi_14"][0]
        assert last_rsi is not None
        assert last_rsi > 50.0

    def test_rsi_trending_down(self) -> None:
        """Test RSI for consistently falling prices."""
        df = pl.DataFrame(
            {
                "symbol": ["TEST"] * 50,
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)],
                "close": [float(100 - i) for i in range(50)],
            }
        )
        result = compute_rsi(df, period=14)

        # For falling prices, RSI should be low (< 50)
        last_rsi = result.tail(1)["rsi_14"][0]
        assert last_rsi is not None
        assert last_rsi < 50.0


class TestComputeMACD:
    """Tests for compute_macd function."""

    def test_compute_macd_basic(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test basic MACD computation."""
        result = compute_macd(sample_ohlcv_data)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        assert result.height == sample_ohlcv_data.height

    def test_macd_histogram_is_difference(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test that MACD histogram is the difference between MACD and signal."""
        result = compute_macd(sample_ohlcv_data)

        # Filter out nulls and check the relationship
        non_null_result = result.filter(
            pl.col("macd_line").is_not_null()
            & pl.col("macd_signal").is_not_null()
            & pl.col("macd_histogram").is_not_null()
        )

        for row in non_null_result.iter_rows(named=True):
            expected_histogram = row["macd_line"] - row["macd_signal"]
            actual_histogram = row["macd_histogram"]
            assert abs(expected_histogram - actual_histogram) < 1e-10

    def test_macd_custom_periods(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test MACD with custom periods."""
        result = compute_macd(sample_ohlcv_data, fast=8, slow=21, signal=5)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns


class TestComputeBollingerBands:
    """Tests for compute_bollinger_bands function."""

    def test_compute_bollinger_bands_basic(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test basic Bollinger Bands computation."""
        result = compute_bollinger_bands(sample_ohlcv_data)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert result.height == sample_ohlcv_data.height

    def test_bollinger_bands_relationship(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test that upper > middle > lower."""
        result = compute_bollinger_bands(sample_ohlcv_data)

        # Filter out nulls
        non_null_result = result.filter(
            pl.col("bb_upper").is_not_null()
            & pl.col("bb_middle").is_not_null()
            & pl.col("bb_lower").is_not_null()
        )

        for row in non_null_result.iter_rows(named=True):
            assert row["bb_upper"] >= row["bb_middle"]
            assert row["bb_middle"] >= row["bb_lower"]

    def test_bollinger_bands_custom_params(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test Bollinger Bands with custom parameters."""
        result = compute_bollinger_bands(sample_ohlcv_data, period=10, std_dev=1.5)
        assert "bb_upper" in result.columns


class TestComputeATR:
    """Tests for compute_atr function."""

    def test_compute_atr_basic(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test basic ATR computation."""
        result = compute_atr(sample_ohlcv_data, period=14)
        assert "atr_14" in result.columns
        assert result.height == sample_ohlcv_data.height

    def test_atr_is_positive(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test that ATR values are positive."""
        result = compute_atr(sample_ohlcv_data, period=14)

        # Filter out null values
        atr_values = result.select(pl.col("atr_14")).drop_nulls()

        assert (atr_values["atr_14"] >= 0.0).all()

    def test_atr_custom_period(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test ATR with custom period."""
        result = compute_atr(sample_ohlcv_data, period=7)
        assert "atr_7" in result.columns


class TestComputeVolumeSMA:
    """Tests for compute_volume_sma function."""

    def test_compute_volume_sma_basic(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test basic volume SMA computation."""
        result = compute_volume_sma(sample_ohlcv_data, period=20)
        assert "volume_sma_20" in result.columns
        assert result.height == sample_ohlcv_data.height

    def test_volume_sma_custom_period(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test volume SMA with custom period."""
        result = compute_volume_sma(sample_ohlcv_data, period=10)
        assert "volume_sma_10" in result.columns


class TestComputeTechnicalIndicators:
    """Tests for compute_technical_indicators function."""

    def test_compute_all_indicators(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test computation of all indicators with default config."""
        result = compute_technical_indicators(sample_ohlcv_data)

        # Check that key expected columns exist (using new naming)
        expected_columns = [
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_50",
            "sma_200",
            "ema_5",
            "ema_10",
            "ema_20",
            "ema_50",
            "rsi_14",
            "rsi_21",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr_14",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_compute_indicators_custom_config(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test computation with custom configuration."""
        config = FeatureConfig(
            sma_periods=[5, 10],
            ema_periods=[8, 21],
            rsi_periods=[10],
        )
        result = compute_technical_indicators(sample_ohlcv_data, config)

        # Custom SMA periods should be present
        assert "sma_5" in result.columns
        assert "sma_10" in result.columns
        assert "ema_8" in result.columns
        assert "ema_21" in result.columns
        assert "rsi_10" in result.columns

        # Default RSI periods should not be present (we specified only [10])
        assert "rsi_14" not in result.columns
        assert "rsi_21" not in result.columns

    def test_compute_indicators_preserves_original_data(
        self, sample_ohlcv_data: pl.DataFrame
    ) -> None:
        """Test that original columns are preserved."""
        result = compute_technical_indicators(sample_ohlcv_data)

        original_cols = [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        for col in original_cols:
            assert col in result.columns

        # Check that original data is unchanged
        assert result.height == sample_ohlcv_data.height

    def test_compute_indicators_empty_dataframe(self, empty_ohlcv_data: pl.DataFrame) -> None:
        """Test handling of empty DataFrame."""
        result = compute_technical_indicators(empty_ohlcv_data)
        assert result.height == 0
        assert "symbol" in result.columns

    def test_compute_indicators_missing_columns_raise_error(self) -> None:
        """Test that missing required columns raise ValueError."""
        df = pl.DataFrame({"symbol": ["TEST"], "close": [100.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_technical_indicators(df)

    def test_compute_indicators_minimal_data(self, minimal_ohlcv_data: pl.DataFrame) -> None:
        """Test with minimal data that may not have enough rows for all indicators."""
        # Use config with smaller periods
        config = FeatureConfig(
            sma_periods=[10, 20],
            ema_periods=[12, 26],
        )
        result = compute_technical_indicators(minimal_ohlcv_data, config)

        # Should complete without errors
        assert result.height == minimal_ohlcv_data.height
        assert "sma_10" in result.columns

    def test_compute_indicators_with_multiple_symbols(self) -> None:
        """Test computation with multiple symbols in DataFrame."""
        df = pl.DataFrame(
            {
                "symbol": ["AAPL"] * 50 + ["GOOGL"] * 50,
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)] * 2,
                "open": [100.0] * 100,
                "high": [105.0] * 100,
                "low": [95.0] * 100,
                "close": [100.0 + i * 0.1 for i in range(100)],
                "volume": [1000000] * 100,
            }
        )

        config = FeatureConfig(sma_periods=[10], ema_periods=[12])
        result = compute_technical_indicators(df, config)

        # Should handle multiple symbols
        assert result.height == 100
        assert result["symbol"].n_unique() == 2

    def test_compute_indicators_adds_columns(
        self, sample_ohlcv_data: pl.DataFrame
    ) -> None:
        """Test that indicators are added as new columns."""
        original_cols = sample_ohlcv_data.columns
        result = compute_technical_indicators(sample_ohlcv_data)

        # Should have more columns than original
        assert len(result.columns) > len(original_cols)

    def test_compute_indicators_handles_flat_prices(self, minimal_ohlcv_data: pl.DataFrame) -> None:
        """Test that flat prices are handled correctly."""
        config = FeatureConfig(sma_periods=[10], ema_periods=[12], rsi_periods=[14])
        result = compute_technical_indicators(minimal_ohlcv_data, config)

        # For flat prices, SMA should equal close price
        non_null = result.filter(pl.col("sma_10").is_not_null())
        if not non_null.is_empty():
            # SMA should be close to close price for flat data
            assert abs(non_null["sma_10"][0] - 100.0) < 0.01


class TestTechnicalFeatureEngine:
    """Tests for TechnicalFeatureEngine class."""

    def test_engine_initialization(self) -> None:
        """Test engine initialization with default config."""
        engine = TechnicalFeatureEngine()
        assert engine.config is not None
        assert isinstance(engine.config, FeatureConfig)

    def test_engine_custom_config(self) -> None:
        """Test engine initialization with custom config."""
        config = FeatureConfig(sma_periods=[5, 10])
        engine = TechnicalFeatureEngine(config)
        assert engine.config.sma_periods == [5, 10]

    def test_engine_compute_all(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test computing all indicators via engine."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_all(sample_ohlcv_data)
        assert result.height == sample_ohlcv_data.height
        assert "sma_5" in result.columns

    def test_engine_compute_selective(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test computing selective indicators."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["sma", "ema"])

        # Should have SMA and EMA columns
        assert "sma_5" in result.columns
        assert "ema_5" in result.columns

    def test_engine_get_feature_names(self) -> None:
        """Test getting feature names."""
        engine = TechnicalFeatureEngine()
        names = engine.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert any("sma" in name for name in names)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_row_dataframe(self) -> None:
        """Test handling of single row DataFrame."""
        df = pl.DataFrame(
            {
                "symbol": ["TEST"],
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000000],
            }
        )

        # Should not raise, but indicators will be null
        result = compute_technical_indicators(df)
        assert result.height == 1

    def test_data_with_nulls(self) -> None:
        """Test handling of DataFrame with null values."""
        df = pl.DataFrame(
            {
                "symbol": ["TEST"] * 50,
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)],
                "open": [100.0 if i % 5 != 0 else None for i in range(50)],
                "high": [105.0] * 50,
                "low": [95.0] * 50,
                "close": [102.0] * 50,
                "volume": [1000000] * 50,
            }
        )

        config = FeatureConfig(sma_periods=[10], ema_periods=[12])
        result = compute_technical_indicators(df, config)
        assert result.height == 50

    def test_extreme_volatility(self) -> None:
        """Test handling of extreme price volatility."""
        prices = [100.0, 200.0, 50.0, 300.0, 10.0, 500.0, 25.0, 400.0, 75.0, 350.0]
        df = pl.DataFrame(
            {
                "symbol": ["TEST"] * 10,
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)],
                "open": [p - 10 for p in prices],
                "high": [p + 50 for p in prices],
                "low": [p - 50 for p in prices],
                "close": prices,
                "volume": [1000000] * 10,
            }
        )

        # Should handle without errors
        result = compute_sma(df, period=3)
        assert result.height == 10
        assert "sma_3" in result.columns
