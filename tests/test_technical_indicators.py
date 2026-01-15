"""Tests for technical indicators computation."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalforge.ml.features.technical import (
    IndicatorConfig,
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
    """Tests for IndicatorConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = IndicatorConfig()
        assert config.sma_periods == (10, 20, 50, 200)
        assert config.ema_periods == (12, 26)
        assert config.rsi_period == 14
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9
        assert config.bb_period == 20
        assert config.bb_std == 2.0
        assert config.atr_period == 14
        assert config.volume_sma_period == 20

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = IndicatorConfig(
            sma_periods=(5, 10),
            ema_periods=(8, 21),
            rsi_period=10,
            macd_fast=10,
            macd_slow=20,
            macd_signal=5,
        )
        assert config.sma_periods == (5, 10)
        assert config.ema_periods == (8, 21)
        assert config.rsi_period == 10
        assert config.macd_fast == 10

    def test_invalid_sma_period_raises_error(self) -> None:
        """Test that invalid SMA period raises ValueError."""
        with pytest.raises(ValueError, match="All SMA periods must be positive"):
            IndicatorConfig(sma_periods=(10, 0, 20))

    def test_invalid_ema_period_raises_error(self) -> None:
        """Test that invalid EMA period raises ValueError."""
        with pytest.raises(ValueError, match="All EMA periods must be positive"):
            IndicatorConfig(ema_periods=(-12, 26))

    def test_invalid_rsi_period_raises_error(self) -> None:
        """Test that invalid RSI period raises ValueError."""
        with pytest.raises(ValueError, match="RSI period must be positive"):
            IndicatorConfig(rsi_period=0)

    def test_invalid_macd_periods_raise_error(self) -> None:
        """Test that invalid MACD periods raise ValueError."""
        with pytest.raises(ValueError, match="All MACD periods must be positive"):
            IndicatorConfig(macd_fast=-12)

        with pytest.raises(ValueError, match="MACD fast period must be less than slow period"):
            IndicatorConfig(macd_fast=26, macd_slow=12)

    def test_invalid_bb_config_raises_error(self) -> None:
        """Test that invalid Bollinger Bands config raises ValueError."""
        with pytest.raises(ValueError, match="Bollinger Bands period must be positive"):
            IndicatorConfig(bb_period=0)

        with pytest.raises(ValueError, match="Bollinger Bands std must be positive"):
            IndicatorConfig(bb_std=-1.0)

    def test_invalid_atr_period_raises_error(self) -> None:
        """Test that invalid ATR period raises ValueError."""
        with pytest.raises(ValueError, match="ATR period must be positive"):
            IndicatorConfig(atr_period=-14)

    def test_invalid_volume_sma_period_raises_error(self) -> None:
        """Test that invalid volume SMA period raises ValueError."""
        with pytest.raises(ValueError, match="Volume SMA period must be positive"):
            IndicatorConfig(volume_sma_period=0)


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

    def test_sma_custom_column(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test SMA on custom column."""
        result = compute_sma(sample_ohlcv_data, period=5, column="high")
        assert "high_sma_5" in result.columns

    def test_sma_missing_column_raises_error(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test that missing column raises ValueError."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            compute_sma(sample_ohlcv_data, period=10, column="nonexistent")

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

    def test_ema_custom_column(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test EMA on custom column."""
        result = compute_ema(sample_ohlcv_data, period=26, column="low")
        assert "low_ema_26" in result.columns

    def test_ema_missing_column_raises_error(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test that missing column raises ValueError."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            compute_ema(sample_ohlcv_data, period=12, column="nonexistent")


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

    def test_rsi_missing_close_raises_error(self) -> None:
        """Test that missing close column raises ValueError."""
        df = pl.DataFrame({"symbol": ["TEST"], "open": [100.0]})
        with pytest.raises(ValueError, match="Column 'close' not found"):
            compute_rsi(df, period=14)


class TestComputeMACD:
    """Tests for compute_macd function."""

    def test_compute_macd_basic(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test basic MACD computation."""
        result = compute_macd(sample_ohlcv_data)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        assert result.height == sample_ohlcv_data.height

    def test_macd_histogram_is_difference(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test that MACD histogram is the difference between MACD and signal."""
        result = compute_macd(sample_ohlcv_data)

        # Filter out nulls and check the relationship
        non_null_result = result.filter(
            pl.col("macd").is_not_null()
            & pl.col("macd_signal").is_not_null()
            & pl.col("macd_histogram").is_not_null()
        )

        for row in non_null_result.iter_rows(named=True):
            expected_histogram = row["macd"] - row["macd_signal"]
            actual_histogram = row["macd_histogram"]
            assert abs(expected_histogram - actual_histogram) < 1e-10

    def test_macd_custom_periods(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test MACD with custom periods."""
        result = compute_macd(sample_ohlcv_data, fast_period=8, slow_period=21, signal_period=5)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns

    def test_macd_invalid_periods_raise_error(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test that invalid periods raise ValueError."""
        with pytest.raises(ValueError, match="Fast period must be less than slow period"):
            compute_macd(sample_ohlcv_data, fast_period=26, slow_period=12)

    def test_macd_missing_close_raises_error(self) -> None:
        """Test that missing close column raises ValueError."""
        df = pl.DataFrame({"symbol": ["TEST"], "open": [100.0]})
        with pytest.raises(ValueError, match="Column 'close' not found"):
            compute_macd(df)


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
        result = compute_bollinger_bands(sample_ohlcv_data, period=10, std_multiplier=1.5)
        assert "bb_upper" in result.columns

    def test_bollinger_bands_missing_close_raises_error(self) -> None:
        """Test that missing close column raises ValueError."""
        df = pl.DataFrame({"symbol": ["TEST"], "open": [100.0]})
        with pytest.raises(ValueError, match="Column 'close' not found"):
            compute_bollinger_bands(df)


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

    def test_atr_missing_columns_raise_error(self) -> None:
        """Test that missing required columns raise ValueError."""
        df = pl.DataFrame({"symbol": ["TEST"], "close": [100.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_atr(df)


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

    def test_volume_sma_missing_column_raises_error(self) -> None:
        """Test that missing volume column raises ValueError."""
        df = pl.DataFrame({"symbol": ["TEST"], "close": [100.0]})
        with pytest.raises(ValueError, match="Column 'volume' not found"):
            compute_volume_sma(df)


class TestComputeTechnicalIndicators:
    """Tests for compute_technical_indicators function."""

    def test_compute_all_indicators(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test computation of all indicators with default config."""
        result = compute_technical_indicators(sample_ohlcv_data)

        # Check that all expected columns exist
        expected_columns = [
            "sma_10",
            "sma_20",
            "sma_50",
            "sma_200",
            "ema_12",
            "ema_26",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr_14",
            "volume_sma_20",
        ]

        for col in expected_columns:
            assert col in result.columns

    def test_compute_indicators_custom_config(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test computation with custom configuration."""
        config = IndicatorConfig(
            sma_periods=(5, 10),
            ema_periods=(8, 21),
            rsi_period=10,
        )
        result = compute_technical_indicators(sample_ohlcv_data, config)

        assert "sma_5" in result.columns
        assert "sma_10" in result.columns
        assert "ema_8" in result.columns
        assert "ema_21" in result.columns
        assert "rsi_10" in result.columns

        # Default periods should not be present
        assert "sma_20" not in result.columns
        assert "sma_50" not in result.columns

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
        config = IndicatorConfig(
            sma_periods=(10, 20),
            ema_periods=(12, 26),
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

        config = IndicatorConfig(sma_periods=(10,), ema_periods=(12,))
        result = compute_technical_indicators(df, config)

        # Should handle multiple symbols
        assert result.height == 100
        assert result["symbol"].n_unique() == 2

    def test_compute_indicators_adds_correct_number_of_columns(
        self, sample_ohlcv_data: pl.DataFrame
    ) -> None:
        """Test that correct number of indicator columns are added."""
        original_cols = len(sample_ohlcv_data.columns)
        result = compute_technical_indicators(sample_ohlcv_data)

        # Default config should add:
        # 4 SMA + 2 EMA + 1 RSI + 3 MACD + 3 BB + 1 ATR + 1 Volume SMA = 15 columns
        expected_new_cols = 15
        assert len(result.columns) == original_cols + expected_new_cols

    def test_compute_indicators_handles_flat_prices(self) -> None:
        """Test indicators with completely flat prices."""
        df = pl.DataFrame(
            {
                "symbol": ["TEST"] * 100,
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)],
                "open": [100.0] * 100,
                "high": [100.0] * 100,
                "low": [100.0] * 100,
                "close": [100.0] * 100,
                "volume": [1000000] * 100,
            }
        )

        result = compute_technical_indicators(df)

        # All SMAs should equal the price
        assert result.filter(pl.col("sma_10").is_not_null())["sma_10"].unique().to_list() == [100.0]

        # RSI should be around 50 (neutral) for flat prices
        rsi_values = result.filter(pl.col("rsi_14").is_not_null())["rsi_14"]
        if len(rsi_values) > 0:
            # For flat prices, RSI should be close to 50
            assert all(45.0 <= val <= 55.0 or val == 100.0 for val in rsi_values)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_row_dataframe(self) -> None:
        """Test with single row DataFrame."""
        df = pl.DataFrame(
            {
                "symbol": ["TEST"],
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [100.0],
                "volume": [1000000],
            }
        )

        config = IndicatorConfig(sma_periods=(5,), ema_periods=(12,))
        result = compute_technical_indicators(df, config)

        # Should not crash, but most indicators will be null
        assert result.height == 1
        assert result["sma_5"][0] is None  # Not enough data

    def test_data_with_nulls(self) -> None:
        """Test handling of null values in price data."""
        df = pl.DataFrame(
            {
                "symbol": ["TEST"] * 30,
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)],
                "open": [100.0] * 30,
                "high": [105.0] * 30,
                "low": [95.0] * 30,
                "close": [100.0 if i != 15 else None for i in range(30)],
                "volume": [1000000] * 30,
            }
        )

        config = IndicatorConfig(sma_periods=(10,), ema_periods=(12,))
        result = compute_technical_indicators(df, config)

        # Should handle nulls gracefully
        assert result.height == 30

    def test_extreme_volatility(self) -> None:
        """Test with extremely volatile prices."""
        df = pl.DataFrame(
            {
                "symbol": ["TEST"] * 100,
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)],
                "open": [100.0 if i % 2 == 0 else 200.0 for i in range(100)],
                "high": [250.0] * 100,
                "low": [50.0] * 100,
                "close": [100.0 if i % 2 == 0 else 200.0 for i in range(100)],
                "volume": [1000000] * 100,
            }
        )

        result = compute_technical_indicators(df)

        # Should complete without errors
        assert result.height == 100
        # ATR should be high due to volatility
        atr_values = result.filter(pl.col("atr_14").is_not_null())["atr_14"]
        assert all(val > 0 for val in atr_values)
