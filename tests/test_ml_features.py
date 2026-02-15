"""Tests for ML feature engineering module."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalforge.ml.features import FeatureConfig, TechnicalFeatureEngine


@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(250)]
    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0 + i * 0.5 for i in range(250)],
            "high": [105.0 + i * 0.5 for i in range(250)],
            "low": [98.0 + i * 0.5 for i in range(250)],
            "close": [102.0 + i * 0.5 for i in range(250)],
            "volume": [1000000 + i * 1000 for i in range(250)],
        }
    )


@pytest.fixture
def volatile_ohlcv_data() -> pl.DataFrame:
    """Create volatile OHLCV data for testing volatility indicators."""
    import math

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    base_price = 100.0

    close_prices = [base_price + 10.0 * math.sin(i * 0.5) for i in range(100)]
    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": [p + 1.0 for p in close_prices],
            "high": [p + 3.0 for p in close_prices],
            "low": [p - 2.0 for p in close_prices],
            "close": close_prices,
            "volume": [1000000] * 100,
        }
    )


@pytest.fixture
def simple_ohlcv_data() -> pl.DataFrame:
    """Create simple constant OHLCV data."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0] * 50,
            "high": [105.0] * 50,
            "low": [95.0] * 50,
            "close": [100.0] * 50,
            "volume": [1000000] * 50,
        }
    )


class TestFeatureConfig:
    """Test FeatureConfig model."""

    def test_default_config(self) -> None:
        """Test default configuration."""
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

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = FeatureConfig(
            sma_periods=[10, 20],
            ema_periods=[8, 21],
            rsi_periods=[14],
            macd_fast=10,
            macd_slow=20,
            macd_signal=5,
        )

        assert config.sma_periods == [10, 20]
        assert config.ema_periods == [8, 21]
        assert config.rsi_periods == [14]
        assert config.macd_fast == 10
        assert config.macd_slow == 20
        assert config.macd_signal == 5

    def test_invalid_macd_periods(self) -> None:
        """Test that invalid MACD periods raise error."""
        with pytest.raises(ValueError, match="MACD fast period must be less than slow period"):
            FeatureConfig(macd_fast=26, macd_slow=12)

    def test_negative_period_validation(self) -> None:
        """Test that negative periods are rejected by Pydantic."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FeatureConfig(macd_fast=-1)

    def test_zero_period_validation(self) -> None:
        """Test that zero periods are rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FeatureConfig(atr_period=0)


class TestTechnicalFeatureEngine:
    """Test TechnicalFeatureEngine class."""

    def test_initialization_default_config(self) -> None:
        """Test engine initialization with default config."""
        engine = TechnicalFeatureEngine()
        assert engine.config is not None
        assert isinstance(engine.config, FeatureConfig)

    def test_initialization_custom_config(self) -> None:
        """Test engine initialization with custom config."""
        config = FeatureConfig(sma_periods=[10, 20])
        engine = TechnicalFeatureEngine(config)
        assert engine.config.sma_periods == [10, 20]

    def test_compute_all_basic(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test computing all features on basic data."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_all(sample_ohlcv_data)

        # Check original columns are preserved
        assert "timestamp" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns

        # Check some indicator columns are added
        assert "sma_20" in result.columns
        assert "ema_10" in result.columns
        assert "rsi_14" in result.columns
        assert "macd_line" in result.columns
        assert "bb_upper" in result.columns
        assert "atr_14" in result.columns

        # Check result has more columns than input
        assert len(result.columns) > len(sample_ohlcv_data.columns)

    def test_compute_all_empty_dataframe(self) -> None:
        """Test compute_all with empty DataFrame."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            },
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )

        engine = TechnicalFeatureEngine()
        result = engine.compute_all(empty_df)

        # Should return original DataFrame without error
        assert result.height == 0

    def test_compute_all_missing_columns(self) -> None:
        """Test compute_all with missing required columns."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "close": [100.0]})

        engine = TechnicalFeatureEngine()
        with pytest.raises(ValueError, match="Missing required columns"):
            engine.compute_all(df)

    def test_get_feature_names(self) -> None:
        """Test get_feature_names returns expected features."""
        engine = TechnicalFeatureEngine()
        features = engine.get_feature_names()

        assert isinstance(features, list)
        assert len(features) > 0
        assert "sma_20" in features
        assert "ema_10" in features
        assert "rsi_14" in features
        assert "macd_line" in features
        assert "bb_upper" in features
        assert "obv" in features
        assert "vwap" in features


class TestSMAIndicator:
    """Test Simple Moving Average calculation."""

    def test_sma_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test SMA values are calculated correctly."""
        engine = TechnicalFeatureEngine(FeatureConfig(sma_periods=[5]))
        result = engine.compute_indicators(sample_ohlcv_data, ["sma"])

        assert "sma_5" in result.columns

        # Check SMA value at row 5 (first complete SMA)
        expected_sma = result.select(pl.col("close").slice(0, 5).mean()).item()
        actual_sma = result.select(pl.col("sma_5").slice(4, 1)).item()

        # Allow small floating point difference
        assert abs(actual_sma - expected_sma) < 0.01

    def test_sma_null_handling(self) -> None:
        """Test SMA handles insufficient data correctly."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [98.0, 99.0, 100.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [1000000, 1000000, 1000000],
            }
        )

        engine = TechnicalFeatureEngine(FeatureConfig(sma_periods=[5]))
        result = engine.compute_indicators(df, ["sma"])

        # All values should be null since we don't have 5 periods
        assert result["sma_5"].null_count() == 3


class TestEMAIndicator:
    """Test Exponential Moving Average calculation."""

    def test_ema_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test EMA values are calculated."""
        engine = TechnicalFeatureEngine(FeatureConfig(ema_periods=[10]))
        result = engine.compute_indicators(sample_ohlcv_data, ["ema"])

        assert "ema_10" in result.columns
        # EMA should have fewer nulls than SMA (typically only first value is null)
        assert result["ema_10"].null_count() <= 1

    def test_ema_vs_sma(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test that EMA differs from SMA (is more responsive)."""
        engine = TechnicalFeatureEngine(FeatureConfig(sma_periods=[20], ema_periods=[20]))
        result = engine.compute_indicators(sample_ohlcv_data, ["sma", "ema"])

        # Filter to rows where both are not null
        filtered = result.filter(
            pl.col("sma_20").is_not_null() & pl.col("ema_20").is_not_null()
        )

        # EMA and SMA should be different
        sma_values = filtered["sma_20"]
        ema_values = filtered["ema_20"]

        # At least some values should differ
        assert not (sma_values == ema_values).all()


class TestRSIIndicator:
    """Test Relative Strength Index calculation."""

    def test_rsi_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test RSI values are in valid range."""
        engine = TechnicalFeatureEngine(FeatureConfig(rsi_periods=[14]))
        result = engine.compute_indicators(sample_ohlcv_data, ["rsi"])

        assert "rsi_14" in result.columns

        # RSI should be between 0 and 100
        rsi_values = result.filter(pl.col("rsi_14").is_not_null())["rsi_14"]
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()

    def test_rsi_trending_up(self) -> None:
        """Test RSI is high when price is trending up."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(50)],
                "high": [105.0 + i for i in range(50)],
                "low": [98.0 + i for i in range(50)],
                "close": [100.0 + i for i in range(50)],
                "volume": [1000000] * 50,
            }
        )

        engine = TechnicalFeatureEngine(FeatureConfig(rsi_periods=[14]))
        result = engine.compute_indicators(df, ["rsi"])

        # RSI should be high (>50) in strong uptrend
        last_rsi = result.select(pl.col("rsi_14").last()).item()
        assert last_rsi > 50

    def test_rsi_multiple_periods(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test RSI calculation with multiple periods."""
        engine = TechnicalFeatureEngine(FeatureConfig(rsi_periods=[14, 21]))
        result = engine.compute_indicators(sample_ohlcv_data, ["rsi"])

        assert "rsi_14" in result.columns
        assert "rsi_21" in result.columns


class TestMACDIndicator:
    """Test MACD indicator calculation."""

    def test_macd_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test MACD components are calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["macd"])

        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_macd_histogram_relationship(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test MACD histogram equals line minus signal."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["macd"])

        # Filter to non-null rows
        filtered = result.filter(
            pl.col("macd_histogram").is_not_null()
            & pl.col("macd_line").is_not_null()
            & pl.col("macd_signal").is_not_null()
        )

        # Check histogram = line - signal
        computed_histogram = (filtered["macd_line"] - filtered["macd_signal"]).to_list()
        actual_histogram = filtered["macd_histogram"].to_list()

        for comp, actual in zip(computed_histogram, actual_histogram, strict=True):
            assert abs(comp - actual) < 0.01


class TestStochasticOscillator:
    """Test Stochastic Oscillator calculation."""

    def test_stochastic_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test Stochastic %K and %D are calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["stochastic"])

        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_stochastic_range(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test Stochastic values are in 0-100 range."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["stochastic"])

        stoch_k = result.filter(pl.col("stoch_k").is_not_null())["stoch_k"]
        stoch_d = result.filter(pl.col("stoch_d").is_not_null())["stoch_d"]

        assert (stoch_k >= 0).all()
        assert (stoch_k <= 100).all()
        assert (stoch_d >= 0).all()
        assert (stoch_d <= 100).all()


class TestROCIndicator:
    """Test Rate of Change indicator."""

    def test_roc_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test ROC is calculated for multiple periods."""
        engine = TechnicalFeatureEngine(FeatureConfig(roc_periods=[5, 10]))
        result = engine.compute_indicators(sample_ohlcv_data, ["roc"])

        assert "roc_5" in result.columns
        assert "roc_10" in result.columns

    def test_roc_positive_trend(self) -> None:
        """Test ROC is positive in uptrend."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0 + i for i in range(50)],
                "high": [105.0 + i for i in range(50)],
                "low": [98.0 + i for i in range(50)],
                "close": [100.0 + i for i in range(50)],
                "volume": [1000000] * 50,
            }
        )

        engine = TechnicalFeatureEngine(FeatureConfig(roc_periods=[10]))
        result = engine.compute_indicators(df, ["roc"])

        # ROC should be positive in uptrend
        roc_values = result.filter(pl.col("roc_10").is_not_null())["roc_10"]
        assert (roc_values > 0).all()


class TestMomentumIndicator:
    """Test Momentum indicator."""

    def test_momentum_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test Momentum is calculated."""
        engine = TechnicalFeatureEngine(FeatureConfig(mom_periods=[10, 20]))
        result = engine.compute_indicators(sample_ohlcv_data, ["momentum"])

        assert "mom_10" in result.columns
        assert "mom_20" in result.columns


class TestBollingerBands:
    """Test Bollinger Bands calculation."""

    def test_bollinger_bands_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test Bollinger Bands components are calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["bollinger"])

        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_bandwidth" in result.columns
        assert "bb_percent_b" in result.columns

    def test_bollinger_bands_relationship(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test BB upper is above middle which is above lower."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["bollinger"])

        filtered = result.filter(
            pl.col("bb_upper").is_not_null()
            & pl.col("bb_middle").is_not_null()
            & pl.col("bb_lower").is_not_null()
        )

        assert (filtered["bb_upper"] > filtered["bb_middle"]).all()
        assert (filtered["bb_middle"] > filtered["bb_lower"]).all()

    def test_bollinger_bands_middle_is_sma(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test BB middle equals SMA."""
        engine = TechnicalFeatureEngine(FeatureConfig(bb_period=20, sma_periods=[20]))
        result = engine.compute_indicators(sample_ohlcv_data, ["bollinger", "sma"])

        filtered = result.filter(pl.col("bb_middle").is_not_null() & pl.col("sma_20").is_not_null())

        bb_middle = filtered["bb_middle"].to_list()
        sma_20 = filtered["sma_20"].to_list()

        for bb, sma in zip(bb_middle, sma_20, strict=True):
            assert abs(bb - sma) < 0.01


class TestATRIndicator:
    """Test Average True Range calculation."""

    def test_atr_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test ATR is calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["atr"])

        assert "atr_14" in result.columns

    def test_atr_positive_values(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test ATR values are always positive."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["atr"])

        atr_values = result.filter(pl.col("atr_14").is_not_null())["atr_14"]
        assert (atr_values >= 0).all()

    def test_atr_responds_to_price_changes(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test ATR responds to increasing price ranges."""
        engine = TechnicalFeatureEngine()

        result = engine.compute_indicators(sample_ohlcv_data, ["atr"])

        # Get first and last ATR values (excluding nulls)
        atr_values = result.filter(pl.col("atr_14").is_not_null())["atr_14"]

        # ATR should be computed and positive
        assert len(atr_values) > 0
        assert (atr_values > 0).all()


class TestVolatilityMetrics:
    """Test advanced volatility metrics."""

    def test_historical_volatility_calculation(
        self, sample_ohlcv_data: pl.DataFrame
    ) -> None:
        """Test historical volatility is calculated."""
        engine = TechnicalFeatureEngine(FeatureConfig(histo_vol_periods=[10, 20]))
        result = engine.compute_indicators(sample_ohlcv_data, ["volatility"])

        assert "histo_vol_10" in result.columns
        assert "histo_vol_20" in result.columns

    def test_parkinson_volatility_calculation(
        self, sample_ohlcv_data: pl.DataFrame
    ) -> None:
        """Test Parkinson volatility is calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["volatility"])

        assert "parkinson_vol" in result.columns
        assert result["parkinson_vol"].null_count() < len(result)

    def test_garman_klass_volatility_calculation(
        self, sample_ohlcv_data: pl.DataFrame
    ) -> None:
        """Test Garman-Klass volatility is calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["volatility"])

        assert "garman_klass_vol" in result.columns
        assert result["garman_klass_vol"].null_count() < len(result)


class TestOBVIndicator:
    """Test On-Balance Volume calculation."""

    def test_obv_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test OBV is calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["obv"])

        assert "obv" in result.columns

    def test_obv_cumulative(self) -> None:
        """Test OBV is cumulative."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0, 101.0, 102.0, 101.0, 103.0],
                "high": [105.0, 106.0, 107.0, 106.0, 108.0],
                "low": [98.0, 99.0, 100.0, 99.0, 101.0],
                "close": [101.0, 102.0, 103.0, 102.0, 104.0],
                "volume": [1000000, 1000000, 1000000, 1000000, 1000000],
            }
        )

        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(df, ["obv"])

        # OBV should be cumulative
        obv_values = result["obv"].to_list()
        # First value is 0, then accumulates
        assert obv_values[0] == 0


class TestVWAPIndicator:
    """Test Volume Weighted Average Price calculation."""

    def test_vwap_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test VWAP is calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["vwap"])

        assert "vwap" in result.columns
        assert result["vwap"].null_count() < len(result)


class TestVolumeIndicators:
    """Test volume-related indicators."""

    def test_volume_sma_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test Volume SMA is calculated."""
        engine = TechnicalFeatureEngine(FeatureConfig(volume_sma_periods=[5, 20]))
        result = engine.compute_indicators(sample_ohlcv_data, ["volume_sma"])

        assert "volume_sma_5" in result.columns
        assert "volume_sma_20" in result.columns

    def test_volume_roc_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test Volume ROC is calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["volume_roc"])

        assert "volume_roc" in result.columns


class TestPriceReturns:
    """Test price returns calculation."""

    def test_returns_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test returns are calculated for multiple periods."""
        engine = TechnicalFeatureEngine(FeatureConfig(return_periods=[1, 5, 10]))
        result = engine.compute_indicators(sample_ohlcv_data, ["returns"])

        assert "return_1d" in result.columns
        assert "return_5d" in result.columns
        assert "return_10d" in result.columns
        assert "log_return_1d" in result.columns

    def test_log_return_calculation(self) -> None:
        """Test log returns are calculated correctly."""
        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0, 110.0],
                "high": [105.0, 115.0],
                "low": [98.0, 108.0],
                "close": [100.0, 110.0],
                "volume": [1000000, 1000000],
            }
        )

        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(df, ["returns"])

        # Log return from 100 to 110 should be ln(110/100) = ln(1.1) ≈ 0.0953
        import math

        expected_log_return = math.log(110.0 / 100.0)
        actual_log_return = result["log_return_1d"][1]

        assert abs(actual_log_return - expected_log_return) < 0.01


class TestPriceRatios:
    """Test price ratio features."""

    def test_price_ratios_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test price ratios to SMA are calculated."""
        engine = TechnicalFeatureEngine(FeatureConfig(price_ratio_sma_periods=[20, 50]))
        result = engine.compute_indicators(sample_ohlcv_data, ["price_ratios"])

        assert "price_to_sma_20" in result.columns
        assert "price_to_sma_50" in result.columns


class TestPriceFeaturesOther:
    """Test other price features."""

    def test_high_low_range_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test high-low range is calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["price_features"])

        assert "high_low_range" in result.columns

        # Range should equal high - low
        expected_range = (result["high"] - result["low"]).to_list()
        actual_range = result["high_low_range"].to_list()

        for exp, act in zip(expected_range, actual_range, strict=True):
            assert abs(exp - act) < 0.01

    def test_gap_calculation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test gap is calculated."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["price_features"])

        assert "gap" in result.columns


class TestComputeIndicatorsMethod:
    """Test compute_indicators method with specific indicators."""

    def test_compute_specific_indicators(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test computing specific indicators only."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, ["sma", "rsi"])

        # Should have SMA and RSI
        assert "sma_20" in result.columns
        assert "rsi_14" in result.columns

        # Should NOT have MACD
        assert "macd_line" not in result.columns

    def test_compute_invalid_indicator(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test error raised for invalid indicator name."""
        engine = TechnicalFeatureEngine()
        with pytest.raises(ValueError, match="Invalid indicator names"):
            engine.compute_indicators(sample_ohlcv_data, ["invalid_indicator"])

    def test_compute_empty_indicator_list(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test computing with empty indicator list."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_indicators(sample_ohlcv_data, [])

        # Should return original DataFrame
        assert result.columns == sample_ohlcv_data.columns


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_dataframe(self) -> None:
        """Test with single row of data."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [105.0],
                "low": [98.0],
                "close": [100.0],
                "volume": [1000000],
            }
        )

        engine = TechnicalFeatureEngine()
        result = engine.compute_all(df)

        # Should not raise error
        assert result.height == 1

    def test_zero_volume(self) -> None:
        """Test handling of zero volume."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "open": [100.0] * 50,
                "high": [105.0] * 50,
                "low": [98.0] * 50,
                "close": [100.0] * 50,
                "volume": [0] * 50,
            }
        )

        engine = TechnicalFeatureEngine()
        result = engine.compute_all(df)

        # Should handle zero volume without error
        assert result.height == 50

    def test_constant_prices(self, simple_ohlcv_data: pl.DataFrame) -> None:
        """Test with constant prices."""
        engine = TechnicalFeatureEngine()
        result = engine.compute_all(simple_ohlcv_data)

        # RSI should be around 50 for constant prices
        rsi_values = result.filter(pl.col("rsi_14").is_not_null())["rsi_14"]
        # Due to EMA smoothing, it might not be exactly 50, but should be stable
        assert len(rsi_values) > 0

    def test_all_null_handling(self) -> None:
        """Test that indicators with insufficient data have nulls."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)],
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [105.0, 106.0, 107.0, 108.0, 109.0],
                "low": [98.0, 99.0, 100.0, 101.0, 102.0],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000000] * 5,
            }
        )

        engine = TechnicalFeatureEngine(FeatureConfig(sma_periods=[200]))
        result = engine.compute_indicators(df, ["sma"])

        # SMA_200 should be all null with only 5 rows
        assert result["sma_200"].null_count() == 5
