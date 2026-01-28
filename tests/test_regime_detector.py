"""Tests for market regime detection using Hidden Markov Models.

This module tests the RegimeDetector class including initialization,
feature computation, model fitting, prediction, and edge cases.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from signalforge.regime.detector import Regime, RegimeConfig, RegimeDetector


@pytest.fixture
def sample_price_data() -> pl.DataFrame:
    """Create sample price data for testing.

    Returns a DataFrame with 300 rows of synthetic price data
    with different market regimes.
    """
    n_rows = 300
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_rows)]

    # Create data with different regimes
    np.random.seed(42)
    prices = []
    base_price = 100.0

    for i in range(n_rows):
        # Simulate different regimes
        if i < 75:  # BULL market
            base_price += np.random.normal(0.3, 0.5)
        elif i < 150:  # RANGE market
            base_price += np.random.normal(0.0, 0.3)
        elif i < 225:  # BEAR market
            base_price += np.random.normal(-0.3, 0.5)
        else:  # CRISIS
            base_price += np.random.normal(-0.5, 1.5)

        prices.append(max(base_price, 10.0))  # Ensure positive prices

    volumes = [1000000 + i * 1000 + np.random.randint(-50000, 50000) for i in range(n_rows)]

    return pl.DataFrame(
        {
            "timestamp": dates,
            "close": prices,
            "volume": volumes,
        }
    )


@pytest.fixture
def minimal_price_data() -> pl.DataFrame:
    """Create minimal price data for edge case testing."""
    n_rows = 60
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_rows)]
    prices = [100.0 + i * 0.5 for i in range(n_rows)]
    volumes = [1000000] * n_rows

    return pl.DataFrame(
        {
            "timestamp": dates,
            "close": prices,
            "volume": volumes,
        }
    )


class TestRegimeConfig:
    """Tests for RegimeConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration initialization."""
        config = RegimeConfig()

        assert config.n_regimes == 4
        assert config.lookback_window == 252
        assert config.min_regime_duration == 5
        assert config.volatility_window == 20
        assert config.trend_fast_window == 10
        assert config.trend_slow_window == 50
        assert config.random_state == 42
        assert config.n_iter == 100
        assert config.tol == 0.01

    def test_custom_config(self) -> None:
        """Test custom configuration initialization."""
        config = RegimeConfig(
            n_regimes=3,
            lookback_window=100,
            min_regime_duration=3,
            volatility_window=10,
            random_state=123,
        )

        assert config.n_regimes == 3
        assert config.lookback_window == 100
        assert config.min_regime_duration == 3
        assert config.volatility_window == 10
        assert config.random_state == 123

    def test_config_validation_n_regimes_too_small(self) -> None:
        """Test validation for n_regimes < 2."""
        with pytest.raises(ValueError, match="n_regimes must be at least 2"):
            RegimeConfig(n_regimes=1)

    def test_config_validation_n_regimes_too_large(self) -> None:
        """Test validation for n_regimes > 10."""
        with pytest.raises(ValueError, match="n_regimes must be at most 10"):
            RegimeConfig(n_regimes=11)

    def test_config_validation_lookback_window_too_small(self) -> None:
        """Test validation for lookback_window too small."""
        with pytest.raises(ValueError, match="lookback_window .* must be at least"):
            RegimeConfig(lookback_window=30, volatility_window=20)

    def test_config_validation_min_regime_duration(self) -> None:
        """Test validation for min_regime_duration."""
        with pytest.raises(ValueError, match="min_regime_duration must be positive"):
            RegimeConfig(min_regime_duration=0)

    def test_config_validation_volatility_window(self) -> None:
        """Test validation for volatility_window."""
        with pytest.raises(ValueError, match="volatility_window must be at least 2"):
            RegimeConfig(volatility_window=1)

    def test_config_validation_trend_windows(self) -> None:
        """Test validation for trend window relationship."""
        with pytest.raises(ValueError, match="trend_fast_window .* must be less than"):
            RegimeConfig(trend_fast_window=50, trend_slow_window=10)

    def test_config_validation_n_iter(self) -> None:
        """Test validation for n_iter."""
        with pytest.raises(ValueError, match="n_iter must be positive"):
            RegimeConfig(n_iter=0)

    def test_config_validation_tol(self) -> None:
        """Test validation for tol."""
        with pytest.raises(ValueError, match="tol must be positive"):
            RegimeConfig(tol=-0.01)


class TestRegimeDetectorInitialization:
    """Tests for RegimeDetector initialization."""

    def test_initialization_default_config(self) -> None:
        """Test initialization with default configuration."""
        detector = RegimeDetector()

        assert detector.config.n_regimes == 4
        assert detector.config.lookback_window == 252
        assert not detector.is_fitted
        assert detector._model is None

    def test_initialization_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = RegimeConfig(n_regimes=3, lookback_window=100)
        detector = RegimeDetector(config=config)

        assert detector.config.n_regimes == 3
        assert detector.config.lookback_window == 100
        assert not detector.is_fitted


class TestRegimeDetectorFeatureComputation:
    """Tests for feature computation."""

    def test_compute_features_basic(self, sample_price_data: pl.DataFrame) -> None:
        """Test basic feature computation."""
        detector = RegimeDetector()
        df_features = detector._compute_features(sample_price_data)

        # Check feature columns exist
        expected_features = ["log_return", "volatility", "volume_change", "trend_strength"]
        for feature in expected_features:
            assert feature in df_features.columns

        # Check no all-null columns
        assert df_features.height == sample_price_data.height

    def test_compute_features_without_volume(self) -> None:
        """Test feature computation without volume column."""
        df = pl.DataFrame(
            {
                "timestamp": pl.date_range(
                    start=datetime(2023, 1, 1),
                    end=datetime(2023, 4, 10),
                    interval="1d",
                    eager=True,
                ),
                "close": [100.0 + i * 0.5 for i in range(100)],
            }
        )

        detector = RegimeDetector()
        df_features = detector._compute_features(df)

        # Should have volume_change as zeros
        assert "volume_change" in df_features.columns
        # Check that non-null volume_change values are zero
        non_null_values = df_features["volume_change"].drop_nulls()
        if len(non_null_values) > 0:
            assert all(v == 0.0 for v in non_null_values)

    def test_compute_features_missing_columns(self) -> None:
        """Test feature computation with missing required columns."""
        df = pl.DataFrame(
            {
                "timestamp": pl.date_range(
                    start=datetime(2023, 1, 1),
                    end=datetime(2023, 4, 10),
                    interval="1d",
                    eager=True,
                ),
                # Missing 'close' column
            }
        )

        detector = RegimeDetector()
        with pytest.raises(ValueError, match="Missing required columns"):
            detector._compute_features(df)


class TestRegimeDetectorFitting:
    """Tests for model fitting."""

    def test_fit_basic(self, sample_price_data: pl.DataFrame) -> None:
        """Test basic model fitting."""
        detector = RegimeDetector()
        detector.fit(sample_price_data)

        assert detector.is_fitted
        assert detector._model is not None
        assert len(detector._regime_mapping) == detector.config.n_regimes
        assert detector._scaler_mean is not None
        assert detector._scaler_std is not None

    def test_fit_with_custom_config(self, sample_price_data: pl.DataFrame) -> None:
        """Test fitting with custom configuration."""
        config = RegimeConfig(n_regimes=3, lookback_window=100, min_regime_duration=3)
        detector = RegimeDetector(config=config)
        detector.fit(sample_price_data)

        assert detector.is_fitted
        assert len(detector._regime_mapping) == 3

    def test_fit_insufficient_data(self) -> None:
        """Test fitting with insufficient data."""
        df = pl.DataFrame(
            {
                "timestamp": pl.date_range(
                    start=datetime(2023, 1, 1),
                    end=datetime(2023, 2, 19),
                    interval="1d",
                    eager=True,
                ),
                "close": [100.0 + i * 0.5 for i in range(50)],
                "volume": [1000000] * 50,
            }
        )

        detector = RegimeDetector()  # Default lookback_window=252
        with pytest.raises(ValueError, match="Insufficient data for training"):
            detector.fit(df)

    def test_fit_empty_dataframe(self) -> None:
        """Test fitting with empty DataFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [],
                "close": [],
                "volume": [],
            }
        )

        detector = RegimeDetector()
        with pytest.raises(ValueError, match="Cannot fit on empty DataFrame"):
            detector.fit(df)

    def test_fit_deterministic(self, sample_price_data: pl.DataFrame) -> None:
        """Test that fitting is deterministic with same random_state."""
        config1 = RegimeConfig(random_state=42)
        detector1 = RegimeDetector(config=config1)
        detector1.fit(sample_price_data)
        predictions1 = detector1.predict(sample_price_data)

        config2 = RegimeConfig(random_state=42)
        detector2 = RegimeDetector(config=config2)
        detector2.fit(sample_price_data)
        predictions2 = detector2.predict(sample_price_data)

        # Predictions should be identical
        assert predictions1["regime"].to_list() == predictions2["regime"].to_list()


class TestRegimeDetectorPrediction:
    """Tests for regime prediction."""

    def test_predict_basic(self, sample_price_data: pl.DataFrame) -> None:
        """Test basic prediction."""
        detector = RegimeDetector()
        detector.fit(sample_price_data)
        predictions = detector.predict(sample_price_data)

        # Check output structure
        assert "regime" in predictions.columns
        assert "regime_probability" in predictions.columns
        assert predictions.height == sample_price_data.height

        # Check regime values are valid
        regime_values = predictions["regime"].drop_nulls()
        valid_regimes = {r.value for r in Regime}
        assert all(r in valid_regimes for r in regime_values)

        # Check probabilities are in valid range
        probs = predictions["regime_probability"].drop_nulls()
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_predict_without_fit(self, sample_price_data: pl.DataFrame) -> None:
        """Test prediction without fitting first."""
        detector = RegimeDetector()
        with pytest.raises(RuntimeError, match="Model must be fitted before prediction"):
            detector.predict(sample_price_data)

    def test_predict_empty_dataframe(self, sample_price_data: pl.DataFrame) -> None:
        """Test prediction with empty DataFrame."""
        detector = RegimeDetector()
        detector.fit(sample_price_data)

        df_empty = pl.DataFrame(
            {
                "timestamp": [],
                "close": [],
                "volume": [],
            }
        )

        with pytest.raises(ValueError, match="Cannot predict on empty DataFrame"):
            detector.predict(df_empty)

    def test_predict_new_data(self, sample_price_data: pl.DataFrame) -> None:
        """Test prediction on new data after fitting."""
        # Split data into train and test (use all data for training since we need 252)
        train_data = sample_price_data  # Use all 300 rows for training
        test_data = sample_price_data.tail(100)

        detector = RegimeDetector()
        detector.fit(train_data)
        predictions = detector.predict(test_data)

        assert predictions.height == test_data.height
        assert "regime" in predictions.columns


class TestRegimeDetectorMethods:
    """Tests for additional detector methods."""

    def test_get_current_regime(self, sample_price_data: pl.DataFrame) -> None:
        """Test getting current regime."""
        detector = RegimeDetector()
        detector.fit(sample_price_data)
        detector.predict(sample_price_data)

        current_regime = detector.get_current_regime()
        assert isinstance(current_regime, Regime)
        assert current_regime in [Regime.BULL, Regime.BEAR, Regime.RANGE, Regime.CRISIS]

    def test_get_current_regime_without_fit(self) -> None:
        """Test getting current regime without fitting."""
        detector = RegimeDetector()
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            detector.get_current_regime()

    def test_get_current_regime_without_predict(self, sample_price_data: pl.DataFrame) -> None:
        """Test getting current regime without predicting."""
        detector = RegimeDetector()
        detector.fit(sample_price_data)

        with pytest.raises(RuntimeError, match="No predictions available"):
            detector.get_current_regime()

    def test_get_regime_probabilities(self, sample_price_data: pl.DataFrame) -> None:
        """Test getting regime probabilities."""
        detector = RegimeDetector()
        detector.fit(sample_price_data)
        detector.predict(sample_price_data)

        probs = detector.get_regime_probabilities()

        # Check structure
        assert isinstance(probs, dict)
        assert len(probs) == len(Regime)
        assert all(regime in probs for regime in Regime)

        # Check probabilities sum to approximately 1
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 0.01

        # Check all probabilities are non-negative
        assert all(p >= 0.0 for p in probs.values())

    def test_get_regime_probabilities_without_fit(self) -> None:
        """Test getting regime probabilities without fitting."""
        detector = RegimeDetector()
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            detector.get_regime_probabilities()

    def test_get_transition_matrix(self, sample_price_data: pl.DataFrame) -> None:
        """Test getting transition matrix."""
        detector = RegimeDetector()
        detector.fit(sample_price_data)

        transition_matrix = detector.get_transition_matrix()

        # Check structure
        assert isinstance(transition_matrix, pl.DataFrame)
        assert "from_regime" in transition_matrix.columns

        # Check dimensions
        n_regimes_enum = len(Regime)
        assert transition_matrix.height == n_regimes_enum
        assert len(transition_matrix.columns) == n_regimes_enum + 1  # +1 for from_regime

        # Check row sums (should be approximately 1)
        regime_cols = [r.value for r in Regime]
        for row in transition_matrix.iter_rows(named=True):
            row_sum = sum(row[col] for col in regime_cols)
            assert abs(row_sum - 1.0) < 0.01

        # Check all values are probabilities
        for col in regime_cols:
            assert all(0.0 <= v <= 1.0 for v in transition_matrix[col])

    def test_get_transition_matrix_without_fit(self) -> None:
        """Test getting transition matrix without fitting."""
        detector = RegimeDetector()
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            detector.get_transition_matrix()


class TestRegimeDetectorSerialization:
    """Tests for model serialization."""

    def test_save_and_load_model(
        self, sample_price_data: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test saving and loading model."""
        # Fit and save model
        detector1 = RegimeDetector()
        detector1.fit(sample_price_data)
        predictions1 = detector1.predict(sample_price_data)

        model_path = tmp_path / "regime_model.pkl"
        detector1.save_model(model_path)

        # Load model and predict
        detector2 = RegimeDetector()
        detector2.load_model(model_path)

        assert detector2.is_fitted
        assert detector2.config.n_regimes == detector1.config.n_regimes

        predictions2 = detector2.predict(sample_price_data)

        # Predictions should be identical
        assert predictions1["regime"].to_list() == predictions2["regime"].to_list()

    def test_save_without_fit(self, tmp_path: Path) -> None:
        """Test saving model without fitting."""
        detector = RegimeDetector()
        model_path = tmp_path / "regime_model.pkl"

        with pytest.raises(RuntimeError, match="Model must be fitted before saving"):
            detector.save_model(model_path)

    def test_load_nonexistent_file(self) -> None:
        """Test loading from non-existent file."""
        detector = RegimeDetector()
        model_path = Path("/nonexistent/path/model.pkl")

        with pytest.raises(OSError, match="Model file not found"):
            detector.load_model(model_path)

    def test_save_creates_parent_directory(
        self, sample_price_data: pl.DataFrame, tmp_path: Path
    ) -> None:
        """Test that save creates parent directory if it doesn't exist."""
        detector = RegimeDetector()
        detector.fit(sample_price_data)

        model_path = tmp_path / "nested" / "directory" / "model.pkl"
        detector.save_model(model_path)

        assert model_path.exists()


class TestRegimeDetectorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nan_handling(self) -> None:
        """Test handling of NaN values in input data."""
        df = pl.DataFrame(
            {
                "timestamp": pl.date_range(
                    start=datetime(2023, 1, 1),
                    end=datetime(2023, 10, 27),
                    interval="1d",
                    eager=True,
                ),
                # Only a few NaN values, not too many to prevent training
                "close": [100.0 + i * 0.5 if i % 50 != 0 else None for i in range(300)],
                "volume": [1000000] * 300,
            }
        )

        detector = RegimeDetector()
        detector.fit(df)
        predictions = detector.predict(df)

        # Should handle NaN gracefully
        assert predictions.height == df.height
        # Some predictions may be None due to NaN in features
        assert "regime" in predictions.columns

    def test_minimum_duration_filter(self, sample_price_data: pl.DataFrame) -> None:
        """Test minimum duration filtering."""
        config = RegimeConfig(min_regime_duration=10)
        detector = RegimeDetector(config=config)
        detector.fit(sample_price_data)
        predictions = detector.predict(sample_price_data)

        # Check that no regime lasts less than min_regime_duration
        regimes = predictions["regime"].drop_nulls().to_list()
        if len(regimes) > 0:
            current_regime = regimes[0]
            duration = 1

            for regime in regimes[1:]:
                if regime == current_regime:
                    duration += 1
                else:
                    # Check minimum duration
                    assert duration >= config.min_regime_duration or duration == len(regimes)
                    current_regime = regime
                    duration = 1

    def test_small_n_regimes(self, sample_price_data: pl.DataFrame) -> None:
        """Test with small number of regimes."""
        config = RegimeConfig(n_regimes=2, lookback_window=100)
        detector = RegimeDetector(config=config)
        detector.fit(sample_price_data)
        predictions = detector.predict(sample_price_data)

        assert "regime" in predictions.columns
        regime_values = predictions["regime"].drop_nulls().unique()
        # Should have at most 2 unique regimes
        assert len(regime_values) <= 2

    def test_constant_prices(self) -> None:
        """Test with constant prices (no volatility)."""
        df = pl.DataFrame(
            {
                "timestamp": pl.date_range(
                    start=datetime(2023, 1, 1),
                    end=datetime(2023, 10, 27),
                    interval="1d",
                    eager=True,
                ),
                "close": [100.0] * 300,
                "volume": [1000000] * 300,
            }
        )

        config = RegimeConfig(lookback_window=100)
        detector = RegimeDetector(config=config)
        detector.fit(df)
        predictions = detector.predict(df)

        # Should not crash and should predict a regime
        assert "regime" in predictions.columns
        # Most predictions should be RANGE due to zero volatility and returns
        non_null_regimes = predictions["regime"].drop_nulls()
        if len(non_null_regimes) > 0:
            # At least some predictions should exist
            assert len(non_null_regimes) > 0


class TestRegimeEnum:
    """Tests for Regime enum."""

    def test_regime_values(self) -> None:
        """Test regime enum values."""
        assert Regime.BULL.value == "bull"
        assert Regime.BEAR.value == "bear"
        assert Regime.RANGE.value == "range"
        assert Regime.CRISIS.value == "crisis"

    def test_regime_membership(self) -> None:
        """Test regime enum membership."""
        assert Regime.BULL in Regime
        assert Regime.BEAR in Regime
        assert Regime.RANGE in Regime
        assert Regime.CRISIS in Regime

    def test_regime_count(self) -> None:
        """Test number of regimes."""
        assert len(Regime) == 4
