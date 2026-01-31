"""Tests for quantile regression models.

This module tests quantile regression implementations including:
- Configuration validation
- Model fitting and prediction
- Prediction interval generation
- Coverage calculation and calibration
- Winkler score computation
- Integration with feature engineering

Tests use synthetic data with known properties to verify correctness.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalforge.ml.models.base import BasePredictor
from signalforge.ml.models.quantile_regression import (
    QuantileGradientBoostingRegressor,
    QuantilePrediction,
    QuantileRegressionConfig,
    QuantileRegressor,
    calculate_coverage,
    create_quantile_regressor,
    winkler_score,
)


@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    """Create synthetic OHLCV data with trend.

    Returns:
        DataFrame with 100 rows of OHLCV data with upward trend.
    """
    n_rows = 100
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_rows)]

    # Create realistic price data with trend and volatility
    np.random.seed(42)
    base_price = 100.0
    trend = np.linspace(0, 20, n_rows)
    noise = np.random.normal(0, 2, n_rows)
    close_prices = base_price + trend + noise

    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices * 0.99,
            "high": close_prices * 1.02,
            "low": close_prices * 0.98,
            "close": close_prices,
            "volume": np.random.randint(1000000, 2000000, n_rows),
        }
    )


@pytest.fixture
def sample_with_features() -> pl.DataFrame:
    """Create data with technical indicators for feature testing.

    Returns:
        DataFrame with OHLCV and technical indicators.
    """
    n_rows = 100
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_rows)]

    np.random.seed(42)
    base = 100.0
    trend = np.linspace(0, 15, n_rows)
    noise = np.random.normal(0, 1.5, n_rows)
    close = base + trend + noise

    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000000, 2000000, n_rows),
            "sma_20": close + np.random.normal(0, 0.5, n_rows),
            "rsi": np.random.uniform(30, 70, n_rows),
            "macd": np.random.normal(0, 0.3, n_rows),
        }
    )


@pytest.fixture
def simple_quantile_config() -> QuantileRegressionConfig:
    """Create basic configuration for testing."""
    return QuantileRegressionConfig(quantiles=[0.1, 0.5, 0.9], alpha=0.01, max_iter=500, n_lags=3)


class TestQuantileRegressionConfig:
    """Tests for QuantileRegressionConfig dataclass."""

    def test_default_config(self) -> None:
        """Test configuration with default values."""
        config = QuantileRegressionConfig()

        assert config.quantiles == [0.1, 0.5, 0.9]
        assert config.alpha == 0.1
        assert config.max_iter == 1000
        assert config.features is None
        assert config.solver == "highs-ds"
        assert config.n_lags == 5

    def test_custom_config(self) -> None:
        """Test configuration with custom values."""
        config = QuantileRegressionConfig(
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            alpha=0.5,
            max_iter=2000,
            features=["close", "volume"],
            solver="highs-ipm",
            n_lags=10,
        )

        assert len(config.quantiles) == 5
        assert config.alpha == 0.5
        assert config.max_iter == 2000
        assert config.features == ["close", "volume"]
        assert config.solver == "highs-ipm"
        assert config.n_lags == 10

    def test_invalid_quantiles_range(self) -> None:
        """Test validation of quantile range."""
        with pytest.raises(ValueError, match="must be in range"):
            QuantileRegressionConfig(quantiles=[0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match="must be in range"):
            QuantileRegressionConfig(quantiles=[-0.1, 0.5, 0.9])

        with pytest.raises(ValueError, match="must be in range"):
            QuantileRegressionConfig(quantiles=[0.1, 0.5, 1.5])

    def test_empty_quantiles(self) -> None:
        """Test validation for empty quantiles list."""
        with pytest.raises(ValueError, match="At least one quantile"):
            QuantileRegressionConfig(quantiles=[])

    def test_duplicate_quantiles(self) -> None:
        """Test validation for duplicate quantiles."""
        with pytest.raises(ValueError, match="must be unique"):
            QuantileRegressionConfig(quantiles=[0.1, 0.5, 0.5, 0.9])

    def test_negative_alpha(self) -> None:
        """Test validation of alpha parameter."""
        with pytest.raises(ValueError, match="must be non-negative"):
            QuantileRegressionConfig(alpha=-0.1)

    def test_invalid_max_iter(self) -> None:
        """Test validation of max_iter parameter."""
        with pytest.raises(ValueError, match="must be positive"):
            QuantileRegressionConfig(max_iter=0)

        with pytest.raises(ValueError, match="must be positive"):
            QuantileRegressionConfig(max_iter=-100)

    def test_negative_lags(self) -> None:
        """Test validation of n_lags parameter."""
        with pytest.raises(ValueError, match="must be non-negative"):
            QuantileRegressionConfig(n_lags=-1)


class TestQuantilePrediction:
    """Tests for QuantilePrediction dataclass."""

    def test_valid_prediction(self) -> None:
        """Test creation of valid prediction."""
        pred = QuantilePrediction(
            point_forecast=100.0,
            lower_bound=95.0,
            upper_bound=105.0,
            all_quantiles={0.1: 95.0, 0.5: 100.0, 0.9: 105.0},
            coverage=0.8,
            timestamp=datetime(2024, 1, 1),
        )

        assert pred.point_forecast == 100.0
        assert pred.lower_bound == 95.0
        assert pred.upper_bound == 105.0
        assert pred.coverage == 0.8
        assert len(pred.all_quantiles) == 3

    def test_invalid_coverage(self) -> None:
        """Test validation of coverage parameter."""
        with pytest.raises(ValueError, match="Coverage must be in"):
            QuantilePrediction(
                point_forecast=100.0,
                lower_bound=95.0,
                upper_bound=105.0,
                all_quantiles={0.5: 100.0},
                coverage=1.5,
            )

    def test_inconsistent_bounds_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning for inconsistent prediction bounds."""
        # Lower bound exceeds point forecast
        _pred = QuantilePrediction(
            point_forecast=100.0,
            lower_bound=102.0,
            upper_bound=105.0,
            all_quantiles={0.5: 100.0},
            coverage=0.8,
        )

        assert "exceeds point forecast" in caplog.text

    def test_none_timestamp(self) -> None:
        """Test prediction with no timestamp."""
        pred = QuantilePrediction(
            point_forecast=100.0,
            lower_bound=95.0,
            upper_bound=105.0,
            all_quantiles={0.5: 100.0},
            coverage=0.8,
        )

        assert pred.timestamp is None


class TestQuantileRegressor:
    """Tests for linear QuantileRegressor."""

    def test_initialization(self, simple_quantile_config: QuantileRegressionConfig) -> None:
        """Test model initialization."""
        model = QuantileRegressor(simple_quantile_config)

        assert not model.is_fitted
        assert model.config == simple_quantile_config
        assert len(model._models) == 0
        assert model._training_data is None

    def test_implements_base_predictor(
        self, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test that model implements BasePredictor interface."""
        model = QuantileRegressor(simple_quantile_config)

        assert isinstance(model, BasePredictor)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "evaluate")
        assert hasattr(model, "is_fitted")

    def test_fit_basic(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test basic model fitting."""
        model = QuantileRegressor(simple_quantile_config)
        model.fit(sample_ohlcv_data, target_column="close")

        assert model.is_fitted
        assert len(model._models) == 3  # Three quantiles
        assert model._training_data is not None
        assert model._target_column == "close"
        assert len(model._feature_columns) > 0

    def test_fit_with_features(self, sample_with_features: pl.DataFrame) -> None:
        """Test fitting with specified features."""
        config = QuantileRegressionConfig(
            quantiles=[0.1, 0.5, 0.9], features=["open", "high", "low", "volume", "sma_20"]
        )
        model = QuantileRegressor(config)
        model.fit(sample_with_features, target_column="close")

        assert model.is_fitted
        # Should have specified features + lag features
        assert "open" in model._feature_columns
        assert "sma_20" in model._feature_columns

    def test_fit_empty_dataframe(self, simple_quantile_config: QuantileRegressionConfig) -> None:
        """Test fitting with empty DataFrame."""
        model = QuantileRegressor(simple_quantile_config)
        empty_df = pl.DataFrame()

        with pytest.raises(ValueError, match="empty DataFrame"):
            model.fit(empty_df)

    def test_fit_missing_target(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test fitting with missing target column."""
        model = QuantileRegressor(simple_quantile_config)

        with pytest.raises(ValueError, match="not found"):
            model.fit(sample_ohlcv_data, target_column="nonexistent")

    def test_fit_missing_features(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test fitting with specified features that don't exist."""
        config = QuantileRegressionConfig(features=["nonexistent_feature"])
        model = QuantileRegressor(config)

        with pytest.raises(ValueError, match="not found in DataFrame"):
            model.fit(sample_ohlcv_data)

    def test_fit_insufficient_data(self, simple_quantile_config: QuantileRegressionConfig) -> None:
        """Test fitting with very small dataset."""
        small_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "open": [99.0, 100.0, 101.0, 102.0, 103.0],
            }
        )

        model = QuantileRegressor(simple_quantile_config)

        with pytest.raises(ValueError, match="Insufficient training data"):
            model.fit(small_df)

    def test_predict_basic(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test basic prediction generation."""
        model = QuantileRegressor(simple_quantile_config)
        model.fit(sample_ohlcv_data, target_column="close")

        predictions = model.predict(horizon=10)

        assert predictions.height == 10
        assert "timestamp" in predictions.columns
        assert "prediction" in predictions.columns
        assert "lower_bound" in predictions.columns
        assert "upper_bound" in predictions.columns
        assert "coverage" in predictions.columns
        assert "quantile_0.1" in predictions.columns
        assert "quantile_0.5" in predictions.columns
        assert "quantile_0.9" in predictions.columns

    def test_predict_without_fit(self, simple_quantile_config: QuantileRegressionConfig) -> None:
        """Test prediction before fitting."""
        model = QuantileRegressor(simple_quantile_config)

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(horizon=10)

    def test_predict_invalid_horizon(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test prediction with invalid horizon."""
        model = QuantileRegressor(simple_quantile_config)
        model.fit(sample_ohlcv_data)

        with pytest.raises(ValueError, match="must be positive"):
            model.predict(horizon=0)

        with pytest.raises(ValueError, match="must be positive"):
            model.predict(horizon=-5)

    def test_prediction_intervals_ordered(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test that prediction intervals are properly ordered."""
        model = QuantileRegressor(simple_quantile_config)
        model.fit(sample_ohlcv_data)

        predictions = model.predict(horizon=5)

        # Check that lower <= point <= upper for each prediction
        for i in range(predictions.height):
            row = predictions.row(i, named=True)
            # Allow small numerical errors
            assert row["lower_bound"] <= row["prediction"] + 0.01
            assert row["prediction"] <= row["upper_bound"] + 0.01

    def test_evaluate_basic(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test basic model evaluation."""
        # Split data
        train_df = sample_ohlcv_data.head(70)
        test_df = sample_ohlcv_data.tail(30)

        model = QuantileRegressor(simple_quantile_config)
        model.fit(train_df)

        metrics = model.evaluate(test_df)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "empirical_coverage" in metrics
        assert "expected_coverage" in metrics
        assert "coverage_deviation" in metrics
        assert "winkler_score" in metrics
        assert "interval_width" in metrics

        # Check metric validity
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["mape"] >= 0
        assert 0 <= metrics["empirical_coverage"] <= 1
        assert metrics["expected_coverage"] == 0.8  # 0.9 - 0.1
        assert metrics["interval_width"] > 0

    def test_evaluate_without_fit(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test evaluation before fitting."""
        model = QuantileRegressor(simple_quantile_config)

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.evaluate(sample_ohlcv_data)

    def test_evaluate_empty_dataframe(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test evaluation with empty test set."""
        model = QuantileRegressor(simple_quantile_config)
        model.fit(sample_ohlcv_data)

        empty_df = pl.DataFrame()

        with pytest.raises(ValueError, match="cannot be empty"):
            model.evaluate(empty_df)

    def test_evaluate_missing_target(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test evaluation with missing target column."""
        model = QuantileRegressor(simple_quantile_config)
        model.fit(sample_ohlcv_data.head(70), target_column="close")

        test_df = sample_ohlcv_data.tail(30).drop("close")

        with pytest.raises(ValueError, match="not found"):
            model.evaluate(test_df)


class TestQuantileGradientBoostingRegressor:
    """Tests for gradient boosting quantile regressor."""

    def test_initialization(self, simple_quantile_config: QuantileRegressionConfig) -> None:
        """Test model initialization."""
        model = QuantileGradientBoostingRegressor(
            simple_quantile_config, n_estimators=50, learning_rate=0.05, max_depth=4
        )

        assert not model.is_fitted
        assert model.config == simple_quantile_config
        assert model.n_estimators == 50
        assert model.learning_rate == 0.05
        assert model.max_depth == 4

    def test_implements_base_predictor(
        self, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test that model implements BasePredictor interface."""
        model = QuantileGradientBoostingRegressor(simple_quantile_config)

        assert isinstance(model, BasePredictor)

    def test_fit_basic(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test basic model fitting."""
        model = QuantileGradientBoostingRegressor(
            simple_quantile_config, n_estimators=20, random_state=42
        )
        model.fit(sample_ohlcv_data, target_column="close")

        assert model.is_fitted
        assert len(model._models) == 3

    def test_predict_basic(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test prediction generation."""
        model = QuantileGradientBoostingRegressor(
            simple_quantile_config, n_estimators=20, random_state=42
        )
        model.fit(sample_ohlcv_data)

        predictions = model.predict(horizon=5)

        assert predictions.height == 5
        assert "prediction" in predictions.columns
        assert "lower_bound" in predictions.columns
        assert "upper_bound" in predictions.columns

    def test_evaluate_basic(
        self, sample_ohlcv_data: pl.DataFrame, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test model evaluation."""
        train_df = sample_ohlcv_data.head(70)
        test_df = sample_ohlcv_data.tail(30)

        model = QuantileGradientBoostingRegressor(
            simple_quantile_config, n_estimators=20, random_state=42
        )
        model.fit(train_df)

        metrics = model.evaluate(test_df)

        assert "rmse" in metrics
        assert "empirical_coverage" in metrics
        assert metrics["rmse"] >= 0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_quantile_regressor_linear(
        self, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test factory function for linear model."""
        model = create_quantile_regressor(simple_quantile_config, method="linear")

        assert isinstance(model, QuantileRegressor)
        assert isinstance(model, BasePredictor)

    def test_create_quantile_regressor_gbm(
        self, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test factory function for GBM model."""
        model = create_quantile_regressor(simple_quantile_config, method="gbm")

        assert isinstance(model, QuantileGradientBoostingRegressor)
        assert isinstance(model, BasePredictor)

    def test_create_quantile_regressor_invalid_method(
        self, simple_quantile_config: QuantileRegressionConfig
    ) -> None:
        """Test factory function with invalid method."""
        with pytest.raises(ValueError, match="Invalid method"):
            create_quantile_regressor(simple_quantile_config, method="invalid")  # type: ignore

    def test_calculate_coverage_basic(self) -> None:
        """Test coverage calculation."""
        predictions = pl.DataFrame(
            {
                "lower_bound": [95.0, 98.0, 101.0, 104.0],
                "upper_bound": [105.0, 108.0, 111.0, 114.0],
            }
        )
        actuals = pl.Series([100.0, 107.0, 115.0, 109.0])  # 3 out of 4 in interval

        coverage = calculate_coverage(predictions, actuals)

        assert coverage == 0.75

    def test_calculate_coverage_perfect(self) -> None:
        """Test coverage with all values in interval."""
        predictions = pl.DataFrame(
            {"lower_bound": [95.0, 95.0, 95.0], "upper_bound": [105.0, 105.0, 105.0]}
        )
        actuals = pl.Series([100.0, 98.0, 102.0])

        coverage = calculate_coverage(predictions, actuals)

        assert coverage == 1.0

    def test_calculate_coverage_none(self) -> None:
        """Test coverage with no values in interval."""
        predictions = pl.DataFrame(
            {"lower_bound": [95.0, 95.0, 95.0], "upper_bound": [100.0, 100.0, 100.0]}
        )
        actuals = pl.Series([110.0, 120.0, 115.0])

        coverage = calculate_coverage(predictions, actuals)

        assert coverage == 0.0

    def test_calculate_coverage_custom_columns(self) -> None:
        """Test coverage with custom column names."""
        predictions = pl.DataFrame({"pred_lower": [95.0, 98.0], "pred_upper": [105.0, 108.0]})
        actuals = pl.Series([100.0, 107.0])

        coverage = calculate_coverage(
            predictions, actuals, lower_col="pred_lower", upper_col="pred_upper"
        )

        assert coverage == 1.0

    def test_calculate_coverage_missing_columns(self) -> None:
        """Test coverage with missing columns."""
        predictions = pl.DataFrame({"lower": [95.0], "upper": [105.0]})
        actuals = pl.Series([100.0])

        with pytest.raises(ValueError, match="must exist"):
            calculate_coverage(predictions, actuals)

    def test_calculate_coverage_length_mismatch(self) -> None:
        """Test coverage with length mismatch."""
        predictions = pl.DataFrame({"lower_bound": [95.0, 98.0], "upper_bound": [105.0, 108.0]})
        actuals = pl.Series([100.0])

        with pytest.raises(ValueError, match="Length mismatch"):
            calculate_coverage(predictions, actuals)

    def test_winkler_score_perfect(self) -> None:
        """Test Winkler score with perfect intervals (no violations)."""
        lower = np.array([95.0, 98.0, 101.0])
        upper = np.array([105.0, 108.0, 111.0])
        actual = np.array([100.0, 103.0, 106.0])
        alpha = 0.2

        score = winkler_score(lower, upper, actual, alpha)

        # Should equal average interval width (10.0)
        assert score == 10.0

    def test_winkler_score_with_violations(self) -> None:
        """Test Winkler score with interval violations."""
        lower = np.array([95.0, 98.0])
        upper = np.array([105.0, 108.0])
        actual = np.array([110.0, 90.0])  # Both violate
        alpha = 0.2

        score = winkler_score(lower, upper, actual, alpha)

        # Should be higher due to penalties
        assert score > 10.0  # Greater than just interval width

    def test_winkler_score_narrow_intervals(self) -> None:
        """Test that narrower intervals get better scores (if no violations)."""
        lower_wide = np.array([90.0, 90.0])
        upper_wide = np.array([110.0, 110.0])
        lower_narrow = np.array([95.0, 95.0])
        upper_narrow = np.array([105.0, 105.0])
        actual = np.array([100.0, 100.0])
        alpha = 0.2

        score_wide = winkler_score(lower_wide, upper_wide, actual, alpha)
        score_narrow = winkler_score(lower_narrow, upper_narrow, actual, alpha)

        assert score_narrow < score_wide

    def test_winkler_score_invalid_alpha(self) -> None:
        """Test Winkler score with invalid alpha."""
        lower = np.array([95.0])
        upper = np.array([105.0])
        actual = np.array([100.0])

        with pytest.raises(ValueError, match="Alpha must be in"):
            winkler_score(lower, upper, actual, alpha=0.0)

        with pytest.raises(ValueError, match="Alpha must be in"):
            winkler_score(lower, upper, actual, alpha=1.5)

    def test_winkler_score_length_mismatch(self) -> None:
        """Test Winkler score with mismatched array lengths."""
        lower = np.array([95.0, 98.0])
        upper = np.array([105.0])
        actual = np.array([100.0])

        with pytest.raises(ValueError, match="same length"):
            winkler_score(lower, upper, actual, alpha=0.2)


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_workflow_linear(self, sample_with_features: pl.DataFrame) -> None:
        """Test complete workflow with linear model."""
        # Split data
        train_df = sample_with_features.head(70)
        test_df = sample_with_features.tail(30)

        # Create and train model
        config = QuantileRegressionConfig(quantiles=[0.05, 0.5, 0.95], alpha=0.01, n_lags=3)
        model = QuantileRegressor(config)
        model.fit(train_df, target_column="close")

        # Generate predictions
        predictions = model.predict(horizon=10)
        assert predictions.height == 10

        # Evaluate
        metrics = model.evaluate(test_df)
        assert metrics["rmse"] >= 0
        assert 0 <= metrics["empirical_coverage"] <= 1

        # Check calibration
        coverage_diff = abs(metrics["empirical_coverage"] - metrics["expected_coverage"])
        # Allow some deviation due to small sample
        assert coverage_diff <= 0.3

    def test_full_workflow_gbm(self, sample_with_features: pl.DataFrame) -> None:
        """Test complete workflow with gradient boosting."""
        train_df = sample_with_features.head(70)
        test_df = sample_with_features.tail(30)

        config = QuantileRegressionConfig(quantiles=[0.1, 0.5, 0.9], n_lags=2)
        model = QuantileGradientBoostingRegressor(
            config, n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42
        )

        model.fit(train_df)
        predictions = model.predict(horizon=5)
        metrics = model.evaluate(test_df)

        assert predictions.height == 5
        assert metrics["rmse"] >= 0

    def test_multiple_quantiles(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test with multiple quantiles for detailed uncertainty."""
        config = QuantileRegressionConfig(
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], alpha=0.01, n_lags=2
        )
        model = QuantileRegressor(config)
        model.fit(sample_ohlcv_data.head(80))

        predictions = model.predict(horizon=3)

        # Check all quantile columns exist
        for q in config.quantiles:
            assert f"quantile_{q}" in predictions.columns

        # Verify ordering: q0.05 <= q0.25 <= q0.5 <= q0.75 <= q0.95
        # Allow small numerical tolerance for floating point errors
        first_pred = predictions.row(0, named=True)
        tolerance = 1e-6
        assert first_pred["quantile_0.05"] <= first_pred["quantile_0.25"] + tolerance
        assert first_pred["quantile_0.25"] <= first_pred["quantile_0.5"] + tolerance
        assert first_pred["quantile_0.5"] <= first_pred["quantile_0.75"] + tolerance
        assert first_pred["quantile_0.75"] <= first_pred["quantile_0.95"] + tolerance

    def test_no_lag_features(self, sample_with_features: pl.DataFrame) -> None:
        """Test model with zero lag features."""
        config = QuantileRegressionConfig(
            quantiles=[0.1, 0.5, 0.9],
            features=["open", "high", "low", "volume"],
            n_lags=0,
        )
        model = QuantileRegressor(config)
        model.fit(sample_with_features)

        assert model.is_fitted
        predictions = model.predict(horizon=5)
        assert predictions.height == 5
