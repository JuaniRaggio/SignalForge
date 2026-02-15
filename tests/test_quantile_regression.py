"""Tests for quantile regression models.

This module tests quantile regression implementations including:
- Configuration validation
- Model fitting and prediction
- Prediction interval generation
- Coverage calculation
- Winkler score computation
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from signalforge.ml.models.base import BasePredictor
from signalforge.ml.models.quantile_regression import (
    QuantilePrediction,
    QuantileRegressionConfig,
    QuantileRegressionPredictor,
    calculate_coverage,
    create_quantile_regressor,
    winkler_score,
)


@pytest.fixture
def sample_features() -> pl.DataFrame:
    """Create synthetic feature data for testing.

    Returns:
        DataFrame with 100 rows of feature data.
    """
    n_rows = 100
    np.random.seed(42)

    return pl.DataFrame(
        {
            "feature1": np.random.randn(n_rows),
            "feature2": np.random.randn(n_rows),
            "feature3": np.random.randn(n_rows),
        }
    )


@pytest.fixture
def sample_target() -> pl.Series:
    """Create synthetic target data for testing.

    Returns:
        Series with 100 target values.
    """
    np.random.seed(42)
    n_rows = 100
    # Target with some pattern + noise
    return pl.Series(np.linspace(100, 120, n_rows) + np.random.randn(n_rows) * 2)


@pytest.fixture
def simple_quantile_config() -> QuantileRegressionConfig:
    """Create basic configuration for testing."""
    return QuantileRegressionConfig(
        quantiles=[0.1, 0.5, 0.9],
        base_model="linear",
        alpha=1.0,
    )


class TestQuantileRegressionConfig:
    """Tests for QuantileRegressionConfig dataclass."""

    def test_default_config(self) -> None:
        """Test configuration with default values."""
        config = QuantileRegressionConfig()

        assert config.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]
        assert config.base_model == "gradient_boosting"
        assert config.n_estimators == 100
        assert config.learning_rate == 0.1
        assert config.max_depth == 3
        assert config.alpha == 1.0

    def test_custom_config(self) -> None:
        """Test configuration with custom values."""
        config = QuantileRegressionConfig(
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            base_model="linear",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            alpha=0.5,
        )

        assert len(config.quantiles) == 5
        assert config.base_model == "linear"
        assert config.n_estimators == 200
        assert config.learning_rate == 0.05
        assert config.max_depth == 5
        assert config.alpha == 0.5

    def test_invalid_n_estimators(self) -> None:
        """Test validation of n_estimators parameter."""
        with pytest.raises(ValueError):
            QuantileRegressionConfig(n_estimators=0)

        with pytest.raises(ValueError):
            QuantileRegressionConfig(n_estimators=-100)

    def test_invalid_learning_rate(self) -> None:
        """Test validation of learning_rate parameter."""
        with pytest.raises(ValueError):
            QuantileRegressionConfig(learning_rate=0.0)

        with pytest.raises(ValueError):
            QuantileRegressionConfig(learning_rate=-0.1)

    def test_invalid_max_depth(self) -> None:
        """Test validation of max_depth parameter."""
        with pytest.raises(ValueError):
            QuantileRegressionConfig(max_depth=0)

    def test_invalid_alpha(self) -> None:
        """Test validation of alpha parameter."""
        with pytest.raises(ValueError):
            QuantileRegressionConfig(alpha=0.0)

        with pytest.raises(ValueError):
            QuantileRegressionConfig(alpha=-0.1)


class TestQuantilePrediction:
    """Tests for QuantilePrediction dataclass."""

    def test_valid_prediction(self) -> None:
        """Test creation of valid prediction."""
        pred = QuantilePrediction(
            prediction=100.0,
            quantile_values={0.1: 95.0, 0.5: 100.0, 0.9: 105.0},
            lower_bound=95.0,
            upper_bound=105.0,
            confidence=0.8,
        )

        assert pred.prediction == 100.0
        assert pred.lower_bound == 95.0
        assert pred.upper_bound == 105.0
        assert pred.confidence == 0.8
        assert len(pred.quantile_values) == 3

    def test_default_values(self) -> None:
        """Test prediction with default values."""
        pred = QuantilePrediction(prediction=100.0)

        assert pred.prediction == 100.0
        assert pred.quantile_values == {}
        assert pred.lower_bound == 0.0
        assert pred.upper_bound == 0.0
        assert pred.confidence == 0.0


class TestQuantileRegressionPredictor:
    """Tests for QuantileRegressionPredictor class."""

    def test_initialization_default(self) -> None:
        """Test predictor initialization with defaults."""
        predictor = QuantileRegressionPredictor()

        assert predictor.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]
        assert predictor.base_model == "gradient_boosting"
        assert predictor._fitted is False

    def test_initialization_custom(self) -> None:
        """Test predictor initialization with custom values."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )

        assert predictor.quantiles == [0.1, 0.5, 0.9]
        assert predictor.base_model == "linear"

    def test_initialization_invalid_empty_quantiles(self) -> None:
        """Test error with empty quantiles."""
        with pytest.raises(ValueError, match="At least one quantile"):
            QuantileRegressionPredictor(quantiles=[])

    def test_initialization_invalid_quantile_range(self) -> None:
        """Test error with quantiles outside (0, 1)."""
        with pytest.raises(ValueError, match="must be in range"):
            QuantileRegressionPredictor(quantiles=[0.0, 0.5, 0.9])

        with pytest.raises(ValueError, match="must be in range"):
            QuantileRegressionPredictor(quantiles=[0.1, 0.5, 1.0])

        with pytest.raises(ValueError, match="must be in range"):
            QuantileRegressionPredictor(quantiles=[-0.1, 0.5, 0.9])

    def test_initialization_duplicate_quantiles(self) -> None:
        """Test error with duplicate quantiles."""
        with pytest.raises(ValueError, match="must be unique"):
            QuantileRegressionPredictor(quantiles=[0.1, 0.5, 0.5, 0.9])

    def test_implements_base_predictor(self) -> None:
        """Test that predictor implements BasePredictor interface."""
        predictor = QuantileRegressionPredictor()

        assert isinstance(predictor, BasePredictor)
        assert hasattr(predictor, "fit")
        assert hasattr(predictor, "predict")
        assert hasattr(predictor, "model_name")
        assert hasattr(predictor, "model_version")

    def test_fit_basic(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test basic model fitting."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )
        result = predictor.fit(sample_features, sample_target)

        assert predictor._fitted is True
        assert len(predictor._models) == 3  # Three quantiles
        assert result is predictor  # Returns self

    def test_fit_gradient_boosting(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test fitting with gradient boosting."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="gradient_boosting",
        )
        predictor.fit(sample_features, sample_target, n_estimators=20)

        assert predictor._fitted is True
        assert len(predictor._models) == 3

    def test_fit_empty_dataframe(self) -> None:
        """Test fitting with empty DataFrame."""
        predictor = QuantileRegressionPredictor()
        empty_df = pl.DataFrame({"a": []})
        empty_series = pl.Series([])

        with pytest.raises(ValueError, match="Cannot fit on empty data"):
            predictor.fit(empty_df, empty_series)

    def test_fit_mismatched_lengths(
        self, sample_features: pl.DataFrame
    ) -> None:
        """Test fitting with mismatched X and y lengths."""
        predictor = QuantileRegressionPredictor()
        short_target = pl.Series([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="same length"):
            predictor.fit(sample_features, short_target)

    def test_predict_basic(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test basic prediction generation."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )
        predictor.fit(sample_features, sample_target)

        predictions = predictor.predict(sample_features)

        assert len(predictions) == sample_features.height
        assert all(hasattr(p, "prediction") for p in predictions)
        assert all(hasattr(p, "lower_bound") for p in predictions)
        assert all(hasattr(p, "upper_bound") for p in predictions)
        assert all(hasattr(p, "confidence") for p in predictions)

    def test_predict_without_fit(self, sample_features: pl.DataFrame) -> None:
        """Test prediction before fitting."""
        predictor = QuantileRegressionPredictor()

        with pytest.raises(RuntimeError, match="must be fitted"):
            predictor.predict(sample_features)

    def test_predict_feature_mismatch(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test prediction with wrong features."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )
        predictor.fit(sample_features, sample_target)

        wrong_features = pl.DataFrame({"wrong": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="Feature mismatch"):
            predictor.predict(wrong_features)

    def test_prediction_intervals_ordered(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test that prediction intervals are properly ordered."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )
        predictor.fit(sample_features, sample_target)

        predictions = predictor.predict(sample_features)

        # Check that lower <= prediction <= upper for most predictions
        for pred in predictions:
            assert pred.lower_bound <= pred.upper_bound

    def test_predict_proba(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test predict_proba returns DataFrame with quantiles."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )
        predictor.fit(sample_features, sample_target)

        proba_df = predictor.predict_proba(sample_features)

        assert isinstance(proba_df, pl.DataFrame)
        assert proba_df.height == sample_features.height
        assert "prediction" in proba_df.columns
        assert "lower_bound" in proba_df.columns
        assert "upper_bound" in proba_df.columns
        assert "quantile_0.1" in proba_df.columns
        assert "quantile_0.5" in proba_df.columns
        assert "quantile_0.9" in proba_df.columns

    def test_predict_intervals(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test prediction interval generation."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )
        predictor.fit(sample_features, sample_target)

        intervals_df = predictor.predict_intervals(sample_features, confidence=0.8)

        assert isinstance(intervals_df, pl.DataFrame)
        assert "prediction" in intervals_df.columns
        assert "lower_bound" in intervals_df.columns
        assert "upper_bound" in intervals_df.columns
        assert "confidence" in intervals_df.columns
        assert "interval_width" in intervals_df.columns

    def test_predict_intervals_invalid_confidence(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test predict_intervals with invalid confidence."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )
        predictor.fit(sample_features, sample_target)

        with pytest.raises(ValueError, match="Confidence must be in"):
            predictor.predict_intervals(sample_features, confidence=0.0)

        with pytest.raises(ValueError, match="Confidence must be in"):
            predictor.predict_intervals(sample_features, confidence=1.0)

    def test_calibrate_intervals(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test interval calibration."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )
        predictor.fit(sample_features, sample_target)

        calibration = predictor.calibrate_intervals(sample_features, sample_target)

        assert isinstance(calibration, dict)
        assert len(calibration) > 0
        # Check coverage values are in valid range
        for coverage in calibration.values():
            assert 0.0 <= coverage <= 1.0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_quantile_regressor_default(self) -> None:
        """Test factory function with default config."""
        predictor = create_quantile_regressor()

        assert isinstance(predictor, QuantileRegressionPredictor)
        assert predictor.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]

    def test_create_quantile_regressor_with_config(self) -> None:
        """Test factory function with custom config."""
        config = QuantileRegressionConfig(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )
        predictor = create_quantile_regressor(config)

        assert isinstance(predictor, QuantileRegressionPredictor)
        assert predictor.quantiles == [0.1, 0.5, 0.9]
        assert predictor.base_model == "linear"

    def test_calculate_coverage_basic(self) -> None:
        """Test coverage calculation."""
        y_actual = np.array([100.0, 107.0, 115.0, 109.0])
        lower = np.array([95.0, 98.0, 101.0, 104.0])
        upper = np.array([105.0, 108.0, 111.0, 114.0])

        coverage = calculate_coverage(y_actual, lower, upper)

        assert coverage == 0.75  # 3 out of 4 in interval

    def test_calculate_coverage_perfect(self) -> None:
        """Test coverage with all values in interval."""
        y_actual = np.array([100.0, 98.0, 102.0])
        lower = np.array([95.0, 95.0, 95.0])
        upper = np.array([105.0, 105.0, 105.0])

        coverage = calculate_coverage(y_actual, lower, upper)

        assert coverage == 1.0

    def test_calculate_coverage_none(self) -> None:
        """Test coverage with no values in interval."""
        y_actual = np.array([110.0, 120.0, 115.0])
        lower = np.array([95.0, 95.0, 95.0])
        upper = np.array([100.0, 100.0, 100.0])

        coverage = calculate_coverage(y_actual, lower, upper)

        assert coverage == 0.0

    def test_calculate_coverage_boundary(self) -> None:
        """Test coverage with values on boundary."""
        y_actual = np.array([95.0, 105.0])  # Exactly on bounds
        lower = np.array([95.0, 95.0])
        upper = np.array([105.0, 105.0])

        coverage = calculate_coverage(y_actual, lower, upper)

        assert coverage == 1.0  # Boundary values should count as in interval

    def test_winkler_score_perfect(self) -> None:
        """Test Winkler score with perfect intervals (no violations)."""
        y_actual = np.array([100.0, 103.0, 106.0])
        lower = np.array([95.0, 98.0, 101.0])
        upper = np.array([105.0, 108.0, 111.0])
        alpha = 0.2

        score = winkler_score(y_actual, lower, upper, alpha)

        # Should equal average interval width (10.0)
        assert score == 10.0

    def test_winkler_score_with_violations(self) -> None:
        """Test Winkler score with interval violations."""
        y_actual = np.array([110.0, 90.0])  # Both violate
        lower = np.array([95.0, 98.0])
        upper = np.array([105.0, 108.0])
        alpha = 0.2

        score = winkler_score(y_actual, lower, upper, alpha)

        # Should be higher due to penalties
        assert score > 10.0  # Greater than just interval width

    def test_winkler_score_narrow_intervals(self) -> None:
        """Test that narrower intervals get better scores (if no violations)."""
        y_actual = np.array([100.0, 100.0])
        lower_wide = np.array([90.0, 90.0])
        upper_wide = np.array([110.0, 110.0])
        lower_narrow = np.array([95.0, 95.0])
        upper_narrow = np.array([105.0, 105.0])
        alpha = 0.2

        score_wide = winkler_score(y_actual, lower_wide, upper_wide, alpha)
        score_narrow = winkler_score(y_actual, lower_narrow, upper_narrow, alpha)

        assert score_narrow < score_wide

    def test_winkler_score_below_lower(self) -> None:
        """Test Winkler score when actual is below lower bound."""
        y_actual = np.array([90.0])  # Below lower
        lower = np.array([95.0])
        upper = np.array([105.0])
        alpha = 0.2

        score = winkler_score(y_actual, lower, upper, alpha)

        # Penalty: (2/0.2) * (95 - 90) = 10 * 5 = 50, plus interval width 10 = 60
        assert score == pytest.approx(60.0, rel=0.01)

    def test_winkler_score_above_upper(self) -> None:
        """Test Winkler score when actual is above upper bound."""
        y_actual = np.array([110.0])  # Above upper
        lower = np.array([95.0])
        upper = np.array([105.0])
        alpha = 0.2

        score = winkler_score(y_actual, lower, upper, alpha)

        # Penalty: (2/0.2) * (110 - 105) = 10 * 5 = 50, plus interval width 10 = 60
        assert score == pytest.approx(60.0, rel=0.01)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_linear(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test complete workflow with linear model."""
        # Split data
        train_size = 70
        train_X = sample_features.head(train_size)
        train_y = sample_target.head(train_size)
        test_X = sample_features.tail(sample_features.height - train_size)

        # Create and train model
        predictor = QuantileRegressionPredictor(
            quantiles=[0.05, 0.5, 0.95],
            base_model="linear",
        )
        predictor.fit(train_X, train_y)

        # Generate predictions
        predictions = predictor.predict(test_X)
        assert len(predictions) == test_X.height

        # Generate probability predictions
        proba_df = predictor.predict_proba(test_X)
        assert proba_df.height == test_X.height
        assert "quantile_0.05" in proba_df.columns
        assert "quantile_0.95" in proba_df.columns

    def test_full_workflow_gbm(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test complete workflow with gradient boosting."""
        train_size = 70
        train_X = sample_features.head(train_size)
        train_y = sample_target.head(train_size)
        test_X = sample_features.tail(sample_features.height - train_size)

        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="gradient_boosting",
        )
        predictor.fit(train_X, train_y, n_estimators=20)

        predictions = predictor.predict(test_X)
        assert len(predictions) == test_X.height

    def test_multiple_quantiles(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test with multiple quantiles for detailed uncertainty."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            base_model="linear",
        )
        predictor.fit(sample_features, sample_target)

        proba_df = predictor.predict_proba(sample_features)

        # Check all quantile columns exist
        for q in predictor.quantiles:
            assert f"quantile_{q}" in proba_df.columns

        # Verify ordering: q0.05 <= q0.25 <= q0.5 <= q0.75 <= q0.95
        first_row = proba_df.row(0, named=True)
        tolerance = 1e-6
        assert first_row["quantile_0.05"] <= first_row["quantile_0.25"] + tolerance
        assert first_row["quantile_0.25"] <= first_row["quantile_0.5"] + tolerance
        assert first_row["quantile_0.5"] <= first_row["quantile_0.75"] + tolerance
        assert first_row["quantile_0.75"] <= first_row["quantile_0.95"] + tolerance

    def test_calibration_workflow(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test calibration workflow."""
        train_size = 70
        train_X = sample_features.head(train_size)
        train_y = sample_target.head(train_size)
        val_X = sample_features.tail(sample_features.height - train_size)
        val_y = sample_target.tail(sample_target.len() - train_size)

        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )
        predictor.fit(train_X, train_y)

        # Calibrate
        calibration = predictor.calibrate_intervals(val_X, val_y)

        assert len(calibration) > 0
        for nominal, empirical in calibration.items():
            assert 0.0 <= empirical <= 1.0

    def test_refit_model(
        self, sample_features: pl.DataFrame, sample_target: pl.Series
    ) -> None:
        """Test that model can be refit with new data."""
        predictor = QuantileRegressionPredictor(
            quantiles=[0.1, 0.5, 0.9],
            base_model="linear",
        )

        # First fit
        predictor.fit(sample_features.head(50), sample_target.head(50))
        assert predictor._fitted is True

        # Refit with different data
        predictor.fit(sample_features.tail(50), sample_target.tail(50))
        assert predictor._fitted is True

        # Should work with new fit
        predictions = predictor.predict(sample_features.head(10))
        assert len(predictions) == 10
