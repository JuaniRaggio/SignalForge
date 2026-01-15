"""Tests for baseline statistical models.

This module tests the baseline prediction models including ARIMA and
RollingMean predictors. Tests cover initialization, fitting, prediction,
evaluation, and edge cases.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from signalforge.ml.models.base import BasePredictor
from signalforge.ml.models.baseline import (
    ARIMAPredictor,
    RollingMeanPredictor,
    calculate_metrics,
)


@pytest.fixture
def sample_time_series() -> pl.DataFrame:
    """Create sample time series data for testing.

    Returns a DataFrame with 100 rows of synthetic time series data
    with a slight upward trend.
    """
    n_rows = 100
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_rows)]

    # Create data with trend and some noise
    base_value = 100.0
    values = [base_value + i * 0.5 + (i % 5) * 0.2 for i in range(n_rows)]

    return pl.DataFrame(
        {
            "timestamp": dates,
            "close": values,
            "volume": [1000000] * n_rows,
        }
    )


@pytest.fixture
def small_time_series() -> pl.DataFrame:
    """Create small time series for edge case testing."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    values = [100.0 + i for i in range(10)]

    return pl.DataFrame(
        {
            "timestamp": dates,
            "close": values,
        }
    )


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_calculate_metrics_basic(self) -> None:
        """Test basic metrics calculation."""
        actual = pl.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        predicted = pl.Series([100.5, 100.8, 102.2, 103.1, 103.9])

        metrics = calculate_metrics(actual, predicted)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "direction_accuracy" in metrics

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert metrics["mape"] > 0
        assert 0 <= metrics["direction_accuracy"] <= 100

    def test_calculate_metrics_perfect_prediction(self) -> None:
        """Test metrics with perfect predictions."""
        actual = pl.Series([100.0, 101.0, 102.0, 103.0])
        predicted = pl.Series([100.0, 101.0, 102.0, 103.0])

        metrics = calculate_metrics(actual, predicted)

        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["mape"] == 0.0
        assert metrics["direction_accuracy"] == 100.0

    def test_calculate_metrics_with_nulls(self) -> None:
        """Test metrics calculation with null values."""
        actual = pl.Series([100.0, None, 102.0, 103.0])
        predicted = pl.Series([100.5, 101.0, None, 103.1])

        metrics = calculate_metrics(actual, predicted)

        # Should only use non-null pairs
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

    def test_calculate_metrics_length_mismatch(self) -> None:
        """Test error when series lengths don't match."""
        actual = pl.Series([100.0, 101.0, 102.0])
        predicted = pl.Series([100.0, 101.0])

        with pytest.raises(ValueError, match="Series length mismatch"):
            calculate_metrics(actual, predicted)

    def test_calculate_metrics_empty_series(self) -> None:
        """Test error with empty series."""
        actual = pl.Series([])
        predicted = pl.Series([])

        with pytest.raises(ValueError, match="Cannot calculate metrics for empty series"):
            calculate_metrics(actual, predicted)

    def test_calculate_metrics_all_nulls(self) -> None:
        """Test error when all values are null."""
        actual = pl.Series([None, None, None])
        predicted = pl.Series([None, None, None])

        with pytest.raises(ValueError, match="No valid"):
            calculate_metrics(actual, predicted)

    def test_direction_accuracy(self) -> None:
        """Test direction accuracy calculation."""
        # All directions correct
        actual = pl.Series([100.0, 101.0, 102.0, 103.0])
        predicted = pl.Series([100.0, 100.5, 101.5, 102.5])

        metrics = calculate_metrics(actual, predicted)
        assert metrics["direction_accuracy"] == 100.0

        # All directions wrong
        actual = pl.Series([100.0, 101.0, 102.0, 103.0])
        predicted = pl.Series([104.0, 103.0, 102.0, 101.0])

        metrics = calculate_metrics(actual, predicted)
        assert metrics["direction_accuracy"] == 0.0


class TestARIMAPredictor:
    """Tests for ARIMAPredictor class."""

    def test_initialization_default(self) -> None:
        """Test ARIMA predictor initialization with defaults."""
        model = ARIMAPredictor()

        assert model.order == (1, 1, 1)
        assert model.seasonal_order is None
        assert not model.is_fitted

    def test_initialization_custom_order(self) -> None:
        """Test ARIMA predictor initialization with custom order."""
        model = ARIMAPredictor(order=(2, 1, 2))

        assert model.order == (2, 1, 2)
        assert not model.is_fitted

    def test_initialization_with_seasonal(self) -> None:
        """Test ARIMA predictor initialization with seasonal order."""
        model = ARIMAPredictor(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
        )

        assert model.order == (1, 1, 1)
        assert model.seasonal_order == (1, 1, 1, 12)

    def test_initialization_invalid_order(self) -> None:
        """Test error with invalid order."""
        with pytest.raises(ValueError, match="order must be a tuple of 3 integers"):
            ARIMAPredictor(order=(1, 1))  # type: ignore

        with pytest.raises(ValueError, match="order parameters must be non-negative"):
            ARIMAPredictor(order=(-1, 1, 1))

    def test_initialization_invalid_seasonal(self) -> None:
        """Test error with invalid seasonal order."""
        with pytest.raises(ValueError, match="seasonal_order must be a tuple of 4 integers"):
            ARIMAPredictor(seasonal_order=(1, 1, 1))  # type: ignore

    def test_isinstance_base_predictor(self) -> None:
        """Test that ARIMA predictor is instance of BasePredictor."""
        model = ARIMAPredictor()
        assert isinstance(model, BasePredictor)

    @patch("signalforge.ml.models.baseline.log_params")
    def test_fit_basic(self, mock_log_params: MagicMock, sample_time_series: pl.DataFrame) -> None:
        """Test fitting ARIMA model."""
        model = ARIMAPredictor(order=(1, 0, 0))
        model.fit(sample_time_series, target_column="close")

        assert model.is_fitted
        assert model._training_data is not None
        assert model._target_column == "close"
        assert model._fitted_model is not None

    def test_fit_empty_dataframe(self) -> None:
        """Test error when fitting with empty DataFrame."""
        model = ARIMAPredictor()
        empty_df = pl.DataFrame({"timestamp": [], "close": []})

        with pytest.raises(ValueError, match="Cannot fit model on empty DataFrame"):
            model.fit(empty_df)

    def test_fit_missing_target_column(self, sample_time_series: pl.DataFrame) -> None:
        """Test error when target column is missing."""
        model = ARIMAPredictor()

        with pytest.raises(ValueError, match="Target column 'price' not found"):
            model.fit(sample_time_series, target_column="price")

    def test_fit_missing_timestamp_column(self) -> None:
        """Test error when timestamp column is missing."""
        model = ARIMAPredictor()
        df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})

        with pytest.raises(ValueError, match="DataFrame must contain 'timestamp' column"):
            model.fit(df)

    @patch("signalforge.ml.models.baseline.log_params")
    def test_predict_basic(
        self, mock_log_params: MagicMock, sample_time_series: pl.DataFrame
    ) -> None:
        """Test generating predictions."""
        model = ARIMAPredictor(order=(1, 0, 0))
        model.fit(sample_time_series, target_column="close")

        predictions = model.predict(horizon=10)

        assert predictions.height == 10
        assert "timestamp" in predictions.columns
        assert "prediction" in predictions.columns
        assert "lower_ci" in predictions.columns
        assert "upper_ci" in predictions.columns

        # Check that timestamps are in the future
        last_training_timestamp = sample_time_series["timestamp"].max()
        first_prediction_timestamp = predictions["timestamp"][0]
        assert first_prediction_timestamp > last_training_timestamp

    def test_predict_not_fitted(self) -> None:
        """Test error when predicting without fitting."""
        model = ARIMAPredictor()

        with pytest.raises(RuntimeError, match="Model must be fitted before prediction"):
            model.predict(horizon=5)

    def test_predict_invalid_horizon(self, sample_time_series: pl.DataFrame) -> None:
        """Test error with invalid horizon."""
        model = ARIMAPredictor(order=(1, 0, 0))
        model.fit(sample_time_series)

        with pytest.raises(ValueError, match="horizon must be positive"):
            model.predict(horizon=0)

        with pytest.raises(ValueError, match="horizon must be positive"):
            model.predict(horizon=-5)

    @patch("signalforge.ml.models.baseline.log_params")
    @patch("signalforge.ml.models.baseline.log_metrics")
    def test_evaluate_basic(
        self,
        mock_log_metrics: MagicMock,
        mock_log_params: MagicMock,
        sample_time_series: pl.DataFrame,
    ) -> None:
        """Test evaluating model on test data."""
        # Split data into train and test
        train_df = sample_time_series.head(80)
        test_df = sample_time_series.tail(20)

        model = ARIMAPredictor(order=(1, 0, 0))
        model.fit(train_df, target_column="close")

        metrics = model.evaluate(test_df)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "direction_accuracy" in metrics

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

    def test_evaluate_not_fitted(self, sample_time_series: pl.DataFrame) -> None:
        """Test error when evaluating without fitting."""
        model = ARIMAPredictor()

        with pytest.raises(RuntimeError, match="Model must be fitted before evaluation"):
            model.evaluate(sample_time_series)

    def test_evaluate_empty_dataframe(self, sample_time_series: pl.DataFrame) -> None:
        """Test error when evaluating with empty DataFrame."""
        model = ARIMAPredictor(order=(1, 0, 0))
        model.fit(sample_time_series)

        empty_df = pl.DataFrame({"timestamp": [], "close": []})

        with pytest.raises(ValueError, match="Cannot evaluate on empty DataFrame"):
            model.evaluate(empty_df)

    def test_evaluate_missing_target_column(self, sample_time_series: pl.DataFrame) -> None:
        """Test error when target column missing in test data."""
        model = ARIMAPredictor(order=(1, 0, 0))
        model.fit(sample_time_series, target_column="close")

        test_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 5, 1)],
                "price": [150.0],
            }
        )

        with pytest.raises(ValueError, match="Target column 'close' not found"):
            model.evaluate(test_df)


class TestRollingMeanPredictor:
    """Tests for RollingMeanPredictor class."""

    def test_initialization_default(self) -> None:
        """Test rolling mean predictor initialization with defaults."""
        model = RollingMeanPredictor()

        assert model.window == 20
        assert not model.is_fitted

    def test_initialization_custom_window(self) -> None:
        """Test rolling mean predictor initialization with custom window."""
        model = RollingMeanPredictor(window=10)

        assert model.window == 10
        assert not model.is_fitted

    def test_initialization_invalid_window(self) -> None:
        """Test error with invalid window."""
        with pytest.raises(ValueError, match="window must be positive"):
            RollingMeanPredictor(window=0)

        with pytest.raises(ValueError, match="window must be positive"):
            RollingMeanPredictor(window=-5)

    def test_isinstance_base_predictor(self) -> None:
        """Test that rolling mean predictor is instance of BasePredictor."""
        model = RollingMeanPredictor()
        assert isinstance(model, BasePredictor)

    @patch("signalforge.ml.models.baseline.log_params")
    def test_fit_basic(self, mock_log_params: MagicMock, sample_time_series: pl.DataFrame) -> None:
        """Test fitting rolling mean model."""
        model = RollingMeanPredictor(window=10)
        model.fit(sample_time_series, target_column="close")

        assert model.is_fitted
        assert model._training_data is not None
        assert model._target_column == "close"

    def test_fit_empty_dataframe(self) -> None:
        """Test error when fitting with empty DataFrame."""
        model = RollingMeanPredictor()
        empty_df = pl.DataFrame({"timestamp": [], "close": []})

        with pytest.raises(ValueError, match="Cannot fit model on empty DataFrame"):
            model.fit(empty_df)

    def test_fit_insufficient_data(self, small_time_series: pl.DataFrame) -> None:
        """Test error when data is smaller than window."""
        model = RollingMeanPredictor(window=20)

        with pytest.raises(ValueError, match="Insufficient data"):
            model.fit(small_time_series)

    def test_fit_missing_target_column(self, sample_time_series: pl.DataFrame) -> None:
        """Test error when target column is missing."""
        model = RollingMeanPredictor()

        with pytest.raises(ValueError, match="Target column 'price' not found"):
            model.fit(sample_time_series, target_column="price")

    def test_fit_missing_timestamp_column(self) -> None:
        """Test error when timestamp column is missing."""
        model = RollingMeanPredictor()
        df = pl.DataFrame({"close": [100.0 + i for i in range(30)]})

        with pytest.raises(ValueError, match="DataFrame must contain 'timestamp' column"):
            model.fit(df)

    @patch("signalforge.ml.models.baseline.log_params")
    def test_predict_basic(
        self, mock_log_params: MagicMock, sample_time_series: pl.DataFrame
    ) -> None:
        """Test generating predictions."""
        model = RollingMeanPredictor(window=10)
        model.fit(sample_time_series, target_column="close")

        predictions = model.predict(horizon=5)

        assert predictions.height == 5
        assert "timestamp" in predictions.columns
        assert "prediction" in predictions.columns

        # All predictions should be the same (rolling mean)
        unique_predictions = predictions["prediction"].n_unique()
        assert unique_predictions == 1

        # Check that timestamps are in the future
        last_training_timestamp = sample_time_series["timestamp"].max()
        first_prediction_timestamp = predictions["timestamp"][0]
        assert first_prediction_timestamp > last_training_timestamp

    def test_predict_not_fitted(self) -> None:
        """Test error when predicting without fitting."""
        model = RollingMeanPredictor()

        with pytest.raises(RuntimeError, match="Model must be fitted before prediction"):
            model.predict(horizon=5)

    def test_predict_invalid_horizon(self, sample_time_series: pl.DataFrame) -> None:
        """Test error with invalid horizon."""
        model = RollingMeanPredictor(window=10)
        model.fit(sample_time_series)

        with pytest.raises(ValueError, match="horizon must be positive"):
            model.predict(horizon=0)

        with pytest.raises(ValueError, match="horizon must be positive"):
            model.predict(horizon=-5)

    @patch("signalforge.ml.models.baseline.log_params")
    @patch("signalforge.ml.models.baseline.log_metrics")
    def test_evaluate_basic(
        self,
        mock_log_metrics: MagicMock,
        mock_log_params: MagicMock,
        sample_time_series: pl.DataFrame,
    ) -> None:
        """Test evaluating model on test data."""
        # Split data into train and test
        train_df = sample_time_series.head(80)
        test_df = sample_time_series.tail(20)

        model = RollingMeanPredictor(window=10)
        model.fit(train_df, target_column="close")

        metrics = model.evaluate(test_df)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "direction_accuracy" in metrics

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

    def test_evaluate_not_fitted(self, sample_time_series: pl.DataFrame) -> None:
        """Test error when evaluating without fitting."""
        model = RollingMeanPredictor()

        with pytest.raises(RuntimeError, match="Model must be fitted before evaluation"):
            model.evaluate(sample_time_series)

    def test_evaluate_empty_dataframe(self, sample_time_series: pl.DataFrame) -> None:
        """Test error when evaluating with empty DataFrame."""
        model = RollingMeanPredictor(window=10)
        model.fit(sample_time_series)

        empty_df = pl.DataFrame({"timestamp": [], "close": []})

        with pytest.raises(ValueError, match="Cannot evaluate on empty DataFrame"):
            model.evaluate(empty_df)

    def test_evaluate_missing_target_column(self, sample_time_series: pl.DataFrame) -> None:
        """Test error when target column missing in test data."""
        model = RollingMeanPredictor(window=10)
        model.fit(sample_time_series, target_column="close")

        test_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 5, 1)],
                "price": [150.0],
            }
        )

        with pytest.raises(ValueError, match="Target column 'close' not found"):
            model.evaluate(test_df)

    @patch("signalforge.ml.models.baseline.log_params")
    def test_prediction_value_correctness(
        self, mock_log_params: MagicMock, sample_time_series: pl.DataFrame
    ) -> None:
        """Test that prediction value is correct rolling mean."""
        window = 10
        model = RollingMeanPredictor(window=window)
        model.fit(sample_time_series, target_column="close")

        predictions = model.predict(horizon=3)

        # Calculate expected mean manually
        expected_mean = float(sample_time_series["close"].tail(window).mean())
        actual_prediction = float(predictions["prediction"][0])

        # Should be approximately equal (floating point tolerance)
        assert abs(actual_prediction - expected_mean) < 1e-6


class TestModelComparison:
    """Tests comparing ARIMA and RollingMean models."""

    @patch("signalforge.ml.models.baseline.log_params")
    @patch("signalforge.ml.models.baseline.log_metrics")
    def test_both_models_can_fit_and_predict(
        self,
        mock_log_metrics: MagicMock,
        mock_log_params: MagicMock,
        sample_time_series: pl.DataFrame,
    ) -> None:
        """Test that both models can fit and predict on same data."""
        arima_model = ARIMAPredictor(order=(1, 0, 0))
        rolling_model = RollingMeanPredictor(window=10)

        # Both should fit without error
        arima_model.fit(sample_time_series)
        rolling_model.fit(sample_time_series)

        # Both should predict without error
        arima_predictions = arima_model.predict(horizon=5)
        rolling_predictions = rolling_model.predict(horizon=5)

        assert arima_predictions.height == 5
        assert rolling_predictions.height == 5

    @patch("signalforge.ml.models.baseline.log_params")
    @patch("signalforge.ml.models.baseline.log_metrics")
    def test_both_models_produce_different_predictions(
        self,
        mock_log_metrics: MagicMock,
        mock_log_params: MagicMock,
        sample_time_series: pl.DataFrame,
    ) -> None:
        """Test that ARIMA and RollingMean produce different predictions."""
        arima_model = ARIMAPredictor(order=(1, 1, 1))
        rolling_model = RollingMeanPredictor(window=10)

        arima_model.fit(sample_time_series)
        rolling_model.fit(sample_time_series)

        arima_predictions = arima_model.predict(horizon=5)
        rolling_predictions = rolling_model.predict(horizon=5)

        # Predictions should be different (models use different approaches)
        arima_values = arima_predictions["prediction"].to_list()
        rolling_values = rolling_predictions["prediction"].to_list()

        # At least one prediction should differ
        assert arima_values != rolling_values
