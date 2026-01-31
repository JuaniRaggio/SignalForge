"""Tests for ensemble models.

This module tests the ensemble framework including:
- EnsembleConfig validation
- WeightedEnsemble with different combination methods
- StackingEnsemble with meta-learner
- Weight optimization
- Integration with base models
- Edge cases and error handling
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import polars as pl
import pytest

from signalforge.ml.models.base import BasePredictor
from signalforge.ml.models.baseline import ARIMAPredictor, RollingMeanPredictor
from signalforge.ml.models.ensemble import (
    EnsembleConfig,
    EnsemblePrediction,
    StackingEnsemble,
    WeightedEnsemble,
    create_ensemble,
    optimize_weights,
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
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    values = [100.0 + i * 0.5 for i in range(30)]

    return pl.DataFrame(
        {
            "timestamp": dates,
            "close": values,
        }
    )


class MockPredictor(BasePredictor):
    """Mock predictor for testing ensemble logic without real models."""

    def __init__(self, prediction_value: float = 100.0) -> None:
        """Initialize mock predictor.

        Args:
            prediction_value: Constant value to return for predictions.
        """
        self.prediction_value = prediction_value
        self._fitted = False
        self._training_data: pl.DataFrame | None = None

    def fit(self, df: pl.DataFrame, target_column: str = "close") -> None:
        """Mock fit method."""
        if df.height == 0:
            raise ValueError("Cannot fit on empty DataFrame")
        self._training_data = df
        self._fitted = True

    def predict(self, horizon: int) -> pl.DataFrame:
        """Mock predict method."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(horizon)]
        return pl.DataFrame(
            {
                "timestamp": dates,
                "prediction": [self.prediction_value] * horizon,
            }
        )

    def evaluate(self, test_df: pl.DataFrame) -> dict[str, float]:
        """Mock evaluate method."""
        return {"rmse": 1.0, "mae": 1.0, "mape": 1.0, "direction_accuracy": 50.0}

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._fitted


class TestEnsembleConfig:
    """Tests for EnsembleConfig dataclass."""

    def test_config_basic(self) -> None:
        """Test basic configuration creation."""
        config = EnsembleConfig(
            models=["model1", "model2"],
            method="weighted",
        )

        assert config.models == ["model1", "model2"]
        assert config.method == "weighted"
        assert config.optimize_weights is True
        assert config.weights is None

    def test_config_with_custom_weights(self) -> None:
        """Test configuration with custom weights."""
        config = EnsembleConfig(
            models=["model1", "model2"],
            weights=[0.6, 0.4],
            method="weighted",
            optimize_weights=False,
        )

        assert config.weights == [0.6, 0.4]
        assert config.optimize_weights is False

    def test_config_weight_length_mismatch(self) -> None:
        """Test that mismatched weights length raises error."""
        with pytest.raises(ValueError, match="Number of weights.*must match"):
            EnsembleConfig(
                models=["model1", "model2"],
                weights=[0.5],
            )

    def test_config_negative_weights(self) -> None:
        """Test that negative weights raise error."""
        with pytest.raises(ValueError, match="All weights must be non-negative"):
            EnsembleConfig(
                models=["model1", "model2"],
                weights=[0.6, -0.4],
            )

    def test_config_zero_sum_weights(self) -> None:
        """Test that weights summing to zero raise error."""
        with pytest.raises(ValueError, match="Sum of weights must be greater than zero"):
            EnsembleConfig(
                models=["model1", "model2"],
                weights=[0.0, 0.0],
            )


class TestEnsemblePrediction:
    """Tests for EnsemblePrediction dataclass."""

    def test_prediction_basic(self) -> None:
        """Test basic prediction result creation."""
        pred = EnsemblePrediction(
            prediction=105.5,
            model_predictions={"model1": 105.0, "model2": 106.0},
            weights_used={"model1": 0.5, "model2": 0.5},
        )

        assert pred.prediction == 105.5
        assert pred.model_predictions == {"model1": 105.0, "model2": 106.0}
        assert pred.weights_used == {"model1": 0.5, "model2": 0.5}
        assert pred.confidence_interval is None

    def test_prediction_with_confidence_interval(self) -> None:
        """Test prediction with confidence interval."""
        pred = EnsemblePrediction(
            prediction=105.5,
            model_predictions={"model1": 105.0, "model2": 106.0},
            weights_used={"model1": 0.5, "model2": 0.5},
            confidence_interval=(103.0, 108.0),
        )

        assert pred.confidence_interval == (103.0, 108.0)


class TestWeightedEnsemble:
    """Tests for WeightedEnsemble class."""

    def test_ensemble_initialization(self) -> None:
        """Test ensemble initialization."""
        config = EnsembleConfig(models=["model1", "model2"])
        ensemble = WeightedEnsemble(config)

        assert ensemble.config == config
        assert len(ensemble._models) == 0
        assert ensemble.is_fitted is False

    def test_add_model(self) -> None:
        """Test adding models to ensemble."""
        config = EnsembleConfig(models=["model1", "model2"])
        ensemble = WeightedEnsemble(config)

        model1 = MockPredictor(100.0)
        model2 = MockPredictor(110.0)

        ensemble.add_model("model1", model1)
        ensemble.add_model("model2", model2)

        assert len(ensemble._models) == 2
        assert "model1" in ensemble._models
        assert "model2" in ensemble._models

    def test_add_model_not_in_config(self) -> None:
        """Test that adding model not in config raises error."""
        config = EnsembleConfig(models=["model1"])
        ensemble = WeightedEnsemble(config)

        with pytest.raises(ValueError, match="not found in configuration"):
            ensemble.add_model("model2", MockPredictor())

    def test_add_duplicate_model(self) -> None:
        """Test that adding duplicate model raises error."""
        config = EnsembleConfig(models=["model1"])
        ensemble = WeightedEnsemble(config)

        ensemble.add_model("model1", MockPredictor())

        with pytest.raises(ValueError, match="already exists"):
            ensemble.add_model("model1", MockPredictor())

    def test_fit_with_equal_weights(self, small_time_series: pl.DataFrame) -> None:
        """Test fitting with equal weights."""
        config = EnsembleConfig(
            models=["model1", "model2"],
            method="mean",
            optimize_weights=False,
        )
        ensemble = WeightedEnsemble(config)

        ensemble.add_model("model1", MockPredictor(100.0))
        ensemble.add_model("model2", MockPredictor(110.0))

        ensemble.fit(small_time_series, target_column="close")

        assert ensemble.is_fitted
        weights = ensemble.get_weights()
        assert weights["model1"] == 0.5
        assert weights["model2"] == 0.5

    def test_fit_with_custom_weights(self, small_time_series: pl.DataFrame) -> None:
        """Test fitting with custom weights."""
        config = EnsembleConfig(
            models=["model1", "model2"],
            weights=[0.7, 0.3],
            method="weighted",
            optimize_weights=False,
        )
        ensemble = WeightedEnsemble(config)

        ensemble.add_model("model1", MockPredictor(100.0))
        ensemble.add_model("model2", MockPredictor(110.0))

        ensemble.fit(small_time_series, target_column="close")

        assert ensemble.is_fitted
        weights = ensemble.get_weights()
        assert weights["model1"] == 0.7
        assert weights["model2"] == 0.3

    def test_fit_missing_models(self, small_time_series: pl.DataFrame) -> None:
        """Test that fitting with missing models raises error."""
        config = EnsembleConfig(models=["model1", "model2"])
        ensemble = WeightedEnsemble(config)

        ensemble.add_model("model1", MockPredictor(100.0))
        # Don't add model2

        with pytest.raises(RuntimeError, match="Missing models"):
            ensemble.fit(small_time_series, target_column="close")

    def test_fit_empty_dataframe(self) -> None:
        """Test that fitting on empty DataFrame raises error."""
        config = EnsembleConfig(models=["model1"])
        ensemble = WeightedEnsemble(config)
        ensemble.add_model("model1", MockPredictor())

        empty_df = pl.DataFrame({"timestamp": [], "close": []})

        with pytest.raises(ValueError, match="empty DataFrame"):
            ensemble.fit(empty_df, target_column="close")

    def test_predict_mean_method(self, small_time_series: pl.DataFrame) -> None:
        """Test prediction with mean method."""
        config = EnsembleConfig(models=["model1", "model2"], method="mean")
        ensemble = WeightedEnsemble(config)

        ensemble.add_model("model1", MockPredictor(100.0))
        ensemble.add_model("model2", MockPredictor(110.0))

        ensemble.fit(small_time_series, target_column="close")
        result = ensemble.predict(horizon=5)

        assert isinstance(result, EnsemblePrediction)
        assert result.prediction == 105.0  # Mean of 100 and 110
        assert result.model_predictions["model1"] == 100.0
        assert result.model_predictions["model2"] == 110.0
        assert result.confidence_interval is not None

    def test_predict_weighted_method(self, small_time_series: pl.DataFrame) -> None:
        """Test prediction with weighted method."""
        config = EnsembleConfig(
            models=["model1", "model2"],
            weights=[0.7, 0.3],
            method="weighted",
            optimize_weights=False,
        )
        ensemble = WeightedEnsemble(config)

        ensemble.add_model("model1", MockPredictor(100.0))
        ensemble.add_model("model2", MockPredictor(110.0))

        ensemble.fit(small_time_series, target_column="close")
        result = ensemble.predict(horizon=5)

        expected = 0.7 * 100.0 + 0.3 * 110.0  # 103.0
        assert result.prediction == pytest.approx(expected)
        assert result.weights_used["model1"] == 0.7
        assert result.weights_used["model2"] == 0.3

    def test_predict_median_method(self, small_time_series: pl.DataFrame) -> None:
        """Test prediction with median method."""
        config = EnsembleConfig(
            models=["model1", "model2", "model3"],
            method="median",
        )
        ensemble = WeightedEnsemble(config)

        ensemble.add_model("model1", MockPredictor(100.0))
        ensemble.add_model("model2", MockPredictor(110.0))
        ensemble.add_model("model3", MockPredictor(105.0))

        ensemble.fit(small_time_series, target_column="close")
        result = ensemble.predict(horizon=5)

        assert result.prediction == 105.0  # Median of [100, 110, 105]

    def test_predict_not_fitted(self) -> None:
        """Test that predicting without fitting raises error."""
        config = EnsembleConfig(models=["model1"])
        ensemble = WeightedEnsemble(config)

        with pytest.raises(RuntimeError, match="must be fitted"):
            ensemble.predict(horizon=5)

    def test_predict_invalid_horizon(self, small_time_series: pl.DataFrame) -> None:
        """Test that invalid horizon raises error."""
        config = EnsembleConfig(models=["model1"])
        ensemble = WeightedEnsemble(config)
        ensemble.add_model("model1", MockPredictor())
        ensemble.fit(small_time_series)

        with pytest.raises(ValueError, match="horizon must be positive"):
            ensemble.predict(horizon=0)

    def test_get_weights_not_fitted(self) -> None:
        """Test that getting weights before fitting raises error."""
        config = EnsembleConfig(models=["model1"])
        ensemble = WeightedEnsemble(config)

        with pytest.raises(RuntimeError, match="must be fitted"):
            ensemble.get_weights()


class TestStackingEnsemble:
    """Tests for StackingEnsemble class."""

    def test_ensemble_initialization(self) -> None:
        """Test stacking ensemble initialization."""
        config = EnsembleConfig(models=["model1", "model2"], method="stacking")
        ensemble = StackingEnsemble(config)

        assert ensemble.config == config
        assert len(ensemble._models) == 0
        assert ensemble.is_fitted is False
        assert ensemble._meta_model is None

    def test_add_model(self) -> None:
        """Test adding models to stacking ensemble."""
        config = EnsembleConfig(models=["model1", "model2"], method="stacking")
        ensemble = StackingEnsemble(config)

        model1 = MockPredictor(100.0)
        model2 = MockPredictor(110.0)

        ensemble.add_model("model1", model1)
        ensemble.add_model("model2", model2)

        assert len(ensemble._models) == 2

    def test_fit_insufficient_data(self) -> None:
        """Test that fitting with insufficient data raises error."""
        config = EnsembleConfig(models=["model1"], method="stacking")
        ensemble = StackingEnsemble(config)
        ensemble.add_model("model1", MockPredictor())

        small_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)],
                "close": [100.0] * 10,
            }
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            ensemble.fit(small_df)

    def test_fit_and_predict(self, small_time_series: pl.DataFrame) -> None:
        """Test fitting and predicting with stacking ensemble."""
        config = EnsembleConfig(models=["model1", "model2"], method="stacking")
        ensemble = StackingEnsemble(config)

        ensemble.add_model("model1", MockPredictor(100.0))
        ensemble.add_model("model2", MockPredictor(110.0))

        ensemble.fit(small_time_series, target_column="close")

        assert ensemble.is_fitted
        assert ensemble._meta_model is not None

        result = ensemble.predict(horizon=5)

        assert isinstance(result, EnsemblePrediction)
        assert isinstance(result.prediction, float)
        assert "model1" in result.model_predictions
        assert "model2" in result.model_predictions
        assert "model1" in result.weights_used
        assert "model2" in result.weights_used

    def test_get_weights(self, small_time_series: pl.DataFrame) -> None:
        """Test getting meta-model weights."""
        config = EnsembleConfig(models=["model1", "model2"], method="stacking")
        ensemble = StackingEnsemble(config)

        ensemble.add_model("model1", MockPredictor(100.0))
        ensemble.add_model("model2", MockPredictor(110.0))

        ensemble.fit(small_time_series, target_column="close")
        weights = ensemble.get_weights()

        assert isinstance(weights, dict)
        assert "model1" in weights
        assert "model2" in weights
        assert all(isinstance(w, float) for w in weights.values())

    def test_predict_not_fitted(self) -> None:
        """Test that predicting without fitting raises error."""
        config = EnsembleConfig(models=["model1"], method="stacking")
        ensemble = StackingEnsemble(config)

        with pytest.raises(RuntimeError, match="must be fitted"):
            ensemble.predict(horizon=5)


class TestCreateEnsemble:
    """Tests for create_ensemble factory function."""

    def test_create_weighted_ensemble(self) -> None:
        """Test creating weighted ensemble."""
        config = EnsembleConfig(models=["model1"], method="weighted")
        ensemble = create_ensemble(config)

        assert isinstance(ensemble, WeightedEnsemble)

    def test_create_mean_ensemble(self) -> None:
        """Test creating mean ensemble."""
        config = EnsembleConfig(models=["model1"], method="mean")
        ensemble = create_ensemble(config)

        assert isinstance(ensemble, WeightedEnsemble)

    def test_create_median_ensemble(self) -> None:
        """Test creating median ensemble."""
        config = EnsembleConfig(models=["model1"], method="median")
        ensemble = create_ensemble(config)

        assert isinstance(ensemble, WeightedEnsemble)

    def test_create_stacking_ensemble(self) -> None:
        """Test creating stacking ensemble."""
        config = EnsembleConfig(models=["model1"], method="stacking")
        ensemble = create_ensemble(config)

        assert isinstance(ensemble, StackingEnsemble)

    def test_create_unsupported_method(self) -> None:
        """Test that unsupported method raises error."""
        config = EnsembleConfig(models=["model1"])
        config.method = "unsupported"  # type: ignore

        with pytest.raises(ValueError, match="Unsupported ensemble method"):
            create_ensemble(config)


class TestOptimizeWeights:
    """Tests for optimize_weights function."""

    def test_optimize_weights_basic(self, small_time_series: pl.DataFrame) -> None:
        """Test basic weight optimization."""
        model1 = MockPredictor(100.0)
        model2 = MockPredictor(110.0)

        model1.fit(small_time_series)
        model2.fit(small_time_series)

        models = {"model1": model1, "model2": model2}

        weights = optimize_weights(models, small_time_series, "close")

        assert isinstance(weights, dict)
        assert "model1" in weights
        assert "model2" in weights
        assert all(0.0 <= w <= 1.0 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_optimize_weights_unfitted_model(self, small_time_series: pl.DataFrame) -> None:
        """Test that optimization with unfitted model falls back to equal weights."""
        model1 = MockPredictor(100.0)
        model2 = MockPredictor(110.0)

        # Only fit model1
        model1.fit(small_time_series)

        models = {"model1": model1, "model2": model2}

        # Should fall back to equal weights instead of raising
        weights = optimize_weights(models, small_time_series, "close")

        # Verify fallback to equal weights
        assert weights["model1"] == 0.5
        assert weights["model2"] == 0.5

    @patch("scipy.optimize.minimize")
    def test_optimize_weights_optimization_failure(
        self,
        mock_minimize: Mock,
        small_time_series: pl.DataFrame,
    ) -> None:
        """Test fallback to equal weights when optimization fails."""
        # Mock optimization failure
        mock_result = Mock()
        mock_result.success = False
        mock_result.message = "Optimization failed"
        mock_minimize.return_value = mock_result

        model1 = MockPredictor(100.0)
        model2 = MockPredictor(110.0)

        model1.fit(small_time_series)
        model2.fit(small_time_series)

        models = {"model1": model1, "model2": model2}

        weights = optimize_weights(models, small_time_series, "close")

        # Should fall back to equal weights
        assert weights["model1"] == 0.5
        assert weights["model2"] == 0.5


class TestIntegrationWithRealModels:
    """Integration tests with real baseline models."""

    def test_ensemble_with_arima_and_rolling_mean(self, sample_time_series: pl.DataFrame) -> None:
        """Test ensemble with ARIMA and RollingMean models."""
        config = EnsembleConfig(
            models=["arima", "rolling_mean"],
            method="weighted",
            optimize_weights=False,
            weights=[0.5, 0.5],
        )
        ensemble = WeightedEnsemble(config)

        # Add real models
        arima = ARIMAPredictor(order=(1, 1, 1))
        rolling_mean = RollingMeanPredictor(window=20)

        ensemble.add_model("arima", arima)
        ensemble.add_model("rolling_mean", rolling_mean)

        # Fit and predict
        ensemble.fit(sample_time_series, target_column="close")
        result = ensemble.predict(horizon=10)

        assert isinstance(result, EnsemblePrediction)
        assert result.prediction > 0
        assert "arima" in result.model_predictions
        assert "rolling_mean" in result.model_predictions
        assert result.confidence_interval is not None

    def test_stacking_ensemble_with_real_models(self, sample_time_series: pl.DataFrame) -> None:
        """Test stacking ensemble with real baseline models."""
        config = EnsembleConfig(
            models=["arima", "rolling_mean"],
            method="stacking",
        )
        ensemble = StackingEnsemble(config)

        arima = ARIMAPredictor(order=(1, 1, 1))
        rolling_mean = RollingMeanPredictor(window=20)

        ensemble.add_model("arima", arima)
        ensemble.add_model("rolling_mean", rolling_mean)

        ensemble.fit(sample_time_series, target_column="close")
        result = ensemble.predict(horizon=10)

        assert isinstance(result, EnsemblePrediction)
        assert result.prediction > 0
        assert ensemble._meta_model is not None


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_model_ensemble(self, small_time_series: pl.DataFrame) -> None:
        """Test ensemble with single model."""
        config = EnsembleConfig(models=["model1"], method="mean")
        ensemble = WeightedEnsemble(config)

        ensemble.add_model("model1", MockPredictor(100.0))
        ensemble.fit(small_time_series)

        result = ensemble.predict(horizon=5)
        assert result.prediction == 100.0

    def test_ensemble_with_many_models(self, small_time_series: pl.DataFrame) -> None:
        """Test ensemble with many models."""
        n_models = 10
        model_names = [f"model{i}" for i in range(n_models)]
        config = EnsembleConfig(models=model_names, method="mean")
        ensemble = WeightedEnsemble(config)

        for i, name in enumerate(model_names):
            ensemble.add_model(name, MockPredictor(100.0 + i * 10))

        ensemble.fit(small_time_series)
        result = ensemble.predict(horizon=5)

        # Mean of 100, 110, 120, ..., 190
        expected_mean = sum(100.0 + i * 10 for i in range(n_models)) / n_models
        assert result.prediction == pytest.approx(expected_mean)

    def test_mlflow_integration(self, small_time_series: pl.DataFrame) -> None:
        """Test that MLflow logging doesn't break ensemble."""
        config = EnsembleConfig(models=["model1"], method="weighted")
        ensemble = WeightedEnsemble(config)

        ensemble.add_model("model1", MockPredictor(100.0))

        # Should not raise even without active MLflow run
        ensemble.fit(small_time_series)
        result = ensemble.predict(horizon=5)

        assert result.prediction == 100.0
