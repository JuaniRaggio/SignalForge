"""Tests for ensemble models.

This module tests the ensemble framework including:
- EnsembleConfig validation
- EnsemblePredictor with different combination methods
- WeightedEnsemble and StackingEnsemble
- Weight optimization
- Integration with base models
- Edge cases and error handling
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from signalforge.ml.models.baseline import ARIMAPredictor, RollingMeanPredictor
from signalforge.ml.models.ensemble import (
    EnsembleConfig,
    EnsemblePrediction,
    EnsemblePredictor,
    StackingEnsemble,
    WeightedEnsemble,
    create_ensemble,
    optimize_weights,
)
from signalforge.ml.models.base import PredictionResult


class MockModel:
    """Mock model that supports sklearn-like fit(X, y) API for testing."""

    def __init__(self, name: str = "mock", offset: float = 0.0) -> None:
        self.name = name
        self.offset = offset
        self._fitted = False
        self._last_y_value: float = 0.0

    def fit(self, X: pl.DataFrame, y: pl.Series) -> None:
        """Fit the mock model."""
        self._fitted = True
        # Store the last y value for simple predictions
        if isinstance(y, pl.Series):
            self._last_y_value = float(y[-1])
        else:
            self._last_y_value = 100.0

    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        """Generate predictions."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        results = []
        for i in range(X.height):
            pred_value = self._last_y_value + self.offset + i * 0.1
            # Get timestamp from X if available, otherwise use current time
            if "timestamp" in X.columns:
                ts = X["timestamp"][i]
            else:
                ts = datetime.now()
            results.append(
                PredictionResult(
                    symbol="TEST",
                    timestamp=ts,
                    horizon_days=1,
                    prediction=pred_value,
                    confidence=0.8,
                    lower_bound=pred_value - 1.0,
                    upper_bound=pred_value + 1.0,
                    model_name=self.name,
                    model_version="1.0.0",
                )
            )
        return results

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Generate probability predictions."""
        predictions = self.predict(X)
        return pl.DataFrame({
            "prediction": [p.prediction for p in predictions],
            "confidence": [p.confidence for p in predictions],
        })


@pytest.fixture
def sample_time_series() -> tuple[pl.DataFrame, pl.Series]:
    """Create sample time series data for testing.

    Returns X (features DataFrame) and y (target Series).
    The ensemble expects sklearn-like fit(X, y) API.
    """
    n_rows = 100
    dates = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(days=n_rows - 1),
        interval="1d",
        eager=True,
    )

    # Create data with trend and some noise
    np.random.seed(42)
    values = 100.0 + np.arange(n_rows) * 0.5 + np.random.randn(n_rows) * 0.2

    X = pl.DataFrame({"timestamp": dates})
    y = pl.Series("target", values)

    return X, y


@pytest.fixture
def small_time_series() -> tuple[pl.DataFrame, pl.Series]:
    """Create small time series for edge case testing.

    Returns X (features DataFrame) and y (target Series).
    """
    n_rows = 30
    dates = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(days=n_rows - 1),
        interval="1d",
        eager=True,
    )
    values = 100.0 + np.arange(n_rows) * 0.5

    X = pl.DataFrame({"timestamp": dates})
    y = pl.Series("target", values)

    return X, y


class TestEnsembleConfig:
    """Tests for EnsembleConfig Pydantic model."""

    def test_config_default(self) -> None:
        """Test default configuration creation."""
        config = EnsembleConfig()

        assert config.combination_method == "weighted_mean"
        assert config.weights is None
        assert config.optimize_weights is False
        assert config.optimization_metric == "mse"

    def test_config_with_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = EnsembleConfig(
            combination_method="mean",
            weights=[0.6, 0.4],
            optimize_weights=True,
            optimization_metric="mae",
        )

        assert config.combination_method == "mean"
        assert config.weights == [0.6, 0.4]
        assert config.optimize_weights is True
        assert config.optimization_metric == "mae"

    def test_config_stack_method(self) -> None:
        """Test stacking configuration."""
        config = EnsembleConfig(combination_method="stack")

        assert config.combination_method == "stack"


class TestEnsemblePrediction:
    """Tests for EnsemblePrediction dataclass."""

    def test_prediction_basic(self) -> None:
        """Test basic prediction result creation."""
        pred = EnsemblePrediction(
            prediction=105.5,
            confidence=0.8,
            lower_bound=103.0,
            upper_bound=108.0,
        )

        assert pred.prediction == 105.5
        assert pred.confidence == 0.8
        assert pred.lower_bound == 103.0
        assert pred.upper_bound == 108.0
        assert pred.model_predictions == {}
        assert pred.weights_used == []

    def test_prediction_with_model_details(self) -> None:
        """Test prediction with model contribution details."""
        pred = EnsemblePrediction(
            prediction=105.5,
            confidence=0.8,
            lower_bound=103.0,
            upper_bound=108.0,
            model_predictions={"model1": 105.0, "model2": 106.0},
            weights_used=[0.5, 0.5],
        )

        assert pred.model_predictions == {"model1": 105.0, "model2": 106.0}
        assert pred.weights_used == [0.5, 0.5]


class TestEnsemblePredictor:
    """Tests for EnsemblePredictor class."""

    def test_ensemble_initialization(
        self, small_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test ensemble initialization."""
        X, y = small_time_series
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        ensemble = EnsemblePredictor([model1, model2])

        assert len(ensemble.models) == 2
        assert ensemble.combination_method == "weighted_mean"
        assert ensemble._weights == [0.5, 0.5]

    def test_ensemble_with_custom_weights(self) -> None:
        """Test ensemble with custom weights."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        ensemble = EnsemblePredictor([model1, model2], weights=[0.7, 0.3])

        assert ensemble._weights == [0.7, 0.3]

    def test_ensemble_empty_models_raises(self) -> None:
        """Test that empty models list raises error."""
        with pytest.raises(ValueError, match="At least one model"):
            EnsemblePredictor([])

    def test_ensemble_weight_mismatch_raises(self) -> None:
        """Test that mismatched weights raise error."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        with pytest.raises(ValueError, match="Number of weights.*must match"):
            EnsemblePredictor([model1, model2], weights=[0.5])

    def test_ensemble_negative_weights_raises(self) -> None:
        """Test that negative weights raise error."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        with pytest.raises(ValueError, match="non-negative"):
            EnsemblePredictor([model1, model2], weights=[0.7, -0.3])

    def test_ensemble_weights_not_sum_one_raises(self) -> None:
        """Test that weights not summing to 1 raise error."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        with pytest.raises(ValueError, match="must sum to 1.0"):
            EnsemblePredictor([model1, model2], weights=[0.6, 0.6])

    def test_ensemble_fit_and_predict(
        self, small_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test fitting and predicting with ensemble."""
        X, y = small_time_series
        model1 = MockModel("model1")
        model2 = MockModel("model2", offset=1.0)

        ensemble = EnsemblePredictor([model1, model2])
        ensemble.fit(X, y)

        assert ensemble._fitted

        # Create future timestamps
        X_future = pl.DataFrame({
            "timestamp": pl.date_range(
                start=X["timestamp"].max() + timedelta(days=1),
                end=X["timestamp"].max() + timedelta(days=5),
                interval="1d",
                eager=True,
            )
        })

        predictions = ensemble.predict(X_future)
        assert len(predictions) == 5

    def test_ensemble_predict_before_fit_raises(self) -> None:
        """Test that predicting before fitting raises error."""
        model1 = MockModel("model1")
        ensemble = EnsemblePredictor([model1])

        X_future = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)]})

        with pytest.raises(RuntimeError, match="must be fitted"):
            ensemble.predict(X_future)

    def test_ensemble_mean_method(
        self, small_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test ensemble with mean combination method."""
        X, y = small_time_series
        model1 = MockModel("model1")
        model2 = MockModel("model2", offset=1.0)

        ensemble = EnsemblePredictor([model1, model2], combination_method="mean")
        ensemble.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(
                start=X["timestamp"].max() + timedelta(days=1),
                end=X["timestamp"].max() + timedelta(days=5),
                interval="1d",
                eager=True,
            )
        })

        predictions = ensemble.predict(X_future)
        assert len(predictions) == 5

    def test_ensemble_median_method(
        self, small_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test ensemble with median combination method."""
        X, y = small_time_series
        model1 = MockModel("model1")
        model2 = MockModel("model2", offset=1.0)
        model3 = MockModel("model3", offset=2.0)

        ensemble = EnsemblePredictor(
            [model1, model2, model3], combination_method="median"
        )
        ensemble.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(
                start=X["timestamp"].max() + timedelta(days=1),
                end=X["timestamp"].max() + timedelta(days=5),
                interval="1d",
                eager=True,
            )
        })

        predictions = ensemble.predict(X_future)
        assert len(predictions) == 5


class TestWeightedEnsemble:
    """Tests for WeightedEnsemble class."""

    def test_weighted_ensemble_initialization(self) -> None:
        """Test weighted ensemble initialization."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        ensemble = WeightedEnsemble([model1, model2])

        assert ensemble.model_name == "weighted_ensemble"
        assert ensemble.combination_method == "weighted_mean"

    def test_weighted_ensemble_with_custom_weights(self) -> None:
        """Test weighted ensemble with custom weights."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        ensemble = WeightedEnsemble([model1, model2], weights=[0.7, 0.3])

        assert ensemble._weights == [0.7, 0.3]

    def test_weighted_ensemble_fit_predict(
        self, small_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test fitting and predicting with weighted ensemble."""
        X, y = small_time_series
        model1 = MockModel("model1")
        model2 = MockModel("model2", offset=1.0)

        ensemble = WeightedEnsemble([model1, model2], weights=[0.6, 0.4])
        ensemble.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(
                start=X["timestamp"].max() + timedelta(days=1),
                end=X["timestamp"].max() + timedelta(days=5),
                interval="1d",
                eager=True,
            )
        })

        predictions = ensemble.predict(X_future)
        assert len(predictions) == 5


class TestStackingEnsemble:
    """Tests for StackingEnsemble class."""

    def test_stacking_ensemble_initialization(self) -> None:
        """Test stacking ensemble initialization."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        ensemble = StackingEnsemble([model1, model2])

        assert ensemble.model_name == "stacking_ensemble"
        assert ensemble.combination_method == "stack"

    def test_stacking_ensemble_fit_predict(
        self, sample_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test fitting and predicting with stacking ensemble."""
        X, y = sample_time_series
        model1 = MockModel("model1")
        model2 = MockModel("model2", offset=1.0)

        ensemble = StackingEnsemble([model1, model2])
        ensemble.fit(X, y)

        assert ensemble._fitted
        assert ensemble._meta_model is not None

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(
                start=X["timestamp"].max() + timedelta(days=1),
                end=X["timestamp"].max() + timedelta(days=5),
                interval="1d",
                eager=True,
            )
        })

        predictions = ensemble.predict(X_future)
        assert len(predictions) == 5


class TestCreateEnsemble:
    """Tests for create_ensemble factory function."""

    def test_create_weighted_ensemble(self) -> None:
        """Test creating weighted ensemble."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        config = EnsembleConfig(combination_method="weighted_mean")
        ensemble = create_ensemble([model1, model2], config)

        assert isinstance(ensemble, WeightedEnsemble)

    def test_create_stacking_ensemble(self) -> None:
        """Test creating stacking ensemble."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        config = EnsembleConfig(combination_method="stack")
        ensemble = create_ensemble([model1, model2], config)

        assert isinstance(ensemble, StackingEnsemble)

    def test_create_ensemble_default_config(self) -> None:
        """Test creating ensemble with default config."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        ensemble = create_ensemble([model1, model2])

        assert isinstance(ensemble, WeightedEnsemble)

    def test_create_ensemble_with_weights(self) -> None:
        """Test creating ensemble with weights in config."""
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        config = EnsembleConfig(weights=[0.7, 0.3])
        ensemble = create_ensemble([model1, model2], config)

        assert ensemble._weights == [0.7, 0.3]


class TestOptimizeWeights:
    """Tests for optimize_weights function."""

    def test_optimize_weights_basic(
        self, sample_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test basic weight optimization."""
        X, y = sample_time_series

        model1 = MockModel("model1")
        model2 = MockModel("model2", offset=1.0)

        # Fit models first
        model1.fit(X, y)
        model2.fit(X, y)

        weights = optimize_weights([model1, model2], X, y)

        assert len(weights) == 2
        assert all(0.0 <= w <= 1.0 for w in weights)
        assert sum(weights) == pytest.approx(1.0)

    def test_optimize_weights_mae_metric(
        self, sample_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test weight optimization with MAE metric."""
        X, y = sample_time_series

        model1 = MockModel("model1")
        model2 = MockModel("model2", offset=1.0)

        model1.fit(X, y)
        model2.fit(X, y)

        weights = optimize_weights([model1, model2], X, y, metric="mae")

        assert len(weights) == 2
        assert sum(weights) == pytest.approx(1.0)


class TestIntegrationWithMockModels:
    """Integration tests with mock models."""

    def test_ensemble_with_multiple_mock_models(
        self, sample_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test ensemble with multiple mock models."""
        X, y = sample_time_series

        model1 = MockModel("model1")
        model2 = MockModel("model2", offset=1.0)
        model3 = MockModel("model3", offset=2.0)

        ensemble = WeightedEnsemble([model1, model2, model3])
        ensemble.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(
                start=X["timestamp"].max() + timedelta(days=1),
                end=X["timestamp"].max() + timedelta(days=10),
                interval="1d",
                eager=True,
            )
        })

        predictions = ensemble.predict(X_future)
        assert len(predictions) == 10
        assert all(p.prediction > 0 for p in predictions)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_model_ensemble(
        self, small_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test ensemble with single model."""
        X, y = small_time_series
        model = MockModel("model1")

        ensemble = EnsemblePredictor([model])
        ensemble.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(
                start=X["timestamp"].max() + timedelta(days=1),
                end=X["timestamp"].max() + timedelta(days=5),
                interval="1d",
                eager=True,
            )
        })

        predictions = ensemble.predict(X_future)
        assert len(predictions) == 5

    def test_ensemble_predict_proba(
        self, small_time_series: tuple[pl.DataFrame, pl.Series]
    ) -> None:
        """Test ensemble predict_proba method."""
        X, y = small_time_series
        model1 = MockModel("model1")
        model2 = MockModel("model2", offset=1.0)

        ensemble = EnsemblePredictor([model1, model2])
        ensemble.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(
                start=X["timestamp"].max() + timedelta(days=1),
                end=X["timestamp"].max() + timedelta(days=5),
                interval="1d",
                eager=True,
            )
        })

        proba_df = ensemble.predict_proba(X_future)

        assert proba_df.height == 5
        assert "prediction" in proba_df.columns
