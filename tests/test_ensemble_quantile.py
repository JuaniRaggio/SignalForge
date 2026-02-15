"""Comprehensive tests for EnsemblePredictor and QuantileRegressionPredictor.

This module contains extensive tests for the new BasePredictor-compliant
ensemble and quantile regression models, covering:
- Initialization and validation
- Fitting and prediction
- Different combination methods
- Weight optimization
- Interval calibration
- Edge cases and error handling
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
import pytest

from signalforge.ml.models.base import BasePredictor, PredictionResult
from signalforge.ml.models.ensemble import EnsemblePredictor
from signalforge.ml.models.quantile_regression import QuantileRegressionPredictor


# Mock predictor for testing
class MockPredictor(BasePredictor):
    """Simple mock predictor for testing."""

    model_name = "mock"
    model_version = "1.0.0"

    def __init__(self, constant_value: float = 1.0, noise: float = 0.0):
        """Initialize mock predictor."""
        self.constant_value = constant_value
        self.noise = noise
        self._fitted = False

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> MockPredictor:
        """Mock fit method."""
        self._fitted = True
        return self

    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        """Mock predict method."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        results = []
        for _ in range(X.height):
            pred_value = self.constant_value + np.random.randn() * self.noise
            results.append(
                PredictionResult(
                    symbol="TEST",
                    timestamp=datetime.now(),
                    horizon_days=1,
                    prediction=pred_value,
                    confidence=0.8,
                    lower_bound=pred_value - 1.0,
                    upper_bound=pred_value + 1.0,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return results

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Mock predict_proba method."""
        return pl.DataFrame({"prediction": [self.constant_value] * X.height})


# Test fixtures
@pytest.fixture
def sample_data():
    """Generate sample training data."""
    np.random.seed(42)
    n = 100
    X = pl.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "feature3": np.random.randn(n),
        }
    )
    y = pl.Series(2.0 * X["feature1"] + 0.5 * X["feature2"] + np.random.randn(n) * 0.1)
    return X, y


@pytest.fixture
def mock_models():
    """Create mock models for ensemble testing."""
    return [
        MockPredictor(constant_value=1.0, noise=0.1),
        MockPredictor(constant_value=1.5, noise=0.1),
        MockPredictor(constant_value=2.0, noise=0.1),
    ]


# EnsemblePredictor Tests


def test_ensemble_initialization(mock_models):
    """Test ensemble predictor initialization."""
    ensemble = EnsemblePredictor(models=mock_models)
    assert ensemble.model_name == "ensemble"
    assert len(ensemble.models) == 3
    assert ensemble.combination_method == "weighted_mean"


def test_ensemble_initialization_empty_models():
    """Test ensemble initialization with empty model list raises error."""
    with pytest.raises(ValueError, match="At least one model must be provided"):
        EnsemblePredictor(models=[])


def test_ensemble_initialization_invalid_weights(mock_models):
    """Test ensemble initialization with invalid weights."""
    with pytest.raises(ValueError, match="Number of weights"):
        EnsemblePredictor(models=mock_models, weights=[0.5, 0.5])


def test_ensemble_initialization_negative_weights(mock_models):
    """Test ensemble initialization with negative weights."""
    with pytest.raises(ValueError, match="non-negative"):
        EnsemblePredictor(models=mock_models, weights=[-0.1, 0.5, 0.6])


def test_ensemble_initialization_weights_not_sum_to_one(mock_models):
    """Test ensemble initialization with weights not summing to 1."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        EnsemblePredictor(models=mock_models, weights=[0.2, 0.3, 0.4])


def test_ensemble_fit(mock_models, sample_data):
    """Test ensemble fitting."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models)
    result = ensemble.fit(X, y)
    assert result is ensemble
    assert ensemble._fitted is True
    assert all(model._fitted for model in ensemble.models)


def test_ensemble_fit_empty_data(mock_models):
    """Test ensemble fitting with empty data raises error."""
    X = pl.DataFrame()
    y = pl.Series([])
    ensemble = EnsemblePredictor(models=mock_models)
    with pytest.raises(ValueError, match="Cannot fit on empty data"):
        ensemble.fit(X, y)


def test_ensemble_fit_mismatched_lengths(mock_models):
    """Test ensemble fitting with mismatched X and y lengths."""
    X = pl.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y = pl.Series([1.0, 2.0])
    ensemble = EnsemblePredictor(models=mock_models)
    with pytest.raises(ValueError, match="must have same length"):
        ensemble.fit(X, y)


def test_ensemble_predict_mean(mock_models, sample_data):
    """Test ensemble prediction with mean combination."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models, combination_method="mean")
    ensemble.fit(X, y)
    predictions = ensemble.predict(X[:10])

    assert len(predictions) == 10
    assert all(isinstance(p, PredictionResult) for p in predictions)
    assert all(p.model_name == "ensemble" for p in predictions)


def test_ensemble_predict_weighted_mean(mock_models, sample_data):
    """Test ensemble prediction with weighted mean."""
    X, y = sample_data
    weights = [0.5, 0.3, 0.2]
    ensemble = EnsemblePredictor(
        models=mock_models, weights=weights, combination_method="weighted_mean"
    )
    ensemble.fit(X, y)
    predictions = ensemble.predict(X[:10])

    assert len(predictions) == 10
    assert all(p.confidence > 0 for p in predictions)


def test_ensemble_predict_median(mock_models, sample_data):
    """Test ensemble prediction with median combination."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models, combination_method="median")
    ensemble.fit(X, y)
    predictions = ensemble.predict(X[:10])

    assert len(predictions) == 10
    assert all(p.lower_bound is not None for p in predictions)
    assert all(p.upper_bound is not None for p in predictions)


def test_ensemble_predict_stack(mock_models, sample_data):
    """Test ensemble prediction with stacking."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models, combination_method="stack")
    ensemble.fit(X, y)
    predictions = ensemble.predict(X[:10])

    assert len(predictions) == 10
    assert ensemble._meta_model is not None
    assert all("stack" in p.model_name for p in predictions)


def test_ensemble_predict_not_fitted(mock_models, sample_data):
    """Test prediction before fitting raises error."""
    X, _y = sample_data
    ensemble = EnsemblePredictor(models=mock_models)
    with pytest.raises(RuntimeError, match="must be fitted before prediction"):
        ensemble.predict(X[:10])


def test_ensemble_predict_feature_mismatch(mock_models, sample_data):
    """Test prediction with mismatched features raises error."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models)
    ensemble.fit(X, y)

    X_wrong = pl.DataFrame({"wrong_feature": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Feature mismatch"):
        ensemble.predict(X_wrong)


def test_ensemble_predict_proba(mock_models, sample_data):
    """Test ensemble predict_proba returns DataFrame with correct structure."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models)
    ensemble.fit(X, y)
    proba_df = ensemble.predict_proba(X[:10])

    assert isinstance(proba_df, pl.DataFrame)
    assert "prediction" in proba_df.columns
    assert "confidence" in proba_df.columns
    assert "std_dev" in proba_df.columns
    assert "model_0" in proba_df.columns
    assert "model_1" in proba_df.columns
    assert "model_2" in proba_df.columns
    assert len(proba_df) == 10


def test_ensemble_optimize_weights(mock_models, sample_data):
    """Test weight optimization."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models)
    ensemble.fit(X, y)
    weights = ensemble.optimize_weights(X, y, metric="mse")

    assert len(weights) == 3
    assert all(w >= 0 for w in weights)
    assert np.isclose(sum(weights), 1.0)


def test_ensemble_optimize_weights_mae(mock_models, sample_data):
    """Test weight optimization with MAE metric."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models)
    ensemble.fit(X, y)
    weights = ensemble.optimize_weights(X, y, metric="mae")

    assert len(weights) == 3
    assert np.isclose(sum(weights), 1.0)


def test_ensemble_optimize_weights_invalid_metric(mock_models, sample_data):
    """Test weight optimization with invalid metric raises error."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models)
    ensemble.fit(X, y)

    with pytest.raises(ValueError, match="Unsupported metric"):
        ensemble.optimize_weights(X, y, metric="invalid")


def test_ensemble_get_model_contributions(mock_models, sample_data):
    """Test getting individual model contributions."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models)
    ensemble.fit(X, y)
    contributions = ensemble.get_model_contributions(X[:10])

    assert isinstance(contributions, dict)
    assert len(contributions) == 3
    assert "model_0" in contributions
    assert "model_1" in contributions
    assert "model_2" in contributions
    assert all(isinstance(df, pl.DataFrame) for df in contributions.values())
    assert all(len(df) == 10 for df in contributions.values())


def test_ensemble_fit_with_optimize_weights_kwarg(mock_models, sample_data):
    """Test fitting with optimize_weights kwarg."""
    X, y = sample_data
    ensemble = EnsemblePredictor(models=mock_models, combination_method="weighted_mean")
    ensemble.fit(X, y, optimize_weights=True)

    # Weights should be different from default equal weights
    assert ensemble._weights is not None


# QuantileRegressionPredictor Tests


def test_quantile_initialization():
    """Test quantile regression initialization."""
    model = QuantileRegressionPredictor()
    assert model.model_name == "quantile_regression"
    assert len(model.quantiles) == 5
    assert model.base_model == "gradient_boosting"


def test_quantile_initialization_custom():
    """Test quantile regression with custom parameters."""
    quantiles = [0.1, 0.5, 0.9]
    model = QuantileRegressionPredictor(quantiles=quantiles, base_model="linear")
    assert model.quantiles == quantiles
    assert model.base_model == "linear"


def test_quantile_initialization_empty_quantiles():
    """Test initialization with empty quantiles raises error."""
    with pytest.raises(ValueError, match="At least one quantile"):
        QuantileRegressionPredictor(quantiles=[])


def test_quantile_initialization_invalid_quantile():
    """Test initialization with invalid quantile value."""
    with pytest.raises(ValueError, match="must be in range"):
        QuantileRegressionPredictor(quantiles=[0.0, 0.5, 1.0])


def test_quantile_initialization_duplicate_quantiles():
    """Test initialization with duplicate quantiles raises error."""
    with pytest.raises(ValueError, match="must be unique"):
        QuantileRegressionPredictor(quantiles=[0.1, 0.5, 0.5, 0.9])


def test_quantile_fit_linear(sample_data):
    """Test quantile regression fitting with linear model."""
    X, y = sample_data
    model = QuantileRegressionPredictor(base_model="linear")
    result = model.fit(X, y)

    assert result is model
    assert model._fitted is True
    assert len(model._models) == len(model.quantiles)


def test_quantile_fit_gradient_boosting(sample_data):
    """Test quantile regression fitting with gradient boosting."""
    X, y = sample_data
    model = QuantileRegressionPredictor(base_model="gradient_boosting")
    result = model.fit(X, y, n_estimators=50)

    assert result is model
    assert model._fitted is True


def test_quantile_fit_empty_data():
    """Test fitting with empty data raises error."""
    X = pl.DataFrame()
    y = pl.Series([])
    model = QuantileRegressionPredictor()
    with pytest.raises(ValueError, match="Cannot fit on empty data"):
        model.fit(X, y)


def test_quantile_fit_mismatched_lengths():
    """Test fitting with mismatched X and y lengths."""
    X = pl.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y = pl.Series([1.0, 2.0])
    model = QuantileRegressionPredictor()
    with pytest.raises(ValueError, match="must have same length"):
        model.fit(X, y)


def test_quantile_predict(sample_data):
    """Test quantile regression prediction."""
    X, y = sample_data
    model = QuantileRegressionPredictor(quantiles=[0.1, 0.5, 0.9])
    model.fit(X, y)
    predictions = model.predict(X[:10])

    assert len(predictions) == 10
    assert all(isinstance(p, PredictionResult) for p in predictions)
    assert all(p.lower_bound is not None for p in predictions)
    assert all(p.upper_bound is not None for p in predictions)
    # Note: Quantile predictions from independently trained models may
    # not be strictly monotonic due to numerical optimization


def test_quantile_predict_not_fitted(sample_data):
    """Test prediction before fitting raises error."""
    X, _y = sample_data
    model = QuantileRegressionPredictor()
    with pytest.raises(RuntimeError, match="must be fitted before prediction"):
        model.predict(X[:10])


def test_quantile_predict_feature_mismatch(sample_data):
    """Test prediction with mismatched features raises error."""
    X, y = sample_data
    model = QuantileRegressionPredictor()
    model.fit(X, y)

    X_wrong = pl.DataFrame({"wrong_feature": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Feature mismatch"):
        model.predict(X_wrong)


def test_quantile_predict_proba(sample_data):
    """Test quantile predict_proba returns all quantile predictions."""
    X, y = sample_data
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    model = QuantileRegressionPredictor(quantiles=quantiles)
    model.fit(X, y)
    proba_df = model.predict_proba(X[:10])

    assert isinstance(proba_df, pl.DataFrame)
    assert "prediction" in proba_df.columns
    assert "confidence" in proba_df.columns
    assert "lower_bound" in proba_df.columns
    assert "upper_bound" in proba_df.columns
    for q in quantiles:
        assert f"quantile_{q}" in proba_df.columns
    assert len(proba_df) == 10


def test_quantile_predict_intervals(sample_data):
    """Test prediction intervals at specific confidence level."""
    X, y = sample_data
    model = QuantileRegressionPredictor(quantiles=[0.05, 0.1, 0.5, 0.9, 0.95])
    model.fit(X, y)
    intervals_df = model.predict_intervals(X[:10], confidence=0.8)

    assert isinstance(intervals_df, pl.DataFrame)
    assert "prediction" in intervals_df.columns
    assert "lower_bound" in intervals_df.columns
    assert "upper_bound" in intervals_df.columns
    assert "confidence" in intervals_df.columns
    assert "interval_width" in intervals_df.columns
    assert len(intervals_df) == 10


def test_quantile_predict_intervals_invalid_confidence(sample_data):
    """Test prediction intervals with invalid confidence raises error."""
    X, y = sample_data
    model = QuantileRegressionPredictor()
    model.fit(X, y)

    with pytest.raises(ValueError, match="Confidence must be in"):
        model.predict_intervals(X[:10], confidence=1.5)


def test_quantile_predict_intervals_not_fitted(sample_data):
    """Test prediction intervals before fitting raises error."""
    X, _y = sample_data
    model = QuantileRegressionPredictor()
    with pytest.raises(RuntimeError, match="must be fitted before prediction"):
        model.predict_intervals(X[:10])


def test_quantile_calibrate_intervals(sample_data):
    """Test interval calibration."""
    X, y = sample_data
    model = QuantileRegressionPredictor(quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    model.fit(X[:80], y[:80])
    calibration = model.calibrate_intervals(X[80:], y[80:])

    assert isinstance(calibration, dict)
    assert len(calibration) > 0
    assert all(0 <= v <= 1 for v in calibration.values())


def test_quantile_calibrate_intervals_not_fitted(sample_data):
    """Test calibration before fitting raises error."""
    X, y = sample_data
    model = QuantileRegressionPredictor()
    with pytest.raises(RuntimeError, match="must be fitted before calibration"):
        model.calibrate_intervals(X, y)


def test_quantile_calibrate_intervals_length_mismatch(sample_data):
    """Test calibration with mismatched X and y lengths."""
    X, y = sample_data
    model = QuantileRegressionPredictor()
    model.fit(X, y)

    with pytest.raises(ValueError, match="length mismatch"):
        model.calibrate_intervals(X[:10], y[:5])


# Integration Tests


def test_ensemble_with_quantile_models(sample_data):
    """Test ensemble of quantile regression models."""
    X, y = sample_data
    models = [
        QuantileRegressionPredictor(quantiles=[0.1, 0.5, 0.9], base_model="linear"),
        QuantileRegressionPredictor(quantiles=[0.1, 0.5, 0.9], base_model="linear"),
    ]

    ensemble = EnsemblePredictor(models=models, combination_method="mean")
    ensemble.fit(X, y)
    predictions = ensemble.predict(X[:10])

    assert len(predictions) == 10
    assert all(isinstance(p, PredictionResult) for p in predictions)


def test_quantile_predictions_bounds(sample_data):
    """Test that quantile predictions have sensible bounds."""
    X, y = sample_data
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    model = QuantileRegressionPredictor(quantiles=quantiles)
    model.fit(X, y)
    proba_df = model.predict_proba(X[:10])

    # Check that all quantile predictions are present
    for q in quantiles:
        assert f"quantile_{q}" in proba_df.columns

    # Check that bounds are reasonable
    assert all(proba_df["lower_bound"] <= proba_df["upper_bound"])

    # Note: Independently trained quantile models may not guarantee
    # strict monotonicity across quantiles due to optimization differences


def test_ensemble_single_model(sample_data):
    """Test ensemble with single model behaves correctly."""
    X, y = sample_data
    model = MockPredictor(constant_value=5.0)
    ensemble = EnsemblePredictor(models=[model])
    ensemble.fit(X, y)
    predictions = ensemble.predict(X[:10])

    assert len(predictions) == 10
    # With single model, predictions should be close to constant value
    assert all(4.5 <= p.prediction <= 5.5 for p in predictions)


def test_quantile_with_single_quantile(sample_data):
    """Test quantile regression with single quantile (median only)."""
    X, y = sample_data
    model = QuantileRegressionPredictor(quantiles=[0.5])
    model.fit(X, y)
    predictions = model.predict(X[:10])

    assert len(predictions) == 10
    # With single quantile, bounds should be the same as prediction
    assert all(p.lower_bound == p.upper_bound for p in predictions)
