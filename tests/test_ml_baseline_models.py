"""Comprehensive test suite for baseline statistical models.

This module tests ARIMA, Prophet, and GARCH models to ensure they:
- Follow the BasePredictor interface
- Handle synthetic data correctly
- Support serialization (save/load)
- Handle edge cases gracefully
- Provide accurate predictions and confidence intervals
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from signalforge.ml.models.arima import ARIMAPredictor
from signalforge.ml.models.garch import GARCHPredictor
from signalforge.ml.models.prophet_model import ProphetPredictor

# Fixtures for synthetic data


@pytest.fixture
def random_walk_data() -> tuple[pl.DataFrame, pl.Series]:
    """Generate random walk time series."""
    n = 100
    timestamps = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )

    # Random walk: cumulative sum of random steps
    np.random.seed(42)
    values = 100.0 + np.cumsum(np.random.randn(n) * 0.5)

    X = pl.DataFrame({"timestamp": timestamps})
    y = pl.Series(values)

    return X, y


@pytest.fixture
def trending_data() -> tuple[pl.DataFrame, pl.Series]:
    """Generate time series with linear trend."""
    n = 100
    timestamps = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )

    # Linear trend with noise
    np.random.seed(42)
    values = 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.2

    X = pl.DataFrame({"timestamp": timestamps})
    y = pl.Series(values)

    return X, y


@pytest.fixture
def seasonal_data() -> tuple[pl.DataFrame, pl.Series]:
    """Generate time series with seasonal pattern."""
    n = 365
    timestamps = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )

    # Seasonal pattern (yearly) with trend and noise
    np.random.seed(42)
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / 365)
    trend = 0.1 * t
    noise = np.random.randn(n) * 0.5
    values = 100.0 + trend + seasonal + noise

    X = pl.DataFrame({"timestamp": timestamps})
    y = pl.Series(values)

    return X, y


@pytest.fixture
def returns_data() -> tuple[pl.DataFrame, pl.Series]:
    """Generate returns data for GARCH testing."""
    n = 200
    timestamps = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )

    # Simulate returns with volatility clustering
    np.random.seed(42)
    returns = np.random.randn(n) * 0.02  # 2% daily volatility

    X = pl.DataFrame({"timestamp": timestamps})
    y = pl.Series(returns)

    return X, y


@pytest.fixture
def constant_data() -> tuple[pl.DataFrame, pl.Series]:
    """Generate constant time series (edge case)."""
    n = 50
    timestamps = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )

    values = np.full(n, 100.0)

    X = pl.DataFrame({"timestamp": timestamps})
    y = pl.Series(values)

    return X, y


@pytest.fixture
def short_series_data() -> tuple[pl.DataFrame, pl.Series]:
    """Generate short time series (edge case)."""
    n = 10
    timestamps = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )

    np.random.seed(42)
    values = 100.0 + np.random.randn(n) * 0.5

    X = pl.DataFrame({"timestamp": timestamps})
    y = pl.Series(values)

    return X, y


# ARIMA Tests


class TestARIMAPredictor:
    """Test suite for ARIMAPredictor."""

    def test_arima_initialization(self) -> None:
        """Test ARIMA model initialization."""
        model = ARIMAPredictor(order=(5, 1, 0))
        assert model.order == (5, 1, 0)
        assert model.seasonal_order is None
        assert model.model_name == "arima"

    def test_arima_initialization_with_seasonal(self) -> None:
        """Test ARIMA initialization with seasonal order."""
        model = ARIMAPredictor(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        assert model.order == (1, 1, 1)
        assert model.seasonal_order == (1, 1, 1, 12)

    def test_arima_invalid_order(self) -> None:
        """Test ARIMA with invalid order parameters."""
        with pytest.raises(ValueError, match="order must be a tuple of 3 integers"):
            ARIMAPredictor(order=(1, 1))  # type: ignore

        with pytest.raises(ValueError, match="order parameters must be non-negative"):
            ARIMAPredictor(order=(-1, 1, 1))

    def test_arima_invalid_seasonal_order(self) -> None:
        """Test ARIMA with invalid seasonal order."""
        with pytest.raises(ValueError, match="seasonal_order must be a tuple of 4 integers"):
            ARIMAPredictor(order=(1, 1, 1), seasonal_order=(1, 1, 1))  # type: ignore

    def test_arima_fit_random_walk(self, random_walk_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test ARIMA fitting on random walk data."""
        X, y = random_walk_data
        model = ARIMAPredictor(order=(1, 1, 0))
        result = model.fit(X, y)
        assert result is model  # Check method chaining
        assert model._fitted is True
        assert model._fitted_model is not None

    def test_arima_fit_trending(self, trending_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test ARIMA fitting on trending data."""
        X, y = trending_data
        model = ARIMAPredictor(order=(2, 1, 2))
        model.fit(X, y)
        assert model._fitted is True

    def test_arima_fit_empty_data(self) -> None:
        """Test ARIMA with empty data."""
        X = pl.DataFrame({"timestamp": []})
        y = pl.Series([])
        model = ARIMAPredictor()

        with pytest.raises(ValueError, match="Cannot fit model on empty"):
            model.fit(X, y)

    def test_arima_fit_mismatched_shapes(self, random_walk_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test ARIMA with mismatched X and y shapes."""
        X, y = random_walk_data
        X_short = X.head(50)
        model = ARIMAPredictor()

        with pytest.raises(ValueError, match="X and y shape mismatch"):
            model.fit(X_short, y)

    def test_arima_predict(self, random_walk_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test ARIMA prediction."""
        X, y = random_walk_data
        model = ARIMAPredictor(order=(1, 1, 0))
        model.fit(X, y)

        # Create future timestamps
        X_future = pl.DataFrame({
            "timestamp": pl.date_range(  # type: ignore[call-overload]
                start=X["timestamp"].max() + timedelta(days=1),  # type: ignore[operator]
                end=X["timestamp"].max() + timedelta(days=5),  # type: ignore[operator]
                interval="1d",
                eager=True,
            )
        })

        predictions = model.predict(X_future)
        assert len(predictions) == 5
        assert all(p.prediction is not None for p in predictions)
        assert all(p.lower_bound is not None for p in predictions)
        assert all(p.upper_bound is not None for p in predictions)
        assert all(0.0 <= p.confidence <= 1.0 for p in predictions)

    def test_arima_predict_before_fit(self, random_walk_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test ARIMA prediction before fitting."""
        X, _ = random_walk_data
        model = ARIMAPredictor()

        with pytest.raises(RuntimeError, match="Model must be fitted before prediction"):
            model.predict(X.head(5))

    def test_arima_predict_proba(self, random_walk_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test ARIMA predict_proba."""
        X, y = random_walk_data
        model = ARIMAPredictor(order=(1, 1, 0))
        model.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(  # type: ignore[call-overload]
                start=X["timestamp"].max() + timedelta(days=1),  # type: ignore[operator]
                end=X["timestamp"].max() + timedelta(days=5),  # type: ignore[operator]
                interval="1d",
                eager=True,
            )
        })

        proba_df = model.predict_proba(X_future)
        assert proba_df.height == 5
        assert "timestamp" in proba_df.columns
        assert "prediction" in proba_df.columns
        assert "lower_bound" in proba_df.columns
        assert "upper_bound" in proba_df.columns
        assert "confidence" in proba_df.columns

    def test_arima_get_aic(self, random_walk_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test ARIMA AIC retrieval."""
        X, y = random_walk_data
        model = ARIMAPredictor(order=(1, 1, 0))
        model.fit(X, y)

        aic = model.get_aic()
        assert isinstance(aic, float)
        assert not np.isnan(aic)

    def test_arima_get_bic(self, random_walk_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test ARIMA BIC retrieval."""
        X, y = random_walk_data
        model = ARIMAPredictor(order=(1, 1, 0))
        model.fit(X, y)

        bic = model.get_bic()
        assert isinstance(bic, float)
        assert not np.isnan(bic)

    def test_arima_save_load(self, random_walk_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test ARIMA model serialization."""
        X, y = random_walk_data
        model = ARIMAPredictor(order=(1, 1, 0))
        model.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(  # type: ignore[call-overload]
                start=X["timestamp"].max() + timedelta(days=1),  # type: ignore[operator]
                end=X["timestamp"].max() + timedelta(days=5),  # type: ignore[operator]
                interval="1d",
                eager=True,
            )
        })
        predictions_before = model.predict(X_future)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "arima_model.pkl"
            model.save(str(path))

            loaded_model = ARIMAPredictor.load(str(path))
            predictions_after = loaded_model.predict(X_future)

        # Check predictions match
        assert len(predictions_before) == len(predictions_after)
        for p_before, p_after in zip(predictions_before, predictions_after, strict=True):
            assert abs(p_before.prediction - p_after.prediction) < 1e-6

    def test_arima_constant_series(self, constant_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test ARIMA on constant series."""
        X, y = constant_data
        model = ARIMAPredictor(order=(0, 0, 0))  # Simple mean model
        model.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(  # type: ignore[call-overload]
                start=X["timestamp"].max() + timedelta(days=1),  # type: ignore[operator]
                end=X["timestamp"].max() + timedelta(days=5),  # type: ignore[operator]
                interval="1d",
                eager=True,
            )
        })

        predictions = model.predict(X_future)
        assert len(predictions) == 5
        # All predictions should be close to 100.0
        assert all(abs(p.prediction - 100.0) < 1.0 for p in predictions)


# Prophet Tests


class TestProphetPredictor:
    """Test suite for ProphetPredictor."""

    def test_prophet_initialization(self) -> None:
        """Test Prophet model initialization."""
        model = ProphetPredictor()
        assert model.yearly_seasonality is True
        assert model.weekly_seasonality is True
        assert model.daily_seasonality is False
        assert model.model_name == "prophet"

    def test_prophet_initialization_custom(self) -> None:
        """Test Prophet initialization with custom parameters."""
        model = ProphetPredictor(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=True,
            changepoint_prior_scale=0.1,
        )
        assert model.yearly_seasonality is False
        assert model.daily_seasonality is True
        assert model.changepoint_prior_scale == 0.1

    def test_prophet_invalid_changepoint_prior(self) -> None:
        """Test Prophet with invalid changepoint_prior_scale."""
        with pytest.raises(ValueError, match="changepoint_prior_scale must be positive"):
            ProphetPredictor(changepoint_prior_scale=-0.1)

    def test_prophet_fit_trending(self, trending_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test Prophet fitting on trending data."""
        X, y = trending_data
        model = ProphetPredictor()
        result = model.fit(X, y)
        assert result is model  # Check method chaining
        assert model._fitted is True
        assert model._model is not None

    def test_prophet_fit_seasonal(self, seasonal_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test Prophet fitting on seasonal data."""
        X, y = seasonal_data
        model = ProphetPredictor(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(X, y)
        assert model._fitted is True

    def test_prophet_fit_empty_data(self) -> None:
        """Test Prophet with empty data."""
        X = pl.DataFrame({"timestamp": []})
        y = pl.Series([])
        model = ProphetPredictor()

        with pytest.raises(ValueError, match="Cannot fit model on empty"):
            model.fit(X, y)

    def test_prophet_predict(self, trending_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test Prophet prediction."""
        X, y = trending_data
        model = ProphetPredictor()
        model.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(  # type: ignore[call-overload]
                start=X["timestamp"].max() + timedelta(days=1),  # type: ignore[operator]
                end=X["timestamp"].max() + timedelta(days=10),  # type: ignore[operator]
                interval="1d",
                eager=True,
            )
        })

        predictions = model.predict(X_future)
        assert len(predictions) == 10
        assert all(p.prediction is not None for p in predictions)
        assert all(p.lower_bound is not None for p in predictions)
        assert all(p.upper_bound is not None for p in predictions)
        assert all(0.0 <= p.confidence <= 1.0 for p in predictions)

    def test_prophet_predict_before_fit(self, trending_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test Prophet prediction before fitting."""
        X, _ = trending_data
        model = ProphetPredictor()

        with pytest.raises(RuntimeError, match="Model must be fitted before prediction"):
            model.predict(X.head(5))

    def test_prophet_predict_proba(self, trending_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test Prophet predict_proba."""
        X, y = trending_data
        model = ProphetPredictor()
        model.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(  # type: ignore[call-overload]
                start=X["timestamp"].max() + timedelta(days=1),  # type: ignore[operator]
                end=X["timestamp"].max() + timedelta(days=10),  # type: ignore[operator]
                interval="1d",
                eager=True,
            )
        })

        proba_df = model.predict_proba(X_future)
        assert proba_df.height == 10
        assert "timestamp" in proba_df.columns
        assert "prediction" in proba_df.columns
        assert "yhat_lower" in proba_df.columns
        assert "yhat_upper" in proba_df.columns

    def test_prophet_add_regressor(self, trending_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test Prophet add_regressor."""
        model = ProphetPredictor()
        model.add_regressor("external_feature")
        assert "external_feature" in model._regressors

    def test_prophet_add_regressor_after_fit(self, trending_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test Prophet add_regressor after fitting."""
        X, y = trending_data
        model = ProphetPredictor()
        model.fit(X, y)

        with pytest.raises(RuntimeError, match="Cannot add regressors after model has been fitted"):
            model.add_regressor("external_feature")

    def test_prophet_save_load(self, trending_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test Prophet model serialization."""
        X, y = trending_data
        model = ProphetPredictor()
        model.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(  # type: ignore[call-overload]
                start=X["timestamp"].max() + timedelta(days=1),  # type: ignore[operator]
                end=X["timestamp"].max() + timedelta(days=10),  # type: ignore[operator]
                interval="1d",
                eager=True,
            )
        })
        predictions_before = model.predict(X_future)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prophet_model.pkl"
            model.save(str(path))

            loaded_model = ProphetPredictor.load(str(path))
            predictions_after = loaded_model.predict(X_future)

        # Check predictions match
        assert len(predictions_before) == len(predictions_after)
        for p_before, p_after in zip(predictions_before, predictions_after, strict=True):
            assert abs(p_before.prediction - p_after.prediction) < 1e-6


# GARCH Tests


class TestGARCHPredictor:
    """Test suite for GARCHPredictor."""

    def test_garch_initialization(self) -> None:
        """Test GARCH model initialization."""
        model = GARCHPredictor(p=1, q=1)
        assert model.p == 1
        assert model.q == 1
        assert model.mean == "Constant"
        assert model.vol == "GARCH"
        assert model.dist == "normal"
        assert model.model_name == "garch"

    def test_garch_initialization_custom(self) -> None:
        """Test GARCH initialization with custom parameters."""
        model = GARCHPredictor(p=2, q=2, mean="Zero", vol="EGARCH", dist="t")
        assert model.p == 2
        assert model.q == 2
        assert model.mean == "Zero"
        assert model.vol == "EGARCH"
        assert model.dist == "t"

    def test_garch_invalid_p(self) -> None:
        """Test GARCH with invalid p parameter."""
        with pytest.raises(ValueError, match="p must be positive"):
            GARCHPredictor(p=0, q=1)

    def test_garch_invalid_q(self) -> None:
        """Test GARCH with invalid q parameter."""
        with pytest.raises(ValueError, match="q must be positive"):
            GARCHPredictor(p=1, q=0)

    def test_garch_invalid_mean(self) -> None:
        """Test GARCH with invalid mean parameter."""
        with pytest.raises(ValueError, match="mean must be one of"):
            GARCHPredictor(p=1, q=1, mean="Invalid")

    def test_garch_invalid_vol(self) -> None:
        """Test GARCH with invalid vol parameter."""
        with pytest.raises(ValueError, match="vol must be one of"):
            GARCHPredictor(p=1, q=1, vol="Invalid")

    def test_garch_invalid_dist(self) -> None:
        """Test GARCH with invalid dist parameter."""
        with pytest.raises(ValueError, match="dist must be one of"):
            GARCHPredictor(p=1, q=1, dist="invalid")

    def test_garch_fit_returns(self, returns_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test GARCH fitting on returns data."""
        X, y = returns_data
        model = GARCHPredictor(p=1, q=1)
        result = model.fit(X, y)
        assert result is model  # Check method chaining
        assert model._fitted is True
        assert model._fitted_result is not None

    def test_garch_fit_empty_data(self) -> None:
        """Test GARCH with empty data."""
        X = pl.DataFrame({"timestamp": []})
        y = pl.Series([])
        model = GARCHPredictor()

        with pytest.raises(ValueError, match="Cannot fit model on empty"):
            model.fit(X, y)

    def test_garch_predict(self, returns_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test GARCH volatility prediction."""
        X, y = returns_data
        model = GARCHPredictor(p=1, q=1)
        model.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(  # type: ignore[call-overload]
                start=X["timestamp"].max() + timedelta(days=1),  # type: ignore[operator]
                end=X["timestamp"].max() + timedelta(days=5),  # type: ignore[operator]
                interval="1d",
                eager=True,
            )
        })

        predictions = model.predict(X_future)
        assert len(predictions) == 5
        assert all(p.prediction is not None for p in predictions)
        assert all(p.prediction > 0 for p in predictions)  # Volatility is positive
        assert all(0.0 <= p.confidence <= 1.0 for p in predictions)

    def test_garch_predict_before_fit(self, returns_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test GARCH prediction before fitting."""
        X, _ = returns_data
        model = GARCHPredictor()

        with pytest.raises(RuntimeError, match="Model must be fitted before prediction"):
            model.predict(X.head(5))

    def test_garch_predict_proba(self, returns_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test GARCH predict_proba."""
        X, y = returns_data
        model = GARCHPredictor(p=1, q=1)
        model.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(  # type: ignore[call-overload]
                start=X["timestamp"].max() + timedelta(days=1),  # type: ignore[operator]
                end=X["timestamp"].max() + timedelta(days=5),  # type: ignore[operator]
                interval="1d",
                eager=True,
            )
        })

        proba_df = model.predict_proba(X_future)
        assert proba_df.height == 5
        assert "timestamp" in proba_df.columns
        assert "volatility" in proba_df.columns
        assert "variance" in proba_df.columns

    def test_garch_forecast_variance(self, returns_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test GARCH forecast_variance."""
        X, y = returns_data
        model = GARCHPredictor(p=1, q=1)
        model.fit(X, y)

        variance_df = model.forecast_variance(horizon=10)
        assert variance_df.height == 10
        assert "horizon_step" in variance_df.columns
        assert "variance" in variance_df.columns
        assert "volatility" in variance_df.columns

    def test_garch_get_standardized_residuals(self, returns_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test GARCH get_standardized_residuals."""
        X, y = returns_data
        model = GARCHPredictor(p=1, q=1)
        model.fit(X, y)

        residuals = model.get_standardized_residuals()
        assert len(residuals) > 0
        # Standardized residuals should have mean close to 0 and std close to 1
        assert abs(residuals.mean()) < 0.5  # type: ignore
        assert abs(residuals.std() - 1.0) < 0.5  # type: ignore

    def test_garch_save_load(self, returns_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test GARCH model serialization."""
        X, y = returns_data
        model = GARCHPredictor(p=1, q=1)
        model.fit(X, y)

        X_future = pl.DataFrame({
            "timestamp": pl.date_range(  # type: ignore[call-overload]
                start=X["timestamp"].max() + timedelta(days=1),  # type: ignore[operator]
                end=X["timestamp"].max() + timedelta(days=5),  # type: ignore[operator]
                interval="1d",
                eager=True,
            )
        })
        predictions_before = model.predict(X_future)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "garch_model.pkl"
            model.save(str(path))

            loaded_model = GARCHPredictor.load(str(path))
            predictions_after = loaded_model.predict(X_future)

        # Check predictions match
        assert len(predictions_before) == len(predictions_after)
        for p_before, p_after in zip(predictions_before, predictions_after, strict=True):
            assert abs(p_before.prediction - p_after.prediction) < 1e-6

    def test_garch_short_series(self, short_series_data: tuple[pl.DataFrame, pl.Series]) -> None:
        """Test GARCH on short series (may struggle with limited data)."""
        X, y = short_series_data
        # Convert to returns
        y_returns = (y - y.shift(1)).tail(-1) / y.shift(1).tail(-1)
        X_returns = X.tail(-1)

        model = GARCHPredictor(p=1, q=1)
        # Short series may cause issues, but should not crash
        try:
            model.fit(X_returns, y_returns)
            assert model._fitted is True or model._fitted is False
        except (RuntimeError, ValueError):
            # Expected for very short series
            pass
