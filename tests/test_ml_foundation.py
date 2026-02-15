"""Comprehensive tests for ML foundation modules.

This module tests:
- BasePredictor interface and PredictionResult
- WalkForwardValidator with purging and embargo
- ValidationResult aggregation
- MLflowTracker wrapper

Tests ensure proper temporal ordering, data leakage prevention,
and correct metric calculation for time series models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest

from signalforge.ml.models.base import BasePredictor, PredictionResult
from signalforge.ml.training.mlflow_utils import MLflowTracker
from signalforge.ml.training.validation import ValidationResult, WalkForwardValidator


# Mock predictor for testing
class MockPredictor(BasePredictor):
    """Mock predictor for testing purposes."""

    model_name = "MockPredictor"
    model_version = "1.0.0"

    def __init__(self) -> None:
        self._fitted = False
        self._train_mean = 0.0

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> BasePredictor:
        """Fit by storing the mean of y."""
        self._fitted = True
        self._train_mean = float(y.mean())  # type: ignore[arg-type]
        return self

    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        """Predict by returning the training mean for each row."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        results = []
        for _i in range(len(X)):
            results.append(
                PredictionResult(
                    symbol="TEST",
                    timestamp=datetime.now(),
                    horizon_days=1,
                    prediction=self._train_mean,
                    confidence=0.5,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return results

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return confidence scores."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        return pl.DataFrame({"confidence": [0.5] * len(X)})


# Test fixtures
@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create a sample time series DataFrame."""
    dates = pl.date_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 12, 31), interval="1d", eager=True
    )
    n = len(dates)
    return pl.DataFrame(
        {
            "timestamp": dates,
            "feature1": list(range(n)),
            "feature2": [float(i * 2) for i in range(n)],
            "value": [100.0 + i * 0.5 for i in range(n)],
        }
    )


@pytest.fixture
def small_dataframe() -> pl.DataFrame:
    """Create a small DataFrame for specific tests."""
    dates = pl.date_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 31), interval="1d", eager=True)
    n = len(dates)
    return pl.DataFrame(
        {
            "timestamp": dates,
            "feature1": list(range(n)),
            "value": [float(i) for i in range(n)],
        }
    )


@pytest.fixture
def mock_model() -> MockPredictor:
    """Create a mock predictor instance."""
    return MockPredictor()


# Test PredictionResult
class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_prediction_result_creation(self) -> None:
        """Test creating a PredictionResult with all fields."""
        result = PredictionResult(
            symbol="AAPL",
            timestamp=datetime(2020, 1, 1),
            horizon_days=5,
            prediction=150.25,
            confidence=0.85,
            lower_bound=148.0,
            upper_bound=152.5,
            model_name="TestModel",
            model_version="1.0",
        )

        assert result.symbol == "AAPL"
        assert result.timestamp == datetime(2020, 1, 1)
        assert result.horizon_days == 5
        assert result.prediction == 150.25
        assert result.confidence == 0.85
        assert result.lower_bound == 148.0
        assert result.upper_bound == 152.5
        assert result.model_name == "TestModel"
        assert result.model_version == "1.0"

    def test_prediction_result_optional_fields(self) -> None:
        """Test PredictionResult with optional fields as None."""
        result = PredictionResult(
            symbol="AAPL",
            timestamp=datetime(2020, 1, 1),
            horizon_days=1,
            prediction=150.0,
            confidence=0.8,
        )

        assert result.lower_bound is None
        assert result.upper_bound is None
        assert result.model_name == ""
        assert result.model_version == ""


# Test BasePredictor interface
class TestBasePredictor:
    """Tests for BasePredictor abstract interface."""

    def test_mock_predictor_fit(self, small_dataframe: pl.DataFrame) -> None:
        """Test that mock predictor can be fitted."""
        model = MockPredictor()
        X = small_dataframe.select(["timestamp", "feature1"])
        y = small_dataframe["value"]

        result = model.fit(X, y)

        assert result is model  # Should return self
        assert model._fitted is True
        assert model._train_mean == y.mean()

    def test_mock_predictor_predict(self, small_dataframe: pl.DataFrame) -> None:
        """Test that mock predictor can generate predictions."""
        model = MockPredictor()
        X = small_dataframe.select(["timestamp", "feature1"])
        y = small_dataframe["value"]

        model.fit(X, y)
        predictions = model.predict(X[:5])

        assert len(predictions) == 5
        assert all(isinstance(p, PredictionResult) for p in predictions)
        assert all(p.prediction == model._train_mean for p in predictions)

    def test_mock_predictor_predict_without_fit(self, small_dataframe: pl.DataFrame) -> None:
        """Test that predict raises error when model not fitted."""
        model = MockPredictor()
        X = small_dataframe.select(["timestamp", "feature1"])

        with pytest.raises(RuntimeError, match="Model not fitted"):
            model.predict(X)

    def test_mock_predictor_predict_proba(self, small_dataframe: pl.DataFrame) -> None:
        """Test that mock predictor can generate probability predictions."""
        model = MockPredictor()
        X = small_dataframe.select(["timestamp", "feature1"])
        y = small_dataframe["value"]

        model.fit(X, y)
        probas = model.predict_proba(X[:5])

        assert isinstance(probas, pl.DataFrame)
        assert len(probas) == 5
        assert "confidence" in probas.columns

    def test_base_predictor_save_not_implemented(self, mock_model: MockPredictor) -> None:
        """Test that save raises NotImplementedError by default."""
        with pytest.raises(NotImplementedError, match="does not implement save"):
            mock_model.save("/tmp/model.pkl")

    def test_base_predictor_load_not_implemented(self) -> None:
        """Test that load raises NotImplementedError by default."""
        with pytest.raises(NotImplementedError, match="does not implement load"):
            MockPredictor.load("/tmp/model.pkl")

    def test_base_predictor_get_feature_importance_default(
        self, mock_model: MockPredictor
    ) -> None:
        """Test that get_feature_importance returns empty dict by default."""
        importance = mock_model.get_feature_importance()
        assert importance == {}

    def test_base_predictor_to_onnx_not_implemented(self, mock_model: MockPredictor) -> None:
        """Test that to_onnx raises NotImplementedError by default."""
        with pytest.raises(NotImplementedError, match="does not support ONNX export"):
            mock_model.to_onnx("/tmp/model.onnx")


# Test WalkForwardValidator initialization
class TestWalkForwardValidatorInit:
    """Tests for WalkForwardValidator initialization and validation."""

    def test_init_valid_parameters(self) -> None:
        """Test initialization with valid parameters."""
        validator = WalkForwardValidator(
            n_splits=5, train_size=100, test_size=20, purge_days=5, embargo_days=2
        )

        assert validator.n_splits == 5
        assert validator.train_size == 100
        assert validator.test_size == 20
        assert validator.purge_days == 5
        assert validator.embargo_days == 2

    def test_init_expanding_window(self) -> None:
        """Test initialization with expanding window (train_size=None)."""
        validator = WalkForwardValidator(n_splits=3, train_size=None, test_size=10)

        assert validator.train_size is None
        assert validator.n_splits == 3

    def test_init_invalid_n_splits(self) -> None:
        """Test that n_splits < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            WalkForwardValidator(n_splits=1)

    def test_init_invalid_test_size(self) -> None:
        """Test that test_size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="test_size must be positive"):
            WalkForwardValidator(n_splits=3, test_size=0)

    def test_init_invalid_train_size(self) -> None:
        """Test that train_size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="train_size must be positive"):
            WalkForwardValidator(n_splits=3, train_size=-10)

    def test_init_invalid_purge_days(self) -> None:
        """Test that purge_days < 0 raises ValueError."""
        with pytest.raises(ValueError, match="purge_days must be non-negative"):
            WalkForwardValidator(n_splits=3, purge_days=-1)

    def test_init_invalid_embargo_days(self) -> None:
        """Test that embargo_days < 0 raises ValueError."""
        with pytest.raises(ValueError, match="embargo_days must be non-negative"):
            WalkForwardValidator(n_splits=3, embargo_days=-1)


# Test WalkForwardValidator split generation
class TestWalkForwardValidatorSplit:
    """Tests for WalkForwardValidator split generation."""

    def test_split_generates_correct_number_of_folds(
        self, sample_dataframe: pl.DataFrame
    ) -> None:
        """Test that split generates the correct number of folds."""
        validator = WalkForwardValidator(n_splits=3, test_size=20, purge_days=0, embargo_days=0)

        splits = list(validator.split(sample_dataframe))

        assert len(splits) == 3

    def test_split_train_test_sizes(self, sample_dataframe: pl.DataFrame) -> None:
        """Test that train and test sets have expected sizes."""
        validator = WalkForwardValidator(
            n_splits=2, train_size=100, test_size=20, purge_days=0, embargo_days=0
        )

        splits = list(validator.split(sample_dataframe))

        for train_df, test_df in splits:
            # Test set should be exactly test_size
            assert len(test_df) == 20
            # Train set should be at most train_size (can be less due to purging)
            assert len(train_df) <= 100

    def test_split_temporal_ordering(self, sample_dataframe: pl.DataFrame) -> None:
        """Test that train always comes before test in time."""
        validator = WalkForwardValidator(n_splits=3, test_size=20, purge_days=0, embargo_days=0)

        splits = list(validator.split(sample_dataframe))

        for train_df, test_df in splits:
            train_end = train_df["timestamp"].max()
            test_start = test_df["timestamp"].min()
            assert train_end < test_start  # type: ignore[operator]

    def test_split_with_purging(self, sample_dataframe: pl.DataFrame) -> None:
        """Test that purging removes correct days from training set."""
        purge_days = 5
        validator = WalkForwardValidator(n_splits=2, test_size=20, purge_days=purge_days)

        splits = list(validator.split(sample_dataframe))

        for train_df, test_df in splits:
            train_end = train_df["timestamp"].max()
            test_start = test_df["timestamp"].min()
            # Gap should be at least purge_days
            gap = (test_start - train_end).days  # type: ignore[operator, union-attr]
            assert gap >= purge_days

    def test_split_with_embargo(self, sample_dataframe: pl.DataFrame) -> None:
        """Test that embargo excludes recent observations from training."""
        embargo_days = 3
        validator = WalkForwardValidator(n_splits=2, test_size=20, embargo_days=embargo_days, purge_days=0)

        # With embargo, training set should end earlier
        splits_with_embargo = list(validator.split(sample_dataframe))

        validator_no_embargo = WalkForwardValidator(n_splits=2, test_size=20, embargo_days=0, purge_days=0)
        splits_no_embargo = list(validator_no_embargo.split(sample_dataframe))

        # First fold comparison
        train_with, _ = splits_with_embargo[0]
        train_without, _ = splits_no_embargo[0]

        # Training with embargo should have fewer or equal samples (due to filtering)
        # The embargo reduces training data by filtering out recent data before test period
        assert len(train_with) <= len(train_without)

    def test_split_missing_date_column(self, sample_dataframe: pl.DataFrame) -> None:
        """Test that missing date column raises ValueError."""
        validator = WalkForwardValidator(n_splits=2, test_size=10)

        with pytest.raises(ValueError, match="Date column .* not found"):
            list(validator.split(sample_dataframe, date_column="nonexistent"))

    def test_split_insufficient_data(self, small_dataframe: pl.DataFrame) -> None:
        """Test that insufficient data raises ValueError."""
        validator = WalkForwardValidator(n_splits=10, test_size=20)

        with pytest.raises(ValueError, match="Insufficient data"):
            list(validator.split(small_dataframe))

    def test_split_auto_sorts_data(self) -> None:
        """Test that split automatically sorts unsorted data."""
        # Create unsorted DataFrame
        dates = [datetime(2020, 1, 5), datetime(2020, 1, 1), datetime(2020, 1, 3)]
        df = pl.DataFrame({"timestamp": dates, "value": [1.0, 2.0, 3.0]})

        validator = WalkForwardValidator(n_splits=2, test_size=1)

        splits = list(validator.split(df))

        # Should still work and maintain temporal ordering
        for train_df, test_df in splits:
            train_dates = train_df["timestamp"].to_list()
            test_dates = test_df["timestamp"].to_list()

            # Check that each is sorted
            assert train_dates == sorted(train_dates)
            assert test_dates == sorted(test_dates)


# Test WalkForwardValidator cross_validate
class TestWalkForwardValidatorCrossValidate:
    """Tests for WalkForwardValidator cross_validate method."""

    def test_cross_validate_returns_validation_result(
        self, sample_dataframe: pl.DataFrame, mock_model: MockPredictor
    ) -> None:
        """Test that cross_validate returns a ValidationResult."""
        validator = WalkForwardValidator(n_splits=2, test_size=20, purge_days=0, embargo_days=0)

        X = sample_dataframe.select(["timestamp", "feature1", "feature2"])
        y = sample_dataframe["value"]

        result = validator.cross_validate(mock_model, X, y)

        assert isinstance(result, ValidationResult)

    def test_cross_validate_calculates_metrics(
        self, sample_dataframe: pl.DataFrame, mock_model: MockPredictor
    ) -> None:
        """Test that cross_validate calculates requested metrics."""
        validator = WalkForwardValidator(n_splits=2, test_size=20)

        X = sample_dataframe.select(["timestamp", "feature1", "feature2"])
        y = sample_dataframe["value"]

        result = validator.cross_validate(mock_model, X, y, metrics=["mse", "mae"])

        assert "mse" in result.metrics
        assert "mae" in result.metrics
        assert len(result.metrics["mse"]) == 2  # One value per fold
        assert len(result.metrics["mae"]) == 2

    def test_cross_validate_aggregates_metrics(
        self, sample_dataframe: pl.DataFrame, mock_model: MockPredictor
    ) -> None:
        """Test that cross_validate calculates mean and std of metrics."""
        validator = WalkForwardValidator(n_splits=3, test_size=20)

        X = sample_dataframe.select(["timestamp", "feature1", "feature2"])
        y = sample_dataframe["value"]

        result = validator.cross_validate(mock_model, X, y, metrics=["mse"])

        assert "mse" in result.mean_metrics
        assert "mse" in result.std_metrics
        assert isinstance(result.mean_metrics["mse"], float)
        assert isinstance(result.std_metrics["mse"], float)

    def test_cross_validate_stores_predictions(
        self, sample_dataframe: pl.DataFrame, mock_model: MockPredictor
    ) -> None:
        """Test that cross_validate stores all predictions."""
        validator = WalkForwardValidator(n_splits=2, test_size=20)

        X = sample_dataframe.select(["timestamp", "feature1", "feature2"])
        y = sample_dataframe["value"]

        result = validator.cross_validate(mock_model, X, y)

        assert not result.predictions.is_empty()
        assert "timestamp" in result.predictions.columns
        assert "actual" in result.predictions.columns
        assert "predicted" in result.predictions.columns
        assert "fold" in result.predictions.columns
        assert len(result.predictions) == 40  # 2 folds * 20 test samples

    def test_cross_validate_stores_fold_info(
        self, sample_dataframe: pl.DataFrame, mock_model: MockPredictor
    ) -> None:
        """Test that cross_validate stores fold metadata."""
        validator = WalkForwardValidator(n_splits=2, test_size=20)

        X = sample_dataframe.select(["timestamp", "feature1", "feature2"])
        y = sample_dataframe["value"]

        result = validator.cross_validate(mock_model, X, y)

        assert len(result.fold_info) == 2
        for fold_info in result.fold_info:
            assert "fold" in fold_info
            assert "train_start" in fold_info
            assert "train_end" in fold_info
            assert "test_start" in fold_info
            assert "test_end" in fold_info

    def test_cross_validate_incompatible_shapes(
        self, sample_dataframe: pl.DataFrame, mock_model: MockPredictor
    ) -> None:
        """Test that cross_validate raises error for incompatible X and y."""
        validator = WalkForwardValidator(n_splits=2, test_size=20)

        X = sample_dataframe.select(["timestamp", "feature1"])
        y = pl.Series([1.0, 2.0, 3.0])  # Wrong length

        with pytest.raises(ValueError, match="X and y must have same length"):
            validator.cross_validate(mock_model, X, y)

    def test_cross_validate_default_metrics(
        self, sample_dataframe: pl.DataFrame, mock_model: MockPredictor
    ) -> None:
        """Test that cross_validate uses default metrics when none specified."""
        validator = WalkForwardValidator(n_splits=2, test_size=20)

        X = sample_dataframe.select(["timestamp", "feature1", "feature2"])
        y = sample_dataframe["value"]

        result = validator.cross_validate(mock_model, X, y, metrics=None)

        # Default metrics
        assert "mse" in result.metrics
        assert "mae" in result.metrics
        assert "directional_accuracy" in result.metrics


# Test MLflowTracker
class TestMLflowTracker:
    """Tests for MLflowTracker wrapper."""

    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    def test_tracker_initialization(self, mock_get_experiment: Mock, mock_create_experiment: Mock) -> None:
        """Test MLflowTracker initialization."""
        mock_get_experiment.return_value = None
        mock_create_experiment.return_value = "exp123"

        tracker = MLflowTracker(experiment_name="test_experiment")

        assert tracker.experiment_name == "test_experiment"
        assert tracker.experiment_id == "exp123"
        mock_create_experiment.assert_called_once_with("test_experiment")

    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    def test_tracker_uses_existing_experiment(self, mock_get_experiment: Mock, mock_create_experiment: Mock) -> None:
        """Test that tracker uses existing experiment if available."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "existing_exp"
        mock_get_experiment.return_value = mock_experiment

        tracker = MLflowTracker(experiment_name="test_experiment")

        assert tracker.experiment_id == "existing_exp"
        mock_create_experiment.assert_not_called()

    @patch("mlflow.start_run")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    def test_start_run(self, mock_get_experiment: Mock, mock_create_experiment: Mock, mock_start_run: Mock) -> None:
        """Test starting an MLflow run."""
        mock_get_experiment.return_value = None
        mock_create_experiment.return_value = "exp123"
        mock_run = MagicMock()
        mock_run.info.run_id = "run456"
        mock_start_run.return_value = mock_run

        tracker = MLflowTracker()
        tracker.start_run(run_name="test_run", tags={"version": "1.0"})

        assert tracker.active_run is not None
        mock_start_run.assert_called_once()

    @patch("mlflow.log_params")
    @patch("mlflow.start_run")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    def test_log_params(self, mock_get_exp: Mock, mock_create_exp: Mock, mock_start: Mock, mock_log_params: Mock) -> None:
        """Test logging parameters."""
        mock_get_exp.return_value = None
        mock_create_exp.return_value = "exp123"
        mock_run = MagicMock()
        mock_start.return_value = mock_run

        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_params({"learning_rate": 0.001, "epochs": 100})

        mock_log_params.assert_called_once()

    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    def test_log_params_without_run_raises_error(self, mock_get_exp: Mock, mock_create_exp: Mock) -> None:
        """Test that log_params raises error when no run is active."""
        mock_get_exp.return_value = None
        mock_create_exp.return_value = "exp123"

        tracker = MLflowTracker()

        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_params({"param": "value"})

    @patch("mlflow.log_metrics")
    @patch("mlflow.start_run")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    def test_log_metrics(self, mock_get_exp: Mock, mock_create_exp: Mock, mock_start: Mock, mock_log_metrics: Mock) -> None:
        """Test logging metrics."""
        mock_get_exp.return_value = None
        mock_create_exp.return_value = "exp123"
        mock_run = MagicMock()
        mock_start.return_value = mock_run

        tracker = MLflowTracker()
        tracker.start_run()
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=10)

        mock_log_metrics.assert_called_once()

    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    def test_log_metrics_without_run_raises_error(self, mock_get_exp: Mock, mock_create_exp: Mock) -> None:
        """Test that log_metrics raises error when no run is active."""
        mock_get_exp.return_value = None
        mock_create_exp.return_value = "exp123"

        tracker = MLflowTracker()

        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_metrics({"metric": 1.0})

    @patch("mlflow.end_run")
    @patch("mlflow.start_run")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    def test_end_run(self, mock_get_exp: Mock, mock_create_exp: Mock, mock_start: Mock, mock_end_run: Mock) -> None:
        """Test ending an MLflow run."""
        mock_get_exp.return_value = None
        mock_create_exp.return_value = "exp123"
        mock_run = MagicMock()
        mock_start.return_value = mock_run

        tracker = MLflowTracker()
        tracker.start_run()
        tracker.end_run()

        assert tracker.active_run is None
        mock_end_run.assert_called_once()

    @patch("mlflow.end_run")
    @patch("mlflow.start_run")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    def test_run_context_manager(self, mock_get_exp: Mock, mock_create_exp: Mock, mock_start: Mock, mock_end_run: Mock) -> None:
        """Test using run_context as a context manager."""
        mock_get_exp.return_value = None
        mock_create_exp.return_value = "exp123"
        mock_run = MagicMock()
        mock_start.return_value = mock_run

        tracker = MLflowTracker()

        with tracker.run_context(run_name="test_run"):
            assert tracker.active_run is not None

        assert tracker.active_run is None
        mock_end_run.assert_called_once()

    @patch("mlflow.end_run")
    @patch("mlflow.start_run")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    def test_run_context_manager_handles_exceptions(self, mock_get_exp: Mock, mock_create_exp: Mock, mock_start: Mock, mock_end_run: Mock) -> None:
        """Test that run_context ends run even when exception occurs."""
        mock_get_exp.return_value = None
        mock_create_exp.return_value = "exp123"
        mock_run = MagicMock()
        mock_start.return_value = mock_run

        tracker = MLflowTracker()

        with pytest.raises(ValueError), tracker.run_context(run_name="test_run"):
            raise ValueError("Test error")

        # Run should still be ended despite exception
        assert tracker.active_run is None
        mock_end_run.assert_called_once()


# Test ValidationResult
class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self) -> None:
        """Test creating a ValidationResult."""
        result = ValidationResult(
            metrics={"mse": [0.1, 0.15, 0.12], "mae": [0.05, 0.06, 0.055]},
            mean_metrics={"mse": 0.125, "mae": 0.055},
            std_metrics={"mse": 0.025, "mae": 0.005},
            predictions=pl.DataFrame(),
            fold_info=[{"fold": 1, "train_samples": 100, "test_samples": 20}],
        )

        assert result.metrics["mse"] == [0.1, 0.15, 0.12]
        assert result.mean_metrics["mse"] == 0.125
        assert result.std_metrics["mse"] == 0.025
        assert isinstance(result.predictions, pl.DataFrame)
        assert len(result.fold_info) == 1
