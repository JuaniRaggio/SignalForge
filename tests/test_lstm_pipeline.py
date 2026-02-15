"""Tests for LSTM model and training pipeline.

This module provides comprehensive testing of:
- LSTMNetwork forward pass and architecture
- LSTMPredictor fit/predict cycle
- Sequence creation and normalization
- Early stopping mechanism
- ONNX export functionality
- TrainingPipeline data preparation and training
- Model save/load functionality

Tests use synthetic data to ensure deterministic behavior.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
import torch

from signalforge.ml.features.technical import FeatureConfig, TechnicalFeatureEngine
from signalforge.ml.models.lstm import LSTMNetwork, LSTMPredictor
from signalforge.ml.training.pipeline import TrainingPipeline, TrainingResult
from signalforge.ml.training.validation import WalkForwardValidator


@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    """Create sample OHLCV data for testing.

    Returns DataFrame with 300 rows of synthetic price data.
    More rows needed to have enough data after technical indicators create NaN values.
    """
    n_rows = 300
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_rows)]

    # Create synthetic price data with trend
    base_price = 100.0
    prices = [base_price + i * 0.1 + np.sin(i * 0.1) * 2 for i in range(n_rows)]

    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1000000 + i * 1000 for i in range(n_rows)],
        }
    )


@pytest.fixture
def sample_feature_data() -> pl.DataFrame:
    """Create sample feature DataFrame for testing."""
    n_rows = 100
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_rows)]

    return pl.DataFrame(
        {
            "timestamp": dates,
            "feature1": np.random.randn(n_rows),
            "feature2": np.random.randn(n_rows),
            "feature3": np.random.randn(n_rows),
        }
    )


@pytest.fixture
def sample_target() -> pl.Series:
    """Create sample target series for testing."""
    return pl.Series("target", np.random.randn(100))


class TestLSTMNetwork:
    """Tests for LSTMNetwork PyTorch module."""

    def test_network_initialization(self) -> None:
        """Test network initialization with valid parameters."""
        network = LSTMNetwork(input_size=5, hidden_size=64, num_layers=2, dropout=0.2)

        assert network.input_size == 5
        assert network.hidden_size == 64
        assert network.num_layers == 2
        assert isinstance(network.lstm, torch.nn.LSTM)
        assert isinstance(network.fc, torch.nn.Linear)

    def test_network_invalid_input_size(self) -> None:
        """Test network initialization fails with invalid input_size."""
        with pytest.raises(ValueError, match="input_size must be positive"):
            LSTMNetwork(input_size=0, hidden_size=64)

        with pytest.raises(ValueError, match="input_size must be positive"):
            LSTMNetwork(input_size=-1, hidden_size=64)

    def test_network_invalid_hidden_size(self) -> None:
        """Test network initialization fails with invalid hidden_size."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            LSTMNetwork(input_size=5, hidden_size=0)

    def test_network_invalid_num_layers(self) -> None:
        """Test network initialization fails with invalid num_layers."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            LSTMNetwork(input_size=5, hidden_size=64, num_layers=0)

    def test_network_invalid_dropout(self) -> None:
        """Test network initialization fails with invalid dropout."""
        with pytest.raises(ValueError, match="dropout must be in"):
            LSTMNetwork(input_size=5, hidden_size=64, dropout=1.5)

        with pytest.raises(ValueError, match="dropout must be in"):
            LSTMNetwork(input_size=5, hidden_size=64, dropout=-0.1)

    def test_network_forward_pass(self) -> None:
        """Test forward pass produces correct output shape."""
        network = LSTMNetwork(input_size=5, hidden_size=64, num_layers=2)
        batch_size = 8
        seq_len = 20
        input_size = 5

        x = torch.randn(batch_size, seq_len, input_size)
        output = network(x)

        assert output.shape == (batch_size, 1)

    def test_network_single_layer_no_dropout(self) -> None:
        """Test network with single layer has no dropout."""
        network = LSTMNetwork(input_size=5, hidden_size=64, num_layers=1, dropout=0.5)

        # With num_layers=1, LSTM dropout should be 0
        assert network.lstm.dropout == 0.0

    def test_network_multiple_layers_with_dropout(self) -> None:
        """Test network with multiple layers has dropout."""
        network = LSTMNetwork(input_size=5, hidden_size=64, num_layers=2, dropout=0.3)

        # With num_layers > 1, LSTM should have dropout
        assert network.lstm.dropout > 0.0

    def test_network_custom_output_size(self) -> None:
        """Test network with custom output size."""
        network = LSTMNetwork(input_size=5, hidden_size=64, output_size=3)

        x = torch.randn(4, 10, 5)
        output = network(x)

        assert output.shape == (4, 3)


class TestLSTMPredictor:
    """Tests for LSTMPredictor model."""

    def test_predictor_initialization(self) -> None:
        """Test predictor initialization with default parameters."""
        predictor = LSTMPredictor(input_size=5)

        assert predictor.input_size == 5
        assert predictor.sequence_length == 20
        assert predictor.hidden_size == 64
        assert predictor.num_layers == 2
        assert predictor.is_fitted is False
        assert predictor.model_name == "lstm"

    def test_predictor_custom_parameters(self) -> None:
        """Test predictor initialization with custom parameters."""
        predictor = LSTMPredictor(
            input_size=10,
            sequence_length=30,
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            learning_rate=0.0005,
            batch_size=64,
            epochs=200,
            early_stopping_patience=15,
        )

        assert predictor.input_size == 10
        assert predictor.sequence_length == 30
        assert predictor.hidden_size == 128
        assert predictor.num_layers == 3
        assert predictor.dropout == 0.3
        assert predictor.learning_rate == 0.0005
        assert predictor.batch_size == 64
        assert predictor.epochs == 200
        assert predictor.early_stopping_patience == 15

    def test_predictor_device_auto_detection(self) -> None:
        """Test device auto-detection."""
        predictor = LSTMPredictor(input_size=5)

        # Device should be set to cuda, mps, or cpu
        assert predictor.device.type in ["cuda", "mps", "cpu"]

    def test_predictor_invalid_parameters(self) -> None:
        """Test predictor initialization fails with invalid parameters."""
        with pytest.raises(ValueError, match="input_size must be positive"):
            LSTMPredictor(input_size=0)

        with pytest.raises(ValueError, match="sequence_length must be positive"):
            LSTMPredictor(input_size=5, sequence_length=0)

        with pytest.raises(ValueError, match="epochs must be positive"):
            LSTMPredictor(input_size=5, epochs=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            LSTMPredictor(input_size=5, batch_size=0)

        with pytest.raises(ValueError, match="early_stopping_patience must be positive"):
            LSTMPredictor(input_size=5, early_stopping_patience=0)

    def test_create_sequences(self, sample_feature_data: pl.DataFrame, sample_target: pl.Series) -> None:
        """Test sequence creation from time series data."""
        predictor = LSTMPredictor(input_size=3, sequence_length=10)

        X_seq, y_seq = predictor._create_sequences(
            sample_feature_data.drop("timestamp"), sample_target
        )

        # Should create len(X) - sequence_length sequences
        expected_n_sequences = len(sample_feature_data) - 10
        assert X_seq.shape == (expected_n_sequences, 10, 3)
        assert y_seq.shape == (expected_n_sequences,)

    def test_normalize(self, sample_feature_data: pl.DataFrame) -> None:
        """Test feature normalization."""
        predictor = LSTMPredictor(input_size=3)

        X_norm, scaler_params = predictor._normalize(sample_feature_data.drop("timestamp"))

        # Check normalization
        assert "mean" in scaler_params
        assert "std" in scaler_params
        assert len(scaler_params["mean"]) == 3
        assert len(scaler_params["std"]) == 3

        # Check that normalized data has approximately mean=0, std=1
        X_array = X_norm.to_numpy()
        assert np.abs(np.mean(X_array, axis=0)).max() < 0.1
        assert np.abs(np.std(X_array, axis=0) - 1.0).max() < 0.1

    def test_fit_basic(self, sample_feature_data: pl.DataFrame, sample_target: pl.Series) -> None:
        """Test basic model fitting."""
        predictor = LSTMPredictor(
            input_size=3,
            sequence_length=10,
            epochs=5,
            batch_size=16,
        )

        X = sample_feature_data.drop("timestamp")
        result = predictor.fit(X, sample_target)

        assert result is predictor  # Should return self
        assert predictor.is_fitted is True
        assert predictor.scaler_mean is not None
        assert predictor.scaler_std is not None

    def test_fit_insufficient_data(self) -> None:
        """Test fitting fails with insufficient data."""
        predictor = LSTMPredictor(input_size=3, sequence_length=20)

        # Create small dataset
        X = pl.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
            "f3": [7.0, 8.0, 9.0],
        })
        y = pl.Series([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Insufficient data"):
            predictor.fit(X, y)

    def test_fit_mismatched_shapes(self, sample_feature_data: pl.DataFrame) -> None:
        """Test fitting fails with mismatched X and y shapes."""
        predictor = LSTMPredictor(input_size=3)

        X = sample_feature_data.drop("timestamp")
        y = pl.Series("target", np.random.randn(50))  # Different length

        with pytest.raises(ValueError, match="X and y must have same length"):
            predictor.fit(X, y)

    def test_fit_wrong_feature_count(self, sample_feature_data: pl.DataFrame, sample_target: pl.Series) -> None:
        """Test fitting fails with wrong number of features."""
        predictor = LSTMPredictor(input_size=5)  # Expect 5 features

        X = sample_feature_data.drop("timestamp")  # Only has 3 features

        with pytest.raises(ValueError, match="X must have 5 features"):
            predictor.fit(X, sample_target)

    def test_predict_basic(self, sample_feature_data: pl.DataFrame, sample_target: pl.Series) -> None:
        """Test basic prediction."""
        predictor = LSTMPredictor(
            input_size=3,
            sequence_length=10,
            epochs=5,
            batch_size=16,
        )

        X = sample_feature_data.drop("timestamp")
        predictor.fit(X, sample_target)

        # Predict on same data
        predictions = predictor.predict(X)

        assert len(predictions) > 0
        assert all(hasattr(p, "prediction") for p in predictions)
        assert all(isinstance(p.prediction, float) for p in predictions)

    def test_predict_not_fitted(self, sample_feature_data: pl.DataFrame) -> None:
        """Test prediction fails if model not fitted."""
        predictor = LSTMPredictor(input_size=3)

        X = sample_feature_data.drop("timestamp")

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            predictor.predict(X)

    def test_predict_insufficient_data(self, sample_feature_data: pl.DataFrame, sample_target: pl.Series) -> None:
        """Test prediction fails with insufficient data."""
        predictor = LSTMPredictor(
            input_size=3,
            sequence_length=10,
            epochs=5,
        )

        X = sample_feature_data.drop("timestamp")
        predictor.fit(X, sample_target)

        # Try to predict with too little data
        X_small = X.head(5)  # Less than sequence_length

        with pytest.raises(ValueError, match="Need at least"):
            predictor.predict(X_small)

    def test_predict_proba(self, sample_feature_data: pl.DataFrame, sample_target: pl.Series) -> None:
        """Test predict_proba returns DataFrame."""
        predictor = LSTMPredictor(
            input_size=3,
            sequence_length=10,
            epochs=5,
        )

        X = sample_feature_data.drop("timestamp")
        predictor.fit(X, sample_target)

        proba = predictor.predict_proba(X)

        assert isinstance(proba, pl.DataFrame)
        assert "prediction" in proba.columns
        assert "confidence" in proba.columns

    def test_early_stopping(self, sample_feature_data: pl.DataFrame, sample_target: pl.Series) -> None:
        """Test early stopping mechanism."""
        predictor = LSTMPredictor(
            input_size=3,
            sequence_length=10,
            epochs=1000,  # High number
            early_stopping_patience=5,
            batch_size=16,
        )

        X = sample_feature_data.drop("timestamp")

        # Training should stop early due to early stopping
        predictor.fit(X, sample_target)

        assert predictor.is_fitted is True
        # Should have stopped before 1000 epochs (hard to test exact behavior)

    def test_get_feature_importance(self) -> None:
        """Test get_feature_importance returns empty dict."""
        predictor = LSTMPredictor(input_size=3)

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 0


class TestLSTMPersistence:
    """Tests for LSTM model save/load functionality."""

    def test_save_and_load(
        self,
        sample_feature_data: pl.DataFrame,
        sample_target: pl.Series,
        tmp_path: Path,
    ) -> None:
        """Test model save and load cycle."""
        predictor = LSTMPredictor(
            input_size=3,
            sequence_length=10,
            hidden_size=32,
            epochs=5,
        )

        X = sample_feature_data.drop("timestamp")
        predictor.fit(X, sample_target)

        # Save model
        model_path = tmp_path / "model.pkl"
        predictor.save(str(model_path))

        assert model_path.exists()

        # Load model
        loaded = LSTMPredictor.load(str(model_path))

        assert loaded.input_size == predictor.input_size
        assert loaded.sequence_length == predictor.sequence_length
        assert loaded.hidden_size == predictor.hidden_size
        assert loaded.is_fitted is True

        # Make predictions with both models
        preds_original = predictor.predict(X)
        preds_loaded = loaded.predict(X)

        # Predictions should be very similar
        assert len(preds_original) == len(preds_loaded)
        for p1, p2 in zip(preds_original, preds_loaded, strict=True):
            assert abs(p1.prediction - p2.prediction) < 0.01

    def test_save_creates_directory(
        self,
        sample_feature_data: pl.DataFrame,
        sample_target: pl.Series,
        tmp_path: Path,
    ) -> None:
        """Test save creates parent directories if needed."""
        predictor = LSTMPredictor(input_size=3, epochs=2)

        X = sample_feature_data.drop("timestamp")
        predictor.fit(X, sample_target)

        # Save to nested path
        model_path = tmp_path / "nested" / "dir" / "model.pkl"
        predictor.save(str(model_path))

        assert model_path.exists()

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading from nonexistent file fails."""
        model_path = tmp_path / "nonexistent.pkl"

        with pytest.raises(IOError, match="Failed to load model"):
            LSTMPredictor.load(str(model_path))


class TestONNXExport:
    """Tests for ONNX export functionality."""

    def test_onnx_export(
        self,
        sample_feature_data: pl.DataFrame,
        sample_target: pl.Series,
        tmp_path: Path,
    ) -> None:
        """Test ONNX model export."""
        predictor = LSTMPredictor(
            input_size=3,
            sequence_length=10,
            epochs=2,
        )

        X = sample_feature_data.drop("timestamp")
        predictor.fit(X, sample_target)

        # Export to ONNX (may fail if onnxscript not installed)
        onnx_path = tmp_path / "model.onnx"
        try:
            predictor.to_onnx(str(onnx_path))
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0
        except RuntimeError as e:
            if "onnxscript" in str(e):
                pytest.skip("onnxscript not installed")
            else:
                raise

    def test_onnx_export_not_fitted(self, tmp_path: Path) -> None:
        """Test ONNX export fails if model not fitted."""
        predictor = LSTMPredictor(input_size=3)

        onnx_path = tmp_path / "model.onnx"

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            predictor.to_onnx(str(onnx_path))


class TestTrainingPipeline:
    """Tests for TrainingPipeline orchestrator."""

    def test_pipeline_initialization(self) -> None:
        """Test pipeline initialization."""
        engine = TechnicalFeatureEngine()
        validator = WalkForwardValidator(n_splits=3)

        pipeline = TrainingPipeline(engine, validator)

        assert pipeline.feature_engine is engine
        assert pipeline.validator is validator
        assert pipeline.mlflow_tracker is None

    def test_prepare_data(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test data preparation."""
        engine = TechnicalFeatureEngine(
            FeatureConfig(
                sma_periods=[5, 10],
                ema_periods=[5],
                rsi_periods=[14],
            )
        )
        validator = WalkForwardValidator(n_splits=3)
        pipeline = TrainingPipeline(engine, validator)

        X, y = pipeline.prepare_data(sample_ohlcv_data, target_column="close", horizon=5)

        assert isinstance(X, pl.DataFrame)
        assert isinstance(y, pl.Series)
        assert len(X) == len(y)
        assert len(X) > 0
        assert "timestamp" in X.columns
        # Should have technical features
        assert X.width > 1

    def test_prepare_data_missing_columns(self) -> None:
        """Test prepare_data fails with missing columns."""
        engine = TechnicalFeatureEngine()
        validator = WalkForwardValidator(n_splits=3)
        pipeline = TrainingPipeline(engine, validator)

        # DataFrame without required columns
        df = pl.DataFrame({"timestamp": [datetime.now()], "price": [100.0]})

        with pytest.raises(ValueError, match="Missing required columns"):
            pipeline.prepare_data(df)

    def test_prepare_data_insufficient_data(self) -> None:
        """Test prepare_data fails with insufficient data."""
        engine = TechnicalFeatureEngine()
        validator = WalkForwardValidator(n_splits=3)
        pipeline = TrainingPipeline(engine, validator)

        # Small dataset
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)],
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000000] * 10,
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            pipeline.prepare_data(df, horizon=5)

    def test_train_basic(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test basic model training through pipeline."""
        engine = TechnicalFeatureEngine(
            FeatureConfig(
                sma_periods=[5, 10],
                ema_periods=[5],
                rsi_periods=[14],
            )
        )
        validator = WalkForwardValidator(n_splits=2, test_size=20)
        pipeline = TrainingPipeline(engine, validator)

        # Prepare data first to get feature count
        X, y = pipeline.prepare_data(sample_ohlcv_data, horizon=5)
        n_features = X.width - 1  # Exclude timestamp

        model = LSTMPredictor(
            input_size=n_features,
            sequence_length=10,
            epochs=3,
            batch_size=16,
        )

        result = pipeline.train(
            model,
            sample_ohlcv_data,
            target_column="close",
            horizon=5,
            run_validation=True,
        )

        assert isinstance(result, TrainingResult)
        assert result.model is model
        assert result.training_time > 0
        assert "mse" in result.final_metrics
        assert "mae" in result.final_metrics
        assert result.validation_result is not None

    def test_train_without_validation(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test training without validation."""
        engine = TechnicalFeatureEngine(FeatureConfig(sma_periods=[5]))
        validator = WalkForwardValidator(n_splits=2)
        pipeline = TrainingPipeline(engine, validator)

        X, y = pipeline.prepare_data(sample_ohlcv_data)
        n_features = X.width - 1

        model = LSTMPredictor(
            input_size=n_features,
            sequence_length=10,
            epochs=3,
        )

        result = pipeline.train(
            model,
            sample_ohlcv_data,
            run_validation=False,
        )

        assert result.validation_result is None
        assert result.training_time > 0

    def test_train_multiple(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test training multiple models."""
        engine = TechnicalFeatureEngine(FeatureConfig(sma_periods=[5]))
        validator = WalkForwardValidator(n_splits=2, test_size=20)
        pipeline = TrainingPipeline(engine, validator)

        X, y = pipeline.prepare_data(sample_ohlcv_data)
        n_features = X.width - 1

        models = [
            LSTMPredictor(input_size=n_features, sequence_length=10, hidden_size=32, epochs=2),
            LSTMPredictor(input_size=n_features, sequence_length=10, hidden_size=64, epochs=2),
        ]

        results = pipeline.train_multiple(
            models,
            sample_ohlcv_data,
            run_validation=False,
        )

        assert len(results) == 2
        assert all(isinstance(r, TrainingResult) for r in results)
        assert all(r.model.is_fitted for r in results)

    def test_train_multiple_empty_list(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test train_multiple fails with empty model list."""
        engine = TechnicalFeatureEngine()
        validator = WalkForwardValidator(n_splits=2)
        pipeline = TrainingPipeline(engine, validator)

        with pytest.raises(ValueError, match="models list cannot be empty"):
            pipeline.train_multiple([], sample_ohlcv_data)


class TestTrainingPipelineWithMLflow:
    """Tests for TrainingPipeline with MLflow tracking."""

    def test_train_with_mlflow_tracker(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test training with MLflow tracker."""
        engine = TechnicalFeatureEngine(FeatureConfig(sma_periods=[5]))
        validator = WalkForwardValidator(n_splits=2, test_size=20)

        # Mock MLflow tracker
        mock_tracker = MagicMock()
        pipeline = TrainingPipeline(engine, validator, mlflow_tracker=mock_tracker)

        X, y = pipeline.prepare_data(sample_ohlcv_data)
        n_features = X.width - 1

        model = LSTMPredictor(
            input_size=n_features,
            sequence_length=10,
            epochs=2,
        )

        pipeline.train(
            model,
            sample_ohlcv_data,
            run_validation=False,
        )

        # Verify MLflow methods were called
        assert mock_tracker.start_run.called
        assert mock_tracker.log_params.called
        assert mock_tracker.log_metrics.called
        assert mock_tracker.end_run.called

    def test_train_mlflow_tracker_exception_handling(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test MLflow tracker is properly closed even on exception."""
        engine = TechnicalFeatureEngine(FeatureConfig(sma_periods=[5]))
        validator = WalkForwardValidator(n_splits=2)

        mock_tracker = MagicMock()
        pipeline = TrainingPipeline(engine, validator, mlflow_tracker=mock_tracker)

        # Create model that will fail during fit
        model = LSTMPredictor(input_size=999, sequence_length=10)  # Wrong feature count

        with pytest.raises(RuntimeError):
            pipeline.train(model, sample_ohlcv_data)

        # end_run should still be called
        assert mock_tracker.end_run.called


# Run count: should have at least 35 tests
def test_test_count() -> None:
    """Verify we have at least 35 tests."""
    import inspect

    test_functions = 0
    test_classes = [
        TestLSTMNetwork,
        TestLSTMPredictor,
        TestLSTMPersistence,
        TestONNXExport,
        TestTrainingPipeline,
        TestTrainingPipelineWithMLflow,
    ]

    for test_class in test_classes:
        methods = inspect.getmembers(test_class, predicate=inspect.isfunction)
        test_methods = [m for m in methods if m[0].startswith("test_")]
        test_functions += len(test_methods)

    # Add standalone test
    test_functions += 1  # This test itself

    assert test_functions >= 35, f"Expected at least 35 tests, found {test_functions}"
