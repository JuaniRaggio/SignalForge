"""Comprehensive tests for ML inference service.

This test suite covers:
- PredictionService functionality
- Caching behavior
- Batch predictions
- ONNXPredictor operations
- ModelRegistry CRUD operations
- Error handling and edge cases
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import polars as pl
import pytest

from signalforge.ml.features.technical import TechnicalFeatureEngine
from signalforge.ml.inference.model_registry import ModelInfo, ModelRegistry
from signalforge.ml.inference.onnx_runtime import ONNXPredictor
from signalforge.ml.inference.predictor import PredictionResponse, PredictionService
from signalforge.ml.models.base import BasePredictor, PredictionResult

# Check if onnxruntime is available
try:
    import onnxruntime  # noqa: F401

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


# Mock predictor for testing
class MockPredictor(BasePredictor):
    """Mock predictor for testing."""

    model_name = "MockPredictor"
    model_version = "1.0.0"

    def __init__(self, prediction_value: float = 2.5) -> None:
        """Initialize mock predictor."""
        self.prediction_value = prediction_value
        self.fit_called = False
        self.predict_called = False

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs: Any) -> BasePredictor:
        """Mock fit method."""
        self.fit_called = True
        return self

    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        """Mock predict method."""
        self.predict_called = True
        results = []
        for _ in range(X.height):
            results.append(
                PredictionResult(
                    symbol="TEST",
                    timestamp=datetime.now(),
                    horizon_days=5,
                    prediction=self.prediction_value,
                    confidence=0.85,
                    lower_bound=self.prediction_value - 1.0,
                    upper_bound=self.prediction_value + 1.0,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return results

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Mock predict_proba method."""
        return pl.DataFrame({"confidence": [0.85] * X.height})


# Mock predictor that raises error in predict_proba
class ErrorPredictor(MockPredictor):
    """Mock predictor that raises error in confidence calculation."""

    model_name = "ErrorPredictor"

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """Mock predict_proba that raises error."""
        raise RuntimeError("Confidence calculation failed")


# Fixtures
@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model() -> MockPredictor:
    """Create mock model."""
    return MockPredictor(prediction_value=2.5)


@pytest.fixture
def model_registry(temp_dir: Path) -> ModelRegistry:
    """Create model registry with temp storage."""
    return ModelRegistry(storage_path=str(temp_dir))


@pytest.fixture
def feature_engine() -> TechnicalFeatureEngine:
    """Create feature engine."""
    return TechnicalFeatureEngine()


@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    """Create sample OHLCV data."""
    dates = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 3, 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": [150.0] * len(dates),
            "high": [155.0] * len(dates),
            "low": [148.0] * len(dates),
            "close": [152.0 + i * 0.1 for i in range(len(dates))],
            "volume": [1000000] * len(dates),
        }
    )


# PredictionService Tests
class TestPredictionService:
    """Tests for PredictionService."""

    def test_initialization(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
    ) -> None:
        """Test service initialization."""
        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
            cache_ttl=300,
        )
        assert service.model_registry == model_registry
        assert service.feature_engine == feature_engine
        assert service.cache_ttl == 300
        assert len(service._cache) == 0

    def test_initialization_default_cache_ttl(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
    ) -> None:
        """Test service initialization with default cache TTL."""
        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )
        assert service.cache_ttl == 300

    @pytest.mark.asyncio
    async def test_predict_raises_not_implemented(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        mock_model: MockPredictor,
    ) -> None:
        """Test predict raises NotImplementedError for data fetching."""
        # Register model as default
        model_id = model_registry.register(mock_model)
        model_registry.set_default(model_id)

        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )

        with pytest.raises(NotImplementedError, match="Data fetching not implemented"):
            await service.predict("AAPL", horizon_days=5)

    @pytest.mark.asyncio
    async def test_predict_with_mock_data(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        mock_model: MockPredictor,
        sample_ohlcv_data: pl.DataFrame,
    ) -> None:
        """Test predict with mocked data fetching."""
        # Register model as default
        model_id = model_registry.register(mock_model)
        model_registry.set_default(model_id)

        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )

        # Mock the data fetching method
        service._fetch_recent_data = AsyncMock(return_value=sample_ohlcv_data)

        result = await service.predict("AAPL", horizon_days=5, use_cache=False)

        assert isinstance(result, PredictionResponse)
        assert result.symbol == "AAPL"
        assert result.horizon_days == 5
        assert result.predicted_return == 2.5
        assert result.confidence == 0.85
        assert result.model_used == "MockPredictor"
        assert result.model_version == "1.0.0"
        assert result.latency_ms > 0
        assert result.predicted_price is not None

    @pytest.mark.asyncio
    async def test_predict_caching(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        mock_model: MockPredictor,
        sample_ohlcv_data: pl.DataFrame,
    ) -> None:
        """Test prediction caching."""
        model_id = model_registry.register(mock_model)
        model_registry.set_default(model_id)

        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
            cache_ttl=300,
        )
        service._fetch_recent_data = AsyncMock(return_value=sample_ohlcv_data)

        # First call - should cache
        result1 = await service.predict("AAPL", horizon_days=5, use_cache=True)
        assert len(service._cache) == 1

        # Second call - should use cache
        result2 = await service.predict("AAPL", horizon_days=5, use_cache=True)
        assert result1.timestamp == result2.timestamp
        assert result1.latency_ms == result2.latency_ms

        # Verify data fetch was only called once
        assert service._fetch_recent_data.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_expiration(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        mock_model: MockPredictor,
        sample_ohlcv_data: pl.DataFrame,
    ) -> None:
        """Test cache expiration."""
        model_id = model_registry.register(mock_model)
        model_registry.set_default(model_id)

        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
            cache_ttl=1,  # 1 second TTL
        )
        service._fetch_recent_data = AsyncMock(return_value=sample_ohlcv_data)

        # First call
        await service.predict("AAPL", horizon_days=5, use_cache=True)
        assert len(service._cache) == 1

        # Wait for cache to expire
        time.sleep(1.1)

        # Second call - cache should be expired
        await service.predict("AAPL", horizon_days=5, use_cache=True)

        # Verify data fetch was called twice
        assert service._fetch_recent_data.call_count == 2

    @pytest.mark.asyncio
    async def test_predict_without_cache(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        mock_model: MockPredictor,
        sample_ohlcv_data: pl.DataFrame,
    ) -> None:
        """Test prediction without caching."""
        model_id = model_registry.register(mock_model)
        model_registry.set_default(model_id)

        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )
        service._fetch_recent_data = AsyncMock(return_value=sample_ohlcv_data)

        # Call twice without cache
        await service.predict("AAPL", horizon_days=5, use_cache=False)
        await service.predict("AAPL", horizon_days=5, use_cache=False)

        # Cache should be empty
        assert len(service._cache) == 0
        # Data fetch should be called twice
        assert service._fetch_recent_data.call_count == 2

    @pytest.mark.asyncio
    async def test_predict_specific_model(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        sample_ohlcv_data: pl.DataFrame,
    ) -> None:
        """Test prediction with specific model."""
        # Register two models
        model1 = MockPredictor(prediction_value=2.5)
        model2 = MockPredictor(prediction_value=5.0)
        model2.model_name = "MockPredictor2"

        model1_id = model_registry.register(model1)
        model2_id = model_registry.register(model2)
        model_registry.set_default(model1_id)

        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )
        service._fetch_recent_data = AsyncMock(return_value=sample_ohlcv_data)

        # Predict with specific model
        result = await service.predict(
            "AAPL",
            horizon_days=5,
            model_name=model2_id,
            use_cache=False,
        )

        assert result.predicted_return == 5.0
        assert result.model_used == "MockPredictor2"

    @pytest.mark.asyncio
    async def test_predict_batch(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        mock_model: MockPredictor,
        sample_ohlcv_data: pl.DataFrame,
    ) -> None:
        """Test batch prediction."""
        model_id = model_registry.register(mock_model)
        model_registry.set_default(model_id)

        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )
        service._fetch_recent_data = AsyncMock(return_value=sample_ohlcv_data)

        symbols = ["AAPL", "GOOGL", "MSFT"]
        results = await service.predict_batch(symbols, horizon_days=5)

        assert len(results) == 3
        assert all(isinstance(r, PredictionResponse) for r in results)
        assert [r.symbol for r in results] == symbols

    @pytest.mark.asyncio
    async def test_predict_batch_empty_list(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
    ) -> None:
        """Test batch prediction with empty list."""
        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )

        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            await service.predict_batch([])

    @pytest.mark.asyncio
    async def test_predict_batch_with_errors(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        mock_model: MockPredictor,
        sample_ohlcv_data: pl.DataFrame,
    ) -> None:
        """Test batch prediction with some failures."""
        model_id = model_registry.register(mock_model)
        model_registry.set_default(model_id)

        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )

        # Mock to fail for specific symbol
        async def mock_fetch(symbol: str) -> pl.DataFrame:
            if symbol == "INVALID":
                raise ValueError("Invalid symbol")
            return sample_ohlcv_data

        service._fetch_recent_data = mock_fetch

        symbols = ["AAPL", "INVALID", "MSFT"]
        results = await service.predict_batch(symbols, horizon_days=5)

        # Should return results for successful symbols only
        assert len(results) == 2
        assert all(r.symbol in ["AAPL", "MSFT"] for r in results)

    def test_clear_cache(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
    ) -> None:
        """Test cache clearing."""
        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )

        # Add items to cache manually
        response = PredictionResponse(
            symbol="AAPL",
            timestamp=datetime.now(),
            horizon_days=5,
            predicted_return=2.5,
            predicted_price=152.0,
            confidence=0.85,
            interval_lower=150.0,
            interval_upper=154.0,
            model_used="MockPredictor",
            model_version="1.0.0",
            features_used=45,
            latency_ms=10.0,
        )
        service._cache["key1"] = (response, time.time())
        service._cache["key2"] = (response, time.time())

        assert len(service._cache) == 2

        service.clear_cache()
        assert len(service._cache) == 0

    def test_prepare_features(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        sample_ohlcv_data: pl.DataFrame,
    ) -> None:
        """Test feature preparation."""
        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )

        # Compute features
        df_with_features = feature_engine.compute_all(sample_ohlcv_data)

        # Prepare for model
        X = service._prepare_features(df_with_features)

        # Should not contain OHLCV columns
        assert "open" not in X.columns
        assert "high" not in X.columns
        assert "low" not in X.columns
        assert "close" not in X.columns
        assert "volume" not in X.columns
        assert "timestamp" not in X.columns

        # Should contain feature columns
        feature_names = feature_engine.get_feature_names()
        for feature in feature_names:
            if feature in df_with_features.columns:
                assert feature in X.columns

        # No NaN or null values
        assert X.null_count().sum_horizontal().item() == 0

    def test_prepare_features_handles_missing_values(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
    ) -> None:
        """Test feature preparation handles missing values."""
        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )

        # Create DataFrame with NaN values
        df = pl.DataFrame(
            {
                "sma_5": [1.0, float("nan"), 3.0],
                "rsi_14": [50.0, 60.0, None],
            }
        )

        X = service._prepare_features(df)

        # All values should be filled
        assert X.null_count().sum_horizontal().item() == 0
        # Check for NaN values - sum all is_nan across all columns
        nan_df = X.select(pl.col("*").is_nan())
        total_nans = nan_df.select(pl.sum_horizontal(nan_df.columns)).sum().item()
        assert total_nans == 0


# ModelRegistry Tests
class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_initialization(self, temp_dir: Path) -> None:
        """Test registry initialization."""
        registry = ModelRegistry(storage_path=str(temp_dir))
        assert registry.storage_path == temp_dir
        assert registry.index_file == temp_dir / "registry_index.json"
        assert len(registry.models_index) == 0

    def test_initialization_creates_directory(self, temp_dir: Path) -> None:
        """Test registry creates storage directory."""
        new_dir = temp_dir / "new_models"
        assert not new_dir.exists()

        _ = ModelRegistry(storage_path=str(new_dir))
        assert new_dir.exists()

    def test_register_model(
        self,
        model_registry: ModelRegistry,
        mock_model: MockPredictor,
    ) -> None:
        """Test model registration."""
        metrics = {"mse": 0.025, "mae": 0.12}
        tags = {"type": "mock", "dataset": "test"}

        model_id = model_registry.register(mock_model, metrics=metrics, tags=tags)

        assert model_id in model_registry.models_index
        info = model_registry.models_index[model_id]
        assert info.model_name == "MockPredictor"
        assert info.model_version == "1.0.0"
        assert info.metrics == metrics
        assert info.tags == tags
        assert not info.is_default

    def test_register_model_without_metadata(
        self,
        model_registry: ModelRegistry,
        mock_model: MockPredictor,
    ) -> None:
        """Test model registration without metrics or tags."""
        model_id = model_registry.register(mock_model)

        info = model_registry.models_index[model_id]
        assert info.metrics == {}
        assert info.tags == {}

    def test_get_model(
        self,
        model_registry: ModelRegistry,
        mock_model: MockPredictor,
    ) -> None:
        """Test model retrieval."""
        model_id = model_registry.register(mock_model)

        loaded_model = model_registry.get(model_id)
        assert isinstance(loaded_model, MockPredictor)
        assert loaded_model.model_name == "MockPredictor"
        assert loaded_model.model_version == "1.0.0"

    def test_get_nonexistent_model(self, model_registry: ModelRegistry) -> None:
        """Test getting nonexistent model raises error."""
        with pytest.raises(KeyError, match="Model not found"):
            model_registry.get("nonexistent-id")

    def test_set_default_model(
        self,
        model_registry: ModelRegistry,
        mock_model: MockPredictor,
    ) -> None:
        """Test setting default model."""
        model_id = model_registry.register(mock_model)
        model_registry.set_default(model_id)

        info = model_registry.models_index[model_id]
        assert info.is_default

    def test_set_default_updates_previous_default(
        self,
        model_registry: ModelRegistry,
    ) -> None:
        """Test setting new default unsets previous default."""
        model1 = MockPredictor(prediction_value=1.0)
        model2 = MockPredictor(prediction_value=2.0)

        model1_id = model_registry.register(model1)
        model2_id = model_registry.register(model2)

        model_registry.set_default(model1_id)
        assert model_registry.models_index[model1_id].is_default

        model_registry.set_default(model2_id)
        assert not model_registry.models_index[model1_id].is_default
        assert model_registry.models_index[model2_id].is_default

    def test_get_default_model(
        self,
        model_registry: ModelRegistry,
        mock_model: MockPredictor,
    ) -> None:
        """Test getting default model."""
        model_id = model_registry.register(mock_model)
        model_registry.set_default(model_id)

        default_model = model_registry.get_default()
        assert isinstance(default_model, MockPredictor)

    def test_get_default_no_default_set(
        self,
        model_registry: ModelRegistry,
    ) -> None:
        """Test getting default when none set raises error."""
        with pytest.raises(ValueError, match="No default model set"):
            model_registry.get_default()

    def test_get_best_model(self, model_registry: ModelRegistry) -> None:
        """Test getting best model by metric."""
        model1 = MockPredictor(prediction_value=1.0)
        model2 = MockPredictor(prediction_value=2.0)
        model3 = MockPredictor(prediction_value=3.0)

        model_registry.register(model1, metrics={"mse": 0.030})
        model_registry.register(model2, metrics={"mse": 0.020})
        model_registry.register(model3, metrics={"mse": 0.025})

        best_model = model_registry.get_best(metric="mse")
        assert isinstance(best_model, MockPredictor)
        assert best_model.prediction_value == 2.0

    def test_get_best_model_by_type(self, model_registry: ModelRegistry) -> None:
        """Test getting best model filtered by type."""
        model1 = MockPredictor(prediction_value=1.0)
        model2 = MockPredictor(prediction_value=2.0)
        model2.model_name = "OtherPredictor"

        model_registry.register(
            model1, metrics={"mse": 0.030}, tags={"type": "mock"}
        )
        model_registry.register(
            model2, metrics={"mse": 0.020}, tags={"type": "other"}
        )

        best_mock = model_registry.get_best(metric="mse", model_type="mock")
        assert best_mock.model_name == "MockPredictor"

    def test_get_best_no_models(self, model_registry: ModelRegistry) -> None:
        """Test getting best model when no models exist."""
        with pytest.raises(ValueError, match="No models found"):
            model_registry.get_best(metric="mse")

    def test_get_best_metric_not_found(
        self,
        model_registry: ModelRegistry,
        mock_model: MockPredictor,
    ) -> None:
        """Test getting best model when metric doesn't exist."""
        model_registry.register(mock_model, metrics={"mae": 0.1})

        with pytest.raises(KeyError, match="No models found with metric"):
            model_registry.get_best(metric="mse")

    def test_list_models(self, model_registry: ModelRegistry) -> None:
        """Test listing all models."""
        model1 = MockPredictor(prediction_value=1.0)
        model2 = MockPredictor(prediction_value=2.0)

        model_registry.register(model1)
        time.sleep(0.01)  # Ensure different timestamps
        model_registry.register(model2)

        models = model_registry.list_models()
        assert len(models) == 2
        # Should be sorted by creation time, newest first
        assert models[0].created_at > models[1].created_at

    def test_list_models_filtered_by_type(
        self,
        model_registry: ModelRegistry,
    ) -> None:
        """Test listing models filtered by type."""
        model1 = MockPredictor(prediction_value=1.0)
        model2 = MockPredictor(prediction_value=2.0)

        model_registry.register(model1, tags={"type": "mock"})
        model_registry.register(model2, tags={"type": "other"})

        models = model_registry.list_models(model_type="mock")
        assert len(models) == 1
        assert models[0].tags["type"] == "mock"

    def test_delete_model(
        self,
        model_registry: ModelRegistry,
        mock_model: MockPredictor,
    ) -> None:
        """Test deleting a model."""
        model_id = model_registry.register(mock_model)
        assert model_id in model_registry.models_index

        model_registry.delete(model_id)
        assert model_id not in model_registry.models_index

    def test_delete_nonexistent_model(
        self,
        model_registry: ModelRegistry,
    ) -> None:
        """Test deleting nonexistent model raises error."""
        with pytest.raises(KeyError, match="Model not found"):
            model_registry.delete("nonexistent-id")

    def test_delete_default_model_raises_error(
        self,
        model_registry: ModelRegistry,
        mock_model: MockPredictor,
    ) -> None:
        """Test deleting default model raises error."""
        model_id = model_registry.register(mock_model)
        model_registry.set_default(model_id)

        with pytest.raises(ValueError, match="Cannot delete default model"):
            model_registry.delete(model_id)

    def test_persistence(self, temp_dir: Path, mock_model: MockPredictor) -> None:
        """Test registry persists to disk."""
        # Create registry and register model
        registry1 = ModelRegistry(storage_path=str(temp_dir))
        model_id = registry1.register(
            mock_model,
            metrics={"mse": 0.025},
            tags={"type": "mock"},
        )
        registry1.set_default(model_id)

        # Create new registry instance (should load from disk)
        registry2 = ModelRegistry(storage_path=str(temp_dir))
        assert model_id in registry2.models_index
        assert registry2.models_index[model_id].is_default
        assert registry2.models_index[model_id].metrics == {"mse": 0.025}


# ONNXPredictor Tests
class TestONNXPredictor:
    """Tests for ONNXPredictor."""

    def test_initialization_missing_file(self) -> None:
        """Test initialization with missing file raises error."""
        if not HAS_ONNXRUNTIME:
            pytest.skip("onnxruntime not installed")
        with pytest.raises((FileNotFoundError, ImportError)):
            ONNXPredictor("nonexistent.onnx")

    def test_initialization_missing_onnxruntime(self) -> None:
        """Test initialization when onnxruntime not installed."""
        # This test would only pass if onnxruntime is not installed
        # Since we can't easily mock module imports, we skip this
        pytest.skip("Skipping onnxruntime import test - requires onnxruntime not installed")

    @pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="onnxruntime not installed")
    @patch("onnxruntime.InferenceSession")
    def test_predict_with_mock_onnx(self, mock_session_class: Mock, temp_dir: Path) -> None:
        """Test predict with mocked ONNX runtime."""
        # Create a dummy ONNX file
        model_path = temp_dir / "model.onnx"
        model_path.write_bytes(b"dummy onnx content")

        # Mock InferenceSession
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[1.5], [2.0], [2.5]])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session_class.return_value = mock_session

        predictor = ONNXPredictor(str(model_path))

        X = pl.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [4.0, 5.0, 6.0]})
        result = predictor.predict(X)

        assert isinstance(result, pl.DataFrame)
        assert "prediction" in result.columns
        assert result.height == 3
        assert mock_session.run.called

    @pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="onnxruntime not installed")
    @patch("onnxruntime.InferenceSession")
    def test_predict_empty_input(self, mock_session_class: Mock, temp_dir: Path) -> None:
        """Test predict with empty input raises error."""
        model_path = temp_dir / "model.onnx"
        model_path.write_bytes(b"dummy")

        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session_class.return_value = mock_session

        predictor = ONNXPredictor(str(model_path))

        X_empty = pl.DataFrame({"feature1": []})
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            predictor.predict(X_empty)

    @pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="onnxruntime not installed")
    @patch("onnxruntime.InferenceSession")
    def test_get_input_names(self, mock_session_class: Mock, temp_dir: Path) -> None:
        """Test getting input names."""
        model_path = temp_dir / "model.onnx"
        model_path.write_bytes(b"dummy")

        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session_class.return_value = mock_session

        predictor = ONNXPredictor(str(model_path))
        names = predictor.get_input_names()

        assert names == ["input"]

    @pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="onnxruntime not installed")
    @patch("onnxruntime.InferenceSession")
    def test_get_output_names(self, mock_session_class: Mock, temp_dir: Path) -> None:
        """Test getting output names."""
        model_path = temp_dir / "model.onnx"
        model_path.write_bytes(b"dummy")

        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session = MagicMock()
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session_class.return_value = mock_session

        predictor = ONNXPredictor(str(model_path))
        names = predictor.get_output_names()

        assert names == ["output"]

    @pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="onnxruntime not installed")
    @patch("onnxruntime.InferenceSession")
    def test_benchmark(self, mock_session_class: Mock, temp_dir: Path) -> None:
        """Test benchmark functionality."""
        model_path = temp_dir / "model.onnx"
        model_path.write_bytes(b"dummy")

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[1.5]])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session_class.return_value = mock_session

        predictor = ONNXPredictor(str(model_path))

        X = pl.DataFrame({"feature1": [1.0], "feature2": [2.0]})
        stats = predictor.benchmark(X, iterations=10)

        assert "mean_ms" in stats
        assert "median_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert "std_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert "throughput_samples_per_sec" in stats

    @pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="onnxruntime not installed")
    @patch("onnxruntime.InferenceSession")
    def test_benchmark_invalid_iterations(self, mock_session_class: Mock, temp_dir: Path) -> None:
        """Test benchmark with invalid iterations."""
        model_path = temp_dir / "model.onnx"
        model_path.write_bytes(b"dummy")

        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session_class.return_value = mock_session

        predictor = ONNXPredictor(str(model_path))

        X = pl.DataFrame({"feature1": [1.0]})
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            predictor.benchmark(X, iterations=0)


# Additional edge case tests
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_prediction_response_creation(self) -> None:
        """Test PredictionResponse creation."""
        response = PredictionResponse(
            symbol="AAPL",
            timestamp=datetime.now(),
            horizon_days=5,
            predicted_return=2.5,
            predicted_price=152.0,
            confidence=0.85,
            interval_lower=150.0,
            interval_upper=154.0,
            model_used="MockPredictor",
            model_version="1.0.0",
            features_used=45,
            latency_ms=10.0,
        )

        assert response.symbol == "AAPL"
        assert response.horizon_days == 5
        assert response.predicted_return == 2.5
        assert response.confidence == 0.85

    def test_model_info_creation(self) -> None:
        """Test ModelInfo creation."""
        info = ModelInfo(
            model_id="test-id",
            model_name="TestModel",
            model_version="1.0.0",
            created_at=datetime.now(),
            metrics={"mse": 0.025},
            tags={"type": "test"},
            is_default=False,
            file_path="/path/to/model.pkl",
        )

        assert info.model_id == "test-id"
        assert info.model_name == "TestModel"
        assert info.metrics["mse"] == 0.025

    @pytest.mark.asyncio
    async def test_prediction_service_with_confidence_error(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
        sample_ohlcv_data: pl.DataFrame,
    ) -> None:
        """Test prediction when confidence calculation fails."""
        # Create model that raises error in predict_proba
        model = ErrorPredictor()
        model_id = model_registry.register(model)
        model_registry.set_default(model_id)

        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )
        service._fetch_recent_data = AsyncMock(return_value=sample_ohlcv_data)

        result = await service.predict("AAPL", horizon_days=5, use_cache=False)

        # Should use default confidence of 0.5
        assert result.confidence == 0.5

    def test_cache_key_generation(
        self,
        model_registry: ModelRegistry,
        feature_engine: TechnicalFeatureEngine,
    ) -> None:
        """Test cache key generation."""
        service = PredictionService(
            model_registry=model_registry,
            feature_engine=feature_engine,
        )

        key1 = service._get_cache_key("AAPL", 5, None)
        key2 = service._get_cache_key("AAPL", 5, "model1")
        key3 = service._get_cache_key("GOOGL", 5, None)

        assert key1 == "AAPL:5:default"
        assert key2 == "AAPL:5:model1"
        assert key3 == "GOOGL:5:default"
        assert key1 != key2
        assert key1 != key3
