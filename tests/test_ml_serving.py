"""Comprehensive tests for ML serving and optimization modules.

Tests cover:
- Batch prediction with request accumulation
- Redis-based prediction caching
- Model routing and A/B testing
- ONNX conversion and validation
- Model quantization
- Model registry enhancements
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import polars as pl
import pytest

from signalforge.ml.inference.model_registry import ModelInfo, ModelRegistry
from signalforge.ml.models.base import BasePredictor, PredictionResult
from signalforge.ml.optimization.onnx_converter import ONNXConverter
from signalforge.ml.optimization.quantization import ModelQuantizer, QuantizationConfig
from signalforge.ml.serving.batch_predictor import BatchPredictor, BatchRequest
from signalforge.ml.serving.model_router import ModelRouter, TrafficConfig, VersionStats
from signalforge.ml.serving.prediction_cache import PredictionCache


# Mock models for testing
class MockPredictor(BasePredictor):
    """Mock predictor for testing."""

    model_name = "MockModel"
    model_version = "1.0.0"

    def fit(self, X: pl.DataFrame, y: pl.Series, **kwargs) -> BasePredictor:
        return self

    def predict(self, X: pl.DataFrame) -> list[PredictionResult]:
        results = []
        for i in range(X.height):
            results.append(
                PredictionResult(
                    symbol="TEST",
                    timestamp=pl.datetime(2024, 1, 1),
                    horizon_days=5,
                    prediction=100.0 + i,
                    confidence=0.85,
                    lower_bound=95.0,
                    upper_bound=105.0,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return results

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame({"confidence": [0.85] * X.height})


# Batch Predictor Tests
class TestBatchPredictor:
    """Tests for BatchPredictor."""

    @pytest.fixture
    def mock_onnx_predictor(self):
        """Create mock ONNX predictor."""
        predictor = Mock()
        predictor.predict = Mock(
            return_value=pl.DataFrame({"prediction": [100.0, 101.0, 102.0]})
        )
        return predictor

    @pytest.fixture
    async def batch_predictor(self, mock_onnx_predictor):
        """Create batch predictor instance."""
        predictor = BatchPredictor(
            mock_onnx_predictor,
            batch_size=32,
            window_ms=50,
        )
        await predictor.start()
        yield predictor
        await predictor.stop()

    def test_batch_predictor_init(self, mock_onnx_predictor):
        """Test batch predictor initialization."""
        predictor = BatchPredictor(
            mock_onnx_predictor,
            batch_size=16,
            window_ms=100,
        )

        assert predictor.batch_size == 16
        assert predictor.window_ms == 100
        assert predictor.window_sec == 0.1

    def test_batch_predictor_invalid_params(self, mock_onnx_predictor):
        """Test validation of initialization parameters."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            BatchPredictor(mock_onnx_predictor, batch_size=0)

        with pytest.raises(ValueError, match="window_ms must be >= 1"):
            BatchPredictor(mock_onnx_predictor, window_ms=0)

    @pytest.mark.asyncio
    async def test_batch_prediction(self, batch_predictor):
        """Test single prediction request."""
        features = pl.DataFrame({"feature1": [1.0], "feature2": [2.0]})

        result = await batch_predictor.predict(features)

        assert isinstance(result, pl.DataFrame)
        assert "prediction" in result.columns
        assert result.height == 1

    @pytest.mark.asyncio
    async def test_batch_accumulation(self, batch_predictor):
        """Test that multiple requests are batched together."""
        features = pl.DataFrame({"feature1": [1.0], "feature2": [2.0]})

        # Submit multiple requests concurrently
        tasks = [batch_predictor.predict(features) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, pl.DataFrame)

    @pytest.mark.asyncio
    async def test_batch_stats(self, batch_predictor):
        """Test statistics collection."""
        features = pl.DataFrame({"feature1": [1.0], "feature2": [2.0]})

        await batch_predictor.predict(features)
        await asyncio.sleep(0.1)  # Wait for processing

        stats = batch_predictor.get_stats()

        assert "total_requests" in stats
        assert "total_batches" in stats
        assert stats["total_requests"] > 0


# Prediction Cache Tests
class TestPredictionCache:
    """Tests for PredictionCache."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.setex = AsyncMock()
        redis.delete = AsyncMock()
        redis.scan_iter = AsyncMock(return_value=iter([]))
        return redis

    @pytest.fixture
    def cache(self, mock_redis):
        """Create prediction cache instance."""
        return PredictionCache(mock_redis, default_ttl=300)

    def test_cache_init(self, mock_redis):
        """Test cache initialization."""
        cache = PredictionCache(mock_redis, default_ttl=600, key_prefix="test:")

        assert cache.default_ttl == 600
        assert cache.key_prefix == "test:"

    def test_cache_invalid_ttl(self, mock_redis):
        """Test validation of TTL parameter."""
        with pytest.raises(ValueError, match="default_ttl must be >= 1"):
            PredictionCache(mock_redis, default_ttl=0)

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache, mock_redis):
        """Test cache miss scenario."""
        mock_redis.get.return_value = None

        result = await cache.get_cached_prediction("AAPL:5")

        assert result is None
        assert cache._cache_misses == 1

    @pytest.mark.asyncio
    async def test_cache_hit(self, cache, mock_redis):
        """Test cache hit scenario."""
        cached_data = {"prediction": 150.5, "confidence": 0.85}
        mock_redis.get.return_value = json.dumps(cached_data)

        result = await cache.get_cached_prediction("AAPL:5")

        assert result == cached_data
        assert cache._cache_hits == 1

    @pytest.mark.asyncio
    async def test_cache_set(self, cache, mock_redis):
        """Test caching a prediction."""
        prediction_data = {"prediction": 150.5, "confidence": 0.85}

        await cache.cache_prediction("AAPL:5", prediction_data, ttl=600)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "pred:AAPL:5"
        assert call_args[1] == 600

    def test_generate_cache_key(self, cache):
        """Test cache key generation."""
        key = cache.generate_cache_key("AAPL", 5, model_id="v1")

        assert key == "AAPL:5:v1"

    def test_generate_cache_key_with_features(self, cache):
        """Test cache key generation with feature hashing."""
        features = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})

        key = cache.generate_cache_key("AAPL", 5, features=features)

        assert "AAPL:5:" in key
        # Key should include hash
        parts = key.split(":")
        assert len(parts) == 3

    def test_feature_hashing(self, cache):
        """Test feature hash consistency."""
        features1 = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})
        features2 = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})

        hash1 = cache._hash_features(features1)
        hash2 = cache._hash_features(features2)

        assert hash1 == hash2

    @pytest.mark.asyncio
    async def test_invalidate(self, cache, mock_redis):
        """Test cache invalidation."""
        await cache.invalidate("AAPL:5")

        mock_redis.delete.assert_called_once_with("pred:AAPL:5")

    def test_cache_hit_rate(self, cache):
        """Test hit rate calculation."""
        cache._cache_hits = 8
        cache._cache_misses = 2

        hit_rate = cache.get_hit_rate()

        assert hit_rate == 80.0

    def test_cache_stats(self, cache):
        """Test statistics retrieval."""
        cache._cache_hits = 10
        cache._cache_misses = 5

        stats = cache.get_stats()

        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["total_requests"] == 15
        assert stats["hit_rate_pct"] == pytest.approx(66.67, rel=0.1)


# Model Router Tests
class TestModelRouter:
    """Tests for ModelRouter."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock model registry."""
        registry = Mock()
        registry.get = Mock(return_value=MockPredictor())
        return registry

    @pytest.fixture
    def router(self, mock_registry):
        """Create model router instance."""
        return ModelRouter(mock_registry)

    def test_router_init(self, mock_registry):
        """Test router initialization."""
        router = ModelRouter(mock_registry)

        assert router.registry == mock_registry
        assert len(router.traffic_configs) == 0

    @pytest.mark.asyncio
    async def test_set_traffic_split(self, router):
        """Test setting traffic split configuration."""
        await router.set_traffic_split(
            "test_model",
            {"v1": 0.7, "v2": 0.3},
        )

        assert "test_model" in router.traffic_configs
        config = router.traffic_configs["test_model"]
        assert config.version_weights == {"v1": 0.7, "v2": 0.3}

    @pytest.mark.asyncio
    async def test_traffic_split_validation(self, router):
        """Test validation of traffic weights."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            await router.set_traffic_split(
                "test_model",
                {"v1": 0.5, "v2": 0.3},  # Sum is 0.8
            )

    @pytest.mark.asyncio
    async def test_route_request(self, router):
        """Test request routing."""
        await router.set_traffic_split(
            "test_model",
            {"v1": 0.8, "v2": 0.2},
        )

        model = await router.route_request("test_model", "req_123")

        assert isinstance(model, MockPredictor)

    @pytest.mark.asyncio
    async def test_consistent_hashing(self, router):
        """Test that same request_id routes to same version."""
        await router.set_traffic_split(
            "test_model",
            {"v1": 0.5, "v2": 0.5},
        )

        # Same request should route to same version
        version1 = router._select_version("req_123", {"v1": 0.5, "v2": 0.5})
        version2 = router._select_version("req_123", {"v1": 0.5, "v2": 0.5})

        assert version1 == version2

    def test_record_request(self, router):
        """Test recording request metrics."""
        router.record_request("test_model", "v1", latency_ms=50.0)

        stats = router.get_version_stats("test_model", "v1")
        assert stats["request_count"] == 1
        assert stats["avg_latency_ms"] == 50.0

    def test_record_request_with_error(self, router):
        """Test recording failed request."""
        router.record_request(
            "test_model",
            "v1",
            latency_ms=100.0,
            error="Test error",
        )

        stats = router.get_version_stats("test_model", "v1")
        assert stats["error_count"] == 1
        assert stats["last_error"] == "Test error"

    def test_version_stats(self, router):
        """Test version statistics calculation."""
        # Record multiple requests
        router.record_request("test_model", "v1", latency_ms=50.0)
        router.record_request("test_model", "v1", latency_ms=70.0)
        router.record_request("test_model", "v1", latency_ms=60.0)

        stats = router.get_version_stats("test_model", "v1")

        assert stats["request_count"] == 3
        assert stats["avg_latency_ms"] == 60.0

    def test_should_rollback_insufficient_data(self, router):
        """Test rollback decision with insufficient data."""
        router.record_request("test_model", "v1", latency_ms=50.0)

        should_rollback = router.should_rollback(
            "test_model",
            "v1",
            min_requests=100,
        )

        assert not should_rollback

    def test_should_rollback_high_error_rate(self, router):
        """Test rollback decision with high error rate."""
        # Record requests with high error rate
        for _ in range(50):
            router.record_request("test_model", "v1", latency_ms=50.0)
        for _ in range(50):
            router.record_request(
                "test_model",
                "v1",
                latency_ms=50.0,
                error="Test error",
            )

        should_rollback = router.should_rollback(
            "test_model",
            "v1",
            error_rate_threshold=5.0,
            min_requests=100,
        )

        assert should_rollback


# Model Registry Enhancement Tests
class TestModelRegistryEnhancements:
    """Tests for enhanced ModelRegistry functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def registry(self, temp_storage):
        """Create model registry instance."""
        return ModelRegistry(temp_storage)

    def test_register_with_status(self, registry):
        """Test registering model with status."""
        model = MockPredictor()

        model_id = registry.register(
            model,
            metrics={"mse": 0.025},
            status="staging",
        )

        info = registry.models_index[model_id]
        assert info.status == "staging"
        assert info.traffic_percentage == 0.0

    def test_promote_to_production(self, registry):
        """Test promoting model to production."""
        model = MockPredictor()
        model_id = registry.register(model, status="staging")

        registry.promote_to_production(model_id, traffic_percentage=0.1)

        info = registry.models_index[model_id]
        assert info.status == "production"
        assert info.traffic_percentage == 0.1

    def test_set_traffic_percentage(self, registry):
        """Test setting traffic percentage."""
        model = MockPredictor()
        model_id = registry.register(model, status="production")

        registry.set_traffic_percentage(model_id, 0.5)

        info = registry.models_index[model_id]
        assert info.traffic_percentage == 0.5

    def test_archive_model(self, registry):
        """Test archiving a model."""
        model = MockPredictor()
        model_id = registry.register(model, status="production")
        registry.set_default(model_id)

        registry.archive_model(model_id)

        info = registry.models_index[model_id]
        assert info.status == "archived"
        assert info.traffic_percentage == 0.0
        assert not info.is_default

    def test_get_production_models(self, registry):
        """Test retrieving production models."""
        model1 = MockPredictor()
        model2 = MockPredictor()

        id1 = registry.register(model1, status="staging")
        id2 = registry.register(model2, status="production")

        prod_models = registry.get_production_models()

        assert len(prod_models) == 1
        assert prod_models[0].model_id == id2

    def test_list_versions(self, registry):
        """Test listing model versions."""
        model1 = MockPredictor()
        model1.model_version = "1.0.0"
        model2 = MockPredictor()
        model2.model_version = "2.0.0"

        registry.register(model1)
        registry.register(model2)

        versions = registry.list_versions("MockModel")

        assert len(versions) == 2
        # Should be sorted newest first
        assert versions[0].model_version == "2.0.0"


# ONNX Converter Tests
class TestONNXConverter:
    """Tests for ONNXConverter."""

    @pytest.fixture
    def converter(self):
        """Create ONNX converter instance."""
        return ONNXConverter(opset_version=14)

    def test_converter_init(self):
        """Test converter initialization."""
        converter = ONNXConverter(opset_version=15, optimization_level=3)

        assert converter.opset_version == 15
        assert converter.optimization_level == 3

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None),
        reason="PyTorch not installed",
    )
    def test_convert_pytorch_model(self, converter):
        """Test converting PyTorch model to ONNX."""
        import torch
        import torch.nn as nn

        model = nn.Linear(10, 1)
        sample_input = torch.randn(1, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        try:
            converter.convert(
                model,
                sample_input,
                output_path,
                verify=False,  # Skip verification in test
            )

            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)


# Quantization Tests
class TestModelQuantizer:
    """Tests for ModelQuantizer."""

    @pytest.fixture
    def quantizer(self):
        """Create model quantizer instance."""
        config = QuantizationConfig(dtype="qint8", backend="fbgemm")
        return ModelQuantizer(config)

    def test_quantizer_init(self):
        """Test quantizer initialization."""
        config = QuantizationConfig(dtype="qint8", per_channel=True)
        quantizer = ModelQuantizer(config)

        assert quantizer.config.dtype == "qint8"
        assert quantizer.config.per_channel

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None),
        reason="PyTorch not installed",
    )
    def test_dynamic_quantization(self, quantizer):
        """Test dynamic quantization."""
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

        quantized = quantizer.quantize_dynamic(model)

        assert quantized is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
