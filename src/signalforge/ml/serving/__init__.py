"""ML Serving module for production inference.

This module provides optimized serving capabilities for ML models including:
- Batch prediction with request accumulation
- Redis-based prediction caching
- A/B testing and traffic splitting
- Model routing and canary deployments

Key Components:
    BatchPredictor: Accumulates requests and processes in batches
    PredictionCache: Redis-based caching for predictions
    ModelRouter: A/B testing and traffic splitting

Examples:
    Batch prediction:

    >>> from signalforge.ml.serving import BatchPredictor
    >>> predictor = BatchPredictor(model, batch_size=32, window_ms=50)
    >>> result = await predictor.predict(features)

    Prediction caching:

    >>> from signalforge.ml.serving import PredictionCache
    >>> cache = PredictionCache(redis_client, default_ttl=300)
    >>> cached = await cache.get_cached_prediction(cache_key)

    A/B testing:

    >>> from signalforge.ml.serving import ModelRouter
    >>> router = ModelRouter(registry)
    >>> await router.set_traffic_split("model_v1", {"v1": 0.8, "v2": 0.2})
"""

from signalforge.ml.serving.batch_predictor import BatchPredictor, BatchRequest
from signalforge.ml.serving.model_router import ModelRouter, TrafficConfig
from signalforge.ml.serving.prediction_cache import PredictionCache

__all__ = [
    "BatchPredictor",
    "BatchRequest",
    "PredictionCache",
    "ModelRouter",
    "TrafficConfig",
]
