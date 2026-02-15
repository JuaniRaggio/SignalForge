# ML Serving & Optimization

This module provides production-ready serving capabilities for SignalForge ML models.

## Features

### Batch Prediction (`batch_predictor.py`)

Accumulates requests over a configurable time window and processes them in batches for optimal throughput.

```python
from signalforge.ml.serving import BatchPredictor
from signalforge.ml.inference import ONNXPredictor

# Initialize predictor
onnx_model = ONNXPredictor("model.onnx")
batch_predictor = BatchPredictor(
    onnx_model,
    batch_size=32,      # Max batch size
    window_ms=50        # Accumulation window
)

# Start background processing
await batch_predictor.start()

# Submit requests (automatically batched)
features = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})
result = await batch_predictor.predict(features)

# Get statistics
stats = batch_predictor.get_stats()
print(f"Avg batch size: {stats['avg_batch_size']:.1f}")
print(f"Throughput: {stats['avg_batch_time_ms']:.2f}ms per batch")
```

**Benefits:**
- 2-5x throughput improvement via vectorization
- Configurable latency/throughput tradeoff
- Automatic request distribution
- Built-in metrics

**Target Latency:** <50ms per request (including batching overhead)

### Prediction Cache (`prediction_cache.py`)

Redis-based caching with intelligent key generation and TTL management.

```python
from signalforge.ml.serving import PredictionCache
from signalforge.core.redis import get_redis

redis = await get_redis()
cache = PredictionCache(redis, default_ttl=300)

# Generate cache key with feature hashing
features = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})
cache_key = cache.generate_cache_key(
    ticker="AAPL",
    horizon=5,
    features=features,
    model_id="v1"
)

# Check cache
cached = await cache.get_cached_prediction(cache_key)
if cached:
    return cached

# Cache new prediction
await cache.cache_prediction(
    cache_key,
    {"prediction": 150.5, "confidence": 0.85},
    ttl=600
)

# Monitor performance
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate_pct']:.1f}%")
```

**Features:**
- Feature-based cache key generation
- Configurable TTL per prediction type
- Pattern-based invalidation
- Hit/miss metrics

### Model Router (`model_router.py`)

A/B testing and canary deployment support with traffic splitting.

```python
from signalforge.ml.serving import ModelRouter
from signalforge.ml.inference import ModelRegistry

registry = ModelRegistry()
router = ModelRouter(registry)

# Configure traffic split (80/20)
await router.set_traffic_split(
    "lstm_model",
    {
        "v1.0": 0.8,
        "v2.0": 0.2
    }
)

# Route request (consistent hashing)
model = await router.route_request("lstm_model", "req_123")

# Record metrics
router.record_request("lstm_model", "v2.0", latency_ms=45.0)

# Check for rollback
if router.should_rollback("lstm_model", "v2.0", error_rate_threshold=5.0):
    # Rollback to v1.0
    await router.set_traffic_split("lstm_model", {"v1.0": 1.0})

# Get performance metrics
stats = router.get_version_stats("lstm_model", "v2.0")
print(f"Error rate: {stats['error_rate_pct']:.2f}%")
print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
```

**Features:**
- Consistent hashing for deterministic routing
- Per-version metrics tracking
- Automatic rollback detection
- Shadow mode support

## Model Optimization

### ONNX Conversion (`onnx_converter.py`)

Convert PyTorch models to ONNX for 2-4x faster inference.

```python
from signalforge.ml.optimization import ONNXConverter
import torch
import torch.nn as nn

# Define model
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

# Convert to ONNX
converter = ONNXConverter(opset_version=14)
sample_input = torch.randn(1, 100)

converter.convert(
    model,
    sample_input,
    "model.onnx",
    input_names=["features"],
    output_names=["predictions"],
    dynamic_axes={
        "features": {0: "batch_size"},
        "predictions": {0: "batch_size"}
    }
)

# Benchmark improvement
results = converter.benchmark_comparison(
    model,
    "model.onnx",
    torch.randn(100, 100),
    iterations=1000
)
print(f"Speedup: {results['speedup']:.2f}x")
```

### Model Quantization (`quantization.py`)

INT8 quantization for 4x smaller models and 2-4x faster CPU inference.

```python
from signalforge.ml.optimization import ModelQuantizer
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

quantizer = ModelQuantizer()

# Dynamic quantization (easiest)
quantized = quantizer.quantize_dynamic(model, "model_quantized.pt")

# Static quantization (best performance)
calibration_data = pl.DataFrame(...)  # Representative data
quantized = quantizer.quantize_static(
    model,
    calibration_data,
    "model_static_quantized.pt"
)

# Compare accuracy and performance
results = quantizer.benchmark_quantized(
    model,
    quantized,
    torch.randn(100, 100),
    iterations=1000
)
print(f"Speedup: {results['speedup']:.2f}x")
print(f"Size reduction: {results['size_reduction_pct']:.1f}%")
```

## Enhanced Model Registry

The ModelRegistry now supports production workflows:

```python
from signalforge.ml.inference import ModelRegistry

registry = ModelRegistry("models/")

# Register model in staging
model_id = registry.register(
    model,
    metrics={"mse": 0.025, "latency_ms": 45.0},
    tags={"type": "lstm", "dataset": "sp500"},
    status="staging"
)

# Canary deployment (5% traffic)
registry.promote_to_production(model_id, traffic_percentage=0.05)

# Monitor metrics, then increase traffic
registry.set_traffic_percentage(model_id, 0.50)

# Full rollout
registry.set_traffic_percentage(model_id, 1.0)

# List production models
prod_models = registry.get_production_models()

# Archive old version
registry.archive_model(old_model_id)
```

**New Features:**
- Model status tracking (staging/production/archived)
- Traffic percentage for gradual rollouts
- ONNX path tracking
- Version listing

## Performance Targets

- **Batch Prediction:** <50ms latency per request
- **Cache Hit Rate:** >80% for repeat predictions
- **ONNX Speedup:** 2-4x vs PyTorch
- **Quantization:** 4x size reduction, 2-4x speedup

## Testing

Comprehensive test suite in `tests/test_ml_serving.py`:

```bash
pytest tests/test_ml_serving.py -v
```

Tests cover:
- Batch accumulation and processing
- Cache hit/miss scenarios
- Traffic routing consistency
- ONNX conversion validation
- Quantization accuracy
