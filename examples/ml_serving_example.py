"""Complete example of ML Serving & Optimization workflow.

This example demonstrates:
1. Converting PyTorch model to ONNX
2. Quantizing model for CPU inference
3. Setting up batch prediction
4. Configuring prediction caching
5. A/B testing with model router
6. Production deployment workflow
"""

import asyncio

import polars as pl
import torch
import torch.nn as nn

from signalforge.core.redis import get_redis
from signalforge.ml.inference import ModelRegistry, ONNXPredictor
from signalforge.ml.optimization import ModelQuantizer, ONNXConverter
from signalforge.ml.serving import BatchPredictor, ModelRouter, PredictionCache


# Step 1: Define and train a simple model
class SimplePredictor(nn.Module):
    """Simple LSTM-based predictor for demonstration."""

    def __init__(self, input_size: int = 50, hidden_size: int = 100):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


async def main() -> None:
    """Run complete serving workflow example."""
    print("=" * 60)
    print("SignalForge ML Serving & Optimization Example")
    print("=" * 60)

    # Initialize model
    print("\n1. Creating PyTorch model...")
    model = SimplePredictor(input_size=50, hidden_size=100)
    sample_input = torch.randn(1, 10, 50)  # (batch, seq_len, features)

    # Step 2: Convert to ONNX
    print("\n2. Converting to ONNX...")
    converter = ONNXConverter(opset_version=14)
    onnx_path = "models/predictor_v1.onnx"

    converter.convert(
        model,
        sample_input,
        onnx_path,
        input_names=["features"],
        output_names=["predictions"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
        verify=False,
    )
    print(f"   Model saved to {onnx_path}")

    # Benchmark ONNX performance
    print("\n3. Benchmarking ONNX vs PyTorch...")
    benchmark_input = torch.randn(100, 10, 50)
    results = converter.benchmark_comparison(
        model,
        onnx_path,
        benchmark_input,
        iterations=100,
    )
    print(f"   PyTorch: {results['pytorch_mean_ms']:.2f}ms")
    print(f"   ONNX: {results['onnx_mean_ms']:.2f}ms")
    print(f"   Speedup: {results['speedup']:.2f}x")

    # Step 3: Quantize model
    print("\n4. Quantizing model...")
    quantizer = ModelQuantizer()
    quantized_model = quantizer.quantize_dynamic(
        model,
        output_path="models/predictor_v1_quantized.pt",
    )
    print("   Model quantized successfully")

    # Benchmark quantization
    quant_results = quantizer.benchmark_quantized(
        model,
        quantized_model,
        torch.randn(100, 10, 50),
        iterations=100,
    )
    print(f"   Original: {quant_results['original_latency_ms']:.2f}ms")
    print(f"   Quantized: {quant_results['quantized_latency_ms']:.2f}ms")
    print(f"   Speedup: {quant_results['speedup']:.2f}x")
    print(f"   Size reduction: {quant_results['size_reduction_pct']:.1f}%")

    # Step 4: Set up model registry
    print("\n5. Setting up Model Registry...")
    registry = ModelRegistry("models/")

    # Register ONNX model
    from signalforge.ml.models.base import BasePredictor

    class DummyModel(BasePredictor):
        model_name = "LSTMPredictor"
        model_version = "1.0.0"

        def fit(self, X, y, **kwargs):
            return self

        def predict(self, X):
            return []

        def predict_proba(self, X):
            return pl.DataFrame()

    model_v1 = DummyModel()
    model_v2 = DummyModel()
    model_v2.model_version = "2.0.0"

    v1_id = registry.register(
        model_v1,
        metrics={"mse": 0.025, "latency_ms": 45.0},
        tags={"type": "lstm", "dataset": "sp500"},
        status="production",
        onnx_path=onnx_path,
    )
    registry.set_traffic_percentage(v1_id, 0.8)

    v2_id = registry.register(
        model_v2,
        metrics={"mse": 0.020, "latency_ms": 42.0},
        tags={"type": "lstm", "dataset": "sp500"},
        status="production",
        onnx_path=onnx_path,
    )
    registry.set_traffic_percentage(v2_id, 0.2)

    print(f"   Registered v1 (80% traffic): {v1_id[:8]}...")
    print(f"   Registered v2 (20% traffic): {v2_id[:8]}...")

    # Step 5: Set up model router for A/B testing
    print("\n6. Configuring Model Router for A/B testing...")
    router = ModelRouter(registry)
    await router.set_traffic_split(
        "LSTMPredictor",
        {v1_id: 0.8, v2_id: 0.2},
    )
    print("   Traffic split: 80/20 between v1 and v2")

    # Simulate routing requests
    print("\n7. Simulating request routing...")
    for i in range(10):
        request_id = f"req_{i:03d}"
        model = await router.route_request("LSTMPredictor", request_id)
        router.record_request(
            "LSTMPredictor",
            v1_id if model.model_version == "1.0.0" else v2_id,
            latency_ms=45.0 if i % 2 == 0 else 42.0,
        )

    # Show statistics
    print("\n8. A/B Test Statistics:")
    for version_id in [v1_id, v2_id]:
        stats = router.get_version_stats("LSTMPredictor", version_id)
        version = registry.models_index[version_id].model_version
        print(f"   v{version}:")
        print(f"      Requests: {stats['request_count']}")
        print(f"      Avg latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"      Error rate: {stats['error_rate_pct']:.2f}%")

    # Step 6: Set up batch predictor
    print("\n9. Setting up Batch Predictor...")
    onnx_predictor = ONNXPredictor(onnx_path)
    batch_predictor = BatchPredictor(
        onnx_predictor,
        batch_size=32,
        window_ms=50,
    )
    await batch_predictor.start()
    print("   Batch predictor started (batch_size=32, window=50ms)")

    # Simulate batch predictions
    print("\n10. Simulating batch predictions...")
    tasks = []
    for i in range(20):
        features = pl.DataFrame(
            {f"feature_{j}": [float(i + j)] for j in range(50)}
        )
        tasks.append(batch_predictor.predict(features))

    results = await asyncio.gather(*tasks)
    await batch_predictor.stop()

    stats = batch_predictor.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Total batches: {stats['total_batches']}")
    print(f"   Avg batch size: {stats['avg_batch_size']:.1f}")
    print(f"   Avg batch time: {stats['avg_batch_time_ms']:.2f}ms")

    # Step 7: Set up prediction cache
    print("\n11. Setting up Prediction Cache...")
    redis = await get_redis()
    cache = PredictionCache(redis, default_ttl=300)

    # Cache some predictions
    for i in range(5):
        features = pl.DataFrame(
            {f"feature_{j}": [float(i + j)] for j in range(50)}
        )
        cache_key = cache.generate_cache_key(
            ticker="AAPL",
            horizon=5,
            features=features,
            model_id="v1",
        )
        await cache.cache_prediction(
            cache_key,
            {"prediction": 150.0 + i, "confidence": 0.85},
            ttl=600,
        )

    # Test cache hits
    for i in range(5):
        features = pl.DataFrame(
            {f"feature_{j}": [float(i + j)] for j in range(50)}
        )
        cache_key = cache.generate_cache_key(
            ticker="AAPL",
            horizon=5,
            features=features,
            model_id="v1",
        )
        cached = await cache.get_cached_prediction(cache_key)
        if cached:
            print(f"   Cache hit: {cached}")

    cache_stats = cache.get_stats()
    print(f"\n   Cache statistics:")
    print(f"      Hits: {cache_stats['hits']}")
    print(f"      Misses: {cache_stats['misses']}")
    print(f"      Hit rate: {cache_stats['hit_rate_pct']:.1f}%")

    # Step 8: Production deployment workflow
    print("\n12. Production Deployment Workflow:")
    print("   Canary deployment (5% traffic)...")
    registry.set_traffic_percentage(v2_id, 0.05)
    registry.set_traffic_percentage(v1_id, 0.95)

    print("   Monitor metrics for 24h...")
    print("   If metrics good, increase to 50%...")
    registry.set_traffic_percentage(v2_id, 0.50)
    registry.set_traffic_percentage(v1_id, 0.50)

    print("   If metrics still good, full rollout...")
    registry.set_traffic_percentage(v2_id, 1.0)
    registry.set_traffic_percentage(v1_id, 0.0)

    print("   Archive old version...")
    registry.archive_model(v1_id)

    prod_models = registry.get_production_models()
    print(f"\n   Production models: {len(prod_models)}")
    for model_info in prod_models:
        print(f"      {model_info.model_name} v{model_info.model_version}")
        print(f"         Status: {model_info.status}")
        print(f"         Traffic: {model_info.traffic_percentage*100:.0f}%")
        print(f"         MSE: {model_info.metrics.get('mse', 0):.4f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
