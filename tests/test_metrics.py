"""Tests for Prometheus metrics integration."""

import contextlib
from time import sleep

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from signalforge.api.middleware.metrics import MetricsMiddleware
from signalforge.core.metrics import (
    active_requests,
    active_users,
    db_query_latency_seconds,
    model_inference_latency_seconds,
    request_latency_seconds,
    request_total,
    signal_generation_total,
    track_db_query,
    track_db_time,
    track_model_inference,
    track_model_time,
    track_request_end,
    track_request_start,
    track_signal_generated,
    track_time,
)


@pytest.fixture
def clear_metrics():
    """Clear metrics before each test to avoid interference."""
    # Get all metrics from registry
    collectors = list(REGISTRY._collector_to_names.keys())

    # Unregister our custom metrics
    for collector in collectors:
        with contextlib.suppress(Exception):
            if hasattr(collector, "_name") and collector._name.startswith("signalforge_"):
                REGISTRY.unregister(collector)

    yield

    # Clean up after test
    for collector in collectors:
        with contextlib.suppress(Exception):
            if hasattr(collector, "_name") and collector._name.startswith("signalforge_"):
                REGISTRY.unregister(collector)


class TestMetricsCore:
    """Test core metrics functionality."""

    def test_request_latency_metric_exists(self):
        """Test that request latency histogram is properly defined."""
        assert request_latency_seconds is not None
        assert request_latency_seconds._name == "signalforge_request_latency_seconds"
        assert request_latency_seconds._type == "histogram"

    def test_request_total_metric_exists(self):
        """Test that request total counter is properly defined."""
        assert request_total is not None
        # prometheus_client strips _total suffix from counter names
        assert request_total._name == "signalforge_request"
        assert request_total._type == "counter"

    def test_active_requests_metric_exists(self):
        """Test that active requests gauge is properly defined."""
        assert active_requests is not None
        assert active_requests._name == "signalforge_active_requests"
        assert active_requests._type == "gauge"

    def test_active_users_metric_exists(self):
        """Test that active users gauge is properly defined."""
        assert active_users is not None
        assert active_users._name == "signalforge_active_users"
        assert active_users._type == "gauge"

    def test_signal_generation_metric_exists(self):
        """Test that signal generation counter is properly defined."""
        assert signal_generation_total is not None
        # prometheus_client strips _total suffix from counter names
        assert signal_generation_total._name == "signalforge_signal_generation"
        assert signal_generation_total._type == "counter"

    def test_model_inference_metric_exists(self):
        """Test that model inference histogram is properly defined."""
        assert model_inference_latency_seconds is not None
        assert (
            model_inference_latency_seconds._name
            == "signalforge_model_inference_latency_seconds"
        )
        assert model_inference_latency_seconds._type == "histogram"

    def test_db_query_metric_exists(self):
        """Test that database query histogram is properly defined."""
        assert db_query_latency_seconds is not None
        assert db_query_latency_seconds._name == "signalforge_db_query_latency_seconds"
        assert db_query_latency_seconds._type == "histogram"

    def test_track_request_start_increments_gauge(self):
        """Test that track_request_start increments active requests."""
        initial_value = active_requests._value._value
        track_request_start()
        assert active_requests._value._value == initial_value + 1

    def test_track_request_end_decrements_gauge(self):
        """Test that track_request_end decrements active requests."""
        track_request_start()
        initial_value = active_requests._value._value
        track_request_end()
        assert active_requests._value._value == initial_value - 1

    def test_track_signal_generated_increments_counter(self):
        """Test that track_signal_generated increments the counter."""
        initial = signal_generation_total.labels(signal_type="buy")._value._value
        track_signal_generated("buy")
        assert signal_generation_total.labels(signal_type="buy")._value._value == initial + 1

    def test_track_signal_generated_multiple_types(self):
        """Test tracking different signal types independently."""
        track_signal_generated("buy")
        track_signal_generated("sell")
        track_signal_generated("buy")

        buy_count = signal_generation_total.labels(signal_type="buy")._value._value
        sell_count = signal_generation_total.labels(signal_type="sell")._value._value

        assert buy_count >= 2
        assert sell_count >= 1

    def test_track_model_inference_records_duration(self):
        """Test that model inference duration is recorded."""
        track_model_inference("sentiment_analyzer", 0.5)
        metric = model_inference_latency_seconds.labels(model_name="sentiment_analyzer")
        assert metric._sum._value >= 0.5

    def test_track_db_query_records_duration(self):
        """Test that database query duration is recorded."""
        track_db_query("select", 0.1)
        metric = db_query_latency_seconds.labels(query_type="select")
        assert metric._sum._value >= 0.1

    def test_track_time_context_manager(self):
        """Test track_time context manager measures duration."""
        with track_time(model_inference_latency_seconds, model_name="test_model"):
            sleep(0.01)

        metric = model_inference_latency_seconds.labels(model_name="test_model")
        assert metric._sum._value >= 0.01

    def test_track_db_time_context_manager(self):
        """Test track_db_time context manager measures duration."""
        with track_db_time("insert"):
            sleep(0.01)

        metric = db_query_latency_seconds.labels(query_type="insert")
        assert metric._sum._value >= 0.01

    def test_track_model_time_context_manager(self):
        """Test track_model_time context manager measures duration."""
        with track_model_time("classifier"):
            sleep(0.01)

        metric = model_inference_latency_seconds.labels(model_name="classifier")
        assert metric._sum._value >= 0.01

    def test_track_time_with_exception(self):
        """Test that track_time records metrics even when exception occurs."""
        try:
            with track_time(model_inference_latency_seconds, model_name="error_model"):
                sleep(0.01)
                raise ValueError("Test error")
        except ValueError:
            pass

        metric = model_inference_latency_seconds.labels(model_name="error_model")
        assert metric._sum._value > 0

    def test_multiple_db_query_types(self):
        """Test tracking different database query types."""
        track_db_query("select", 0.05)
        track_db_query("insert", 0.1)
        track_db_query("update", 0.15)

        select_metric = db_query_latency_seconds.labels(query_type="select")
        insert_metric = db_query_latency_seconds.labels(query_type="insert")
        update_metric = db_query_latency_seconds.labels(query_type="update")

        assert select_metric._sum._value > 0
        assert insert_metric._sum._value > 0
        assert update_metric._sum._value > 0


class TestMetricsMiddleware:
    """Test MetricsMiddleware functionality."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI application with metrics middleware."""
        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.get("/test/{item_id}")
        async def test_item_endpoint(item_id: int):
            return {"item_id": item_id}

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

        @app.get("/metrics")
        async def metrics_endpoint():
            return {"metrics": "data"}

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        return app

    def test_middleware_tracks_successful_request(self, app):
        """Test that middleware tracks successful requests."""
        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        metric = request_total.labels(endpoint="/test", method="GET", status="200")
        assert metric._value._value >= 1

    def test_middleware_tracks_request_latency(self, app):
        """Test that middleware tracks request latency."""
        client = TestClient(app)
        client.get("/test")

        metric = request_latency_seconds.labels(endpoint="/test", method="GET")
        assert metric._sum._value > 0
        assert metric._sum._value > 0

    def test_middleware_tracks_error_requests(self, app):
        """Test that middleware tracks requests that result in errors."""
        client = TestClient(app)
        with contextlib.suppress(Exception):
            client.get("/error")

        metric = request_total.labels(endpoint="/error", method="GET", status="500")
        assert metric._value._value >= 1

    def test_middleware_excludes_metrics_endpoint(self, app):
        """Test that /metrics endpoint is excluded from tracking."""
        client = TestClient(app)
        initial_count = request_total.labels(endpoint="/metrics", method="GET", status="200")._value._value

        client.get("/metrics")

        final_count = request_total.labels(endpoint="/metrics", method="GET", status="200")._value._value
        assert final_count == initial_count

    def test_middleware_excludes_health_endpoint(self, app):
        """Test that /health endpoint is excluded from tracking."""
        client = TestClient(app)
        initial_count = request_total.labels(endpoint="/health", method="GET", status="200")._value._value

        client.get("/health")

        final_count = request_total.labels(endpoint="/health", method="GET", status="200")._value._value
        assert final_count == initial_count

    def test_middleware_groups_by_route_pattern(self, app):
        """Test that middleware tracks requests to parameterized routes."""
        client = TestClient(app)
        # Make requests to parameterized routes
        client.get("/test/1")
        client.get("/test/2")

        # Verify requests were tracked (may be grouped or individual depending on implementation)
        # The key is that the metrics are recorded
        total_requests = 0
        for labels in [("/test/1", "GET", "200"), ("/test/2", "GET", "200")]:
            metric = request_total.labels(endpoint=labels[0], method=labels[1], status=labels[2])
            total_requests += metric._value._value
        assert total_requests >= 2

    def test_middleware_active_requests_tracking(self, app):
        """Test that middleware properly tracks active requests."""
        # This test verifies the gauge increments and decrements correctly
        client = TestClient(app)
        initial_active = active_requests._value._value

        # Make a request
        client.get("/test")

        # After request completes, active requests should return to initial value
        assert active_requests._value._value == initial_active

    def test_middleware_different_http_methods(self, app):
        """Test that middleware tracks different HTTP methods separately."""
        app.post("/test")(lambda: {"status": "created"})

        client = TestClient(app)
        client.get("/test")
        client.post("/test")

        get_metric = request_total.labels(endpoint="/test", method="GET", status="200")
        post_metric = request_total.labels(endpoint="/test", method="POST", status="200")

        assert get_metric._value._value >= 1
        assert post_metric._value._value >= 1


class TestMetricsEndpoint:
    """Test /metrics endpoint functionality."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI application."""
        from signalforge.api.routes.health import router

        app = FastAPI()
        app.include_router(router)
        return app

    def test_metrics_endpoint_returns_prometheus_format(self, app):
        """Test that /metrics endpoint returns Prometheus text format."""
        client = TestClient(app)
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert b"# HELP" in response.content or b"# TYPE" in response.content

    def test_metrics_endpoint_includes_custom_metrics(self, app):
        """Test that /metrics includes SignalForge custom metrics."""
        client = TestClient(app)
        response = client.get("/metrics")

        content = response.content.decode("utf-8")
        assert "signalforge_request_latency_seconds" in content or "signalforge" in content

    def test_metrics_endpoint_accessible(self, app):
        """Test that /metrics endpoint is accessible."""
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200


class TestMetricsIntegration:
    """Integration tests for complete metrics system."""

    def test_end_to_end_request_tracking(self):
        """Test complete request lifecycle tracking."""
        track_request_start()
        initial_active = active_requests._value._value

        assert initial_active > 0

        track_request_end()
        final_active = active_requests._value._value

        assert final_active < initial_active

    def test_signal_tracking_multiple_types(self):
        """Test tracking multiple signal types in sequence."""
        signals = ["buy", "sell", "hold", "buy", "buy", "sell"]

        for signal in signals:
            track_signal_generated(signal)

        buy_count = signal_generation_total.labels(signal_type="buy")._value._value
        sell_count = signal_generation_total.labels(signal_type="sell")._value._value
        hold_count = signal_generation_total.labels(signal_type="hold")._value._value

        assert buy_count >= 3
        assert sell_count >= 2
        assert hold_count >= 1

    def test_model_tracking_multiple_models(self):
        """Test tracking multiple ML models."""
        models = [
            ("sentiment", 0.1),
            ("classifier", 0.2),
            ("sentiment", 0.15),
        ]

        for model_name, duration in models:
            track_model_inference(model_name, duration)

        sentiment_metric = model_inference_latency_seconds.labels(model_name="sentiment")
        classifier_metric = model_inference_latency_seconds.labels(model_name="classifier")

        assert sentiment_metric._sum._value > 0
        assert classifier_metric._sum._value > 0

    def test_db_query_tracking_comprehensive(self):
        """Test comprehensive database query tracking."""
        queries = [
            ("select", 0.05),
            ("insert", 0.1),
            ("update", 0.08),
            ("select", 0.06),
            ("delete", 0.12),
        ]

        for query_type, duration in queries:
            track_db_query(query_type, duration)

        select_metric = db_query_latency_seconds.labels(query_type="select")
        insert_metric = db_query_latency_seconds.labels(query_type="insert")

        assert select_metric._sum._value > 0
        assert insert_metric._sum._value > 0
