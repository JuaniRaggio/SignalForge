"""Prometheus metrics for monitoring SignalForge application."""

from collections.abc import Iterator
from contextlib import contextmanager
from time import time

from prometheus_client import Counter, Gauge, Histogram

# Request metrics
request_latency_seconds = Histogram(
    "signalforge_request_latency_seconds",
    "Latency of HTTP requests in seconds",
    ["endpoint", "method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
)

request_total = Counter(
    "signalforge_request_total",
    "Total number of HTTP requests",
    ["endpoint", "method", "status"],
)

active_requests = Gauge(
    "signalforge_active_requests",
    "Number of active HTTP requests",
)

# User metrics
active_users = Gauge(
    "signalforge_active_users",
    "Number of active users",
)

# Signal generation metrics
signal_generation_total = Counter(
    "signalforge_signal_generation_total",
    "Total number of signals generated",
    ["signal_type"],
)

# Model inference metrics
model_inference_latency_seconds = Histogram(
    "signalforge_model_inference_latency_seconds",
    "Latency of ML model inference in seconds",
    ["model_name"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# Database query metrics
db_query_latency_seconds = Histogram(
    "signalforge_db_query_latency_seconds",
    "Latency of database queries in seconds",
    ["query_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


def track_request_start() -> None:
    """Increment the active requests counter."""
    active_requests.inc()


def track_request_end() -> None:
    """Decrement the active requests counter."""
    active_requests.dec()


def track_signal_generated(signal_type: str) -> None:
    """Track a generated signal.

    Args:
        signal_type: Type of signal generated (e.g., 'buy', 'sell', 'hold')
    """
    signal_generation_total.labels(signal_type=signal_type).inc()


def track_model_inference(model_name: str, duration: float) -> None:
    """Track ML model inference latency.

    Args:
        model_name: Name of the ML model
        duration: Duration of the inference in seconds
    """
    model_inference_latency_seconds.labels(model_name=model_name).observe(duration)


def track_db_query(query_type: str, duration: float) -> None:
    """Track database query latency.

    Args:
        query_type: Type of query (e.g., 'select', 'insert', 'update', 'delete')
        duration: Duration of the query in seconds
    """
    db_query_latency_seconds.labels(query_type=query_type).observe(duration)


@contextmanager
def track_time(metric: Histogram, **labels: str) -> Iterator[None]:
    """Context manager to track execution time of a code block.

    Args:
        metric: Prometheus Histogram metric to observe
        **labels: Labels to apply to the metric

    Example:
        with track_time(model_inference_latency_seconds, model_name="sentiment_analyzer"):
            result = model.predict(data)
    """
    start = time()
    try:
        yield
    finally:
        duration = time() - start
        metric.labels(**labels).observe(duration)


@contextmanager
def track_db_time(query_type: str) -> Iterator[None]:
    """Context manager to track database query execution time.

    Args:
        query_type: Type of query (e.g., 'select', 'insert', 'update', 'delete')

    Example:
        with track_db_time("select"):
            results = await db.execute(query)
    """
    start = time()
    try:
        yield
    finally:
        duration = time() - start
        track_db_query(query_type, duration)


@contextmanager
def track_model_time(model_name: str) -> Iterator[None]:
    """Context manager to track ML model inference time.

    Args:
        model_name: Name of the ML model

    Example:
        with track_model_time("sentiment_analyzer"):
            result = model.predict(data)
    """
    start = time()
    try:
        yield
    finally:
        duration = time() - start
        track_model_inference(model_name, duration)
