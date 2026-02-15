"""Batch prediction service for optimized inference.

This module provides batched prediction capabilities to improve throughput
by accumulating requests over a time window and processing them together.

Benefits of batch processing:
- Higher throughput via vectorized operations
- Better GPU utilization
- Reduced per-request overhead
- Configurable latency/throughput tradeoff

Key Classes:
    BatchRequest: Individual prediction request
    BatchPredictor: Service that accumulates and batches requests

Examples:
    Basic usage:

    >>> from signalforge.ml.serving import BatchPredictor
    >>> from signalforge.ml.inference import ONNXPredictor
    >>>
    >>> predictor = ONNXPredictor("model.onnx")
    >>> batch_predictor = BatchPredictor(
    ...     predictor,
    ...     batch_size=32,
    ...     window_ms=50
    ... )
    >>>
    >>> # Requests are automatically batched
    >>> result = await batch_predictor.predict(features)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from signalforge.ml.inference.onnx_runtime import ONNXPredictor

logger = get_logger(__name__)


@dataclass
class BatchRequest:
    """Individual prediction request in a batch.

    Attributes:
        request_id: Unique identifier for this request
        features: Input features as Polars DataFrame
        timestamp: When the request was received
        future: Future that will be resolved with the prediction
    """

    request_id: str
    features: pl.DataFrame
    timestamp: float
    future: asyncio.Future[pl.DataFrame]


class BatchPredictor:
    """Batch prediction service for optimized inference.

    This service accumulates prediction requests over a configurable time window
    and processes them together in batches. This improves throughput by leveraging
    vectorized operations and reducing per-request overhead.

    Attributes:
        predictor: The underlying predictor (ONNX or other)
        batch_size: Maximum batch size
        window_ms: Time window for accumulation in milliseconds
        pending_requests: Queue of pending requests
        processing_task: Background task that processes batches
    """

    def __init__(
        self,
        predictor: ONNXPredictor,
        batch_size: int = 32,
        window_ms: int = 50,
    ) -> None:
        """Initialize batch predictor.

        Args:
            predictor: Predictor instance (ONNX or compatible)
            batch_size: Maximum batch size (default: 32)
            window_ms: Accumulation window in milliseconds (default: 50ms)

        Raises:
            ValueError: If batch_size or window_ms are invalid
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if window_ms < 1:
            raise ValueError("window_ms must be >= 1")

        self.predictor = predictor
        self.batch_size = batch_size
        self.window_ms = window_ms
        self.window_sec = window_ms / 1000.0

        self.pending_requests: list[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._processing_task: asyncio.Task[None] | None = None
        self._shutdown = False

        # Metrics
        self._total_requests = 0
        self._total_batches = 0
        self._total_batch_time = 0.0

        logger.info(
            "batch_predictor_initialized",
            batch_size=batch_size,
            window_ms=window_ms,
        )

    async def start(self) -> None:
        """Start the background batch processing task."""
        if self._processing_task is not None:
            logger.warning("batch_predictor_already_started")
            return

        self._shutdown = False
        self._processing_task = asyncio.create_task(self._process_batches())
        logger.info("batch_predictor_started")

    async def stop(self) -> None:
        """Stop the background batch processing task."""
        if self._processing_task is None:
            return

        self._shutdown = True

        # Wait for processing task to complete
        if self._processing_task is not None:
            await self._processing_task
            self._processing_task = None

        # Process remaining requests
        if self.pending_requests:
            await self._process_current_batch()

        logger.info(
            "batch_predictor_stopped",
            total_requests=self._total_requests,
            total_batches=self._total_batches,
            avg_batch_time_ms=(
                self._total_batch_time / self._total_batches * 1000
                if self._total_batches > 0
                else 0
            ),
        )

    async def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        """Submit prediction request and wait for result.

        This method is thread-safe and can be called concurrently.
        Requests are automatically batched in the background.

        Args:
            features: Input features as Polars DataFrame (single row)

        Returns:
            Prediction result as DataFrame

        Raises:
            RuntimeError: If predictor is not started or prediction fails

        Examples:
            >>> features = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})
            >>> result = await batch_predictor.predict(features)
            >>> print(result["prediction"][0])
        """
        if self._processing_task is None:
            raise RuntimeError("BatchPredictor not started. Call start() first.")

        # Create request
        request_id = f"req_{self._total_requests}"
        future: asyncio.Future[pl.DataFrame] = asyncio.Future()

        request = BatchRequest(
            request_id=request_id,
            features=features,
            timestamp=time.perf_counter(),
            future=future,
        )

        # Add to pending queue
        async with self._lock:
            self.pending_requests.append(request)
            self._total_requests += 1

        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(
                "prediction_request_failed",
                request_id=request_id,
                error=str(e),
            )
            raise

    async def _process_batches(self) -> None:
        """Background task that processes batches periodically."""
        while not self._shutdown:
            try:
                # Wait for window period
                await asyncio.sleep(self.window_sec)

                # Process current batch
                await self._process_current_batch()

            except Exception as e:
                logger.error(
                    "batch_processing_error",
                    error=str(e),
                )

    async def _process_current_batch(self) -> None:
        """Process all pending requests as batches."""
        while True:
            # Get next batch
            async with self._lock:
                if not self.pending_requests:
                    break

                # Take up to batch_size requests
                batch = self.pending_requests[: self.batch_size]
                self.pending_requests = self.pending_requests[self.batch_size :]

            if not batch:
                break

            # Process batch
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[BatchRequest]) -> None:
        """Process a single batch of requests.

        Args:
            batch: List of requests to process
        """
        if not batch:
            return

        start_time = time.perf_counter()
        batch_size = len(batch)

        try:
            # Combine all features into single DataFrame
            all_features = pl.concat([req.features for req in batch])

            # Run batch prediction
            predictions = self.predictor.predict(all_features)

            # Distribute results to individual requests
            for i, request in enumerate(batch):
                # Extract prediction for this request
                request_pred = predictions[i : i + 1]

                # Resolve future with result
                if not request.future.done():
                    request.future.set_result(request_pred)

            # Update metrics
            batch_time = time.perf_counter() - start_time
            self._total_batches += 1
            self._total_batch_time += batch_time

            # Calculate latencies
            latencies = [
                (time.perf_counter() - req.timestamp) * 1000 for req in batch
            ]
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)

            logger.debug(
                "batch_processed",
                batch_size=batch_size,
                batch_time_ms=batch_time * 1000,
                avg_latency_ms=avg_latency,
                max_latency_ms=max_latency,
            )

        except Exception as e:
            logger.error(
                "batch_prediction_failed",
                batch_size=batch_size,
                error=str(e),
            )

            # Fail all requests in batch
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(
                        RuntimeError(f"Batch prediction failed: {e}")
                    )

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary with statistics including:
            - total_requests: Total requests processed
            - total_batches: Total batches processed
            - avg_batch_size: Average batch size
            - avg_batch_time_ms: Average batch processing time
            - pending_requests: Current pending request count

        Examples:
            >>> stats = batch_predictor.get_stats()
            >>> print(f"Throughput: {stats['avg_batch_size'] / stats['avg_batch_time_ms'] * 1000:.0f} req/s")
        """
        avg_batch_size = (
            self._total_requests / self._total_batches
            if self._total_batches > 0
            else 0
        )
        avg_batch_time_ms = (
            self._total_batch_time / self._total_batches * 1000
            if self._total_batches > 0
            else 0
        )

        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "avg_batch_size": avg_batch_size,
            "avg_batch_time_ms": avg_batch_time_ms,
            "pending_requests": len(self.pending_requests),
        }


__all__ = [
    "BatchRequest",
    "BatchPredictor",
]
