"""ONNX Runtime integration for optimized inference.

This module provides efficient model inference using ONNX Runtime, which offers:
- Cross-platform compatibility
- Optimized execution for CPU and GPU
- Significantly faster inference than PyTorch/TensorFlow in production
- Smaller model footprint

The ONNXPredictor class wraps ONNX Runtime and provides a Polars-friendly
interface for predictions.

Key Classes:
    ONNXPredictor: ONNX Runtime wrapper for fast inference

Examples:
    Loading and using an ONNX model:

    >>> from signalforge.ml.inference import ONNXPredictor
    >>> import polars as pl
    >>>
    >>> # Load ONNX model
    >>> predictor = ONNXPredictor("models/lstm_v1.onnx")
    >>>
    >>> # Prepare input features
    >>> X = pl.DataFrame({
    ...     "rsi_14": [65.0, 45.0],
    ...     "macd": [0.5, -0.3],
    ... })
    >>>
    >>> # Run inference
    >>> predictions = predictor.predict(X)
    >>> print(predictions)

    Converting PyTorch model to ONNX:

    >>> import torch
    >>> import torch.nn as nn
    >>>
    >>> # Define your PyTorch model
    >>> model = nn.Sequential(
    ...     nn.Linear(10, 20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... )
    >>>
    >>> # Create sample input for export
    >>> sample_input = torch.randn(1, 10)
    >>>
    >>> # Convert and create predictor
    >>> predictor = ONNXPredictor.from_pytorch(
    ...     model, sample_input, "model.onnx"
    ... )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    import torch.nn as nn

logger = get_logger(__name__)


class ONNXPredictor:
    """ONNX Runtime wrapper for fast inference.

    This class provides efficient model inference using ONNX Runtime,
    with a Polars-friendly interface that matches the rest of SignalForge.

    Attributes:
        model_path: Path to the ONNX model file
        session: ONNX Runtime inference session
    """

    def __init__(self, model_path: str) -> None:
        """Initialize ONNX predictor.

        Args:
            model_path: Path to ONNX model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If ONNX Runtime initialization fails
            ImportError: If onnxruntime is not installed
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for ONNXPredictor. "
                "Install with: pip install onnxruntime"
            ) from e

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = model_path

        # Create inference session
        # Use available execution providers (CPU, CUDA, etc)
        try:
            self.session = ort.InferenceSession(
                str(model_path_obj),
                providers=["CPUExecutionProvider"],
            )
            logger.info(
                "onnx_predictor_initialized",
                model_path=model_path,
                providers=self.session.get_providers(),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX Runtime: {e}") from e

    def predict(self, X: pl.DataFrame) -> pl.DataFrame:
        """Run inference using ONNX Runtime.

        Args:
            X: Input features as Polars DataFrame

        Returns:
            DataFrame with predictions

        Raises:
            ValueError: If input shape is incompatible
            RuntimeError: If inference fails

        Examples:
            >>> X = pl.DataFrame({"feature1": [1.0, 2.0], "feature2": [3.0, 4.0]})
            >>> predictions = predictor.predict(X)
            >>> print(predictions)
        """
        if X.height == 0:
            raise ValueError("Input DataFrame is empty")

        # Convert Polars DataFrame to numpy array
        X_numpy = X.to_numpy().astype(np.float32)

        # Get input name
        input_name = self.session.get_inputs()[0].name

        # Run inference
        try:
            start_time = time.perf_counter()
            outputs = self.session.run(None, {input_name: X_numpy})
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                "onnx_inference_completed",
                num_samples=X.height,
                latency_ms=latency_ms,
            )
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {e}") from e

        # Convert output to Polars DataFrame
        # Handle different output shapes
        output_array = outputs[0]

        # Flatten if needed
        if output_array.ndim > 1 and output_array.shape[1] == 1:
            output_array = output_array.flatten()

        result = pl.DataFrame({"prediction": output_array})

        return result

    @classmethod
    def from_pytorch(
        cls,
        model: nn.Module,
        sample_input: Any,
        output_path: str,
    ) -> ONNXPredictor:
        """Convert PyTorch model to ONNX and create predictor.

        This method exports a PyTorch model to ONNX format and
        returns an ONNXPredictor instance for the exported model.

        Args:
            model: PyTorch model to convert
            sample_input: Sample input tensor for tracing
            output_path: Path where ONNX model will be saved

        Returns:
            ONNXPredictor instance for the exported model

        Raises:
            RuntimeError: If ONNX export fails
            ImportError: If PyTorch is not installed

        Examples:
            >>> import torch
            >>> import torch.nn as nn
            >>>
            >>> model = nn.Linear(10, 1)
            >>> sample_input = torch.randn(1, 10)
            >>> predictor = ONNXPredictor.from_pytorch(
            ...     model, sample_input, "model.onnx"
            ... )
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for ONNX export. "
                "Install with: pip install torch"
            ) from e

        # Set model to eval mode
        model.eval()

        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                sample_input,
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            logger.info(
                "pytorch_model_exported_to_onnx",
                output_path=output_path,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to export PyTorch model to ONNX: {e}") from e

        # Create and return predictor
        return cls(output_path)

    def get_input_names(self) -> list[str]:
        """Return model input names.

        Returns:
            List of input tensor names

        Examples:
            >>> predictor = ONNXPredictor("model.onnx")
            >>> print(predictor.get_input_names())
            ['input']
        """
        return [input_meta.name for input_meta in self.session.get_inputs()]

    def get_output_names(self) -> list[str]:
        """Return model output names.

        Returns:
            List of output tensor names

        Examples:
            >>> predictor = ONNXPredictor("model.onnx")
            >>> print(predictor.get_output_names())
            ['output']
        """
        return [output_meta.name for output_meta in self.session.get_outputs()]

    def get_input_shape(self) -> tuple[int | str, ...]:
        """Get expected input shape.

        Returns:
            Tuple representing input shape (may contain dynamic dimensions)

        Examples:
            >>> predictor = ONNXPredictor("model.onnx")
            >>> print(predictor.get_input_shape())
            ('batch_size', 10)
        """
        input_meta = self.session.get_inputs()[0]
        shape = input_meta.shape
        # Convert None to 'batch_size' for dynamic dimensions
        return tuple("batch_size" if dim is None else dim for dim in shape)

    def benchmark(
        self,
        X: pl.DataFrame,
        iterations: int = 100,
    ) -> dict[str, float]:
        """Benchmark inference latency.

        Runs multiple inference iterations and reports statistics.

        Args:
            X: Input features for benchmarking
            iterations: Number of iterations to run

        Returns:
            Dictionary with latency statistics (mean, median, min, max, std)

        Examples:
            >>> X = pl.DataFrame({"feature1": [1.0] * 100, "feature2": [2.0] * 100})
            >>> stats = predictor.benchmark(X, iterations=1000)
            >>> print(f"Mean latency: {stats['mean_ms']:.2f} ms")
            >>> print(f"P95 latency: {stats['p95_ms']:.2f} ms")
        """
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        logger.info(
            "starting_benchmark",
            num_samples=X.height,
            iterations=iterations,
        )

        latencies: list[float] = []

        # Warm up
        for _ in range(min(10, iterations // 10)):
            _ = self.predict(X)

        # Actual benchmark
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = self.predict(X)
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        latency_array = np.array(latencies)
        stats = {
            "mean_ms": float(np.mean(latency_array)),
            "median_ms": float(np.median(latency_array)),
            "min_ms": float(np.min(latency_array)),
            "max_ms": float(np.max(latency_array)),
            "std_ms": float(np.std(latency_array)),
            "p95_ms": float(np.percentile(latency_array, 95)),
            "p99_ms": float(np.percentile(latency_array, 99)),
            "throughput_samples_per_sec": (X.height * iterations) / (sum(latencies) / 1000),
        }

        logger.info(
            "benchmark_completed",
            **stats,
        )

        return stats


__all__ = ["ONNXPredictor"]
