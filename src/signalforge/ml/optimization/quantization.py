"""Model quantization for optimized CPU inference.

This module provides tools for quantizing ML models to reduce size and
improve inference performance on CPU. Quantization converts floating-point
weights to lower precision (e.g., INT8) while maintaining accuracy.

Benefits of quantization:
- 4x smaller model size (FP32 to INT8)
- 2-4x faster inference on CPU
- Lower memory bandwidth usage
- Minimal accuracy loss

Key Classes:
    QuantizationConfig: Configuration for quantization
    ModelQuantizer: Quantize PyTorch models

Examples:
    Dynamic quantization (easiest):

    >>> from signalforge.ml.optimization import ModelQuantizer
    >>> import torch.nn as nn
    >>>
    >>> model = nn.Sequential(
    ...     nn.Linear(100, 50),
    ...     nn.ReLU(),
    ...     nn.Linear(50, 10)
    ... )
    >>>
    >>> quantizer = ModelQuantizer()
    >>> quantized = quantizer.quantize_dynamic(model)
    >>> quantized.save("model_quantized.pt")

    Static quantization (best performance):

    >>> import polars as pl
    >>> calibration_data = pl.DataFrame(...)  # Representative data
    >>> quantized = quantizer.quantize_static(
    ...     model,
    ...     calibration_data,
    ...     "model_quantized.pt"
    ... )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    import polars as pl
    import torch
    import torch.nn as nn

logger = get_logger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        dtype: Target quantization dtype (qint8, quint8)
        backend: Quantization backend (fbgemm, qnnpack)
        per_channel: Use per-channel quantization
        reduce_range: Reduce quantization range for compatibility
    """

    dtype: str = "qint8"
    backend: str = "fbgemm"
    per_channel: bool = True
    reduce_range: bool = False


class ModelQuantizer:
    """Quantize PyTorch models for optimized CPU inference.

    This class provides utilities for quantizing models using dynamic
    and static quantization techniques.

    Attributes:
        config: Quantization configuration
    """

    def __init__(self, config: QuantizationConfig | None = None) -> None:
        """Initialize model quantizer.

        Args:
            config: Quantization configuration (uses defaults if None)
        """
        self.config = config or QuantizationConfig()

        logger.info(
            "model_quantizer_initialized",
            dtype=self.config.dtype,
            backend=self.config.backend,
        )

    def quantize_dynamic(
        self,
        model: nn.Module,
        output_path: str | None = None,
    ) -> nn.Module:
        """Apply dynamic quantization to model.

        Dynamic quantization quantizes weights ahead of time but activations
        are quantized on-the-fly during inference. This is the easiest method
        and works well for RNN/LSTM models.

        Args:
            model: PyTorch model to quantize
            output_path: Optional path to save quantized model

        Returns:
            Quantized model

        Raises:
            ImportError: If torch is not installed
            RuntimeError: If quantization fails

        Examples:
            >>> import torch.nn as nn
            >>> model = nn.LSTM(100, 50, num_layers=2)
            >>> quantizer = ModelQuantizer()
            >>> quantized = quantizer.quantize_dynamic(model)
            >>> # Use quantized model for inference
            >>> output = quantized(input_tensor)
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for quantization. "
                "Install with: pip install torch"
            ) from e

        logger.info("starting_dynamic_quantization")

        start_time = time.perf_counter()

        try:
            # Set model to eval mode
            model.eval()

            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(  # type: ignore[attr-defined]
                model,
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8,
            )

            quantization_time = time.perf_counter() - start_time

            logger.info(
                "dynamic_quantization_completed",
                quantization_time_ms=quantization_time * 1000,
            )

            # Save if path provided
            if output_path:
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                torch.save(quantized_model.state_dict(), str(output_path_obj))
                logger.info(
                    "quantized_model_saved",
                    output_path=output_path,
                )

            return quantized_model  # type: ignore[no-any-return]

        except Exception as e:
            raise RuntimeError(f"Dynamic quantization failed: {e}") from e

    def quantize_static(
        self,
        model: nn.Module,
        calibration_data: pl.DataFrame,
        output_path: str,
    ) -> nn.Module:
        """Apply static quantization to model.

        Static quantization quantizes both weights and activations ahead of time
        using calibration data. This provides the best performance but requires
        representative calibration data.

        Args:
            model: PyTorch model to quantize
            calibration_data: Representative data for calibration
            output_path: Path to save quantized model

        Returns:
            Quantized model

        Raises:
            ImportError: If torch is not installed
            RuntimeError: If quantization fails

        Examples:
            >>> import polars as pl
            >>> calibration_data = pl.DataFrame({
            ...     "feature1": [1.0, 2.0, 3.0],
            ...     "feature2": [4.0, 5.0, 6.0]
            ... })
            >>> quantizer = ModelQuantizer()
            >>> quantized = quantizer.quantize_static(
            ...     model,
            ...     calibration_data,
            ...     "model_static_quantized.pt"
            ... )
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for quantization. "
                "Install with: pip install torch"
            ) from e

        logger.info(
            "starting_static_quantization",
            calibration_samples=calibration_data.height,
        )

        start_time = time.perf_counter()

        try:
            # Set quantization backend
            torch.backends.quantized.engine = self.config.backend

            # Set model to eval mode
            model.eval()

            # Fuse modules for better performance
            model = self._fuse_modules(model)

            # Prepare for quantization
            model.qconfig = torch.quantization.get_default_qconfig(  # type: ignore[attr-defined]
                self.config.backend
            )
            torch.quantization.prepare(model, inplace=True)  # type: ignore[attr-defined]

            # Calibrate with representative data
            self._calibrate(model, calibration_data)

            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=False)  # type: ignore[attr-defined]

            quantization_time = time.perf_counter() - start_time

            logger.info(
                "static_quantization_completed",
                quantization_time_ms=quantization_time * 1000,
            )

            # Save model
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            torch.save(quantized_model.state_dict(), str(output_path_obj))

            logger.info(
                "quantized_model_saved",
                output_path=output_path,
            )

            return quantized_model  # type: ignore[no-any-return]

        except Exception as e:
            raise RuntimeError(f"Static quantization failed: {e}") from e

    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse consecutive modules for better quantization.

        Fuses Conv-BN-ReLU and Linear-ReLU patterns.

        Args:
            model: PyTorch model

        Returns:
            Model with fused modules
        """
        try:
            import torch
        except ImportError:
            return model

        # This is a simplified version
        # In production, you'd need to specify which modules to fuse
        try:
            model = torch.quantization.fuse_modules(  # type: ignore[no-untyped-call]
                model,
                [["0", "1"]],  # Example: fuse first two layers
                inplace=True,
            )
        except Exception as e:
            logger.warning(
                "module_fusion_skipped",
                error=str(e),
            )

        return model

    def _calibrate(self, model: nn.Module, calibration_data: pl.DataFrame) -> None:
        """Calibrate quantized model with representative data.

        Args:
            model: Prepared model for quantization
            calibration_data: Calibration dataset
        """
        try:
            import torch
        except ImportError:
            return

        logger.info(
            "calibrating_model",
            num_samples=calibration_data.height,
        )

        # Convert to numpy then tensor
        data_numpy = calibration_data.to_numpy().astype(np.float32)
        data_tensor = torch.from_numpy(data_numpy)

        # Run forward passes for calibration
        with torch.no_grad():
            # Process in batches to avoid OOM
            batch_size = 32
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i : i + batch_size]
                _ = model(batch)

        logger.info("calibration_completed")

    def compare_accuracy(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_data: pl.DataFrame,
        test_labels: pl.DataFrame,
    ) -> dict[str, Any]:
        """Compare accuracy between original and quantized models.

        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            test_data: Test features
            test_labels: Test labels

        Returns:
            Dictionary with comparison metrics

        Examples:
            >>> comparison = quantizer.compare_accuracy(
            ...     original_model,
            ...     quantized_model,
            ...     test_data,
            ...     test_labels
            ... )
            >>> print(f"Accuracy loss: {comparison['accuracy_loss']:.2%}")
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch required for accuracy comparison") from e

        logger.info("comparing_model_accuracy")

        # Convert to tensors
        X = torch.from_numpy(test_data.to_numpy().astype(np.float32))
        y = torch.from_numpy(test_labels.to_numpy().astype(np.float32))

        # Get predictions
        with torch.no_grad():
            original_preds = original_model(X)
            quantized_preds = quantized_model(X)

        # Calculate MSE
        original_mse = float(
            torch.mean((original_preds - y) ** 2).item()
        )
        quantized_mse = float(
            torch.mean((quantized_preds - y) ** 2).item()
        )

        # Calculate difference
        pred_diff = float(
            torch.mean((original_preds - quantized_preds) ** 2).item()
        )

        results = {
            "original_mse": original_mse,
            "quantized_mse": quantized_mse,
            "mse_increase": quantized_mse - original_mse,
            "mse_increase_pct": (
                (quantized_mse - original_mse) / original_mse * 100
                if original_mse > 0
                else 0
            ),
            "prediction_diff_mse": pred_diff,
        }

        logger.info(
            "accuracy_comparison_completed",
            **results,
        )

        return results

    def benchmark_quantized(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        sample_input: torch.Tensor,
        iterations: int = 100,
    ) -> dict[str, Any]:
        """Benchmark performance improvement from quantization.

        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            sample_input: Input for benchmarking
            iterations: Number of iterations

        Returns:
            Dictionary with benchmark results

        Examples:
            >>> import torch
            >>> sample = torch.randn(100, 10)
            >>> results = quantizer.benchmark_quantized(
            ...     original_model,
            ...     quantized_model,
            ...     sample,
            ...     iterations=1000
            ... )
            >>> print(f"Speedup: {results['speedup']:.2f}x")
            >>> print(f"Size reduction: {results['size_reduction_pct']:.1f}%")
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch required for benchmarking") from e

        logger.info(
            "benchmarking_quantized_model",
            iterations=iterations,
        )

        # Benchmark original model
        original_model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = original_model(sample_input)

            # Benchmark
            original_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = original_model(sample_input)
                original_times.append(time.perf_counter() - start)

        original_mean = np.mean(original_times) * 1000

        # Benchmark quantized model
        quantized_model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = quantized_model(sample_input)

            # Benchmark
            quantized_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = quantized_model(sample_input)
                quantized_times.append(time.perf_counter() - start)

        quantized_mean = np.mean(quantized_times) * 1000

        # Calculate size reduction
        original_size = sum(
            p.numel() * p.element_size() for p in original_model.parameters()
        )
        quantized_size = sum(
            p.numel() * p.element_size() for p in quantized_model.parameters()
        )

        results = {
            "original_latency_ms": float(original_mean),
            "quantized_latency_ms": float(quantized_mean),
            "speedup": float(original_mean / quantized_mean),
            "latency_reduction_pct": float(
                (original_mean - quantized_mean) / original_mean * 100
            ),
            "original_size_mb": float(original_size / 1024 / 1024),
            "quantized_size_mb": float(quantized_size / 1024 / 1024),
            "size_reduction_pct": float(
                (original_size - quantized_size) / original_size * 100
            ),
        }

        logger.info(
            "quantization_benchmark_completed",
            **results,
        )

        return results


__all__ = [
    "QuantizationConfig",
    "ModelQuantizer",
]
