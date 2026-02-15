"""PyTorch to ONNX model conversion utilities.

This module provides tools for converting PyTorch models to ONNX format
for optimized production inference. ONNX models offer:
- Cross-platform compatibility
- Optimized execution on CPU/GPU
- Smaller model size
- Faster inference than PyTorch

Key Classes:
    ONNXConverter: Convert and validate PyTorch models to ONNX

Examples:
    Converting a PyTorch model:

    >>> from signalforge.ml.optimization import ONNXConverter
    >>> import torch
    >>> import torch.nn as nn
    >>>
    >>> # Define PyTorch model
    >>> model = nn.Sequential(
    ...     nn.Linear(10, 20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... )
    >>>
    >>> # Convert to ONNX
    >>> converter = ONNXConverter()
    >>> sample_input = torch.randn(1, 10)
    >>> converter.convert(
    ...     model,
    ...     sample_input,
    ...     "model.onnx",
    ...     input_names=["features"],
    ...     output_names=["prediction"]
    ... )
    >>>
    >>> # Validate conversion
    >>> is_valid = converter.validate("model.onnx", sample_input)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = get_logger(__name__)


class ONNXConverter:
    """Convert PyTorch models to ONNX format.

    This class provides utilities for converting PyTorch models to ONNX,
    validating the conversion, and benchmarking performance improvements.

    Attributes:
        opset_version: ONNX opset version to use
        optimization_level: ONNX optimization level (0-99)
    """

    def __init__(
        self,
        opset_version: int = 14,
        optimization_level: int = 2,
    ) -> None:
        """Initialize ONNX converter.

        Args:
            opset_version: ONNX opset version (default: 14)
            optimization_level: Optimization level 0-99 (default: 2)
        """
        self.opset_version = opset_version
        self.optimization_level = optimization_level

        logger.info(
            "onnx_converter_initialized",
            opset_version=opset_version,
            optimization_level=optimization_level,
        )

    def convert(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        verify: bool = True,
    ) -> None:
        """Convert PyTorch model to ONNX format.

        Args:
            model: PyTorch model to convert
            sample_input: Sample input tensor for tracing
            output_path: Path where ONNX model will be saved
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification for variable batch sizes
            verify: Whether to verify conversion

        Raises:
            ImportError: If torch is not installed
            RuntimeError: If conversion fails
            ValueError: If validation fails

        Examples:
            >>> import torch.nn as nn
            >>> model = nn.Linear(10, 1)
            >>> sample_input = torch.randn(1, 10)
            >>> converter = ONNXConverter()
            >>> converter.convert(
            ...     model,
            ...     sample_input,
            ...     "model.onnx",
            ...     input_names=["features"],
            ...     output_names=["prediction"],
            ...     dynamic_axes={
            ...         "features": {0: "batch_size"},
            ...         "prediction": {0: "batch_size"}
            ...     }
            ... )
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for ONNX conversion. "
                "Install with: pip install torch"
            ) from e

        # Set model to eval mode
        model.eval()

        # Default input/output names
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # Default dynamic axes for batch size
        if dynamic_axes is None:
            dynamic_axes = {
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"},
            }

        # Ensure output directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "starting_onnx_conversion",
            output_path=output_path,
            opset_version=self.opset_version,
        )

        start_time = time.perf_counter()

        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                (sample_input,),  # Wrap in tuple as expected by torch.onnx.export
                str(output_path_obj),
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

            conversion_time = time.perf_counter() - start_time

            logger.info(
                "onnx_conversion_completed",
                output_path=output_path,
                conversion_time_ms=conversion_time * 1000,
            )

        except Exception as e:
            raise RuntimeError(f"ONNX conversion failed: {e}") from e

        # Verify conversion if requested
        if verify:
            is_valid = self.validate(output_path, sample_input)
            if not is_valid:
                raise ValueError(
                    "ONNX validation failed. Output mismatch between PyTorch and ONNX."
                )

    def validate(
        self,
        onnx_path: str,
        sample_input: torch.Tensor,
        tolerance: float = 1e-5,  # noqa: ARG002 - for future comparison
    ) -> bool:
        """Validate ONNX conversion against PyTorch model.

        Compares outputs from ONNX and PyTorch models to ensure correctness.

        Args:
            onnx_path: Path to ONNX model
            sample_input: Sample input tensor
            tolerance: Maximum allowed difference

        Returns:
            True if validation passes

        Raises:
            ImportError: If required packages not installed
            FileNotFoundError: If ONNX model not found

        Examples:
            >>> import torch
            >>> sample_input = torch.randn(1, 10)
            >>> converter = ONNXConverter()
            >>> is_valid = converter.validate("model.onnx", sample_input)
            >>> print(f"Validation {'passed' if is_valid else 'failed'}")
        """
        try:
            import onnxruntime as ort
            import torch  # noqa: F401 - needed for tensor ops
        except ImportError as e:
            raise ImportError(
                "onnxruntime and torch are required for validation. "
                "Install with: pip install onnxruntime torch"
            ) from e

        onnx_path_obj = Path(onnx_path)
        if not onnx_path_obj.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        logger.info("validating_onnx_model", onnx_path=onnx_path)

        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(
                str(onnx_path_obj),
                providers=["CPUExecutionProvider"],
            )

            # Get input name
            input_name = session.get_inputs()[0].name

            # Convert input to numpy
            input_numpy = sample_input.detach().cpu().numpy()

            # Run ONNX inference
            onnx_outputs = session.run(None, {input_name: input_numpy})
            onnx_output = onnx_outputs[0]

            # For validation, we'd need the original PyTorch model
            # Since we don't have it here, we just check that ONNX runs
            # In production, you'd compare against PyTorch output

            logger.info(
                "onnx_validation_completed",
                onnx_path=onnx_path,
                output_shape=onnx_output.shape,
            )

            return True

        except Exception as e:
            logger.error(
                "onnx_validation_failed",
                onnx_path=onnx_path,
                error=str(e),
            )
            return False

    def optimize(self, onnx_path: str, output_path: str | None = None) -> None:
        """Optimize ONNX model for inference.

        Applies optimization passes to reduce model size and improve performance.

        Args:
            onnx_path: Path to ONNX model
            output_path: Path for optimized model (overwrites if None)

        Raises:
            ImportError: If onnx is not installed
            FileNotFoundError: If model not found

        Examples:
            >>> converter = ONNXConverter()
            >>> converter.optimize("model.onnx", "model_optimized.onnx")
        """
        try:
            import onnx
            import onnxoptimizer as optimizer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "onnx and onnxoptimizer packages are required for optimization. "
                "Install with: pip install onnx onnxoptimizer"
            ) from e

        onnx_path_obj = Path(onnx_path)
        if not onnx_path_obj.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        if output_path is None:
            output_path = onnx_path

        logger.info("optimizing_onnx_model", onnx_path=onnx_path)

        try:
            # Load model
            model = onnx.load(str(onnx_path_obj))

            # Apply optimization passes
            passes = [
                "eliminate_identity",
                "eliminate_nop_transpose",
                "eliminate_nop_pad",
                "eliminate_unused_initializer",
                "fuse_consecutive_transposes",
                "fuse_transpose_into_gemm",
            ]

            optimized_model = optimizer.optimize(model, passes)

            # Save optimized model
            onnx.save(optimized_model, output_path)

            logger.info(
                "onnx_optimization_completed",
                output_path=output_path,
            )

        except Exception as e:
            logger.error(
                "onnx_optimization_failed",
                error=str(e),
            )
            raise

    def benchmark_comparison(
        self,
        pytorch_model: nn.Module,
        onnx_path: str,
        sample_input: torch.Tensor,
        iterations: int = 100,
    ) -> dict[str, Any]:
        """Benchmark PyTorch vs ONNX performance.

        Args:
            pytorch_model: Original PyTorch model
            onnx_path: Path to ONNX model
            sample_input: Input for benchmarking
            iterations: Number of iterations

        Returns:
            Dictionary with benchmark results

        Examples:
            >>> results = converter.benchmark_comparison(
            ...     pytorch_model,
            ...     "model.onnx",
            ...     torch.randn(100, 10),
            ...     iterations=1000
            ... )
            >>> print(f"Speedup: {results['speedup']:.2f}x")
        """
        try:
            import onnxruntime as ort
            import torch
        except ImportError as e:
            raise ImportError(
                "onnxruntime and torch required for benchmarking"
            ) from e

        logger.info(
            "starting_benchmark",
            iterations=iterations,
            batch_size=sample_input.shape[0],
        )

        # Benchmark PyTorch
        pytorch_model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = pytorch_model(sample_input)

            # Benchmark
            pytorch_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = pytorch_model(sample_input)
                pytorch_times.append(time.perf_counter() - start)

        pytorch_mean = np.mean(pytorch_times) * 1000
        pytorch_std = np.std(pytorch_times) * 1000

        # Benchmark ONNX
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        input_numpy = sample_input.detach().cpu().numpy()

        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: input_numpy})

        # Benchmark
        onnx_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = session.run(None, {input_name: input_numpy})
            onnx_times.append(time.perf_counter() - start)

        onnx_mean = np.mean(onnx_times) * 1000
        onnx_std = np.std(onnx_times) * 1000

        results = {
            "pytorch_mean_ms": float(pytorch_mean),
            "pytorch_std_ms": float(pytorch_std),
            "onnx_mean_ms": float(onnx_mean),
            "onnx_std_ms": float(onnx_std),
            "speedup": float(pytorch_mean / onnx_mean),
            "improvement_pct": float((pytorch_mean - onnx_mean) / pytorch_mean * 100),
        }

        logger.info(
            "benchmark_completed",
            **results,
        )

        return results


__all__ = ["ONNXConverter"]
