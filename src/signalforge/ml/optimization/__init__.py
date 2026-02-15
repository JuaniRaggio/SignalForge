"""ML Model Optimization module.

This module provides tools for model optimization and conversion:
- PyTorch to ONNX conversion
- Model quantization for CPU inference
- Validation and benchmarking

Key Components:
    ONNXConverter: Convert PyTorch models to ONNX
    ModelQuantizer: INT8 quantization for CPU inference

Examples:
    Converting PyTorch model to ONNX:

    >>> from signalforge.ml.optimization import ONNXConverter
    >>> converter = ONNXConverter()
    >>> converter.convert(pytorch_model, sample_input, "model.onnx")

    Quantizing model:

    >>> from signalforge.ml.optimization import ModelQuantizer
    >>> quantizer = ModelQuantizer()
    >>> quantized_model = quantizer.quantize_dynamic(model)
"""

from signalforge.ml.optimization.onnx_converter import ONNXConverter
from signalforge.ml.optimization.quantization import ModelQuantizer, QuantizationConfig

__all__ = [
    "ONNXConverter",
    "ModelQuantizer",
    "QuantizationConfig",
]
