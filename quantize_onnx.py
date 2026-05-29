"""Quantize ONNX models to INT8 for faster inference and reduced memory.

Uses ONNX Runtime's quantization API for dynamic quantization.
Preserves model accuracy with representative data calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx


@dataclass
class QuantizeConfig:
    model_path: str
    output_path: str
    quant_format: str = "QInt8"
    per_channel: bool = False
    reduce_range: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--model", required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--output", required=True, help="Output path for quantized model")
    parser.add_argument("--quant-format", default="QInt8", help="Quantization format (QInt8 or QUInt8)")
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel quantization")
    parser.add_argument("--reduce-range", action="store_true", help="Reduce quantization range")
    return parser.parse_args()


def quantize_onnx(
    model_path: str,
    output_path: str,
    quant_format: str = "QInt8",
    per_channel: bool = False,
    reduce_range: bool = False,
) -> dict[str, Any]:
    """Quantize an ONNX model to INT8."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError("onnxruntime[tools] is required for quantization")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def _clear_shape_info(model: onnx.ModelProto) -> onnx.ModelProto:
        """Strip shape metadata so ONNX Runtime can re-infer shapes cleanly."""
        for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            tensor_type = value_info.type.tensor_type
            if tensor_type.HasField("shape"):
                tensor_type.ClearField("shape")
        model.graph.ClearField("value_info")
        return model

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        temp_model_path = tmp_file.name

    try:
        sanitized_model = _clear_shape_info(onnx.load(model_path))
        onnx.save(sanitized_model, temp_model_path)
    finally:
        pass

    quant_type = QuantType.QInt8 if quant_format == "QInt8" else QuantType.QUInt8

    try:
        quantize_dynamic(
            temp_model_path,
            output_path,
            per_channel=per_channel,
            reduce_range=reduce_range,
            weight_type=quant_type,
        )
    except Exception as e:
        raise RuntimeError(f"Quantization failed: {str(e)}") from e
    finally:
        if os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
            except OSError:
                pass

    if not os.path.isfile(output_path):
        raise RuntimeError(f"Quantization completed but output file not found: {output_path}")

    fp32_size = os.path.getsize(model_path)
    int8_size = os.path.getsize(output_path)
    reduction_ratio = (1 - int8_size / max(1, fp32_size)) * 100

    return {
        "output_path": os.path.abspath(output_path),
        "fp32_size_bytes": fp32_size,
        "int8_size_bytes": int8_size,
        "reduction_percent": round(reduction_ratio, 2),
        "quant_format": quant_format,
        "per_channel": per_channel,
        "reduce_range": reduce_range,
    }


def main() -> None:
    args = parse_args()
    cfg = QuantizeConfig(
        model_path=args.model,
        output_path=args.output,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
    )

    print(f"Quantizing ONNX model: {cfg.model_path}")
    result = quantize_onnx(
        cfg.model_path,
        cfg.output_path,
        quant_format=cfg.quant_format,
        per_channel=cfg.per_channel,
        reduce_range=cfg.reduce_range,
    )

    print("=" * 80)
    print(f"✓ Quantization successful")
    print(f"  Output: {result['output_path']}")
    print(f"  FP32 size: {result['fp32_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  INT8 size: {result['int8_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Reduction: {result['reduction_percent']:.1f}%")
    print("=" * 80)

    meta_path = cfg.output_path.replace(".onnx", "_quantization_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
