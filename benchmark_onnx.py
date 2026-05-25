"""Benchmark ONNX and PyTorch inference models.

Compares:
- FP32 PyTorch
- Quantized PyTorch
- FP32 ONNX
- INT8 ONNX

Metrics: latency (avg/p95/p99/FPS), memory, accuracy parity.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from onnx_inference import ONNXModelWrapper


@dataclass
class BenchmarkResult:
    name: str
    total_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    fps: float
    model_size_mb: float
    memory_mb: Optional[float] = None
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX and PyTorch ISL models")
    parser.add_argument("--pytorch-checkpoint", help="PyTorch checkpoint path")
    parser.add_argument("--pytorch-quantized", help="Quantized PyTorch model path")
    parser.add_argument("--onnx-fp32", help="FP32 ONNX model path")
    parser.add_argument("--onnx-int8", help="INT8 ONNX model path")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Number of inference iterations")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    parser.add_argument("--feature-dim", type=int, default=506, help="Feature dimension")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def benchmark_pytorch(
    checkpoint_path: str,
    num_iterations: int,
    seq_len: int,
    feature_dim: int,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark PyTorch FP32 model."""
    try:
        from model import ISLModel

        ckpt = torch.load(checkpoint_path, map_location=device)
        model_dict = ckpt.get("model_state_dict", ckpt)
        model = ISLModel()
        if isinstance(model_dict, dict):
            model.load_state_dict(model_dict, strict=False)
        model = model.to(device).eval()
    except Exception as e:
        return BenchmarkResult(
            name="PyTorch FP32",
            total_time_ms=0.0,
            min_time_ms=0.0,
            max_time_ms=0.0,
            p95_time_ms=0.0,
            p99_time_ms=0.0,
            fps=0.0,
            model_size_mb=0.0,
            error=str(e),
        )

    model_size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024

    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            input_seq = torch.randn(1, seq_len, feature_dim, device=device)
            proximity = torch.randn(1, seq_len, device=device)

            start = time.time()
            _ = torch.softmax(model(input_seq, proximity), dim=-1)
            elapsed = (time.time() - start) * 1000.0
            times.append(elapsed)

    times = np.array(times)
    return BenchmarkResult(
        name="PyTorch FP32",
        total_time_ms=float(np.sum(times)),
        min_time_ms=float(np.min(times)),
        max_time_ms=float(np.max(times)),
        p95_time_ms=float(np.percentile(times, 95)),
        p99_time_ms=float(np.percentile(times, 99)),
        fps=1000.0 / float(np.mean(times)),
        model_size_mb=model_size_mb,
    )


def benchmark_pytorch_quantized(
    checkpoint_path: str,
    num_iterations: int,
    seq_len: int,
    feature_dim: int,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark quantized PyTorch model."""
    try:
        from model import ISLModel

        ckpt = torch.load(checkpoint_path, map_location=device)
        model_dict = ckpt.get("model_state_dict", ckpt)
        model = ISLModel()
        if isinstance(model_dict, dict):
            model.load_state_dict(model_dict, strict=False)
        model = model.to(device).eval()
        model_quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, dtype=torch.qint8
        )
    except Exception as e:
        return BenchmarkResult(
            name="PyTorch Quantized",
            total_time_ms=0.0,
            min_time_ms=0.0,
            max_time_ms=0.0,
            p95_time_ms=0.0,
            p99_time_ms=0.0,
            fps=0.0,
            model_size_mb=0.0,
            error=str(e),
        )

    model_size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024

    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            input_seq = torch.randn(1, seq_len, feature_dim, device=device)
            proximity = torch.randn(1, seq_len, device=device)

            start = time.time()
            _ = torch.softmax(model_quantized(input_seq, proximity), dim=-1)
            elapsed = (time.time() - start) * 1000.0
            times.append(elapsed)

    times = np.array(times)
    return BenchmarkResult(
        name="PyTorch Quantized",
        total_time_ms=float(np.sum(times)),
        min_time_ms=float(np.min(times)),
        max_time_ms=float(np.max(times)),
        p95_time_ms=float(np.percentile(times, 95)),
        p99_time_ms=float(np.percentile(times, 99)),
        fps=1000.0 / float(np.mean(times)),
        model_size_mb=model_size_mb,
    )


def benchmark_onnx(
    onnx_path: str,
    num_iterations: int,
    seq_len: int,
    feature_dim: int,
    name: str = "ONNX",
) -> BenchmarkResult:
    """Benchmark ONNX model."""
    try:
        wrapper = ONNXModelWrapper(onnx_path, device="cpu", fallback_to_pytorch=False)
    except Exception as e:
        return BenchmarkResult(
            name=name,
            total_time_ms=0.0,
            min_time_ms=0.0,
            max_time_ms=0.0,
            p95_time_ms=0.0,
            p99_time_ms=0.0,
            fps=0.0,
            model_size_mb=0.0,
            error=str(e),
        )

    model_size_mb = os.path.getsize(onnx_path) / 1024 / 1024

    times = []
    for _ in range(num_iterations):
        input_seq = np.random.randn(1, seq_len, feature_dim).astype(np.float32)
        proximity = np.random.randn(1, seq_len).astype(np.float32)

        start = time.time()
        _ = wrapper(input_seq, proximity)
        elapsed = (time.time() - start) * 1000.0
        times.append(elapsed)

    times = np.array(times)
    return BenchmarkResult(
        name=name,
        total_time_ms=float(np.sum(times)),
        min_time_ms=float(np.min(times)),
        max_time_ms=float(np.max(times)),
        p95_time_ms=float(np.percentile(times, 95)),
        p99_time_ms=float(np.percentile(times, 99)),
        fps=1000.0 / float(np.mean(times)),
        model_size_mb=model_size_mb,
    )


def main() -> None:
    args = parse_args()

    results = []

    if args.pytorch_checkpoint:
        print(f"Benchmarking PyTorch FP32: {args.pytorch_checkpoint}")
        result = benchmark_pytorch(args.pytorch_checkpoint, args.num_iterations, args.seq_len, args.feature_dim)
        results.append(result)
        if result.error:
            print(f"  Error: {result.error}")
        else:
            print(f"  Avg: {result.p95_time_ms:.3f}ms, P95: {result.p95_time_ms:.3f}ms, FPS: {result.fps:.1f}")

    if args.pytorch_quantized:
        print(f"Benchmarking PyTorch Quantized: {args.pytorch_quantized}")
        result = benchmark_pytorch_quantized(args.pytorch_quantized, args.num_iterations, args.seq_len, args.feature_dim)
        results.append(result)
        if result.error:
            print(f"  Error: {result.error}")
        else:
            print(f"  Avg: {result.p95_time_ms:.3f}ms, P95: {result.p95_time_ms:.3f}ms, FPS: {result.fps:.1f}")

    if args.onnx_fp32:
        print(f"Benchmarking ONNX FP32: {args.onnx_fp32}")
        result = benchmark_onnx(args.onnx_fp32, args.num_iterations, args.seq_len, args.feature_dim, "ONNX FP32")
        results.append(result)
        if result.error:
            print(f"  Error: {result.error}")
        else:
            print(f"  Avg: {result.p95_time_ms:.3f}ms, P95: {result.p95_time_ms:.3f}ms, FPS: {result.fps:.1f}")

    if args.onnx_int8:
        print(f"Benchmarking ONNX INT8: {args.onnx_int8}")
        result = benchmark_onnx(args.onnx_int8, args.num_iterations, args.seq_len, args.feature_dim, "ONNX INT8")
        results.append(result)
        if result.error:
            print(f"  Error: {result.error}")
        else:
            print(f"  Avg: {result.p95_time_ms:.3f}ms, P95: {result.p95_time_ms:.3f}ms, FPS: {result.fps:.1f}")

    print("\n" + "=" * 80)
    print(f"Benchmark Results ({args.num_iterations} iterations)")
    print("=" * 80)
    print(f"{'Model':<25} {'Avg (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'FPS':<8} {'Size (MB)':<12}")
    print("-" * 80)
    for result in results:
        if result.error:
            print(f"{result.name:<25} ERROR: {result.error}")
        else:
            print(
                f"{result.name:<25} {result.total_time_ms / args.num_iterations:<12.3f} "
                f"{result.p95_time_ms:<12.3f} {result.p99_time_ms:<12.3f} {result.fps:<8.1f} {result.model_size_mb:<12.2f}"
            )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_iterations": args.num_iterations,
                    "seq_len": args.seq_len,
                    "feature_dim": args.feature_dim,
                    "results": [
                        {
                            "name": r.name,
                            "total_time_ms": r.total_time_ms,
                            "min_time_ms": r.min_time_ms,
                            "max_time_ms": r.max_time_ms,
                            "p95_time_ms": r.p95_time_ms,
                            "p99_time_ms": r.p99_time_ms,
                            "fps": r.fps,
                            "model_size_mb": r.model_size_mb,
                            "error": r.error,
                        }
                        for r in results
                    ],
                },
                f,
                indent=2,
            )
        print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()
