"""Quantize a trained ISL checkpoint for CPU inference.

This script applies PyTorch dynamic quantization to Linear/GRU/LSTM layers,
saves a reloadable quantized bundle, and prints size/latency comparisons.
It can also batch-quantize an entire ensemble directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from quantization_utils import (
    benchmark_model_latency,
    build_model_from_checkpoint,
    load_checkpoint,
    load_model_artifact,
    model_size_mb,
    quantize_sign_language_model,
    save_quantized_bundle,
    set_quantized_engine,
)


def _load_sample(sample_path: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    if sample_path:
        sample = np.load(sample_path).astype(np.float32)
        proximity = sample[:, -1] if sample.shape[1] > 0 else None
        return sample, proximity

    from config import get_config

    cfg = get_config()
    sample = np.zeros((cfg.preprocessing.num_frames, cfg.frame_features.input_sequence_dim), dtype=np.float32)
    proximity = None
    if cfg.spatial.proximity_dim > 0:
        proximity = sample[:, cfg.frame_features.proximity_index]
    return sample, proximity


def _default_output_path(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)
    return str(path.with_name(f"{path.stem}_quantized.pt"))


def _quantize_single_checkpoint(checkpoint_path: str, output_path: str, sample: np.ndarray, proximity: np.ndarray | None, warmup_iters: int, benchmark_iters: int) -> dict:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("quantized") is True:
        raise ValueError(f"Checkpoint is already quantized: {checkpoint_path}")

    fp32_model = build_model_from_checkpoint(checkpoint)
    fp32_size_mb = model_size_mb(fp32_model)
    fp32_benchmark = benchmark_model_latency(
        fp32_model,
        sample,
        proximity=proximity,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
    )

    quantized_model = quantize_sign_language_model(fp32_model, inplace=False)
    quantized_size_mb = model_size_mb(quantized_model)
    quantized_benchmark = benchmark_model_latency(
        quantized_model,
        sample,
        proximity=proximity,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
    )

    model_kwargs = {
        "hidden_size": int(getattr(fp32_model, "hidden_size", 0) or fp32_model.input_proj[0].out_features),
        "num_layers": int(getattr(fp32_model, "num_layers", 1)),
        "bidirectional": bool(getattr(fp32_model, "bidirectional", True)),
        "dropout": float(getattr(fp32_model, "dropout_rate", 0.3)),
    }

    bundle = save_quantized_bundle(
        output_path,
        quantized_model=quantized_model,
        num_classes=int(checkpoint["num_classes"]),
        classes=checkpoint.get("classes"),
        source_checkpoint=os.path.abspath(checkpoint_path),
        model_kwargs=model_kwargs,
        metadata={
            "warmup_iters": warmup_iters,
            "benchmark_iters": benchmark_iters,
            "fp32_size_mb": fp32_size_mb,
            "quantized_size_mb": quantized_size_mb,
            "fp32_avg_ms": fp32_benchmark.avg_ms,
            "quantized_avg_ms": quantized_benchmark.avg_ms,
        },
    )

    reloaded_model, _, _, _, _ = load_model_artifact(output_path, map_location="cpu")
    reloaded_benchmark = benchmark_model_latency(
        reloaded_model,
        sample,
        proximity=proximity,
        warmup_iters=max(5, warmup_iters // 2),
        benchmark_iters=max(10, benchmark_iters // 2),
    )

    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "output": os.path.abspath(output_path),
        "fp32": {
            "model_size_mb": round(fp32_size_mb, 4),
            "avg_ms": round(fp32_benchmark.avg_ms, 4),
            "fps": round(fp32_benchmark.fps, 4),
        },
        "quantized": {
            "model_size_mb": round(quantized_size_mb, 4),
            "avg_ms": round(quantized_benchmark.avg_ms, 4),
            "fps": round(quantized_benchmark.fps, 4),
        },
        "reloaded_quantized": {
            "avg_ms": round(reloaded_benchmark.avg_ms, 4),
            "fps": round(reloaded_benchmark.fps, 4),
        },
        "bundle_keys": sorted(bundle.keys()),
        "size_reduction_pct": round(((fp32_size_mb - quantized_size_mb) / fp32_size_mb * 100.0) if fp32_size_mb > 0 else 0.0, 2),
        "latency_reduction_pct": round(((fp32_benchmark.avg_ms - quantized_benchmark.avg_ms) / fp32_benchmark.avg_ms * 100.0) if fp32_benchmark.avg_ms > 0 else 0.0, 2),
    }


def _batch_quantize_ensemble(ensemble_dir: str, output_dir: str, sample: np.ndarray, proximity: np.ndarray | None, warmup_iters: int, benchmark_iters: int) -> list[dict]:
    results: list[dict] = []
    os.makedirs(output_dir, exist_ok=True)
    for entry in sorted(os.listdir(ensemble_dir)):
        if not entry.endswith(".pth"):
            continue
        input_path = os.path.join(ensemble_dir, entry)
        output_path = os.path.join(output_dir, f"{Path(entry).stem}_quantized.pt")
        try:
            results.append(
                _quantize_single_checkpoint(
                    input_path,
                    output_path,
                    sample,
                    proximity,
                    warmup_iters,
                    benchmark_iters,
                )
            )
        except Exception as exc:
            results.append({
                "checkpoint": os.path.abspath(input_path),
                "output": os.path.abspath(output_path),
                "error": str(exc),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic quantization for the ISL CPU pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Path to a fp32 checkpoint (e.g. model.pth)")
    group.add_argument("--ensemble-dir", help="Quantize every .pth file in an ensemble directory")
    parser.add_argument("--output", help="Output quantized bundle path or directory")
    parser.add_argument("--sample-npy", help="Optional representative .npy sequence for benchmarking")
    parser.add_argument("--warmup-iters", type=int, default=25, help="Warmup iterations before timing")
    parser.add_argument("--benchmark-iters", type=int, default=100, help="Timed benchmark iterations")
    parser.add_argument("--backend", choices=["auto", "fbgemm", "qnnpack"], default="auto", help="Quantized CPU backend")
    args = parser.parse_args()

    if args.backend == "auto":
        backend = set_quantized_engine()
    else:
        import torch

        torch.backends.quantized.engine = args.backend
        backend = args.backend

    sample, proximity = _load_sample(args.sample_npy)

    if args.checkpoint:
        output_path = args.output or _default_output_path(args.checkpoint)
        result = _quantize_single_checkpoint(
            args.checkpoint,
            output_path,
            sample,
            proximity,
            args.warmup_iters,
            args.benchmark_iters,
        )
        result["backend"] = backend
        print(json.dumps(result, indent=2))
        return

    output_dir = args.output or os.path.join(args.ensemble_dir, "quantized")
    results = _batch_quantize_ensemble(
        args.ensemble_dir,
        output_dir,
        sample,
        proximity,
        args.warmup_iters,
        args.benchmark_iters,
    )
    print(json.dumps({"backend": backend, "results": results}, indent=2))


if __name__ == "__main__":
    main()