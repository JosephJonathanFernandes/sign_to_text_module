"""Evaluate a quantized ISL checkpoint or ensemble bundle on CPU.

The script focuses on latency, FPS, and optional validation accuracy using the
existing isolated-word dataset split.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from quantization_utils import benchmark_ensemble_latency, benchmark_model_latency, load_checkpoint, load_model_artifact


def _load_sample(sample_npy: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    if sample_npy:
        sample = np.load(sample_npy).astype(np.float32)
        proximity = sample[:, -1] if sample.shape[1] > 0 else None
        return sample, proximity

    from config import get_config

    cfg = get_config()
    sample = np.zeros((cfg.preprocessing.num_frames, cfg.frame_features.input_sequence_dim), dtype=np.float32)
    proximity = None
    if cfg.spatial.proximity_dim > 0:
        proximity = sample[:, cfg.frame_features.proximity_index]
    return sample, proximity


def _validation_accuracy_for_single_model(model) -> float | None:
    try:
        from train import create_data_loaders
    except Exception:
        return None

    _, val_loader, _, _, _ = create_data_loaders()
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for sequences, proximity, labels in val_loader:
            logits = model(sequences.to("cpu"), proximity=proximity.to("cpu"))
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels.to("cpu")).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a quantized ISL artifact")
    parser.add_argument("--checkpoint", required=True, help="Quantized bundle path or fp32 checkpoint")
    parser.add_argument("--sample-npy", help="Representative .npy sequence for latency benchmarking")
    parser.add_argument("--warmup-iters", type=int, default=25, help="Warmup iterations before timing")
    parser.add_argument("--benchmark-iters", type=int, default=100, help="Timed iterations")
    parser.add_argument("--evaluate-accuracy", action="store_true", help="Measure validation accuracy using the existing split")
    parser.add_argument("--ensemble-latency", action="store_true", help="Treat the checkpoint as a directory of quantized models and benchmark sequential ensemble latency")
    args = parser.parse_args()

    sample, proximity = _load_sample(args.sample_npy)

    if args.ensemble_latency:
        if not os.path.isdir(args.checkpoint):
            raise ValueError("--ensemble-latency requires --checkpoint to be a directory")

        from model import SignLanguageGRU
        import torch

        models = []
        for entry in sorted(os.listdir(args.checkpoint)):
            if not entry.endswith(".pth"):
                continue
            artifact_path = os.path.join(args.checkpoint, entry)
            model, classes, num_classes, is_quantized, checkpoint = load_model_artifact(artifact_path, map_location="cpu")
            models.append(model)

        benchmark = benchmark_ensemble_latency(
            models,
            sample,
            proximity=proximity,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
        )
        report = {
            "artifact": os.path.abspath(args.checkpoint),
            "mode": "ensemble",
            "benchmark": {
                "avg_ms": round(benchmark.avg_ms, 4),
                "median_ms": round(benchmark.median_ms, 4),
                "p95_ms": round(benchmark.p95_ms, 4),
                "min_ms": round(benchmark.min_ms, 4),
                "max_ms": round(benchmark.max_ms, 4),
                "std_ms": round(benchmark.std_ms, 4),
                "fps": round(benchmark.fps, 4),
                "warmup_iters": benchmark.warmup_iters,
                "benchmark_iters": benchmark.benchmark_iters,
            },
        }
        print(json.dumps(report, indent=2))
        return

    model, classes, num_classes, is_quantized, checkpoint = load_model_artifact(args.checkpoint, map_location="cpu")
    benchmark = benchmark_model_latency(
        model,
        sample,
        proximity=proximity,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )

    accuracy = None
    if args.evaluate_accuracy:
        accuracy = _validation_accuracy_for_single_model(model)

    report = {
        "artifact": os.path.abspath(args.checkpoint),
        "is_quantized": is_quantized,
        "num_classes": num_classes,
        "benchmark": {
            "avg_ms": round(benchmark.avg_ms, 4),
            "median_ms": round(benchmark.median_ms, 4),
            "p95_ms": round(benchmark.p95_ms, 4),
            "min_ms": round(benchmark.min_ms, 4),
            "max_ms": round(benchmark.max_ms, 4),
            "std_ms": round(benchmark.std_ms, 4),
            "fps": round(benchmark.fps, 4),
            "warmup_iters": benchmark.warmup_iters,
            "benchmark_iters": benchmark.benchmark_iters,
        },
        "validation_accuracy": round(float(accuracy), 4) if accuracy is not None else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()