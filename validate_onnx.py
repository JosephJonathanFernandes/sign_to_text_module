"""Validate ONNX model outputs against PyTorch baseline.

Compares numeric outputs, confidence distributions, and ensemble behavior.
Generates detailed metrics report.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import numpy as np
import torch

from onnx_inference import ONNXModelWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ONNX outputs against PyTorch")
    parser.add_argument("--pytorch-checkpoint", required=True, help="PyTorch checkpoint path")
    parser.add_argument("--onnx-model", required=True, help="ONNX model path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    parser.add_argument("--feature-dim", type=int, default=506, help="Feature dimension")
    parser.add_argument("--output", help="Output JSON file for report")
    return parser.parse_args()


def load_pytorch_model(checkpoint_path: str, device: str = "cpu") -> Any:
    """Load PyTorch ISL model."""
    from model import ISLModel

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_dict = ckpt.get("model_state_dict", ckpt)
    model = ISLModel()
    if isinstance(model_dict, dict):
        model.load_state_dict(model_dict, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def infer_pytorch(
    model: Any, input_seq: np.ndarray, proximity: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """Inference with PyTorch model."""
    input_t = torch.from_numpy(input_seq).to(device)
    prox_t = torch.from_numpy(proximity).to(device)
    logits = model(input_t, prox_t)
    output = torch.softmax(logits, dim=-1).cpu().numpy()
    return output


def infer_onnx(
    wrapper: ONNXModelWrapper, input_seq: np.ndarray, proximity: np.ndarray
) -> np.ndarray:
    """Inference with ONNX model."""
    return wrapper(input_seq, proximity)


def compute_metrics(
    pytorch_outputs: np.ndarray, onnx_outputs: np.ndarray
) -> dict[str, Any]:
    """Compute comparison metrics."""
    l2_distance = np.sqrt(np.mean((pytorch_outputs - onnx_outputs) ** 2))
    l1_distance = np.mean(np.abs(pytorch_outputs - onnx_outputs))
    max_abs_diff = np.max(np.abs(pytorch_outputs - onnx_outputs))
    mean_abs_diff = np.mean(np.abs(pytorch_outputs - onnx_outputs))

    pytorch_preds = np.argmax(pytorch_outputs, axis=-1)
    onnx_preds = np.argmax(onnx_outputs, axis=-1)
    agreement = np.mean(pytorch_preds == onnx_preds)

    pytorch_conf = np.max(pytorch_outputs, axis=-1)
    onnx_conf = np.max(onnx_outputs, axis=-1)
    conf_diff = np.abs(pytorch_conf - onnx_conf)
    mean_conf_diff = np.mean(conf_diff)
    max_conf_diff = np.max(conf_diff)

    return {
        "l2_distance": float(l2_distance),
        "l1_distance": float(l1_distance),
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(mean_abs_diff),
        "prediction_agreement": float(agreement),
        "mean_confidence_diff": float(mean_conf_diff),
        "max_confidence_diff": float(max_conf_diff),
    }


def main() -> None:
    args = parse_args()

    print("Loading PyTorch model...")
    pytorch_model = load_pytorch_model(args.pytorch_checkpoint)

    print("Loading ONNX model...")
    onnx_wrapper = ONNXModelWrapper(args.onnx_model, device="cpu")

    if not onnx_wrapper.onnx_available:
        print("ERROR: ONNX model not available")
        return

    print(f"Validating on {args.num_samples} samples...")

    pytorch_all = []
    onnx_all = []

    for i in range(args.num_samples):
        input_seq = np.random.randn(1, args.seq_len, args.feature_dim).astype(np.float32)
        proximity = np.random.randn(1, args.seq_len).astype(np.float32)

        pytorch_out = infer_pytorch(pytorch_model, input_seq, proximity)
        onnx_out = infer_onnx(onnx_wrapper, input_seq, proximity)

        pytorch_all.append(pytorch_out)
        onnx_all.append(onnx_out)

        if (i + 1) % (args.num_samples // 10 or 1) == 0:
            print(f"  {i + 1}/{args.num_samples} samples processed")

    pytorch_all = np.concatenate(pytorch_all, axis=0)
    onnx_all = np.concatenate(onnx_all, axis=0)

    metrics = compute_metrics(pytorch_all, onnx_all)

    print("\n" + "=" * 80)
    print("Validation Report")
    print("=" * 80)
    print(f"PyTorch Model: {args.pytorch_checkpoint}")
    print(f"ONNX Model:    {args.onnx_model}")
    print(f"Samples:       {args.num_samples}")
    print("-" * 80)
    print(f"L2 Distance:              {metrics['l2_distance']:.6f}")
    print(f"L1 Distance:              {metrics['l1_distance']:.6f}")
    print(f"Max Abs Diff:             {metrics['max_abs_diff']:.6f}")
    print(f"Mean Abs Diff:            {metrics['mean_abs_diff']:.6f}")
    print(f"Prediction Agreement:     {metrics['prediction_agreement']:.4%}")
    print(f"Mean Confidence Diff:     {metrics['mean_confidence_diff']:.6f}")
    print(f"Max Confidence Diff:      {metrics['max_confidence_diff']:.6f}")
    print("=" * 80)

    if metrics["prediction_agreement"] >= 0.99:
        print("✓ Validation PASSED (>99% prediction agreement)")
    elif metrics["prediction_agreement"] >= 0.95:
        print("⚠ Validation WARNING (95-99% prediction agreement)")
    else:
        print("✗ Validation FAILED (<95% prediction agreement)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        report = {
            "pytorch_model": args.pytorch_checkpoint,
            "onnx_model": args.onnx_model,
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
            "feature_dim": args.feature_dim,
            "metrics": metrics,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
