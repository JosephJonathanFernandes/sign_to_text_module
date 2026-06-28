"""Export PyTorch models to ONNX format with metadata preservation.

Supports:
- Single model export
- K-fold ensemble export
- Dynamic batch size
- FP32 export (quantization-ready)
- Metadata serialization
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.onnx

from src.training.model import SignLanguageGRU


@dataclass
class ExportConfig:
    checkpoint: str
    output_path: str
    seq_len: int = 20
    feature_dim: int = 506
    opset_version: int = 18
    device: str = "cpu"
    verbose: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch ISL model to ONNX format")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--output", required=True, help="Output path for ONNX model (.onnx)")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    parser.add_argument("--feature-dim", type=int, default=506, help="Feature dimension per frame")
    parser.add_argument("--opset-version", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--device", default="cpu", help="Device for export (cpu or cuda)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def _infer_num_classes(model_dict: dict[str, Any], ckpt_meta: dict[str, Any]) -> int:
    """Infer the classification head size from checkpoint metadata/state dict."""
    if isinstance(ckpt_meta.get("num_classes"), int):
        return int(ckpt_meta["num_classes"])

    for key in ("fc.3.weight", "classifier.weight", "head.weight"):
        if key in model_dict and hasattr(model_dict[key], "shape"):
            return int(model_dict[key].shape[0])

    raise ValueError("Unable to infer num_classes from checkpoint")


def load_pytorch_model(checkpoint_path: str, device: str) -> tuple[SignLanguageGRU, dict[str, Any]]:
    """Load a PyTorch ISL model checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_dict = ckpt.get("model_state_dict", ckpt)
    ckpt_meta = ckpt if isinstance(ckpt, dict) and "model_state_dict" in ckpt else {}

    num_classes = _infer_num_classes(model_dict, ckpt_meta)
    model = SignLanguageGRU(num_classes=num_classes)
    if isinstance(model_dict, dict):
        model.load_state_dict(model_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, ckpt_meta


@torch.no_grad()
def export_to_onnx(
    model: SignLanguageGRU,
    output_path: str,
    *,
    seq_len: int = 20,
    feature_dim: int = 506,
    opset_version: int = 18,
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, Any]:
    """Export PyTorch model to ONNX with dynamic axes."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if opset_version < 18:
        print(f"[WARN] opset_version={opset_version} is too low for the installed exporter; using 18 instead.")
        opset_version = 18

    dummy_input = torch.randn(1, seq_len, feature_dim, dtype=torch.float32, device=device)
    dummy_proximity = torch.randn(1, seq_len, dtype=torch.float32, device=device)

    try:
        torch.onnx.export(
            model,
            (dummy_input, dummy_proximity),
            output_path,
            input_names=["input_seq", "proximity"],
            output_names=["logits"],
            dynamic_axes={
                "input_seq": {0: "batch"},
                "proximity": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=verbose,
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {str(e)}") from e

    if not os.path.isfile(output_path):
        raise RuntimeError(f"ONNX export completed but file not found: {output_path}")

    model_size_bytes = os.path.getsize(output_path)

    return {
        "output_path": os.path.abspath(output_path),
        "model_size_bytes": model_size_bytes,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "opset_version": opset_version,
        "device": device,
    }


def _json_safe(value: Any) -> Any:
    """Convert tensors/arrays and nested containers into JSON-serializable values."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value, key=lambda item: str(item))]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def save_metadata(output_path: str, metadata: dict[str, Any], checkpoint_meta: dict[str, Any]) -> None:
    """Save export metadata alongside the ONNX model."""
    meta_path = output_path.replace(".onnx", "_metadata.json")
    combined_meta = {
        "export": metadata,
        "checkpoint": checkpoint_meta,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(combined_meta), f, indent=2)
    print(f"Saved metadata: {meta_path}")


def main() -> None:
    args = parse_args()
    cfg = ExportConfig(
        checkpoint=args.checkpoint,
        output_path=args.output,
        seq_len=args.seq_len,
        feature_dim=args.feature_dim,
        opset_version=args.opset_version,
        device=args.device,
        verbose=args.verbose,
    )

    print(f"Loading checkpoint: {cfg.checkpoint}")
    model, ckpt_meta = load_pytorch_model(cfg.checkpoint, cfg.device)

    print(f"Exporting to ONNX: {cfg.output_path}")
    export_meta = export_to_onnx(
        model,
        cfg.output_path,
        seq_len=cfg.seq_len,
        feature_dim=cfg.feature_dim,
        opset_version=cfg.opset_version,
        device=cfg.device,
        verbose=cfg.verbose,
    )

    save_metadata(cfg.output_path, export_meta, ckpt_meta)

    print("=" * 80)
    print(f"✓ Export successful")
    print(f"  Model: {export_meta['output_path']}")
    print(f"  Size: {export_meta['model_size_bytes'] / 1024 / 1024:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
