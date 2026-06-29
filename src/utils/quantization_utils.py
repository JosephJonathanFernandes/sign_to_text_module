"""Dynamic quantization utilities for CPU inference.

This module keeps the existing SignLanguageGRU pipeline intact while providing
helpers to:
  - apply PyTorch dynamic quantization to Linear / GRU / LSTM modules
  - save and load quantized bundles
  - benchmark latency and FPS on representative inputs
  - estimate model artifact size for before/after comparisons

Dynamic quantization is a runtime optimization for CPU deployment. It is not a
replacement for fp32 training checkpoints, and it is not the preferred format
for ONNX export later.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from config import get_config
from src.training.model import SignLanguageGRU

cfg = get_config()

try:
    from torch.ao.quantization import quantize_dynamic as _quantize_dynamic
except Exception:  # pragma: no cover - older torch fallback
    from torch.quantization import quantize_dynamic as _quantize_dynamic


QUANTIZABLE_MODULES = (nn.Linear, nn.LSTM, nn.GRU)


@dataclass
class BenchmarkResult:
    """Summary of a latency benchmark run."""

    avg_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    fps: float
    warmup_iters: int
    benchmark_iters: int


def preferred_quantized_engine() -> str:
    """Return the best available quantized CPU backend for this machine."""
    preferred = "fbgemm"
    supported = getattr(torch.backends.quantized, "supported_engines", [])
    if preferred in supported:
        return preferred
    if supported:
        return supported[0]
    return getattr(torch.backends.quantized, "engine", preferred)


def set_quantized_engine() -> str:
    """Set the preferred quantized backend when the runtime supports it."""
    engine = preferred_quantized_engine()
    try:
        torch.backends.quantized.engine = engine
    except Exception:
        pass
    return engine


def model_size_mb(model: nn.Module) -> float:
    """Estimate serialized state_dict size in MB."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name

    try:
        torch.save(model.state_dict(), temp_path)
        return os.path.getsize(temp_path) / (1024.0 * 1024.0)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def bundle_size_mb(bundle: dict) -> float:
    """Estimate serialized bundle size in MB."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name

    try:
        torch.save(bundle, temp_path)
        return os.path.getsize(temp_path) / (1024.0 * 1024.0)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict:
    """Load a torch checkpoint bundle from disk."""
    return torch.load(path, map_location=map_location, weights_only=False)


def is_quantized_bundle(checkpoint: dict) -> bool:
    """Return True if a checkpoint is already a saved quantized bundle."""
    return bool(checkpoint.get("quantized", False))


def quantize_sign_language_model(
    model: nn.Module,
    *,
    inplace: bool = False,
    modules: Sequence[type[nn.Module]] = QUANTIZABLE_MODULES,
) -> nn.Module:
    """Apply dynamic quantization to the supported module types.

    The current model uses Linear layers in the encoder, attention blocks,
    optional GNN branch, and the classifier head. Quantizing all Linear layers
    plus recurrent layers gives the best CPU gains with minimal code changes.
    """
    set_quantized_engine()
    quantized = _quantize_dynamic(
        model,
        {module_type for module_type in modules},
        dtype=torch.qint8,
        inplace=inplace,
    )
    return quantized


def build_model_from_checkpoint(checkpoint: dict) -> SignLanguageGRU:
    """Instantiate a SignLanguageGRU and load weights from a checkpoint."""
    num_classes = int(checkpoint["num_classes"])
    model_kwargs = checkpoint.get("model_kwargs") or {}
    state_dict = checkpoint["model_state_dict"]

    if model_kwargs:
        model = SignLanguageGRU(num_classes=num_classes, **model_kwargs)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    hidden_size = int(state_dict["input_proj.0.weight"].shape[0])
    bidirectional = any(key.endswith("_reverse") for key in state_dict.keys() if key.startswith("gru.weight_ih_l"))

    # Infer the number of recurrent layers from the highest GRU layer index.
    layer_indices = []
    for key in state_dict.keys():
        if key.startswith("gru.weight_ih_l") and not key.endswith("_reverse"):
            suffix = key.split("gru.weight_ih_l", 1)[1]
            try:
                layer_indices.append(int(suffix))
            except ValueError:
                continue
    num_layers = (max(layer_indices) + 1) if layer_indices else 1

    model = SignLanguageGRU(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def build_quantized_model_from_bundle(checkpoint: dict) -> SignLanguageGRU:
    """Rebuild a quantized SignLanguageGRU from a saved bundle."""
    num_classes = int(checkpoint["num_classes"])
    model_kwargs = checkpoint.get("model_kwargs") or {}
    if model_kwargs:
        model = SignLanguageGRU(num_classes=num_classes, **model_kwargs)
    else:
        # Fall back to inferring architecture from the quantized state dict.
        model = build_model_from_checkpoint({
            "num_classes": num_classes,
            "model_state_dict": checkpoint["quantized_state_dict"],
        })
    model = quantize_sign_language_model(model)

    quantized_state = checkpoint.get("quantized_state_dict")
    if quantized_state is None:
        raise KeyError("Quantized bundle is missing 'quantized_state_dict'")

    model.load_state_dict(quantized_state, strict=False)
    model.eval()
    return model


def load_model_artifact(path: str, map_location: str | torch.device = "cpu") -> tuple[nn.Module, list[str] | None, int, bool, dict]:
    """Load either a fp32 checkpoint or a quantized bundle.

    Returns:
        model, classes, num_classes, is_quantized, checkpoint
    """
    checkpoint = load_checkpoint(path, map_location=map_location)

    if is_quantized_bundle(checkpoint):
        model = build_quantized_model_from_bundle(checkpoint)
        quantized = True
    else:
        model = build_model_from_checkpoint(checkpoint)
        quantized = False

    classes = checkpoint.get("classes")
    num_classes = int(checkpoint["num_classes"])
    return model, classes, num_classes, quantized, checkpoint


def save_quantized_bundle(
    path: str,
    *,
    quantized_model: nn.Module,
    num_classes: int,
    classes: Sequence[str] | None = None,
    source_checkpoint: str | None = None,
    model_kwargs: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Save a quantized model bundle that can be reloaded later.

    The bundle stores the quantized state_dict and metadata rather than a raw
    pickled module, which is more robust across environments.
    """
    bundle = {
        "format": "sign_language_dynamic_quantized_v1",
        "quantized": True,
        "num_classes": int(num_classes),
        "classes": list(classes) if classes is not None else None,
        "source_checkpoint": source_checkpoint,
        "model_kwargs": dict(model_kwargs) if model_kwargs is not None else None,
        "quantization_engine": preferred_quantized_engine(),
        "quantization_spec": [module.__name__ for module in QUANTIZABLE_MODULES],
        "quantized_state_dict": quantized_model.state_dict(),
        "config_version": getattr(cfg, "CONFIG_VERSION", None),
    }
    if metadata:
        bundle["metadata"] = dict(metadata)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(bundle, path)
    return bundle


def _as_tensor_sequence(sequence: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(sequence, torch.Tensor):
        return sequence.detach().cpu().float()
    return torch.as_tensor(sequence, dtype=torch.float32)


@torch.inference_mode()
def benchmark_model_latency(
    model: nn.Module,
    sequence: np.ndarray | torch.Tensor,
    *,
    proximity: np.ndarray | torch.Tensor | None = None,
    warmup_iters: int = 20,
    benchmark_iters: int = 100,
) -> BenchmarkResult:
    """Benchmark a single model on CPU and return latency/FPS stats."""
    model.eval()
    seq_t = _as_tensor_sequence(sequence).unsqueeze(0)
    prox_t = None
    if proximity is not None:
        prox_t = _as_tensor_sequence(proximity).unsqueeze(0)

    for _ in range(max(0, warmup_iters)):
        if prox_t is None:
            _ = model(seq_t)
        else:
            _ = model(seq_t, proximity=prox_t)

    timings_ms: list[float] = []
    for _ in range(max(1, benchmark_iters)):
        start = time.perf_counter()
        if prox_t is None:
            _ = model(seq_t)
        else:
            _ = model(seq_t, proximity=prox_t)
        end = time.perf_counter()
        timings_ms.append((end - start) * 1000.0)

    arr = np.asarray(timings_ms, dtype=np.float64)
    avg_ms = float(arr.mean())
    median_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95))
    min_ms = float(arr.min())
    max_ms = float(arr.max())
    std_ms = float(arr.std(ddof=0))
    fps = float(1000.0 / avg_ms) if avg_ms > 0 else 0.0

    return BenchmarkResult(
        avg_ms=avg_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        std_ms=std_ms,
        fps=fps,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
    )


def benchmark_ensemble_latency(
    models: Iterable[nn.Module],
    sequence: np.ndarray | torch.Tensor,
    *,
    proximity: np.ndarray | torch.Tensor | None = None,
    warmup_iters: int = 10,
    benchmark_iters: int = 50,
) -> BenchmarkResult:
    """Benchmark a list of models by executing them sequentially.

    This approximates current ensemble latency because each model is executed
    independently in the existing inference path.
    """
    models = list(models)
    if not models:
        raise ValueError("No models were provided for benchmarking")

    seq_t = _as_tensor_sequence(sequence).unsqueeze(0)
    prox_t = None
    if proximity is not None:
        prox_t = _as_tensor_sequence(proximity).unsqueeze(0)

    for _ in range(max(0, warmup_iters)):
        for model in models:
            model.eval()
            if prox_t is None:
                _ = model(seq_t)
            else:
                _ = model(seq_t, proximity=prox_t)

    timings_ms: list[float] = []
    for _ in range(max(1, benchmark_iters)):
        start = time.perf_counter()
        for model in models:
            model.eval()
            if prox_t is None:
                _ = model(seq_t)
            else:
                _ = model(seq_t, proximity=prox_t)
        end = time.perf_counter()
        timings_ms.append((end - start) * 1000.0)

    arr = np.asarray(timings_ms, dtype=np.float64)
    avg_ms = float(arr.mean())
    median_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95))
    min_ms = float(arr.min())
    max_ms = float(arr.max())
    std_ms = float(arr.std(ddof=0))
    fps = float(1000.0 / avg_ms) if avg_ms > 0 else 0.0

    return BenchmarkResult(
        avg_ms=avg_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        std_ms=std_ms,
        fps=fps,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
    )
