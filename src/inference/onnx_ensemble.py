"""ONNX ensemble support module.

Provides seamless integration of ONNX models into the existing ensemble loader.
Automatically detects and loads .onnx models alongside PyTorch models.
Supports mixed ONNX/PyTorch ensembles with automatic fallback.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.inference.onnx_inference import ONNXModelWrapper

logger = logging.getLogger(__name__)


_ONNX_FP32_SUFFIXES = ("_fp32", "_float32", "_fp")
_ONNX_INT8_SUFFIXES = ("_int8", "_quantized", "_int8_quantized")
_SKIP_DIR_NAMES = {"venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".idea", ".vscode"}


class EnsembleModel:
    """Wrapper for both PyTorch and ONNX models in ensemble."""

    def __init__(self, model: Any, model_type: str = "pytorch", name: str = ""):
        self.model = model
        self.model_type = model_type  # "pytorch" or "onnx"
        self.name = name
        self.is_pytorch = model_type == "pytorch"
        self.is_onnx = model_type == "onnx"

    def infer(self, tensor: torch.Tensor, proximity: Optional[torch.Tensor] = None) -> np.ndarray:
        """Run inference and return logits as numpy array."""
        if self.is_pytorch:
            return self._infer_pytorch(tensor, proximity)
        elif self.is_onnx:
            return self._infer_onnx(tensor, proximity)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    @torch.no_grad()
    def _infer_pytorch(self, tensor: torch.Tensor, proximity: Optional[torch.Tensor] = None) -> np.ndarray:
        """PyTorch inference."""
        logits = self.model(tensor, proximity=proximity)
        if isinstance(logits, dict):
            logits = logits["sign_logits"]
        return logits.cpu().detach().numpy()[0]

    def _infer_onnx(self, tensor: torch.Tensor, proximity: Optional[torch.Tensor] = None) -> np.ndarray:
        """ONNX inference."""
        input_seq = tensor.cpu().numpy().astype(np.float32)
        if proximity is not None:
            prox_np = proximity.cpu().numpy().astype(np.float32)
        else:
            prox_np = np.zeros((input_seq.shape[0], input_seq.shape[1]), dtype=np.float32)

        probs = self.model(input_seq, prox_np)

        from scipy.special import logit

        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        logits = logit(probs)
        return logits[0]


def load_onnx_model(
    onnx_path: str,
    pytorch_fallback_path: Optional[str] = None,
    device: str = "cpu",
) -> EnsembleModel:
    """Load ONNX model with optional PyTorch fallback."""
    try:
        wrapper = ONNXModelWrapper(
            onnx_path,
            pytorch_checkpoint=pytorch_fallback_path,
            device=device,
            fallback_to_pytorch=True,
            enable_profiling=False,
        )
        return EnsembleModel(wrapper, model_type="onnx", name=os.path.basename(onnx_path))
    except Exception as e:
        logger.error(f"Failed to load ONNX model {onnx_path}: {str(e)}")
        raise


def _normalize_model_family(filename: str) -> str:
    """Collapse format-specific filenames to a shared family stem."""
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    for suffix in _ONNX_FP32_SUFFIXES + _ONNX_INT8_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _artifact_priority(filename: str) -> tuple[int, str]:
    """Lower is better: ONNX FP32 -> ONNX INT8 -> PyTorch."""
    lower = filename.lower()
    if lower.endswith(".onnx"):
        if any(lower.endswith(f"{suffix}.onnx") for suffix in _ONNX_INT8_SUFFIXES):
            return (1, lower)
        return (0, lower)
    if lower.endswith(".pth"):
        return (2, lower)
    return (99, lower)


def _coerce_search_dirs(model_dirs: str | Sequence[str]) -> list[str]:
    if isinstance(model_dirs, str):
        return [model_dirs]
    return [d for d in model_dirs if d]


def _iter_model_files(search_dir: str):
    """Yield model artifact files recursively under a search directory."""
    for root, dirs, files in os.walk(search_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES and not d.startswith(".")]
        for fname in sorted(files):
            if fname.endswith((".onnx", ".pth")):
                yield root, fname


def detect_and_load_models(
    ensemble_dir: str | Sequence[str],
    max_models: int = 5,
    device: str = "cpu",
) -> tuple[list[EnsembleModel], dict[str, Any]]:
    """Auto-detect and load .pth and .onnx models from one or more directories.

    Args:
        ensemble_dir: directory or list of directories containing model files
        max_models: maximum number of models to load
        device: device for PyTorch models

    Returns:
        (models_list, metadata_dict)
        metadata_dict includes counts of ONNX vs PyTorch models loaded
    """
    models = []
    metadata = {
        "pytorch_models": 0,
        "onnx_models": 0,
        "total_models": 0,
        "selected_artifacts": [],
    }

    search_dirs = [d for d in _coerce_search_dirs(ensemble_dir) if os.path.isdir(d)]
    if not search_dirs:
        return models, metadata

    candidates: dict[str, dict[str, Any]] = {}
    pth_fallbacks: dict[str, str] = {}

    for dir_index, search_dir in enumerate(search_dirs):
        for root, fname in _iter_model_files(search_dir):
            if not (fname.endswith(".onnx") or fname.endswith(".pth")):
                continue

            family = _normalize_model_family(fname)
            full_path = os.path.join(root, fname)
            priority = _artifact_priority(fname)
            candidate_key = (priority[0], dir_index, priority[1], fname)

            if fname.endswith(".pth"):
                current_fallback = pth_fallbacks.get(family)
                if current_fallback is None or candidate_key < candidates.get(f"pth::{family}", {}).get("sort_key", (99, 99, "", "")):
                    pth_fallbacks[family] = full_path
                candidates.setdefault(f"pth::{family}", {"sort_key": candidate_key, "path": full_path})
                continue

            current = candidates.get(family)
            if current is None or candidate_key < current["sort_key"]:
                candidates[family] = {
                    "sort_key": candidate_key,
                    "path": full_path,
                    "family": family,
                    "filename": fname,
                    "priority": priority[0],
                    "dir_index": dir_index,
                }

    ordered = sorted(candidates.values(), key=lambda item: item["sort_key"])
    remaining = max_models
    loaded_pth = 0
    loaded_onnx = 0

    for item in ordered:
        if remaining <= 0:
            break

        fname = item["filename"]
        fpath = item["path"]
        family = item["family"]
        if fname.endswith(".onnx"):
            fallback_pth = pth_fallbacks.get(family)
            try:
                model = load_onnx_model(fpath, pytorch_fallback_path=fallback_pth, device=device)
                models.append(model)
                loaded_onnx += 1
                remaining -= 1
                metadata["selected_artifacts"].append(
                    {
                        "family": family,
                        "kind": "onnx",
                        "path": fpath,
                        "fallback_pth": fallback_pth,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load ONNX model {fname}: {str(e)}")
            continue

        try:
            from src.utils.quantization_utils import load_model_artifact

            model, _, _, _, _ = load_model_artifact(fpath, map_location=device)
            models.append(EnsembleModel(model, model_type="pytorch", name=fname))
            loaded_pth += 1
            remaining -= 1
            metadata["selected_artifacts"].append(
                {
                    "family": family,
                    "kind": "pytorch",
                    "path": fpath,
                    "fallback_pth": None,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to load PyTorch model {fname}: {str(e)}")

    metadata["pytorch_models"] = loaded_pth
    metadata["onnx_models"] = loaded_onnx
    metadata["total_models"] = len(models)

    return models, metadata


def ensemble_predict_mixed(
    models: list[EnsembleModel],
    sequence: np.ndarray,
    device: str = "cpu",
    proximity_feat_dim: int = 1,
    frame_feat_dim: int = 506,
    proximity_index: int = 20,
) -> tuple[int, float, np.ndarray]:
    """Ensemble inference with mixed ONNX/PyTorch models.

    Args:
        models: list of EnsembleModel objects
        sequence: numpy array of shape (NUM_FRAMES, feat_dim)
        device: torch device
        proximity_feat_dim: dimension of proximity features
        frame_feat_dim: feature dimension per frame
        proximity_index: index in sequence to extract proximity

    Returns:
        (pred_idx, confidence, probs)
    """
    if not models:
        raise ValueError("No models provided for ensemble prediction")

    all_logits = []
    device_obj = torch.device(device)

    seq = _align_sequence_dim(sequence, frame_feat_dim)
    tensor = torch.from_numpy(seq).unsqueeze(0).float().to(device_obj)

    if proximity_feat_dim > 0 and tensor.shape[-1] >= frame_feat_dim:
        proximity = tensor[:, :, proximity_index : proximity_index + proximity_feat_dim]
        if proximity.shape[-1] == 1:
            proximity = proximity.squeeze(-1)
    else:
        proximity = None

    for model in models:
        try:
            logits = model.infer(tensor, proximity)
            all_logits.append(logits)
        except Exception as e:
            logger.warning(f"Inference failed for {model.name}: {str(e)}; skipping")
            continue

    if not all_logits:
        raise RuntimeError("All models failed inference; ensemble unavailable")

    avg_logits = np.mean(all_logits, axis=0)
    avg_logits_tensor = torch.from_numpy(avg_logits).unsqueeze(0).float().to(device_obj)
    avg_probs = F.softmax(avg_logits_tensor, dim=1).cpu().detach().numpy()[0]

    pred_idx = int(np.argmax(avg_probs))
    confidence = float(avg_probs[pred_idx])

    return pred_idx, confidence, avg_probs


def _align_sequence_dim(seq: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad/truncate sequence feature dimension."""
    feat_dim = seq.shape[1]
    if feat_dim == target_dim:
        return seq
    if feat_dim > target_dim:
        return seq[:, :target_dim]

    pad = np.zeros((seq.shape[0], target_dim - feat_dim), dtype=np.float32)
    return np.concatenate([seq, pad], axis=1)
