"""ONNX ensemble support module.

Provides seamless integration of ONNX models into the existing ensemble loader.
Automatically detects and loads .onnx models alongside PyTorch models.
Supports mixed ONNX/PyTorch ensembles with automatic fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from onnx_inference import ONNXModelWrapper

logger = logging.getLogger(__name__)


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


def detect_and_load_models(
    ensemble_dir: str,
    max_models: int = 5,
    device: str = "cpu",
) -> tuple[list[EnsembleModel], dict[str, Any]]:
    """Auto-detect and load .pth and .onnx models from ensemble directory.

    Args:
        ensemble_dir: directory containing model files
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
    }

    if not os.path.isdir(ensemble_dir):
        return models, metadata

    pth_files = sorted([f for f in os.listdir(ensemble_dir) if f.endswith(".pth")])
    onnx_files = sorted([f for f in os.listdir(ensemble_dir) if f.endswith(".onnx")])

    remaining = max_models
    loaded_pth = 0
    loaded_onnx = 0

    for fname in pth_files:
        if remaining <= 0:
            break
        fpath = os.path.join(ensemble_dir, fname)
        try:
            from quantization_utils import load_model_artifact

            model, _, _, _, _ = load_model_artifact(fpath, map_location=device)
            models.append(EnsembleModel(model, model_type="pytorch", name=fname))
            loaded_pth += 1
            remaining -= 1
        except Exception as e:
            logger.warning(f"Failed to load PyTorch model {fname}: {str(e)}")

    for fname in onnx_files:
        if remaining <= 0:
            break
        fpath = os.path.join(ensemble_dir, fname)
        try:
            model = load_onnx_model(fpath, device=device)
            models.append(model)
            loaded_onnx += 1
            remaining -= 1
        except Exception as e:
            logger.warning(f"Failed to load ONNX model {fname}: {str(e)}")

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
