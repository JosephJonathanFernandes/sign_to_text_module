"""Integration extension for ensemble.py to support ONNX models.

This module provides drop-in functions that can be used alongside
existing ensemble.py functions to enable ONNX model support.

No modifications to ensemble.py required - use these functions
to enable ONNX inference while maintaining full backward compatibility.

Usage:
    from ensemble import load_ensemble, ensemble_predict
    from onnx_ensemble_integration import load_ensemble_with_onnx
    
    # Option 1: Use existing PyTorch ensemble (unchanged)
    models, classes, num_classes = load_ensemble()
    
    # Option 2: Use ONNX ensemble (new)
    models, classes, num_classes = load_ensemble_with_onnx()
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import get_config
from onnx_ensemble import detect_and_load_models, ensemble_predict_mixed

logger = logging.getLogger(__name__)

cfg = get_config()

ENSEMBLE_DIR = cfg.paths.ensemble_dir
PROCESSED_DIR = cfg.paths.processed_dir
DEVICE = cfg.hardware.torch_device


def load_ensemble_with_onnx(
    max_models: int | None = None,
    fallback_to_pytorch: bool = True,
) -> tuple[list[Any], list[str], int]:
    """Load ensemble with automatic ONNX+PyTorch detection.

    This function is a drop-in replacement for load_ensemble() that
    automatically detects and loads both .onnx and .pth models.

    Args:
        max_models: Maximum number of models to load. If None, uses
                   cfg.live_inference.ensemble_size
        fallback_to_pytorch: If True, includes PyTorch models in ensemble
                            even if ONNX models are available

    Returns:
        (models_list, classes, num_classes) - same as ensemble.load_ensemble()
    """
    if max_models is None:
        max_models = cfg.live_inference.ensemble_size

    current_classes = sorted([
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ])

    models, meta = detect_and_load_models(
        ENSEMBLE_DIR,
        max_models=max_models,
        device=str(DEVICE),
    )

    if not models:
        logger.warning(f"No ONNX/PyTorch models found in {ENSEMBLE_DIR}")
        return [], current_classes, len(current_classes)

    logger.info(
        f"[ONNX Ensemble] Loaded {meta['pytorch_models']} PyTorch + "
        f"{meta['onnx_models']} ONNX models"
    )

    return models, current_classes, len(current_classes)


@torch.no_grad()
def ensemble_predict_with_onnx(
    models: list[Any],
    sequence: np.ndarray,
    use_tta: bool = False,
) -> tuple[int, float, np.ndarray]:
    """Ensemble prediction with mixed ONNX/PyTorch models.

    This function is a drop-in replacement for ensemble_predict()
    that works with both PyTorch and ONNX models.

    Args:
        models: list of models (from load_ensemble_with_onnx)
        sequence: input sequence array (NUM_FRAMES, feat_dim)
        use_tta: test-time augmentation (only for PyTorch models)

    Returns:
        (pred_idx, confidence, probs)
    """
    try:
        pred_idx, confidence, probs = ensemble_predict_mixed(
            models,
            sequence,
            device=str(DEVICE),
            proximity_feat_dim=cfg.spatial.proximity_dim,
            frame_feat_dim=cfg.frame_features.frame_features_dim,
            proximity_index=cfg.frame_features.proximity_index,
        )
        return pred_idx, confidence, probs
    except Exception as e:
        logger.error(f"ONNX ensemble prediction failed: {str(e)}")
        raise


def check_onnx_models_available(ensemble_dir: str) -> bool:
    """Check if ONNX models are available in ensemble directory."""
    if not os.path.isdir(ensemble_dir):
        return False

    onnx_files = [f for f in os.listdir(ensemble_dir) if f.endswith(".onnx")]
    return len(onnx_files) > 0


def get_ensemble_status(ensemble_dir: str | None = None) -> dict[str, Any]:
    """Get detailed status of ensemble models.

    Returns:
        dict with keys:
        - 'total_pytorch': number of .pth files
        - 'total_onnx': number of .onnx files
        - 'pytorch_models': list of .pth filenames
        - 'onnx_models': list of .onnx filenames
        - 'mixed_ensemble': True if both types exist
    """
    if ensemble_dir is None:
        ensemble_dir = ENSEMBLE_DIR

    if not os.path.isdir(ensemble_dir):
        return {
            "total_pytorch": 0,
            "total_onnx": 0,
            "pytorch_models": [],
            "onnx_models": [],
            "mixed_ensemble": False,
        }

    pth_files = sorted([f for f in os.listdir(ensemble_dir) if f.endswith(".pth")])
    onnx_files = sorted([f for f in os.listdir(ensemble_dir) if f.endswith(".onnx")])

    return {
        "total_pytorch": len(pth_files),
        "total_onnx": len(onnx_files),
        "pytorch_models": pth_files,
        "onnx_models": onnx_files,
        "mixed_ensemble": len(pth_files) > 0 and len(onnx_files) > 0,
    }


def recommend_backend() -> str:
    """Recommend inference backend based on available models.

    Returns:
        'onnx' if ONNX models available, 'pytorch' otherwise
    """
    status = get_ensemble_status()

    if status["total_onnx"] > 0:
        return "onnx"
    else:
        return "pytorch"


def print_ensemble_info(ensemble_dir: str | None = None) -> None:
    """Print detailed ensemble information."""
    if ensemble_dir is None:
        ensemble_dir = ENSEMBLE_DIR

    status = get_ensemble_status(ensemble_dir)

    print("=" * 80)
    print("Ensemble Status")
    print("=" * 80)
    print(f"Directory: {ensemble_dir}")
    print(f"PyTorch models: {status['total_pytorch']}")
    if status["pytorch_models"]:
        for fname in status["pytorch_models"][:5]:
            print(f"  - {fname}")
        if len(status["pytorch_models"]) > 5:
            print(f"  ... and {len(status['pytorch_models']) - 5} more")

    print(f"ONNX models: {status['total_onnx']}")
    if status["onnx_models"]:
        for fname in status["onnx_models"][:5]:
            print(f"  - {fname}")
        if len(status["onnx_models"]) > 5:
            print(f"  ... and {len(status['onnx_models']) - 5} more")

    print(f"Mixed ensemble: {status['mixed_ensemble']}")
    recommended = recommend_backend()
    print(f"Recommended backend: {recommended}")
    print("=" * 80)
