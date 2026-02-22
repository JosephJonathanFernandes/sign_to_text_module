"""
Ensemble inference for image-based ISL letter/number model.
Loads K-fold image models and averages softmax outputs.
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from config_image import (
    DEVICE, IMG_ENSEMBLE_DIR, IMG_MODEL_PATH, IMG_SIZE,
)
from model_image import SignImageCNN
from dataset_image import ALL_CLASSES


def load_image_ensemble():
    """
    Load all fold image models.
    Falls back to single model_image.pth.
    Returns (models, classes, num_classes).
    """
    models = []

    if os.path.isdir(IMG_ENSEMBLE_DIR):
        fold_files = sorted([
            f for f in os.listdir(IMG_ENSEMBLE_DIR)
            if f.endswith(".pth")
        ])
        for fname in fold_files:
            fpath = os.path.join(IMG_ENSEMBLE_DIR, fname)
            ckpt = torch.load(
                fpath, map_location=DEVICE,
                weights_only=False,
            )
            nc = ckpt["num_classes"]
            m = SignImageCNN(nc).to(DEVICE)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            models.append(m)

    if models:
        print(
            f"[ImageEnsemble] Loaded {len(models)} "
            f"fold models"
        )
        return models, ALL_CLASSES, len(ALL_CLASSES)

    # Fallback to single model
    if os.path.exists(IMG_MODEL_PATH):
        ckpt = torch.load(
            IMG_MODEL_PATH, map_location=DEVICE,
            weights_only=False,
        )
        nc = ckpt["num_classes"]
        m = SignImageCNN(nc).to(DEVICE)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        print("[ImageEnsemble] Fallback: single model")
        return [m], ALL_CLASSES, len(ALL_CLASSES)

    raise FileNotFoundError(
        "No image model found. Train with "
        "--mode letter --kfold first."
    )


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess a BGR image for prediction.
    Returns numpy array (3, 128, 128) float32.
    """
    if img_bgr.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img


@torch.no_grad()
def image_ensemble_predict(
    models: list,
    img_chw: np.ndarray,
) -> tuple:
    """
    Ensemble prediction on a single preprocessed image.

    Args:
        models: list of SignImageCNN models
        img_chw: (3, H, W) float32 array

    Returns:
        (pred_idx, confidence, probs_array)
    """
    tensor = torch.from_numpy(img_chw).unsqueeze(0).to(DEVICE)
    all_probs = []

    for model in models:
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy()[0])

    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = int(np.argmax(avg_probs))
    confidence = float(avg_probs[pred_idx])

    return pred_idx, confidence, avg_probs
