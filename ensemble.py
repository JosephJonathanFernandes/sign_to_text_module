"""
Ensemble inference: load all K-fold models and average their softmax
outputs for more robust predictions. Includes test-time augmentation (TTA).
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    DEVICE, ENSEMBLE_DIR, PROCESSED_DIR,
    MODEL_SAVE_PATH,
    INPUT_SIZE,
    FRAME_FEAT_DIM, PROXIMITY_FEAT_DIM, PROXIMITY_INDEX,
)
from model import SignLanguageGRU


# Number of TTA (test-time augmentation) forward passes
TTA_ROUNDS = 5


def _tta_augment(seq: np.ndarray) -> np.ndarray:
    """
    Light test-time augmentation: small noise + tiny scaling.
    Much gentler than training augmentation.
    """
    seq = seq.copy()
    # Small Gaussian noise
    seq += np.random.randn(*seq.shape).astype(np.float32) * 0.008
    # Tiny scale jitter
    scale = np.random.uniform(0.96, 1.04)
    seq *= scale
    return seq


def _align_sequence_dim(seq: np.ndarray) -> np.ndarray:
    """Pad/truncate sequence feature dimension to current INPUT_SIZE."""
    feat_dim = seq.shape[1]
    if feat_dim == INPUT_SIZE:
        return seq
    if feat_dim > INPUT_SIZE:
        return seq[:, :INPUT_SIZE]

    pad = np.zeros((seq.shape[0], INPUT_SIZE - feat_dim), dtype=np.float32)
    return np.concatenate([seq, pad], axis=1)


def load_ensemble():
    """
    Load all fold models from ENSEMBLE_DIR.
    Returns (models_list, classes, num_classes).
    Falls back to single model.pth if ensemble dir is empty.
    """
    models = []
    classes = None
    current_classes = sorted([
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ])

    # Try loading ensemble fold models
    if os.path.isdir(ENSEMBLE_DIR):
        fold_files = sorted([
            f for f in os.listdir(ENSEMBLE_DIR) if f.endswith(".pth")
        ])
        for fname in fold_files:
            fpath = os.path.join(ENSEMBLE_DIR, fname)
            ckpt = torch.load(fpath, map_location=DEVICE, weights_only=False)
            ckpt_classes = ckpt.get("classes")

            # Skip stale folds that don't match current processed classes
            if ckpt_classes is not None:
                if sorted(ckpt_classes) != current_classes:
                    print(
                        f"[Ensemble] Skipping stale fold: {fname} "
                        f"(checkpoint classes != current processed classes)"
                    )
                    continue

            num_classes = ckpt["num_classes"]
            model = SignLanguageGRU(num_classes=num_classes).to(DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            models.append(model)
            if classes is None and "classes" in ckpt:
                classes = ckpt["classes"]

    if models:
        if classes is None:
            classes = current_classes
        print(
            f"[Ensemble] Loaded {len(models)} fold models, "
            f"{len(classes)} classes"
        )
        return models, classes, len(classes)

    # Fallback: single model
    if os.path.exists(MODEL_SAVE_PATH):
        ckpt = torch.load(
            MODEL_SAVE_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
        num_classes = ckpt["num_classes"]
        model = SignLanguageGRU(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        classes = ckpt.get("classes")
        if classes is None or sorted(classes) != current_classes:
            classes = current_classes
        print(
            f"[Ensemble] Fallback: loaded single model, "
            f"{len(classes)} classes"
        )
        return [model], classes, num_classes

    raise FileNotFoundError(
        "No models found. Train with --kfold or --train first."
    )


def load_merged_ensemble_10_2():
    """
    Load ensemble with 5-fold main + 1 single model fallback.
    (Old models skipped due to feature dimension mismatch)
    
    Returns:
        (main_models, fallback_models, classes, num_classes)
    """
    main_models = []
    fallback_models = []
    classes = None
    
    print("[Ensemble] Loading ensemble...")
    
    # Load new folds (5 models for main ensemble)
    if os.path.isdir(ENSEMBLE_DIR):
        print(f"  Loading fold models from {ENSEMBLE_DIR}...")
        new_fold_files = sorted([
            f for f in os.listdir(ENSEMBLE_DIR) if f.endswith(".pth")
        ])
        for fname in new_fold_files[:5]:  # Limit to 5
            fpath = os.path.join(ENSEMBLE_DIR, fname)
            try:
                ckpt = torch.load(fpath, map_location=DEVICE, weights_only=False)
                num_classes = ckpt["num_classes"]
                model = SignLanguageGRU(num_classes=num_classes).to(DEVICE)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                main_models.append(model)
                if classes is None and "classes" in ckpt:
                    classes = ckpt["classes"]
                print(f"    ✓ {fname}")
            except Exception as e:
                print(f"    ✗ {fname} - {e}")
    
    # Load single fallback model
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"  Loading fallback model from {MODEL_SAVE_PATH}...")
        try:
            ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
            num_classes = ckpt["num_classes"]
            model = SignLanguageGRU(num_classes=num_classes).to(DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            fallback_models.append(model)
            if classes is None and "classes" in ckpt:
                classes = ckpt["classes"]
            print(f"    ✓ model.pth")
        except Exception as e:
            print(f"    ✗ model.pth - {e}")
    
    if classes is None:
        current_classes = sorted([
            d for d in os.listdir(PROCESSED_DIR)
            if os.path.isdir(os.path.join(PROCESSED_DIR, d))
        ])
        classes = current_classes
    
    if not main_models and not fallback_models:
        raise FileNotFoundError("No models found for ensemble.")
    
    print(f"[Ensemble] Loaded {len(main_models)} main models + {len(fallback_models)} fallback, {len(classes)} classes")
    
    return main_models, fallback_models, classes, len(classes)



@torch.no_grad()
def ensemble_predict(
    models: list,
    sequence: np.ndarray,
    use_tta: bool = True,
) -> tuple:
    """
    Run ensemble prediction with optional test-time augmentation.

    Args:
        models: list of trained SignLanguageGRU models
        sequence: numpy array of shape (NUM_FRAMES, feat_dim)
        use_tta: whether to apply test-time augmentation

    Returns:
        (pred_idx, confidence, all_probs) where
        all_probs is a numpy array of shape (num_classes,)
    """
    all_probs = []

    tta_seqs = [sequence]
    if use_tta and TTA_ROUNDS > 1:
        for _ in range(TTA_ROUNDS - 1):
            tta_seqs.append(_tta_augment(sequence))

    for seq in tta_seqs:
        seq = _align_sequence_dim(seq)
        tensor = torch.from_numpy(seq).unsqueeze(0).float().to(DEVICE)
        if PROXIMITY_FEAT_DIM > 0 and tensor.shape[-1] >= FRAME_FEAT_DIM:
            proximity = tensor[:, :, PROXIMITY_INDEX]
        else:
            proximity = None

        for model in models:
            logits = model(tensor, proximity=proximity)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])

    # Average all probabilities (models x TTA rounds)
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = int(np.argmax(avg_probs))
    confidence = float(avg_probs[pred_idx])

    return pred_idx, confidence, avg_probs


@torch.no_grad()
def merged_ensemble_predict(
    main_models: list,
    fallback_models: list,
    sequence: np.ndarray,
    use_tta: bool = True,
) -> dict:
    """
    Run merged 10+2 ensemble prediction.
    
    - Primary: predictions from 10-model main ensemble (5 old + 5 new folds)
    - Fallback: predictions from 2-model fallback ensemble (old + new single models)
    - Returns both for analysis and fallback capability
    
    Args:
        main_models: list of 10 fold models (5 old + 5 new)
        fallback_models: list of 2 single models (1 old + 1 new)
        sequence: numpy array of shape (NUM_FRAMES, feat_dim)
        use_tta: whether to apply test-time augmentation
    
    Returns:
        dict with keys:
        - 'main': (pred_idx, confidence, probs) from 10-model ensemble
        - 'fallback': (pred_idx, confidence, probs) from 2-model ensemble
        - 'pred_idx': final prediction index
        - 'confidence': final confidence
        - 'probs': final probability distribution
    """
    result = {}
    
    # Main ensemble prediction (10 models)
    if main_models:
        idx, conf, probs = ensemble_predict(main_models, sequence, use_tta=use_tta)
        result['main'] = (idx, conf, probs)
        result['pred_idx'] = idx
        result['confidence'] = conf
        result['probs'] = probs
    else:
        result['main'] = None
    
    # Fallback ensemble prediction (2 models)
    if fallback_models:
        idx, conf, probs = ensemble_predict(fallback_models, sequence, use_tta=use_tta)
        result['fallback'] = (idx, conf, probs)
        
        # If main not available, use fallback
        if 'pred_idx' not in result:
            result['pred_idx'] = idx
            result['confidence'] = conf
            result['probs'] = probs
    else:
        result['fallback'] = None
    
    return result

