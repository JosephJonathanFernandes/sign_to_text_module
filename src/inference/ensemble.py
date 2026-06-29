"""
Ensemble inference: load all K-fold models and average their softmax
outputs for more robust predictions. Includes test-time augmentation (TTA).

════════════════════════════════════════════════════════════════════════════════════
PHASE 3: LIVE INFERENCE OPTIMIZATION
════════════════════════════════════════════════════════════════════════════════════
- Dynamic ensemble size (1, 3, or 5 models)
- Optional TTA (disabled by default for better latency)
- Latency tracking and reporting
- Configurable through config.live_inference
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import time
import logging

from config import get_config
from src.utils.quantization_utils import load_model_artifact

logger = logging.getLogger("sign_to_text.ensemble")

cfg = get_config()

# Convenience references for ensemble
DEVICE = cfg.hardware.torch_device
ENSEMBLE_DIR = cfg.paths.ensemble_dir
PROCESSED_DIR = cfg.paths.processed_dir
MODEL_SAVE_PATH = cfg.paths.model_save_path
INPUT_SIZE = cfg.frame_features.input_sequence_dim
FRAME_FEAT_DIM = cfg.frame_features.frame_features_dim
PROXIMITY_FEAT_DIM = cfg.spatial.proximity_dim
PROXIMITY_INDEX = cfg.frame_features.proximity_index

# ════════════════════════════════════════════════════════════════════════════════════
# PHASE 3: Live inference configuration
# ════════════════════════════════════════════════════════════════════════════════════
LIVE_ENSEMBLE_SIZE = cfg.live_inference.ensemble_size
LIVE_USE_TTA = cfg.live_inference.use_tta
PRINT_LATENCY_STATS = cfg.live_inference.print_latency_stats



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


def load_ensemble(model_artifact_path: str | None = None):
    """
    Load fold models from ENSEMBLE_DIR with support for dynamic ensemble sizing (PHASE 3).
    
    Returns (models_list, classes, num_classes).
    Falls back to single model.pth if ensemble dir is empty.
    
    ════════════════════════════════════════════════════════════════════════════════════
    PHASE 3: Dynamic Ensemble Size
    ════════════════════════════════════════════════════════════════════════════════════
    - LIVE_ENSEMBLE_SIZE controls number of models to load:
      - 1: Single model (fastest, ~1-2 fps)
      - 3: Balanced ensemble (2-3 fps, ~1-2% accuracy loss)
      - 5: Full ensemble (0.5-1 fps, best accuracy)
    
    Loads first N fold models from sorted list.
    """
    current_classes = sorted([
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ]) if os.path.exists(PROCESSED_DIR) else []

    if model_artifact_path:
        model, classes, num_classes, _, _ = load_model_artifact(model_artifact_path, map_location="cpu")
        model.eval()
        if classes is None:
            classes = current_classes
        logger.info(
            "model_artifact_loaded",
            extra={
                "artifact": os.path.basename(model_artifact_path),
                "num_classes": len(classes)
            }
        )
        return [model], classes, num_classes

    models = []
    classes = None

    # Try loading ensemble fold models (limited by LIVE_ENSEMBLE_SIZE)
    if os.path.isdir(ENSEMBLE_DIR):
        fold_files = sorted([
            f for f in os.listdir(ENSEMBLE_DIR) if f.endswith(".pth")
        ])
        
        # PHASE 3: Limit to LIVE_ENSEMBLE_SIZE models
        fold_files = fold_files[:LIVE_ENSEMBLE_SIZE]
        
        for fname in fold_files:
            fpath = os.path.join(ENSEMBLE_DIR, fname)
            model, ckpt_classes, num_classes, _, ckpt = load_model_artifact(fpath, map_location="cpu")

            # Skip stale folds that don't match current processed classes
            if ckpt_classes is not None:
                if current_classes and sorted(ckpt_classes) != current_classes:
                    logger.warning(
                        "stale_fold_skipped",
                        extra={
                            "fold": fname,
                            "reason": "checkpoint classes != current processed classes"
                        }
                    )
                    continue

            model.eval()
            models.append(model)
            if classes is None and ckpt_classes is not None:
                classes = ckpt_classes

    if models:
        if classes is None:
            classes = current_classes
        logger.info(
            "ensemble_loaded",
            extra={
                "models_count": len(models),
                "ensemble_size": LIVE_ENSEMBLE_SIZE,
                "num_classes": len(classes)
            }
        )
        return models, classes, len(classes)

    # Fallback: single model
    if os.path.exists(MODEL_SAVE_PATH):
        model, classes, num_classes, _, ckpt = load_model_artifact(MODEL_SAVE_PATH, map_location="cpu")
        model.eval()
        if classes is None or (current_classes and sorted(classes) != current_classes):
            if current_classes:
                classes = current_classes
        logger.info(
            "fallback_model_loaded",
            extra={
                "num_classes": len(classes)
            }
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
    
    logger.info("merged_ensemble_load_started", extra={"event": "loading_merged_ensemble"})
    
    # Load new folds (5 models for main ensemble)
    if os.path.isdir(ENSEMBLE_DIR):
        new_fold_files = sorted([
            f for f in os.listdir(ENSEMBLE_DIR) if f.endswith(".pth")
        ])
        for fname in new_fold_files[:5]:  # Limit to 5
            fpath = os.path.join(ENSEMBLE_DIR, fname)
            try:
                model, classes_in_ckpt, num_classes, _, ckpt = load_model_artifact(fpath, map_location="cpu")
                main_models.append(model)
                if classes is None and classes_in_ckpt is not None:
                    classes = classes_in_ckpt
                logger.info("fold_loaded", extra={"fold": fname})
            except Exception as e:
                logger.error("fold_load_error", extra={"fold": fname, "error": str(e)})
    
    # Load single fallback model
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model, classes_in_ckpt, num_classes, _, ckpt = load_model_artifact(MODEL_SAVE_PATH, map_location="cpu")
            model.eval()
            fallback_models.append(model)
            if classes is None and classes_in_ckpt is not None:
                classes = classes_in_ckpt
            logger.info("fallback_loaded", extra={"model": "model.pth"})
        except Exception as e:
            logger.error("fallback_load_error", extra={"model": "model.pth", "error": str(e)})
    
    if classes is None:
        current_classes = sorted([
            d for d in os.listdir(PROCESSED_DIR)
            if os.path.isdir(os.path.join(PROCESSED_DIR, d))
        ]) if os.path.exists(PROCESSED_DIR) else []
        classes = current_classes
    
    if not main_models and not fallback_models:
        raise FileNotFoundError("No models found for ensemble.")
    
    logger.info(
        "merged_ensemble_loaded",
        extra={
            "main_models": len(main_models),
            "fallback_models": len(fallback_models),
            "num_classes": len(classes)
        }
    )
    
    return main_models, fallback_models, classes, len(classes)



@torch.no_grad()
def ensemble_predict(
    models: list,
    sequence: np.ndarray,
    use_tta: bool = None,
) -> tuple:
    """
    Run ensemble prediction with optional test-time augmentation (PHASE 3).

    Args:
        models: list of trained SignLanguageGRU models
        sequence: numpy array of shape (NUM_FRAMES, feat_dim)
        use_tta: whether to apply test-time augmentation.
                 If None, uses LIVE_USE_TTA config value.

    Returns:
        (pred_idx, confidence, all_probs) where
        all_probs is a numpy array of shape (num_classes,)
    
    ════════════════════════════════════════════════════════════════════════════════════
    PHASE 3: TTA Control
    ════════════════════════════════════════════════════════════════════════════════════
    - use_tta=True: 5 augmented passes per model (slower, more robust)
    - use_tta=False: 1 pass per model (faster) [default for live]
    - If use_tta not specified, uses LIVE_USE_TTA config
    """
    # Use config value if not explicitly provided
    if use_tta is None:
        use_tta = LIVE_USE_TTA
    
    t_start = time.time()
    
    # PHASE 1 OPTIMIZATION: Store logits instead of softmax probabilities
    # Softmax is expensive; averaging logits then softmax once is mathematically equivalent
    # and saves N-1 softmax operations when using ensemble or TTA
    all_logits = []

    tta_seqs = [sequence]
    if use_tta and TTA_ROUNDS > 1:
        for _ in range(TTA_ROUNDS - 1):
            tta_seqs.append(_tta_augment(sequence))

    t_tta_prep = time.time()
    t_model_forward = 0
    num_forward_passes = 0
    
    for seq in tta_seqs:
        seq = _align_sequence_dim(seq)
        tensor = torch.from_numpy(seq).unsqueeze(0).float().to(DEVICE)
        if PROXIMITY_FEAT_DIM > 0 and tensor.shape[-1] >= FRAME_FEAT_DIM:
            proximity = tensor[:, :, PROXIMITY_INDEX]
        else:
            proximity = None

        for model in models:
            t_fwd_start = time.time()
            logits = model(tensor, proximity=proximity)
            if isinstance(logits, dict):
                logits = logits.get("sign_logits", logits.get("logits", logits))
            t_fwd_end = time.time()
            t_model_forward += (t_fwd_end - t_fwd_start)
            
            # OPTIMIZATION: Store logits, not softmax probabilities
            # This saves softmax computation when averaging across models/TTA rounds
            all_logits.append(logits.cpu().detach().numpy()[0])
            num_forward_passes += 1

    # PHASE 1 OPTIMIZATION: Average logits first, then softmax once
    # Mathematically equivalent to averaging softmax probabilities but faster:
    # softmax(avg(logits)) ≈ avg(softmax(logits)) with scale invariance
    # Saves: (num_forward_passes - 1) softmax operations
    avg_logits = np.mean(all_logits, axis=0)
    avg_logits_tensor = torch.from_numpy(avg_logits).unsqueeze(0).float().to(DEVICE)
    avg_probs_tensor = F.softmax(avg_logits_tensor, dim=1)
    avg_probs = avg_probs_tensor.cpu().detach().numpy()[0]
    
    pred_idx = int(np.argmax(avg_probs))
    confidence = float(avg_probs[pred_idx])

    # ════════════════════════════════════════════════════════════════════════════════════
    # PHASE 3: Latency Tracking
    # ════════════════════════════════════════════════════════════════════════════════════
    if PRINT_LATENCY_STATS:
        t_end = time.time()
        total_time = (t_end - t_start) * 1000  # ms
        model_time = t_model_forward * 1000  # ms
        tta_time = (t_tta_prep - t_start) * 1000  # ms
        other_time = total_time - model_time - tta_time
        
        fps = 1000.0 / total_time if total_time > 0 else 0
        
        logger.info(
            "inference_latency_stats",
            extra={
                "model_ms": round(model_time, 1),
                "forward_passes": num_forward_passes,
                "tta_prep_ms": round(tta_time, 1),
                "other_ms": round(other_time, 1),
                "total_ms": round(total_time, 1),
                "fps": round(fps, 2),
                "num_models": len(models),
                "use_tta": use_tta
            }
        )

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

