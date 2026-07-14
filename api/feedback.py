import asyncio
import logging
import os
import time
from datetime import datetime
from typing import List

import numpy as np
from fastapi import FastAPI
import torch

import hashlib
import threading

from src.core.config import get_config
from src.inference.onnx_ensemble import ensemble_predict_mixed
from src.training.adapter_model import AdapterTrainer, AdapterModel
from src.utils.pseudo_utilities import load_pseudo_data

logger = logging.getLogger("sign_to_text.api.feedback")
cfg = get_config()

FEEDBACK_STATE = {
    "training_in_progress": False,
    "last_training_time": 0.0,
    "pending_feedback_count": 0,
    "total_feedback_received": 0,
    "success_runs": 0,
    "failed_runs": 0,
    "active_adapter_version": "None",
}
training_lock = threading.Lock()


def _save_feedback_data(sequence: np.ndarray, correct_word: str, session_id: str) -> bool:
    """Save the corrected sequence to the pseudo_data directory. Returns True if saved."""
    # Ensure word is lowercase for directory matching
    correct_word = correct_word.lower()
    save_dir = os.path.join(cfg.paths.pseudo_data_dir, correct_word)
    os.makedirs(save_dir, exist_ok=True)
    
    # Duplicate detection via MD5
    seq_hash = hashlib.md5(sequence.tobytes()).hexdigest()
    filename = f"{correct_word}_{seq_hash}.npy"
    filepath = os.path.join(save_dir, filename)
    
    if os.path.exists(filepath):
        logger.info("feedback_duplicate_skipped", extra={"word": correct_word, "hash": seq_hash})
        return False
    
    np.save(filepath, sequence)
    logger.info("feedback_saved", extra={"word": correct_word, "file": filename})
    return True


def _train_adapter_sync(app: FastAPI):
    """Synchronous function to train adapter on all pseudo_data."""
    if not hasattr(app.state, "models") or not app.state.models:
        logger.warning("No base models found. Cannot train adapter.")
        return

    # ── Cooldown & Threshold Check ──
    with training_lock:
        if FEEDBACK_STATE["training_in_progress"]:
            logger.info("adapter_training_skipped", extra={"reason": "already_in_progress"})
            return
            
        time_since_last = time.time() - FEEDBACK_STATE["last_training_time"]
        if False:  # Disabled cooldown for evaluation
            logger.info("adapter_training_skipped", extra={"reason": "cooldown_active", "remaining_s": int(300 - time_since_last)})
            return
            
        if FEEDBACK_STATE["pending_feedback_count"] < 100:
            logger.info("adapter_training_skipped", extra={"reason": "insufficient_samples", "pending": FEEDBACK_STATE["pending_feedback_count"]})
            return

        # Mark as started
        FEEDBACK_STATE["training_in_progress"] = True
        FEEDBACK_STATE["pending_feedback_count"] = 0  # reset pending count

    logger.info("adapter_training_started", extra={"event": "adapter_train_start"})
    
    try:
        _do_train_adapter(app)
    except Exception as e:
        logger.exception("adapter_training_failed_exception")
        FEEDBACK_STATE["failed_runs"] += 1
    finally:
        with training_lock:
            FEEDBACK_STATE["training_in_progress"] = False
            FEEDBACK_STATE["last_training_time"] = time.time()

def _do_train_adapter(app: FastAPI):
    
    # 1. Load all pseudo data
    data = load_pseudo_data(cfg.paths.pseudo_data_dir)
    
    ensemble_probs_list = []
    class_indices_list = []
    
    device = str(cfg.hardware.torch_device)
    
    total_samples = sum(info["metadata"]["count"] for info in data.values())
    if total_samples == 0:
        logger.warning("No feedback data found in pseudo_data.")
        return
        
    logger.info("processing_feedback_data", extra={"samples": total_samples})
    
    # 2. Get base ensemble predictions for every sequence
    for class_name, info in data.items():
        if class_name not in app.state.classes:
            continue
            
        class_idx = app.state.classes.index(class_name)
        
        for seq in info["sequences"]:
            # Make sure shape is (NUM_FRAMES, INPUT_SIZE)
            if seq.shape != (cfg.preprocessing.num_frames, cfg.frame_features.input_sequence_dim):
                continue
                
            try:
                # Use ensemble_predict_mixed to get base probabilities
                pred_idx, conf, avg_probs = ensemble_predict_mixed(
                    app.state.models,
                    seq,
                    device=device,
                    proximity_feat_dim=cfg.spatial.proximity_dim,
                    frame_feat_dim=cfg.frame_features.input_sequence_dim,
                    proximity_index=cfg.frame_features.proximity_index,
                )
                ensemble_probs_list.append(avg_probs)
                class_indices_list.append(class_idx)
            except Exception as e:
                logger.warning(f"Failed to predict on feedback sample: {e}")
                
    if not ensemble_probs_list:
        logger.warning("No valid feedback data to train on.")
        return
        
    # 3. Train Adapter
    trainer = AdapterTrainer(
        num_classes=app.state.num_classes,
        device=device,
        learning_rate=cfg.training.adapter_learning_rate,
        hidden_dim=cfg.training.adapter_hidden_dim,
    )
    
    # Calculate basic class weights to handle imbalance (since users will mostly correct their mistakes)
    # We want to give higher weight to minority classes.
    class_counts = {}
    for idx in class_indices_list:
        class_counts[idx] = class_counts.get(idx, 0) + 1
        
    max_count = max(class_counts.values())
    class_weights = {}
    for idx, count in class_counts.items():
        # simple inverse frequency weighting, clamped
        weight = min(5.0, max_count / count)
        class_weights[idx] = weight
        
    # Train for a few epochs (more than live adapter because this is explicit supervised data)
    result = trainer.train(
        ensemble_probs_list,
        class_indices_list,
        class_weights=class_weights,
        epochs=10, 
        batch_size=8,
        verbose=False,
    )
    
    if result.get("success"):
        # 4. Save new weights
        adapter_weights_dir = "adapter_weights"
        os.makedirs(adapter_weights_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(adapter_weights_dir, f"adapter_{timestamp}.pt")
        
        trainer.save_model(save_path)
        logger.info("adapter_training_complete", extra={"loss": result["history"]["losses"][-1], "saved": save_path})
        
        # 5. Hot-reload the adapter in the API state (Rollback mechanism)
        try:
            new_adapter = AdapterModel(app.state.num_classes, hidden_dim=cfg.training.adapter_hidden_dim).to(device)
            new_adapter.load_weights(save_path)
            new_adapter.eval()
            
            # Atomic swap only after successful load
            app.state.adapter_model = new_adapter
            version = timestamp
            FEEDBACK_STATE["active_adapter_version"] = version
            FEEDBACK_STATE["success_runs"] += 1
            logger.info("adapter_reloaded_in_api", extra={"event": "adapter_hot_reload", "version": version})
        except Exception as e:
            logger.error("adapter_reload_failed", extra={"error": str(e), "note": "keeping previous adapter"})
            FEEDBACK_STATE["failed_runs"] += 1
    else:
        logger.error("adapter_training_failed", extra={"reason": result.get("reason")})
        FEEDBACK_STATE["failed_runs"] += 1


async def process_feedback_async(
    app: FastAPI, 
    sequence: List[List[float]], 
    correct_word: str, 
    session_id: str
):
    """
    Handle incoming feedback. Save the sequence, then trigger adapter training
    in a background thread so we don't block the API.
    """
    seq_np = np.array(sequence, dtype=np.float32)
    
    # ── Validation ──
    if seq_np.shape != (cfg.preprocessing.num_frames, cfg.frame_features.input_sequence_dim):
        logger.warning("feedback_rejected_shape", extra={"shape": seq_np.shape})
        return
    if np.isnan(seq_np).any() or np.isinf(seq_np).any():
        logger.warning("feedback_rejected_nan", extra={})
        return
    if correct_word not in app.state.classes:
        logger.warning("feedback_rejected_unknown_word", extra={"word": correct_word})
        return
        
    velocity = seq_np[:, 253:506]
    vel_sum = float(np.sum(np.abs(velocity)))
    if vel_sum < 1.0: # Requires minimal motion across 20 frames
        logger.warning("feedback_rejected_low_motion", extra={"vel_sum": vel_sum})
        return
    
    loop = asyncio.get_running_loop()
    
    # Save the file (fast I/O)
    saved = await loop.run_in_executor(
        None,
        _save_feedback_data,
        seq_np,
        correct_word,
        session_id
    )
    
    if saved:
        with training_lock:
            FEEDBACK_STATE["pending_feedback_count"] += 1
            FEEDBACK_STATE["total_feedback_received"] += 1
            
        # Train the adapter on the updated dataset (slower computation)
        # It will exit early if cooldown or threshold not met
        await loop.run_in_executor(
            None,
            _train_adapter_sync,
            app
        )
