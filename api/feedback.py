import asyncio
import logging
import os
import time
from datetime import datetime
from typing import List

import numpy as np
from fastapi import FastAPI
import torch

from src.core.config import get_config
from src.inference.onnx_ensemble import ensemble_predict_mixed
from src.training.adapter_model import AdapterTrainer, AdapterModel
from src.utils.pseudo_utilities import load_pseudo_data

logger = logging.getLogger("sign_to_text.api.feedback")
cfg = get_config()


def _save_feedback_data(sequence: np.ndarray, correct_word: str, session_id: str):
    """Save the corrected sequence to the pseudo_data directory."""
    # Ensure word is lowercase for directory matching
    correct_word = correct_word.lower()
    save_dir = os.path.join(cfg.paths.pseudo_data_dir, correct_word)
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = int(time.time() * 1000)
    filename = f"{timestamp}_{session_id}.npy"
    filepath = os.path.join(save_dir, filename)
    
    np.save(filepath, sequence)
    logger.info("feedback_saved", extra={"word": correct_word, "file": filename})


def _train_adapter_sync(app: FastAPI):
    """Synchronous function to train adapter on all pseudo_data."""
    if not hasattr(app.state, "models") or not app.state.models:
        logger.warning("No base models found. Cannot train adapter.")
        return

    logger.info("adapter_training_started", extra={"event": "adapter_train_start"})
    
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
        
        # 5. Hot-reload the adapter in the API state!
        new_adapter = AdapterModel(app.state.num_classes, hidden_dim=cfg.training.adapter_hidden_dim).to(device)
        new_adapter.load_weights(save_path)
        new_adapter.eval()
        app.state.adapter_model = new_adapter
        logger.info("adapter_reloaded_in_api", extra={"event": "adapter_hot_reload"})
    else:
        logger.error("adapter_training_failed", extra={"reason": result.get("reason")})


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
    
    loop = asyncio.get_running_loop()
    
    # Save the file (fast I/O)
    await loop.run_in_executor(
        None,
        _save_feedback_data,
        seq_np,
        correct_word,
        session_id
    )
    
    # Train the adapter on the updated dataset (slower computation)
    await loop.run_in_executor(
        None,
        _train_adapter_sync,
        app
    )
