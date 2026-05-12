"""
Quick smoke training: run one fold for 1 epoch with reduced batch size.
Run: python smoke_train.py
"""
import os
import numpy as np
import torch
import train
from config import get_config
from pipeline_logger import setup_pipeline_logger

# Adjust for quick smoke
train.NUM_EPOCHS = 1
train.BATCH_SIZE = 8

cfg = get_config()
full_ds = train.ISLDataset(augment=False, min_samples=2)
num_classes = full_ds.num_classes
labels = np.array([lbl for _, lbl in full_ds.samples])

# Build folds
folds = train._build_source_aware_folds(full_ds.samples, labels, cfg.paths.num_folds, cfg.training.random_seed)
val_idx = folds[0]
train_idx = np.setdiff1d(np.arange(len(labels)), val_idx)

save_path = os.path.join(cfg.paths.ensemble_dir, "smoke_fold.pth")
print(f"Running smoke train: train={len(train_idx)} val={len(val_idx)}")

logger = setup_pipeline_logger("smoke")
best_acc = train._train_fold(full_ds, train_idx, val_idx, num_classes, 0, save_path, pipeline_log=logger)
print(f"Smoke fold best acc: {best_acc:.2f}%")
