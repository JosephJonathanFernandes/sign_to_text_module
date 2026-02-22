"""
Configuration for the image-based ISL letter/number recognition pipeline.
Uses static hand gesture images from archive/, archive (1)/ and archive (2)/.
"""

import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE1_DIR = os.path.join(BASE_DIR, "archive (1)", "Indian")
ARCHIVE2_DIR = os.path.join(BASE_DIR, "archive (2)", "ISL_Dataset")
ARCHIVE3_DIR = os.path.join(BASE_DIR, "archive", "original_images")
IMG_PROCESSED_DIR = os.path.join(BASE_DIR, "processed_images")
IMG_MODEL_PATH = os.path.join(BASE_DIR, "model_image.pth")
IMG_ENSEMBLE_DIR = os.path.join(BASE_DIR, "ensemble_image")

# ─── Image settings ─────────────────────────────────────────────────
IMG_SIZE = 128                # Native resolution of archive images
IMG_CHANNELS = 3              # RGB
NUM_IMG_CLASSES = 36          # A-Z (26) + 0-9 (10)

# ─── Model ───────────────────────────────────────────────────────────
IMG_DROPOUT = 0.4

# ─── Training ────────────────────────────────────────────────────────
IMG_BATCH_SIZE = 128
IMG_NUM_EPOCHS = 15
IMG_LEARNING_RATE = 2e-3
IMG_WEIGHT_DECAY = 1e-4
IMG_LABEL_SMOOTHING = 0.1
IMG_PATIENCE = 5              # Early stopping
IMG_SCHEDULER_PATIENCE = 3
IMG_GRAD_CLIP = 1.0
IMG_VAL_SPLIT = 0.15
IMG_NUM_FOLDS = 3
IMG_NUM_WORKERS = 2           # Parallel data loading
IMG_RANDOM_SEED = 42

# ─── Device ──────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")
NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)
