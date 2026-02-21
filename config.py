"""
Configuration constants for ISL Word Recognition Pipeline.
CPU-optimized settings for Intel Iris Xe.
"""

import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model.pth")

# ─── Preprocessing ──────────────────────────────────────────────────
NUM_FRAMES = 20              # Frames sampled per video (CPU-optimized)
NUM_LANDMARKS = 21           # MediaPipe hand landmarks
NUM_COORDS = 3               # x, y, z per landmark
INPUT_SIZE = NUM_LANDMARKS * NUM_COORDS  # 63
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
HAND_LANDMARKER_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")

# ─── Model ───────────────────────────────────────────────────────────
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.4

# ─── Training ────────────────────────────────────────────────────────
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4            # L2 regularization
LABEL_SMOOTHING = 0.1          # Softens targets
PATIENCE = 10                  # Early stopping patience
SCHEDULER_PATIENCE = 5         # LR scheduler patience
GRAD_CLIP = 1.0                # Gradient clipping max norm
VAL_SPLIT = 0.2              # 80/20 train/val split
RANDOM_SEED = 42

# ─── Device (CPU-only for Intel Iris Xe) ─────────────────────────────
DEVICE = torch.device("cpu")
NUM_THREADS = 4               # Adjust based on your CPU core count
torch.set_num_threads(NUM_THREADS)

print(f"[Config] Device: {DEVICE} | Threads: {NUM_THREADS}")
print(f"[Config] Sequence shape: (batch, {NUM_FRAMES}, {INPUT_SIZE})")
