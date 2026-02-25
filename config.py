"""
Configuration constants for ISL Word Recognition Pipeline.
CPU-optimized settings for Intel Iris Xe.
"""

import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model.pth")
ENSEMBLE_DIR = os.path.join(BASE_DIR, "ensemble")  # K-fold models
NUM_FOLDS = 5                 # K-fold cross-validation

# ─── Preprocessing ──────────────────────────────────────────────────
NUM_FRAMES = 20              # Frames sampled per video (must match webcam)
NUM_LANDMARKS = 21           # MediaPipe hand landmarks
NUM_COORDS = 3               # x, y, z per landmark
NUM_HANDS = 2                # Both hands captured (right slot 0, left slot 1)
LANDMARK_DIM = NUM_LANDMARKS * NUM_COORDS          # 63  — per hand
FRAME_FEAT_DIM = LANDMARK_DIM * NUM_HANDS          # 126 — both hands per frame
USE_VELOCITY = True           # Append frame-to-frame deltas
INPUT_SIZE = FRAME_FEAT_DIM * 2 if USE_VELOCITY else FRAME_FEAT_DIM  # 252 or 126
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
HAND_LANDMARKER_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")

# ─── Model ───────────────────────────────────────────────────────────
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.40

# ─── Training ────────────────────────────────────────────────────────
BATCH_SIZE = 4
NUM_EPOCHS = 25
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 3e-4            # L2 regularization
LABEL_SMOOTHING = 0.1          # Softens targets
PATIENCE = 20                  # Early stopping patience
SCHEDULER_PATIENCE = 7         # LR scheduler patience
GRAD_CLIP = 1.0                # Gradient clipping max norm
VAL_SPLIT = 0.2              # 80/20 train/val split
RANDOM_SEED = 42

# ─── Device (CPU-only for Intel Iris Xe) ─────────────────────────────
DEVICE = torch.device("cpu")
NUM_THREADS = 4               # Adjust based on your CPU core count
torch.set_num_threads(NUM_THREADS)

# ─── Webcam ──────────────────────────────────────────────────────
WEBCAM_RECORD_FRAMES = 90      # Raw frames to capture (~3s at 30fps)
                                # Uniformly sub-sampled to NUM_FRAMES
WEBCAM_COUNTDOWN = 3           # Countdown seconds before recording

# ─── Prediction ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75    # Show prediction only above 75%
PREDICTION_SMOOTHING_WINDOW = 10  # Majority vote window size

print(f"[Config] Device: {DEVICE} | Threads: {NUM_THREADS}")
print(f"[Config] Sequence shape: (batch, {NUM_FRAMES}, {INPUT_SIZE})")
