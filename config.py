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
WEBCAM_WIDTH = 640           # Webcam frame width
WEBCAM_HEIGHT = 480          # Webcam frame height
CROP_TO_WEBCAM_SIZE = True   # Center-crop videos to match webcam dimensions
NUM_LANDMARKS = 21           # MediaPipe hand landmarks
NUM_COORDS = 3               # x, y, z per landmark
NUM_HANDS = 2                # Both hands captured (right slot 0, left slot 1)
LANDMARK_DIM = NUM_LANDMARKS * NUM_COORDS          # 63  — per hand
RAW_FRAME_FEAT_DIM = LANDMARK_DIM * NUM_HANDS      # 126 — raw hands per frame
USE_FACE_RELATIVE = True      # Append hand coords relative to face anchors
USE_SPATIAL_DISTANCE = False  # Disabled (not implemented in extraction yet)
SPATIAL_DISTANCE_DIM = NUM_LANDMARKS * 4 if USE_SPATIAL_DISTANCE else 0  # 84 per hand if enabled
REL_FRAME_FEAT_DIM = (LANDMARK_DIM + SPATIAL_DISTANCE_DIM) * NUM_HANDS if USE_FACE_RELATIVE else 0
PROXIMITY_FEAT_DIM = 1 if USE_FACE_RELATIVE else 0
FRAME_FEAT_DIM = (
    RAW_FRAME_FEAT_DIM + REL_FRAME_FEAT_DIM + PROXIMITY_FEAT_DIM
)
PROXIMITY_INDEX = FRAME_FEAT_DIM - 1 if PROXIMITY_FEAT_DIM else -1
USE_VELOCITY = True           # Append frame-to-frame deltas
INPUT_SIZE = FRAME_FEAT_DIM * 2 if USE_VELOCITY else FRAME_FEAT_DIM
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
HAND_LANDMARKER_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")
FACE_LANDMARKER_MODEL = os.path.join(BASE_DIR, "face_landmarker.task")
FACE_NOSE_INDEX = 1
FACE_LEFT_EYE_INDEX = 33
FACE_RIGHT_EYE_INDEX = 263
DEBUG_DRAW_FACE_CENTER = True

# ─── Model ───────────────────────────────────────────────────────────
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.40
USE_FACE_PROXIMITY_ATTENTION = True
PROXIMITY_SIGMA = 0.10
LEARNABLE_PROXIMITY_SIGMA = False

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
WEBCAM_RECORD_FRAMES = 90      # Raw frames to capture, then sub-sample
WEBCAM_COUNTDOWN = 3           # Countdown seconds before recording

# ─── Prediction ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35  # Base confidence threshold (adjusted dynamically)
PREDICTION_SMOOTHING_WINDOW = 10  # Majority vote window size

# ─── Motion Gating ──────────────────────────────────────────────────────
MOTION_GATING_ENABLED = True  # Gate predictions based on hand motion
MOTION_THRESHOLD = 8.0  # Pixels/frame to consider motion (normalized to 640x480)
MOTION_SMOOTHING = 0.7  # Exponential moving average factor for motion detection
IDLE_CONFIDENCE_THRESHOLD = 0.70  # Higher threshold for static hands

# ─── Dynamic Thresholds ──────────────────────────────────────────────────
DYNAMIC_THRESHOLD_ENABLED = True  # Adjust confidence threshold based on motion & stability
MOTION_BOOST_FACTOR = 0.15  # Reduce threshold by this amount when motion detected
STABILITY_BOOST_FACTOR = 0.10  # Reduce threshold as sign becomes more stable
DYNAMIC_THRESHOLD_MIN = 0.20  # Don't go below this threshold

# ─── Transition Logic ────────────────────────────────────────────────────
TRANSITION_HYSTERESIS = 0.12  # Min confidence delta to switch predictions
SIGN_IDLE_TIMEOUT = 30  # Frames before considering hands idle (1s @ 30fps)
SIMILAR_CLASS_PENALTY = 0.08  # Extra threshold for easily-confused classes

print(f"[Config] Device: {DEVICE} | Threads: {NUM_THREADS}")
print(f"[Config] Sequence shape: (batch, {NUM_FRAMES}, {INPUT_SIZE})")
