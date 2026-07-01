# Inference Pipeline

## Overview

The live inference pipeline runs at 30 FPS and delivers end-to-end predictions in under 200 ms. It uses ONNX INT8 as the primary inference backend with automatic PyTorch FP32 fallback.

## Stage-by-Stage Latency Budget

| Stage | Target Latency | Notes |
|---|---|---|
| Frame capture | ~0.5 ms | OpenCV VideoCapture |
| MediaPipe (detect frame) | 30–40 ms | Every 5 frames |
| MediaPipe (cached frame) | < 5 ms | Reused landmarks |
| Feature construction | < 2 ms | Pre-allocated buffers |
| ONNX INT8 inference | 15–30 ms | Full 20-frame window |
| Heuristic adjustment | < 1 ms | Multiplicative penalty |
| Temporal post-processing | < 1 ms | Sliding window |
| Sentence builder | < 1 ms | Deque + state machine |
| **Total (detect frame)** | **~80–120 ms** | |
| **Total (cached frame)** | **~30–50 ms** | |

## Stage 1 — Adaptive Landmark Detection

**File:** `src/core/webcam.py`

Detection is gated by an adaptive interval counter:

```
base interval = 5 frames
low-motion multiplier = 2.0 (up to max 8 frames)
forced re-detect = every 15 frames
```

- **Hand detection (on interval):** MediaPipe HandLandmarker detects up to 2 hands, returning 21 normalized (x,y,z) landmarks per hand and `handedness` (Left/Right label)
- **Face detection (on interval):** MediaPipe FaceLandmarker returns 478 face landmarks; only 3 are used (nose tip=1, left eye=33, right eye=263)
- **Cached frames:** Previous landmark results reused directly, skipping MediaPipe entirely

HOG-based person detection is **disabled** (`disable_hog_detection=True`) — saves ~8 ms/frame, no accuracy loss.

## Stage 2 — Feature Vector Construction

**File:** `src/preprocessing/preprocess.py` → `extract_landmarks_with_face_relative()`

For each frame:

1. **Raw hand block** (126 dims): left hand 63 dims + right hand 63 dims, in MediaPipe normalized coordinates
2. **Face-relative block** (126 dims): each landmark expressed as `(lm - nose_tip) / inter_eye_distance`; zero-filled if face not detected
3. **Proximity scalar** (1 dim): L2 distance from hand centroid to nose tip
4. **Velocity delta** (253 dims): `f_t - f_{t-1}`; zero at frame 0

Total: **506 dims per frame**

Pre-allocated module-level buffers (`_LANDMARK_BUFFERS`) avoid per-frame NumPy allocation.

## Stage 3 — Sequence Buffer

`collections.deque(maxlen=20)` accumulates frames. Once full, inference is triggered on every new frame (sliding window). Buffer → `np.array` of shape `(1, 20, 506)`.

## Stage 4 — ONNX INT8 Inference

**File:** `src/inference/onnx_inference.py` → `ONNXModelWrapper`

1. Feature dimension alignment (pad/truncate if input dim differs from model's expected dim)
2. Proximity vector rank adjustment
3. ONNX Runtime session invoke
4. On failure → automatic PyTorch FP32 fallback

Output: logits `(1, 78)` → softmax → probability vector `(78,)`

**Ensemble mode:** `src/inference/onnx_ensemble_integration.py` averages predictions from up to 5 fold checkpoints (configured via `LiveInferenceConfig.ensemble_size`).

## Stage 5 — Soft Heuristic Adjustment Layer

**File:** `src/core/webcam.py`

Before temporal smoothing, the raw GRU probabilities are adjusted via deterministic heuristic rules based on JSON metadata (`data/hand_sign_classification.json`):

1. **Feature Extraction:** Live webcam features (e.g., hand count) are continuously tracked.
2. **Confidence Gating:** Heuristics are only applied if the live visual confidence (e.g., MediaPipe hand presence) exceeds `0.7`.
3. **Multiplicative Penalty:** If the live feature mismatches the expected sign feature (e.g., seeing 1 hand for a `two_hands` sign), a multiplicative penalty (e.g., `0.6`) is applied to that sign's probability.
4. **Dominance:** This soft penalty ensures the model remains the primary decider, safely reshaping probabilities without catastrophic hard-filtering.

## Stage 6 — Temporal Post-Processing

**File:** `src/inference/temporal_postprocessor.py`

### ConfidenceSmoother

- Maintains a sliding window deque of 8 probability vectors
- Each entry weighted by: `confidence_score × exp_decay^(age)`; decay factor = 0.3
- More recent frames carry proportionally more weight
- Weighted average is renormalized to sum to 1

### StablePredictor

- Maintains a candidate class + patience counter
- A class switch is confirmed only when:
  - Same class predicted for **3 consecutive frames**
  - Smoothed confidence exceeds current stable class by **≥ 0.12** (hysteresis)

## Stage 7 — Momentum Commit

**File:** `src/core/webcam.py`

A sign is committed to the sentence when:
- **3-of-5 majority:** the class appears ≥ 3 times in the 5 most recent stable predictions
- **Minimum confidence:** average confidence of agreeing predictions ≥ 0.60
- **Ambiguity delay:** if top-1 minus top-2 probability < 0.05, wait 4 additional frames

## Stage 8 — Sentence Builder + NLP

**File:** `src/inference/sentence_builder.py`, `src/inference/nlp_postprocessor.py`

The `SentenceBuilder` uses a debouncing state machine to prevent noise during rapid gesture sequences:

- **State Machine Debounce:** `__reject__` or `__transition__` classes are ignored unless they persist for ≥ 3 consecutive frames (`separator_counter`).
- **Duplicate Suppression:** Prevents appending the exact same word consecutively.
- **Idle Timeout:** If no hands are detected for `30 frames`, the state resets.
- `nlp_postprocessor` applies: capitalization, grammatical connectors, punctuation normalization

## Confidence Threshold System

| Threshold | Value | Purpose |
|---|---|---|
| Base | 0.12 | Minimum confidence to accept any prediction |
| Hysteresis | 0.12 | Minimum delta to switch stable class |
| Ambiguity margin | 0.05 | Top-1 minus Top-2 gap; triggers ambiguity delay below |
| Similar-class penalty | +0.08 | Extra threshold for visually confusable sign pairs |
| Momentum confidence | 0.60 | Min average confidence to commit a sign |

## Running Live Inference

```bash
python main.py --webcam

# With quantized ONNX model explicitly:
python main.py --webcam --quantized --quantized-model models/model_int8.onnx

# Single model only (faster, less accurate):
# Set LiveInferenceConfig.ensemble_size = 1 in config.py
```

## Keyboard Controls

| Key | Action |
|---|---|
| `U` | Undo last committed word |
| `C` | Clear entire sentence |
| `Q` / `ESC` | Quit webcam |
