# System Design — ISL Sign-to-Text

## Overview

The ISL Sign-to-Text system is a sequential, multi-stage pipeline that transforms raw webcam frames into English text via Indian Sign Language word recognition.

## High-Level Data Flow

```
Webcam Frame
    │
    ▼
[Stage 1] Adaptive Landmark Detection
    │  HandLandmarker (every 5 frames, forced re-detect at 15)
    │  FaceLandmarker (every 5 frames)
    │
    ▼
[Stage 2] Feature Vector Construction (per frame)
    │  Raw hand coords:      126 dims (21 landmarks × 3 × 2 hands)
    │  Face-relative coords: 126 dims (normalized by inter-eye distance)
    │  Proximity scalar:       1 dim  (L2 hand-to-face distance)
    │  Velocity delta:       253 dims (frame-to-frame difference)
    │  Total:                506 dims
    │
    ▼
[Stage 3] Sequence Buffer
    │  Circular deque of 20 frames → shape (20, 506)
    │
    ▼
[Stage 4] Deep Learning Inference (ONNX INT8 → PyTorch fallback)
    │  Input: (1, 20, 506)
    │  Output: logits over 78 sign classes → softmax probabilities
    │
    ▼
[Stage 5] Temporal Post-Processing
    │  ConfidenceSmoother: 8-frame exponential decay weighted average
    │  StablePredictor:    3-frame patience + 0.12 hysteresis margin
    │
    ▼
[Stage 6] Momentum-Based Commit
    │  Require 3-of-5 majority + avg confidence ≥ 0.60
    │
    ▼
[Stage 7] Sentence Builder + NLP Post-Processor
       Accumulated text output → grammar cleanup → display
```

## Module Dependencies

```
src/core/main.py
    ├── src/preprocessing/preprocess.py    (landmark extraction)
    ├── src/training/train.py              (training loop)
    ├── src/inference/ensemble.py          (model loading)
    └── src/core/webcam.py                 (live inference)
            ├── src/core/config.py             (all settings)
            ├── src/inference/temporal_postprocessor.py
            ├── src/inference/sentence_builder.py
            ├── src/inference/nlp_postprocessor.py
            ├── src/inference/onnx_ensemble_integration.py
            │       └── src/inference/onnx_inference.py
            ├── src/inference/hand_selector.py
            └── src/training/adapter_training.py
                    └── src/training/adapter_model.py
```

## Configuration System

All pipeline parameters are centralized in `src/core/config.py` as validated Python dataclasses:

| Config Class | Responsibility |
|---|---|
| `PathsConfig` | All file and directory paths |
| `LandmarkConfig` | Feature dimensions (21 landmarks × 3 coords × 2 hands) |
| `SpatialFeaturesConfig` | Face-relative feature toggles |
| `FrameFeaturesConfig` | Per-frame dimension calculations |
| `PreprocessingConfig` | Frame sampling, webcam resolution, detection intervals |
| `ModelConfig` | Hidden size, layers, dropout, attention |
| `TrainingConfig` | Batch size, LR, epochs, mixup, class weighting |
| `InferenceConfig` | Confidence threshold, hysteresis, similar-class penalty |
| `MotionConfig` | Motion threshold (resolution-independent, normalized) |
| `HardwareConfig` | Device (CPU/CUDA), thread count |
| `LiveInferenceConfig` | Adapter training, temporal smoothing, momentum commit |
| `ArchitectureImprovementsConfig` | Phase 1–10 architecture toggles |

The master `Config` class aggregates all subsystems and exposes a `validate()` method and a `summary()` pretty-printer.

## Compatibility Shims

Root-level stub files preserve backward compatibility for legacy scripts:

```python
# root/config.py
from src.core.config import *

# root/model.py
from src.training.model import *

# root/train.py
from src.training.train import *
```

This means all existing scripts using `from config import get_config` continue to work without modification.
