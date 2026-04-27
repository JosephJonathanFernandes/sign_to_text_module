# ISL Sign To Text

Real-time Indian Sign Language word recognition using hand landmarks and a BiGRU-based classifier.

## Features

- **Advanced landmark extraction:** Hand + face-relative features (506 dims per frame)
- **BiGRU + Attention model** with face-proximity biased attention weighting
- **Single model training and K-fold ensemble** (5 models, 95.83% accuracy)
- **Live webcam inference** with real-time sentence construction
- **Temporal Post-Processing** (NEW): Confidence-weighted smoothing + anti-flicker stabilization
- **Hand Selection** (NEW): Face-centric single-person hand filtering for multi-person robustness
- **Automatic sentence building** — signs accumulate into coherent text with NLP post-processing
- **Webcam data collection** for new samples
- **Runtime signer validation:**
  - Detects and validates both hands belong to same person (IoU-based)
  - Shows on-screen bounding boxes and confidence scores
- **Advanced augmentation:** Mixup, CutMix, noise, scale/rotation jitter
- **Loss functions:** Focal Loss (hard sample mining) + class-weighted Cross-Entropy
- **NLP post-processing:** Grammar correction, punctuation insertion, text normalization
- **Motion gating & dynamic thresholds** — adaptive confidence based on hand velocity

## Sign Classes (56+)

**Pronouns:** I, he, she, it, we, you, you all, they

**Adjectives:** beautiful, ugly, loud, quiet, happy, sad, deaf, blind, nice, rich, poor, thick, thin, expensive, cheap, flat, curved, male, female, tight, loose

**Greetings:** Hello, How are you, Alright, Good Morning, Good afternoon, Good evening, Good night

**Other:** Thank you, Pleased, Good, Idle, Morning

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Minimal packages in requirements:

- torch
- numpy
- opencv-python
- mediapipe

## Project Layout

- **main.py** — CLI entry point & orchestration  
- **preprocess.py** — Video to landmark preprocessing (MediaPipe extraction)
- **dataset.py** — PyTorch dataset with on-the-fly augmentation & oversampling
- **model.py** — BiGRU + Attention architecture
- **train.py** — Training loop, K-fold cross-validation, loss functions
- **ensemble.py** — K-fold ensemble loading & test-time augmentation
- **webcam.py** — Live webcam prediction with signer validation
- **sentence_builder.py** — Continuous sign→text conversion with word smoothing
- **nlp_postprocessor.py** — Grammar correction, punctuation, normalization  
- **collect_data.py** — Webcam sample collection utility
- **config.py** — Hyperparameters & settings
- **model.pth** — Trained single model checkpoint
- **ensemble/fold_*.pth** — 5 K-fold ensemble model checkpoints
- **Dataset/** — Raw video files organized by class
- **processed/** — Preprocessed landmark .npy files (20 frames × 506 dims)

## Quick Start

### 1) Preprocess and Train

```bash
python main.py --preprocess
python main.py --train
```

### 2) Optional: K-fold Ensemble

```bash
python main.py --kfold
```

### 3) Predict From a Video

```bash
python main.py --predict path/to/video.mp4
```

### 4) Run Live Webcam

```bash
python main.py --webcam
```

Press Q or ESC to quit.

### 5) Collect New Webcam Samples

Interactive mode:

```bash
python main.py --collect
```

Direct class mode:

```bash
python main.py --collect --cls happy --n 10
```

## Model Architecture

**BiGRU + Attention with Face-Proximity Weighting**

```
Input (20 frames × 506 features)
  ├─ Raw coordinates: 126 dims (21 landmarks × 3 coords × 2 hands)
  ├─ Face-relative coordinates: 126 dims
  ├─ Proximity score: 1 dim
  └─ Velocity (deltas): ×2 all above
    ↓
LayerNorm + FC projection
    ↓
Bidirectional GRU (64 hidden, 40% dropout)
    ↓
Face-Proximity Attention (learnable soft attention with biased weighting)
    ↓
FC classifier (56 classes)
    ↓
Output logits
```

**Key Technical Features:**
- **Focal Loss** (γ=2.0) — Focuses on hard-to-classify samples
- **Class-weighted loss** — Handles imbalanced data using inverse frequency
- **Mixup augmentation** (α=0.3) — Blends training samples for regularization
- **Face-proximity biased attention** — Frames with hands near face weighted higher
- **Motion gating** — Dynamically adjusts confidence based on hand velocity
- **Label smoothing** (15%) — Prevents overconfident predictions

## Performance

- **K-Fold Ensemble:** **95.83% mean accuracy** (506 dims, 5 models averaged)
- **Real-time webcam:** ~30 FPS on Intel Iris Xe GPU

## Webcam Signer Validation (What You Will See)

In webcam mode, the app draws:

- Person boxes (if detected)
- Hand boxes with Left/Right labels
- Assignment label per hand
- Status text:
  - Single-hand sign mode
  - Same person: YES
  - Same person: NO
  - Same person: waiting

Prediction update logic:

- One-hand sign visible: prediction is allowed
- Two-hand sign visible: prediction is allowed when pair is validated as same signer
- Invalid pair/no hand for a short period: rolling window resets

## Notes

- The same-person check is a runtime safety gate. It does not require re-recording the dataset.
- If person detection is unstable in a specific environment, use better lighting and a clean background for best results.

## Troubleshooting

- Webcam not opening:
  - Close apps using the camera
  - Check camera permissions in Windows privacy settings

- No model found warning:
  - Run training first or ensure ensemble weights exist in the ensemble folder

- Low confidence predictions:
  - Improve lighting
  - Keep hand signs centered and steady
  - Collect more balanced samples per class

## License

For academic and learning use unless a separate license is added.
