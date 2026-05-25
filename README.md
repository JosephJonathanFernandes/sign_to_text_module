# ISL Sign To Text

Real-time Indian Sign Language word recognition using MediaPipe landmarks, a BiGRU-based sequence classifier, and lightweight sentence post-processing.

This repository currently implements a strong isolated-word recognition pipeline with continuous webcam inference, confidence smoothing, and rule-based text cleanup. It is a good base for adding context-aware sign translation on top.

## Features

- **Landmark-based pipeline:** MediaPipe hand + face-relative keypoints stored as compact `.npy` sequences
- **BiGRU + Attention classifier:** temporal sequence model for isolated sign recognition
- **K-fold ensemble inference:** more robust prediction by averaging multiple checkpoints
- **Real-time webcam mode:** sliding-window recognition with signer validation, confidence gating, periodic full hand re-detection, and drift-triggered cache refresh
- **Sentence builder:** accumulates recognized signs into continuous text
- **Rule-based NLP post-processing:** grammar cleanup, punctuation insertion, text normalization
- **Data collection utilities:** webcam capture for new training samples
- **Raw-video augmentation:** controlled dataset expansion before preprocessing
- **Landmark augmentation:** sequence-level augmentation on processed `.npy` files
- **Dataset balancing:** duplicate webcam samples and trim overfull classes to a fixed target
- **Balanced training:** class weights, mixup, optional focal loss, oversampling
- **Pseudo-label and adapter hooks:** experimental live adaptation pipeline for future personalization

## Sign Classes (78)

**Pronouns:** I, he, she, it, we, you, you all, they

**Adjectives:** beautiful, ugly, loud, quiet, happy, sad, deaf, blind, nice, rich, poor, thick, thin, expensive, cheap, flat, curved, male, female, tight, loose

**Greetings:** Hello, How are you, Alright, Good Morning, Good afternoon, Good evening, Good night

**Other:** Thank you, Pleased, Good, Idle, Morning

The active processed dataset currently contains 78 classes. If you add or remove class folders in `processed/`, retrain the model and ensemble checkpoints.

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

- **main.py** — CLI entry point and pipeline orchestration  
- **preprocess.py** — MediaPipe extraction and `.npy` generation
- **augmentations.py** — Landmark-sequence augmentation utilities
- **augment_pipeline.py** — Raw video augmentation orchestrator
- **dataset.py** — PyTorch dataset with augmentation and oversampling
- **model.py** — BiGRU + Attention sequence classifier
- **train.py** — Training loop, K-fold cross-validation, and loss functions
- **ensemble.py** — Ensemble loading and test-time augmentation
- **webcam.py** — Live webcam prediction, smoothing, and sentence building
- **sentence_builder.py** — Continuous sign-to-text assembly
- **temporal_postprocessor.py** — Optional temporal smoothing and stabilization
- **nlp_postprocessor.py** — Rule-based grammar and punctuation cleanup
- **collect_data.py** — Webcam sample collection utility
- **adapter_model.py** / **adapter_training.py** — Experimental live adaptation components
- **config.py** — Hyperparameters and derived dimensions
- **model.pth** — Trained single-model checkpoint
- **ensemble/** — K-fold ensemble checkpoints
- **Dataset/** — Raw video files organized by class
- **processed/** — Preprocessed landmark `.npy` files (20 frames × 506 dims)

## Developer utilities

Developer and debugging tools are collected in `DEVELOPER.md`. It lists quick verification commands, profiling helpers, data QC scripts, K-fold helpers, and checkpoint naming conventions. See: [DEVELOPER.md](DEVELOPER.md)

For a quick webcam-pipeline measurement, run:

```bash
python -u scripts/benchmark_webcam_pipeline.py --frames 120 --warmup 10
```

To rebalance `processed/` so every class ends at 850 samples, run:

```bash
python balance_processed_dataset.py --target 850
```

Use `--dry-run` first if you want to inspect the planned additions and removals without changing files.


## Quick Start

### 1) Preprocess and Train

```bash
python main.py --preprocess
python main.py --train
```

### 1b) Build a Controlled Augmented Video Dataset

Run this on your training split only, not validation/test:

```bash
python main.py --augment-videos --augment-input-dir Dataset --augment-output-dir augmented_dataset
```

Each class keeps the original videos and generates up to **8 separate augmentations** per source video while capping the total output per class.

**Augmentation types** (preserves hand visibility for MediaPipe):
- **aug1:** Center crop (baseline)
- **aug2:** Left-shifted crop (hand positioned left)
- **aug3:** Right-shifted crop (hand positioned right)
- **aug4:** Center crop + random effect (scale/zoom, rotation, contrast, color jitter, or noise)
- **aug5:** Left crop + random effect
- **aug6:** Right crop + random effect
- **aug7:** Center crop + stacked effects (two sequential transformations)
- **aug8:** Center crop + different random effect

**Example: Generate 8 variants per video (default)**
```bash
python main.py --augment-videos --augment-input-dir Dataset --augment-output-dir augmented_dataset
```
With 10 source videos per class → ~80 augmented + 10 originals = 90 total per class.

**Higher resolution (sharper quality)**
```bash
python main.py --augment-videos --augment-input-dir Dataset --augment-output-dir augmented_dataset --augment-width 320 --augment-height 320
```
Or even larger: `--augment-width 480 --augment-height 480`

**Generate even more per class (e.g., 200 total)**
Increase the cap per class:
```bash
python main.py --augment-videos --augment-input-dir Dataset --augment-output-dir augmented_dataset --augment-max-per-class 200
```

**Non-destructive append (keep existing augmentations)**
```bash
python main.py --augment-videos --augment-input-dir Dataset --augment-output-dir augmented_dataset --no-clear
```

**Combine options (high-res, more per-class, 8 variants, non-destructive)**
```bash
python main.py --augment-videos --augment-input-dir Dataset --augment-output-dir augmented_dataset --augment-width 320 --augment-height 320 --augment-max-per-class 150 --no-clear
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

The collector uses image-mode MediaPipe detection per frame, which keeps the live webcam capture path compatible with the sample extraction code.

### 6) Dynamic Quantization for CPU Inference

Quantize the main checkpoint:

```bash
python quantize_model.py --checkpoint model.pth --output model_quantized.pt
```

Quantize every ensemble checkpoint into a separate directory:

```bash
python quantize_model.py --ensemble-dir ensemble --output ensemble_quantized
```

Benchmark and optionally measure validation accuracy:

```bash
python evaluate_quantized_model.py --checkpoint model_quantized.pt --evaluate-accuracy
```

The live pipeline can load quantized bundles through the existing ensemble loader. Opt in with:

```bash
python main.py --webcam --quantized
python main.py --predict sample.mp4 --quantized
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
- **Label smoothing** — Prevents overconfident predictions
- **Continuous sentence building** — Sign predictions are accumulated and normalized into text

## Performance

- **K-Fold Ensemble:** **95.83% mean accuracy** on the earlier smaller benchmark set
- **Real-time webcam:** approximately 30 FPS on Intel Iris Xe GPU in the current setup
- **Current dataset:** 78 classes in `processed/`

Performance depends heavily on camera quality, lighting, signer distance, and whether the sign is held long enough for the 20-frame window.

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
- The current NLP layer is rule-based. A true context-aware translation model is not yet implemented.
- If person detection is unstable in a specific environment, use better lighting and a clean background for best results.
- If you change the number of classes or feature layout, retrain both the single model and ensemble checkpoints.

## Translation Roadmap

The current system is optimized for isolated-word recognition. A practical upgrade path to context-aware translation is:

1. Keep the landmark recognizer and sentence builder as the base.
2. Add a context buffer that stores recent committed words and confidence scores.
3. Rescore top-k predictions with a lightweight language model or rule-based prior.
4. Apply grammar correction only after sentence completion.
5. Move to gloss-to-text or CSLT only after collecting sentence-level data.

This keeps the system lightweight and deployable while adding incremental intelligence.

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

## Recent changes (2026-05-21)

These notes summarize recent development pushes so the repo README matches the latest code.

- Added a lightweight per-frame Spatial GNN and integrated it into the model pipeline. The GNN feature is concatenated before the model `input_proj` and controlled by config flags under `arch_improvements`.
- Updated training policy: removed GNN warmup. Training now uses direct full-model fine-tuning only (single-split and K-fold).
- K-fold training runs full fine-tuning directly with `--kfold`; no staged warmup flags remain in `train.py`.
- Current CLI flags for training control are: `--kfold`, `--epochs`, `--lr`.
- Added quick scripts for testing and benchmarking: `scripts/smoke_gnn_test.py` and `scripts/benchmark_gnn.py`.

Example usage:
```bash
# Single full-training run
python train.py --epochs 8 --lr 1e-4

# K-fold full fine-tuning (5 folds)
python train.py --kfold 5 --epochs 8 --lr 1e-4
```

If you'd like, I can expand these notes into a longer changelog section or add per-file developer notes.
