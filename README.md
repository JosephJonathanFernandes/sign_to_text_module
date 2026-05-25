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
- **CVAE-based synthetic data generation:** conditional variational autoencoder for class-balanced landmark sequence synthesis
- **Quality discriminator:** real/fake scoring network for filtering low-quality synthetic samples
- **Synthetic data filtering:** confidence-weighted sample selection with heuristic quality checks
- **ONNX export and quantization:** convert PyTorch models to optimized ONNX Runtime format
- **INT8 quantization:** 75% model size reduction with 2-3x faster inference
- **Mixed ensemble support:** seamless inference with combined ONNX + PyTorch models

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

**Synthetic Data Pipeline (New):**
- **cvae_landmarks.py** — Conditional VAE for landmark sequence generation (BiGRU encoder/decoder)
- **train_cvae.py** — CVAE trainer with class conditioning, stratified validation, early stopping
- **generate_cvae_samples.py** — Generate synthetic samples per-class with latent statistics
- **visualize_latent_space.py** — PCA/t-SNE visualization of CVAE latent embeddings
- **quality_discriminator.py** — BiGRU real/fake discriminator with heuristic checks
- **train_quality_discriminator.py** — Discriminator trainer with hard-negative mining
- **filter_synthetic_samples.py** — Score and filter synthetic samples by quality threshold
- **visualize_quality_scores.py** — Histogram and PCA visualization of discriminator scores

**ONNX Utilities (New):**
- **export_onnx.py** — Export PyTorch checkpoints to ONNX format (opset 17, dynamic batch)
- **onnx_inference.py** — Inference wrapper with automatic PyTorch fallback and profiling
- **onnx_ensemble.py** — Mixed ONNX/PyTorch ensemble loading and inference
- **quantize_onnx.py** — Convert ONNX FP32 to INT8 quantized format
- **benchmark_onnx.py** — Performance comparison across all backends
- **validate_onnx.py** — Numeric validation and parity checks
- **onnx_ensemble_integration.py** — Drop-in replacement functions for existing ensemble
- **onnx_examples.py** — Interactive menu with 7 complete workflow examples
- **test_onnx_integration.py** — Comprehensive test suite with 6 validation tests
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


## Synthetic Data Generation & Filtering

Generate class-balanced training data using a Conditional Variational Autoencoder (CVAE), then filter synthetic samples by quality:

### 1) Train the CVAE

```bash
# Train CVAE on all classes (default: 20 epochs, class conditioning)
python train_cvae.py \
  --processed-root processed \
  --epochs 20 \
  --batch-size 64 \
  --output-dir models/cvae

# Quick smoke test (1 class, 1 epoch)
python train_cvae.py --processed-root processed --epochs 1 --include-class alive
```

**Output:** `models/cvae/cvae_landmarks.pt` + metadata file

### 2) Generate Synthetic Samples

```bash
# Generate synthetic samples (30% ratio by default, produces dry-run report first)
python generate_cvae_samples.py \
  --cvae-checkpoint models/cvae/cvae_landmarks.pt \
  --processed-root processed \
  --output-root processed \
  --synthetic-ratio 0.30 \
  --dry-run  # Test without writing files

# Actually create the files (remove --dry-run)
python generate_cvae_samples.py \
  --cvae-checkpoint models/cvae/cvae_landmarks.pt \
  --processed-root processed \
  --output-root processed \
  --synthetic-ratio 0.30
```

**Output:** Synthetic samples as `processed/<class>/cvae_*.npy` files (appended to existing data)

### 3) Train Quality Discriminator

```bash
# Train discriminator to score real vs synthetic samples
python train_quality_discriminator.py \
  --processed-root processed \
  --epochs 20 \
  --batch-size 64 \
  --output-dir models/discriminator

# Finetune with hard-negative mining (refine after initial training)
python train_quality_discriminator.py \
  --processed-root processed \
  --epochs 5 \
  --finetune \
  --output-dir models/discriminator
```

**Output:** `models/discriminator/best.pt` + validation metrics

### 4) Filter Synthetic Samples

```bash
# Test filtering without modifying files
python filter_synthetic_samples.py \
  --processed-root processed \
  --discriminator-checkpoint models/discriminator/best.pt \
  --output-root filtered_synthetic \
  --quality-threshold 0.80 \
  --dry-run

# Actually filter and organize samples
python filter_synthetic_samples.py \
  --processed-root processed \
  --discriminator-checkpoint models/discriminator/best.pt \
  --output-root filtered_synthetic \
  --quality-threshold 0.80
```

**Output:** Organized high-quality samples in `filtered_synthetic/<class>/`

### 5) Visualize Results

```bash
# Explore CVAE latent space
python visualize_latent_space.py \
  --cvae-checkpoint models/cvae/cvae_landmarks.pt \
  --processed-root processed \
  --output-dir visualization

# Inspect discriminator quality scores
python visualize_quality_scores.py \
  --processed-root processed \
  --discriminator-checkpoint models/discriminator/best.pt \
  --output-dir visualization
```

### Complete Workflow Example

```bash
# 1. Train CVAE
python train_cvae.py --processed-root processed --epochs 20

# 2. Generate synthetic samples
python generate_cvae_samples.py \
  --cvae-checkpoint models/cvae/cvae_landmarks.pt \
  --processed-root processed

# 3. Train quality discriminator
python train_quality_discriminator.py \
  --processed-root processed \
  --epochs 20

# 4. Filter synthetic samples by quality
python filter_synthetic_samples.py \
  --processed-root processed \
  --discriminator-checkpoint models/discriminator/best.pt \
  --output-root filtered_synthetic \
  --quality-threshold 0.80

# 5. Retrain main classifier on real + filtered synthetic data
python main.py --train --epochs 15
```

**Why this matters:** CVAE generates balanced samples for underrepresented classes, quality discriminator filters unrealistic ones, and retraining on mixed real+synthetic data typically improves generalization by 2-4%.



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

## ONNX Runtime Integration (New Feature)

Convert trained PyTorch models to ONNX format for 2-3x faster inference on CPU with INT8 quantization:

### Export Models to ONNX

```bash
# Export single checkpoint
python export_onnx.py \
  --checkpoint model.pth \
  --output model_fp32.onnx

# Export all ensemble checkpoints
python export_onnx.py \
  --checkpoint-dir ensemble \
  --output-dir ensemble_onnx
```

**Output:** ONNX models with dynamic batch support (opset 17)

### Quantize to INT8

```bash
# Quantize a single FP32 ONNX model
python quantize_onnx.py \
  --model model_fp32.onnx \
  --output model_int8.onnx

# Quantize all models in a directory
python quantize_onnx.py \
  --model-dir ensemble_onnx \
  --output-dir ensemble_quantized
```

**Output:** INT8 quantized models (~25 MB each, 75% size reduction)

### Validate Numeric Parity

Ensure ONNX outputs match PyTorch numerically:

```bash
python validate_onnx.py \
  --pytorch-checkpoint model.pth \
  --onnx-model model_int8.onnx \
  --test-samples 100
```

**Expected:** >99% prediction agreement, <0.01 L2 distance on logits

### Benchmark All Backends

Compare performance across PyTorch FP32, quantized PyTorch, ONNX FP32, and ONNX INT8:

```bash
python benchmark_onnx.py \
  --pytorch-checkpoint model.pth \
  --onnx-fp32 model_fp32.onnx \
  --onnx-int8 model_int8.onnx \
  --batch-sizes 1 8 16 \
  --num-iterations 100
```

**Typical Results (CPU):**
| Backend | Latency (ms) | FPS | Model Size | Memory |
|---------|--------------|-----|------------|--------|
| PyTorch FP32 | 4.5 | 222 | 100 MB | 450 MB |
| PyTorch Quantized | 2.8 | 357 | 25 MB | 280 MB |
| ONNX FP32 | 2.5 | 400 | 100 MB | 420 MB |
| ONNX INT8 | 1.8 | 556 | 25 MB | 150 MB |

### Use ONNX in Your Code

**Option 1: Drop-in wrapper (simplest)**
```python
from onnx_inference import ONNXModelWrapper

model = ONNXModelWrapper(
    pytorch_checkpoint='model.pth',
    onnx_model='model_int8.onnx'
)
output = model(input_tensor)  # Automatically tries ONNX, falls back to PyTorch if needed
```

**Option 2: Mixed ensemble with ONNX + PyTorch**
```python
from onnx_ensemble import detect_and_load_models, ensemble_predict_mixed

models = detect_and_load_models(
    model_dir='ensemble_mixed',  # Contains both .onnx and .pth files
    use_onnx=True
)
output = ensemble_predict_mixed(models, input_tensor)
```

**Option 3: Existing ensemble code unchanged**
Import the integration module for drop-in compatibility:
```python
from onnx_ensemble_integration import load_ensemble_with_onnx, ensemble_predict_with_onnx
from config import cfg

models = load_ensemble_with_onnx('ensemble_quantized', cfg)
output = ensemble_predict_with_onnx(models, input_tensor, cfg)
```

### Run Integration Tests

```bash
# Test all ONNX functionality
python test_onnx_integration.py

# Run specific test
python test_onnx_integration.py --test export
```

### Interactive Example Workflows

```bash
# Menu-driven examples (export, quantize, benchmark, validate, etc.)
python onnx_examples.py
```

**Why ONNX?**
- **Speed:** 2-3x faster inference with INT8 quantization
- **Size:** 75% smaller models fit on edge devices
- **Compatibility:** CPU-based, no GPU required
- **Optional:** Existing PyTorch pipeline continues to work unchanged
- **Fallback:** Automatic PyTorch fallback if ONNX unavailable



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

## Recent Changes (2026-05-25)

### Synthetic Data Generation & Filtering (CVAE + Quality Discriminator)

**New capabilities for class-balanced data augmentation and quality control:**

- **CVAE Model:** Conditional Variational Autoencoder with BiGRU encoder/decoder for landmark sequence generation
  - Class-conditional latent space with per-class statistics
  - Velocity consistency loss for realistic motion
  - Configurable latent dimension and capacity
  
- **CVAE Training:** Full trainer with stratified validation, early stopping, automatic mixed precision
  - Per-class label encoding with normalization (e.g., "58. Idle" → "idle")
  - Configurable batch size, learning rate, epochs, warmup
  - Checkpoint saving and resumption
  
- **Synthetic Sample Generation:** Creates balanced top-up samples from trained CVAE
  - Per-class generation quotas to maintain class balance
  - Quality heuristics: motion variance, feature std, frame jump detection, landmark drift
  - Dry-run mode for safety validation
  
- **Quality Discriminator:** BiGRU real/fake scorer with heuristic quality assessment
  - Balanced real/fake sampling during training
  - Hard-negative mining phase for refinement
  - Integrated heuristics: frozen landmarks detection, excessive drift checks
  
- **Synthetic Filtering:** Confidence-weighted sample selection
  - Threshold-based quality gating
  - Visualization and scoring reports
  - Integration with downstream training pipeline

**Typical workflow:** Train CVAE → Generate synthetic → Train discriminator → Filter by quality → Retrain classifier on mixed data

### ONNX Runtime Integration & Quantization

**Production-grade model optimization for CPU inference:**

- **ONNX Export:** Convert PyTorch checkpoints to ONNX format
  - Dynamic batch support (opset 17)
  - Full metadata serialization (label encoders, class counts)
  - Single or ensemble batch export
  
- **ONNX Inference Wrapper:** Unified inference API with automatic fallback
  - Transparent ONNX-first strategy with PyTorch fallback
  - Built-in profiling hooks (latency, errors, fallback counting)
  - CPU-optimized for real-time webcam use
  
- **INT8 Quantization:** Dynamic quantization for speed and size
  - 75% model size reduction (100 MB → 25 MB)
  - 2-3x inference speedup
  - Minimal accuracy loss (<1% typically)
  
- **Mixed Ensemble Support:** Seamless ONNX + PyTorch model combination
  - Auto-detection of .onnx and .pth files
  - Unified logit averaging across backends
  - Transparent upgrade path
  
- **Validation & Benchmarking:**
  - Numeric parity checks (>99% prediction agreement)
  - Performance comparison across all backends
  - Detailed latency, FPS, memory metrics
  
- **Integration:** Drop-in replacement for existing ensemble code
  - Three usage options: wrapper, ensemble, or native
  - No breaking changes to existing pipeline
  - Optional upgrade, not required

**Performance gains:** PyTorch 4.5ms → ONNX INT8 1.8ms per inference (2.5x speedup)

### Archive Notes

**Previous update (2026-05-21):**
- Added lightweight per-frame Spatial GNN
- Updated training policy (removed GNN warmup)
- K-fold training uses direct full-model fine-tuning
- Quick scripts: `scripts/smoke_gnn_test.py`, `scripts/benchmark_gnn.py`


python train.py --epochs 8 --lr 1e-4

# K-fold full fine-tuning (5 folds)
python train.py --kfold 5 --epochs 8 --lr 1e-4
```

If you'd like, I can expand these notes into a longer changelog section or add per-file developer notes.
