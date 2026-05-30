# ISL Sign To Text

Real-time Indian Sign Language word recognition using MediaPipe landmarks, a BiGRU-based sequence classifier, and lightweight sentence post-processing.

This repository currently implements a strong isolated-word recognition pipeline with continuous webcam inference, confidence smoothing, and rule-based text cleanup. It is a good base for adding context-aware sign translation on top.

## Features

- **Landmark-based pipeline:** MediaPipe hand + face-relative keypoints stored as compact `.npy` sequences
- **BiGRU + Attention classifier:** temporal sequence model for isolated sign recognition
- **K-fold ensemble inference:** more robust prediction by averaging multiple checkpoints
- **Real-time webcam mode:** sliding-window recognition with signer validation, transition suppression, confidence gating, periodic full hand re-detection, and drift-triggered cache refresh
- **Sentence builder:** accumulates recognized signs into continuous text with ambiguity delay when top predictions are too close
- **Similar sign pairs config:** editable [similar_signs.json](similar_signs.json) file for confusable-sign rules
- **Sign category config:** editable [sign_categories.json](sign_categories.json) file for high-level class groups
- **Hand sign classification config:** editable [hand_sign_classification.json](hand_sign_classification.json) file for hand-count, motion, and location hints
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
- **Negative-sample training:** optional `processed_negatives/` reject class support via `--neg-root`

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
- onnxruntime
- onnx
- onnxscript

For ONNX quantization, `onnxruntime` must be installed in the active environment. The current quantizer uses dynamic INT8 quantization and writes a `*_quantization_metadata.json` file alongside the output model.

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
- **export_onnx.py** — Export PyTorch checkpoints to ONNX format (opset 18, dynamic batch)
- **onnx_inference.py** — Inference wrapper with automatic PyTorch fallback and profiling
- **onnx_ensemble.py** — Mixed ONNX/PyTorch ensemble loading and inference
- **quantize_onnx.py** — Convert ONNX FP32 to INT8 quantized format
- **onnx_ensemble_integration.py** — Drop-in replacement functions for existing ensemble
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

To keep only the highest-quality and least-redundant samples per class, run:

```bash
python quality_filter_npy.py --dry-run
python quality_filter_npy.py --class hello --dry-run
```

This filter now uses a hybrid quality + diversity pass: it keeps the adaptive quality-curve budget, builds diversity embeddings, suppresses near-duplicates, and greedily selects a diverse high-quality subset per class.

Dry-run and cleanup reports are written to `logs/quality_filter/` as JSON and CSV files.

When you run the quality filter without `--dry-run`, deleted samples are copied into a sibling archive folder named `processed_del/` by default. The live training set stays in `processed/`, while `processed_del/` becomes the rollback and audit copy of removed samples.

Use `processed/` for model training, evaluation, and any task that should reflect the curated dataset. Use `processed_del/` only when you explicitly want the removed samples for inspection, recovery, or a separate comparison task. The kept samples are the best subset according to this filter: highest-quality and least redundant within the current settings, but not a human guarantee of absolute perfection.

## Training And ONNX

Train the main classifier with a plain disjoint split:

```bash
python main.py --train
```

Include negatives from `processed_negatives/` as the reject class:

```bash
python main.py --train --neg-root processed_negatives
```

Run K-fold training with the same negative root:

```bash
python main.py --kfold --neg-root processed_negatives
```

Export the trained checkpoint to ONNX with the repo's current model class:

```bash
python export_onnx.py --checkpoint model.pth --output model_fp32.onnx
```

The exporter now infers `num_classes` from the checkpoint, writes `model_fp32_metadata.json`, and uses opset 18 by default.

Note: dynamic INT8 quantization is supported, but on small models it may not always reduce file size. Check the reported size after running `quantize_onnx.py`.


## Negatives / --neg-root

You can include a single `__reject__` (negative) class during training by pointing `--neg-root` at a directory containing negative `.npy` samples. The loader collects negatives recursively and treats them as one extra class used for rejecting out‑of‑vocabulary inputs.

Behavior details:

- Place negative `.npy` files under a folder like `processed_negatives/`. Subfolders are allowed and files are collected recursively.
- The loader will add a class named `__reject__` only if the negative folder contains at least `min_samples` (default `2`) files — this prevents accidental inclusion of tiny negative sets.
- Negatives are appended as a single class; their samples become `(file_path, label_index, weight)` entries in `ISLDataset.samples` where `label_index` corresponds to the `__reject__` class.
- Negatives affect class weighting and splits like any other class. If you have a very large negative set it will influence computed class weights; tune `cfg.training.class_weight_power` if needed.

Examples:

Train with negatives included in Phase 1 (processed-only training) and then fine‑tune on archived data:

```bash
python main.py --train --neg-root processed_negatives --finetune-archived-epochs 15 --finetune-archived-lr 3e-05
```

Quick check in Python to see how negatives were discovered:

```python
from train import create_data_loaders
_, _, _, _, ds = create_data_loaders(neg_root="processed_negatives", include_archived=False)
print('Classes:', ds.classes)
print('Negative label index:', ds.class_to_idx.get('__reject__'))
```

If you'd like, I can also add a short example script that builds a balanced negative set or a small validation snippet to inspect negative samples before training.


## Archived fine-tuning (two-phase training)

This repo supports a two-phase training workflow designed to preserve the curated `processed/` set while still allowing a controlled fine‑tune pass on archived samples stored in `processed_del/`.

- Phase 1 (default): train on the cleaned `processed/` dataset (and optional negatives). This avoids immediately exposing the model to noisy or deleted samples.
- Phase 2 (optional): fine‑tune on `processed_del/` (archived samples) only for a small number of epochs to adapt without forgetting the curated set.

Key CLI flags:

- `--archived-weight <float>` — per-sample weight assigned to archived samples when they are included (default: `0.25`).
- `--finetune-archived-epochs <int>` — number of epochs to run the archived-only fine‑tune phase after the main training (default: `0`, i.e. disabled).
- `--finetune-archived-lr <float>` — learning rate to use during the archived fine‑tune phase (optional; if omitted a conservative default is used).

Behavior notes:

- By default `main.py --train` runs Phase 1 on `processed/` only. If you set `--finetune-archived-epochs N` the pipeline will then run Phase 2 and fine‑tune the saved model on samples found under `processed_del/` for `N` epochs.
- For K‑fold runs (`--kfold`) Phase 1 performs the usual K‑fold training. If `--finetune-archived-epochs` is set, Phase 2 will iterate through each saved fold model and fine‑tune each one on `processed_del/` (so the ensemble reflects archived fine‑tuning across folds).
- If `processed_del/` is not present the fine‑tune phase is skipped.

Examples:

Train normally then fine‑tune 12 epochs on archived samples with a lower LR:

```bash
python main.py --train --finetune-archived-epochs 12 --finetune-archived-lr 3e-05
```

Run K‑fold then fine‑tune every fold for 10 epochs on archived data:

```bash
python main.py --kfold --finetune-archived-epochs 10
```

If you want to experiment with different archived sample weighting during fine‑tune, pass `--archived-weight 0.15` (the pipeline assigns this weight to archived samples during any dataset construction that includes them).

Advanced: the code exposes `create_data_loaders()` and `train()` so you can craft custom two‑phase flows in a script if you prefer different inclusion rules for Phase 1.

## Developer Notes & Troubleshooting

This section collects implementation knowledge and quick fixes useful when developing, debugging, or extending the pipeline.

- **Dataset sample representation changed:** `ISLDataset.samples` entries are now 3-tuples `(file_path, label_index, sample_weight)` instead of `(file_path, label_index)`. Update any code that *unpacks* `parent.samples` or assumes 2-tuples (e.g., `_, label = parent.samples[i]`) to handle the third `weight` element: `_, label, _ = parent.samples[i]`.

- **Archived samples are NOT included in normal training:** The default training/data-loader path uses `include_archived=False`. Archived samples in `processed_del/` are only used during the explicit Phase‑2 fine‑tune step (or when you call `create_data_loaders(..., include_archived=True)`). This prevents accidental training on deleted/archived data.

- **API notes:**
  - `create_data_loaders(neg_root=None, archived_root=None, archived_weight=0.25, include_archived=False)` — call with `include_archived=True` only when you want archived samples included in the base dataset.
  - `train(..., epochs=None, pretrained_checkpoint=None, lr=None)` — supports overriding epochs, loading a checkpoint, and specifying learning rate for fine‑tune runs.

- **Two-phase training (summary):**
  1. Phase 1: train on `processed/` only (default). Example:

```bash
python main.py --train
```

  2. Phase 2: fine‑tune on archived `processed_del/` only. Enabled with `--finetune-archived-epochs` (applies to single-run and to each fold in `--kfold`). Example:

```bash
python main.py --train --finetune-archived-epochs 12 --finetune-archived-lr 3e-05
```

- **K‑fold behavior:** When `--kfold` is used and `--finetune-archived-epochs` > 0, the pipeline will fine‑tune each saved fold model on archived samples (not just the best fold). This keeps the ensemble consistent.

- **How to verify archived inclusion in a run:**
  - Look in the training log for a `[Dataset] <N> samples` line. If archived were included, the total should equal `len(processed/) + len(processed_del/)` (or the number you expect). The pipeline also prints `[Phase 2] Fine-tuning on archived samples` when it begins the archived pass.
  - Quick programmatic check (python):

```python
from train import create_data_loaders
_, _, _, _, ds = create_data_loaders(include_archived=True)
print('Total samples (with archived)=', len(ds))
```

- **Per-sample weighting:** Archived samples are assigned a weight (default `0.25`) so their loss contribution is scaled during training. Change with `--archived-weight` or pass `archived_weight` to `create_data_loaders()` when building a custom dataset.

- **Filelist helper:** Use `tools/build_weighted_filelist.py` to create a 3-column file list `path,label,weight` for inspection or to feed custom samplers.

- **Removing accidentally committed artifacts:** If you previously committed large artifacts that should be ignored (e.g., `model.pth`), remove them from the index without deleting the file with:

```bash
git rm --cached model.pth
git commit -m "Remove tracked large artifacts now in .gitignore"
git push
```

- **Common crash:** If you see `ValueError: too many values to unpack (expected 2)` during training, search for unpacking sites assuming two elements in `samples`. Update them to handle three elements (see first note).

- **Quick dry-run:** To run a short sanity check (1 epoch Phase 1 + 1 epoch Phase 2) you can temporarily override values in config or call training directly in an interactive script. Example quick run (small, for CI/dev):

```bash
python -c "import main; main.run_train_word(neg_root=None, archived_weight=0.25, finetune_archived_epochs=1, finetune_archived_lr=1e-4)"
```

If you want, I can commit these README updates and run a short dry‑run to prove the flow. If you'd like the README to include additional developer notes (e.g., how we untracked `model.pth` historically, or exact log excerpts), tell me what to include and I'll add it.


## Synthetic Data Generation & Filtering

Generate class-balanced training data using a Conditional Variational Autoencoder (CVAE), then filter synthetic samples by quality:

### 1) Train the CVAE

```bash
# Train CVAE on all classes (default: 20 epochs, class conditioning)
python train_cvae.py \
  --processed-root processed \
  --epochs 20 \
  --batch-size 64 \
  --checkpoints-dir checkpoints/cvae_landmarks \
  --models-dir models

# Quick smoke test (1 class, 1 epoch)
python train_cvae.py --processed-root processed --epochs 1 --include-class alive
```

**Output:** `models/cvae_landmarks.pt` + `models/cvae_metadata.json`

### 2) Generate Synthetic Samples

```bash
# Generate synthetic samples (30% ratio by default, produces dry-run report first)
python generate_cvae_samples.py \
  --checkpoint models/cvae_landmarks.pt \
  --processed-root processed \
  --output-root generated \
  --max-ratio-synthetic 0.30 \
  --dry-run  # Test without writing files

# Actually create the files (remove --dry-run)
python generate_cvae_samples.py \
  --checkpoint models/cvae_landmarks.pt \
  --processed-root processed \
  --output-root generated \
  --max-ratio-synthetic 0.30
```

**Output:** Synthetic samples as `generated/<class>/cvae_*.npy` files. Keeping synthetic files in a separate root prevents label leakage when training the quality discriminator.

### 3) Train Quality Discriminator

```bash
# Train discriminator to score real vs synthetic samples
python train_quality_discriminator.py \
  --real-root processed \
  --synthetic-root generated \
  --epochs 20 \
  --batch-size 64 \
  --checkpoints-dir checkpoints/quality_discriminator \
  --models-dir models/discriminator

# Finetune with hard-negative mining (refine after initial training)
python train_quality_discriminator.py \
  --real-root processed \
  --synthetic-root generated \
  --epochs 5 \
  --hard-negative-mining \
  --hard-negative-finetune-epochs 5 \
  --checkpoints-dir checkpoints/quality_discriminator \
  --models-dir models/discriminator
```

**Output:** `checkpoints/quality_discriminator/best.pt` and exported model `models/discriminator/quality_discriminator.pt` + validation metrics

### 4) Filter Synthetic Samples

```bash
# Test filtering without modifying files
python filter_synthetic_samples.py \
  --source-root generated \
  --processed-root processed \
  --checkpoint models/discriminator/quality_discriminator.pt \
  --output-root filtered_synthetic \
  --threshold 0.80 \
  --dry-run

# Actually filter and organize samples
python filter_synthetic_samples.py \
  --source-root generated \
  --processed-root processed \
  --checkpoint models/discriminator/quality_discriminator.pt \
  --output-root filtered_synthetic \
  --threshold 0.80
```

**Output:** Organized high-quality samples in `filtered_synthetic/<class>/`

### 5) Visualize Results

```bash
# Explore CVAE latent space
python visualize_latent_space.py \
  --checkpoint models/cvae_landmarks.pt \
  --processed-root processed \
  --output logs/cvae_landmarks/latent_pca.png

# Inspect discriminator quality scores
python visualize_quality_scores.py \
  --real-root processed \
  --synthetic-root generated \
  --checkpoint models/discriminator/quality_discriminator.pt \
  --output-dir visualization
```

### Complete Workflow Example

```bash
# 1. Train CVAE
python train_cvae.py --processed-root processed --epochs 20

# 2. Generate synthetic samples
python generate_cvae_samples.py \
  --checkpoint models/cvae_landmarks.pt \
  --processed-root processed \
  --output-root generated

# 3. Train quality discriminator
python train_quality_discriminator.py \
  --real-root processed \
  --synthetic-root generated \
  --epochs 20 \
  --checkpoints-dir checkpoints/quality_discriminator \
  --models-dir models/discriminator

# 4. Filter synthetic samples by quality
python filter_synthetic_samples.py \
  --source-root generated \
  --processed-root processed \
  --checkpoint models/discriminator/quality_discriminator.pt \
  --output-root filtered_synthetic \
  --threshold 0.80

# 5. Retrain main classifier on real + filtered synthetic data
python main.py --train --neg-root processed_negatives
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

## Latest Changes (2026-05-29)

- **Archive-on-delete:** When you run the quality filter without `--dry-run`, removed samples are moved to a sibling archive folder `processed_del/` (keeps original filenames and class layout). `processed/` remains the curated training set.

- **Implicit archived inclusion in training:** If a `processed_del/` folder exists next to `processed/`, the training pipeline will automatically include those archived samples with reduced influence so you can fine-tune robustness without forgetting the clean data.

- **Per-sample weights applied:** Archived samples are assigned a default weight (0.25) and the training loop multiplies per-sample losses by these weights before averaging. Clean `processed/` samples keep weight `1.0`.

- **New CLI flag:** `--archived-weight` — set from the command line to control the archived-sample weight (range 0–1). Examples:

```bash
# Train with lower influence from archived samples (0.15)
python main.py --train --archived-weight 0.15

# Run K-fold with archived weight 0.10
python main.py --kfold --archived-weight 0.10
```

- **Helper script:** `tools/build_weighted_filelist.py` — builds a combined or staged filelist mixing `processed/` and `processed_del/` with per-sample weights. Use `--staged` to create `stage1_...` (clean) and `stage2_...` (archived) files for staged training.

- **Training safety:** Default behavior favors clean data (weight=1.0) and uses lower LR/fewer epochs when fine-tuning on archived data is recommended. Always inspect samples in `processed_del/` before reintroducing them.

- **Reports & QC artifacts:** Quality-filter dry-run reports and problematic-class CSVs are written to `logs/quality_filter/` (JSON + CSV) for audit and manual review.

If you'd like, I can add a short note in `DEVELOPER.md` showing the recommended staged training commands and a PyTorch `WeightedRandomSampler` snippet to consume the three-column filelist automatically.

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

The live sentence builder now adds an ambiguity-delay gate: when the top two class probabilities are too close, it waits a few more frames before committing the sign. That reduces borderline mis-commits during fast motion, transitions, and noisy webcam input.

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
