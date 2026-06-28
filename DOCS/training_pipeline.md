# Training Pipeline

## Overview

The ISL Sign-to-Text training pipeline is a 10-stage process from raw video recording to a deployable ONNX INT8 model.

## Quick Reference

```bash
# Full pipeline (steps 1–10)
python main.py --collect --cls hello --n 50    # Step 1: collect
python main.py --augment-videos                # Step 3: augment videos
python main.py --preprocess                    # Step 4: extract landmarks
python main.py --augment-landmarks             # Step 5: augment landmarks
python main.py --merge                         # Step 6: merge augmentation
python main.py --cleanup                       # Step 7: diversity cleanup
python main.py --kfold                         # Step 8: K-fold training
python scripts/export_onnx.py                  # Step 9: ONNX export
python scripts/quantize_onnx.py                # Step 10: INT8 quantize
```

---

## Step-by-Step

### Step 1 — Data Collection

```bash
python main.py --collect --cls <sign_name> --n 50
```

- Records 50 training samples via webcam
- 3-second countdown before each recording
- Each sample: 90 raw frames at 640×480 px
- Saved to `assets/Dataset/<sign_name>/`

### Step 2 — (Optional) Negative Samples

Collect background / non-sign sequences:

```bash
python main.py --collect --cls __reject__ --n 50
# Move to processed_negatives/ after preprocessing
```

### Step 3 — Video Augmentation

```bash
python main.py --augment-videos \
    --augment-input-dir assets/Dataset \
    --augment-output-dir assets/augmented_dataset \
    --augment-max-per-class 900
```

Generates up to 54 augmented video variants per source video:
- 17 visual effects (brightness, contrast, motion blur, JPEG artifacts, etc.)
- 3 crop positions (center, left 15%, right 85%)

### Step 4 — Landmark Extraction (Preprocessing)

```bash
python main.py --preprocess
# or from augmented:
python main.py --preprocess --preprocess-dir assets/augmented_dataset
```

For each video:
1. Uniformly samples 20 frames via `np.linspace`
2. Runs MediaPipe HandLandmarker + FaceLandmarker per frame
3. Constructs 253-dim base feature vector (126 raw + 126 face-relative + 1 proximity)
4. Appends 253-dim velocity delta → **506-dim per frame**
5. Saves `(20, 506)` NumPy array to `assets/processed/<class>/`

### Step 5 — Landmark Augmentation

```bash
python main.py --augment-landmarks \
    --augment-landmarks-n 14
```

Applies 14 of 20 available deterministic augmentations to each `.npy` file:
- 3D rotation (±15°), scaling (0.88–1.12×), translation
- Temporal: speed warp, time shift, frame dropout (1–3 frames)
- Occlusion: per-hand dropout, fog noise, coarse dropout
- Recomputes proximity scalar and velocity after coordinate changes

### Step 6 — Merge Augmentation

```bash
python main.py --merge --merge-n 3 --merge-mode crossfade_splice
```

Splices frame ranges from different recordings of the same class:
- Modes: `splice`, `crossfade_splice`, `blend`, `hand_swap`, `tempo_aligned_splice`
- Creates stylistically diverse synthetic samples from real data

### Step 7 — Diversity Cleanup

```bash
python main.py --cleanup \
    --cleanup-max-aug 600 \
    --cleanup-max-merge 200
```

Removes near-duplicates using L2-normalized cosine distance, then selects the most diverse subset via Farthest Point Sampling (FPS).

### Step 8 — K-Fold Training

```bash
python main.py --kfold
# With negatives:
python main.py --kfold --neg-root processed_negatives
```

**Per fold:**
1. Disjoint stratified split (70% train / 30% validation)
2. `ISLDataset` loads `.npy` files with on-the-fly augmentation
3. Training: AdamW, cosine LR scheduler, label smoothing 0.05, mixup α=0.3
4. Best checkpoint saved to `assets/ensemble/fold_N.pth`
5. Fold accuracy recorded in `assets/ensemble/kfold_manifest.json`

**Two-phase training (optional):**
- Phase 1: train on `processed/` only
- Phase 2: fine-tune adding `processed_del/` at weight 0.25

```bash
python main.py --kfold \
    --finetune-archived-epochs 15 \
    --finetune-archived-lr 3e-5
```

### Step 9 — ONNX Export

```bash
python scripts/export_onnx.py \
    --checkpoint models/model.pth \
    --output models/model_fp32.onnx
```

- Opset 18, dynamic batch size
- Traces model with a synthetic input
- Writes `model_fp32_metadata.json` with class list and config hash

### Step 10 — INT8 Quantization

```bash
python scripts/quantize_onnx.py \
    --input models/model_fp32.onnx \
    --output models/model_int8.onnx
```

- Dynamic INT8 quantization via `onnxruntime.quantization.quantize_dynamic`
- Result: ~1.05 MB (from ~4.2 MB FP32)
- Writes `*_quantization_metadata.json`

---

## Training Configuration

All hyperparameters are in `src/core/config.py` → `TrainingConfig`:

| Parameter | Default | Notes |
|---|---|---|
| `batch_size` | 8 | Small batches for limited per-class counts |
| `learning_rate` | 3e-4 | Reduced for stability |
| `weight_decay` | 5e-4 | L2 regularization |
| `grad_clip` | 1.0 | Gradient norm clipping |
| `num_epochs` | 50 | With early stopping |
| `patience` | 10 | Early stopping patience |
| `val_split` | 0.30 | Stratified 70/30 split |
| `label_smoothing` | 0.05 | Prevents overconfident predictions |
| `use_class_weights` | True | Inverse frequency, power=1.0 |
| `use_mixup` | True | α=0.3, probability=0.5 |
| `lr_scheduler` | cosine | ReduceLROnPlateau with cosine decay |
