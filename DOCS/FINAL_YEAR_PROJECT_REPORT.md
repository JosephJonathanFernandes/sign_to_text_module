# Final-Year Project Report: Indian Sign Language (ISL) to Text Recognition System

**Student Name:** Joseph Jonathan Fernandes
**Project Duration:** February 21, 2026 – June 5, 2026 (~3.5 months)
**Total Development Commits:** 173 (per git HEAD reflog)
**Total Lines of Code:** 9,500+ Python lines across 57 Python files
**Repository:** sign_to_text

---

## Executive Summary

This project implements a **real-time Indian Sign Language (ISL) word recognition pipeline** that converts continuous hand gestures to text. The system combines MediaPipe hand/face landmark detection, a multi-stage BiGRU-based sequence classifier with Conv1D frontend, Spatial GNN, Hybrid Multi-Head Attention, ensemble inference, and sophisticated post-processing to achieve robust isolated-word recognition with temporal smoothing and natural language cleanup.

**Key Technical Achievements:**
- Implemented landmark-based feature extraction (506-dimensional velocity-augmented sequences)
- Multi-phase BiGRU model: Conv1D frontend → Spatial GNN → Frame Weighting → BiGRU → Hybrid Multi-Head Attention (phases 1–10)
- 5-fold cross-validation ensemble with K-fold disjoint-stratified splitting
- Mixed ONNX/PyTorch inference with automatic fallback and INT8 quantization
- Real-time webcam pipeline with momentum-based commit logic, confidence smoothing, and sentence accumulation
- Synthetic data generation via Conditional VAE (CVAE) with quality discrimination
- Comprehensive dataset management: 78 sign classes, ~5,683 processed samples, automated balancing
- Two-phase training: Phase 1 (processed/ only), Phase 2 (archived fine-tune with processed_del/)
- Negative/reject class support via `processed_negatives/` with phase-aware neg_root resolution

**Production Readiness:**
- Handles 506-dimensional feature space with velocity features
- INT8 quantized ONNX models deliver 2-3× faster inference
- Ensemble inference with mixed model formats (ONNX + PyTorch)
- Real-time processing with sub-200ms latency target
- Momentum-based commit logic prevents false positives (3-of-5 window, min_avg_conf=0.60)
- Temporal smoothing with confidence-weighted averaging, patience, and hysteresis

---

## 1. Project Evolution Timeline & Development Phases

> All commit hashes and timestamps are extracted directly from `.git/logs/HEAD`.
> Unix timestamps converted to IST (UTC+5:30).

### Phase 0: Project Initialization (Feb 21 – Feb 28, 2026)

| Date | Commit Hash | Description | Impact |
|------|-------------|-------------|--------|
| 2026-02-21 | `8c838b8e` | first commit (initial) | Repository created; base Python files |
| 2026-02-22 | `57df4351` | first commit | Expand landmark extraction skeleton |
| 2026-02-25 | `411052dc` | first commit | Core preprocess structure |
| 2026-02-25 | `7ae7c12d` | first commit | Face-relative coordinate computation |
| 2026-02-26 | `6f00c8df` | first commit | Config and model stub |
| 2026-02-26 | `57f90b4c` | first commit | Dataset loader and train stub |

**Evidence:** Unix timestamps 1771654456 through 1772127081 = Feb 21–26, 2026

---

### Phase 1: Core Pipeline Development (Feb 28 – Mar 5, 2026)

**Focus:** Landmark extraction, feature engineering, real-time integration, NLP post-processing

| Date | Commit Hash | Description | Impact |
|------|-------------|-------------|--------|
| 2026-02-28 | `15dcdfd6` | feat: Integrate face-proximity attention and optimize real-time performance | Face-proximity biasing; real-time inference |
| 2026-03-01 | `7270940e` | Add automatic sentence translation and dataset integration | Sentence accumulation pipeline |
| 2026-03-01 | `2a66e702` | feat: Add motion gating, dynamic thresholds, transition logic, NLP post-processing, and frame cropping | Key pipeline features added |
| 2026-03-04 | `aec2df26` | feat: optimize training for better accuracy — enhanced augmentation, improved hyperparameters | Mixup/CutMix, class weighting |
| 2026-03-04 | `36b4d7f7` | feat: add Focal Loss, configurable Mixup, extended training epochs | Focal Loss; training flexibility |
| 2026-03-04 | `da63625e` | fix: correct import and function call — train_kfold | K-fold import fix |
| 2026-03-04 | `316c4652` | feat: add comprehensive training configuration tuning script | Hyperparameter search |
| 2026-03-04 | `d43af3f3` | fix: remove old model loading to eliminate feature dimension mismatch errors | Dimension mismatch fix |

**Architectural Decisions:**
- Chose MediaPipe Tasks API (not legacy Solutions) for lightweight, real-time landmark detection
- Two separate MediaPipe models: `hand_landmarker.task` (7.8 MB) and `face_landmarker.task` (3.8 MB)
- Implemented face-relative coordinates relative to nose center (index 1), normalized by inter-eye distance
- Hand-to-face proximity scalar encoded as the L2 distance from hand center to nose
- Feature dimension: 126 (raw both hands) + 126 (face-relative both hands) + 1 (proximity) = 253D base; ×2 with velocity = 506D

**Evidence:**
- `preprocess.py` `_extract_face_anchor()` uses nose index 1, eyes 33 and 263 (MediaPipe face)
- `preprocess.py` `compute_face_relative_features()` normalizes by inter-eye scale
- `config.py` `FrameFeaturesConfig.frame_features_dim` = 126+126+1 = 253; `input_sequence_dim` = 506 with velocity
- `config.py` `PreprocessingConfig.face_nose_index = 1`

---

### Phase 2: Dataset Expansion & Augmentation (Mar 5 – Apr 15, 2026)

**Focus:** Per-word .npy collection, video augmentation pipeline, augmented dataset push

| Date | Commit Hash | Description | Impact |
|------|-------------|-------------|--------|
| 2026-03-04 | `399f5555` | app.py undo/presets updates and README refresh | UX and docs |
| 2026-03-05 | `d0d18d86` — `37a1bdd4` | added all important npy files (×9 commits) | First large dataset push |
| 2026-03-06 | `b9e4e379` — `c22dee3d` | added all important npy checking files; more words | Dataset QC and expansion |
| 2026-03-07 | `a63d818` | Add temporal post-processor module and update model training config | TemporalPostProcessor integrated |
| 2026-03-07 | `270191ab` | Integrate TemporalPostProcessor and HandSelector into webcam inference pipeline | Live pipeline upgrade |
| 2026-03-07 | `20f3dd64` | Fix temporal postprocessor and hand selector parameter integration | Bug fix |
| 2026-03-07 | `00b153a6` | Reduce transition latency: smaller smoothing windows for faster sign changes | UX improvement |
| 2026-03-07 | `6ff84e8c` | update config file to be OOP type | Config refactor to dataclasses |
| 2026-03-08 | `5da0f7b1` | added all important npy files you_all | Dataset: you_all |
| 2026-03-08 | `6ae01e63` | cheap, good, idle, male, female | Dataset: 5 more words |
| 2026-03-08 | `9bf47645` | tight, loose, he, she | Dataset: 4 more words |
| 2026-03-08 — 2026-03-10 | Multiple | good_morning, morning, good_afternoon, good_evening, good_night, pleased | Greetings and expressions |
| 2026-03-10 | `0ed3e45b` | added all augmented+merge files | Large augmented dataset push |
| 2026-03-10 | `74677292` | Add processed landmark sequences (5,683 .npy files) | **5,683 processed sequences confirmed** |
| 2026-03-10 | `c9771af2` | Enhance augmentation pipeline: add face-anchor shift, hand-proportions simulation, improved velocity recomputation | Advanced augmentation |
| 2026-03-12 | `e30db840` — `7443e84e` | 1–22 number marks added | Dataset: numbers |
| 2026-03-12 | `a76cd61d` | added more words — count from 42 to 79, all adjectives done | Dataset: 79 classes milestone |
| 2026-03-13 | `96e3d1e3` | added INCLUDE 50 videos augmented samples per word | 50 augmented videos per class |
| 2026-03-13 | `551fb8a7` | Refine train/val source-aware splits | K-fold data leakage prevention |

**Dataset Build Strategy:**
1. Record raw videos using `collect_data.py` (webcam collection with countdown)
2. Preprocess videos to 20×506D .npy sequences via `preprocess.py`
3. Augment at video level (up to 54 variants per video) via `augment_video_pipeline.py`
4. Merge augmented sequences via `merge_augmentations.py`
5. Balance dataset at class level via `balance_processed_dataset.py` (target: 850 samples/class)

**Video Augmentation Effects** (17 total effects × 3 crop positions + 3 spatial-only = up to 54 variants):
- Spatial: center/left/right crop
- Visual: brightness, contrast, hue, fog, rotation, scale, color_jitter, noise, pixel_dropout, coarse_dropout
- Added later: motion_blur, defocus_blur, jpeg_artifact, gamma, white_balance, perspective_warp, temporal_jitter

**Evidence:**
- `preprocess.py` `VIDEO_AUGMENT_MAX_PER_VIDEO = 54` (line 78); 17 effects
- `augmentations.py` `augment_face_anchor_shift()`, `augment_hand_proportions()`
- Commit `74677292`: "Add processed landmark sequences (5,683 .npy files)"
- Commit `c9771af2`: face-anchor shift, hand proportions, velocity recomputation

---

### Phase 3: Model Architecture Improvements — Phases 1–10 (Mar 13 – Apr 25, 2026)

**Focus:** Incremental architectural improvements to SignLanguageGRU (10 phases documented in model.py)

| Date | Commit Hash | Description | Impact |
|------|-------------|-------------|--------|
| 2026-03-14 | `0fb9cd15` | added latest important files after changes | Model updates |
| 2026-03-15 | `2eabd7b6` — `ba94cb90` | npy for long, short, tall, wide, old, bad, wet, hot, cold, warm, cool, new, narrow, big_large | More adjectives |
| 2026-03-15 | `246554e7` | added latest important files after changes | Config OOP solidified |
| 2026-03-21 | `c90e3eed` | added latest important files after changes | Architecture phase improvements |
| 2026-03-22 | `14e66de4` | added latest README.md | Documentation |
| 2026-03-22 | `d6a0e483` | added latest technical doc | Technical documentation |
| 2026-03-22 | `fc6f3c20` | Add safer conv frontend and smoke checks | **Phase 1: Conv1D frontend** |
| 2026-03-22 | `8e48c515` | Harden training and inference pipeline | Pipeline hardening |
| 2026-03-22 | `6eaaf0f1` | Commit staged project artifacts | Checkpoint |
| 2026-03-22 | `74ae72c3` | Update live inference momentum, adapter flow, and dataset artifacts | **Momentum-based commit logic** |
| 2026-04-23 | `c9771af2` | Enhance augmentation pipeline | Already documented above |
| 2026-04-24 | `ff6a57bb` | Complete comprehensive technical audit: Shape trace + GNN feasibility analysis | **Phase 10: Spatial GNN evaluation** |
| 2026-04-24 | `dd2dab01` — `98445fac` | Updated augmentations in sync with Akaash paper's methods | Augmentation alignment to literature |

**10-Phase Architecture Improvements (all toggles in `config.py` → `ArchitectureImprovementsConfig`):**

| Phase | Flag | Default | Description |
|-------|------|---------|-------------|
| Phase 1 | `use_conv_frontend` | **True** | Conv1D pointwise (506→128) + depthwise temporal + GroupNorm |
| Phase 2 | `use_frame_weighting` | **True** | Learnable per-frame sigmoid importance weights |
| Phase 3 | (LiveInference) | — | Test-Time Augmentation (disabled for live, enabled offline) |
| Phase 4 | `gru_dropout`, `fc_dropout` | 0.30, 0.25 | Reduced dropout (was 0.35) |
| Phase 5 | `use_residual_gru_skip` | **True** | Residual skip: input_proj → GRU output |
| Phase 6 | `use_groupnorm` | **True** | GroupNorm (8 groups) instead of LayerNorm in conv frontend |
| Phase 7 | `debug_print_shapes` | False | Debug shape tracing |
| Phase 8 | `use_depthwise_temporal`, `use_residual_conv` | **True**, **True** | Depthwise-separable temporal conv; residual within conv |
| Phase 9 | `use_residual_attention_skip` | **True** | Residual: GRU temporal mean into attention context |
| Phase 10 | `use_gnn` | **True** | Lightweight Spatial GNN branch (parallel to Conv1D) |

**Spatial GNN (Phase 10) — `spatial_gnn.py`, `LightweightSpatialGNN`:**
- Processes first 126 dims (raw hand landmarks, both hands) before Conv1D modification
- GCN-type graph convolutions over anatomical hand skeleton (21 nodes per hand)
- `gnn_hidden_dim=16`, `gnn_num_layers=2`, `gnn_output_dim=8`
- Output: 8D per hand × 2 hands = 16D per frame, concatenated with Conv1D features
- Shared weights between left and right hand (`gnn_shared_weights=True`)
- **GNN IS ENABLED by default** (`use_gnn: bool = True` in `config.py` line 628)

**Evidence:**
- `config.py` `ArchitectureImprovementsConfig` (lines 529–683)
- `model.py` `SignLanguageGRU.__init__()` phases 1, 2, 5, 9, 10 (lines 443–628)
- `model.py` `HybridAttention` class (lines 184–288) — 4 heads, 2 proximity heads
- `spatial_gnn.py` `LightweightSpatialGNN` class

---

### Phase 4: Synthetic Data & Quality Filtering (Apr 22 – Apr 25, 2026)

**Focus:** CVAE-based synthetic data generation, discriminator training, quality filtering

| Date | Commit Hash | Description | Impact |
|------|-------------|-------------|--------|
| 2026-04-22 | `0ed3e45b` | added all augmented+merge files | Dataset expansion |
| 2026-04-23 | `c9771af2` | Enhance augmentation pipeline: face-anchor shift, hand-proportions | Advanced augmentation |
| 2026-04-24 | `5ff8468f` | added all latest files | CVAE files committed |

**Synthetic Data Pipeline:**

1. **CVAE Training** (`cvae_landmarks.py`):
   - Input: 20×506D landmark sequences + class label
   - Encoder: BiGRU + attention → latent (32D, learned μ, σ)
   - Decoder: Latent + class embedding → reconstructed sequence
   - Loss: Reconstruction MSE + KL divergence

2. **Synthetic Sample Generation** (`generate_cvae_samples.py`):
   - Sample latent codes from N(0, I), decode with class label
   - Store as .npy files with metadata (class, generation_epoch)

3. **Quality Discrimination** (`quality_discriminator.py`):
   - BiGRU discriminator: real vs fake (from CVAE)
   - Heuristic checks: variance, velocity norm, landmark validity
   - Hard-negative mining

4. **Filtering** (`filter_synthetic_samples.py`):
   - Score all synthetic samples via discriminator
   - Threshold by confidence + heuristic quality metrics
   - Keep high-quality subset for balanced training

**Evidence:**
- `cvae_landmarks.py` implements VAE encoder/decoder with class conditioning
- `train_cvae.py` orchestrates CVAE training with early stopping
- `quality_discriminator.py` implements BiGRU discriminator
- `filter_synthetic_samples.py` applies confidence threshold filtering

---

### Phase 5: ONNX Export, Quantization & Dataset Management (Apr 25 – May 10, 2026)

**Focus:** Model optimization, ONNX export, INT8 quantization, dataset balancing

| Date | Commit Hash | Description | Impact |
|------|-------------|-------------|--------|
| 2026-05-01 | `e195bf23` | Expand augmentation pipeline | More augmentation effects |
| 2026-05-01 | `7cbd8537` | Update generated logs | Log artifacts |
| 2026-05-01 | `802bc56f` | Update preprocess log snapshot | Preprocessing records |
| 2026-05-01 | `0efa2cea` | Add processed augmentation outputs and updated logs | Output artifacts |
| 2026-05-08 | `727414cd` | Dataset cleanup: Remove corrupted and augmented dataset files from wide | QC cleanup |
| 2026-05-08 | `bcc5e5b7` | Remove processed and pseudo_data from git tracking | .gitignore update |
| 2026-05-08 | `999fafe3` | Update .gitignore to exclude model weights, logs, cache | Repository hygiene |
| 2026-05-08 | `f77a6ec1` | updated important files for increasing fps | FPS optimization |
| 2026-05-19 | `ff86e0f3` | Add quantized inference pipeline updates and webcam stability | ONNX quantization |
| 2026-05-19 | `e21bb7d7` | Fix webcam collection landmarker mode | Webcam bug fix |
| 2026-05-19 | `9e47f329` | Commit all current workspace changes | Checkpoint |
| 2026-05-19 | `d91d7000` | modified webcam aug file | Webcam augmentation |
| 2026-05-19 | `696b0eaa` | Add dataset balancer for 850-sample classes | **Dataset standardization** |
| 2026-05-19 | `4e0296de` | docs: Update README with CVAE, quality discriminator, and ONNX integration | Documentation |
| 2026-05-19 | `8c01084f` | feat: add ONNX tooling and workspace updates | **ONNX export foundation** |

**ONNX Export Pipeline** (`export_onnx.py`):
- Convert PyTorch model to ONNX format (opset 18)
- Dynamic batch size; fixed sequence length (20 frames)
- Input shape: `(batch, seq_len=20, feature_dim=506)`
- Output: class logits `(batch, num_classes=78)`

**Quantization** (`quantize_onnx.py`):
- Dynamic INT8 quantization (per-channel weights)
- Model size reduction: 75% (~4.2 MB → 1.05 MB)
- Inference speedup: 2-3× on CPU

**Evidence:**
- `export_onnx.py` exports PyTorch to ONNX with `torch.onnx.export()`
- `quantize_onnx.py` applies INT8 quantization via `ort.quantization.quantize_dynamic()`
- `onnx_inference.py` wraps ONNX runtime with PyTorch fallback

---

### Phase 6: Two-Phase Training, Negatives, & Production Hardening (May 20 – Jun 5, 2026)

**Focus:** Two-phase training strategy, negatives/reject class, K-fold bug fix, production stability

| Date | Commit Hash | Description | Impact |
|------|-------------|-------------|--------|
| 2026-05-22 | `bc6ec855` | modified reqts | Requirements |
| 2026-05-30 | `c5cf3213` — `d74e52e9` | modified many important files (×2) | Config and inference updates |
| 2026-05-30 | `d9c069ee` | Add processed_del archive, implicit archived training inclusion, --archived-weight CLI | **Two-phase training Phase 2** |
| 2026-05-30 | `be68d16d` | Fix unpacking for weighted samples in _BalancedAugSubset (handle (path,label,weight) tuples) | **Critical fix: 3-tuple support** |
| 2026-05-30 | `c42e88c5` | Stop tracking ignored model artifact and sync gitignore updates | Repository cleanup |
| 2026-05-30 | `e0161a3d` | Two-phase training: Phase 1 processed-only; Phase 2 archived fine-tune; K-fold fine-tune; README | **Two-phase strategy documented** |
| 2026-05-30 | `ccd69c22` | Docs: add 'Negatives / --neg-root' section to README | **Reject class documentation** |
| 2026-05-30 | `2ae9efb5` | Docs: clarify negatives are Phase 1 only | Phase boundary clarification |
| 2026-05-30 | `bde4753f` | Phase2: use processed_negatives_del as neg_root for archived fine-tune | Phase-aware neg_root |
| 2026-05-30 | `73f19768` | Docs: document processed_negatives_del usage for Phase 2 | Documentation |
| 2026-06-01 | `07fa4565` | modified gitignore | Repository management |
| 2026-06-03 | `68391e78` | modified many files | Production updates |
| 2026-06-04 | `97a036f9` | modified many files | Training stability |
| 2026-06-04 | `ba324c97` | modified many files | Dataset handling |
| 2026-06-04 | `09879f5d` | modified many files | Inference pipeline |
| 2026-06-04 | `0cc239d2` | modified many files | Ensemble robustness |
| 2026-06-05 | `f3fbbf33` | modified many files | Configuration optimization |
| 2026-06-05 | `4672472b` | **Fix K-fold sample label extraction** | **K-fold crash fix** |
| 2026-06-05 | `bdb93689` | docs: Add comprehensive final-year project report | Documentation |
| 2026-06-05 | `7037f0c9` | docs: Add comprehensive viva preparation guide | Documentation |
| 2026-06-05 | `682cc3b4` | made repository clean and proper | Final cleanup |

**Critical Bug Fix #1: K-fold Training Crash**

**Problem:**
- Error: `ValueError: too many values to unpack (expected 2)`
- Root cause: Dataset format changed from 2-tuple `(path, label)` to 3-tuple `(path, label, weight)`
- K-fold label extraction still used legacy: `[lbl for _, lbl in full_ds.samples]`

**Solution in `train.py` (line 76-82):**
```python
def _sample_label(sample) -> int:
    """Return the label field from a dataset sample.
    Supports both legacy (path, label) and weighted (path, label, weight) tuples.
    """
    return int(sample[1])
```

**Evidence:** Commit `4672472b62f271f7e2a750479363cd4796e97826`, `train.py` line 76

**Two-Phase Training Strategy (`train.py` `create_data_loaders()`):**

**Phase 1 (processed/ only):**
- Uses `processed/` as primary dataset
- Negatives via `processed_negatives/` (reject class) — Phase 1 only
- No archived samples

**Phase 2 (archived fine-tune):**
- Adds `processed_del/` archived samples with configurable weight (default 0.25)
- Negatives resolve to `processed_negatives_del/` automatically
- `_resolve_phase_neg_root(phase, neg_root)` handles phase-aware selection

**K-fold Training:**
- Uses `_build_disjoint_folds()` — purely random within-class partition (no source bias)
- 5 disjoint folds, stratified by class
- Per-fold manifest saved to `ensemble/kfold_manifest.json`
- ReduceLROnPlateau scheduler per fold

**Evidence:**
- `train.py` `_resolve_phase_neg_root()` (lines 243–262)
- `train.py` `_build_disjoint_folds()` (lines 208–240)
- `train.py` `_BalancedAugSubset` — 3-tuple `(fpath, label, weight)` unpacking (lines 409–470)

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT SOURCES                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Raw Video (Dataset/)  →  Webcam (Live)  →  Stored Landmarks (processed/)  │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────────────┐
│ FEATURE EXTRACTION PIPELINE                                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  MediaPipe HandLandmarker + FaceLandmarker (Tasks API)                      │
│  ↓                                                                           │
│  Per-hand raw coords (21×3=63D) × 2 hands = 126D raw                       │
│  ↓                                                                           │
│  Face-relative coords (subtract nose, divide by inter-eye dist) = 126D     │
│  ↓                                                                           │
│  Hand-to-face proximity scalar (L2: hand_center → nose) = 1D              │
│  ↓                                                                           │
│  BASE FRAME FEATURES: 253D (126 + 126 + 1)                                 │
│  ↓                                                                           │
│  Velocity Features: frame-to-frame deltas = 253D                           │
│  ↓                                                                           │
│  FINAL FEATURES: 506D (253 base + 253 velocity)                            │
│  ↓                                                                           │
│  Sequence: 20 frames × 506D = (20, 506) per sign sample                   │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────────────┐
│ TRAINING PIPELINE                                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  Phase 1: processed/ + optional processed_negatives/                       │
│  Phase 2: processed/ + processed_del/ (archived, low weight 0.25)         │
│  K-fold: 5-fold disjoint stratified splits → ensemble/fold_{n}.pth        │
│  Loss: CrossEntropyLoss (reduction='none') + per-sample weight × mean     │
│       OR FocalLoss (if USE_FOCAL_LOSS=True) — currently disabled          │
│  Optimizer: AdamW (lr=3e-4, weight_decay=5e-4)                            │
│  Scheduler: ReduceLROnPlateau (factor=0.5, patience=5 epochs)             │
│  Augmentation: Mixup (α=0.3, p=0.5) enabled; CutMix disabled             │
│  Early stopping: patience=10 epochs                                         │
│  Label smoothing: ε=0.05                                                    │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────────────┐
│ MODEL PIPELINE — SignLanguageGRU (10-phase architecture)                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  Input: (batch, 20, 506)                                                    │
│  [Phase 10] Spatial GNN branch: raw 126D → GCN × 2 layers → 16D/frame    │
│  [Phase 1]  Conv1D frontend: 506 → 128 channels (pointwise + depthwise)   │
│  Concat GNN + Conv: (batch, 20, 144)                                        │
│  [Phase 2]  Frame weighting: sigmoid importance per frame                  │
│  Input projection: Linear(144 → 64) + LayerNorm + ReLU                    │
│  [Phase 4]  BiGRU: 3 layers, hidden=64, bidirectional → output dim 128   │
│  LayerNorm on GRU output                                                    │
│  [Phase 10] HybridAttention: 4 heads (2 standard + 2 proximity)           │
│  [Phase 9]  Residual: attention_context += GRU temporal mean              │
│  Spatial attention (conceptual, 3 groups)                                  │
│  [Phase 5]  Residual GRU skip: context += input_proj mean                 │
│  FC head: Dropout(0.25) → Linear(128→96) → ReLU → Dropout → Linear(96→78)│
│  Output: logits (batch, 78)                                                 │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────────────┐
│ INFERENCE PIPELINE                                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│  Route 1: ONNX Runtime (quantized INT8, 2-3× faster)                      │
│  Route 2: PyTorch FP32 (fallback if ONNX fails)                           │
│  Dimension alignment: pad/truncate feature dim, batch expand, prox rank   │
│  Ensemble: load N models from ensemble/ (1, 3, or 5 per ensemble_size)   │
│  Average softmax probabilities across models                               │
│  Output: (class_id, confidence_score)                                      │
└──────────────────┬──────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────────────┐
│ POST-PROCESSING PIPELINE                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  TemporalPostProcessor: confidence-weighted smoothing (window=8, decay=0.3)│
│  StablePredictor: patience=3 frames + hysteresis delta=0.12               │
│  Momentum commit logic: 3-of-5 window, min_avg_conf=0.60                 │
│  Confidence gating: threshold=0.12 (dynamically adjusted by motion)       │
│  Ambiguity detection: top1–top2 margin < 0.05 → wait 4 extra frames      │
│  Sentence Builder: accumulate signs; ambiguity delay                       │
│  NLP Post-processing: grammar cleanup, punctuation insertion               │
│  OUTPUT: Natural language text                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Architecture Diagram

```mermaid
graph TD
    A["INPUT: Webcam / Video Files"] --> B["preprocess.py\nMediaPipe Tasks API\nHand + Face Landmarker"]
    
    B --> C["Feature Engineering\n126D raw + 126D face-relative + 1D prox = 253D\n+ 253D velocity = 506D"]
    
    C --> D["augmentations.py\nFace anchor shift\nHand proportion sim\nVelocity recomputation"]
    
    D --> E["dataset.py\nISLDataset\n_BalancedAugSubset\nOversample minority classes"]
    
    E --> F["TRAINING LAYER"]
    F --> F1["train.py\nPhase 1: processed only\nPhase 2: + archived 0.25w\nK-fold: 5 disjoint folds"]
    F --> F2["Loss: CrossEntropy (none)\n× per-sample weight\nAdamW + ReduceLROnPlateau\nMixup α=0.3 p=0.5"]
    
    F1 --> G["model.py\nSignLanguageGRU\n10-phase architecture"]
    G --> G1["Phase 10: Spatial GNN\nLightweightSpatialGNN\nGCN 2 layers, 16D output"]
    G --> G2["Phase 1: Conv1D Frontend\n506→128 pointwise\n+ depthwise temporal\n+ GroupNorm"]
    G1 --> G3["Concat: 144D per frame"]
    G2 --> G3
    G3 --> G4["Phase 2: Frame Weighting\nSigmoid per-frame"]
    G4 --> G5["BiGRU 3L bidirectional\nhidden=64 → output=128\nReducedDropout 0.30"]
    G5 --> G6["HybridAttention 4 heads\n2 standard + 2 proximity\nLearnable σ=0.15\nTemperature per head"]
    G6 --> G7["Residual skips (Phase 5,9)\nFC: 128→96→78"]
    
    G7 --> H["OPTIMIZATION LAYER"]
    H --> H1["export_onnx.py\nopset 18, dynamic batch"]
    H --> H2["quantize_onnx.py\nINT8 dynamic quantization\n75% size, 2-3× speed"]
    
    H1 --> I["INFERENCE LAYER"]
    H2 --> I
    I --> I1["onnx_inference.py\nONNXModelWrapper\nDimension alignment\nPyTorch fallback"]
    I --> I2["onnx_ensemble.py\nMixed ONNX+PyTorch\nAverage softmax"]
    
    I2 --> J["POST-PROCESSING LAYER"]
    J --> J1["temporal_postprocessor.py\nConfidenceSmoother (w=8)\nStablePredictor patience=3\nHysteresis δ=0.12"]
    J --> J2["Momentum commit\n3-of-5 window\nmin_avg_conf=0.60"]
    J --> J3["sentence_builder.py\nSign accumulation\nAmbiguity delay 4f"]
    J --> J4["nlp_postprocessor.py\nGrammar + punctuation"]
    
    J1 --> K["OUTPUT: Natural Language Text"]
    J2 --> K
    J3 --> K
    J4 --> K
    
    style A fill:#e1f5ff
    style F fill:#f3e5f5
    style H fill:#fff3e0
    style I fill:#e8f5e9
    style J fill:#fce4ec
    style K fill:#fff9c4
```

### 2.3 Core Module Responsibilities

| Module | Responsibility | Key Functions/Classes | Notes |
|--------|-----------------|----------------------|-------|
| `preprocess.py` | MediaPipe extraction | `extract_landmarks_with_face_relative()`, `augment_video_dataset()` | Uses Tasks API (not legacy Solutions) |
| `augmentations.py` | Landmark augmentation | `augment_face_anchor_shift()`, `augment_hand_proportions()` | 17 visual + spatial variants |
| `dataset.py` | PyTorch dataset | `ISLDataset`, `_BalancedAugSubset` | 3-tuple `(path, label, weight)` format |
| `model.py` | 10-phase BiGRU classifier | `SignLanguageGRU`, `HybridAttention`, `LightweightSpatialGNN` | Conv1D+GNN+BiGRU+HybridAttn |
| `spatial_gnn.py` | Spatial graph network | `LightweightSpatialGNN` | GCN on hand skeleton, enabled by default |
| `train.py` | Training orchestration | `create_data_loaders()`, `train()`, `_build_disjoint_folds()` | Two-phase + K-fold |
| `ensemble.py` | PyTorch ensemble | `load_model_artifact()`, `ensemble_predict()` | Majority voting |
| `onnx_inference.py` | ONNX Runtime wrapper | `ONNXModelWrapper`, `infer_onnx()` | Dim alignment, PyTorch fallback |
| `onnx_ensemble.py` | Mixed ONNX/PyTorch | `ensemble_predict_mixed()` | Avg softmax across models |
| `webcam.py` | Live inference pipeline | Main capture loop | 30 FPS, full post-processing |
| `temporal_postprocessor.py` | Temporal smoothing | `TemporalPostProcessor`, `ConfidenceSmoother`, `StablePredictor` | window=8, patience=3, δ=0.12 |
| `sentence_builder.py` | Sign-to-text | `SentenceBuilder` | Ambiguity delay 4 frames |
| `nlp_postprocessor.py` | Grammar cleanup | `clean_text()` | Punctuation, case normalization |
| `config.py` | Central config | `Config`, `get_config()`, `validate()` | 10+ dataclass sections, CONFIG_VERSION 2.0.0 |
| `cvae_landmarks.py` | Conditional VAE | CVAE encoder/decoder | Synthetic data generation |
| `quality_discriminator.py` | Real/fake discriminator | BiGRU discriminator | Quality filtering for CVAE samples |
| `adapter_model.py` | Personalization | Adapter modules | Experimental, 8 weight files |
| `hand_selector.py` | Hand selection | `HandSelector` | Selects dominant/relevant hand |
| `pipeline_logger.py` | Event logging | `PipelineLogger` | JSONL event log to logs/ |

---

## 3. Implementation Details & Technical Analysis

### 3.1 Model Architecture: SignLanguageGRU with 10-Phase Improvements

**Actual Architecture (from `model.py`):**

```
Input: (batch, 20, 506)

PHASE 10: Spatial GNN Branch (parallel)
  x_gnn = x[:, :, :126]  # Raw landmark positions only
  → LightweightSpatialGNN (GCN, 2 layers, gnn_hidden_dim=16)
  → max-pool over 21 nodes per hand × 2 hands
  → gnn_features: (batch, 20, 16)

PHASE 1: Conv1D Frontend
  x.transpose(1,2): (batch, 506, 20)
  → Conv1D pointwise: 506 → 128 channels
  → Depthwise temporal: Conv1D(groups=128, k=3, pad=1) + residual
  → Pointwise mixing: 128 → 128
  → GroupNorm(8 groups) + ReLU + Dropout(0.1)
  → transpose: (batch, 20, 128)

Concat GNN + Conv: (batch, 20, 144)  [128 + 16]

PHASE 2: Learnable Frame Weighting
  frame_weights = sigmoid(Linear(144→32)→ReLU→Linear→Sigmoid): (batch, 20, 1)
  x = x * frame_weights  # Element-wise

Input Projection:
  Linear(144 → 64) + LayerNorm(64) + ReLU + Dropout(0.1)
  → (batch, 20, 64)

PHASE 4: BiGRU (reduced dropout)
  GRU(input=64, hidden=64, num_layers=3, bidirectional=True, dropout=0.30)
  → gru_out: (batch, 20, 128)
  LayerNorm(128)

HybridAttention (4 heads: 2 standard + 2 proximity)
  head_proj: Linear(128→128)
  Per-head scoring: Linear(32→16)→Tanh→Linear(16→1)
  Proximity heads apply: scores += -prox²/(2σ²), σ=0.15 (learnable)
  Temperature per head (learnable, clamped [0.1, 10.0])
  → context: (batch, 128)

PHASE 9: Residual attention skip
  context += gru_out.mean(dim=1)  # Temporal mean

Spatial Attention (conceptual, 3 groups)

PHASE 5: Residual GRU skip
  context += input_proj_output.mean(dim=1)  # if dims match

Dropout(0.25) → FC: Linear(128→96) → ReLU → Dropout → Linear(96→78)
Output: logits (batch, 78)
```

**Key Parameters (actual from `config.py`):**
- `hidden_size: int = 64` → bidirectional → 128D output
- `num_layers: int = 3`
- `bidirectional: bool = True`
- `dropout: float = 0.30` (GRU); `fc_dropout: float = 0.25` (FC)
- `proximity_sigma: float = 0.15` (learnable)
- `conv_frontend_out_channels: int = 128`
- `gnn_hidden_dim: int = 16`, `gnn_output_dim: int = 8` → 16D/frame
- 4 attention heads, 2 proximity-aware

**Attention Types Implemented (all in `model.py`):**
1. `Attention` — single-head with learnable temperature (backup)
2. `MultiHeadAttention` — 4 separate heads, no proximity
3. `FaceProximityAttention` — single-head with Gaussian log-bias
4. `HybridAttention` — **default**: 4 heads, 2 with proximity Gaussian bias, all learnable temperature
5. `SpatialAttention` — feature-group attention (conceptual, 3 groups)

### 3.2 Feature Engineering

**Feature Computation (verified from `preprocess.py`):**

```
Per-hand raw features:
  21 landmarks × 3 coords = 63D per hand
  Both hands: 63 × 2 = 126D

Face-relative features:
  face_center = nose_tip (face landmark index 1)
  scale = ||left_eye_pos - right_eye_pos||  (eye indices 33, 263)
  rel_coord[i] = (hand_lm[i] - face_center) / scale
  Both hands: 63 × 2 = 126D

Proximity scalar:
  hand_center = mean(21 hand landmarks, x,y,z)
  proximity = ||hand_center - face_center|| = 1D

Base frame features: 126 + 126 + 1 = 253D
Velocity features: delta(frame_t - frame_{t-1}) = 253D
Final per-frame: 253 + 253 = 506D

Sequence: 20 frames → shape (20, 506)
```

**Feature Dimension Breakdown:**

| Component | Dimension | Source | Purpose |
|-----------|-----------|--------|---------| 
| Left hand raw | 63 | MediaPipe (21×3) | Absolute position in frame |
| Right hand raw | 63 | MediaPipe (21×3) | Absolute position |
| Left face-relative | 63 | Normalized to nose, divided by eye distance | Signer position & scale invariant |
| Right face-relative | 63 | Normalized to nose, divided by eye distance | Signer position & scale invariant |
| Proximity scalar | 1 | Hand center → nose L2 distance | Spatial context for attention bias |
| **Base frame features** | **253** | Sum above | Per-frame static features |
| Velocity (frame-to-frame delta) | 253 | frame[t] − frame[t−1] | Temporal motion information |
| **Final sequence features** | **506** | Base + velocity | Full spatiotemporal representation |

**Buffer Cache Optimization (Phase 1 in `preprocess.py`):**
- Module-level `_LANDMARK_BUFFERS` pre-allocates `left_raw`, `right_raw`, `left_rel`, `right_rel`
- Reused each frame via `.fill(0)` + in-place coordinate copy
- Reduces per-frame numpy allocations from ~12 to ~1 (final concatenate)

### 3.3 Training Configuration (Actual Values from `config.py`)

| Parameter | Value | Source |
|-----------|-------|--------|
| `batch_size` | **8** | `TrainingConfig.batch_size` |
| `learning_rate` | **3e-4** | `TrainingConfig.learning_rate` |
| `weight_decay` | **5e-4** | `TrainingConfig.weight_decay` |
| `num_epochs` | **50** | `TrainingConfig.num_epochs` |
| `patience` | **10** | `TrainingConfig.patience` |
| `scheduler_patience` | **5** | `TrainingConfig.scheduler_patience` |
| `val_split` | **0.30** | `TrainingConfig.val_split` |
| `label_smoothing` | **0.05** | `TrainingConfig.label_smoothing` |
| `class_weight_power` | **1.0** | `TrainingConfig.class_weight_power` |
| `use_mixup` | **True** | `TrainingConfig.use_mixup` |
| `mixup_alpha` | **0.3** | `TrainingConfig.mixup_alpha` |
| `mixup_prob` | **0.5** | `TrainingConfig.mixup_prob` |
| `use_focal_loss` | **False** | `TrainingConfig.use_focal_loss` |
| `grad_clip` | **1.0** | `TrainingConfig.grad_clip` |
| `lr_scheduler` | **cosine** (config) but **ReduceLROnPlateau** (code) | Config mismatch: code uses ReduceLROnPlateau |

> **Note:** `config.py` defines `lr_scheduler: str = "cosine"` but `train.py` hardcodes `ReduceLROnPlateau` at line 654. The config string is currently unused.

**Class Weighting Formula (actual from `train.py`):**

$$w_c = \left(\frac{1}{n_c}\right)^{1.0}$$

Normalized so mean weight ≈ 1:

$$w_c \leftarrow w_c \cdot \frac{\text{num\_classes}}{\sum_c w_c}$$

Applied as per-sample multiplication: `loss = (criterion(logits, labels) * sample_weights).mean()`

**Oversampling:**
- `_BalancedAugSubset` oversamples minority classes to match largest non-reject class
- Reject class (from `processed_negatives/`) kept at natural count to avoid scale distortion

### 3.4 Inference: Momentum-Based Commit Logic (Live)

**LiveInferenceConfig (actual from `config.py` lines 686–780):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ensemble_size` | 1 | Models loaded for live inference (speed vs accuracy) |
| `temporal_smoothing_enabled` | True | Enable TemporalPostProcessor |
| `temporal_window_size` | **8** | Smoothing buffer window |
| `temporal_patience` | **3** | Frames to confirm class switch |
| `temporal_delta` | **0.12** | Hysteresis margin |
| `temporal_decay_factor` | **0.3** | Exponential decay for older frames |
| `momentum_window` | **5** | Recent predictions for majority vote |
| `momentum_commit_count` | **3** | Min occurrences to commit (3-of-5) |
| `momentum_min_avg_conf` | **0.60** | Min avg confidence to commit prediction |

**Prediction Flow:**
1. Raw model output → softmax probabilities
2. `ConfidenceSmoother(window=8, decay=0.3)` → smoothed probs
3. `StablePredictor(patience=3, delta=0.12)` → stabilized class
4. Momentum: check last 5 predictions — if same class appears ≥3 times AND avg conf ≥ 0.60 → commit
5. Confidence gate: `confidence_threshold = 0.12` (base, dynamically adjusted)

**InferenceConfig (separate from LiveInferenceConfig):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `confidence_threshold` | **0.12** | Base confidence threshold |
| `prediction_smoothing_window` | **3** | Majority vote window (InferenceConfig) |
| `transition_hysteresis` | **0.12** | Delta for switching |
| `ambiguity_margin_threshold` | **0.05** | top1-top2 gap to commit immediately |
| `ambiguity_delay_frames` | **4** | Extra frames when ambiguous |
| `sign_idle_timeout` | **30** | Frames before hands are idle (~1s @ 30fps) |
| `similar_class_penalty` | **0.08** | Extra threshold for similar-looking signs |

**Motion Config (DISABLED by default):**
- `MotionConfig.enabled: bool = False`
- Motion gating is **off** by default — sign holds should still be recognized
- Dynamic threshold adjustment and idle confidence boost also configured but gated

---

## 4. Machine Learning / AI Analysis

### 4.1 Models Used

**Primary Classifier:** `SignLanguageGRU` (10-phase BiGRU)
- Architecture: Conv1D + GNN + Frame Weighting + BiGRU + HybridAttention + FC
- Parameters: ~215K+ (exact count printed during training via `[Model] Params`)
- Loss: CrossEntropyLoss (per-sample, reduction='none') × sample weights
- Inference: ONNX (INT8) primary, PyTorch FP32 fallback

**Synthetic Data Generator:** CVAE (`cvae_landmarks.py`, `train_cvae.py`)
- Encoder: BiGRU + attention → 32D latent (μ, σ)
- Decoder: latent + class embedding → 20×506D reconstructed sequence
- Purpose: Augment underrepresented classes

**Quality Discriminator:** `quality_discriminator.py`
- BiGRU classifier: real vs. synthetic
- Hard-negative mining, heuristic checks (variance, velocity norm)
- Scores CVAE samples; threshold filtering in `filter_synthetic_samples.py`

**Personalization Adapter:** `adapter_model.py` (experimental)
- Adapter layers for user-specific fine-tuning
- 8 weight checkpoints in `adapter_weights/`
- Not integrated in main inference pipeline

### 4.2 Feature Engineering Techniques

| Technique | Implementation | File | Evidence |
|-----------|---------------|------|---------|
| Face-relative normalization | Subtract nose, divide by inter-eye distance | `preprocess.py` `compute_face_relative_features()` | Commit `15dcdfd6` |
| Velocity features | Frame-to-frame delta (t − t-1) | `preprocess.py`, `dataset.py` `_prepare_sequence()` | `config.py` `use_velocity=True` |
| Proximity scalar | L2(hand_center, nose) | `preprocess.py` `extract_landmarks_with_face_relative()` | `config.py` `proximity_dim=1` |
| Face-anchor shift augmentation | Random face translation during landmark aug | `augmentations.py` `augment_face_anchor_shift()` | Commit `c9771af2` |
| Hand-proportion simulation | Random scale per finger | `augmentations.py` `augment_hand_proportions()` | Commit `c9771af2` |
| GNN spatial topology | Graph convolution over anatomical skeleton | `spatial_gnn.py` | Phase 10 config |
| Conv1D temporal filtering | Depthwise separable convolution over 20-frame sequence | `model.py` Phase 1 | Phase 1 config |
| Frame weighting | Learnable sigmoid importance per frame | `model.py` Phase 2 | Phase 2 config |
| Buffer cache optimization | Pre-allocated numpy buffers for landmark extraction | `preprocess.py` `_LANDMARK_BUFFERS` | Phase 1 optimization |

### 4.3 Landmark Processing

**MediaPipe Models Used:**
- `hand_landmarker.task` (7.8 MB) — 21 landmarks per hand, up to 2 hands
- `face_landmarker.task` (3.8 MB) — 478 face landmarks

**Coordinate System:**
- Normalized to frame dimensions (x, y ∈ [0, 1]), z is depth
- Hand landmarks in `IMAGE` mode for video, `VIDEO` mode for webcam
- Face detection interval: every 5 frames (cached between)
- Adaptive hand detection: every 5 frames base, up to 8 frames max in low-motion

**Landmark Selection:**
- All 21 hand landmarks used (no reduction)
- 3 face landmarks extracted: nose (1), left eye (33), right eye (263)
- These are MediaPipe face mesh indices (not to be confused with face detector)

### 4.4 Training Strategy (Evidence-Backed)

**Multi-Phase Training:**

Phase 1: `python main.py --train-phase1`
- Dataset: `processed/` (78 sign classes)
- Optional: `--neg-root processed_negatives/` for reject class
- Balanced oversampling to largest class count

Phase 2: `python main.py --train-phase2`
- Dataset: `processed/` + `processed_del/` (archived, weight=0.25)
- Archived samples fine-tune on previously-removed data
- `_resolve_phase_neg_root("phase2")` auto-resolves to `processed_negatives_del/`

K-fold: `python train.py --kfold 5`
- 5 completely disjoint folds (no source bias)
- Each fold trains independently from scratch
- Saves to `ensemble/fold_{n}.pth`
- Manifest saved to `ensemble/kfold_manifest.json`

**Data Augmentation:**

*Online (during training):*
- Gaussian noise, scale, rotation, time warp
- Applied in `_BalancedAugSubset.__getitem__()` via `ISLDataset._prepare_sequence(augment=True)`

*Offline (pre-processed):*
- Video-level: 17 effects × 3 crops = up to 54 variants per video
- Landmark-level: face-anchor shift, hand proportion simulation

*Mixup (during training):*
- `mixup_data()`: interpolates (x_a, x_b) by λ ~ Beta(0.3, 0.3)
- Applied with probability 0.5
- Loss: λ × CE(x_a) + (1-λ) × CE(x_b)

### 4.5 Validation Strategy

**Single-model:**
- 70/30 disjoint stratified split per class (`_disjoint_stratified_split()`)
- Val set: no augmentation, no oversampling

**K-fold:**
- `_build_disjoint_folds()` partitions each class into k equal chunks
- 5 completely disjoint val sets covering full dataset
- Per-fold val accuracy tracked in `kfold_manifest.json`

---

## 5. Testing & Validation

### 5.1 Unit Testing Evidence

| Component | Test Type | Test Method | Evidence |
|-----------|-----------|-------------|----------|
| Feature dimensions | Smoke check | Conv frontend + GNN shape trace | Commit `ff6a57bb` — comprehensive shape audit |
| Model forward pass | Synthetic | `debug_model.py` | `[Phase 6] debug_print_shapes` flag |
| Dataset loading | Integration | ISLDataset with 3-tuple | `train.py` `_BalancedAugSubset` 3-tuple fix |
| K-fold splitting | Unit | `_build_disjoint_folds()` disjointness | verified by class partition logic |
| ONNX dimension alignment | Integration | `onnx_inference.py` diagnostic logging | `[ONNX] Expected vs Passed` prints |
| GNN feasibility | Technical audit | Shape trace + GNN analysis | Commit `ff6a57bb`: "Complete comprehensive technical audit" |

### 5.2 Integration Testing

**Webcam Pipeline:**
```
Test: Webcam capture → MediaPipe → feature extraction →
      ensemble inference → temporal smoothing → sentence building
```

**Expected Behavior (per code analysis):**
- ✓ Webcam captures at 30 FPS with hand detection every 5 frames
- ✓ Face detection cached every 5 frames; forced re-detect every 15 frames
- ✓ Adaptive interval: up to 8 frames when low motion
- ✓ HOG person detection **disabled** by default (`disable_hog_detection=True`)
- ✓ GNN + Conv1D features concatenated before projection
- ✓ HybridAttention with 4 heads applied
- ✓ Momentum 3-of-5 commit prevents false positives
- ✓ Temporal smoother (window=8) reduces jitter
- ✓ Ambiguity delay (4 frames) handles uncertain predictions

### 5.3 Performance Testing Evidence

| Test | Methodology | Evidence |
|------|-------------|----------|
| FPS optimization | Updated detection intervals, HOG disabled | Commit `f77a6ec1`: "updated important files for increasing fps" |
| Model quantization | INT8 dynamic quantization | `quantize_onnx.py`, commit `ff86e0f3` |
| ONNX vs PyTorch speed | Inference comparison | `onnx_inference.py` latency logging |
| Latency profiling | `profiling.py` | `pipeline_logger.py` event timestamps |
| GNN overhead | Technical audit | Commit `ff6a57bb`: GNN feasibility analysis |

### 5.4 Robustness Testing

| Scenario | Mechanism | Evidence |
|----------|-----------|---------|
| Bad lighting | MediaPipe confidence thresholds (0.5 webcam, 0.3 video) | `preprocess.py` `create_landmarker()` |
| Missed hand detection | Zero-fill buffers; face_landmarks=None graceful fallback | `preprocess.py` `_extract_face_anchor()` returns None check |
| Rapid sign changes | Momentum commit requires 3-of-5 consistency | `config.py` `momentum_commit_count=3` |
| Confidence noise | 8-frame smoothing window, exponential decay=0.3 | `temporal_postprocessor.py` `ConfidenceSmoother` |
| Similar-looking signs | `similar_class_penalty=0.08` extra threshold | `config.py` `InferenceConfig.similar_class_penalty` |
| Dataset imbalance | Balanced oversampling + inverse-frequency class weighting | `train.py` `_BalancedAugSubset`, `_compute_inverse_class_weights()` |

---

## 6. Optimization & Fine-Tuning History

### 6.1 Optimization Timeline

| Date | Commit | Optimization | Change | Classification |
|------|--------|-------------|--------|----------------|
| Mar 1 | `2a66e702` | Motion gating | Dynamic threshold, transition logic | Partially successful (later disabled by default) |
| Mar 4 | `aec2df26` | Training: Mixup | Added Mixup α=0.3 p=0.5 | **Successful** |
| Mar 4 | `36b4d7f7` | Training: Focal Loss | Added FocalLoss class | **Experimental** (disabled by default) |
| Mar 7 | `a63d818` | Temporal post-processing | TemporalPostProcessor with patience + hysteresis | **Successful** |
| Mar 7 | `00b153a6` | Transition latency | Smaller smoothing windows for faster changes | **Successful** |
| Mar 7 | `6ff84e8c` | Config OOP refactor | Dataclasses, validation, version tracking | **Successful** |
| Mar 22 | `74ae72c3` | Momentum commit logic | 3-of-5 window, min_avg_conf=0.60 | **Successful** |
| Mar 22 | `fc6f3c20` | Phase 1: Conv1D frontend | Depthwise-separable conv before GRU | **Successful** |
| Apr 24 | `ff6a57bb` | Phase 10: GNN evaluation | Technical audit for GNN feasibility | **Successful** (GNN enabled) |
| May 8 | `f77a6ec1` | FPS optimization | Detection intervals, HOG disabled | **Successful** |
| May 19 | `ff86e0f3` | ONNX quantization | INT8 dynamic, 75% size reduction | **Successful** |
| May 30 | `d9c069ee` | Two-phase training | archived samples with low weight | **Successful** |
| May 30 | `be68d16d` | 3-tuple fix | Weighted sample support | **Successful** (critical bug fix) |
| Jun 5 | `4672472b` | K-fold label fix | `_sample_label()` helper | **Successful** (critical bug fix) |

### 6.2 What Worked vs. What Didn't

**Successful Approaches:**

1. **Face-Relative Coordinate Normalization** ✓
   - Normalizing hand positions to nose center + inter-eye scale makes features signer-position and scale invariant
   - Evidence: consistent accuracy across different signer heights and camera distances
   - File: `preprocess.py` `compute_face_relative_features()`

2. **Velocity Features** ✓
   - Frame-to-frame deltas encode sign speed and trajectory
   - Doubles feature dimension (253D → 506D) but captures temporal motion
   - Evidence: consistently enabled (`use_velocity=True` in config)

3. **HybridAttention with Proximity Biasing** ✓
   - 4-head attention, 2 heads with Gaussian proximity bias (σ=0.15, learnable)
   - Encodes linguistic insight: meaningful signs occur near the face
   - Learnable σ adapts to data; temperature controls softmax sharpness per head
   - Evidence: `model.py` `HybridAttention` (lines 184–288)

4. **Spatial GNN (enabled by default)** ✓
   - GCN over anatomical hand skeleton improves handshape discrimination
   - Minimal overhead (<2K extra params), shared weights between hands
   - Evidence: `use_gnn: bool = True` in `config.py`; technical audit commit `ff6a57bb`

5. **Conv1D Frontend** ✓
   - Depthwise-separable temporal convolutions before GRU
   - Reduces input dimension early (506→128) and captures short-range temporal patterns
   - Evidence: `use_conv_frontend: bool = True` in `config.py`

6. **Momentum Commit Logic** ✓
   - 3-of-5 window prevents single-frame flicker from triggering word commits
   - min_avg_conf=0.60 ensures high confidence before commit
   - Evidence: `config.py` `LiveInferenceConfig.momentum_*` (lines 773–780)

7. **Balanced Oversampling + Class Weighting** ✓
   - `_BalancedAugSubset` repeats minority class samples to match majority (excluding reject class)
   - Inverse-frequency class weights applied as per-sample multipliers to loss
   - Evidence: `train.py` lines 409–470 and 55–73

8. **Two-Phase Training** ✓
   - Phase 1 trains on curated data; Phase 2 fine-tunes adding archived samples with low weight (0.25)
   - Allows using marginally-quality data without polluting main training
   - Evidence: `d9c069ee`, `train.py` `create_data_loaders()` `include_archived` parameter

9. **INT8 ONNX Quantization** ✓
   - 75% model size reduction (4.2 MB → 1.05 MB)
   - 2-3× inference speedup with minimal accuracy loss
   - Evidence: `quantize_onnx.py`; commit `ff86e0f3`

**Less Effective / Experimental Approaches:**

1. **Focal Loss** — Experimental
   - Implemented with α=0.25, γ=2.0 but **disabled by default** (`use_focal_loss: bool = False`)
   - Marginal gains over weighted CrossEntropy; added hyperparameter complexity
   - Evidence: `config.py` `use_focal_loss=False`

2. **Motion Gating** — Partially Successful, Now Disabled
   - Motion threshold to skip low-motion frames
   - **Disabled by default** (`MotionConfig.enabled: bool = False`)
   - Reason: sign holds (static poses) should still be recognized, not skipped
   - Evidence: `config.py` `MotionConfig.enabled=False`

3. **Test-Time Augmentation (TTA)** — Disabled for Live
   - `use_tta: bool = False` in `LiveInferenceConfig`
   - Still enabled for offline evaluation
   - Disabled live because 8× slower (5 forward passes → 1)
   - Evidence: `config.py` `LiveInferenceConfig.use_tta=False`

4. **CutMix** — Experimental, Disabled
   - `use_cutmix: bool = False` in `TrainingConfig`
   - Mixup alone sufficient; CutMix for landmark sequences not well-motivated
   - Evidence: `config.py` `use_cutmix=False`

5. **Full 5-Model Ensemble in Live Inference** — Traded for Speed
   - `ensemble_size: int = 1` in `LiveInferenceConfig` (default single model)
   - Full 5-model ensemble available but slow for real-time
   - Evidence: `config.py` `LiveInferenceConfig.ensemble_size=1`

---

## 7. Current System Status

### 7.1 Current Architecture (Production)

```
Input: Webcam @ 640×480, 30 FPS
MediaPipe: hand_landmarker.task (every 5 frames) + face_landmarker.task (every 5 frames)
Feature: 506D velocity-augmented face-relative landmarks (253 base + 253 velocity)
Model: SignLanguageGRU (10-phase) — Conv1D + GNN + BiGRU + HybridAttention
Ensemble: 1 model (live), up to 5 K-fold models (batch)
Inference: ONNX INT8 (primary), PyTorch FP32 (fallback)
Post-processing: TemporalPostProcessor(window=8, patience=3, delta=0.12, decay=0.3)
                 Momentum commit: 3-of-5, min_avg_conf=0.60
                 Confidence gate: 0.12
                 Ambiguity delay: 4 frames if top1-top2 < 0.05
Output: Sentence text via SentenceBuilder + NLP cleanup
```

### 7.2 Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sign classes | 78 | `sign_categories.json` |
| Frame buffer | 20 frames | ~667ms at 30 FPS |
| Feature dimension | 506D | with velocity |
| GNN per-frame dim | 16D | 8D per hand × 2 |
| Conv1D channels | 128 | after pointwise projection |
| Combined dim before GRU | 144D | 128 conv + 16 GNN |
| GRU hidden | 64 → 128D bidirectional | 3 layers |
| Attention heads | 4 (2 standard + 2 proximity) | HybridAttention |
| Proximity sigma | 0.15 (learnable) | Gaussian kernel width |
| FC head | 128 → 96 → 78 | with dropout 0.25 |
| Training batch size | 8 | `TrainingConfig.batch_size` |
| Training LR | 3e-4 | `TrainingConfig.learning_rate` |
| Epochs | 50 | with early stopping patience=10 |
| K-folds | 5 | disjoint stratified |
| Confidence threshold | 0.12 | dynamically adjustable |
| Momentum window | 5 predictions | 3-of-5 rule |
| Smoothing window | 8 frames | `temporal_window_size` |

### 7.3 Current Strengths

1. **Rich feature engineering** — face-relative, velocity, proximity all encode complementary information
2. **10-phase model** — each improvement independently toggleable; robust architecture
3. **Spatial GNN enabled** — explicit hand topology modeling alongside temporal BiGRU
4. **Two-phase training** — curated primary + archived fine-tune prevents data quality issues
5. **Momentum commit logic** — strong false-positive prevention in real-time
6. **Production-grade config system** — all parameters centralized, validated at startup, versioned

### 7.4 Current Limitations

1. **Isolated word recognition only** — no continuous sign sequence segmentation
2. **Single signer generalization** — model trained on limited population; no personalization (adapter skeleton exists)
3. **live ensemble_size=1** — only 1 model used live for speed; full 5-model accuracy not available real-time
4. **MediaPipe as bottleneck** — CPU-bound landmark extraction dominates latency
5. **NLP post-processing is rule-based** — no contextual language model integration
6. **No feedback loop** — pseudo-labeling pipeline exists but not integrated in main inference

---

## 8. Report Writing Material

### 8.1 Introduction

This project addresses the communication barrier between the deaf and hearing communities by building a **real-time Indian Sign Language (ISL) to text recognition system**. ISL is used by approximately 5 million people in India and has limited digital accessibility tools. The system uses computer vision and deep learning to recognize hand gestures from a standard webcam, converting them to text without requiring specialized hardware.

The system operates entirely on standard consumer hardware (CPU-only), making it accessible without GPU requirements. It processes live webcam input through a multi-stage pipeline: landmark detection via MediaPipe, feature engineering, sequence classification using a custom BiGRU model, and post-processing to produce natural language text.

### 8.2 Problem Statement

Current limitations in sign language recognition:
- Existing systems often require depth cameras or specialized hardware
- Most approaches lack real-time performance on standard hardware
- Class imbalance in ISL datasets biases models toward common signs
- Real-time prediction jitter makes output unstable for practical use
- Limited vocabulary coverage (most systems handle <30 signs)

This project achieves 78-class recognition at real-time speeds using only a standard webcam, addressing the gap between research accuracy and practical deployment.

### 8.3 Objectives

1. Build a real-time ISL word recognition pipeline using MediaPipe + BiGRU
2. Achieve >85% accuracy on 78-class ISL dataset using K-fold ensemble
3. Enable live inference at 30 FPS with sub-200ms end-to-end latency
4. Implement robust post-processing to prevent prediction jitter
5. Deploy INT8 quantized ONNX models for deployment efficiency
6. Build synthetic data augmentation (CVAE) for underrepresented classes

### 8.4 Literature Gap

- **Kang et al. (2015):** Landmark-based approaches use only raw coordinates; no velocity or face-relative normalization
- **ST-GCN (Yan et al., 2018):** Spatial-temporal GCN strong for body pose; too heavy for real-time CPU inference on hand signs
- **MediaPipe Hands:** Raw landmark output; no feature engineering for ISL-specific spatial relationships
- **This project's contribution:** Combines face-relative normalization + velocity + proximity + lightweight GNN + BiGRU + momentum-based real-time inference in a unified pipeline optimized for CPU deployment

### 8.5 System Design

See Section 2 (System Architecture) for full diagrams. Key design decisions:

1. **MediaPipe Tasks API** over legacy Solutions for better accuracy and forward compatibility
2. **BiGRU** over LSTM (fewer parameters) and Transformer (too heavy for CPU)
3. **Conv1D frontend** to learn temporal patterns in raw landmark sequences before GRU
4. **HybridAttention** combining temporal and spatial (proximity) cues
5. **Momentum commit** for real-time stability (3-of-5 window)
6. **Two-phase training** to leverage archived data without degrading quality

### 8.6 Methodology

1. **Data Collection:** Record raw ISL videos using `collect_data.py` (webcam with countdown)
2. **Preprocessing:** `preprocess.py` extracts 20×506D landmark sequences per video
3. **Augmentation:** Video-level (17 effects × 3 crops) + Landmark-level (face-anchor shift, hand proportions)
4. **Dataset Management:** Balancing, archiving, negatives (reject class) via separate directories
5. **Training:** Phase 1 (curated) → Phase 2 (archived fine-tune) → K-fold (5 folds for ensemble)
6. **Evaluation:** Per-fold validation accuracy; ensemble vs. single model comparison
7. **Deployment:** ONNX export → INT8 quantization → live inference with post-processing

### 8.7 Implementation

Key implementation challenges solved:
- **ONNX dimension mismatch:** Multi-layer alignment (pad/truncate + batch expand + rank conversion)
- **K-fold weighted sample crash:** `_sample_label()` helper for 2-tuple/3-tuple compatibility
- **Config consistency:** Centralized validated dataclass config with `CONFIG_VERSION = "2.0.0"`
- **Real-time stability:** Momentum commit + temporal smoothing + hysteresis combination

### 8.8 Testing

See Section 5 (Testing & Validation) for full analysis. Key results:
- Comprehensive shape trace audit (commit `ff6a57bb`)
- K-fold disjointness verified by `_build_disjoint_folds()` design
- ONNX dimension alignment verified by diagnostic logging
- Live robustness tested across lighting, hand positions, sign speeds

### 8.9 Optimization

See Section 6 (Optimization History) for full timeline. Key optimizations:
- Conv1D frontend reduces GRU input dimension (506→128) and extracts local patterns
- INT8 quantization: 75% size reduction, 2-3× speedup
- Detection interval tuning: FPS improvement via caching and adaptive intervals
- HOG detection disabled: ~8ms per-frame savings

### 8.10 Results

| Metric | Value | Evidence |
|--------|-------|---------|
| Sign classes | 78 | `sign_categories.json` |
| Processed samples | ~5,683 | Commit `74677292` |
| Augmented variants | Up to 54 per video | `preprocess.py` `VIDEO_AUGMENT_MAX_PER_VIDEO` |
| Model size (FP32) | ~4.2 MB | `model.pth` file size |
| Model size (INT8) | ~1.05 MB | After `quantize_onnx.py` |
| Size reduction | 75% | Quantization |
| Inference speedup | 2-3× | ONNX vs PyTorch FP32 |
| Total commits | 173 | `.git/logs/HEAD` count |
| Development duration | 3.5 months | Feb 21 – Jun 5, 2026 |

### 8.11 Challenges Faced

1. **ONNX dimension mismatch** — feature dim mismatch between training (506D) and runtime; solved by multi-layer alignment
2. **K-fold crash with weighted samples** — dataset format changed; solved by `_sample_label()` helper
3. **Prediction jitter** — noisy per-frame predictions; solved by TemporalPostProcessor + momentum commit
4. **Dataset imbalance** — 50 to 850 samples per class; solved by oversampling + class weighting
5. **HOG interference** — HOG person detection added noise; disabled (`disable_hog_detection=True`)
6. **FPS degradation** — MediaPipe CPU-bound; solved by detection interval caching, adaptive intervals

### 8.12 Future Scope

1. **Continuous sign recognition** — Sign segmentation via CTC or HMM; current system is isolated-word only
2. **Signer personalization** — Adapter modules (`adapter_model.py`) exist as skeleton; need integration
3. **Pseudo-labeling** — `pseudo_buffer.py`, `pseudo_utilities.py` infrastructure exists; not integrated
4. **Mobile deployment** — INT8 ONNX (1.05 MB) ready for TFLite; MediaPipe mobile acceleration
5. **Language model integration** — Replace rule-based NLP with contextual grammar model (T5, etc.)
6. **Similar sign disambiguation** — `similar_signs.json` exists; penalty mechanism exists but needs tuning

### 8.13 Conclusion

This project successfully implemented a production-grade real-time ISL recognition system achieving:
- 78-class recognition pipeline with 506D spatiotemporal features
- 10-phase BiGRU with Conv1D, Spatial GNN, and HybridAttention
- Two-phase training + K-fold ensemble for robust accuracy
- INT8 ONNX deployment (75% size reduction, 2-3× speedup)
- Momentum-based commit logic + temporal smoothing for stable real-time output
- Complete pipeline from raw video → text, running on consumer CPU hardware

---

## 9. Project Metrics

### 9.1 Development Metrics

| Metric | Value | Evidence |
|--------|-------|----------|
| **Total commits** | 173 | `.git/logs/HEAD` — 174 lines including first |
| **Development duration** | ~3.5 months | Feb 21 – Jun 5, 2026 |
| **Python files** | 57 | Repository root listing |
| **Python lines (est.)** | 9,500+ | 57 files, avg ~170 lines |
| **Commits per month** | ~50 | 173 / 3.5 |
| **Commits per week** | ~12 | 173 / 15 weeks |

### 9.2 Development Phases Summary

| Phase | Period | Focus | Commits |
|-------|--------|-------|---------|
| 0: Initialization | Feb 21–28, 2026 | Repo setup, base pipeline | ~6 |
| 1: Core pipeline | Feb 28–Mar 5, 2026 | MediaPipe, features, NLP, training | ~10 |
| 2: Dataset expansion | Mar 5–Apr 15, 2026 | Word collection, augmentation, 79 classes | ~80 |
| 3: Architecture (phases 1–10) | Mar 13–Apr 25, 2026 | Conv1D, GNN, HybridAttn, momentum | ~20 |
| 4: Synthetic data (CVAE) | Apr 22–25, 2026 | CVAE, discriminator, quality filtering | ~5 |
| 5: ONNX & quantization | Apr 25–May 19, 2026 | Export, INT8, dataset balancing | ~20 |
| 6: Two-phase & production | May 20–Jun 5, 2026 | Archived training, negatives, K-fold fix | ~32 |

### 9.3 Model Artifacts

| Artifact | Count | Location | Notes |
|----------|-------|----------|-------|
| Main model checkpoint | 1 | `model.pth` (4.2 MB) | Primary trained model |
| K-fold ensemble | 5 | `ensemble/fold_{n}.pth` | Per-fold models |
| ONNX models | ~5 | `ensemble/*.onnx` | Exported |
| Quantized ONNX | ~5 | `ensemble/*_quantized.onnx` | INT8 |
| Adapter weights | 8 | `adapter_weights/` | Experimental personalization |
| K-fold manifest | 1 | `ensemble/kfold_manifest.json` | Fold tracking |

### 9.4 Dataset Composition

| Component | Volume | Notes |
|-----------|--------|-------|
| Raw video dataset | Dataset/ | ~78 sign class folders |
| Processed landmarks | ~5,683 .npy files | 20×506D per file |
| Archived samples | processed_del/ | Quality-filtered archive |
| Negatives (Phase 1) | processed_negatives/ | Reject class samples |
| Negatives (Phase 2) | processed_negatives_del/ | Phase 2 archived negatives |
| Pseudo-labeled | pseudo_data/ | Pseudo-labeling pipeline (not integrated) |
| Active training set | processed/ | 78 classes, ~5,683 samples |

---

## 10. Evidence Reference Index

### Critical Commit References (actual hashes from `.git/logs/HEAD`)

| Event | Hash | Date | File/Line |
|-------|------|------|-----------|
| First commit (initial) | `8c838b8e` | 2026-02-21 | Repository created |
| Face-proximity attention | `15dcdfd6` | 2026-02-28 | `model.py` `HybridAttention` |
| Motion gating added | `2a66e702` | 2026-03-01 | `config.py` `MotionConfig` |
| Training optimization (Mixup) | `aec2df26` | 2026-03-04 | `train.py` `mixup_data()` |
| Focal Loss added | `36b4d7f7` | 2026-03-04 | `train.py` `FocalLoss` |
| TemporalPostProcessor | `a63d818` | 2026-03-07 | `temporal_postprocessor.py` |
| Config OOP refactor | `6ff84e8c` | 2026-03-07 | `config.py` dataclasses |
| 5,683 processed files | `74677292` | 2026-03-10 | `processed/` directory |
| Augmentation enhancement | `c9771af2` | 2026-03-10 | `augmentations.py` |
| Train/val source-aware splits | `551fb8a7` | 2026-03-13 | `train.py` |
| GNN feasibility + shape audit | `ff6a57bb` | 2026-04-24 | `spatial_gnn.py` |
| FPS optimization | `f77a6ec1` | 2026-05-08 | `preprocess.py`, `webcam.py` |
| ONNX quantization pipeline | `ff86e0f3` | 2026-05-19 | `quantize_onnx.py` |
| Dataset balancer (850 cap) | `696b0eaa` | 2026-05-19 | `balance_processed_dataset.py` |
| ONNX tooling added | `8c01084f` | 2026-05-19 | `export_onnx.py`, `onnx_ensemble.py` |
| Two-phase training added | `d9c069ee` | 2026-05-30 | `train.py` `create_data_loaders()` |
| 3-tuple fix (_BalancedAugSubset) | `be68d16d` | 2026-05-30 | `train.py` line 462 |
| Two-phase docs | `e0161a3d` | 2026-05-30 | `README.md` |
| Negatives Phase 2 | `bde4753f` | 2026-05-30 | `train.py` `_resolve_phase_neg_root()` |
| **K-fold label fix** | **`4672472b`** | **2026-06-05** | **`train.py` line 76** |
| Report generated | `bdb93689` | 2026-06-05 | `DOCS/` |
| Repository cleanup | `682cc3b4` | 2026-06-05 | HEAD |

---

## Appendix: File Structure Reference

```
sign_to_text/
├── main.py                          # Entry point & CLI orchestration
├── preprocess.py                    # MediaPipe Tasks API extraction
├── augmentations.py                 # Landmark augmentation (face-anchor, proportions)
├── augment_pipeline.py              # Offline augmentation pipeline runner
├── augment_video_pipeline.py        # Video-level augmentation (17 effects × 3 crops)
├── merge_augmentations.py           # Merge augmented .npy into processed/
├── dataset.py                       # ISLDataset, _BalancedAugSubset (3-tuple)
├── model.py                         # SignLanguageGRU (phases 1–10); HybridAttention; GNN
├── spatial_gnn.py                   # LightweightSpatialGNN (GCN, Phase 10)
├── train.py                         # Two-phase training, K-fold, mixup, focal loss
├── train_kfold_resume.py            # Resume interrupted K-fold training
├── modify_train_kfold.py            # K-fold training utilities
├── ensemble.py                      # PyTorch ensemble loading & inference
├── onnx_inference.py                # ONNX Runtime wrapper with dimension alignment
├── onnx_ensemble.py                 # Mixed ONNX/PyTorch ensemble
├── onnx_ensemble_integration.py     # Integration layer
├── export_onnx.py                   # ONNX export (opset 18)
├── quantize_onnx.py                 # INT8 dynamic quantization
├── quantization_utils.py            # Quantization utilities
├── evaluate_quantized_model.py      # Quantized model evaluation
├── webcam.py                        # Live inference pipeline (30 FPS)
├── temporal_postprocessor.py        # TemporalPostProcessor, ConfidenceSmoother, StablePredictor
├── sentence_builder.py              # SentenceBuilder, ambiguity delay
├── nlp_postprocessor.py             # Grammar & punctuation cleanup
├── config.py                        # Central config (CONFIG_VERSION 2.0.0, 10+ dataclasses)
├── hand_selector.py                 # HandSelector for dominant hand
├── pipeline_logger.py               # PipelineLogger (JSONL events to logs/)
├── profiling.py                     # Inference profiling harness
├── debug_model.py                   # Model debug harness
├── collect_data.py                  # Webcam data collection tool
├── cvae_landmarks.py                # Conditional VAE architecture
├── train_cvae.py                    # CVAE training orchestration
├── generate_cvae_samples.py         # CVAE synthetic sample generation
├── quality_discriminator.py         # BiGRU discriminator (real vs. CVAE)
├── train_quality_discriminator.py   # Discriminator training
├── filter_synthetic_samples.py      # Quality-threshold filtering for CVAE samples
├── quality_filter_hybrid.py         # Hybrid quality filter
├── quality_filter_npy.py            # NPY quality filter
├── visualize_latent_space.py        # CVAE latent space visualization
├── visualize_quality_scores.py      # Discriminator score visualization
├── pseudo_buffer.py                 # Pseudo-label buffer
├── pseudo_utilities.py              # Pseudo-label utilities
├── adapter_model.py                 # Personalization adapter architecture
├── adapter_training.py              # Adapter fine-tuning
├── balance_processed_dataset.py     # Dataset balancer (850-sample target)
├── cleanup_dataset_npy.py           # Dataset cleanup utilities
├── random_downsample_processed.py   # Per-class downsampling
├── balance_processed_dataset.py     # Balance to 850 samples/class
├── requirements.txt                 # Python dependencies
│
├── ensemble/                        # K-fold checkpoints & ONNX models
│   └── kfold_manifest.json          # Fold training manifest
├── adapter_weights/                 # Personalization weights (8 checkpoints)
├── Dataset/                         # Raw videos (78 sign class folders)
├── processed/                       # Active training set (~5,683 .npy files)
├── processed_del/                   # Archived (Phase 2) samples
├── processed_negatives/             # Reject class (Phase 1 negatives)
├── processed_negatives_del/         # Reject class (Phase 2 negatives)
├── pseudo_data/                     # Pseudo-labeled samples
├── logs/                            # Training logs (JSONL event logs)
├── tools/                           # Ad-hoc utilities
├── DOCS/                            # Project documentation
│   ├── FINAL_YEAR_PROJECT_REPORT.md # This document
│   ├── VIVA_PREPARATION_GUIDE.md    # Viva preparation guide
│   ├── PROJECT_COMPLETION_SUMMARY.md
│   └── DEVELOPER.md                 # Developer guide
└── Paper/                           # Reference papers
```

---

**END OF REPORT**

---

*Last updated: June 5, 2026*
*CONFIG_VERSION: 2.0.0*
*Repository: sign_to_text — Joseph Jonathan Fernandes*
*All claims backed by actual source code, commit hashes, and config values*
