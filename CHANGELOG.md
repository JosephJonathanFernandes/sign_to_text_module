# Changelog

All notable changes to the ISL Sign-to-Text project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] — June 2026 (FYP Final Submission)

### Added
- **Spatial GNN branch** (`src/training/spatial_gnn.py`) — lightweight 2-layer GCN operating over the anatomical hand skeleton graph (21 nodes × 2 hands), producing 16 GNN-derived dimensions per frame fused with the Conv1D frontend
- **CVAE synthetic data pipeline** (`experimental/`) — Conditional Variational Autoencoder for class-balanced landmark sequence synthesis, with BiGRU encoder/decoder and velocity-consistency loss
- **Quality discriminator** — BiGRU real/fake classifier with hard-negative mining loop to filter low-quality synthetic samples before dataset inclusion
- **ONNX INT8 export and inference** (`scripts/export_onnx.py`, `scripts/quantize_onnx.py`) — 2–3× faster CPU inference, ~75% model size reduction
- **Mixed ONNX + PyTorch ensemble** (`src/inference/onnx_ensemble_integration.py`) — drop-in replacement with automatic PyTorch fallback
- **Face-relative landmark features** — 126-dimensional face-anchored coordinates normalized by inter-eye distance, position- and scale-invariant
- **Proximity-aware HybridAttention** — 4-head attention with 2 proximity-biased heads using Gaussian proximity kernel; per-head learnable temperature
- **Temporal post-processor** — `ConfidenceSmoother` (8-frame exponential decay window) + `StablePredictor` (3-frame patience + 0.12 hysteresis)
- **Momentum-based sign commit** — 3-of-5 majority window with 0.60 minimum average confidence
- **Modular `src/` package layout** — `core/`, `inference/`, `preprocessing/`, `training/`, `utils/`, `ui/`
- **Adaptive detection interval** — hand/face detection runs every 5 frames (up to 8 during low-motion), forced re-detect every 15
- **Module-level buffer cache** — pre-allocated NumPy buffers for landmark extraction, reducing per-frame allocations
- **Per-class threshold optimization** — `similar_class_penalty` (0.08) for visually confusable sign pairs
- **User-specific adapter** (`src/training/adapter_model.py`, `src/training/adapter_training.py`) — residual log-probability MLP correcting ensemble output, trained asynchronously in a background thread
- **Landmark augmentation pipeline** — 20 deterministic sequence-level augmentations (3D rotation, time warping, per-hand dropout, sensor noise, etc.)
- **Merge augmentation** — frame-splicing between same-class recordings (crossfade, hand-swap, tempo-aligned warping)
- **Video augmentation** — 54 combinations of 17 visual effects × 3 crop positions applied before landmark extraction
- **Diversity cleanup** — near-duplicate removal via L2-normalized cosine distance + Farthest Point Sampling for subset selection
- **Two-phase training** — Phase 1 on curated `processed/`, Phase 2 fine-tunes on archived `processed_del/` at reduced weight (0.25)
- **Hand sign classification** — data-driven update script auto-classifies one-hand vs two-hand signs from processed `.npy` statistics

### Changed
- Migrated from flat root layout to `src/` module hierarchy (backward-compatible root stubs preserved)
- `config.py` moved to `src/core/config.py`; all paths updated with `../../` traversal from `src/core/`
- Config paths now resolve correctly regardless of working directory

### Fixed
- Undefined `cfg` in `run_predict_word()` in `src/core/main.py`
- Double-import syntax errors in `scripts/train_kfold_resume.py` and `src/utils/pseudo_utilities.py`
- Broken path resolution for `processed/`, `assets/`, `models/` after migration to `src/core/`

---

## [1.5.0] — March 2026

### Added
- K-fold cross-validation ensemble (5 folds) with `kfold_manifest.json`
- Conv1D depthwise-separable frontend (Phase 1–3 architectural improvements)
- Learnable frame weighting (Phase 2)
- Residual GRU skip connections (Phase 5 and 9)
- GroupNorm in Conv1D frontend (Phase 6)
- NLP post-processor (`src/inference/nlp_postprocessor.py`) — grammar cleanup, punctuation normalization
- Sentence builder (`src/inference/sentence_builder.py`) — continuous sign-to-text with ambiguity delay
- Negative class training (`__reject__` via `--neg-root`)

### Changed
- Raised face-relative feature set from 126 to 253 total base dimensions (+ proximity scalar)
- Velocity features appended to produce 506-dimensional input
- Label smoothing (0.05), mixup (α=0.3, p=0.5), class weighting (power 1.0)

---

## [1.0.0] — December 2025 (Initial FYP Prototype)

### Added
- BiGRU sequence classifier for ISL word recognition (78 sign classes)
- MediaPipe Tasks API integration (HandLandmarker + FaceLandmarker)
- Raw video preprocessing pipeline
- Webcam real-time inference loop
- Webcam data collection utility (`src/preprocessing/collect_data.py`)
- Initial dataset: ~73 samples per class across 78 ISL sign classes
