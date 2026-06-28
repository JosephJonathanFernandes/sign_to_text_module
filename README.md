# 🤟 ISL Sign-to-Text

> **Real-time Indian Sign Language word recognition** using MediaPipe hand & face landmarks, a BiGRU + Spatial GNN deep learning classifier, and ONNX INT8 inference — running entirely on a CPU.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-FYP%20Final-blueviolet)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20ONNX-orange)
![MediaPipe](https://img.shields.io/badge/Landmarks-MediaPipe%20Tasks%20API-red)

---

## 📌 Problem Statement

Indian Sign Language (ISL) is the primary mode of communication for approximately **18 million Deaf and hard-of-hearing individuals** in India. Despite this, no consumer-grade, hardware-accessible, real-time ISL translation system currently exists. Existing solutions either depend on expensive depth cameras, require body-worn sensors, or are limited to fingerspelling alphabets.

This project delivers a **complete, CPU-deployable ISL word recognition pipeline** that translates live webcam video of ISL gestures into English text — using only a standard RGB camera.

---

## 🎯 Key Innovations

| Innovation | Description |
|---|---|
| 🦴 **Spatial GNN Branch** | 2-layer Graph Convolutional Network over the anatomical hand skeleton (21 nodes × 2 hands), fused with the Conv1D temporal frontend |
| 👁️ **Face-Relative Features** | Hand landmarks normalized by face anchor (nose tip + inter-eye distance) — position- and scale-invariant across signers |
| ⚡ **ONNX INT8 Inference** | 2–3× faster CPU inference vs PyTorch FP32; automatic PyTorch fallback on any failure |
| 🎭 **CVAE Synthetic Data** | Conditional Variational Autoencoder (BiGRU encoder/decoder) generates class-balanced synthetic training sequences |
| 🔄 **Adaptive Detection** | Hand/face landmark detection runs every 5 frames (extended to 8 during low-motion), forced re-detect every 15 — real-time at 30 FPS |
| 🧠 **Proximity Attention** | HybridAttention with 4 heads; 2 are proximity-biased using a Gaussian kernel over hand-to-face distance |
| 👤 **Live User Adapter** | Asynchronous background MLP corrects ensemble output in log-probability space for user-specific personalization |

---

## 🏗️ System Architecture

```
                         ┌─────────────────────────────────────────────────────┐
                         │              WEBCAM CAPTURE (640×480 @ 30 FPS)       │
                         └─────────────────────┬───────────────────────────────┘
                                               │
                         ┌─────────────────────▼───────────────────────────────┐
                         │     ADAPTIVE MEDIAPIPE LANDMARK DETECTION            │
                         │  HandLandmarker (every 5f) + FaceLandmarker (every 5f)│
                         └─────────────────────┬───────────────────────────────┘
                                               │
                         ┌─────────────────────▼───────────────────────────────┐
                         │          FEATURE VECTOR CONSTRUCTION (per frame)     │
                         │  Raw hand (126) + Face-relative (126) + Proximity (1)│
                         │               + Velocity delta = 506 dims            │
                         └─────────────────────┬───────────────────────────────┘
                                               │
                         ┌─────────────────────▼───────────────────────────────┐
                         │         SEQUENCE BUFFER  (20 frames × 506 dims)      │
                         └──────┬──────────────────────────┬────────────────────┘
                                │                          │
                    ┌───────────▼──────┐       ┌──────────▼──────────┐
                    │  Spatial GNN     │       │   Conv1D Frontend    │
                    │  (21 nodes/hand) │       │   (506→128 dims)     │
                    │  2-layer GCN     │       │   Depthwise Temporal │
                    │  16 dims output  │       │   + GroupNorm        │
                    └───────────┬──────┘       └──────────┬──────────┘
                                │      concat (144 dims)   │
                    ┌───────────▼──────────────────────────▼──────────┐
                    │         LEARNABLE FRAME WEIGHTING (sigmoid)      │
                    │         INPUT PROJECTION (144→64) + LayerNorm    │
                    └─────────────────────┬───────────────────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────────┐
                    │    BiGRU × 3 Layers (hidden=64, bidirectional)   │
                    │         Dropout 0.30 between layers              │
                    └─────────────────────┬───────────────────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────────┐
                    │  HybridAttention (4 heads: 2 standard + 2 proximity-aware) │
                    │         + Residual skip connections (Phase 5 & 9)│
                    └─────────────────────┬───────────────────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────────┐
                    │    FC Head: Dropout → Linear(128→96) → ReLU      │
                    │                → Linear(96→78 classes)           │
                    └─────────────────────┬───────────────────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────────┐
                    │     TEMPORAL POST-PROCESSOR                      │
                    │  ConfidenceSmoother (8-frame) + StablePredictor  │
                    │       + Momentum commit (3-of-5 majority)        │
                    └─────────────────────┬───────────────────────────┘
                                          │
                    ┌─────────────────────▼───────────────────────────┐
                    │    SENTENCE BUILDER + NLP POST-PROCESSOR         │
                    │  Ambiguity delay → Grammar cleanup → Output text │
                    └─────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Landmark extraction | MediaPipe Tasks API (`HandLandmarker`, `FaceLandmarker`) |
| Deep learning | PyTorch 2.x |
| Accelerated inference | ONNX Runtime (INT8 quantized) |
| Computer vision | OpenCV 4.x |
| Graph neural network | Custom GCN (PyTorch, no external GNN library) |
| Generative model | CVAE (Conditional VAE — BiGRU encoder/decoder) |
| Dataset format | NumPy `.npy` sequences, `(20, 506)` per sample |
| Configuration | Validated Python dataclasses (`config.py`) |
| Language | Python 3.10+ |
| OS | Windows / Linux / macOS |

---

## ✋ Sign Classes (78 Signs)

**Pronouns (8):** I · he · she · it · we · you · you_all · they

**Adjectives (21):** beautiful · ugly · loud · quiet · happy · sad · deaf · blind · nice · rich · poor · thick · thin · expensive · cheap · flat · curved · male · female · tight · loose

**Descriptors (21):** big_large · small_little · fast · slow · heavy · light · tall · short · long · narrow · wide · deep · shallow · hot · cold · warm · clean · dirty · dry · wet · soft · hard · strong · weak · old · new · young · famous · healthy · sick

**States (4):** dead · alive · high · low

**Greetings & Phrases (7):** Hello · How_are_you · Alright · Good_Morning · Morning · Good_afternoon · Good_evening · Good_night

**Social (6):** Thank_you · Pleased · bad · mean · cool · Idle

**Numeric (3):** 0 · 1 · 2

---

## 📊 Results

| Metric | Value |
|---|---|
| Sign classes | 78 ISL words |
| Dataset size | ~5,683 processed sequences |
| Training augmentation | Up to 54 video variants + 20 landmark augmentations per sample |
| Model parameters | ~180K (BiGRU + GNN) |
| ONNX INT8 model size | ~1.05 MB |
| End-to-end latency | < 200 ms per prediction |
| Target inference rate | 30 FPS webcam |
| K-fold validation folds | 5 |

> **Note:** Per-fold and per-class accuracy metrics are stored in `assets/ensemble/kfold_manifest.json` after training.

---

## 📁 Project Structure

```
sign_to_text/
│
├── main.py                      ← CLI entry point (shim → src/core/main.py)
├── config.py                    ← Config shim (backward compat)
├── model.py                     ← Model shim (backward compat)
├── train.py                     ← Train shim (backward compat)
├── webcam.py                    ← Webcam shim (backward compat)
│
├── requirements.txt             ← Python dependencies
├── pyproject.toml               ← Package metadata
├── LICENSE                      ← MIT License
├── CHANGELOG.md                 ← Version history
├── CONTRIBUTING.md              ← Contribution guide
│
├── src/                         ← All core source code
│   ├── core/
│   │   ├── config.py            ← Master configuration (validated dataclasses)
│   │   ├── main.py              ← Pipeline orchestration and CLI argument parsing
│   │   ├── webcam.py            ← Real-time webcam inference loop
│   │   ├── landmark_processor.py← Landmark math utilities
│   │   ├── motion_tracker.py    ← Frame-to-frame motion estimation
│   │   ├── camera_manager.py    ← Camera initialization
│   │   └── inference_engine.py  ← Inference session wrapper
│   │
│   ├── training/
│   │   ├── model.py             ← SignLanguageGRU (BiGRU + Conv + GNN + Attention)
│   │   ├── spatial_gnn.py       ← Lightweight Spatial GCN over hand skeleton graph
│   │   ├── train.py             ← Training loop, K-fold CV, loss functions
│   │   ├── adapter_model.py     ← Residual log-prob adapter MLP
│   │   └── adapter_training.py  ← Async background adapter training manager
│   │
│   ├── inference/
│   │   ├── ensemble.py          ← Ensemble loading and test-time augmentation
│   │   ├── onnx_inference.py    ← ONNX Runtime wrapper with PyTorch fallback
│   │   ├── onnx_ensemble.py     ← Mixed ONNX + PyTorch ensemble
│   │   ├── onnx_ensemble_integration.py ← Drop-in ensemble replacement
│   │   ├── temporal_postprocessor.py    ← ConfidenceSmoother + StablePredictor
│   │   ├── sentence_builder.py  ← Continuous sign-to-text assembly
│   │   ├── nlp_postprocessor.py ← Rule-based grammar & punctuation cleanup
│   │   ├── hand_selector.py     ← Multi-signer hand assignment logic
│   │   └── pseudo_buffer.py     ← Pseudo-label buffering for adapter
│   │
│   ├── preprocessing/
│   │   ├── preprocess.py        ← MediaPipe extraction → .npy generation
│   │   ├── dataset.py           ← ISLDataset (PyTorch Dataset + augmentation)
│   │   ├── augmentations.py     ← 20 deterministic landmark augmentations
│   │   ├── merge_augmentations.py ← Frame-splicing merge augmentation
│   │   ├── collect_data.py      ← Webcam training data collection tool
│   │   └── cleanup_dataset_npy.py ← Near-duplicate removal + FPS diversity
│   │
│   ├── utils/
│   │   ├── pipeline_logger.py   ← Structured event logging
│   │   ├── profiling.py         ← Lightweight latency profiler
│   │   ├── quantization_utils.py← Checkpoint quantization helpers
│   │   └── pseudo_utilities.py  ← Pseudo-label generation utilities
│   │
│   └── ui/
│       └── renderer.py          ← OpenCV overlay rendering
│
├── scripts/                     ← Standalone utility scripts
│   ├── export_onnx.py           ← Export PyTorch → ONNX (opset 18)
│   ├── quantize_onnx.py         ← FP32 → INT8 quantization
│   ├── evaluate_quantized_model.py ← Quantized model evaluation
│   ├── train_kfold_resume.py    ← K-fold training orchestration
│   ├── augment_pipeline.py      ← Landmark augmentation pipeline
│   ├── augment_video_pipeline.py← Video-level augmentation
│   ├── balance_processed_dataset.py ← Dataset balancing to target count
│   ├── random_downsample_processed.py ← Safe class downsampling
│   ├── quality_filter_hybrid.py ← Hybrid quality + diversity filter
│   ├── debug_model.py           ← Model debug and shape trace
│   ├── quantize_model.py        ← PyTorch model quantization
│   └── update_hand_classification.py ← Auto-update hand count JSON
│
├── experimental/                ← Research experiments (CVAE synthetic pipeline)
│   ├── cvae_landmarks.py        ← Conditional VAE model definition
│   ├── train_cvae.py            ← CVAE trainer
│   ├── generate_cvae_samples.py ← Synthetic sequence generation
│   ├── quality_discriminator.py ← BiGRU realism classifier
│   ├── train_quality_discriminator.py ← Discriminator trainer
│   ├── filter_synthetic_samples.py ← Quality-threshold filtering
│   ├── visualize_latent_space.py ← PCA/t-SNE latent space visualization
│   └── visualize_quality_scores.py  ← Score histogram visualization
│
├── tools/                       ← Developer and analysis tools
│   ├── validate_npy.py          ← .npy integrity checker
│   ├── build_weighted_filelist.py ← Filelist emitter for dataloaders
│   ├── grid_search_archived.py  ← Archived sample weight grid search
│   ├── generate_mermaid.py      ← Dependency graph generator
│   └── debug_onnx_input_check.py ← ONNX input dimension debugger
│
├── data/                        ← JSON configuration files
│   ├── hand_sign_classification.json ← One-hand vs two-hand classification
│   ├── similar_signs.json       ← Confusable sign pairs (threshold penalty)
│   ├── sign_categories.json     ← High-level sign category groups
│   └── dep_graph.json           ← Module dependency graph
│
├── docs/                        ← Technical documentation
│   ├── system_design.md         ← Full system architecture
│   ├── model_architecture.md    ← GRU + GNN + ONNX design details
│   ├── training_pipeline.md     ← End-to-end training guide
│   ├── inference_pipeline.md    ← Live inference walkthrough
│   ├── dataset.md               ← Dataset structure and statistics
│   └── FYP_REPORT_STRUCTURE.md  ← Academic summary for FYP evaluators
│
├── assets/                      ← Data and model artifacts (gitignored)
│   ├── Dataset/                 ← Raw video files organized by class
│   ├── processed/               ← Preprocessed .npy sequences (20×506)
│   ├── augmented_dataset/       ← Augmented videos (before preprocessing)
│   └── ensemble/                ← K-fold model checkpoints
│
├── models/                      ← Model files (gitignored)
│   ├── hand_landmarker.task     ← MediaPipe hand landmarker (7.8 MB)
│   ├── face_landmarker.task     ← MediaPipe face landmarker (3.8 MB)
│   ├── model.pth                ← Trained single-model checkpoint
│   └── model_fp32.onnx / *_int8.onnx ← ONNX models
│
└── logs/                        ← Runtime logs (gitignored)
```

---

## ⚙️ Setup

### Prerequisites

- Python **3.10+**
- A standard USB webcam
- **No GPU required** — fully CPU-optimized

### Installation

```bash
# Clone the repository
git clone https://github.com/JosephJonathanFernandes/sign_to_text_module.git
cd sign_to_text_module

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux / macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Model Files

Place the following MediaPipe task files in the `models/` directory:
- [`hand_landmarker.task`](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
- [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

---

## 🚀 Usage

### Live Webcam Recognition

```bash
python main.py --webcam
```

### Train a Single Model

```bash
python main.py --train
```

### K-Fold Ensemble Training (recommended)

```bash
python main.py --kfold
```

### Preprocess Raw Videos

```bash
python main.py --preprocess
```

### Predict from a Video File

```bash
python main.py --predict path/to/video.mp4
```

### Export to ONNX + Quantize to INT8

```bash
python scripts/export_onnx.py --checkpoint models/model.pth --output models/model_fp32.onnx
python scripts/quantize_onnx.py --input models/model_fp32.onnx --output models/model_int8.onnx
```

### Collect New Training Samples

```bash
python main.py --collect --cls hello --n 50
```

---

## 🔬 Training Pipeline

The full training pipeline runs in this order:

```
1. Record videos             python main.py --collect --cls <sign> --n 50
2. Augment videos            python main.py --augment-videos
3. Preprocess landmarks      python main.py --preprocess
4. Augment landmarks         python main.py --augment-landmarks
5. Merge augmentations       python main.py --merge
6. Cleanup near-duplicates   python main.py --cleanup
7. Train K-fold ensemble     python main.py --kfold
8. Export ONNX               python scripts/export_onnx.py
9. Quantize INT8             python scripts/quantize_onnx.py
10. Run inference            python main.py --webcam
```

---

## 🧪 Developer Utilities

See [`DOCS/DEVELOPER.md`](DOCS/DEVELOPER.md) for:
- Checkpoint naming conventions
- K-fold fine-tuning commands
- Dataset quality control scripts
- Profiling and debug utilities
- Pseudo-label and adapter workflows

---

## 📖 Documentation

| Document | Description |
|---|---|
| [`docs/system_design.md`](docs/system_design.md) | Full system architecture and data flow |
| [`docs/model_architecture.md`](docs/model_architecture.md) | BiGRU + GNN + Attention model details |
| [`docs/training_pipeline.md`](docs/training_pipeline.md) | End-to-end training walkthrough |
| [`docs/inference_pipeline.md`](docs/inference_pipeline.md) | Live inference stages and latency budget |
| [`docs/dataset.md`](docs/dataset.md) | Dataset structure, statistics, and augmentation |
| [`docs/FYP_REPORT_STRUCTURE.md`](docs/FYP_REPORT_STRUCTURE.md) | Academic evaluation structure |
| [`CHANGELOG.md`](CHANGELOG.md) | Full version history |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guide |
| [`DOCS/DEVELOPER.md`](DOCS/DEVELOPER.md) | Developer commands and utilities |

---

## 🔮 Future Work

- [ ] Continuous (sentence-level) ISL recognition beyond isolated words
- [ ] Transformer-based sequence model for richer temporal context
- [ ] Mobile deployment (TFLite / CoreML export)
- [ ] Multi-signer domain adaptation via federated learning
- [ ] Integration with text-to-speech for complete accessibility loop
- [ ] Web-based demo with WebRTC capture

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) — Hand and face landmark detection
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [ONNX Runtime](https://onnxruntime.ai/) — Accelerated CPU inference
- The Deaf community and ISL signers who participated in data collection
