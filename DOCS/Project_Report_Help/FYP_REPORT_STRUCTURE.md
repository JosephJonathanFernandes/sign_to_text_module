# FYP Report Structure — ISL Sign-to-Text

## Academic Context

**Project Title:** Real-Time Indian Sign Language Word Recognition Using BiGRU, Spatial GNN, and ONNX Runtime Inference

**Domain:** Computer Vision · Deep Learning · Human-Computer Interaction · Accessibility Technology

**Institution:** Final Year Project (FYP) Submission

---

## 1. Problem Statement

India has approximately 18 million Deaf and hard-of-hearing individuals who communicate through Indian Sign Language (ISL). Despite this large population, real-time ISL translation technology that is:
- Hardware-accessible (no depth cameras or body-worn sensors)
- Deployable on standard consumer hardware (CPU-only laptops)
- Accurate across different signers and environments

...does not currently exist as an open, production-ready system.

This project addresses this accessibility gap by delivering a complete, CPU-deployable ISL word recognition pipeline.

---

## 2. Research Objectives

1. **O1 — Accurate Isolated Word Recognition:** Achieve reliable recognition of 89 ISL words from standard RGB webcam video using only CPU inference.
2. **O2 — Real-Time Performance:** Deliver end-to-end prediction latency under 200 ms at 30 FPS on consumer hardware.
3. **O3 — Signer-Independent Generalization:** Build a feature representation (face-relative coordinates + velocity) that generalizes across different signers and camera positions.
4. **O4 — Data-Efficient Training:** Achieve competitive accuracy with limited data (~73 samples/class) through a multi-stage augmentation pipeline (video, landmark, merge) and CVAE synthetic data generation.
5. **O5 — Continuous Text Output:** Integrate temporal stability mechanisms (smoothing + momentum commit) to produce coherent continuous text from a live signing stream.

---

## 3. Literature Context

| Reference Area | Relevance |
|---|---|
| MediaPipe Hands (Zhang et al., 2020) | Real-time hand landmark detection foundation |
| Bidirectional GRU for sequence classification | Temporal modeling of gesture sequences |
| Graph Convolutional Networks (Kipf & Welling, 2016) | GCN applied over hand skeleton topology |
| Conditional VAE (Sohn et al., 2015) | Class-conditioned synthetic data generation |
| ONNX Runtime quantization | CPU inference acceleration via INT8 |
| Face-relative gesture features | Position/scale-invariant hand representation |

---

## 4. Innovation Points

What is genuinely novel in this project beyond standard academic implementations:

| Innovation | Description |
|---|---|
| **Face-relative + raw dual features** | Parallel raw coordinates (position info) and face-normalized coordinates (shape info) as complementary feature blocks — not a simple replacement |
| **Fused Spatial GNN + Conv1D** | GNN branch over anatomical hand graph fused with Conv1D temporal branch — without any external GNN library |
| **Proximity-aware HybridAttention** | Attention heads biased by a Gaussian proximity kernel over hand-to-face distance, with per-head learnable temperature |
| **CVAE + Quality Discriminator pipeline** | Full generative pipeline: BiGRU CVAE → hard-negative mining discriminator → quality-filtered dataset injection |
| **Two-phase training with archived weighting** | Phase 2 fine-tunes on previously archived samples at reduced weight (0.25) to safely recover useful historical data |
| **Asynchronous live user adapter** | Background-thread MLP correcting ensemble output in log-probability space, with rollback if adaptation degrades performance |
| **Data-driven hand classification** | Auto-classifies signs as one-hand/two-hand from dataset statistics (ratio of frames with both hands active) |

---

## 5. Methodology

### 5.1 Data Collection

- Custom webcam collection tool (`src/preprocessing/collect_data.py`)
- Controlled + uncontrolled recording environments
- Multiple recordings per sign to capture signer variation
- 89 sign classes, ~73 samples/class initial

### 5.2 Feature Engineering

- MediaPipe Tasks API (HandLandmarker + FaceLandmarker)
- 506-dimensional per-frame feature vectors
- Face-relative normalization for position/scale invariance
- Frame-to-frame velocity encoding

### 5.3 Data Augmentation

- Video-level: 54 photometric + geometric variants per video
- Landmark-level: 20 deterministic sequence-level transforms
- Merge augmentation: frame splicing between same-class recordings
- CVAE synthetic generation: class-balanced generated sequences

### 5.4 Model Architecture

- BiGRU (3 layers, hidden=64, bidirectional) + Conv1D frontend + Spatial GNN
- HybridAttention with proximity bias
- Modular phase-based design (Phases 1–10, all independently toggleable)

### 5.5 Training Strategy

- K-fold cross-validation (5 folds)
- AdamW + cosine LR scheduler
- Label smoothing (0.05), mixup (α=0.3), class weighting
- Two-phase training with archived sample fine-tuning

### 5.6 Inference Optimization

- ONNX INT8 quantization (2–3× speedup, 75% size reduction)
- Adaptive detection intervals (5–8 frames)
- Pre-allocated NumPy buffers (eliminates per-frame allocation overhead)
- Temporal post-processing: ConfidenceSmoother + StablePredictor + momentum commit

---

## 6. Dataset Statistics

| Property | Value |
|---|---|
| Total sign classes | 89 |
| Total processed sequences | ~5,683+ |
| Average per class (before augmentation) | ~73 |
| Target per class (after balancing) | 850 |
| Sequence shape | (20, 506) — float32 |
| Video resolution | 640 × 480 px |
| Frame sampling | 20 frames uniform (np.linspace) |
| Recording environments | Controlled (indoor fixed) + Uncontrolled (varied lighting, background) |

---

## 7. Evaluation Metrics

| Metric | Description |
|---|---|
| Top-1 Accuracy | Fraction of samples where highest-probability class is correct |
| Top-5 Accuracy | Fraction where correct class appears in top 5 predictions |
| Per-class Precision | Precision per sign class (confusion matrix diagonal) |
| Per-class Recall | Recall per sign class |
| K-fold average accuracy | Mean validation accuracy across 5 folds |
| Real-time latency | End-to-end ms from frame input to text output |
| FPS sustained | Average webcam loop frames per second |

> Accuracy values per fold are stored in `assets/ensemble/kfold_manifest.json` after training.

---

## 8. Experimental Setup

| Property | Value |
|---|---|
| Hardware | Consumer laptop CPU (Intel Iris Xe or equivalent) |
| GPU | None (CPU-only) |
| Framework | PyTorch 2.x + ONNX Runtime 1.16+ |
| Landmark extraction | MediaPipe Tasks API 0.10+ |
| Training duration | ~30–60 min per K-fold on CPU |
| Python | 3.10+ |

---

## 9. Limitations

| Limitation | Impact |
|---|---|
| Isolated word recognition only | Cannot recognize continuous sentence-level signing |
| Single-signer-dominant dataset | Some performance degradation on unseen signers |
| ISL dialect variation | Regional ISL variants not represented in training data |
| CPU-only deployment | Cannot achieve < 50 ms latency without GPU |
| Lighting sensitivity | MediaPipe confidence drops under very low or very bright lighting |
| Two-hand detection reliability | MediaPipe occasionally misses the second hand, causing zero-fill artifacts |

---

## 10. Future Scope

- [ ] **Sentence-level ISL recognition** — sliding window over continuous signing streams
- [ ] **Transformer sequence model** — ViT or Temporal Transformer for longer-range context
- [ ] **Mobile deployment** — TFLite export for Android/iOS
- [ ] **Multi-signer generalization** — federated learning or domain adaptation for new users
- [ ] **Text-to-speech integration** — complete accessibility loop
- [ ] **Web-based demo** — WebRTC frame capture + ONNX.js inference in browser
- [ ] **ISL sentence corpus** — expand from isolated words to phrase-level dataset

---

## 11. Infrastructure Contributions

- **Designed** a low-latency FastAPI inference architecture for real-time ISL translation.
- **Introduced** a deterministic frontend/backend feature contract to eliminate preprocessing inconsistencies between MediaPipe and PyTorch pipelines.
- **Implemented** schema validation and compatibility handshakes for robust browser integration.
- **Developed** a shared feature extraction system to ensure zero-drift preprocessing across training, inference, and frontend simulation environments.
- **Optimized** dataset storage using HDF5 with backward-compatible integration, reducing dataset initialization latency from 71.14 s to 0.18 s and reducing first-epoch execution time from 98.58 s to 18.28 s under the evaluated configuration.
- **Added** dataset fingerprinting and metadata lineage tracking to improve reproducibility and experiment consistency.
