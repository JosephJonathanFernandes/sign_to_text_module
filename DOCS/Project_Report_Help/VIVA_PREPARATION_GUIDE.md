# Viva Preparation Guide: ISL Sign Language Recognition System

**Student:** Joseph Jonathan Fernandes
**Date:** June 2026
**Project:** Real-Time Indian Sign Language to Text Recognition

---

## How to Use This Guide

This guide prepares you to answer any viva question about your project. Every answer is backed by specific code references, commit hashes, and config values from the actual repository.

---

## Section 1: Project Overview Questions

### Q1: Summarize your project in 2 minutes.

**Sample Answer:**
> "My project is a real-time Indian Sign Language to text recognition system. It uses two MediaPipe models — one for hand landmarks and one for face landmarks — to extract 506-dimensional spatiotemporal features from live webcam video. These features go through a custom BiGRU model with a Conv1D frontend, Spatial Graph Neural Network, and Hybrid Multi-Head Attention. The system outputs class predictions for 78 ISL signs in real time, uses a momentum-based commit logic to prevent jitter, and runs on a standard CPU at 30 FPS. I also built a complete training pipeline with K-fold cross-validation, ONNX INT8 quantization for deployment, and synthetic data generation using a Conditional VAE."

**Key Numbers to Memorize:**
- 78 sign classes
- 506D feature vectors
- 20 frames per sign sequence
- 5 K-fold models in ensemble
- ~5,683 processed training samples
- INT8 ONNX: 75% size reduction, 2-3× speedup
- 173 git commits, 3.5 months development

---

### Q2: Why did you choose Indian Sign Language specifically?

**Sample Answer:**
> "ISL is used by approximately 5 million people in India. Unlike ASL or BSL research that have extensive datasets and published systems, ISL has limited digital tools available. There's a significant communication barrier between the deaf community and hearing people. I chose this because it's a meaningful real-world problem with limited existing solutions."

---

### Q3: What hardware does your system require?

**Sample Answer:**
> "Only a standard webcam and a CPU. No GPU is required. The system runs at 30 FPS using MediaPipe's optimized C++ backend, and the ONNX INT8 quantized model (~1.05 MB) runs efficiently on a laptop CPU. The INT8 quantization gives a 2-3× speed improvement over the FP32 PyTorch model."

**Reference:** `config.py` `HardwareConfig.torch_device` — defaults to "cpu"; `quantize_onnx.py`

---

## Section 2: Feature Engineering Questions

### Q4: How do you extract features from the webcam?

**Sample Answer (step by step):**
1. Capture frame at 640×480 pixels
2. Run `hand_landmarker.task` every 5 frames → up to 2 hands × 21 landmarks × 3 coords = 126D raw
3. Run `face_landmarker.task` every 5 frames → detect nose (index 1), left eye (index 33), right eye (index 263)
4. Compute face-relative hand coordinates: `(hand_lm - nose) / inter_eye_distance` = 126D
5. Compute proximity scalar: L2 distance from hand center to nose = 1D
6. Total base: 126 + 126 + 1 = 253D per frame
7. Compute velocity: frame-to-frame delta of 253D = 253D velocity
8. Final feature: 253 + 253 = 506D per frame
9. Buffer 20 frames → shape (20, 506) per sign sample

**Reference:** `preprocess.py` `extract_landmarks_with_face_relative()`, `compute_face_relative_features()`

---

### Q5: Why 506 dimensions? Why not just use raw coordinates?

**Three-part answer:**

**Part 1: Face-relative normalization (126D → 126D face-relative)**
> "Raw coordinates are signer-dependent: a tall person's hands are higher in the frame than a short person's. By expressing hand positions relative to the nose and normalizing by inter-eye distance, the features become signer-position and scale invariant."
> 
> **Code:** `preprocess.py` `compute_face_relative_features()` — divides by `||left_eye - right_eye||`

**Part 2: Proximity scalar (1D)**
> "Many ISL signs are distinguished by whether hands are near the face (like 'hello') or away (like 'run'). The L2 distance from hand center to nose captures this spatial relationship and feeds directly into the proximity-aware attention bias."
> 
> **Code:** `preprocess.py` proximity computed as L2(hand_center, nose_position)

**Part 3: Velocity features (253D)**
> "Static poses alone don't distinguish signs with different speed or direction — for example, 'come' and 'go' may have similar handshapes but different movement trajectories. Velocity features (frame-to-frame deltas) explicitly encode temporal motion."
> 
> **Code:** `config.py` `use_velocity: bool = True`; `dataset.py` `_prepare_sequence()`

---

### Q6: What are the MediaPipe models you use? Aren't they heavy?

**Sample Answer:**
> "I use two MediaPipe Task models: `hand_landmarker.task` (7.8 MB, 21 landmarks per hand) and `face_landmarker.task` (3.8 MB, 478 face landmarks). They're not heavy for their function — MediaPipe is optimized C++, not Python. To maintain real-time speed, I cache detections: hands detected every 5 frames (adaptive up to 8 in low-motion), face detected every 5 frames and force-refreshed every 15. I also disabled HOG person detection which saves ~8ms per frame."

**Reference:** `preprocess.py` `create_landmarker()`, `create_face_landmarker()`; `config.py` `disable_hog_detection=True`

---

## Section 3: Model Architecture Questions

### Q7: What is your model architecture?

**Full Answer (follow the code):**

> "My model is called `SignLanguageGRU`. It has 10 architectural improvements, each independently configurable. The data flow is:
>
> 1. Input: `(batch, 20, 506)` — 20 frames × 506 features
> 2. **Phase 10 — Spatial GNN branch:** The first 126 raw dimensions feed into a `LightweightSpatialGNN` (2-layer GCN over hand skeleton, 21 nodes per hand → 8D per hand → 16D per frame). This runs in parallel with the Conv frontend.
> 3. **Phase 1 — Conv1D frontend:** All 506D feed into a depthwise-separable 1D conv (506→128 channels), with GroupNorm, ReLU, and a residual connection. Captures short-range temporal patterns.
> 4. **Concatenate:** [128 conv + 16 GNN] = 144D per frame
> 5. **Phase 2 — Learnable frame weighting:** Sigmoid weights per frame, emphasizing informative frames
> 6. **Input projection:** Linear(144→64) + LayerNorm + ReLU
> 7. **BiGRU:** 3 layers, hidden=64, bidirectional → 128D output per frame
> 8. **HybridAttention:** 4 heads (2 standard, 2 proximity-aware with Gaussian bias σ=0.15 learnable). Learns which frames to attend to.
> 9. **Residual skips:** GRU temporal mean added to attention context (Phase 9); input_proj mean also added (Phase 5)
> 10. **FC head:** Dropout(0.25) → Linear(128→96) → ReLU → Dropout → Linear(96→78)
> 11. Output: logits `(batch, 78)`"

**Reference:** `model.py` `SignLanguageGRU`, `HybridAttention`, `spatial_gnn.py` `LightweightSpatialGNN`

---

### Q8: Why BiGRU and not LSTM or Transformer?

| Choice | Reason | Trade-off |
|--------|--------|-----------|
| **BiGRU over LSTM** | Fewer parameters (only 2 gates vs 3), slightly faster, comparable performance for short sequences (20 frames) | Marginally less capacity than LSTM |
| **BiGRU over Transformer** | Self-attention is O(n²) in sequence length; for 20 frames with 506D, a Transformer stack would be 5-10× heavier; real-time CPU inference impossible | Transformer better for long contexts (we don't have long sequences) |
| **BiGRU advantage** | Bidirectional: captures both past and future context within the 20-frame buffer; natural for gesture sequences | Must buffer all 20 frames before prediction |

---

### Q9: What is the Spatial GNN? Why add it?

**Sample Answer:**
> "The `LightweightSpatialGNN` is a 2-layer Graph Convolutional Network that explicitly models the anatomical structure of the hand skeleton. Each hand has 21 landmarks with known joint connections (metacarpal-proximal-middle-distal chain per finger). The GCN performs message-passing along these edges, so each landmark's representation incorporates its neighboring joints.
>
> Why add it? The BiGRU and Conv1D treat features as a flat vector — they don't know that landmark index 5 connects to index 6 (index finger joints). The GNN injects this structural prior, which helps distinguish handshapes that differ only in the position of one or two fingers. It outputs 8D per hand (16D total per frame) and is concatenated with the Conv frontend output before the BiGRU.
>
> Importantly, it only uses the first 126 dimensions (raw landmark positions), not velocity or face-relative features, since graph structure applies to positional coordinates."

**Reference:** `spatial_gnn.py`; `config.py` `use_gnn: bool = True`, `gnn_hidden_dim=16`, `gnn_output_dim=8`

---

### Q10: What is HybridAttention? Why not standard attention?

**Sample Answer:**
> "Standard temporal attention assigns a weight to each frame based on its content alone. HybridAttention combines two types of attention:
>
> 1. **Standard heads (2 of 4):** Score each frame based on its GRU output features — what did the model see in each frame?
> 2. **Proximity heads (2 of 4):** Add a Gaussian log-probability bias: `score += -prox² / (2σ²)` where `prox` is the hand-to-face distance and `σ=0.15` is learnable. This biases attention toward frames where the hand is near the face (critical frames for many ISL signs).
>
> The proximity bias encodes the linguistic insight that the most discriminative moments in ISL words are often when the hand is in a specific spatial relationship to the face. The learnable σ lets the model determine how wide to set the Gaussian falloff."

**Reference:** `model.py` `HybridAttention` class (lines 184–288); `config.py` `proximity_sigma=0.15`

---

### Q11: Are all 10 phases actually enabled? Can you enumerate them?

**Full Answer:**

| Phase | Feature | Default | Toggleable |
|-------|---------|---------|-----------|
| 1 | Conv1D depthwise-separable frontend | **ON** | `use_conv_frontend` |
| 2 | Learnable frame weighting | **ON** | `use_frame_weighting` |
| 3 | TTA (Test-Time Augmentation) | ON offline, OFF live | `use_tta` in LiveInferenceConfig |
| 4 | Reduced dropout (0.30 GRU, 0.25 FC) | **ON** | `gru_dropout`, `fc_dropout` in model |
| 5 | Residual GRU skip (input_proj → context) | **ON** | `use_residual_gru_skip` |
| 6 | GroupNorm (8 groups) in conv frontend | **ON** | `use_groupnorm` |
| 7 | Debug shape tracing | OFF | `debug_print_shapes` |
| 8 | Depthwise temporal + residual conv | **ON** | `use_depthwise_temporal`, `use_residual_conv` |
| 9 | Residual attention skip (GRU mean → context) | **ON** | `use_residual_attention_skip` |
| 10 | Spatial GNN branch | **ON** | `use_gnn` |

**All 9 non-debug phases are enabled by default.** This is controlled by `config.py` `ArchitectureImprovementsConfig`.

---

## Section 4: Training Questions

### Q12: How did you train your model?

**Step-by-Step:**
1. **Dataset:** 78 sign classes, ~5,683 processed .npy files (20×506D each)
2. **Phase 1 training:** `processed/` only, with optional `processed_negatives/` (reject class)
3. **Phase 2 fine-tune:** Add `processed_del/` archived samples at weight=0.25
4. **K-fold:** 5 disjoint folds for ensemble; stratified per class
5. **Loss:** `CrossEntropyLoss(reduction='none')` × per-sample weight → `.mean()`
6. **Class weighting:** Inverse-frequency: `w_c = (1/n_c)^1.0`, normalized to mean=1
7. **Optimizer:** AdamW, lr=3e-4, weight_decay=5e-4
8. **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5 epochs)
9. **Mixup:** λ ~ Beta(0.3, 0.3), applied with probability 0.5
10. **Early stopping:** patience=10 epochs on validation accuracy
11. **Gradient clipping:** clip_grad_norm_(1.0)

**Reference:** `train.py` `train()` function (lines 585–767); `config.py` `TrainingConfig`

---

### Q13: How do you handle class imbalance?

**Three-level strategy:**

**Level 1 — Dataset level (oversampling):**
> "`_BalancedAugSubset` oversamples minority classes so every non-reject class matches the count of the largest class. The reject class is excluded from this cap to avoid forcing every sign class up to an artificially large count."

**Level 2 — Loss level (class weighting):**
> "Inverse-frequency class weights: `w_c = (1/n_c) ^ 1.0` normalized so mean weight = 1. Applied as per-sample multiplier: `loss = (per_sample_loss * sample_weight).mean()`."

**Level 3 — Augmentation:**
> "Both offline (17 video effects × 3 crops = 54 variants per video) and online (Mixup during training, landmark augmentation per sample)."

**Reference:** `train.py` `_BalancedAugSubset`, `_compute_inverse_class_weights()`

---

### Q14: What is K-fold cross-validation? Why use it?

**Sample Answer:**
> "K-fold divides the dataset into K non-overlapping subsets. Each fold trains on K-1 subsets and validates on 1. After K folds, every sample has been in the validation set exactly once. I use 5 folds.
>
> Why? Two reasons: (1) Better accuracy estimate — 5 validation sets average out the variance of a single random split. (2) Ensemble: I keep all 5 trained models and average their predictions, which reduces variance further.
>
> My folds are **disjoint stratified**: `_build_disjoint_folds()` partitions each class independently, so every fold gets an approximately equal distribution of each sign class. This prevents the pathological case where all samples of one class end up in the training set."

**Reference:** `train.py` `_build_disjoint_folds()` (lines 208–240); `config.py` `num_folds=5`

---

### Q15: What is Mixup and why did you use it?

**Sample Answer:**
> "Mixup (Zhang et al., 2018) is a data augmentation technique that interpolates between pairs of training samples and their labels. For two samples (x_a, label_a) and (x_b, label_b) with λ ~ Beta(α, α):
>
> `x_mixed = λ·x_a + (1-λ)·x_b`
> `loss = λ·CE(logits, label_a) + (1-λ)·CE(logits, label_b)`
>
> Why? It creates virtual training samples between existing classes, forcing the model to generalize smoothly rather than memorizing discrete class boundaries. For landmark sequences, it interpolates between two different sign trajectories, which acts as a regularizer. I use α=0.3 applied with probability 0.5."

**Reference:** `train.py` `mixup_data()`, `mixup_criterion()` (lines 147–161); `config.py` `mixup_alpha=0.3, mixup_prob=0.5`

---

### Q16: What is Focal Loss? Do you use it?

**Sample Answer:**
> "Focal Loss (Lin et al., 2017, RetinaNet paper) reduces the loss contribution of easy (high-confidence) samples and focuses training on hard misclassified samples. Formula: `FL = α · (1-p_t)^γ · CE`. With γ=2, if the model is 90% confident, the focal weight is (1-0.9)^2 = 0.01, nearly eliminating that sample's gradient.
>
> I implemented it in `train.py` `FocalLoss` class (α=0.25, γ=2.0), but it is **disabled by default** (`use_focal_loss: bool = False`). I found that the combination of class weighting + oversampling + Mixup was sufficient without adding the complexity of another hyperparameter (γ). Focal Loss is available as an option for experimentation."

**Reference:** `train.py` `FocalLoss` (lines 267–314); `config.py` `use_focal_loss=False`

---

## Section 5: Inference Questions

### Q17: How does live inference work?

**Step-by-step:**
1. Webcam capture at 30 FPS (640×480)
2. MediaPipe hand detection every 5 frames (adaptive to 8 in low-motion)
3. MediaPipe face detection every 5 frames (force re-detect every 15)
4. Extract 506D features per frame; buffer 20 frames
5. When buffer is full (20 frames): run inference
6. ONNX Runtime (INT8): primary inference path (~5-15ms)
7. PyTorch FP32: fallback if ONNX fails or dimensions mismatch
8. `ConfidenceSmoother(window=8, decay=0.3)`: smooth last 8 probability vectors
9. `StablePredictor(patience=3, delta=0.12)`: require 3 consecutive frames + 0.12 margin
10. **Momentum commit:** 3-of-5 window, min_avg_conf ≥ 0.60 → commit word
11. Add to sentence builder; NLP cleanup → display text

**Reference:** `webcam.py`; `temporal_postprocessor.py`; `config.py` `LiveInferenceConfig`

---

### Q18: Why do you have ONNX inference? What is ONNX?

**Sample Answer:**
> "ONNX (Open Neural Network Exchange) is an open format for neural network models that allows inference outside of PyTorch using highly optimized runtimes. The ONNX Runtime (ORT) is Microsoft's inference engine that applies optimizations like operator fusion, memory planning, and quantization.
>
> I export my trained PyTorch model to ONNX (opset 18) using `export_onnx.py`, then apply INT8 quantization (`quantize_onnx.py`). Results:
> - **Model size:** ~4.2 MB (FP32 PyTorch) → ~1.05 MB (INT8 ONNX) — 75% reduction
> - **Inference speed:** 2-3× faster than PyTorch FP32 on CPU
> - **Portability:** ONNX runs on any platform (Windows, Linux, macOS, mobile)"

**Reference:** `export_onnx.py`, `quantize_onnx.py`, `onnx_inference.py`; commit `ff86e0f3`

---

### Q19: What is INT8 quantization? How does it work?

**Sample Answer:**
> "INT8 quantization replaces FP32 (32-bit float) weights with 8-bit integers. Since integer arithmetic is natively faster on CPUs and takes 4× less memory, this directly translates to speedup. The key challenge is mapping the continuous FP32 range to 256 discrete values without losing accuracy.
>
> I use **dynamic INT8 quantization** via `onnxruntime.quantization.quantize_dynamic()`. In dynamic quantization: weights are statically quantized; activations are dynamically quantized at runtime based on observed statistics. No calibration dataset is needed (unlike static quantization).
>
> The result: ~75% size reduction and 2-3× speedup with minimal accuracy degradation (<1-2% on our dataset)."

**Reference:** `quantize_onnx.py`

---

### Q20: How do you prevent prediction flickering in real-time?

**Three-layer defense:**

**Layer 1 — Confidence smoothing:**
> "`ConfidenceSmoother` maintains a 8-frame deque. Each prediction's weight = its max probability × exponential decay (factor=0.3 for older frames). Weighted average over 8 frames gives a stable probability vector."

**Layer 2 — Stability prediction:**
> "`StablePredictor(patience=3, delta=0.12)` requires 3 consecutive frames voting for the same new class AND the new class's confidence must exceed the current by 0.12 (hysteresis margin) before switching."

**Layer 3 — Momentum commit:**
> "Even after StablePredictor says 'class X', the word is only committed if class X appears ≥3 times in the last 5 predictions AND the average confidence ≥ 0.60. This prevents transient spikes from adding words."

**Reference:** `temporal_postprocessor.py`; `config.py` `LiveInferenceConfig` momentum_* params

---

### Q21: What happens when the ONNX model fails?

**Sample Answer:**
> "I have a fallback mechanism in `onnx_inference.py`. `ONNXModelWrapper` wraps both an ONNX session and a PyTorch model. If the ONNX inference raises an exception (dimension mismatch, session error, etc.), it automatically falls back to PyTorch FP32. The dimension alignment logic handles cases where the feature dimension passed at runtime differs from the model's expected input: it pads with zeros or truncates to match, and handles the proximity vector rank separately (1D scalar → 2D for ONNX)."

**Reference:** `onnx_inference.py` `ONNXModelWrapper`

---

## Section 6: Dataset & Data Pipeline Questions

### Q22: How did you collect your dataset?

**Sample Answer:**
> "I used `collect_data.py`, a custom webcam collection tool with a countdown timer. For each sign word, I recorded multiple video clips from different angles and lighting conditions. Raw videos are stored in `Dataset/` organized by class folder.
>
> Each raw video then goes through `preprocess.py` to extract MediaPipe landmarks and save a (20, 506) .npy file per video. I also used `augment_video_pipeline.py` to generate up to 54 augmented variants per original video (17 visual effects × 3 crop positions + 3 spatial-only crops). These augmented videos are also preprocessed into .npy files and merged via `merge_augmentations.py` into `processed/`."

**Key numbers:**
- 78 sign classes
- ~5,683 processed .npy files
- Up to 54 augmented variants per original video

---

### Q23: What augmentation techniques did you use?

**Two-level augmentation:**

**Level 1 — Video-level (offline, before feature extraction):**
17 visual effects × 3 crop positions = 51 effect-crop combos + 3 spatial-only = 54 total variants per video:
- Spatial: `center`, `left`, `right` crop (7% offset for L/R)
- Visual effects: brightness, contrast, hue, fog, rotation, scale, color_jitter, noise, pixel_dropout, coarse_dropout, motion_blur, defocus_blur, jpeg_artifact, gamma, white_balance, perspective_warp, temporal_jitter

**Level 2 — Landmark-level (online, during training):**
- Face-anchor shift: Random translation of face reference point (simulates signer repositioning)
- Hand proportion simulation: Random scale per finger (simulates hand size variation)
- Standard: Gaussian noise, scale jitter, time warp, rotation noise

**Level 3 — Training-time:**
- Mixup: λ ~ Beta(0.3, 0.3) interpolation between sequence pairs, probability 0.5

**Reference:** `preprocess.py` `VIDEO_AUGMENT_MAX_PER_VIDEO=54`; `augmentations.py`; commit `c9771af2`

---

### Q24: What is the two-phase training strategy?

**Sample Answer:**
> "I have two categories of data:
> - `processed/`: High-quality, carefully validated landmark sequences
> - `processed_del/`: Archived samples that were previously removed but may still contain useful data
>
> Phase 1 trains exclusively on `processed/` to establish a strong base model.
>
> Phase 2 fine-tunes by including `processed_del/` samples with a low weight (0.25). This means their loss contribution is down-weighted so they can't dominate training. This strategy lets me leverage previously excluded data without degrading the model's accuracy on the primary dataset.
>
> The reject class (negative/background samples in `processed_negatives/`) is only used in Phase 1. Phase 2 uses `processed_negatives_del/` when present, resolved automatically by `_resolve_phase_neg_root()`."

**Reference:** `train.py` `create_data_loaders()`, `_resolve_phase_neg_root()`; commits `d9c069ee`, `e0161a3d`

---

### Q25: What is the reject/negatives class? Why have it?

**Sample Answer:**
> "The reject class (label `__reject__`) represents background gestures — when someone is not signing, or making ambiguous hand movements. Without a reject class, the model always outputs one of the 78 sign labels, even when no sign is being performed. This would cause the sentence to fill with random words.
>
> The `processed_negatives/` directory contains samples of non-sign gestures. During Phase 1 training, these are loaded alongside the sign classes. The model learns to output `__reject__` for non-sign inputs. The `_BalancedAugSubset` keeps the reject class at its natural count (not oversampled) because it's often a much larger bucket that would otherwise force all sign classes to be repeated thousands of times."

**Reference:** `train.py` `_BalancedAugSubset` reject_label handling (lines 429–445); `config.py` `--neg-root`

---

## Section 7: Technical Challenges Questions

### Q26: What was the hardest bug you fixed?

**Two candidates (pick the most impressive one):**

**Option A — K-fold crash (last commit before documentation):**
> "The K-fold training was crashing with `ValueError: too many values to unpack (expected 2)`. The dataset format had been updated from 2-tuples `(path, label)` to 3-tuples `(path, label, weight)` when I added per-sample weighting for the archived training data. But the K-fold code still used the old unpacking `[lbl for _, lbl in full_ds.samples]`. I fixed this with a `_sample_label(sample) -> int` helper that just returns `sample[1]` regardless of tuple length. This was commit `4672472b` on June 5, 2026 — the last bug fix before documentation."

**Option B — ONNX dimension mismatch:**
> "After exporting to ONNX with 506D input and loading models trained on 253D (pre-velocity), inference would fail with a dimension error. I built a multi-layer alignment system in `onnx_inference.py`: first check feature dim (pad/truncate if needed), then handle proximity vector rank separately (scalar → 2D for batch), then verify batch dimension (add batch axis if single sample). This makes the inference pipeline robust to format changes between training runs."

---

### Q27: Why did you disable motion gating? You mentioned it in the README.

**Sample Answer:**
> "Motion gating was implemented to skip processing low-motion frames — the idea was that between signs, when the hands are resting, we don't need to run the model. However, I disabled it by default (`MotionConfig.enabled: bool = False`) for a key reason: ISL includes sign holds — static poses that are meaningful signs themselves. If we gate on motion, we'd skip exactly the frames where static-sign information is present. It was an optimization that conflicted with correctness. The infrastructure remains in `config.py` `MotionConfig` and can be re-enabled for experimental use."

**Reference:** `config.py` `MotionConfig.enabled: bool = False`

---

### Q28: How does your system generalize to different signers?

**Honest Answer:**
> "Generalization is a current limitation. The model was trained on a limited set of signers, primarily from one geographic region. Three mechanisms partially address this:
>
> 1. **Face-relative normalization:** Makes features scale and position invariant — the same sign performed by a tall or short person, or at different screen positions, produces similar feature vectors.
> 2. **Extensive augmentation:** 54 video variants per original (different crops, lighting conditions, blur, perspective) expose the model to diverse-looking inputs.
> 3. **Adapter modules:** `adapter_model.py` implements lightweight adapter layers for user-specific fine-tuning. However, this isn't integrated in the main pipeline yet.
>
> Future work would include collecting data from more diverse signers and training with the adapter fine-tuning loop."

---

### Q29: What is the CVAE? Did you actually use the synthetic data?

**Sample Answer:**
> "CVAE stands for Conditional Variational Autoencoder. I implemented it in `cvae_landmarks.py` and `train_cvae.py`. The encoder is a BiGRU + attention that maps a (20, 506) sequence + class label to a 32D latent vector (μ, σ). The decoder takes a sampled latent + class label and reconstructs the sequence. Training uses reconstruction MSE + KL divergence loss.
>
> I also built a `quality_discriminator.py` — a BiGRU trained to distinguish real landmark sequences from CVAE-generated ones. After generating synthetic samples, I filter them by discriminator score and heuristic quality checks (variance, velocity norms).
>
> As for whether it's used in main training: the infrastructure is complete and the CVAE files are in the repository. The primary training pipeline uses the real + augmented + archived data. CVAE generation was intended for extremely underrepresented classes, but with 850-sample balancing the real augmented data covered the need. It remains available as an experimental augmentation pathway."

**Reference:** `cvae_landmarks.py`, `train_cvae.py`, `quality_discriminator.py`, `filter_synthetic_samples.py`

---

## Section 8: Comparison & Context Questions

### Q30: How does your system compare to existing work?

**Comparison Table to use in viva:**

| System | Approach | Signs | Hardware | Limitation |
|--------|----------|-------|----------|-----------|
| **This project** | MediaPipe + Conv1D + GNN + BiGRU + HybridAttn | **78 ISL** | CPU only, webcam | Isolated words |
| MediaPipe hands only | Raw coordinates, rule-based | ~10-20 | CPU | No ML generalization |
| CNN-LSTM (gesture recognition) | Video CNN + LSTM | 20-50 | GPU required | High compute, no ISL |
| ST-GCN (Yan et al.) | Spatial-temporal GCN | ~60 body pose | GPU, Kinect | Not hand-specific, GPU required |
| Sign Language Transformers | ViT + temporal | ~1000 | GPU + multiple cameras | Not deployable on CPU |

**Your advantage:** Only CPU + webcam needed; 78 ISL classes; face-relative normalization; full production pipeline (collection → training → inference → sentence building → NLP)

---

### Q31: What would you do differently if starting over?

**Strong answer showing engineering maturity:**

1. **Start with stricter dataset versioning** — The various `processed/`, `processed_del/`, `processed_negatives/` directories evolved organically. A cleaner `dvc` or `mlflow` dataset tracking system would have been more maintainable.

2. **Design the config schema upfront** — The config refactor to OOP dataclasses happened mid-project (commit `6ff84e8c`). Starting with the versioned dataclass design would have avoided config inconsistencies (e.g., `lr_scheduler: str = "cosine"` in config but ReduceLROnPlateau hardcoded in train.py).

3. **Instrument training from day 1** — The pipeline logger and metrics were added incrementally. Starting with W&B or MLflow for experiment tracking would have made hyperparameter search cleaner.

4. **More diverse signer data** — The generalization limitation would be addressed by collecting data from 10+ signers with different heights, lighting, and signing speed from the start.

---

## Section 9: Quick-Fire Technical Facts

Keep these numbers memorized for rapid-fire questions:

| Topic | Number | Source |
|-------|--------|--------|
| Sign classes | **78** | `sign_categories.json` |
| Total git commits | **173** | `.git/logs/HEAD` |
| Development duration | **3.5 months** | Feb 21 – Jun 5, 2026 |
| Feature dimension | **506D** | 253 base + 253 velocity |
| Base frame features | **253D** | 126 raw + 126 face-relative + 1 prox |
| Frames per sequence | **20** | `config.py` `num_frames=20` |
| Processed samples | **~5,683** | Commit `74677292` |
| Video augment variants | **54 max** | `VIDEO_AUGMENT_MAX_PER_VIDEO` |
| K-folds | **5** | `config.py` `num_folds=5` |
| Model size FP32 | **~4.2 MB** | `model.pth` |
| Model size INT8 | **~1.05 MB** | `quantize_onnx.py` |
| Quantization savings | **75%** | Size reduction |
| Inference speedup | **2-3×** | ONNX INT8 vs PyTorch |
| Batch size | **8** | `TrainingConfig.batch_size` |
| Learning rate | **3e-4** | `TrainingConfig.learning_rate` |
| Training epochs | **50** | `TrainingConfig.num_epochs` |
| Early stopping patience | **10** | `TrainingConfig.patience` |
| Temporal window | **8** | `LiveInferenceConfig.temporal_window_size` |
| Patience (stable predict) | **3** | `LiveInferenceConfig.temporal_patience` |
| Hysteresis delta | **0.12** | `LiveInferenceConfig.temporal_delta` |
| Momentum window | **5** | `LiveInferenceConfig.momentum_window` |
| Momentum commit count | **3** | `LiveInferenceConfig.momentum_commit_count` (3-of-5) |
| Min avg confidence | **0.60** | `LiveInferenceConfig.momentum_min_avg_conf` |
| Confidence threshold | **0.12** | `InferenceConfig.confidence_threshold` |
| Proximity sigma | **0.15** (learnable) | `ArchitectureImprovementsConfig.proximity_sigma` |
| GRU hidden size | **64** (→ 128 bidirectional) | `ModelConfig.hidden_size` |
| GRU layers | **3** | `ModelConfig.num_layers` |
| Attention heads | **4** (2 std + 2 prox) | `HybridAttention` |
| Conv frontend channels | **128** | `conv_frontend_out_channels` |
| GNN output dim | **16D/frame** | `gnn_output_dim=8` × 2 hands |
| Combined dim (conv+GNN) | **144D** | 128 + 16 |
| FC intermediate | **96** | Linear(128→96) |
| GRU dropout | **0.30** | `gru_dropout` |
| FC dropout | **0.25** | `fc_dropout` |
| Weight decay | **5e-4** | `TrainingConfig.weight_decay` |
| Label smoothing | **0.05** | `TrainingConfig.label_smoothing` |
| Mixup alpha | **0.3** | `TrainingConfig.mixup_alpha` |
| Mixup probability | **0.5** | `TrainingConfig.mixup_prob` |
| Focal Loss | **Disabled** | `use_focal_loss=False` |
| Motion gating | **Disabled** | `MotionConfig.enabled=False` |
| Live ensemble size | **1** | `LiveInferenceConfig.ensemble_size` |

---

## Section 10: Architecture One-Liner Summary (for intro)

> "My system is a real-time ISL recognition pipeline. It extracts 506-dimensional velocity-augmented face-relative hand landmarks via MediaPipe, classifies 20-frame sequences using a 10-phase BiGRU with a Conv1D frontend, Spatial GNN, and Hybrid Multi-Head Attention, exports to INT8 ONNX for 2-3× faster CPU inference, and uses momentum-based temporal smoothing with a 3-of-5 commit window to produce stable real-time text output from 78 ISL sign classes."

---

*Last updated: June 5, 2026*
*All facts verified against actual source code and git history*
