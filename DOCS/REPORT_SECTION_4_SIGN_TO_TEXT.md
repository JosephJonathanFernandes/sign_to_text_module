# 4. SIGN TO TEXT MODULE

## 4.1 Module Overview

The Sign-to-Text module constitutes the core computational subsystem of the AI-Powered Indian Sign Language Recognition System. Its primary purpose is to translate isolated Indian Sign Language (ISL) word gestures, captured via a standard RGB webcam, into corresponding English text strings in real time. Unlike conventional sign language recognition approaches that rely on depth cameras or body-worn sensors, this module operates entirely on two-dimensional colour video frames captured from a consumer-grade webcam, making it hardware-accessible and practically deployable.

Within the complete system, the Sign-to-Text module functions as the primary perception and classification layer. It receives raw video frames from the webcam subsystem, performs multi-stage landmark extraction using the MediaPipe Tasks API, constructs 506-dimensional spatiotemporal feature vectors from extracted landmarks, and passes sequences of twenty such frames through a trained Bidirectional Gated Recurrent Unit (BiGRU) deep learning classifier. The output of the module — a predicted sign label and an associated confidence score — is consumed by the Sentence Builder and Natural Language Processing (NLP) post-processor to produce grammatically cleaned text output.

**Module Inputs:**
- Live RGB video frames at 640 × 480 pixels, 30 frames per second, from a USB webcam.
- For offline training and preprocessing: recorded video files in `.mp4`, `.mov`, `.avi`, or `.mkv` format.

**Module Outputs:**
- Recognised ISL word label (one of 78 sign classes).
- Scalar confidence score in the range [0, 1].
- Accumulated sentence string passed to the NLP post-processing layer.

**Overall Workflow:**
The module executes the following pipeline in sequence during live inference: (1) webcam frame capture at 30 FPS; (2) interleaved MediaPipe hand and face landmark detection with caching; (3) construction of 506-dimensional per-frame feature vectors; (4) buffering of 20 consecutive frames; (5) ONNX Runtime INT8 inference with a PyTorch fallback; (6) confidence smoothing and temporal stability filtering; (7) momentum-based sign commit logic; and (8) text generation via the Sentence Builder module.

---

## 4.2 System Architecture

The Sign-to-Text module is architected as a sequential, multi-stage pipeline. Each stage transforms its input into a more abstract representation until a final textual output is produced. A complete description of each stage follows.

### Stage 1 — Webcam Capture

The OpenCV `VideoCapture` interface captures frames at 640 × 480 pixels from the system webcam at approximately 30 frames per second. To maintain consistency between training data and live inference, all frames are centre-cropped to the webcam target resolution. This ensures that the spatial coordinate space experienced by MediaPipe during inference exactly matches that during preprocessing. The HOG-based person detection layer is explicitly disabled (via `disable_hog_detection: bool = True` in `config.py`) to save approximately 8 milliseconds per frame.

### Stage 2 — Adaptive MediaPipe Landmark Detection

To achieve real-time throughput without running MediaPipe on every frame, an adaptive detection interval mechanism is employed. Hand landmarks are detected every 5 frames by default, with cached results reused between detections. A forced re-detection is triggered every 15 frames regardless of motion state, preventing stale landmark tracking from persisting. When motion magnitude is below 60% of the configured threshold, the hand detection interval is extended up to a maximum of 8 frames, further reducing computational load during static sign holds. Face landmarks are similarly detected at a 5-frame interval, with the results cached between frames.

### Stage 3 — Feature Vector Construction

For each processed frame, a 253-dimensional base feature vector is constructed: 126 dimensions of raw hand landmark coordinates (both hands, 21 landmarks × 3 coordinates × 2 hands), 126 dimensions of face-relative hand coordinates (same structure, normalised relative to the face anchor), and 1 scalar proximity dimension encoding the L2 distance from the hand centroid to the nose tip. Frame-to-frame velocity features (253 dimensions) are appended, yielding a final per-frame vector of **506 dimensions**.

### Stage 4 — Sequence Buffering

Twenty consecutive feature vectors (20 frames) are accumulated in a fixed-length circular buffer, producing an input tensor of shape `(20, 506)`. This 20-frame window corresponds to approximately 667 milliseconds at 30 FPS, capturing the full temporal extent of most ISL word gestures.

### Stage 5 — Deep Learning Inference

The buffered sequence is passed to the `SignLanguageGRU` model. Primary inference uses the ONNX Runtime (ORT) with INT8 quantisation, providing 2–3× faster inference than the PyTorch FP32 path. If the ONNX session raises a dimension or runtime error, the system falls back to PyTorch FP32 inference automatically. The model outputs raw logits over 78 classes, which are converted to softmax probabilities.

### Stage 6 — Temporal Post-Processing

The raw per-frame probability vector is processed by the `TemporalPostProcessor`, which combines a `ConfidenceSmoother` (sliding window of 8 frames with confidence weighting and exponential decay factor of 0.3) and a `StablePredictor` (requires 3 consecutive frames voting for the same class with a minimum confidence margin of 0.12 before switching). This two-layer mechanism significantly reduces prediction jitter without introducing excessive latency.

### Stage 7 — Momentum-Based Commit

A sign word is committed to the output sentence only when the predicted class appears at least 3 times in the most recent 5 predictions (a "3-of-5 majority window") and the average confidence across those occurrences equals or exceeds 0.60. This prevents transient or low-confidence predictions from being mistakenly appended to the output sentence.

### Stage 8 — Text Generation

Committed sign labels are passed to the `SentenceBuilder`, which applies an ambiguity delay of 4 additional frames when the margin between the top-1 and top-2 predicted class probabilities is less than 0.05. The assembled sentence is subsequently cleaned by `nlp_postprocessor.py` for grammar and punctuation normalisation.

---

## 4.3 Data Acquisition and Dataset Preparation

### 4.3.1 Sign Collection Process

A custom webcam data collection tool, `collect_data.py`, was developed to standardise the recording process across all 78 sign classes. The tool provides a countdown of 3 seconds before each recording begins, allowing the signer to prepare. For each sign class, 90 raw frames are captured using the OpenCV `VideoCapture` interface at 640 × 480 pixels. These 90 raw frames are subsequently sub-sampled to 20 evenly spaced frames during preprocessing, ensuring temporal consistency across all recordings.

Recordings were conducted in both controlled and uncontrolled environments. Controlled recordings used a fixed-distance position from the camera under consistent indoor lighting. Uncontrolled recordings deliberately introduced variation in lighting temperature (fluorescent, incandescent, and natural daylight), background complexity (plain walls, cluttered rooms), and signer-to-camera distance. This diversity was intentional: a model trained exclusively on controlled data generalises poorly to real-world conditions where users' environments vary significantly.

Multiple recordings per class were collected to increase sample diversity. The dataset reached approximately 5,683 processed `.npy` sequences across 78 sign classes, as evidenced by the commit `74677292` titled "Add processed landmark sequences (5,683 .npy files)".

### 4.3.2 Landmark Extraction

Landmark extraction is implemented in `preprocess.py` using the **MediaPipe Tasks API** — specifically the `HandLandmarker` and `FaceLandmarker` task models. The Tasks API was chosen over the legacy MediaPipe Solutions API for three reasons: improved accuracy on partial occlusion cases, better forward compatibility with future MediaPipe releases, and explicit separation of image-mode and video-mode inference that aligns with the training versus live inference distinction in this system.

**Hand Landmark Extraction:** The `hand_landmarker.task` model (7.8 MB) detects up to 2 hands per frame, extracting 21 landmarks per hand in normalised (x, y, z) coordinates. A minimum hand detection confidence of 0.3 is used during preprocessing, raised to 0.5 during live webcam inference to reduce false positive detections.

**Face Landmark Extraction:** The `face_landmarker.task` model (3.8 MB) extracts 478 facial landmarks. From these, only three indices are used: nose tip (index 1), left eye outer corner (index 33), and right eye outer corner (index 263). These three points are sufficient to define a face anchor (the nose tip as origin) and a spatial scale factor (the inter-eye Euclidean distance), enabling position- and scale-invariant landmark normalisation.

**Intentional Exclusion of Pose Landmarks:** Full body pose landmarks (MediaPipe Pose, producing 33 body landmarks) were deliberately excluded from the feature set. ISL word-level recognition depends on hand configuration and hand position relative to the face — shoulder and trunk landmarks contribute minimal discriminative information for isolated word recognition while adding 99 dimensions of noise (33 × 3 coordinates). Their exclusion reduces the feature dimension, improves computational efficiency, and decreases the risk of overfitting to spurious body-pose correlations.

The extracted feature set therefore comprises:
- **Raw hand landmarks:** 21 landmarks × 3 coords × 2 hands = **126 dimensions**
- **Face-relative hand landmarks:** 21 landmarks × 3 coords × 2 hands = **126 dimensions**
- **Hand-to-face proximity scalar:** **1 dimension**
- **Total base features:** **253 dimensions per frame**
- **With velocity (frame-to-frame delta):** **506 dimensions per frame**

### 4.3.3 Data Preprocessing

Preprocessing is implemented in `preprocess.py` and `dataset.py` and encompasses the following steps:

**Missing Landmark Handling:** When a hand is not detected in a given frame — due to partial occlusion, motion blur, or detection failure — the corresponding 63-dimensional raw block and 63-dimensional face-relative block are filled with zeros. This zero-filling strategy was chosen over interpolation because the absence of a hand is itself a meaningful signal (it indicates the hand is not in the field of view or is not performing a gesture). The model learns to treat zero-filled frames accordingly.

**Face Anchor Absence:** When the face landmarker fails to detect a face (e.g., profile views or extreme lighting), `_extract_face_anchor()` returns `None` and the face-relative feature blocks are zero-filled. A global `_FACE_WARNING_SHOWN` flag prevents repeated console warnings from degrading real-time performance.

**Frame Count Standardisation:** All video clips are sampled to exactly 20 frames via uniform index interpolation (`np.linspace(0, total_frames - 1, 20)`), regardless of the original video length or frame rate. This ensures temporal consistency across the dataset.

**Input Dimension Alignment:** The `_align_input_size()` method in `ISLDataset` pads with zeros or truncates the feature dimension to exactly match the current `INPUT_SIZE` (506). This ensures backward compatibility when loading `.npy` files generated by earlier pipeline versions with different feature dimensions.

**Velocity Recomputation After Augmentation:** Following any augmentation that modifies landmark coordinates (noise injection, rotation, time warping), the proximity scalar and its velocity component are recomputed via `_recompute_proximity()` to maintain feature coherence. This prevents a situation where augmented coordinate features are inconsistent with stale proximity values.

**Data Consistency Checks:** During dataset loading, each `.npy` file is validated by attempting to load and checking that its size is non-zero. Corrupt files are reported and skipped. Up to 3 retries are performed before raising an informative error that includes the file path and suggested remediation steps.

### 4.3.4 Dataset Organisation

The training dataset is organised as a file-system hierarchy under the `processed/` directory, with one subdirectory per sign class named after the sign label (e.g., `processed/hello/`, `processed/thank_you/`). Each subdirectory contains `.npy` files, where each file stores one processed sequence as a NumPy array of shape `(20, 506)`.

The `ISLDataset` class (`dataset.py`) loads these `.npy` files at training time, applies optional on-the-fly augmentation, and returns tuples of `(sequence, proximity, label, sample_weight)`. The `balance_processed_dataset.py` script ensures that each class contains at least 850 samples prior to training, downsampling over-represented classes and oversampling under-represented ones. An archival directory `processed_del/` stores previously removed samples that may be reintroduced during Phase 2 fine-tuning at a reduced sample weight of 0.25. A `processed_negatives/` directory stores background/non-sign sequences that are used to train a reject class (`__reject__`), preventing the model from always outputting a sign label even when no signing is occurring.

---

## 4.4 Feature Engineering

### 4.4.1 Landmark-Based Features

The 126-dimensional raw hand feature block concatenates the 21 MediaPipe hand landmarks of both the left and right hands, each expressed as a normalised (x, y, z) triplet in the range [0, 1] relative to the frame dimensions. These raw coordinates preserve absolute hand position information within the frame. The choice to retain both hands — rather than only the dominant hand — enables the model to distinguish signs that involve different relative positions or configurations of both hands simultaneously.

The 126-dimensional face-relative feature block expresses the same 21 landmarks per hand in face-anchored coordinates. Specifically, each landmark coordinate is transformed as:

```
relative_coord[i] = (hand_lm[i] - nose_tip) / inter_eye_distance
```

where `nose_tip` is the (x, y, z) coordinate of face landmark index 1 and `inter_eye_distance` is the Euclidean distance between face landmarks index 33 (left eye) and index 263 (right eye). The nose tip serves as the origin, and the inter-eye distance normalises for the physical scale of the signer's face in the frame.

### 4.4.2 Relative Feature Generation

The decision to include face-relative coordinates as a distinct feature block (rather than replacing raw coordinates) was motivated by their complementary nature. Raw coordinates encode where in the frame the hands are located — useful when combined with face landmarks to infer spatial relationships. Face-relative coordinates encode where the hands are with respect to the signer's face, which is the primary spatial cue in ISL: most ISL signs are defined by the position and configuration of the hands relative to specific facial regions (near the mouth, at the forehead, at the chin, extended in front of the face, etc.).

The face-relative representation confers three practical benefits. First, it is invariant to the absolute position of the signer within the camera frame, so the same sign performed by a signer sitting close to or far from the camera produces similar face-relative values. Second, it is invariant to the signer's stature, since the inter-eye distance scales proportionally with the face's apparent size in the frame. Third, it enables the model's attention mechanism to apply a physically meaningful spatial bias: the Gaussian proximity kernel in the HybridAttention module uses the proximity scalar — which is derived from face-relative distances — to upweight frames where the hands are near the face, which are typically the most informative frames for discriminating ISL signs.

### 4.4.3 Temporal Representation

A single frame's landmark configuration is insufficient to distinguish many ISL signs. Motion trajectory, speed, and directional change are all critical discriminators. The temporal representation is addressed at two levels.

**Multi-Frame Sequence:** Twenty consecutive frames are buffered and processed as a unit. The 20-frame window was selected based on empirical observation of sign durations in the collected dataset: the majority of ISL words are completed within approximately 0.5 to 0.8 seconds, corresponding to 15 to 24 frames at 30 FPS. A 20-frame window (667 ms) captures the complete gesture while remaining short enough to keep sequence processing computationally tractable.

**Velocity Features:** Frame-to-frame finite differences of all 253 base features are appended to form the 506-dimensional input. The velocity block at frame $t$ is computed as $v_t = f_t - f_{t-1}$, where $f_t$ denotes the base feature vector at time $t$. At frame 0 (the first frame), the velocity block is set to zero. Velocity features explicitly encode motion direction and speed, enabling the model to distinguish signs that share similar peak-frame handshapes but differ in their approach trajectory.

---

## 4.5 Deep Learning Model Development

### 4.5.1 Model Selection

The `SignLanguageGRU` architecture — a multi-phase Bidirectional Gated Recurrent Unit with convolutional and graph neural network frontends — was selected based on three constraints specific to this project:

1. **CPU-only deployment requirement:** The system must operate on a standard laptop or desktop CPU without requiring a GPU. Transformer-based architectures, despite superior accuracy on large datasets, incur $O(n^2)$ self-attention complexity over the sequence length and require substantially more memory bandwidth, making them unsuitable for real-time CPU inference. The GRU's $O(n)$ sequential computation with moderate hidden dimensions is far more tractable.

2. **Short sequence length (20 frames):** The bidirectional GRU is well-suited to sequences of this length. The full 20-frame context is available at inference time, so bidirectionality (reading the sequence in both forward and backward directions) is not computationally prohibitive.

3. **Limited training data:** With approximately 5,683 samples across 78 classes (averaging ~73 samples per class before augmentation), large-parameter Transformer models would overfit severely. The GRU's parameter efficiency — especially when combined with the Conv1D frontend and Spatial GNN — provides sufficient representational capacity without overfitting.

The LSTM architecture was also evaluated. The GRU was preferred because it employs two gating mechanisms (update and reset gates) rather than three (input, forget, output), yielding fewer parameters with comparable performance on short-sequence gesture classification tasks.

### 4.5.2 Network Architecture

The `SignLanguageGRU` model implements 10 independently configurable architectural improvements, all enabled by default in production. The data flow is as follows:

**Input:** Tensor of shape `(batch, 20, 506)` — batch × 20 frames × 506 features.

**Phase 10 — Spatial GNN Branch (`spatial_gnn.py`):**
The first 126 dimensions (raw hand landmark coordinates for both hands) are passed through a `LightweightSpatialGNN`, a 2-layer Graph Convolutional Network operating over the anatomical hand skeleton graph (21 nodes per hand, with edges corresponding to known metacarpal-proximal-middle-distal finger joint connections). The GCN produces 8-dimensional pooled representations per hand (global max-pooling over 21 nodes), concatenated across both hands to yield **16 dimensions per frame**. This GNN branch runs in parallel with the Conv1D frontend.

**Phase 1 — Conv1D Frontend:**
All 506 dimensions are passed through a depthwise-separable 1D convolutional frontend: a pointwise convolution reducing 506 channels to 128, followed by a depthwise temporal convolution (kernel size 3, padding 1, grouped by channel) with a residual connection, and a GroupNorm (8 groups) followed by ReLU and dropout (0.1). This frontend extracts short-range temporal patterns across the 20-frame sequence while reducing input dimensionality. The output is of shape `(batch, 20, 128)`.

**Concatenation:** The 16-dimensional GNN output per frame is concatenated with the 128-dimensional Conv1D output to produce **144 dimensions per frame**.

**Phase 2 — Learnable Frame Weighting:**
A small MLP (`Linear(144→32)→ReLU→Linear→Sigmoid`) produces a scalar importance weight per frame, applied as an element-wise multiplicative mask. This allows the model to soft-suppress uninformative frames (e.g., transition frames between signs) while amplifying informative frames (sign onset and peak).

**Input Projection:** A `Linear(144→64)` layer followed by LayerNorm(64) and ReLU projects the combined features into the GRU input space.

**Phase 4 — Bidirectional GRU:**
Three stacked bidirectional GRU layers with hidden dimension 64 per direction (128-dimensional concatenated output). Inter-layer dropout is 0.30 (reduced from 0.35 as part of Phase 4 refinements). The output is of shape `(batch, 20, 128)`, followed by a LayerNorm.

**HybridAttention (4 heads):**
Two of the four attention heads are standard temporal attention heads (learning which frames carry the most information). The remaining two heads are proximity-aware: their attention scores are additively biased by the Gaussian proximity log-probability $\log \mathcal{N}(\text{prox}; 0, \sigma^2) = -\text{prox}^2 / (2\sigma^2)$, where $\sigma = 0.15$ is a learnable parameter. Each head also has an independent learnable temperature clamped to $[0.1, 10.0]$, controlling the sharpness of its softmax distribution. The four head outputs (each 32-dimensional) are concatenated into the 128-dimensional context vector.

**Residual Skips (Phases 5 and 9):** The temporal mean of the GRU output is added to the attention context (Phase 9 residual). Additionally, the temporal mean of the input projection is added to the context if dimensions align (Phase 5 residual). These skip connections improve gradient flow and training convergence.

**FC Classification Head:** `Dropout(0.25) → Linear(128→96) → ReLU → Dropout → Linear(96→78)` producing raw logits over 78 classes.

### 4.5.3 Training Strategy

The model is trained using the `train.py` module. The training configuration is centralised in `config.py` as a validated dataclass (`TrainingConfig`), with `CONFIG_VERSION = "2.0.0"`.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Batch size | 8 | Small batches suited to limited per-class sample counts |
| Learning rate | 3 × 10⁻⁴ | Reduced from 5 × 10⁻⁴ for improved stability with small datasets |
| Weight decay | 5 × 10⁻⁴ | L2 regularisation to prevent overfitting |
| Gradient clipping | 1.0 | Prevents gradient explosion in deep recurrent networks |
| Epochs | 50 | Sufficient convergence for 78-class problem |
| Early stopping patience | 10 | Terminates training if validation accuracy does not improve for 10 epochs |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 5) | Halves LR when validation accuracy plateaus |
| Validation split | 70 / 30 (stratified) | Disjoint per-class splits via `_disjoint_stratified_split()` |
| Loss function | CrossEntropyLoss (per-sample, reduction='none') × per-sample weight | Enables differential weighting of archived vs. primary samples |
| Label smoothing | 0.05 | Prevents over-confident predictions on ambiguous classes |
| Class weighting | Inverse frequency, power 1.0, normalised to mean = 1 | Compensates for residual class imbalance after oversampling |
| Mixup augmentation | α = 0.3, applied with probability 0.5 | Creates virtual training samples between classes; improves generalisation |
| K-fold cross-validation | 5 folds, disjoint stratified | Full ensemble of 5 models for improved accuracy |

**Two-Phase Training:** Phase 1 trains exclusively on curated data from `processed/`. Phase 2 fine-tunes by adding samples from `processed_del/` (previously archived data) at a reduced sample weight of 0.25, preventing lower-quality archived samples from dominating gradient updates.

**Model Checkpointing:** The best-performing checkpoint per fold (highest validation accuracy) is saved to `model.pth` (single model) or `ensemble/fold_{n}.pth` (K-fold). The K-fold training manifest, saved to `ensemble/kfold_manifest.json`, records per-fold accuracy, checkpoint path, and completion timestamp.

### 4.5.4 Confidence-Based Prediction

During inference, the softmax of the model's logits produces a probability vector over all 78 sign classes. The maximum probability value constitutes the confidence score. A base confidence threshold of 0.12 was established empirically: the ensemble output distribution was observed to concentrate in the 0.1–0.2 range for correct predictions in ambiguous scenarios, and a threshold at 0.12 preserves sensitivity while filtering clear non-detections. An additional penalty of 0.08 is applied to known similar-class pairs (`similar_class_penalty`) to reduce the risk of confusing visually similar signs. Predictions falling below the composite threshold are discarded, and the frame is treated as idle. This multi-threshold approach substantially reduces false positive word commits compared to a single global threshold.

---

## 4.6 Enhancements Implemented After Review 2

The following enhancements were implemented iteratively after Review 2, forming the principal technical contributions of the latter development phase (March–June 2026).

### 4.6.1 Relative Feature Integration

Prior to this enhancement, only raw hand landmark coordinates (126 dimensions) were used as input features. The face-relative coordinate block (an additional 126 dimensions) was integrated following the analysis that raw coordinates are inherently signer-position-dependent. A signer positioned at the left edge of the frame produces systematically different raw coordinate values than the same sign performed by the same signer at the centre of the frame, despite the underlying gesture being identical. By expressing hand positions relative to the face anchor — normalised by inter-eye distance — the feature representation becomes invariant to both the signer's position within the frame and the apparent scale of their face due to camera distance. This directly improves generalisation to unseen signers and recording environments. The implementation in `preprocess.py` `compute_face_relative_features()` was introduced in commit `15dcdfd6` (February 28, 2026) and enhanced with face-anchor shift augmentation in commit `c9771af2`.

### 4.6.2 Per-Class Threshold Optimisation

The initial system employed a single global confidence threshold. Analysis of per-class error patterns revealed that certain sign pairs (e.g., signs involving similar handshapes in proximal facial regions) consistently produced low but non-trivial confidence scores, leading to misclassification. A `similar_class_penalty` parameter (value 0.08 in `config.py` `InferenceConfig`) was introduced to apply an elevated effective threshold to sign pairs identified as visually similar. This does not require retraining: the penalty is applied at inference time by augmenting the base threshold for specific class pairs, effectively requiring higher certainty before committing visually ambiguous predictions. This targeted approach improved per-class precision for the most frequently confused class pairs without degrading recognition speed on well-separated classes.

### 4.6.3 Temporal Stability Improvements

The initial inference pipeline committed a sign as soon as the model's argmax prediction changed, resulting in highly unstable output: a single outlier frame could interrupt a correct sign detection mid-gesture or insert spurious words. The `TemporalPostProcessor` (implemented in `temporal_postprocessor.py`, integrated in commit `a63d818`) addresses this through a two-stage pipeline:

The `ConfidenceSmoother` maintains a sliding window deque of the 8 most recent probability vectors. Each entry is weighted by its confidence score (the maximum softmax probability) multiplied by an exponential decay factor of 0.3 applied to older entries, so that more recent frames carry proportionally greater influence. The weighted average is renormalised to produce a smoothed probability distribution.

The `StablePredictor` operates on the smoothed output. It maintains a candidate class and a patience counter: the candidate class must be predicted for 3 consecutive frames, and its smoothed confidence must exceed that of the current stable class by at least 0.12 (the hysteresis margin), before a class switch is confirmed. This patience-plus-hysteresis mechanism eliminates single-frame transient switches while adapting quickly to genuine sign changes.

### 4.6.4 Transition Suppression Mechanism

During natural signing, the hand transitions between signs — a period of motion during which landmark configurations do not correspond to any well-defined sign. Without suppression, the model confidently misclassifies transition frames, inserting spurious words into the output sentence. The momentum-based commit logic addresses this: a sign is only committed when it appears in at least 3 of the 5 most recent stable predictions and the average confidence across those occurrences is at least 0.60. Because transition frames typically produce low-confidence, inconsistent predictions across the 5-frame window, they rarely achieve 3-of-5 majority. Additionally, an ambiguity delay of 4 frames is imposed when the margin between the top-1 and top-2 softmax probabilities is less than 0.05, providing additional suppression during uncertain moments. The `sign_idle_timeout` of 30 frames (approximately 1 second at 30 FPS) resets the sentence builder when hands are absent, preventing stale predictions from propagating.

### 4.6.5 Real-Time Pipeline Optimisation

Multiple targeted optimisations were implemented to bring the end-to-end latency within the sub-200 ms target:

**Detection interval caching:** MediaPipe hand and face detection run every 5 frames (adaptive up to 8 during low-motion periods), with cached landmarks reused between detection frames. Landmark re-use reduces the per-frame MediaPipe overhead from approximately 30–40 ms to under 5 ms on cached frames.

**HOG detection disabled:** The HOG-based person-presence check was disabled (`disable_hog_detection: bool = True`), saving approximately 8 ms per frame without meaningful accuracy loss, since the MediaPipe face landmarker already serves as the primary anchor.

**Module-level buffer cache:** `preprocess.py` allocates fixed NumPy buffers (`_LANDMARK_BUFFERS`) at module load time for `left_raw`, `right_raw`, `left_rel`, and `right_rel`. These buffers are reset via `.fill(0)` and reused in-place each frame, reducing per-frame NumPy allocation overhead from approximately 12 array allocations to approximately 1 (the final concatenation). This was implemented as the "Phase 1 Optimization" noted in `preprocess.py`.

**ONNX INT8 inference:** The trained PyTorch model is exported to ONNX format using `export_onnx.py` (opset 18, dynamic batch size) and quantised to INT8 via `quantize_onnx.py` using dynamic quantisation (`onnxruntime.quantization.quantize_dynamic`). The resulting INT8 model is approximately 1.05 MB (reduced from approximately 4.2 MB FP32), and runs 2–3× faster on a CPU than the PyTorch FP32 path.

### 4.6.6 Model Robustness Improvements

**Handling Motion Variation:** Eight distinct online augmentation operations are applied during training in `ISLDataset._augment()`: (1) Gaussian noise injection (σ = 0.015, 70% probability); (2) random uniform scaling (0.88–1.12×, 60% probability); (3) temporal frame shift via circular roll (−3 to +3 frames, 50% probability); (4) random frame dropout (1–3 frames zeroed, 30% probability); (5) XY-plane rotation of raw landmark blocks (−15° to +15°, 40% probability); (6) time warping by resampling the 20-frame sequence at 0.75×–1.25× speed (40% probability); (7) per-hand dropout (up to one-third of frames for a randomly selected hand, 20% probability); and (8) stronger localised noise on a random subset of frames (25% probability).

**Handling User Variation:** Video-level augmentation in `preprocess.py` applies up to 54 distinct photometric and geometric transformations to each source video before landmark extraction: 17 visual effects (brightness, contrast, hue shift, fog, rotation, scale, colour jitter, Gaussian noise, pixel dropout, coarse dropout, motion blur, defocus blur, JPEG artefact compression, gamma correction, white balance shift, perspective warp, temporal jitter) combined with 3 crop positions (centre, left-offset at 15%, right-offset at 85%), yielding up to 54 augmented variants per original video. Additionally, `augmentations.py` implements face-anchor shift augmentation (random translation of the face reference point to simulate signer repositioning) and hand-proportion simulation (random per-finger scale factors to simulate different hand sizes), both applied at the landmark sequence level.

**Handling Environmental Changes:** The MediaPipe confidence threshold is set lower during training-data extraction (0.3) than during live webcam inference (0.5), ensuring that the training data includes some lower-confidence detections representative of challenging environments, while live inference applies stricter filtering to reduce false positive detections under good lighting. The face-relative normalisation further decouples the feature representation from ambient lighting and background changes, since it is based on relative spatial ratios rather than absolute pixel intensities.

---

## 4.7 Real-Time Recognition System Implementation

### 4.7.1 Live Webcam Processing

The `webcam.py` module manages the live inference loop. OpenCV's `VideoCapture` interface opens the default webcam device and reads frames at the native capture rate (target: 30 FPS). Each frame is immediately centre-cropped to 640 × 480 pixels to match the preprocessing geometry. Frame timing is monitored: if frame acquisition falls below 25 FPS, a warning is logged via `pipeline_logger.py` to assist in latency diagnosis. The main capture loop is structured as a producer-consumer pattern: landmark detection and feature extraction run synchronously within the loop, while text display updates are performed asynchronously to the OpenCV display window.

### 4.7.2 Real-Time Feature Extraction

Within the webcam loop, hand landmark detection is gated by the adaptive interval logic: a counter tracks the number of frames since the last full detection, running MediaPipe only when the counter exceeds the current adaptive interval (nominally 5, extended to up to 8 during low-motion periods). Cached hand landmarks are used for intermediate frames. Face landmark detection follows the same 5-frame interval, with a hard forced re-detection every 15 frames. When valid hand and face landmarks are available, `extract_landmarks_with_face_relative()` computes the 253-dimensional base feature vector using the pre-allocated buffer cache, computes the proximity scalar, and appends the velocity delta from the previous frame, yielding the 506-dimensional frame feature vector.

### 4.7.3 Model Inference Pipeline

The 20-frame buffer is managed by a `collections.deque(maxlen=20)`. Once the deque reaches full capacity, inference is triggered on every new frame (a sliding window approach). The feature buffer is converted to a NumPy array of shape `(1, 20, 506)` and passed to the `ONNXModelWrapper` in `onnx_inference.py`. The wrapper handles: (1) feature dimension alignment (pad or truncate if the current feature dim differs from the model's expected input); (2) proximity vector rank adjustment (scalar → 2D tensor for batch dimension); (3) batch axis addition if required; and (4) ONNX Runtime session invocation. On failure, it falls back to the PyTorch model. The returned logits are passed through softmax to produce class probabilities.

### 4.7.4 Text Generation Mechanism

The `SentenceBuilder` class in `sentence_builder.py` maintains the current accumulated sentence as a list of committed sign labels. Sign labels are appended only when the momentum commit condition is met (3-of-5 majority window, minimum average confidence 0.60) and the new label differs from the last committed label (preventing immediate repeated word appends). An ambiguity delay of 4 additional frames is applied when the top-1 minus top-2 softmax probability is less than 0.05, requiring stronger evidence before committing visually ambiguous predictions. The `nlp_postprocessor.py` module applies rule-based post-processing: capitalisation of the first word, insertion of grammatical connectors where inferred, and punctuation normalisation. The cleaned sentence string is returned for display.

### 4.7.5 User Interface Integration

The OpenCV-based display window renders the live webcam feed with real-time visual overlays: detected hand landmark skeleton drawn on the frame, the current predicted sign label and confidence score displayed in the upper-left corner, and the accumulated sentence string displayed at the bottom of the frame. A colour-coded confidence bar provides immediate visual feedback on recognition certainty. The `app.py` module provides a keyboard interface: pressing 'U' undoes the last committed word (pop from sentence list), pressing 'C' clears the entire sentence, and pressing 'Q' exits the application. Preset phrases are also configurable via the presets mechanism documented in the README.

---

## 4.8 Testing and Validation

### 4.8.1 Unit Testing

Unit-level validation was performed on each discrete pipeline component to confirm correct behaviour in isolation before integration testing.

**Webcam Capture:** The `VideoCapture` initialisation was tested by verifying that `cap.isOpened()` returns `True` and that frame dimensions match the configured 640 × 480 target. Frame rate consistency was tested by measuring capture intervals over a 5-second window and confirming that the mean frame interval was within ±2 ms of the expected 33.3 ms at 30 FPS.

**Landmark Extraction:** `extract_landmarks_with_face_relative()` was validated with synthetic test inputs: a frame containing a known synthetic hand at a known position. The expected face-relative coordinate values (computed analytically from the known hand and face positions) were compared against the function's output. The proximity scalar was verified to match the analytically computed L2 distance. Zero-fill behaviour was confirmed for frames with no detected hands and for frames with no detected face.

**Dataset Generation:** The `ISLDataset` loader was tested by loading a small synthetic dataset of three classes, verifying that the class count, sample count, and label distribution match expected values. The corrupt-file handling was tested by injecting a zero-byte `.npy` file into the test dataset; the loader was verified to log a warning and skip the file without raising an unhandled exception.

**Model Loading:** The `SignLanguageGRU` model was instantiated with `num_classes=78` and a synthetic input tensor of shape `(2, 20, 506)` was passed through the forward pass. Output logit shape `(2, 78)` was confirmed. The ONNX model was loaded and verified to accept the same input shape, returning matching output shapes. The shape trace audit conducted in commit `ff6a57bb` ("Complete comprehensive technical audit: Shape trace + GNN feasibility analysis") formally documented the expected shape at each layer.

**Prediction Generation:** The ONNX wrapper's dimension-alignment logic was tested by deliberately providing inputs of shape `(1, 20, 253)` (pre-velocity baseline features) and confirming that the wrapper correctly pads to `(1, 20, 506)` and returns valid logits without error.

### 4.8.2 Integration Testing

**End-to-End Pipeline Testing:** The complete pipeline from webcam capture to text output was verified by running the live webcam loop (`webcam.py`) with a known sign (the sign for "hello") performed 20 times and observing that the committed text output was "hello" in at least 18 of 20 trials. Pipeline event logs from `pipeline_logger.py` were inspected to confirm that all stages (feature extraction, inference, temporal smoothing, sentence building) completed within their expected time budgets.

**Data Flow Verification:** The shape of tensors at each stage was verified against expected values during integration: feature vector shape `(20, 506)`, proximity vector shape `(20,)`, logit shape `(78,)`, and softmax output shape `(78,)`. The `debug_print_shapes` flag (Phase 7 in `config.py`) was activated during integration testing to emit per-layer shape logs without modifying the inference code.

**Real-Time Performance Testing:** The end-to-end latency from frame capture to text update was measured using Python's `time.perf_counter()`. Over 500 consecutive frames, the 95th-percentile end-to-end latency was measured to confirm compliance with the sub-200 ms target. The K-fold manifest (`ensemble/kfold_manifest.json`) records per-fold training duration and validation accuracy for reproducibility.

### 4.8.3 Functional Testing

**Recognition Accuracy:** Each of the 78 sign classes was performed 10 times by the primary developer. The sign was recorded as "correctly recognised" if the committed word in the `SentenceBuilder` output matched the ground-truth label within a 3-second window. Recognition accuracy was evaluated per-class and aggregated across all 78 classes.

**Correct Text Generation:** The sentence accumulation pipeline was tested with a scripted sequence of 5 signs performed in order. The `SentenceBuilder` output was compared against the expected word sequence to verify that no signs were omitted, no duplicate words were inserted, and the output sentence was grammatically post-processed correctly.

**Class Detection Verification:** For each of the 78 classes, at least one correctly predicted sample and one correctly rejected non-sign sample were verified. The reject class (`__reject__`) was tested by performing arbitrary background hand movements not corresponding to any trained sign; the model was confirmed to output `__reject__` or a below-threshold confidence, preventing text output.

### 4.8.4 Robustness Testing

**Lighting Variations:** Testing was conducted under three distinct lighting conditions: standard indoor fluorescent lighting (reference condition), dim incandescent lighting at approximately 40% of reference intensity, and direct sunlight through a window (creating strong shadows and high contrast). Recognition accuracy was observed to degrade in dim lighting, primarily due to reduced MediaPipe detection confidence (which drops below the 0.5 webcam threshold). The face-relative normalisation maintained representation consistency between fluorescent and sunlight conditions where MediaPipe detection remained stable.

**Background Variations:** Tests were conducted against a plain white wall (reference), a cluttered bookshelf background, and an outdoor scene visible through a window. Background complexity had negligible impact on recognition accuracy, as MediaPipe's landmark detection operates on hand keypoints rather than background pixels, and the feature representation does not encode background information.

**Camera Distance Variation:** Testing was performed at three signing distances: approximately 0.5 m (very close), 1.0 m (standard), and 1.5 m (distant). At 0.5 m, certain signs where both hands extended beyond the frame edges were partially detected; recognition accuracy dropped for these signs. At 1.0 m and 1.5 m, where hands remained within the frame, the face-relative normalisation successfully compensated for the apparent size change, maintaining recognition accuracy. A recommended signing distance of 0.6–1.2 m was established based on these tests.

**Hand Speed Variation:** Signs were performed at three speeds: deliberate (approximately 1.5× slower than natural), natural pace, and rapid (approximately 1.5× faster than natural). The time-warping augmentation applied during training (0.75×–1.25× resampling) ensures the model has been exposed to speed-varied versions of each sign. Deliberate-speed performance was comparable to natural pace. Rapid signing occasionally caused the 20-frame buffer to capture an incomplete sign (the hand exits the frame before the buffer is full), leading to reduced confidence scores and occasional missed recognitions.

**Multiple Users:** Three additional users (beyond the primary developer) performed each of the 78 signs. Generalisation across users was observed to be strongest for signs performed close to the face (where face-relative normalisation provides strong invariance) and weakest for signs involving extended hand positions far from the face (where absolute position variation is large relative to the face anchor).

### 4.8.5 Performance Evaluation

| Metric | Measured Value |
|---|---|
| Webcam capture rate | 30 FPS (640 × 480) |
| MediaPipe detection per full frame | 30–40 ms (every 5 frames) |
| MediaPipe on cached frames | < 5 ms |
| ONNX INT8 inference (20-frame buffer) | 5–15 ms per inference |
| PyTorch FP32 inference (fallback) | 30–60 ms per inference |
| TemporalPostProcessor per frame | < 1 ms |
| End-to-end latency (95th percentile) | < 200 ms |
| Model size (FP32 PyTorch) | ~4.2 MB |
| Model size (INT8 ONNX) | ~1.05 MB (75% reduction) |
| Inference speedup (ONNX vs PyTorch) | 2–3× |
| Sustained FPS during live inference | 25–30 FPS |

The primary computational bottleneck is the MediaPipe landmark extraction on full-detection frames. The adaptive interval mechanism (5 frames base, up to 8 in low-motion) reduces effective MediaPipe overhead by 60–80% relative to per-frame detection. The HOG detection bypass saves a further 8 ms per full-detection frame.

### 4.8.6 Error Analysis

**Similar Sign Confusion:** The most frequently observed error category was confusion between sign pairs sharing similar handshapes performed near the same facial region. For example, signs differing only in the orientation of the wrist or the extension state of the little finger were occasionally confused when performed quickly. The `similar_class_penalty` mechanism reduces this error rate, though it does not eliminate it for sign pairs with very high visual similarity.

**Transition-Related Errors:** During the interval between consecutive signs, the hand passes through configurations that may superficially resemble known signs at low confidence. Without the momentum commit logic (3-of-5 window) and the confidence threshold (0.60 minimum average), transition frames occasionally committed spurious words. Post-implementation, transition errors were substantially reduced, though rapid multi-sign sequences performed without deliberate pauses between signs still present a challenge.

**Landmark Detection Failures:** In approximately 3–5% of observed frames under nominal lighting, MediaPipe failed to detect one or both hands, particularly when hands were partially occluded by the body or when the signer's skin tone had low contrast against the background under certain lighting angles. These frames produce zero-filled feature blocks, which the trained model handles by outputting low-confidence predictions that fall below the commit threshold.

**Environmental Limitations:** The system is not robust to extreme illumination changes such as backlighting (the signer between the camera and a bright window), which causes MediaPipe to fail on hand detection in the majority of frames. Ambient noise in the webcam feed under dim lighting also degrades landmark precision.

### 4.8.7 Validation Results

The integration of all enhancements described in Section 4.6 produced measurable improvements over the Review 2 baseline:

- The TemporalPostProcessor reduced observable prediction jitter from multiple flickered word outputs per second to zero flicker during stable sign holds, confirmed by visual inspection of the live system over 10-minute operation sessions.
- The momentum commit logic (3-of-5, min confidence 0.60) eliminated spurious transition-frame word insertions in 18 of 20 transition test trials (versus 12 of 20 for the pre-enhancement baseline).
- The ONNX INT8 quantisation reduced model file size by 75% (from approximately 4.2 MB to approximately 1.05 MB) and improved inference speed by 2–3× on CPU, enabling sustained 25–30 FPS live operation on a standard mid-range laptop.
- The face-relative feature integration improved cross-user recognition: the three secondary users achieved mean recognition accuracy approximately 12 percentage points higher than with the raw-coordinate baseline on face-proximate signs.
- The reject class (`__reject__`) training, using `processed_negatives/`, eliminated false positive word insertions during deliberate non-signing periods in all 20 idle-state test trials.

---

## 4.9 Challenges Encountered and Solutions Implemented

### Challenge 1: ONNX Dimension Mismatch at Runtime

**Problem:** After updating the feature pipeline from 253D base features to 506D (with velocity), previously exported ONNX models expected a 253-dimensional input. Attempting inference with the updated 506D features caused the ONNX Runtime to raise a dimension mismatch exception, crashing the live pipeline.

**Root Cause:** The ONNX model graph encodes the input tensor shape at export time. When the feature engineering pipeline was updated without re-exporting the ONNX model, the static shape in the ONNX graph became inconsistent with the runtime input.

**Implemented Solution:** A multi-layer dimension alignment mechanism was implemented in `onnx_inference.py` (`ONNXModelWrapper`). At inference time, the wrapper compares the actual input feature dimension against the ONNX model's expected dimension. If they differ, the input is padded with zeros (if too small) or truncated (if too large) to match. The proximity vector rank is also verified and expanded separately. This runtime alignment eliminates crashes from version mismatches and provides a diagnostic log message identifying the mismatch.

**Outcome:** The live pipeline operates stably regardless of which previously exported ONNX version is loaded, and informative logs alert the operator when a re-export is advisable.

---

### Challenge 2: K-Fold Training Crash with Weighted Samples

**Problem:** K-fold cross-validation training crashed with `ValueError: too many values to unpack (expected 2)` on the label extraction loop.

**Root Cause:** The dataset format was updated from 2-tuples `(path, label)` to 3-tuples `(path, label, sample_weight)` when per-sample weighting was introduced for archived data. The K-fold label extraction in `train.py` still used the legacy unpacking pattern `[lbl for _, lbl in full_ds.samples]`.

**Implemented Solution:** A `_sample_label(sample) -> int` helper function was introduced (`train.py`, lines 76–82, commit `4672472b`). The function always reads `sample[1]`, making it compatible with both 2-tuple and 3-tuple formats. All K-fold label extraction calls were updated to use this helper.

**Outcome:** K-fold training executes successfully without crashes. The 3-tuple format is now the canonical sample representation throughout the training pipeline.

---

### Challenge 3: Real-Time Prediction Instability (Jitter)

**Problem:** The initial inference pipeline committed a new word on every frame where the argmax prediction changed. Under natural lighting and with the hand in motion, the model's argmax flickered between 2–4 different classes at up to 10–15 Hz, making the output sentence unreadable.

**Root Cause:** Individual frame predictions are inherently noisy for a sequence model operating on a sliding window: each new frame shifts the entire buffer by one position, and the model's logit distribution changes rapidly as the sign gesture progresses or transitions.

**Implemented Solution:** The three-layer post-processing stack was implemented: (1) `ConfidenceSmoother` with an 8-frame weighted sliding window; (2) `StablePredictor` with 3-frame patience and 0.12 hysteresis; and (3) the momentum commit logic (3-of-5 window, 0.60 average confidence threshold). Each layer independently reduces jitter, and their combination virtually eliminates flicker under normal operating conditions.

**Outcome:** Zero observable jitter during stable sign holds in all validation sessions. Transition suppression reduced spurious transition-frame word insertions by approximately 90%.

---

### Challenge 4: Dataset Class Imbalance

**Problem:** The initial dataset had highly variable class sizes: some classes (common signs such as "hello" and "thank you") had 150+ samples, while rarer or recently added classes had as few as 30 samples. The standard cross-entropy loss function, trained on this imbalanced distribution, learned to bias predictions towards majority classes.

**Root Cause:** Naturally occurring variation in recording effort and sign complexity resulted in unequal sample counts. The class with the smallest sample count was approximately 5× smaller than the class with the largest sample count.

**Implemented Solution:** Three complementary strategies were applied: (1) `_BalancedAugSubset` oversamples minority classes by repeating their samples until all classes (except the reject class) match the majority class count; (2) inverse-frequency class weights (power 1.0, normalised to mean = 1) are applied as per-sample multipliers to the loss; (3) the `balance_processed_dataset.py` script enforces an 850-sample target per class during dataset preparation, applying downsampling to over-represented classes and oversampling (via augmentation) to under-represented ones.

**Outcome:** The effective class distribution seen by the gradient update step is balanced, and the per-class validation accuracy distribution narrows considerably: the standard deviation of per-class accuracy decreases relative to the unweighted baseline.

---

### Challenge 5: Computational Constraints for Real-Time CPU Inference

**Problem:** Running MediaPipe hand landmark detection on every frame at 30 FPS consumed approximately 40 ms per frame (25 FPS maximum), leaving insufficient budget for model inference, post-processing, and display. Adding face landmark detection to every frame further degraded throughput.

**Root Cause:** MediaPipe's hand and face landmark models, while optimised, require non-trivial CPU time per frame when run at full resolution.

**Implemented Solution:** Four optimisations were applied: (1) adaptive detection interval caching (5 frames nominal, up to 8 in low-motion) reducing effective MediaPipe execution frequency; (2) forced re-detection every 15 frames to prevent stale tracking; (3) HOG person detection disabled (saves ~8 ms per detection frame); and (4) module-level NumPy buffer pre-allocation (`_LANDMARK_BUFFERS` in `preprocess.py`) eliminating per-frame heap allocation overhead. Additionally, ONNX INT8 quantisation reduced model inference time by 2–3×.

**Outcome:** Sustained live inference at 25–30 FPS on a standard mid-range laptop CPU, meeting the real-time usability target.

---

## 4.10 Current Status of the Module

The Sign-to-Text module is functionally complete and production-ready at its current scope.

**Completed Features:**
- Full 78-class ISL word recognition pipeline from raw webcam input to text output.
- 506-dimensional velocity-augmented, face-relative spatiotemporal feature extraction.
- 10-phase `SignLanguageGRU` architecture (Conv1D frontend, Spatial GNN, Frame Weighting, BiGRU ×3, HybridAttention ×4 heads, Residual Skips, FC Head).
- 5-fold K-fold cross-validation training pipeline with per-fold checkpoint saving and manifest.
- Two-phase training strategy (Phase 1: curated data; Phase 2: archived fine-tune at 0.25 weight).
- Reject class training with `processed_negatives/` for false-positive suppression.
- ONNX INT8 quantisation with dimension-aligned PyTorch fallback.
- `TemporalPostProcessor` with confidence smoothing (window = 8, decay = 0.3) and `StablePredictor` (patience = 3, hysteresis = 0.12).
- Momentum-based commit logic (3-of-5 window, minimum average confidence 0.60).
- `SentenceBuilder` with ambiguity delay (4 frames when top-1 − top-2 < 0.05).
- NLP post-processor for grammar and punctuation cleanup.
- Comprehensive dataset pipeline: `collect_data.py`, `preprocess.py`, `augment_video_pipeline.py`, `merge_augmentations.py`, `balance_processed_dataset.py`.
- Conditional VAE (`cvae_landmarks.py`) and quality discriminator (`quality_discriminator.py`) for synthetic data generation.
- Adapter model skeleton (`adapter_model.py`) for user-specific personalisation (experimental).
- Centralised, validated configuration system (`config.py`, CONFIG_VERSION 2.0.0) with 10+ typed dataclasses.

**Testing Completed:**
- Unit testing: landmark extraction correctness, dataset loading, model forward pass, ONNX alignment.
- Integration testing: end-to-end pipeline shape verification, real-time latency measurement.
- Functional testing: 78-class recognition across 10 trials per class, sentence accumulation correctness.
- Robustness testing: three lighting conditions, three background types, three camera distances, three signing speeds, three additional users.

**Current Readiness Level:** Production-ready for isolated ISL word recognition at 78-class scope on CPU hardware. Not yet production-ready for continuous sign sequence recognition or multi-user personalised deployment.

---

## 4.11 Future Enhancements

**Expansion Beyond Current Sign Classes:** The architecture supports an arbitrary number of output classes by modifying the final `Linear(96→N)` layer. Expansion to 100 or 200+ sign classes requires additional data collection, dataset balancing, and retraining, but no architectural changes. The K-fold training framework and ONNX export pipeline are already designed to support larger class sets.

**Hierarchical Classification:** For very large vocabularies (500+ signs), a two-stage hierarchical classifier can be employed: a coarse classifier first identifies the sign category (e.g., greeting, direction, number), followed by a fine classifier specialised to that category. This approach reduces the effective classification problem size at each stage, improving accuracy and reducing confusion between semantically distant but visually similar signs from different categories.

**Continuous Sign Language Recognition:** The current system performs isolated word recognition — it recognises one sign at a time, separated by idle pauses. Continuous sign language recognition, where signs are performed in a flowing sequence without deliberate pauses, requires sequence segmentation (e.g., via Connectionist Temporal Classification or Hidden Markov Models). Integration of a segmentation module would substantially improve the system's naturalness for experienced signers.

**Context-Aware Prediction:** Integrating a language model (e.g., a fine-tuned T5 or BERT model) as a re-ranker over the top-5 sign predictions, conditioned on the previously committed words, would enable context-aware disambiguation of visually similar signs based on linguistic plausibility. The infrastructure for top-5 output already exists in the model's softmax output.

**Signer Personalisation:** The adapter model skeleton in `adapter_model.py` provides the foundation for user-specific fine-tuning. Integrating an online few-shot adaptation mechanism — where 5–10 examples per sign from a new user trigger a lightweight adapter weight update — would substantially improve cross-user generalisation without requiring full retraining.

**Mobile and Edge Deployment:** The INT8 ONNX model (1.05 MB) is well within the size budget for mobile deployment. Conversion to TensorFlow Lite or CoreML format would enable deployment on Android and iOS devices respectively. MediaPipe's mobile SDKs provide compatible hand and face landmarker implementations, making the complete pipeline portable to mobile with relatively small additional engineering effort.

**Multilingual Text Generation:** The NLP post-processor currently outputs English text. Extending it to Hindi, Konkani, or other Indian languages would increase the system's utility for the ISL-using community in India. Since the model outputs sign labels (which are language-neutral gesture identifiers), multilingual support requires only a mapping layer from sign labels to translated text, without any changes to the recognition model itself.

---

## 4.12 Conclusion

The Sign-to-Text module represents a complete, multi-phase implementation of a real-time Indian Sign Language recognition system, developed from first principles over a period of approximately 3.5 months (February to June 2026) across 173 version-controlled commits. The implementation advances beyond a baseline recognition pipeline through ten independently configurable architectural improvements to the core `SignLanguageGRU` model — including a Conv1D depthwise-separable frontend, a lightweight Spatial GNN, learnable frame weighting, multi-layer BiGRU with reduced dropout, and a HybridAttention mechanism combining temporal and proximity-aware attention heads with learnable temperatures — together with a comprehensive feature engineering pipeline that produces 506-dimensional velocity-augmented, face-relative spatiotemporal representations.

The testing programme, encompassing unit, integration, functional, robustness, and performance evaluation, confirms that the module meets its core design objectives: real-time operation at 25–30 FPS on a CPU-only system, correct recognition across 78 ISL classes, stable output under natural lighting and environmental variation, and a text generation mechanism that suppresses the transition and jitter errors that dominated earlier pipeline iterations.

The enhancements implemented after Review 2 — relative feature integration, per-class threshold optimisation, the TemporalPostProcessor, momentum-based commit suppression, ONNX INT8 optimisation, and multi-level augmentation — collectively transformed the module from a functionally incomplete prototype into a deployable system suitable for real-world demonstration. The two-phase training strategy and reject-class suppression mechanism further enhance robustness and practical reliability. The module's modular design — all hyperparameters centralised in a validated dataclass configuration, all architectural phases independently toggleable — provides a solid foundation for future enhancements including continuous recognition, user personalisation, and mobile deployment.

---

*Report Section: Sign to Text Module | Goa College of Engineering Final Year Project | June 2026*














## 4. SIGN TO TEXT MODULE

### 4.1 Module Overview

The Sign-to-Text module constitutes the core predictive engine of the real-time Indian Sign Language (ISL) recognition system. The fundamental purpose of this module is to bridge the communication gap for the approximately 5 million individuals in the Indian deaf community by directly translating continuous spatial hand gestures into natural language text. Operating entirely on standard consumer-grade computing hardware without requiring specialized GPUs or depth cameras, this module functions as the critical translation layer within the complete system architecture.

The input to the module consists of live RGB video frames captured via a standard webcam at a resolution of 640×480 pixels operating at 30 frames per second. The output of the module is grammatically corrected, natural language text corresponding to 78 distinct ISL sign classes. The overall workflow involves capturing the video feed, extracting multi-dimensional spatial keypoints utilizing Google's MediaPipe framework, engineering scale-invariant spatiotemporal features, performing sequence classification over a rolling buffer using a custom multi-phase deep learning architecture, and finally applying rule-based temporal smoothing and linguistic post-processing to generate a stable text output.

### 4.2 System Architecture

The complete system architecture is designed as a high-throughput, low-latency pipeline to facilitate real-time inference. The pipeline begins with the webcam input, which continuously feeds RGB frames into the feature extraction subsystem.

To process the visual data, the architecture integrates the MediaPipe Tasks API, specifically utilizing two distinct models: the `hand_landmarker.task` for detailed anatomical hand parsing and the `face_landmarker.task` for spatial anchoring. Landmark extraction occurs at dynamically adjusted intervals to optimize processing speeds, outputting absolute coordinate values for 21 hand joints per hand, as well as specific facial anchor points.

Following extraction, the raw coordinates pass into the feature generation stage. The architecture does not rely on raw pixel data or basic absolute coordinates; instead, it processes 126 raw hand coordinate dimensions, 126 face-relative normalized dimensions, and a 1-dimensional hand-to-face proximity scalar. To capture motion dynamics, the system calculates frame-to-frame velocity deltas, resulting in a robust 506-dimensional feature vector per frame.

These frame-level features are accumulated into a fixed sequence formation consisting of 20 consecutive frames, creating a $20 \times 506$ spatiotemporal tensor matrix representing a single sign gesture. This tensor is fed into the deep learning inference engine, which utilizes a 10-phase Gated Recurrent Unit architecture featuring a 1D Convolutional frontend and a Spatial Graph Neural Network (GNN) branch to extract both local temporal patterns and spatial joint relationships.

The deep learning model outputs raw softmax probabilities across the 78 sign classes. The final text generation phase passes these probabilities through a Temporal Post-Processor that applies confidence-weighted smoothing, hysteresis thresholding, and a momentum-based commit logic before the `SentenceBuilder` and `nlp_postprocessor.py` modules compile the discrete classifications into grammatically structured text.

### 4.3 Data Acquisition and Dataset Preparation

#### 4.3.1 Sign Collection Process

A robust, webcam-based recording system was developed specifically for this project via the `collect_data.py` utility. This module enables the real-time capture of sign gestures through an interactive countdown interface, ensuring uniform sample lengths. To ensure generalization, multiple recordings per class were generated, ultimately yielding an active training set of approximately 5,683 processed samples distributed across 78 sign classes.

The recording strategy incorporated both controlled and uncontrolled environments. Samples were collected under varying lighting conditions and camera angles to simulate real-world usage. To artificially expand the dataset's robustness, an offline video augmentation pipeline (`augment_video_pipeline.py`) was implemented, generating up to 54 structural variants per original video by applying 17 distinct visual effects (including brightness, contrast, hue, fog, and pixel dropout) across 3 different crop positions.

#### 4.3.2 Landmark Extraction

The extraction mechanism intentionally utilizes the lightweight MediaPipe Tasks API rather than legacy solutions or computationally heavy pose estimation frameworks. The `hand_landmarker.task` model extracts 21 precise 3D anatomical joints for up to two hands. The `face_landmarker.task` extracts 478 points, but the module programmatically isolates only the nose center (index 1), left eye (index 33), and right eye (index 263).

Full-body pose landmarks were intentionally excluded from the extraction pipeline. This reduction of unnecessary features drastically improves computational efficiency, saving approximately 8 milliseconds per frame by disabling internal HOG person detection and bypassing heavy pose estimation networks. The finalized extraction results in approximately 506 features used per frame, combining raw spatial data, facial anchor normalization, and velocity calculations.

#### 4.3.3 Data Preprocessing

Data preprocessing ensures tensor integrity before deep learning inference. Missing landmark handling is explicitly managed; if the `_extract_face_anchor()` function fails to detect a face, or if a hand briefly leaves the camera view, the system gracefully falls back by zero-filling pre-allocated numpy buffers (`_LANDMARK_BUFFERS`), preventing pipeline crashes. Frame normalization is achieved by converting all spatial data relative to the bounding box dimensions. Sequence standardization forces all temporal clips into exactly 20-frame sequences, padding or truncating dynamically. Noise reduction occurs both spatially through MediaPipe's internal exponential moving average and temporally via the sequence dataset builder.

#### 4.3.4 Dataset Organization

The project directory follows a strict hierarchical folder structure. Raw MP4 files are stored in the `Dataset/` directory, organized into 78 class-wise subfolders. The `preprocess.py` module converts these raw videos into highly compressed NPY file generations, stored in the `processed/` directory. To ensure balanced training, the `balance_processed_dataset.py` script automatically manages dataset creation, replicating minority class samples or trimming majority classes to an exact target of 850 samples per class, creating a training-ready distribution.

### 4.4 Feature Engineering

#### 4.4.1 Landmark-Based Features

The foundational features rely on accurate hand landmark coordinates, providing 63 dimensions (21 landmarks $\times$ 3 coordinates) per hand, totaling 126 raw spatial dimensions for both hands. Facial landmark coordinates (nose and eyes) are extracted not for direct classification, but to serve as a topological anchor for the relative spatial information calculations.

#### 4.4.2 Relative Feature Generation

To decouple the model from the physical characteristics of the user, the module implements relative feature generation. Relative positioning between landmarks is computed by subtracting the nose coordinate from every hand landmark coordinate. This value is then divided by the inter-eye distance to normalize the scale. This face-relative normalization yields an additional 126 dimensions that provide improved robustness against camera placement, dramatically reducing the system's user-position dependency, allowing successful inference regardless of whether the user sits close to or far from the webcam.

#### 4.4.3 Temporal Representation

Because static poses cannot distinguish motion-dependent signs (e.g., distinguishing "come" from "go"), temporal representation is strictly enforced. The system computes frame-to-frame velocity by calculating the delta between frame $t$ and frame $t-1$ across the entire 253-dimensional base space. This adds 253 velocity features, explicitly capturing motion dynamics. The final multi-frame sequence creation buffers these 506-dimensional vectors into strict 20-frame sequences, allowing the temporal architecture to analyze the complete spatiotemporal trajectory of the gesture.

### 4.5 Deep Learning Model Development

#### 4.5.1 Model Selection

The Modified 10-Phase Gated Recurrent Unit (MOPGRU) architecture—implemented internally as the `SignLanguageGRU`—was selected specifically for its optimal balance between parameter efficiency and sequence modeling capabilities. Standard Transformers were rejected as their $O(n^2)$ self-attention complexity over 506 dimensions would create prohibitive computational overhead for real-time CPU deployment. Standard LSTMs were bypassed in favor of GRUs because the GRU's two-gate structure requires fewer parameters than the LSTM's three-gate structure, accelerating inference times without sacrificing accuracy over short 20-frame temporal windows.

#### 4.5.2 Network Architecture

The network architecture is a highly engineered multi-branch system. The input layer accepts the $(batch, 20, 506)$ tensor. The sequence processing immediately branches: the first 126 raw dimensions are routed to a Spatial Graph Neural Network (GNN) consisting of 2 Graph Convolutional Network (GCN) layers operating over the 21-node anatomical hand skeleton, outputting a 16-dimensional feature vector per frame. Concurrently, the full 506-dimensional input passes through a depthwise-separable 1D Convolutional frontend, compressing the features into 128 channels.

These branches are concatenated to 144 dimensions and passed through learnable sigmoid frame-weighting before entering the recurrent core. The bidirectional GRU layers consist of 3 stacked layers with a hidden dimension of 64, yielding a 128-dimensional temporal output. This output is analyzed by a Hybrid Multi-Head Attention module featuring 4 heads. Crucially, two of these heads apply a spatial proximity bias using the formula:
$\text{scores} += -\frac{\text{prox}^2}{2\sigma^2}$
where $\sigma$ is a learnable parameter initialized to 0.15. Finally, dense layers apply dropout (0.25) and project the 128-dimensional context vector through a 96-dimensional hidden layer before the final Softmax classification layer outputs the 78 class probabilities.

#### 4.5.3 Training Strategy

The training strategy is engineered to prevent overfitting and ensure robust generalization. The dataset split utilizes a K-fold Disjoint Stratified partition, utilizing `_build_disjoint_folds()` to create 5 perfectly distinct subsets. The model is trained for 50 epochs with a batch size of 8. Optimization is handled by the AdamW optimizer with a weight decay of 5e-4 and a baseline learning rate of 3e-4.

The loss function is a per-sample weighted Cross-Entropy loss. To handle minor class imbalances not caught by oversampling, an inverse-frequency class weighting formula is applied:
$w_c = \left(\frac{1}{n_c}\right)^{1.0}$
Early stopping is enforced with a patience of 10 epochs based on validation accuracy. Furthermore, a `ReduceLROnPlateau` scheduler dynamically halves the learning rate upon a 5-epoch plateau. Model checkpointing saves the best performing weights per fold to the `ensemble/` directory, registering metrics in a `kfold_manifest.json` tracker.

#### 4.5.4 Confidence-Based Prediction

Raw softmax outputs exhibit high variance during live inference; therefore, confidence-based prediction logic is strictly enforced. Confidence score generation relies on a dynamic threshold dynamically adjustable from a 0.12 baseline. Prediction filtering relies on a momentum-based commit logic, explicitly requiring a specific class to appear 3 times within a rolling 5-frame window (the 3-of-5 rule) while maintaining a minimum average confidence of 0.60. This drastically aids in the reduction of false positives caused by transient hand movements between legitimate signs.

### 4.6 Enhancements Implemented After Review 2

Following Review 2, the system underwent an extensive refactoring phase encompassing architectural improvements, real-time stability fixes, and deployment optimizations.

#### 4.6.1 Relative Feature Integration

The most critical mathematical enhancement was the introduction of the relative feature integration. Initially, raw MediaPipe coordinates caused severe misclassifications if the user shifted off-center. By introducing face-anchor subtraction and scaling based on inter-eye Euclidean distance, the resulting features became strictly scale and position invariant. The benefits obtained included a massive increase in recognition consistency across different camera distances and signer heights.

#### 4.6.2 Per-Class Threshold Optimization

A rigid global threshold caused similar signs to be misclassified. To resolve this, individual confidence thresholds were introduced via the `InferenceConfig` architecture. A specific `similar_class_penalty` of 0.08 was implemented. This forced the model to require an extra 8% confidence margin when distinguishing between highly visually similar signs defined in `similar_signs.json`. This resulted in improved class-specific performance and a marked reduction in misclassification between visually identical gestures that differ only in temporal velocity.

#### 4.6.3 Temporal Stability Improvements

Real-time prediction instability was heavily mitigated through prediction smoothing. The `TemporalPostProcessor` module was integrated, specifically leveraging a `ConfidenceSmoother` utilizing an 8-frame rolling deque. Older frames within this window are exponentially decayed using a factor of 0.3. For stable output generation, consecutive frame verification is enforced via a `StablePredictor` which requires 3 consecutive class agreements and a hysteresis delta of 0.12 to confirm a class switch.

#### 4.6.4 Transition Suppression Mechanism

A major flaw identified during Review 2 was the triggering of spurious word predictions while a user transitioned their hands from one sign to another. The prevention of incorrect predictions during sign transitions was achieved by implementing an ambiguity margin threshold. If the difference between the top-1 and top-2 class probabilities is less than 0.05, the `SentenceBuilder` initiates a strict 4-frame ambiguity delay. This transition suppression mechanism forces the model to wait until the user settles into a definitive pose, resulting in highly improved recognition reliability.

#### 4.6.5 Real-Time Pipeline Optimization

To guarantee a sub-200ms latency target, real-time pipeline optimization focused on deployment formats. The native PyTorch FP32 models were exported to the ONNX Runtime architecture (opset 18). Dynamic INT8 quantization was applied via `quantize_onnx.py`, yielding a 75% reduction in model size (from 4.2 MB to 1.05 MB). This resulted in faster inference and reduced latency by 2-3x on standard CPUs, ensuring a significantly better user experience devoid of video lag.

#### 4.6.6 Model Robustness Improvements

Handling motion variations was tackled by dynamically injecting temporal jitter and velocity recomputation during training. Handling user variations was addressed by implementing the 17-effect video augmentation pipeline and specific landmark-level augmentations such as face-anchor shifting and random hand-proportion scaling. Handling environmental changes is now inherently managed by the relative spatial calculation, which ignores lighting-based pixel variance entirely in favor of geometric topology.

### 4.7 Real-Time Recognition System Implementation

#### 4.7.1 Live Webcam Processing

Live webcam processing is orchestrated by the `webcam.py` module, which instantiates an OpenCV `VideoCapture` object targeted at 30 FPS. The module establishes a non-blocking UI thread that renders bounding boxes and assignment labels directly onto the live feed.

#### 4.7.2 Real-Time Feature Extraction

Real-time feature extraction relies on a cached extraction policy to maintain high frame rates. While frames enter at 30 FPS, the MediaPipe `hand_landmarker.task` only executes every 5 frames under normal motion, gracefully adapting to an 8-frame interval during low-motion periods to conserve CPU cycles. The `face_landmarker.task` is similarly cached, executing every 5 frames and forcing a hard refresh every 15 frames.

#### 4.7.3 Model Inference Pipeline

The model inference pipeline in production bypasses standard PyTorch loops, heavily utilizing the `ONNXModelWrapper`. The system extracts the $(20, 506)$ feature matrix and passes it to the INT8 quantized ONNX ensemble. The integration layer performs crucial dimension alignment, automatically padding or truncating feature tensors if runtime dimensions mismatch the exported ONNX expected shapes. If the ONNX session fails, the wrapper seamlessly executes a native PyTorch FP32 fallback.

#### 4.7.4 Text Generation Mechanism

The final raw predictions feed the text generation mechanism handled by `sentence_builder.py` and `nlp_postprocessor.py`. As signs pass the momentum commit logic, they are appended to an active string buffer. The rule-based NLP processor standardizes casing, manages spacing between discrete words, and handles punctuation cleanup, ultimately transforming a list of predicted tokens (e.g., `["hello", "how_are", "you"]`) into coherent natural language text.

#### 4.7.5 User Interface Integration

User interface integration consists of real-time overlay metrics drawn directly via OpenCV. The interface displays the current active sign prediction, the aggregated sentence string, the status of the two-hand validation check ("Same person: YES/NO"), and immediate bounding box visual feedback validating that the MediaPipe Tasks API has successfully acquired the user's spatial topology.

### 4.8 Testing and Validation

Extensive testing protocols were executed to validate system integrity from individual functions up to the entire real-time pipeline.

#### 4.8.1 Unit Testing

Unit testing prioritized internal tensor routing. Smoke checks and manual shape tracing were committed (Commit `ff6a57bb`) to mathematically verify the dimensionality of the Conv1D frontend concatenation with the Spatial GNN branch. Dataset generation unit tests explicitly validated the behavior of `_BalancedAugSubset`, discovering and successfully patching a critical tuple-unpacking crash where the architecture expected 2-tuples but received 3-tuples containing `(path, label, weight)` data. Model loading and prediction generation were verified using isolated dummy tensors initialized via `debug_model.py`.

#### 4.8.2 Integration Testing

Integration testing focused on end-to-end pipeline operability. End-to-end pipeline testing verified the complete data flow from the `collect_data.py` webcam capture through the `preprocess.py` buffer caching to the final `.npy` file generation. Data flow verification explicitly tested the mixed PyTorch and ONNX ensemble script (`onnx_ensemble.py`), confirming that predictions from both backends could be averaged effectively without array shape conflicts. Real-time performance testing utilized the `profiling.py` harness to confirm that the entire cycle—from frame capture to text output—stayed well below the 200ms latency ceiling.

#### 4.8.3 Functional Testing

Functional testing measured the core competency of the AI. Recognition accuracy hit a notable 95.83% mean accuracy on internal benchmark subsets using the K-Fold ensemble strategy. Correct text generation was verified by performing consecutive distinct gestures in front of the webcam and confirming that the `sentence_builder.py` correctly assembled the semantic intent. Class detection verification confirmed that all 78 target sign categories defined in `sign_categories.json` could be successfully triggered by a human user.

#### 4.8.4 Robustness Testing

System robustness testing explicitly sought to break the predictive engine under hostile conditions. Testing under different lighting proved that relying on MediaPipe's topological landmarks (with a low 0.5 confidence threshold for webcam) bypassed standard pixel-level shading vulnerabilities. Different backgrounds failed to confuse the system due to the exclusion of background segmentation in favor of skeletal mapping. Different distances from the camera were successfully mitigated by the inter-eye division embedded within the face-relative feature mathematics. Different hand speeds were normalized by the temporal 20-frame buffering, and different users were tested to ensure the models generalized beyond the primary dataset contributor.

#### 4.8.5 Performance Evaluation

The performance evaluation heavily highlights the success of the optimization phase. Inference speed was increased by 2-3x after replacing native PyTorch FP32 models with INT8 ONNX binaries. The overall response time allows for immediate conversational feedback. Real-time usability is confirmed by the system's ability to maintain a consistent 30 FPS processing loop on standard CPU hardware. Computational efficiency was further proven by the system's tiny memory footprint; the final compiled recognition model requires only 1.05 MB of disk space.

#### 4.8.6 Error Analysis

Comprehensive error analysis identified remaining edge cases. Similar sign confusion occurs when temporal trajectories overlap; this was largely solved by the 0.08 `similar_class_penalty`. Transition-related errors—where hands form accidental sign shapes while moving to resting positions—were effectively neutralized by the momentum-commit logic and the 4-frame ambiguity delay. Landmark detection failures in extreme low light cause graceful zero-buffer fallbacks rather than system crashes. Environmental limitations are strictly tied to the webcam's hardware exposure capabilities.

#### 4.8.7 Validation Results

The validation results confirm that the integration of the 10-phase BiGRU with spatial-temporal inputs achieves production-grade accuracy. By shifting the computational burden away from heavy Transformer models and into elegant feature engineering (face-relative normalization + velocity concatenation), the system definitively proves that high-accuracy, 78-class continuous gesture recognition is highly viable on standard, non-GPU computer architectures.

### 4.9 Challenges Encountered and Solutions Implemented

Several critical engineering challenges emerged during the development cycle, requiring sophisticated programmatic interventions.

* **Problem:** K-fold Training Crash with Weighted Samples.
* **Root Cause:** A system update modified the internal `ISLDataset` representation from a 2-tuple `(path, label)` to a 3-tuple `(path, label, weight)` to accommodate Phase 2 archived fine-tuning data. The older K-fold logic crashed attempting to unpack 3 values into 2 variables.
* **Implemented Solution:** A dedicated `_sample_label(sample)` helper function was engineered in `train.py` to safely extract index 1 regardless of the tuple length.
* **Outcome:** K-fold training resumed successfully, yielding 5 perfectly distinct model folds.


* **Problem:** Real-time prediction instability and UI jitter.
* **Root Cause:** Rapid hand movements and transition frames caused the raw softmax probabilities to wildly oscillate frame-to-frame.
* **Implemented Solution:** Implementation of a strict 3-of-5 momentum commit window requiring a minimum average confidence of 0.60, alongside an 8-frame exponential decay smoother.
* **Outcome:** Transient spikes were suppressed, completely eliminating false-positive word insertions.


* **Problem:** ONNX Dimension Mismatch during deployment.
* **Root Cause:** The input feature dimensions expanded from 253D to 506D when velocity tracking was introduced, causing older PyTorch models exported to ONNX to crash when receiving live 506D arrays.
* **Implemented Solution:** An intelligent multi-layer alignment protocol was embedded into `onnx_inference.py` to dynamically pad or truncate feature tensors and auto-align batch dimensions.
* **Outcome:** Seamless ONNX execution with a reliable automated fallback to native PyTorch.


* **Problem:** Dataset imbalance skewing predictions.
* **Root Cause:** Natural data collection yielded highly uneven class distributions (e.g., 50 samples for hard signs, 850 for easy signs).
* **Implemented Solution:** The `balance_processed_dataset.py` script was deployed to strictly cap max samples at 850, while `_BalancedAugSubset` oversampled minority classes. Inverse-frequency class weighting was applied to the Cross-Entropy loss.
* **Outcome:** The model learned to predict minority classes accurately without being overwhelmed by majority class biases.


* **Problem:** CPU performance bottlenecks.
* **Root Cause:** Heavy background person detection and continuous full-resolution tracking monopolized CPU threads.
* **Implemented Solution:** HOG person detection was explicitly disabled (`disable_hog_detection=True`) and MediaPipe executions were cached on 5-to-8 frame adaptive intervals.
* **Outcome:** Saved approximately 8ms per frame, ensuring a locked 30 FPS processing speed.



### 4.10 Current Status of the Module

The Sign-to-Text module is currently operating at a production-ready capability level. The system successfully processes 506-dimensional features through a 10-phase BiGRU architecture, accurately converting gestures into text. All major components—including the CVAE synthetic data generator, ONNX INT8 dynamic quantization pipeline, multi-phase K-fold training orchestrator, and real-time temporal momentum post-processor—have been successfully integrated and mathematically validated. The software handles 78 unique ISL classifications efficiently on standard CPU hardware.

### 4.11 Future Enhancements

While highly effective, the architectural foundation allows for massive future scalability.

* **Expansion beyond current classes:** The dataset pipeline is dynamically constructed, allowing immediate scaling to support 100+ and eventually 200+ ISL sign classes by simply generating new webcam NPY matrices.
* **Hierarchical classification approach:** As classes grow, implementing a two-stage classifier (grouping signs by physical location first, then by motion) could maintain high accuracy over massive vocabularies.
* **Sentence generation and context-aware prediction:** The current NLP output relies on rule-based processing; future implementations will integrate a lightweight Language Model (such as a quantized T5) to rescore probabilities based on semantic context.
* **Continuous sign recognition:** Implementing Connectionist Temporal Classification (CTC) or Hidden Markov Models (HMM) would allow for true continuous sequence segmentation without relying on isolated 20-frame discrete windows.
* **Edge deployment optimization:** The current 1.05 MB INT8 ONNX footprint is already primed for conversion to TensorFlow Lite (TFLite), paving the way for native mobile deployment on Android devices.
* **Multilingual text generation:** Hooking the final English text output into an on-device translation API would seamlessly enable the generation of Hindi or Marathi text overlays, further bridging communication gaps.

### 4.12 Conclusion

The Sign-to-Text module represents a highly optimized, fully functional engineering solution to visual gesture translation. Implementation achievements include the successful deployment of a 10-phase BiGRU architecture capable of decoding 506-dimensional spatiotemporal matrices at 30 frames per second on consumer hardware. Testing achievements verified the pipeline's 95.83% K-Fold accuracy and immunity to runtime crashes via dynamic fallback wrappers. Massive improvements post-Review 2—most notably the integration of face-relative normalization, ONNX INT8 quantization, and momentum-commit smoothing—transformed a volatile predictive script into a highly stable, latency-free desktop application. The module successfully fulfills its primary objective: acting as a reliable, real-time communication bridge for the Indian Sign Language community.




## 4. SIGN TO TEXT MODULE

### 4.1 Module Overview

The Sign-to-Text module constitutes the core predictive engine of the real-time Indian Sign Language (ISL) recognition system. The fundamental purpose of this module is to bridge the communication gap for the approximately 5 million individuals in the Indian deaf community by directly translating continuous spatial hand gestures into natural language text. Operating entirely on standard consumer-grade computing hardware without requiring specialized GPUs or depth cameras, this module functions as the critical translation layer within the complete system architecture.

The input to the module consists of live RGB video frames captured via a standard webcam at a resolution of 640×480 pixels operating at 30 frames per second. The output of the module is grammatically corrected, natural language text corresponding to 78 distinct ISL sign classes. The overall workflow involves capturing the video feed, extracting multi-dimensional spatial keypoints utilizing Google's MediaPipe framework, engineering scale-invariant spatiotemporal features, performing sequence classification over a rolling buffer using a custom multi-phase deep learning architecture, and finally applying rule-based temporal smoothing and linguistic post-processing to generate a stable text output.

### 4.2 System Architecture

The complete system architecture is designed as a high-throughput, low-latency pipeline to facilitate real-time inference. The pipeline begins with the webcam input, which continuously feeds RGB frames into the feature extraction subsystem.

To process the visual data, the architecture integrates the MediaPipe Tasks API, specifically utilizing two distinct models: the `hand_landmarker.task` for detailed anatomical hand parsing and the `face_landmarker.task` for spatial anchoring. Landmark extraction occurs at dynamically adjusted intervals to optimize processing speeds, outputting absolute coordinate values for 21 hand joints per hand, as well as specific facial anchor points.

Following extraction, the raw coordinates pass into the feature generation stage. The architecture does not rely on raw pixel data or basic absolute coordinates; instead, it processes 126 raw hand coordinate dimensions, 126 face-relative normalized dimensions, and a 1-dimensional hand-to-face proximity scalar. To capture motion dynamics, the system calculates frame-to-frame velocity deltas, resulting in a robust 506-dimensional feature vector per frame.

These frame-level features are accumulated into a fixed sequence formation consisting of 20 consecutive frames, creating a $20 \times 506$ spatiotemporal tensor matrix representing a single sign gesture. This tensor is fed into the deep learning inference engine, which utilizes a 10-phase Gated Recurrent Unit architecture featuring a 1D Convolutional frontend and a Spatial Graph Neural Network (GNN) branch to extract both local temporal patterns and spatial joint relationships.

The deep learning model outputs raw softmax probabilities across the 78 sign classes. The final text generation phase passes these probabilities through a Temporal Post-Processor that applies confidence-weighted smoothing, hysteresis thresholding, and a momentum-based commit logic before the `SentenceBuilder` and `nlp_postprocessor.py` modules compile the discrete classifications into grammatically structured text.

### 4.3 Data Acquisition and Dataset Preparation

#### 4.3.1 Sign Collection Process

A robust, webcam-based recording system was developed specifically for this project via the `collect_data.py` utility. This module enables the real-time capture of sign gestures through an interactive countdown interface, ensuring uniform sample lengths. To ensure generalization, multiple recordings per class were generated, ultimately yielding an active training set of approximately 5,683 processed samples distributed across 78 sign classes.

The recording strategy incorporated both controlled and uncontrolled environments. Samples were collected under varying lighting conditions and camera angles to simulate real-world usage. To artificially expand the dataset's robustness, an offline video augmentation pipeline (`augment_video_pipeline.py`) was implemented, generating up to 54 structural variants per original video by applying 17 distinct visual effects (including brightness, contrast, hue, fog, and pixel dropout) across 3 different crop positions.

#### 4.3.2 Landmark Extraction

The extraction mechanism intentionally utilizes the lightweight MediaPipe Tasks API rather than legacy solutions or computationally heavy pose estimation frameworks. The `hand_landmarker.task` model extracts 21 precise 3D anatomical joints for up to two hands. The `face_landmarker.task` extracts 478 points, but the module programmatically isolates only the nose center (index 1), left eye (index 33), and right eye (index 263).

Full-body pose landmarks were intentionally excluded from the extraction pipeline. This reduction of unnecessary features drastically improves computational efficiency, saving approximately 8 milliseconds per frame by disabling internal HOG person detection and bypassing heavy pose estimation networks. The finalized extraction results in approximately 506 features used per frame, combining raw spatial data, facial anchor normalization, and velocity calculations.

#### 4.3.3 Data Preprocessing

Data preprocessing ensures tensor integrity before deep learning inference. Missing landmark handling is explicitly managed; if the `_extract_face_anchor()` function fails to detect a face, or if a hand briefly leaves the camera view, the system gracefully falls back by zero-filling pre-allocated numpy buffers (`_LANDMARK_BUFFERS`), preventing pipeline crashes. Frame normalization is achieved by converting all spatial data relative to the bounding box dimensions. Sequence standardization forces all temporal clips into exactly 20-frame sequences, padding or truncating dynamically. Noise reduction occurs both spatially through MediaPipe's internal exponential moving average and temporally via the sequence dataset builder.

#### 4.3.4 Dataset Organization

The project directory follows a strict hierarchical folder structure. Raw MP4 files are stored in the `Dataset/` directory, organized into 78 class-wise subfolders. The `preprocess.py` module converts these raw videos into highly compressed NPY file generations, stored in the `processed/` directory. To ensure balanced training, the `balance_processed_dataset.py` script automatically manages dataset creation, replicating minority class samples or trimming majority classes to an exact target of 850 samples per class, creating a training-ready distribution.

### 4.4 Feature Engineering

#### 4.4.1 Landmark-Based Features

The foundational features rely on accurate hand landmark coordinates, providing 63 dimensions (21 landmarks $\times$ 3 coordinates) per hand, totaling 126 raw spatial dimensions for both hands. Facial landmark coordinates (nose and eyes) are extracted not for direct classification, but to serve as a topological anchor for the relative spatial information calculations.

#### 4.4.2 Relative Feature Generation

To decouple the model from the physical characteristics of the user, the module implements relative feature generation. Relative positioning between landmarks is computed by subtracting the nose coordinate from every hand landmark coordinate. This value is then divided by the inter-eye distance to normalize the scale. This face-relative normalization yields an additional 126 dimensions that provide improved robustness against camera placement, dramatically reducing the system's user-position dependency, allowing successful inference regardless of whether the user sits close to or far from the webcam.

#### 4.4.3 Temporal Representation

Because static poses cannot distinguish motion-dependent signs (e.g., distinguishing "come" from "go"), temporal representation is strictly enforced. The system computes frame-to-frame velocity by calculating the delta between frame $t$ and frame $t-1$ across the entire 253-dimensional base space. This adds 253 velocity features, explicitly capturing motion dynamics. The final multi-frame sequence creation buffers these 506-dimensional vectors into strict 20-frame sequences, allowing the temporal architecture to analyze the complete spatiotemporal trajectory of the gesture.

### 4.5 Deep Learning Model Development

#### 4.5.1 Model Selection

The Modified 10-Phase Gated Recurrent Unit (MOPGRU) architecture—implemented internally as the `SignLanguageGRU`—was selected specifically for its optimal balance between parameter efficiency and sequence modeling capabilities. Standard Transformers were rejected as their $O(n^2)$ self-attention complexity over 506 dimensions would create prohibitive computational overhead for real-time CPU deployment. Standard LSTMs were bypassed in favor of GRUs because the GRU's two-gate structure requires fewer parameters than the LSTM's three-gate structure, accelerating inference times without sacrificing accuracy over short 20-frame temporal windows.

#### 4.5.2 Network Architecture

The network architecture is a highly engineered multi-branch system. The input layer accepts the $(batch, 20, 506)$ tensor. The sequence processing immediately branches: the first 126 raw dimensions are routed to a Spatial Graph Neural Network (GNN) consisting of 2 Graph Convolutional Network (GCN) layers operating over the 21-node anatomical hand skeleton, outputting a 16-dimensional feature vector per frame. Concurrently, the full 506-dimensional input passes through a depthwise-separable 1D Convolutional frontend, compressing the features into 128 channels.

These branches are concatenated to 144 dimensions and passed through learnable sigmoid frame-weighting before entering the recurrent core. The bidirectional GRU layers consist of 3 stacked layers with a hidden dimension of 64, yielding a 128-dimensional temporal output. This output is analyzed by a Hybrid Multi-Head Attention module featuring 4 heads. Crucially, two of these heads apply a spatial proximity bias using the formula:
$\text{scores} += -\frac{\text{prox}^2}{2\sigma^2}$
where $\sigma$ is a learnable parameter initialized to 0.15. Finally, dense layers apply dropout (0.25) and project the 128-dimensional context vector through a 96-dimensional hidden layer before the final Softmax classification layer outputs the 78 class probabilities.

#### 4.5.3 Training Strategy

The training strategy is engineered to prevent overfitting and ensure robust generalization. The dataset split utilizes a K-fold Disjoint Stratified partition, utilizing `_build_disjoint_folds()` to create 5 perfectly distinct subsets. The model is trained for 50 epochs with a batch size of 8. Optimization is handled by the AdamW optimizer with a weight decay of 5e-4 and a baseline learning rate of 3e-4.

The loss function is a per-sample weighted Cross-Entropy loss. To handle minor class imbalances not caught by oversampling, an inverse-frequency class weighting formula is applied:
$w_c = \left(\frac{1}{n_c}\right)^{1.0}$
Early stopping is enforced with a patience of 10 epochs based on validation accuracy. Furthermore, a `ReduceLROnPlateau` scheduler dynamically halves the learning rate upon a 5-epoch plateau. Model checkpointing saves the best performing weights per fold to the `ensemble/` directory, registering metrics in a `kfold_manifest.json` tracker.

#### 4.5.4 Confidence-Based Prediction

Raw softmax outputs exhibit high variance during live inference; therefore, confidence-based prediction logic is strictly enforced. Confidence score generation relies on a dynamic threshold dynamically adjustable from a 0.12 baseline. Prediction filtering relies on a momentum-based commit logic, explicitly requiring a specific class to appear 3 times within a rolling 5-frame window (the 3-of-5 rule) while maintaining a minimum average confidence of 0.60. This drastically aids in the reduction of false positives caused by transient hand movements between legitimate signs.

### 4.6 Enhancements Implemented After Review 2

Following Review 2, the system underwent an extensive refactoring phase encompassing architectural improvements, real-time stability fixes, and deployment optimizations.

#### 4.6.1 Relative Feature Integration

The most critical mathematical enhancement was the introduction of the relative feature integration. Initially, raw MediaPipe coordinates caused severe misclassifications if the user shifted off-center. By introducing face-anchor subtraction and scaling based on inter-eye Euclidean distance, the resulting features became strictly scale and position invariant. The benefits obtained included a massive increase in recognition consistency across different camera distances and signer heights.

#### 4.6.2 Per-Class Threshold Optimization

A rigid global threshold caused similar signs to be misclassified. To resolve this, individual confidence thresholds were introduced via the `InferenceConfig` architecture. A specific `similar_class_penalty` of 0.08 was implemented. This forced the model to require an extra 8% confidence margin when distinguishing between highly visually similar signs defined in `similar_signs.json`. This resulted in improved class-specific performance and a marked reduction in misclassification between visually identical gestures that differ only in temporal velocity.

#### 4.6.3 Temporal Stability Improvements

Real-time prediction instability was heavily mitigated through prediction smoothing. The `TemporalPostProcessor` module was integrated, specifically leveraging a `ConfidenceSmoother` utilizing an 8-frame rolling deque. Older frames within this window are exponentially decayed using a factor of 0.3. For stable output generation, consecutive frame verification is enforced via a `StablePredictor` which requires 3 consecutive class agreements and a hysteresis delta of 0.12 to confirm a class switch.

#### 4.6.4 Transition Suppression Mechanism

A major flaw identified during Review 2 was the triggering of spurious word predictions while a user transitioned their hands from one sign to another. The prevention of incorrect predictions during sign transitions was achieved by implementing an ambiguity margin threshold. If the difference between the top-1 and top-2 class probabilities is less than 0.05, the `SentenceBuilder` initiates a strict 4-frame ambiguity delay. This transition suppression mechanism forces the model to wait until the user settles into a definitive pose, resulting in highly improved recognition reliability.

#### 4.6.5 Real-Time Pipeline Optimization

To guarantee a sub-200ms latency target, real-time pipeline optimization focused on deployment formats. The native PyTorch FP32 models were exported to the ONNX Runtime architecture (opset 18). Dynamic INT8 quantization was applied via `quantize_onnx.py`, yielding a 75% reduction in model size (from 4.2 MB to 1.05 MB). This resulted in faster inference and reduced latency by 2-3x on standard CPUs, ensuring a significantly better user experience devoid of video lag.

#### 4.6.6 Model Robustness Improvements

Handling motion variations was tackled by dynamically injecting temporal jitter and velocity recomputation during training. Handling user variations was addressed by implementing the 17-effect video augmentation pipeline and specific landmark-level augmentations such as face-anchor shifting and random hand-proportion scaling. Handling environmental changes is now inherently managed by the relative spatial calculation, which ignores lighting-based pixel variance entirely in favor of geometric topology.

### 4.7 Real-Time Recognition System Implementation

#### 4.7.1 Live Webcam Processing

Live webcam processing is orchestrated by the `webcam.py` module, which instantiates an OpenCV `VideoCapture` object targeted at 30 FPS. The module establishes a non-blocking UI thread that renders bounding boxes and assignment labels directly onto the live feed.

#### 4.7.2 Real-Time Feature Extraction

Real-time feature extraction relies on a cached extraction policy to maintain high frame rates. While frames enter at 30 FPS, the MediaPipe `hand_landmarker.task` only executes every 5 frames under normal motion, gracefully adapting to an 8-frame interval during low-motion periods to conserve CPU cycles. The `face_landmarker.task` is similarly cached, executing every 5 frames and forcing a hard refresh every 15 frames.

#### 4.7.3 Model Inference Pipeline

The model inference pipeline in production bypasses standard PyTorch loops, heavily utilizing the `ONNXModelWrapper`. The system extracts the $(20, 506)$ feature matrix and passes it to the INT8 quantized ONNX ensemble. The integration layer performs crucial dimension alignment, automatically padding or truncating feature tensors if runtime dimensions mismatch the exported ONNX expected shapes. If the ONNX session fails, the wrapper seamlessly executes a native PyTorch FP32 fallback.

#### 4.7.4 Text Generation Mechanism

The final raw predictions feed the text generation mechanism handled by `sentence_builder.py` and `nlp_postprocessor.py`. As signs pass the momentum commit logic, they are appended to an active string buffer. The rule-based NLP processor standardizes casing, manages spacing between discrete words, and handles punctuation cleanup, ultimately transforming a list of predicted tokens (e.g., `["hello", "how_are", "you"]`) into coherent natural language text.

#### 4.7.5 User Interface Integration

User interface integration consists of real-time overlay metrics drawn directly via OpenCV. The interface displays the current active sign prediction, the aggregated sentence string, the status of the two-hand validation check ("Same person: YES/NO"), and immediate bounding box visual feedback validating that the MediaPipe Tasks API has successfully acquired the user's spatial topology.

### 4.8 Testing and Validation

Extensive testing protocols were executed to validate system integrity from individual functions up to the entire real-time pipeline.

#### 4.8.1 Unit Testing

Unit testing prioritized internal tensor routing. Smoke checks and manual shape tracing were committed (Commit `ff6a57bb`) to mathematically verify the dimensionality of the Conv1D frontend concatenation with the Spatial GNN branch. Dataset generation unit tests explicitly validated the behavior of `_BalancedAugSubset`, discovering and successfully patching a critical tuple-unpacking crash where the architecture expected 2-tuples but received 3-tuples containing `(path, label, weight)` data. Model loading and prediction generation were verified using isolated dummy tensors initialized via `debug_model.py`.

#### 4.8.2 Integration Testing

Integration testing focused on end-to-end pipeline operability. End-to-end pipeline testing verified the complete data flow from the `collect_data.py` webcam capture through the `preprocess.py` buffer caching to the final `.npy` file generation. Data flow verification explicitly tested the mixed PyTorch and ONNX ensemble script (`onnx_ensemble.py`), confirming that predictions from both backends could be averaged effectively without array shape conflicts. Real-time performance testing utilized the `profiling.py` harness to confirm that the entire cycle—from frame capture to text output—stayed well below the 200ms latency ceiling.

#### 4.8.3 Functional Testing

Functional testing measured the core competency of the AI. Recognition accuracy hit a notable 95.83% mean accuracy on internal benchmark subsets using the K-Fold ensemble strategy. Correct text generation was verified by performing consecutive distinct gestures in front of the webcam and confirming that the `sentence_builder.py` correctly assembled the semantic intent. Class detection verification confirmed that all 78 target sign categories defined in `sign_categories.json` could be successfully triggered by a human user.

#### 4.8.4 Robustness Testing

System robustness testing explicitly sought to break the predictive engine under hostile conditions. Testing under different lighting proved that relying on MediaPipe's topological landmarks (with a low 0.5 confidence threshold for webcam) bypassed standard pixel-level shading vulnerabilities. Different backgrounds failed to confuse the system due to the exclusion of background segmentation in favor of skeletal mapping. Different distances from the camera were successfully mitigated by the inter-eye division embedded within the face-relative feature mathematics. Different hand speeds were normalized by the temporal 20-frame buffering, and different users were tested to ensure the models generalized beyond the primary dataset contributor.

#### 4.8.5 Performance Evaluation

The performance evaluation heavily highlights the success of the optimization phase. Inference speed was increased by 2-3x after replacing native PyTorch FP32 models with INT8 ONNX binaries. The overall response time allows for immediate conversational feedback. Real-time usability is confirmed by the system's ability to maintain a consistent 30 FPS processing loop on standard CPU hardware. Computational efficiency was further proven by the system's tiny memory footprint; the final compiled recognition model requires only 1.05 MB of disk space.

#### 4.8.6 Error Analysis

Comprehensive error analysis identified remaining edge cases. Similar sign confusion occurs when temporal trajectories overlap; this was largely solved by the 0.08 `similar_class_penalty`. Transition-related errors—where hands form accidental sign shapes while moving to resting positions—were effectively neutralized by the momentum-commit logic and the 4-frame ambiguity delay. Landmark detection failures in extreme low light cause graceful zero-buffer fallbacks rather than system crashes. Environmental limitations are strictly tied to the webcam's hardware exposure capabilities.

#### 4.8.7 Validation Results

The validation results confirm that the integration of the 10-phase BiGRU with spatial-temporal inputs achieves production-grade accuracy. By shifting the computational burden away from heavy Transformer models and into elegant feature engineering (face-relative normalization + velocity concatenation), the system definitively proves that high-accuracy, 78-class continuous gesture recognition is highly viable on standard, non-GPU computer architectures.

### 4.9 Challenges Encountered and Solutions Implemented

Several critical engineering challenges emerged during the development cycle, requiring sophisticated programmatic interventions.

* **Problem:** K-fold Training Crash with Weighted Samples.
* **Root Cause:** A system update modified the internal `ISLDataset` representation from a 2-tuple `(path, label)` to a 3-tuple `(path, label, weight)` to accommodate Phase 2 archived fine-tuning data. The older K-fold logic crashed attempting to unpack 3 values into 2 variables.
* **Implemented Solution:** A dedicated `_sample_label(sample)` helper function was engineered in `train.py` to safely extract index 1 regardless of the tuple length.
* **Outcome:** K-fold training resumed successfully, yielding 5 perfectly distinct model folds.


* **Problem:** Real-time prediction instability and UI jitter.
* **Root Cause:** Rapid hand movements and transition frames caused the raw softmax probabilities to wildly oscillate frame-to-frame.
* **Implemented Solution:** Implementation of a strict 3-of-5 momentum commit window requiring a minimum average confidence of 0.60, alongside an 8-frame exponential decay smoother.
* **Outcome:** Transient spikes were suppressed, completely eliminating false-positive word insertions.


* **Problem:** ONNX Dimension Mismatch during deployment.
* **Root Cause:** The input feature dimensions expanded from 253D to 506D when velocity tracking was introduced, causing older PyTorch models exported to ONNX to crash when receiving live 506D arrays.
* **Implemented Solution:** An intelligent multi-layer alignment protocol was embedded into `onnx_inference.py` to dynamically pad or truncate feature tensors and auto-align batch dimensions.
* **Outcome:** Seamless ONNX execution with a reliable automated fallback to native PyTorch.


* **Problem:** Dataset imbalance skewing predictions.
* **Root Cause:** Natural data collection yielded highly uneven class distributions (e.g., 50 samples for hard signs, 850 for easy signs).
* **Implemented Solution:** The `balance_processed_dataset.py` script was deployed to strictly cap max samples at 850, while `_BalancedAugSubset` oversampled minority classes. Inverse-frequency class weighting was applied to the Cross-Entropy loss.
* **Outcome:** The model learned to predict minority classes accurately without being overwhelmed by majority class biases.


* **Problem:** CPU performance bottlenecks.
* **Root Cause:** Heavy background person detection and continuous full-resolution tracking monopolized CPU threads.
* **Implemented Solution:** HOG person detection was explicitly disabled (`disable_hog_detection=True`) and MediaPipe executions were cached on 5-to-8 frame adaptive intervals.
* **Outcome:** Saved approximately 8ms per frame, ensuring a locked 30 FPS processing speed.



### 4.10 Current Status of the Module

The Sign-to-Text module is currently operating at a production-ready capability level. The system successfully processes 506-dimensional features through a 10-phase BiGRU architecture, accurately converting gestures into text. All major components—including the CVAE synthetic data generator, ONNX INT8 dynamic quantization pipeline, multi-phase K-fold training orchestrator, and real-time temporal momentum post-processor—have been successfully integrated and mathematically validated. The software handles 78 unique ISL classifications efficiently on standard CPU hardware.

### 4.11 Future Enhancements

While highly effective, the architectural foundation allows for massive future scalability.

* **Expansion beyond current classes:** The dataset pipeline is dynamically constructed, allowing immediate scaling to support 100+ and eventually 200+ ISL sign classes by simply generating new webcam NPY matrices.
* **Hierarchical classification approach:** As classes grow, implementing a two-stage classifier (grouping signs by physical location first, then by motion) could maintain high accuracy over massive vocabularies.
* **Sentence generation and context-aware prediction:** The current NLP output relies on rule-based processing; future implementations will integrate a lightweight Language Model (such as a quantized T5) to rescore probabilities based on semantic context.
* **Continuous sign recognition:** Implementing Connectionist Temporal Classification (CTC) or Hidden Markov Models (HMM) would allow for true continuous sequence segmentation without relying on isolated 20-frame discrete windows.
* **Edge deployment optimization:** The current 1.05 MB INT8 ONNX footprint is already primed for conversion to TensorFlow Lite (TFLite), paving the way for native mobile deployment on Android devices.
* **Multilingual text generation:** Hooking the final English text output into an on-device translation API would seamlessly enable the generation of Hindi or Marathi text overlays, further bridging communication gaps.

### 4.12 Conclusion

The Sign-to-Text module represents a highly optimized, fully functional engineering solution to visual gesture translation. Implementation achievements include the successful deployment of a 10-phase BiGRU architecture capable of decoding 506-dimensional spatiotemporal matrices at 30 frames per second on consumer hardware. Testing achievements verified the pipeline's 95.83% K-Fold accuracy and immunity to runtime crashes via dynamic fallback wrappers. Massive improvements post-Review 2—most notably the integration of face-relative normalization, ONNX INT8 quantization, and momentum-commit smoothing—transformed a volatile predictive script into a highly stable, latency-free desktop application. The module successfully fulfills its primary objective: acting as a reliable, real-time communication bridge for the Indian Sign Language community.
