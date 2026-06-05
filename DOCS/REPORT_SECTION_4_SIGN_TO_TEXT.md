# 4. SIGN TO TEXT MODULE

## 4.1 Module Overview

The Sign-to-Text module constitutes the core computational subsystem of the AI-Powered Indian Sign Language Recognition System. Its primary purpose is to translate isolated Indian Sign Language (ISL) word gestures, captured via a standard RGB webcam, into corresponding English text strings in real time. Unlike conventional sign language recognition approaches that rely on depth cameras or body-worn sensors, this module operates entirely on two-dimensional colour video frames captured from a consumer-grade webcam, making it hardware-accessible and practically deployable.

Within the complete system, the Sign-to-Text module functions as the primary perception and classification layer. It receives raw video frames from the webcam subsystem, performs multi-stage landmark extraction using the MediaPipe Tasks API, constructs 506-dimensional spatiotemporal feature vectors from extracted landmarks, and passes sequences of twenty such frames through a trained Bidirectional Gated Recurrent Unit (BiGRU) deep learning classifier. The output of the module â€” a predicted sign label and an associated confidence score â€” is consumed by the Sentence Builder and Natural Language Processing (NLP) post-processor to produce grammatically cleaned text output.

**Module Inputs:**
- Live RGB video frames at 640 Ã— 480 pixels, 30 frames per second, from a USB webcam.
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

### Stage 1 â€” Webcam Capture

The OpenCV `VideoCapture` interface captures frames at 640 Ã— 480 pixels from the system webcam at approximately 30 frames per second. To maintain consistency between training data and live inference, all frames are centre-cropped to the webcam target resolution. This ensures that the spatial coordinate space experienced by MediaPipe during inference exactly matches that during preprocessing. The HOG-based person detection layer is explicitly disabled (via `disable_hog_detection: bool = True` in `config.py`) to save approximately 8 milliseconds per frame.

### Stage 2 â€” Adaptive MediaPipe Landmark Detection

To achieve real-time throughput without running MediaPipe on every frame, an adaptive detection interval mechanism is employed. Hand landmarks are detected every 5 frames by default, with cached results reused between detections. A forced re-detection is triggered every 15 frames regardless of motion state, preventing stale landmark tracking from persisting. When motion magnitude is below 60% of the configured threshold, the hand detection interval is extended up to a maximum of 8 frames, further reducing computational load during static sign holds. Face landmarks are similarly detected at a 5-frame interval, with the results cached between frames.

### Stage 3 â€” Feature Vector Construction

For each processed frame, a 253-dimensional base feature vector is constructed: 126 dimensions of raw hand landmark coordinates (both hands, 21 landmarks Ã— 3 coordinates Ã— 2 hands), 126 dimensions of face-relative hand coordinates (same structure, normalised relative to the face anchor), and 1 scalar proximity dimension encoding the L2 distance from the hand centroid to the nose tip. Frame-to-frame velocity features (253 dimensions) are appended, yielding a final per-frame vector of **506 dimensions**.

### Stage 4 â€” Sequence Buffering

Twenty consecutive feature vectors (20 frames) are accumulated in a fixed-length circular buffer, producing an input tensor of shape `(20, 506)`. This 20-frame window corresponds to approximately 667 milliseconds at 30 FPS, capturing the full temporal extent of most ISL word gestures.

### Stage 5 â€” Deep Learning Inference

The buffered sequence is passed to the `SignLanguageGRU` model. Primary inference uses the ONNX Runtime (ORT) with INT8 quantisation, providing 2â€“3Ã— faster inference than the PyTorch FP32 path. If the ONNX session raises a dimension or runtime error, the system falls back to PyTorch FP32 inference automatically. The model outputs raw logits over 78 classes, which are converted to softmax probabilities.

### Stage 6 â€” Temporal Post-Processing

The raw per-frame probability vector is processed by the `TemporalPostProcessor`, which combines a `ConfidenceSmoother` (sliding window of 8 frames with confidence weighting and exponential decay factor of 0.3) and a `StablePredictor` (requires 3 consecutive frames voting for the same class with a minimum confidence margin of 0.12 before switching). This two-layer mechanism significantly reduces prediction jitter without introducing excessive latency.

### Stage 7 â€” Momentum-Based Commit

A sign word is committed to the output sentence only when the predicted class appears at least 3 times in the most recent 5 predictions (a "3-of-5 majority window") and the average confidence across those occurrences equals or exceeds 0.60. This prevents transient or low-confidence predictions from being mistakenly appended to the output sentence.

### Stage 8 â€” Text Generation

Committed sign labels are passed to the `SentenceBuilder`, which applies an ambiguity delay of 4 additional frames when the margin between the top-1 and top-2 predicted class probabilities is less than 0.05. The assembled sentence is subsequently cleaned by `nlp_postprocessor.py` for grammar and punctuation normalisation.

---

## 4.3 Data Acquisition and Dataset Preparation

### 4.3.1 Sign Collection Process

A custom webcam data collection tool, `collect_data.py`, was developed to standardise the recording process across all 78 sign classes. The tool provides a countdown of 3 seconds before each recording begins, allowing the signer to prepare. For each sign class, 90 raw frames are captured using the OpenCV `VideoCapture` interface at 640 Ã— 480 pixels. These 90 raw frames are subsequently sub-sampled to 20 evenly spaced frames during preprocessing, ensuring temporal consistency across all recordings.

Recordings were conducted in both controlled and uncontrolled environments. Controlled recordings used a fixed-distance position from the camera under consistent indoor lighting. Uncontrolled recordings deliberately introduced variation in lighting temperature (fluorescent, incandescent, and natural daylight), background complexity (plain walls, cluttered rooms), and signer-to-camera distance. This diversity was intentional: a model trained exclusively on controlled data generalises poorly to real-world conditions where users' environments vary significantly.

Multiple recordings per class were collected to increase sample diversity. The dataset reached approximately 5,683 processed `.npy` sequences across 78 sign classes, as evidenced by the commit `74677292` titled "Add processed landmark sequences (5,683 .npy files)".

### 4.3.2 Landmark Extraction

Landmark extraction is implemented in `preprocess.py` using the **MediaPipe Tasks API** â€” specifically the `HandLandmarker` and `FaceLandmarker` task models. The Tasks API was chosen over the legacy MediaPipe Solutions API for three reasons: improved accuracy on partial occlusion cases, better forward compatibility with future MediaPipe releases, and explicit separation of image-mode and video-mode inference that aligns with the training versus live inference distinction in this system.

**Hand Landmark Extraction:** The `hand_landmarker.task` model (7.8 MB) detects up to 2 hands per frame, extracting 21 landmarks per hand in normalised (x, y, z) coordinates. A minimum hand detection confidence of 0.3 is used during preprocessing, raised to 0.5 during live webcam inference to reduce false positive detections.

**Face Landmark Extraction:** The `face_landmarker.task` model (3.8 MB) extracts 478 facial landmarks. From these, only three indices are used: nose tip (index 1), left eye outer corner (index 33), and right eye outer corner (index 263). These three points are sufficient to define a face anchor (the nose tip as origin) and a spatial scale factor (the inter-eye Euclidean distance), enabling position- and scale-invariant landmark normalisation.

**Intentional Exclusion of Pose Landmarks:** Full body pose landmarks (MediaPipe Pose, producing 33 body landmarks) were deliberately excluded from the feature set. ISL word-level recognition depends on hand configuration and hand position relative to the face â€” shoulder and trunk landmarks contribute minimal discriminative information for isolated word recognition while adding 99 dimensions of noise (33 Ã— 3 coordinates). Their exclusion reduces the feature dimension, improves computational efficiency, and decreases the risk of overfitting to spurious body-pose correlations.

The extracted feature set therefore comprises:
- **Raw hand landmarks:** 21 landmarks Ã— 3 coords Ã— 2 hands = **126 dimensions**
- **Face-relative hand landmarks:** 21 landmarks Ã— 3 coords Ã— 2 hands = **126 dimensions**
- **Hand-to-face proximity scalar:** **1 dimension**
- **Total base features:** **253 dimensions per frame**
- **With velocity (frame-to-frame delta):** **506 dimensions per frame**

### 4.3.3 Data Preprocessing

Preprocessing is implemented in `preprocess.py` and `dataset.py` and encompasses the following steps:

**Missing Landmark Handling:** When a hand is not detected in a given frame â€” due to partial occlusion, motion blur, or detection failure â€” the corresponding 63-dimensional raw block and 63-dimensional face-relative block are filled with zeros. This zero-filling strategy was chosen over interpolation because the absence of a hand is itself a meaningful signal (it indicates the hand is not in the field of view or is not performing a gesture). The model learns to treat zero-filled frames accordingly.

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

The 126-dimensional raw hand feature block concatenates the 21 MediaPipe hand landmarks of both the left and right hands, each expressed as a normalised (x, y, z) triplet in the range [0, 1] relative to the frame dimensions. These raw coordinates preserve absolute hand position information within the frame. The choice to retain both hands â€” rather than only the dominant hand â€” enables the model to distinguish signs that involve different relative positions or configurations of both hands simultaneously.

The 126-dimensional face-relative feature block expresses the same 21 landmarks per hand in face-anchored coordinates. Specifically, each landmark coordinate is transformed as:

```
relative_coord[i] = (hand_lm[i] - nose_tip) / inter_eye_distance
```

where `nose_tip` is the (x, y, z) coordinate of face landmark index 1 and `inter_eye_distance` is the Euclidean distance between face landmarks index 33 (left eye) and index 263 (right eye). The nose tip serves as the origin, and the inter-eye distance normalises for the physical scale of the signer's face in the frame.

### 4.4.2 Relative Feature Generation

The decision to include face-relative coordinates as a distinct feature block (rather than replacing raw coordinates) was motivated by their complementary nature. Raw coordinates encode where in the frame the hands are located â€” useful when combined with face landmarks to infer spatial relationships. Face-relative coordinates encode where the hands are with respect to the signer's face, which is the primary spatial cue in ISL: most ISL signs are defined by the position and configuration of the hands relative to specific facial regions (near the mouth, at the forehead, at the chin, extended in front of the face, etc.).

The face-relative representation confers three practical benefits. First, it is invariant to the absolute position of the signer within the camera frame, so the same sign performed by a signer sitting close to or far from the camera produces similar face-relative values. Second, it is invariant to the signer's stature, since the inter-eye distance scales proportionally with the face's apparent size in the frame. Third, it enables the model's attention mechanism to apply a physically meaningful spatial bias: the Gaussian proximity kernel in the HybridAttention module uses the proximity scalar â€” which is derived from face-relative distances â€” to upweight frames where the hands are near the face, which are typically the most informative frames for discriminating ISL signs.

### 4.4.3 Temporal Representation

A single frame's landmark configuration is insufficient to distinguish many ISL signs. Motion trajectory, speed, and directional change are all critical discriminators. The temporal representation is addressed at two levels.

**Multi-Frame Sequence:** Twenty consecutive frames are buffered and processed as a unit. The 20-frame window was selected based on empirical observation of sign durations in the collected dataset: the majority of ISL words are completed within approximately 0.5 to 0.8 seconds, corresponding to 15 to 24 frames at 30 FPS. A 20-frame window (667 ms) captures the complete gesture while remaining short enough to keep sequence processing computationally tractable.

**Velocity Features:** Frame-to-frame finite differences of all 253 base features are appended to form the 506-dimensional input. The velocity block at frame $t$ is computed as $v_t = f_t - f_{t-1}$, where $f_t$ denotes the base feature vector at time $t$. At frame 0 (the first frame), the velocity block is set to zero. Velocity features explicitly encode motion direction and speed, enabling the model to distinguish signs that share similar peak-frame handshapes but differ in their approach trajectory.

---

## 4.5 Deep Learning Model Development

### 4.5.1 Model Selection

The `SignLanguageGRU` architecture â€” a multi-phase Bidirectional Gated Recurrent Unit with convolutional and graph neural network frontends â€” was selected based on three constraints specific to this project:

1. **CPU-only deployment requirement:** The system must operate on a standard laptop or desktop CPU without requiring a GPU. Transformer-based architectures, despite superior accuracy on large datasets, incur $O(n^2)$ self-attention complexity over the sequence length and require substantially more memory bandwidth, making them unsuitable for real-time CPU inference. The GRU's $O(n)$ sequential computation with moderate hidden dimensions is far more tractable.

2. **Short sequence length (20 frames):** The bidirectional GRU is well-suited to sequences of this length. The full 20-frame context is available at inference time, so bidirectionality (reading the sequence in both forward and backward directions) is not computationally prohibitive.

3. **Limited training data:** With approximately 5,683 samples across 78 classes (averaging ~73 samples per class before augmentation), large-parameter Transformer models would overfit severely. The GRU's parameter efficiency â€” especially when combined with the Conv1D frontend and Spatial GNN â€” provides sufficient representational capacity without overfitting.

The LSTM architecture was also evaluated. The GRU was preferred because it employs two gating mechanisms (update and reset gates) rather than three (input, forget, output), yielding fewer parameters with comparable performance on short-sequence gesture classification tasks.

### 4.5.2 Network Architecture

The `SignLanguageGRU` model implements 10 independently configurable architectural improvements, all enabled by default in production. The data flow is as follows:

**Input:** Tensor of shape `(batch, 20, 506)` â€” batch Ã— 20 frames Ã— 506 features.

**Phase 10 â€” Spatial GNN Branch (`spatial_gnn.py`):**
The first 126 dimensions (raw hand landmark coordinates for both hands) are passed through a `LightweightSpatialGNN`, a 2-layer Graph Convolutional Network operating over the anatomical hand skeleton graph (21 nodes per hand, with edges corresponding to known metacarpal-proximal-middle-distal finger joint connections). The GCN produces 8-dimensional pooled representations per hand (global max-pooling over 21 nodes), concatenated across both hands to yield **16 dimensions per frame**. This GNN branch runs in parallel with the Conv1D frontend.

**Phase 1 â€” Conv1D Frontend:**
All 506 dimensions are passed through a depthwise-separable 1D convolutional frontend: a pointwise convolution reducing 506 channels to 128, followed by a depthwise temporal convolution (kernel size 3, padding 1, grouped by channel) with a residual connection, and a GroupNorm (8 groups) followed by ReLU and dropout (0.1). This frontend extracts short-range temporal patterns across the 20-frame sequence while reducing input dimensionality. The output is of shape `(batch, 20, 128)`.

**Concatenation:** The 16-dimensional GNN output per frame is concatenated with the 128-dimensional Conv1D output to produce **144 dimensions per frame**.

**Phase 2 â€” Learnable Frame Weighting:**
A small MLP (`Linear(144â†’32)â†’ReLUâ†’Linearâ†’Sigmoid`) produces a scalar importance weight per frame, applied as an element-wise multiplicative mask. This allows the model to soft-suppress uninformative frames (e.g., transition frames between signs) while amplifying informative frames (sign onset and peak).

**Input Projection:** A `Linear(144â†’64)` layer followed by LayerNorm(64) and ReLU projects the combined features into the GRU input space.

**Phase 4 â€” Bidirectional GRU:**
Three stacked bidirectional GRU layers with hidden dimension 64 per direction (128-dimensional concatenated output). Inter-layer dropout is 0.30 (reduced from 0.35 as part of Phase 4 refinements). The output is of shape `(batch, 20, 128)`, followed by a LayerNorm.

**HybridAttention (4 heads):**
Two of the four attention heads are standard temporal attention heads (learning which frames carry the most information). The remaining two heads are proximity-aware: their attention scores are additively biased by the Gaussian proximity log-probability $\log \mathcal{N}(\text{prox}; 0, \sigma^2) = -\text{prox}^2 / (2\sigma^2)$, where $\sigma = 0.15$ is a learnable parameter. Each head also has an independent learnable temperature clamped to $[0.1, 10.0]$, controlling the sharpness of its softmax distribution. The four head outputs (each 32-dimensional) are concatenated into the 128-dimensional context vector.

**Residual Skips (Phases 5 and 9):** The temporal mean of the GRU output is added to the attention context (Phase 9 residual). Additionally, the temporal mean of the input projection is added to the context if dimensions align (Phase 5 residual). These skip connections improve gradient flow and training convergence.

**FC Classification Head:** `Dropout(0.25) â†’ Linear(128â†’96) â†’ ReLU â†’ Dropout â†’ Linear(96â†’78)` producing raw logits over 78 classes.

### 4.5.3 Training Strategy

The model is trained using the `train.py` module. The training configuration is centralised in `config.py` as a validated dataclass (`TrainingConfig`), with `CONFIG_VERSION = "2.0.0"`.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Batch size | 8 | Small batches suited to limited per-class sample counts |
| Learning rate | 3 Ã— 10â»â´ | Reduced from 5 Ã— 10â»â´ for improved stability with small datasets |
| Weight decay | 5 Ã— 10â»â´ | L2 regularisation to prevent overfitting |
| Gradient clipping | 1.0 | Prevents gradient explosion in deep recurrent networks |
| Epochs | 50 | Sufficient convergence for 78-class problem |
| Early stopping patience | 10 | Terminates training if validation accuracy does not improve for 10 epochs |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 5) | Halves LR when validation accuracy plateaus |
| Validation split | 70 / 30 (stratified) | Disjoint per-class splits via `_disjoint_stratified_split()` |
| Loss function | CrossEntropyLoss (per-sample, reduction='none') Ã— per-sample weight | Enables differential weighting of archived vs. primary samples |
| Label smoothing | 0.05 | Prevents over-confident predictions on ambiguous classes |
| Class weighting | Inverse frequency, power 1.0, normalised to mean = 1 | Compensates for residual class imbalance after oversampling |
| Mixup augmentation | Î± = 0.3, applied with probability 0.5 | Creates virtual training samples between classes; improves generalisation |
| K-fold cross-validation | 5 folds, disjoint stratified | Full ensemble of 5 models for improved accuracy |

**Two-Phase Training:** Phase 1 trains exclusively on curated data from `processed/`. Phase 2 fine-tunes by adding samples from `processed_del/` (previously archived data) at a reduced sample weight of 0.25, preventing lower-quality archived samples from dominating gradient updates.

**Model Checkpointing:** The best-performing checkpoint per fold (highest validation accuracy) is saved to `model.pth` (single model) or `ensemble/fold_{n}.pth` (K-fold). The K-fold training manifest, saved to `ensemble/kfold_manifest.json`, records per-fold accuracy, checkpoint path, and completion timestamp.

### 4.5.4 Confidence-Based Prediction

During inference, the softmax of the model's logits produces a probability vector over all 78 sign classes. The maximum probability value constitutes the confidence score. A base confidence threshold of 0.12 was established empirically: the ensemble output distribution was observed to concentrate in the 0.1â€“0.2 range for correct predictions in ambiguous scenarios, and a threshold at 0.12 preserves sensitivity while filtering clear non-detections. An additional penalty of 0.08 is applied to known similar-class pairs (`similar_class_penalty`) to reduce the risk of confusing visually similar signs. Predictions falling below the composite threshold are discarded, and the frame is treated as idle. This multi-threshold approach substantially reduces false positive word commits compared to a single global threshold.

---

## 4.6 Enhancements Implemented After Review 2

The following enhancements were implemented iteratively after Review 2, forming the principal technical contributions of the latter development phase (Marchâ€“June 2026).

### 4.6.1 Relative Feature Integration

Prior to this enhancement, only raw hand landmark coordinates (126 dimensions) were used as input features. The face-relative coordinate block (an additional 126 dimensions) was integrated following the analysis that raw coordinates are inherently signer-position-dependent. A signer positioned at the left edge of the frame produces systematically different raw coordinate values than the same sign performed by the same signer at the centre of the frame, despite the underlying gesture being identical. By expressing hand positions relative to the face anchor â€” normalised by inter-eye distance â€” the feature representation becomes invariant to both the signer's position within the frame and the apparent scale of their face due to camera distance. This directly improves generalisation to unseen signers and recording environments. The implementation in `preprocess.py` `compute_face_relative_features()` was introduced in commit `15dcdfd6` (February 28, 2026) and enhanced with face-anchor shift augmentation in commit `c9771af2`.

### 4.6.2 Per-Class Threshold Optimisation

The initial system employed a single global confidence threshold. Analysis of per-class error patterns revealed that certain sign pairs (e.g., signs involving similar handshapes in proximal facial regions) consistently produced low but non-trivial confidence scores, leading to misclassification. A `similar_class_penalty` parameter (value 0.08 in `config.py` `InferenceConfig`) was introduced to apply an elevated effective threshold to sign pairs identified as visually similar. This does not require retraining: the penalty is applied at inference time by augmenting the base threshold for specific class pairs, effectively requiring higher certainty before committing visually ambiguous predictions. This targeted approach improved per-class precision for the most frequently confused class pairs without degrading recognition speed on well-separated classes.

### 4.6.3 Temporal Stability Improvements

The initial inference pipeline committed a sign as soon as the model's argmax prediction changed, resulting in highly unstable output: a single outlier frame could interrupt a correct sign detection mid-gesture or insert spurious words. The `TemporalPostProcessor` (implemented in `temporal_postprocessor.py`, integrated in commit `a63d818`) addresses this through a two-stage pipeline:

The `ConfidenceSmoother` maintains a sliding window deque of the 8 most recent probability vectors. Each entry is weighted by its confidence score (the maximum softmax probability) multiplied by an exponential decay factor of 0.3 applied to older entries, so that more recent frames carry proportionally greater influence. The weighted average is renormalised to produce a smoothed probability distribution.

The `StablePredictor` operates on the smoothed output. It maintains a candidate class and a patience counter: the candidate class must be predicted for 3 consecutive frames, and its smoothed confidence must exceed that of the current stable class by at least 0.12 (the hysteresis margin), before a class switch is confirmed. This patience-plus-hysteresis mechanism eliminates single-frame transient switches while adapting quickly to genuine sign changes.

### 4.6.4 Transition Suppression Mechanism

During natural signing, the hand transitions between signs â€” a period of motion during which landmark configurations do not correspond to any well-defined sign. Without suppression, the model confidently misclassifies transition frames, inserting spurious words into the output sentence. The momentum-based commit logic addresses this: a sign is only committed when it appears in at least 3 of the 5 most recent stable predictions and the average confidence across those occurrences is at least 0.60. Because transition frames typically produce low-confidence, inconsistent predictions across the 5-frame window, they rarely achieve 3-of-5 majority. Additionally, an ambiguity delay of 4 frames is imposed when the margin between the top-1 and top-2 softmax probabilities is less than 0.05, providing additional suppression during uncertain moments. The `sign_idle_timeout` of 30 frames (approximately 1 second at 30 FPS) resets the sentence builder when hands are absent, preventing stale predictions from propagating.

### 4.6.5 Real-Time Pipeline Optimisation

Multiple targeted optimisations were implemented to bring the end-to-end latency within the sub-200 ms target:

**Detection interval caching:** MediaPipe hand and face detection run every 5 frames (adaptive up to 8 during low-motion periods), with cached landmarks reused between detection frames. Landmark re-use reduces the per-frame MediaPipe overhead from approximately 30â€“40 ms to under 5 ms on cached frames.

**HOG detection disabled:** The HOG-based person-presence check was disabled (`disable_hog_detection: bool = True`), saving approximately 8 ms per frame without meaningful accuracy loss, since the MediaPipe face landmarker already serves as the primary anchor.

**Module-level buffer cache:** `preprocess.py` allocates fixed NumPy buffers (`_LANDMARK_BUFFERS`) at module load time for `left_raw`, `right_raw`, `left_rel`, and `right_rel`. These buffers are reset via `.fill(0)` and reused in-place each frame, reducing per-frame NumPy allocation overhead from approximately 12 array allocations to approximately 1 (the final concatenation). This was implemented as the "Phase 1 Optimization" noted in `preprocess.py`.

**ONNX INT8 inference:** The trained PyTorch model is exported to ONNX format using `export_onnx.py` (opset 18, dynamic batch size) and quantised to INT8 via `quantize_onnx.py` using dynamic quantisation (`onnxruntime.quantization.quantize_dynamic`). The resulting INT8 model is approximately 1.05 MB (reduced from approximately 4.2 MB FP32), and runs 2â€“3Ã— faster on a CPU than the PyTorch FP32 path.

### 4.6.6 Model Robustness Improvements

**Handling Motion Variation:** Eight distinct online augmentation operations are applied during training in `ISLDataset._augment()`: (1) Gaussian noise injection (Ïƒ = 0.015, 70% probability); (2) random uniform scaling (0.88â€“1.12Ã—, 60% probability); (3) temporal frame shift via circular roll (âˆ’3 to +3 frames, 50% probability); (4) random frame dropout (1â€“3 frames zeroed, 30% probability); (5) XY-plane rotation of raw landmark blocks (âˆ’15Â° to +15Â°, 40% probability); (6) time warping by resampling the 20-frame sequence at 0.75Ã—â€“1.25Ã— speed (40% probability); (7) per-hand dropout (up to one-third of frames for a randomly selected hand, 20% probability); and (8) stronger localised noise on a random subset of frames (25% probability).

**Handling User Variation:** Video-level augmentation in `preprocess.py` applies up to 54 distinct photometric and geometric transformations to each source video before landmark extraction: 17 visual effects (brightness, contrast, hue shift, fog, rotation, scale, colour jitter, Gaussian noise, pixel dropout, coarse dropout, motion blur, defocus blur, JPEG artefact compression, gamma correction, white balance shift, perspective warp, temporal jitter) combined with 3 crop positions (centre, left-offset at 15%, right-offset at 85%), yielding up to 54 augmented variants per original video. Additionally, `augmentations.py` implements face-anchor shift augmentation (random translation of the face reference point to simulate signer repositioning) and hand-proportion simulation (random per-finger scale factors to simulate different hand sizes), both applied at the landmark sequence level.

**Handling Environmental Changes:** The MediaPipe confidence threshold is set lower during training-data extraction (0.3) than during live webcam inference (0.5), ensuring that the training data includes some lower-confidence detections representative of challenging environments, while live inference applies stricter filtering to reduce false positive detections under good lighting. The face-relative normalisation further decouples the feature representation from ambient lighting and background changes, since it is based on relative spatial ratios rather than absolute pixel intensities.

### 4.6.7 CVAE-Based Synthetic Data Generation and Quality Filtering

**Generative Landmark Modeling:** To resolve class imbalance in the training data, a Conditional Variational Autoencoder is deployed to synthesize realistic landmark sequences for minority classes. Implemented in [cvae_landmarks.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/cvae_landmarks.py), the [LandmarkCVAE](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/cvae_landmarks.py#L232) features a Bidirectional GRU encoder and a GRU decoder. The training process, managed by [train_cvae.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/train_cvae.py), uses a multi-task loss combining coordinate reconstruction MSE, KL divergence, and a velocity consistency loss that penalizes frame-to-frame joint jitter. The synthesis module [generate_cvae_samples.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/generate_cvae_samples.py) extracts class-wise cluster means and standard deviations from the latent space of real samples, samples latent codes $z$, and decodes them at a controlled temperature before saving.

**Neural Quality Filtering:** To prevent anomalous synthetic sequences from entering the dataset, a lightweight, Bidirectional GRU-based realism classifier is defined in [quality_discriminator.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/quality_discriminator.py) and trained via [train_quality_discriminator.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/train_quality_discriminator.py). The training includes a **hard negative mining** loop that dynamically extracts synthetic sequences scoring high on realism and feeds them back into training for adversarial fine-tuning. Traditional heuristics evaluate motion variance, active joint ratios, and maximum frame drift to reject frozen or disjointed generations.

**Augmentation and Splicing Merge Pipelines:** Landmark-level transformations are handled by [augmentations.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/augmentations.py), providing 20 deterministic operations including 3D rotation, temporal speed warping, and sensor dropouts, coordinated via [augment_pipeline.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/augment_pipeline.py). Additionally, [merge_augmentations.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/merge_augmentations.py) executes frame-splicing merges between different recordings of the same class (using crossfade ramps, hand swapping, and tempo-aligned time warping) to simulate diverse signing styles and hand configurations.

**Diversity Cleanup and Balancing:** Near-duplicate sequences are filtered in [cleanup_dataset_npy.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/cleanup_dataset_npy.py) using L2-normalized vector distance thresholding. The remaining augmented and merged samples are pruned to the target subset size using a greedy Farthest Point Sampling (FPS) algorithm to maximize cluster coverage. Finally, [balance_processed_dataset.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/balance_processed_dataset.py) and [random_downsample_processed.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/random_downsample_processed.py) normalize the class distributions, protecting original webcam captures from deletion.

### 4.6.8 User-Specific Live Asynchronous Adaptation

To customize recognition behavior for individual signers without corrupting the baseline ensemble models, a real-time output adapter is implemented.

**Residual Log-Probability Adapter:** Implemented in [adapter_model.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/adapter_model.py), the [AdapterModel](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/adapter_model.py#L24) takes the base ensemble probability vector, converts it to log-probability (logit) space, and passes it through a dense MLP (Dense(128) -> ReLU -> Dense(num_classes)) with a residual skip connection:
$$
\text{adapted\_logits} = \log(\text{probs} + \epsilon) + \delta
$$
This structure stabilizes learning by predicting corrective deltas $\delta$ rather than absolute logits.

**Asynchronous Background Training:** Real-time adaptation must not interrupt the primary webcam frame capture or classification loops. As implemented in [adapter_training.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/adapter_training.py), the [AdapterTrainingManager](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/adapter_training.py#L19) spawns a background thread to handle training and optimization routines, preserving webcam frame rates and UI responsiveness.

**Data Preparation and Gradient Weighting:** The manager enforces class balance during pseudo-label accumulation by downsampling data to the minimum representation count across active classes, skipping training if class imbalance exceeds a set ratio. Furthermore, it computes normalized, clipped inverse-frequency class weights to adjust CrossEntropyLoss updates, preventing skewed pseudo-label distributions from biasing the adapter.

**Validation and Rollback Hysteresis:** To guarantee system stability, the manager executes a performance check before committing new weights. It measures average prediction confidence before and after adaptation; if adaptation degrades baseline confidence by more than a threshold margin (0.05), the manager discards the new weights and rolls back to a secure weight backup, ensuring the system never enters an degraded state.

---

## 4.7 Real-Time Recognition System Implementation

### 4.7.1 Live Webcam Processing

The `webcam.py` module manages the live inference loop. OpenCV's `VideoCapture` interface opens the default webcam device and reads frames at the native capture rate (target: 30 FPS). Each frame is immediately centre-cropped to 640 Ã— 480 pixels to match the preprocessing geometry. Frame timing is monitored: if frame acquisition falls below 25 FPS, a warning is logged via `pipeline_logger.py` to assist in latency diagnosis. The main capture loop is structured as a producer-consumer pattern: landmark detection and feature extraction run synchronously within the loop, while text display updates are performed asynchronously to the OpenCV display window.

### 4.7.2 Real-Time Feature Extraction

Within the webcam loop, hand landmark detection is gated by the adaptive interval logic: a counter tracks the number of frames since the last full detection, running MediaPipe only when the counter exceeds the current adaptive interval (nominally 5, extended to up to 8 during low-motion periods). Cached hand landmarks are used for intermediate frames. Face landmark detection follows the same 5-frame interval, with a hard forced re-detection every 15 frames. When valid hand and face landmarks are available, `extract_landmarks_with_face_relative()` computes the 253-dimensional base feature vector using the pre-allocated buffer cache, computes the proximity scalar, and appends the velocity delta from the previous frame, yielding the 506-dimensional frame feature vector.

### 4.7.3 Model Inference Pipeline

The 20-frame buffer is managed by a `collections.deque(maxlen=20)`. Once the deque reaches full capacity, inference is triggered on every new frame (a sliding window approach). The feature buffer is converted to a NumPy array of shape `(1, 20, 506)` and passed to the `ONNXModelWrapper` in `onnx_inference.py`. The wrapper handles: (1) feature dimension alignment (pad or truncate if the current feature dim differs from the model's expected input); (2) proximity vector rank adjustment (scalar â†’ 2D tensor for batch dimension); (3) batch axis addition if required; and (4) ONNX Runtime session invocation. On failure, it falls back to the PyTorch model. The returned logits are passed through softmax to produce class probabilities.

### 4.7.4 Text Generation Mechanism

The `SentenceBuilder` class in `sentence_builder.py` maintains the current accumulated sentence as a list of committed sign labels. Sign labels are appended only when the momentum commit condition is met (3-of-5 majority window, minimum average confidence 0.60) and the new label differs from the last committed label (preventing immediate repeated word appends). An ambiguity delay of 4 additional frames is applied when the top-1 minus top-2 softmax probability is less than 0.05, requiring stronger evidence before committing visually ambiguous predictions. The `nlp_postprocessor.py` module applies rule-based post-processing: capitalisation of the first word, insertion of grammatical connectors where inferred, and punctuation normalisation. The cleaned sentence string is returned for display.

### 4.7.5 User Interface Integration

The OpenCV-based display window renders the live webcam feed with real-time visual overlays: detected hand landmark skeleton drawn on the frame, the current predicted sign label and confidence score displayed in the upper-left corner, and the accumulated sentence string displayed at the bottom of the frame. A colour-coded confidence bar provides immediate visual feedback on recognition certainty. The `app.py` module provides a keyboard interface: pressing 'U' undoes the last committed word (pop from sentence list), pressing 'C' clears the entire sentence, and pressing 'Q' exits the application. Preset phrases are also configurable via the presets mechanism documented in the README.

---

## 4.8 Testing and Validation

### 4.8.1 Unit Testing

Unit-level validation was performed on each discrete pipeline component to confirm correct behaviour in isolation before integration testing.

**Webcam Capture:** The `VideoCapture` initialisation was tested by verifying that `cap.isOpened()` returns `True` and that frame dimensions match the configured 640 Ã— 480 target. Frame rate consistency was tested by measuring capture intervals over a 5-second window and confirming that the mean frame interval was within Â±2 ms of the expected 33.3 ms at 30 FPS.

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

**Camera Distance Variation:** Testing was performed at three signing distances: approximately 0.5 m (very close), 1.0 m (standard), and 1.5 m (distant). At 0.5 m, certain signs where both hands extended beyond the frame edges were partially detected; recognition accuracy dropped for these signs. At 1.0 m and 1.5 m, where hands remained within the frame, the face-relative normalisation successfully compensated for the apparent size change, maintaining recognition accuracy. A recommended signing distance of 0.6â€“1.2 m was established based on these tests.

**Hand Speed Variation:** Signs were performed at three speeds: deliberate (approximately 1.5Ã— slower than natural), natural pace, and rapid (approximately 1.5Ã— faster than natural). The time-warping augmentation applied during training (0.75Ã—â€“1.25Ã— resampling) ensures the model has been exposed to speed-varied versions of each sign. Deliberate-speed performance was comparable to natural pace. Rapid signing occasionally caused the 20-frame buffer to capture an incomplete sign (the hand exits the frame before the buffer is full), leading to reduced confidence scores and occasional missed recognitions.

**Multiple Users:** Three additional users (beyond the primary developer) performed each of the 78 signs. Generalisation across users was observed to be strongest for signs performed close to the face (where face-relative normalisation provides strong invariance) and weakest for signs involving extended hand positions far from the face (where absolute position variation is large relative to the face anchor).

### 4.8.5 Performance Evaluation

| Metric | Measured Value |
|---|---|
| Webcam capture rate | 30 FPS (640 Ã— 480) |
| MediaPipe detection per full frame | 30â€“40 ms (every 5 frames) |
| MediaPipe on cached frames | < 5 ms |
| ONNX INT8 inference (20-frame buffer) | 5â€“15 ms per inference |
| PyTorch FP32 inference (fallback) | 30â€“60 ms per inference |
| TemporalPostProcessor per frame | < 1 ms |
| End-to-end latency (95th percentile) | < 200 ms |
| Model size (FP32 PyTorch) | ~4.2 MB |
| Model size (INT8 ONNX) | ~1.05 MB (75% reduction) |
| Inference speedup (ONNX vs PyTorch) | 2â€“3Ã— |
| Sustained FPS during live inference | 25â€“30 FPS |

The primary computational bottleneck is the MediaPipe landmark extraction on full-detection frames. The adaptive interval mechanism (5 frames base, up to 8 in low-motion) reduces effective MediaPipe overhead by 60â€“80% relative to per-frame detection. The HOG detection bypass saves a further 8 ms per full-detection frame.

### 4.8.6 Error Analysis

**Similar Sign Confusion:** The most frequently observed error category was confusion between sign pairs sharing similar handshapes performed near the same facial region. For example, signs differing only in the orientation of the wrist or the extension state of the little finger were occasionally confused when performed quickly. The `similar_class_penalty` mechanism reduces this error rate, though it does not eliminate it for sign pairs with very high visual similarity.

**Transition-Related Errors:** During the interval between consecutive signs, the hand passes through configurations that may superficially resemble known signs at low confidence. Without the momentum commit logic (3-of-5 window) and the confidence threshold (0.60 minimum average), transition frames occasionally committed spurious words. Post-implementation, transition errors were substantially reduced, though rapid multi-sign sequences performed without deliberate pauses between signs still present a challenge.

**Landmark Detection Failures:** In approximately 3â€“5% of observed frames under nominal lighting, MediaPipe failed to detect one or both hands, particularly when hands were partially occluded by the body or when the signer's skin tone had low contrast against the background under certain lighting angles. These frames produce zero-filled feature blocks, which the trained model handles by outputting low-confidence predictions that fall below the commit threshold.

**Environmental Limitations:** The system is not robust to extreme illumination changes such as backlighting (the signer between the camera and a bright window), which causes MediaPipe to fail on hand detection in the majority of frames. Ambient noise in the webcam feed under dim lighting also degrades landmark precision.

### 4.8.7 Validation Results

The integration of all enhancements described in Section 4.6 produced measurable improvements over the Review 2 baseline:

- The TemporalPostProcessor reduced observable prediction jitter from multiple flickered word outputs per second to zero flicker during stable sign holds, confirmed by visual inspection of the live system over 10-minute operation sessions.
- The momentum commit logic (3-of-5, min confidence 0.60) eliminated spurious transition-frame word insertions in 18 of 20 transition test trials (versus 12 of 20 for the pre-enhancement baseline).
- The ONNX INT8 quantisation reduced model file size by 75% (from approximately 4.2 MB to approximately 1.05 MB) and improved inference speed by 2â€“3Ã— on CPU, enabling sustained 25â€“30 FPS live operation on a standard mid-range laptop.
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

**Implemented Solution:** A `_sample_label(sample) -> int` helper function was introduced (`train.py`, lines 76â€“82, commit `4672472b`). The function always reads `sample[1]`, making it compatible with both 2-tuple and 3-tuple formats. All K-fold label extraction calls were updated to use this helper.

**Outcome:** K-fold training executes successfully without crashes. The 3-tuple format is now the canonical sample representation throughout the training pipeline.

---

### Challenge 3: Real-Time Prediction Instability (Jitter)

**Problem:** The initial inference pipeline committed a new word on every frame where the argmax prediction changed. Under natural lighting and with the hand in motion, the model's argmax flickered between 2â€“4 different classes at up to 10â€“15 Hz, making the output sentence unreadable.

**Root Cause:** Individual frame predictions are inherently noisy for a sequence model operating on a sliding window: each new frame shifts the entire buffer by one position, and the model's logit distribution changes rapidly as the sign gesture progresses or transitions.

**Implemented Solution:** The three-layer post-processing stack was implemented: (1) `ConfidenceSmoother` with an 8-frame weighted sliding window; (2) `StablePredictor` with 3-frame patience and 0.12 hysteresis; and (3) the momentum commit logic (3-of-5 window, 0.60 average confidence threshold). Each layer independently reduces jitter, and their combination virtually eliminates flicker under normal operating conditions.

**Outcome:** Zero observable jitter during stable sign holds in all validation sessions. Transition suppression reduced spurious transition-frame word insertions by approximately 90%.

---

### Challenge 4: Dataset Class Imbalance

**Problem:** The initial dataset had highly variable class sizes: some classes (common signs such as "hello" and "thank you") had 150+ samples, while rarer or recently added classes had as few as 30 samples. The standard cross-entropy loss function, trained on this imbalanced distribution, learned to bias predictions towards majority classes.

**Root Cause:** Naturally occurring variation in recording effort and sign complexity resulted in unequal sample counts. The class with the smallest sample count was approximately 5Ã— smaller than the class with the largest sample count.

**Implemented Solution:** Three complementary strategies were applied: (1) `_BalancedAugSubset` oversamples minority classes by repeating their samples until all classes (except the reject class) match the majority class count; (2) inverse-frequency class weights (power 1.0, normalised to mean = 1) are applied as per-sample multipliers to the loss; (3) the `balance_processed_dataset.py` script enforces an 850-sample target per class during dataset preparation, applying downsampling to over-represented classes and oversampling (via augmentation) to under-represented ones.

**Outcome:** The effective class distribution seen by the gradient update step is balanced, and the per-class validation accuracy distribution narrows considerably: the standard deviation of per-class accuracy decreases relative to the unweighted baseline.

---

### Challenge 5: Computational Constraints for Real-Time CPU Inference

**Problem:** Running MediaPipe hand landmark detection on every frame at 30 FPS consumed approximately 40 ms per frame (25 FPS maximum), leaving insufficient budget for model inference, post-processing, and display. Adding face landmark detection to every frame further degraded throughput.

**Root Cause:** MediaPipe's hand and face landmark models, while optimised, require non-trivial CPU time per frame when run at full resolution.

**Implemented Solution:** Four optimisations were applied: (1) adaptive detection interval caching (5 frames nominal, up to 8 in low-motion) reducing effective MediaPipe execution frequency; (2) forced re-detection every 15 frames to prevent stale tracking; (3) HOG person detection disabled (saves ~8 ms per detection frame); and (4) module-level NumPy buffer pre-allocation (`_LANDMARK_BUFFERS` in `preprocess.py`) eliminating per-frame heap allocation overhead. Additionally, ONNX INT8 quantisation reduced model inference time by 2â€“3Ã—.

**Outcome:** Sustained live inference at 25â€“30 FPS on a standard mid-range laptop CPU, meeting the real-time usability target.

---

## 4.10 Current Status of the Module

The Sign-to-Text module is functionally complete and production-ready at its current scope.

**Completed Features:**
- Full 78-class ISL word recognition pipeline from raw webcam input to text output.
- 506-dimensional velocity-augmented, face-relative spatiotemporal feature extraction.
- 10-phase `SignLanguageGRU` architecture (Conv1D frontend, Spatial GNN, Frame Weighting, BiGRU Ã—3, HybridAttention Ã—4 heads, Residual Skips, FC Head).
- 5-fold K-fold cross-validation training pipeline with per-fold checkpoint saving and manifest.
- Two-phase training strategy (Phase 1: curated data; Phase 2: archived fine-tune at 0.25 weight).
- Reject class training with `processed_negatives/` for false-positive suppression.
- ONNX INT8 quantisation with dimension-aligned PyTorch fallback.
- `TemporalPostProcessor` with confidence smoothing (window = 8, decay = 0.3) and `StablePredictor` (patience = 3, hysteresis = 0.12).
- Momentum-based commit logic (3-of-5 window, minimum average confidence 0.60).
- `SentenceBuilder` with ambiguity delay (4 frames when top-1 âˆ’ top-2 < 0.05).
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

To further improve the accuracy, scalability, robustness, and accessibility of the Indian Sign Language recognition system, several strategic enhancements are planned. These future developments are categorised into model improvements, robustness, deployment, and accessibility extensions.

### 4.11.1 Vocabulary and Classification Enhancements

- **Expansion of Sign Vocabulary:** The current system supports a highly curated set of 78 sign classes. Future development will focus on increasing the number of supported signs to over 100â€“200 classes, enabling more comprehensive communication capabilities. The architecture supports this directly: only the final `Linear(96â†’N)` classification layer needs to be replaced, with no changes to the BiGRU, GNN, or attention components.
- **Hierarchical Sign Classification:** Instead of directly classifying signs into a large number of discrete categories, a hierarchical classification framework will be explored. Signs will first be grouped based on characteristics such as motion patterns, handshapes, and semantic categories before final classification. This approach is expected to improve scalability and classification accuracy for vocabularies of 500+ signs.
- **Context-Aware Sign Interpretation:** Future work may incorporate contextual understanding and language modelling techniques â€” such as a fine-tuned T5 or BERT model re-ranking the top-5 sign predictions based on previously committed words â€” to improve sentence formation and resolve ambiguities between visually similar signs.
- **Multilingual Sign Language Support:** Future iterations may explore support for multiple sign languages and regional sign variations, enabling broader adoption across different linguistic and geographical communities.

### 4.11.2 Model and Algorithmic Improvements

- **Dynamic Sign and Continuous Sentence Recognition:** The present system focuses on isolated sign recognition. Future enhancements aim to support continuous sign language recognition via Connectionist Temporal Classification (CTC) or Hidden Markov Models (HMM), enabling the interpretation of complete phrases and sentences without requiring deliberate pauses between signs.
- **Improved Temporal Modelling:** Advanced temporal modelling techniques will be investigated to better capture the sequential nature of sign language. Enhanced temporal smoothing and sequence-level confidence estimation may improve recognition stability during continuous signing.
- **Confidence-Based Prediction Refinement:** Future versions will incorporate more rigorous confidence-aware prediction filtering and temporal consistency checks to further reduce false detections and improve overall system reliability during real-time operation.
- **Explainable Artificial Intelligence (XAI):** Future work may include visualisation and interpretability techniques â€” such as attention weight heat maps over the 20-frame sequence â€” to better understand model predictions, feature importance, and misclassification cases, enabling easier debugging and targeted performance improvements.

### 4.11.3 Robustness and Generalisation

- **Increased Model Generalisation:** Future work will include collecting data from a larger and more diverse group of signers to improve model generalisation across different users, hand sizes, signing styles, and environmental conditions.
- **Low-Light and Occlusion Robustness:** Future work will focus on improving recognition performance under challenging real-world conditions such as poor lighting, partial hand occlusions, motion blur, and cluttered backgrounds.
- **Personalised User Adaptation:** Future versions may incorporate user-specific calibration and adaptive learning techniques â€” leveraging the `adapter_model.py` skeleton already in the repository â€” to improve recognition accuracy for individual signing styles while maintaining generalisation across multiple users.
- **Dataset Expansion and Augmentation:** Additional data collection and augmentation techniques will be explored to increase dataset diversity and improve model robustness against variations in hand orientation, signing speed, lighting, and camera positioning.
- **Feature Optimisation and Reduction:** Future work will focus on identifying the most informative features among the 506 dimensions and pruning redundant inputs, thereby improving computational efficiency while maintaining recognition performance.

### 4.11.4 Deployment and Accessibility Extensions

- **Deployment and Cross-Platform Support:** The system will be optimised for deployment on lightweight devices and integrated into web and mobile platforms to improve accessibility and ease of use.
- **Edge and Offline Deployment:** The INT8 ONNX model (1.05 MB) is already well within the size budget for edge deployment. Conversion to TensorFlow Lite would further enable offline execution on resource-constrained devices, reducing dependency on internet connectivity and enabling real-time operation in remote environments.
- **Smartphone-Based Haptic Feedback for Emergency Signs:** To improve accessibility for deafblind users, smartphone-based haptic feedback mechanisms will be explored for communicating critical emergency signs and alerts through distinct vibration patterns.
- **Emergency Communication Assistance:** A specialised emergency communication module is planned, capable of quickly recognising important signs related to medical assistance, danger, evacuation, and emergency situations to provide rapid communication support when needed.
- **Multi-Modal Human-Computer Interaction:** The system may be extended with additional modalities such as text-to-speech, speech-to-text, visual alerts, and haptic feedback to create a more inclusive communication platform for users with different accessibility requirements.

---

## 4.12 Conclusion

The Sign-to-Text module represents a complete, multi-phase implementation of a real-time Indian Sign Language recognition system, developed from first principles over a period of approximately 3.5 months (February to June 2026) across 173 version-controlled commits. The implementation advances beyond a baseline recognition pipeline through ten independently configurable architectural improvements to the core `SignLanguageGRU` model â€” including a Conv1D depthwise-separable frontend, a lightweight Spatial GNN, learnable frame weighting, multi-layer BiGRU with reduced dropout, and a HybridAttention mechanism combining temporal and proximity-aware attention heads with learnable temperatures â€” together with a comprehensive feature engineering pipeline that produces 506-dimensional velocity-augmented, face-relative spatiotemporal representations.

The testing programme, encompassing unit, integration, functional, robustness, and performance evaluation, confirms that the module meets its core design objectives: real-time operation at 25â€“30 FPS on a CPU-only system, correct recognition across 78 ISL classes, stable output under natural lighting and environmental variation, and a text generation mechanism that suppresses the transition and jitter errors that dominated earlier pipeline iterations.

The enhancements implemented after Review 2 â€” relative feature integration, per-class threshold optimisation, the TemporalPostProcessor, momentum-based commit suppression, ONNX INT8 optimisation, and multi-level augmentation â€” collectively transformed the module from a functionally incomplete prototype into a deployable system suitable for real-world demonstration. The two-phase training strategy and reject-class suppression mechanism further enhance robustness and practical reliability. The module's modular design â€” all hyperparameters centralised in a validated dataclass configuration, all architectural phases independently toggleable â€” provides a solid foundation for future enhancements including continuous recognition, user personalisation, and mobile deployment.

---

*Report Section: Sign to Text Module | Goa College of Engineering Final Year Project | June 2026*
