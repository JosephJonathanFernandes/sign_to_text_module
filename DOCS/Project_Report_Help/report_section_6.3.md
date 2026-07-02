# 6.3 SIGN-TO-SPEECH PIPELINE

This section provides an in-depth, evidence-based analysis of the entire Sign-to-Speech pipeline as implemented in the `sign_to_text_module` repository. The architecture supports a low-latency local inference engine and a decoupled FastAPI WebSocket service (`api/app.py`) designed for real-time, edge-device execution (≈12 ms inference latency per frame sequence).

## High-Level Overview

The end-to-end pipeline translates continuous Indian Sign Language (ISL) gestures from live video streams into grammatically coherent English sentences. 

The data flow occurs in the following sequence:
1. **Frontend/Camera Capture**: A live webcam or video stream captures the signer.
2. **Preprocessing & Feature Extraction**: MediaPipe extracts 3D hand and face landmarks. These are normalized, anchored to the face, and converted into a 506-dimensional feature vector per frame.
3. **Sign Recognition Inference**: A sliding window of 20 frames is fed into an ONNX INT8-quantized hybrid model (Spatial GNN + Bidirectional GRU + Proximity-aware Attention).
4. **Temporal Smoothing & Sentence Building**: Raw token probabilities are temporally smoothed and accumulated into gloss sequences.
5. **NLP Translation Layer**: A custom rule-based grammar correction module converts raw ISL glosses into readable English sentences.
6. **Text-to-Speech (TTS)**: (Planned future enhancement, currently not fully implemented).

## Pipeline Data Flow & Representations

* **Input:** Raw RGB frames (640×480 px, 30 FPS) captured via `webcam.py` or WebSocket.
* **Intermediate Representation 1 (Raw Landmarks):** 21 3D coordinates per hand + face anchor (nose, eyes).
* **Intermediate Representation 2 (Feature Tensor):** `(batch, 20, 506)` tensor. The 506 dimensions include 126 raw coordinates, 126 face-relative coordinates, 1 proximity scalar, and 253 velocity deltas.
* **Intermediate Representation 3 (Context Vector):** A 128-dimensional temporal context vector outputted by the HybridAttention mechanism.
* **Output 1 (Gloss):** Probability distribution over 89 ISL classes.
* **Output 2 (Sentence):** Structurally coherent English string (e.g., "Hello, how are you?").

## Error Handling & Limitations

* **Flood Protection**: The WebSocket API drops frames if more than 2 pending inference calls are in-flight, keeping latency low.
* **Missing Landmarks**: MediaPipe occasionally drops frames under poor lighting or occlusion. The system mitigates this via interpolation and the structural robustness of the Spatial GNN message passing.
* **Limitations**: The model is bound to a fixed sequence length of 20 frames (approx. 667ms). HOG person detection is deliberately disabled in `PreprocessingConfig` to save ~8ms latency, at the cost of losing person-aware filtering.

---

# 6.3.1 SIGN LANGUAGE RECOGNITION MODULE

## A. Purpose
The core objective of the recognition module is to accurately classify continuous ISL gestures from sequential video frames under strict real-time constraints. It bridges the gap between raw spatial configurations and semantic vocabulary tokens.

## B. Input Processing
* **Input Sources:** Real-time webcam (`webcam.py`), offline video files, and WebSocket streams.
* **Frame Extraction:** Sequences are uniformly sampled using `np.linspace` to exactly 20 frames.
* **Landmark Extraction:** Relies on Google MediaPipe (HandLandmarker and FaceLandmarker).
* **Normalization & Noise Handling:** Hand coordinates are normalized relative to the face anchor (nose center, index 1) to make the model translation-invariant.
* **Augmentation:** 
  - *Video-level*: 17 visual effects (blur, noise, brightness) and 3 crop positions.
  - *Landmark-level*: 3D rotation (±15°), scaling (0.88–1.12×), time masking, scattered dropout, and face-anchor shift (random translation to simulate signer repositioning).

## C. Recognition Architecture
The system employs a custom **SignLanguageGRU**, a multi-branch hybrid deep learning architecture utilizing Spatial GNNs, Conv1D, and BiGRUs.

### Architecture Explanation & Processing Flow
1. **Branch A (Spatial GNN):** Takes the first 126 raw coordinate dimensions. The hand skeleton is treated as a graph (21 nodes per hand). A 2-layer Graph Convolutional Network (GCN) aggregates adjacency-weighted neighbors (Linear 3→16, then 16→8). Outputs 16 dimensions per frame.
2. **Branch B (Conv1D Frontend):** Takes the full 506 dimensions. Applies a Pointwise Conv1D (kernel=1) followed by a Depthwise Temporal Conv1D (kernel=3, 128 groups). Outputs 128 dimensions.
3. **Fusion & Frame Weighting:** The 16 GNN dims and 128 Conv1D dims are concatenated (144 dims). A Learnable Frame Weighting MLP (Linear(144→32) → ReLU → Linear(32→1) → Sigmoid) applies soft-attention to suppress uninformative transition frames.
4. **Bidirectional GRU:** Projected to 64 dims, then passed through 3 stacked BiGRU layers (hidden dim 64 per direction, dropout 0.30), yielding a 128-dimensional temporal sequence.
5. **Hybrid Attention:** 4 independent attention heads (2 standard temporal, 2 proximity-aware). Proximity attention scores are additively biased by a learnable Gaussian kernel $\log \mathcal{N}(\text{prox}; 0, \sigma^2)$ where $\sigma=0.15$ initially.
6. **Classification Head:** Dense layers with Dropout(0.25) → Linear(128→96) → ReLU → Linear(96→num_classes).

## D. Training Details
* **Dataset Structure:** Custom recorded videos converted into `.npy` feature matrices.
* **Validation Split:** 5-Fold Cross-Validation with a 70/30 Stratified Split per fold.
* **Hyperparameters:**
  * Epochs: 50
  * Batch Size: 8
  * Optimizer: AdamW (weight decay 5e-4)
  * Learning Rate: $3 \times 10^{-4}$ with a `ReduceLROnPlateau` scheduler (cosine decay).
* **Loss Function:** Cross-Entropy with Label Smoothing ($0.05$) and Inverse Frequency Class Weighting.
* **Additional Regularization:** Mixup Augmentation ($\alpha = 0.3$).

## E. Inference Pipeline
`Frame Capture → MediaPipe Landmarking → 506-dim Feature Construction → Sliding Deque Buffer (maxlen=20) → ONNX INT8 Execution → Logits → Temporal Smoothing → Output Gloss`

The model is exported to ONNX (Opset 18) and dynamically quantized to INT8, reducing the model size from ~4.2 MB to ~1.05 MB and accelerating CPU inference by 2–3×.

## F. Limitations
* The reliance on MediaPipe causes tracking degradation in severe motion blur or low-light conditions.
* The fixed temporal window (20 frames) makes it difficult to recognize highly prolonged gestures without temporal scaling.

---

# 6.3.2 ISL GLOSSES TO ENGLISH SENTENCE TRANSLATION

## A. Input Format
The input to the translation module consists of a sequence of token strings (raw glosses) emitted by the recognition module's temporal postprocessor, such as `["HELLO", "HOW_ARE_YOU", "GOOD"]`.

## B. Translation Mechanism
The repository does **not** employ a sequence-to-sequence Transformer, LSTM, or external LLM API for translation. Instead, it relies on a **Rule-Based NLP Approach**. 

The logic is embedded in `src/inference/nlp_postprocessor.py` and `src/inference/sentence_builder.py`.

## C. Processing Pipeline
1. **Gloss Sequence Accumulation:** The `SentenceBuilder` aggregates temporally stable words, ignoring idle timeouts and duplicate adjacent predictions.
2. **Grammar Correction:** The raw gloss sequence is passed to `NLPPostprocessor.fix_grammar()`, which applies custom heuristic pattern matching to fix direct ISL-to-English literal translations.
3. **Punctuation Insertion:** `NLPPostprocessor.add_punctuation()` applies rules based on questioning words (e.g., "what", "how") to append question marks or periods.
4. **Post-Processing & Capitalization:** The string is normalized into a capitalized, human-readable English sentence.

## D. Implementation Details & Limitations
* **Files involved:** `nlp_postprocessor.py`, `sentence_builder.py`, `webcam.py` (which displays the translation stream).
* **Limitations:** The rule-based nature is rigid. It works well for known conversational structures hardcoded in the heuristics, but cannot dynamically infer complex, unseen grammatical transformations the way a neural Seq2Seq model or LLM would. 

> [!WARNING]
> **Assumptions and Missing Information**
> A fully contextual LLM/Transformer-based translation module was proposed in early designs but is not fully implemented in the current source code; the system relies exclusively on deterministic Python rule-sets.

---

# 6.3.3 TEXT TO SPEECH TRANSLATION USING SARVAM

## A. Purpose
To convert the final, grammatically corrected English sentence into spoken audio output, providing a complete non-verbal to verbal communication bridge.

## B. Implementation Status

> [!IMPORTANT]
> The repository currently does not contain complete Sarvam API implementation, and this component remains future work.

Thorough inspection of the repository (including `api/`, `src/`, and dependency files) reveals no endpoint usage, authentication logic, or API calls directed towards Sarvam AI. 

## C. Future Scope
Once implemented, the theoretical pipeline would involve:
`Input Text → Request Formatting (JSON) → Authenticated Sarvam API Call → Audio Stream Reception (Speech Synthesis) → Output via Local Speaker`

Limitations of adding this in the future include network latency constraints (which could impact the current 12ms real-time edge performance) and dependence on third-party API availability.

---

# Architecture Diagram

```text
+-------------------------------------------------------------+
|                        USER INPUT                           |
|  (Live Webcam / Video Feed: 640x480, 30 FPS)                |
+-----------------------------+-------------------------------+
                              |
                              v
+-----------------------------+-------------------------------+
|                PREPROCESSING & LANDMARKING                  |
|  MediaPipe Hand & Face Extraction -> 506-dim Feature Tensor |
+-----------------------------+-------------------------------+
                              |
                              v
+-----------------------------+-------------------------------+
|                  SIGN RECOGNITION (ONNX)                    |
| +---------------------------------------------------------+ |
| |  Branch A: Spatial GNN (16 dims)                        | |
| |  Branch B: Conv1D Temporal (128 dims)                   | |
| |  Fusion & Learnable Frame Weighting (144 dims)          | |
| |  3-Layer Bidirectional GRU (128 dims)                   | |
| |  Proximity-Aware Hybrid Attention                       | |
| |  Fully Connected Head (89 Classes)                      | |
| +---------------------------------------------------------+ |
+-----------------------------+-------------------------------+
                              |
                              v
+-----------------------------+-------------------------------+
|                    ISL GLOSS GENERATION                     |
|  (Temporal Smoothing & Hysteresis Thresholding)             |
+-----------------------------+-------------------------------+
                              |
                              v
+-----------------------------+-------------------------------+
|                ENGLISH SENTENCE TRANSLATION                 |
|  (Rule-Based NLP Grammar & Punctuation Correction)          |
+-----------------------------+-------------------------------+
                              |
                              v
+-----------------------------+-------------------------------+
|            SARVAM TEXT-TO-SPEECH (FUTURE WORK)              |
|  (External API Call for Speech Synthesis)                   |
+-----------------------------+-------------------------------+
                              |
                              v
+-----------------------------+-------------------------------+
|                       SPEECH OUTPUT                         |
+-------------------------------------------------------------+
```

---

# Pseudocode

```python
# OVERALL SIGN-TO-SPEECH PIPELINE
initialize_mediapipe_models()
model = load_onnx_int8_model("model_int8.onnx")
buffer = SlidingDeque(maxlen=20)
sentence_builder = SentenceBuilder()
nlp_processor = NLPPostprocessor()

while video_stream_active:
    frame = capture_frame()
    
    # 1. Preprocessing
    landmarks = extract_mediapipe_landmarks(frame)
    if landmarks.is_empty():
        continue
        
    normalized_coords = face_anchor_normalization(landmarks)
    velocity = calculate_frame_velocity(normalized_coords, buffer.last())
    features = concatenate(normalized_coords, velocity) # 506 dims
    
    buffer.append(features)
    
    if len(buffer) == 20:
        # 2. Sign Recognition
        logits = model.predict(buffer)
        raw_prediction, confidence = softmax(logits)
        
        # 3. Temporal Smoothing
        stable_word = apply_temporal_hysteresis(raw_prediction, confidence)
        
        if stable_word:
            # 4. Sentence Building
            sentence_builder.add_word(stable_word)
            
            # 5. NLP Translation
            raw_text = sentence_builder.get_text()
            english_sentence = nlp_processor.fix_grammar(raw_text)
            english_sentence = nlp_processor.add_punctuation(english_sentence)
            
            display_to_screen(english_sentence)
            
            # 6. Text-to-Speech (Placeholder / Future Work)
            # audio = sarvam_api.synthesize(english_sentence)
            # play_audio(audio)
```

---

# Mathematical Formulation

Where implemented in the source code, the network relies on the following mathematical formulations:

**1. Frame Velocity Calculation**
For consecutive frames $t$ and $t-1$, the velocity vector is:
$$ V_t = X_t - X_{t-1} $$

**2. Graph Convolution Network (Spatial GNN)**
For node representations $H^{(l)}$ at layer $l$, and normalized adjacency matrix $\hat{A}$:
$$ H^{(l+1)} = \sigma \left( \hat{A} H^{(l)} W^{(l)} \right) $$
Where $\sigma$ is the ReLU activation function.

**3. Learnable Frame Weighting (Soft Attention)**
Given the fused feature matrix $F$, a scalar weight $w_t$ is learned for each frame:
$$ w_t = \sigma(W_2 \max(0, W_1 F_t + b_1) + b_2) $$
$$ F'_t = F_t \odot w_t $$

**4. Proximity-Aware Attention Bias**
The standard attention score $e_{t}$ is biased by the physical distance $d_t$ between the hand and face:
$$ \alpha_t = \text{Softmax}\left( e_t + \log \mathcal{N}(d_t; 0, \sigma_{learnable}^2) \right) $$
This ensures frames where hands are closer to the face receive dynamically calibrated focus.

**5. Cross Entropy with Label Smoothing**
During training, targets $y$ are smoothed using factor $\alpha = 0.05$:
$$ y'_k = (1 - \alpha) y_k + \frac{\alpha}{K} $$
Where $K$ is the total number of classes (89).

---

# Tables

### 1. Software Libraries
| Library | Version / Usage | Purpose in Pipeline |
|---------|-----------------|---------------------|
| `torch` | 2.x | Model training and architecture definition |
| `onnxruntime` | 1.15+ | Edge inference execution (INT8) |
| `mediapipe` | Holistic 0.10.x | 3D hand and face landmark extraction |
| `fastapi` | Latest | WebSocket API for frontend decoupling |
| `numpy` | 1.24+ | Matrix operations and feature processing |

### 2. Models Used
| Model Component | Type / Arch | Parameters | Purpose |
|-----------------|-------------|------------|---------|
| Feature Extractor | MediaPipe | N/A (Pre-trained) | Frame to 3D coordinate mapping |
| Spatial Graph | GCN | ~2K | Hand anatomy spatial relationships |
| Temporal Engine | BiGRU | ~225K | Sequence dynamics modeling |
| Classification | Dense FC | ~13K | Final gloss token prediction |
| **Total Size** | INT8 Quantized | **~344K / 1.05 MB** | Real-time edge classification |

### 3. Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 8 | Small batches suitable for limited class counts |
| Learning Rate | $3 \times 10^{-4}$ | Ensures stable convergence, cosine decayed |
| Sequence Length | 20 | Optimal temporal window (~667 ms) for single signs |
| Dropout (GRU) | 0.30 | Prevents temporal overfitting |
| Label Smoothing | 0.05 | Prevents overconfident predictions |
| Confidence Threshold | 0.12 | Dynamically adjusted hysteresis threshold |

### 4. Input-Output Formats
| Stage | Input Format | Output Format |
|-------|--------------|---------------|
| Preprocessing | RGB Frame (`640x480`) | 506-dim Numpy Array |
| Inference Engine | `(Batch, 20, 506)` Float Tensor | 89-dim Probability Vector |
| Sentence Builder | `String` (Gloss Token) | `String` (Gloss Sequence) |
| NLP Postprocessor | `String` (ISL Grammar) | `String` (English Grammar) |

### 5. Dataset Details
| Property | Value |
|----------|-------|
| Number of Classes | 89 |
| Augmentations per video | Up to 54 variants |
| Split Ratio | 70% Train / 30% Validation |
| Validation Strategy | 5-Fold Stratified Cross-Validation |
| Class Weighting | Inverse Frequency ($P=1.0$) |
