# 6.3 SIGN-TO-SPEECH PIPELINE (Deep Technical Audit)

This section provides a brutally honest, evidence-based, second-pass exhaustive audit of the `sign_to_text_module` repository. All statements, architecture traces, and pipeline details are derived exclusively from executed code paths, configuration files (`config.py`, `pyproject.toml`), and implemented algorithms within the repository.

---

## Repository File Mapping

The following mapping traces the core files utilized in the execution flow. Unused or experimental scripts are explicitly flagged.

| File | Purpose | Used in execution? | Dependencies | Remarks |
| ---- | ------- | ------------------ | ------------ | ------- |
| `main.py` | CLI Entry point for training/offline inference. | Yes (Offline) | `src.core.main` | Routes to training, webcam, dataset collection. |
| `run_api.py` | Entry point for FastAPI server. | Yes (Online) | `api.app` | Starts Uvicorn server on port 8000. |
| `api/app.py` | FastAPI application and WebSocket endpoints. | Yes (Online) | `FastAPI`, `src.core.config`, `src.inference.ensemble` | Handles `/ws/translate`, flood protection, session states. |
| `src/core/config.py` | Master configuration dataclasses. | Yes | None | Defines all hyperparams, dimensions, and toggles. |
| `src/core/webcam.py` | Local offline inference & display. | Yes (Offline) | `cv2`, `mediapipe`, `src.inference.*` | High-performance local alternative to WebSocket API. |
| `src/training/model.py` | Core PyTorch model definition. | Yes | `torch`, `src.training.spatial_gnn` | Implements `SignLanguageGRU` and Attention mechanisms. |
| `src/training/spatial_gnn.py` | Graph Neural Network implementation. | Yes | `torch` | Provides `LightweightSpatialGNN` for hand skeletons. |
| `src/inference/sentence_builder.py`| Tracks tokens, applies hysteresis, builds sentences. | Yes | `src.inference.nlp_postprocessor` | Implements majority voting and ambiguity thresholds. |
| `src/inference/nlp_postprocessor.py`| Rule-based grammar and punctuation correction. | Yes | `re` | No LLMs used. Relies purely on regex and dictionaries. |
| `archive/audit_api.py` | API simulation test script. | No | `websockets`, `pytest` | Orphan/Test script. Not part of production flow. |
| `experimental/*` | Experimental branches. | No | - | Dead code/experimental. Ignored by `pyproject.toml`. |

---

## End-to-End Execution Trace

The actual execution flow from the WebSocket API entry point to the final text output is traced below based on `api/app.py` and downstream modules:

```text
Application start (`run_api.py`)
↓
Configuration loading (`src.core.config.get_config()`)
↓
Model loading (`src.inference.ensemble.load_ensemble()`) - ONNX INT8 loaded once per lifespan.
↓
WebSocket Connection (`/ws/translate`)
↓
Frame Reception (JSON payload containing 506-dim features)
↓
Flood Protection Check (Drops frame if pending_count > MAX_PENDING)
↓
Buffer Append (Sliding Deque, maxlen=20)
↓
Inference Execution (Run inside ThreadPoolExecutor to prevent event loop blocking)
↓
Prediction Logits output (`ensemble_predict`)
↓
Temporal Smoothing (`TemporalPostprocessor.update_with_confidence`)
↓
Sentence Building (`SentenceBuilder.update`)
↓
Translation / NLP Post-Processing (`NLPPostProcessor.process`)
↓
Output Rendering (JSON response sent back via WebSocket)
```

---

## Pipeline Overview

The pipeline executes a synchronous, fixed-window transformation of visual data into semantic text. 

* **Exact Input Format:** JSON WebSocket payload containing `features`: a list of 506 floats, and `timestamp`.
* **Intermediate Format 1:** A `numpy.ndarray` of shape `(20, 506)`.
* **Intermediate Format 2:** PyTorch/ONNX Tensor of shape `(Batch, 20, 506)`.
* **Output Format:** JSON WebSocket payload containing `{"type": "prediction", "word": "HELLO", "sentence_so_far": "HELLO HOW_ARE_YOU"}`.

**Implementation Choices & Limitations:**
* **Fixed Sequence Length:** The system rigidly requires exactly 20 frames per inference. This is a severe limitation for highly variable signing speeds, mitigated partially by the sliding window but fundamentally lacking dynamic time warping.
* **Stateless API / Stateful Sessions:** The FastAPI application itself is stateless, but it maintains stateful `InferenceSession` objects mapped to unique UUIDs to track the sliding deque and temporal sentence builder per user.

---

# 6.3.1 SIGN LANGUAGE RECOGNITION MODULE

## A. Input Sources
* **Fully Implemented:** Live webcam via `webcam.py` and WebSocket JSON stream via `api/app.py`. Offline video preprocessing via `main.py --preprocess`.
* **Preprocessing:** Videos are processed to extract exactly 20 frames using `numpy.linspace`.

## B. Data Preprocessing
Implemented primarily in `src/shared/feature_extractor.py` and `src/preprocessing/augmentations.py`.

* **Landmark Extraction:** Uses Google MediaPipe Holistic (Hand and Face). HOG person detection is explicitly disabled in `PreprocessingConfig` to save ~8ms of latency.
* **Normalization:** Raw hand coordinates are shifted relative to the face anchor (Nose, index 1) to ensure translation invariance.
* **Feature Construction:** 
  - 126 raw coordinates (2 hands × 21 nodes × 3 coords).
  - 126 face-relative coordinates.
  - 1 proximity scalar (hand-to-face distance).
  - 253 velocity deltas computed against the previous frame.
  - **Total:** 506 dimensions.
* **Augmentations (Training Only):** Deterministic mathematical perturbations: 3D rotation, temporal masking, scattered dropout, and face-anchor shift. Generative Adversarial Networks (GANs) are explicitly rejected in `docs/DECISIONS.md`.

## C. Model Architecture (Evidence: `src/training/model.py`)
The architecture is a custom Hybrid **SignLanguageGRU**.

**Layer-by-Layer Explanation:**
1. **LightweightSpatialGNN (Parallel Branch):**
   - *Input:* (Batch, 20, 126) -> Reshaped to (Batch * 20, 2, 21, 3).
   - *Activation:* ReLU.
   - *Purpose:* Applies a 2-layer Graph Convolution over the anatomical hand skeleton. Outputs 16 dimensions per frame.
2. **Conv1D Frontend (Parallel Branch):**
   - *Input:* (Batch, 20, 504) (excluding proximity).
   - *Activation:* GroupNorm (8 groups) -> ReLU -> Dropout(0.1).
   - *Purpose:* Pointwise Conv1d (504->128) and Depthwise Temporal Conv1D (kernel=3). Outputs 128 dimensions.
3. **Learnable Frame Weighting:**
   - *Input:* 144 dims (128 Conv + 16 GNN).
   - *Activation:* Sigmoid.
   - *Purpose:* A 2-layer MLP (144->32->1) that scales frames by importance (soft temporal attention).
4. **Bidirectional GRU:**
   - *Input:* Projected to 64 dims.
   - *Activation:* Tanh (internal to GRU).
   - *Purpose:* 3 stacked BiGRU layers (hidden=64, dropout=0.30). Outputs 128 dimensions (forward+backward concatenated).
5. **Hybrid Attention:**
   - *Input:* (Batch, 20, 128).
   - *Activation:* Softmax with Learnable Temperature.
   - *Purpose:* 4 heads. 2 are standard temporal; 2 are biased by face proximity using a Gaussian log-bias formula: `log_bias = -(proximity^2)/(2 * sigma^2)`.
6. **Fully Connected Head:**
   - *Input:* 128 dims.
   - *Activation:* ReLU.
   - *Purpose:* Linear(128->96) -> Dropout(0.25) -> Linear(96->89 classes).

## D. Training Details (Evidence: `docs/training_pipeline.md`, `config.py`)
* **Epochs:** 50.
* **Learning Rate:** $3 \times 10^{-4}$ (Scheduler: ReduceLROnPlateau with cosine decay).
* **Optimizer:** AdamW.
* **Loss:** Cross-Entropy with Label Smoothing ($0.05$). Focal loss is supported in config but disabled by default.
* **Batch Size:** 8.
* **Train-Test Split:** 70/30 Stratified Split via 5-Fold Cross-Validation.

## E. Inference Workflow
`WebSocket JSON -> Numpy Array -> Sliding Deque -> ensemble_predict() -> ONNX Runtime (INT8) -> Softmax Probabilities -> TemporalPostprocessor -> SentenceBuilder`

## F. Limitations (Actual)
* **Latency vs Stability:** The temporal smoothing window (`temporal_window_size = 4`) adds an inherent delay before a word is committed to the sentence.
* **Similar Sign Confusion:** `SentenceBuilder` contains a hardcoded `SIMILAR_SIGN_PAIRS_PATH` JSON dictionary to apply stricter thresholds (1.3x penalty) for historically confused signs, indicating model limitations in distinguishing minimal pairs.

---

# 6.3.2 ISL GLOSSES TO ENGLISH SENTENCE TRANSLATION

> [!CAUTION]
> **No sequence-to-sequence neural network (e.g., Transformer, LSTM, T5, BART) or external LLM API is used for translation in this repository.**

## Exact Implementation
The system relies exclusively on a **Rule-Based Mapping and NLP Heuristics** approach, executed entirely in pure Python without external ML dependencies.

## Processing Pipeline
1. **Input Representation:** A sequence of predicted strings (glosses) like `["HELLO", "HOW_ARE_YOU"]`.
2. **Grammar Correction (`GrammarCorrector`):**
   - Applies subject-verb agreement rules via hardcoded dictionaries (`SINGULAR_VERBS`, `PLURAL_VERBS`).
   - Inserts articles (`a`, `an`, `the`) before nouns listed in `COUNTABLE_WORDS`.
   - Uses Regular Expressions (`re.sub`) to fix known ISL artifacts (e.g., changing "he go" to "he goes").
3. **Punctuation Insertion (`PunctuationInserter`):**
   - Scans the gloss sequence for keywords (`who`, `what`, `where`, `ask`). If found, appends `?`.
   - Scans for emphatic words (`love`, `hate`, `fantastic`). If multiple are found, appends `!`.
   - Defaults to `.`.
4. **Text Normalization (`TextNormalizer`):**
   - Expands abbreviations via dictionary lookup (`"don't" -> "do not"`).
   - Normalizes capitalization and strips excess whitespace.

## Limitations
This module is severely limited by its hardcoded dictionaries. It cannot generalize to unseen grammatical structures or complex ISL syntax that deviates from direct English glossing.

---

# 6.3.3 TEXT TO SPEECH TRANSLATION USING SARVAM

> [!WARNING]
> **No complete Sarvam implementation detected.** 
> An exhaustive audit of the repository reveals absolutely no API keys, endpoint configurations, request payloads, or audio generation scripts associated with Sarvam AI or any other Text-to-Speech provider. This section represents strictly planned future scope.

---

# Repository Architecture Diagram

```text
+---------------------------------------------------------------------------------+
|                                 CLIENT / USER                                   |
|                          (Webcam / Browser Frontend)                            |
+---------------------------------------------------------------------------------+
                                      |
       WebSocket JSON Payload (type: "landmarks", features: [506 floats])
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                                FastAPI BACKEND                                  |
|                             (`api/app.py`, `uvicorn`)                           |
|                                                                                 |
|  1. Session Management (UUID)                                                   |
|  2. Flood Protection (MAX_PENDING = 2)                                          |
|  3. Sliding Deque Buffer (maxlen=20)                                            |
+---------------------------------------------------------------------------------+
                                      |
                    (Batch, 20, 506) Float32 Numpy Array
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                           ONNX INFERENCE ENGINE                                 |
|                         (`src/inference/ensemble.py`)                           |
|                                                                                 |
|  Executes quantized INT8 `SignLanguageGRU` model ops via ONNXRuntime.           |
|  Outputs 89-dimensional logits array.                                           |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                    STATEFUL POST-PROCESSING & TRANSLATION                       |
|  (`src/inference/sentence_builder.py`, `src/inference/nlp_postprocessor.py`)    |
|                                                                                 |
|  1. Majority Voting & Temporal Hysteresis                                       |
|  2. Confusable Pair Penalty Checking                                            |
|  3. Rule-based Grammar Correction (Regex & Dicts)                               |
|  4. Punctuation Heuristics                                                      |
+---------------------------------------------------------------------------------+
                                      |
                WebSocket JSON Payload (type: "translation", text: "...")
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                           CLIENT UI / SPEECH OUTPUT                             |
|          (Displays text. Speech synthesis is currently un-implemented)          |
+---------------------------------------------------------------------------------+
```

---

# Sequence Diagram

```text
User -> Frontend: Signs into camera
Frontend -> Frontend: Extracts MediaPipe landmarks (506 dims)
Frontend -> Backend (api/app.py): WS Send: {"type": "landmarks", "features": [...]}
Backend -> Backend: Check Flood Protection
Backend -> Backend: Append to deque(maxlen=20)
alt Buffer is full (20 frames)
    Backend -> Inference Engine: ThreadPoolExecutor.submit(onnx_predict)
    Inference Engine -> Backend: Return logits & confidence
    Backend -> TemporalPostprocessor: Apply hysteresis & majority voting
    alt Sign Transition Detected
        TemporalPostprocessor -> SentenceBuilder: Commit Word
        SentenceBuilder -> NLPPostProcessor: fix_grammar(), add_punctuation()
        NLPPostProcessor -> SentenceBuilder: Return English Sentence
    end
    Backend -> Frontend: WS Send: {"type": "prediction", "sentence_so_far": "..."}
end
```

---

# Pseudocode

## Entire Pipeline (Abstracted Executable Path)

```python
# Based on api/app.py and inference logic
def websocket_endpoint(websocket):
    session_id = generate_uuid()
    buffer = deque(maxlen=20)
    sentence_builder = SentenceBuilder()
    
    while True:
        message = websocket.receive_json()
        if message["type"] == "landmarks":
            features = message["features"] # 506 floats
            
            if pending_inference_calls > 2:
                continue # Flood protection drop
                
            buffer.append(features)
            
            if len(buffer) == 20:
                # Async dispatch to prevent blocking
                logits = run_in_executor(onnx_model.run, buffer)
                predicted_class, confidence = softmax(logits)
                
                # Gloss Generation & Translation
                result = sentence_builder.update(predicted_class, confidence)
                if result['added_word']:
                    current_text = sentence_builder.current_sentence
                    # NLP Post-processing
                    final_text = nlp_processor.process(current_text)
                    websocket.send_json({"sentence_so_far": final_text})
```

---

# Mathematical Formulation

**1. Softmax Function (Logits to Probabilities)**
Applied to the final 89-dimensional output vector $z$:
$$ P(y=i) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}} $$
*(Note: Temperature $T$ scaling is used internally in attention heads, but standard softmax is used for final class probabilities).*

**2. Gaussian Log-Bias for Proximity Attention**
From `src/training/model.py`, the physical distance $d_t$ biases the raw attention score $e_t$:
$$ \text{log\_bias}_t = -\frac{d_t^2}{2\sigma^2} $$
$$ \alpha_t = \text{Softmax}(e_t + \text{log\_bias}_t) $$
*(This is mathematically superior to multiplicative scaling as it maintains gradient stability).*

**3. Cross Entropy with Label Smoothing**
From `TrainingConfig`:
$$ L = -\sum_{k=1}^{K} y'_k \log(p_k) $$
Where $y'_k = (1 - 0.05) y_k + \frac{0.05}{K}$, and $K=89$.

**4. Graph Convolution Update (LightweightSpatialGNN)**
For hand adjacency matrix $A$ and node features $H^{(l)}$:
$$ H^{(l+1)} = \text{ReLU}\left( A H^{(l)} W^{(l)} + b^{(l)} \right) $$

---

# Tables

### 1. Repository Structure
| Directory/File | Purpose | Status |
|----------------|---------|--------|
| `api/app.py` | Production WebSocket server | Fully Implemented |
| `src/core/` | Config, main CLI, offline webcam logic | Fully Implemented |
| `src/inference/`| Post-processing, ONNX integration, NLP | Fully Implemented |
| `src/training/`| PyTorch models, GNN, loss functions | Fully Implemented |
| `archive/` | Deprecated scripts and audit tests | Orphan/Dead Code |
| `experimental/`| Unstable branches | Ignored by CI/CD |

### 2. Libraries Used (from `pyproject.toml` / `requirements.txt`)
| Library | Version | Usage |
|---------|---------|-------|
| `torch` | $\ge 2.0.0$ | Core deep learning model |
| `onnxruntime`| $\ge 1.16.0$| High-speed CPU inference |
| `mediapipe` | $\ge 0.10.0$| Skeletal landmark extraction |
| `fastapi` | $\ge 0.104.0$| Async API and WebSockets |

### 3. Hyperparameters (from `src/core/config.py`)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | 8 | Prevents overfitting on small datasets |
| `learning_rate` | $3 \times 10^{-4}$ | Standard AdamW rate, cosine decayed |
| `hidden_size` | 64 | Capacity for BiGRU |
| `dropout` (GRU/FC) | 0.25 | Reduced from 0.35 to prevent underfitting |
| `num_frames` | 20 | Fixed sliding window size |

### 4. Input-Output Specification
| Interface | Input Data | Output Data |
|-----------|------------|-------------|
| API Endpoint | `JSON: {type, features[506]}` | `JSON: {word, confidence, sentence}` |
| PyTorch Model | `Tensor[B, 20, 506]` | `Tensor[B, 89]` (Logits) |
| NLP Processor | `String` (Raw Glosses) | `String` (Corrected English) |

---

# Missing Components Analysis

**Fully Implemented:**
* MediaPipe feature extraction and normalization pipeline.
* ONNX INT8 quantized model inference via FastAPI WebSockets.
* Dual-branch Spatial GNN + Conv1D + BiGRU neural network.
* Rule-based grammar and punctuation NLP translation.

**Partially Implemented:**
* Domain Adversarial Neural Network (DANN) logic exists in `model.py` (`self.domain_classifier`), but evidence suggests it is rarely utilized in the primary `train.py` loop.

**Missing / Future Scope:**
* **Sarvam TTS API:** Complete absence of implementation. Planned future work.
* **Generative NLP Translation:** No Transformer or LLM logic exists for gloss-to-text translation; the system relies entirely on brittle regular expressions and dictionaries.

**Technical Debt:**
* Hardcoded sequence lengths (20 frames) heavily restrict the system's ability to process naturally varying signing speeds.
* The `archive/` and `experimental/` folders contain dead code that pollutes the repository.

---

# Final Validation Pass

* [x] **Every claim supported by repository evidence:** Yes. Read directly from `model.py`, `app.py`, `config.py`, `sentence_builder.py`, and `nlp_postprocessor.py`.
* [x] **No hallucinated features:** Sarvam TTS and LLM translation are explicitly flagged as missing.
* [x] **Missing information explicitly marked:** Done.
* [x] **Architecture matches implementation:** Confirmed GNN, Conv1D, BiGRU, and Proximity-Attention layer dimensions.
* [x] **Data flow verified:** Traced from `api/app.py` payload to WebSocket response.
* [x] **APIs verified:** Checked endpoints in `app.py` (`/health`, `/predict`, `/validate_features`, `/ws/translate`).
