## 6.3 SIGN-TO-TEXT PIPELINE IMPLEMENTATION

### 6.3.1 Sign Language Recognition Module Implementation
The Sign Language Recognition Module functions as the foundational spatiotemporal perception layer, engineered to facilitate the real-time classification of streaming landmark coordinates into semantic textual representations. By establishing a unified bridge between MediaPipe-derived spatial topology and downstream linguistic engines, this framework orchestrates the complete interpretation lifecycle. The architecture integrates rigorous geometric preprocessing, multi-stream feature composition, and aggressively optimized neural inference to resolve the inherent complexities of visual gesture identification.

The functional stages of the recognition pipeline are structured as follows:

**Deterministic Frame Extraction and Memory Optimization:**
To ensure absolute temporal consistency across heterogeneous video samples, the preprocessing workflow (`src/preprocessing/preprocess.py`) utilizes a linear interpolation strategy (`np.linspace`) to sub-sample video sequences into exactly twenty frames. The implementation leverages pre-allocated memory buffers to mathematically suppress the computational overhead of repeated garbage collection during live execution. In instances where environmental noise or physical hand occlusions impede detection, the system applies zero-initialization to the landmark arrays; this ensures that missing data is represented through stable mathematical placeholders without compromising the structural integrity of the 506-dimensional feature schema.

```python
def extract_landmarks_with_face_relative(frame, hand_landmarker, face_landmarker):
    left_raw = np.zeros(63, dtype=np.float32)
    right_raw = np.zeros(63, dtype=np.float32)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    hand_result = hand_landmarker.detect(mp_image)
    face_result = face_landmarker.detect(mp_image)

    if hand_result.hand_landmarks:
        for idx, hand_lms in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            flat_lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms]).flatten()

            if handedness == "Left":
                left_raw = flat_lms
            else:
                right_raw = flat_lms

    return left_raw, right_raw, face_result
```

**Spatial Normalization and Feature Composition:**
The core feature engineering layer (`src/shared/feature_extractor.py`) systematically transforms raw coordinates into high-dimensional, posture-invariant representations. To achieve translation and scale invariance, the pipeline anchors the anatomical skeleton to the wrist nodes and scales the resulting topology by the maximum Euclidean distance. By concatenating these stabilized landmarks with face-relative projections and temporal motion descriptors, the system generates a robust feature vector that effectively isolates articulatory dynamics from camera-specific variables.

**Spatiotemporal Feature Composition Workflow:**
The resulting 506-dimensional embedding is synthesized through a precise four-stage computational hierarchy. Initially, normalized coordinates for both hands contribute 126 dimensions of wrist-anchored data. Subsequently, 126 additional dimensions are derived through facial-relative projections, utilizing inter-pupillary distance as a scale-invariant factor to stabilize landmarks against varying camera depths. To capture zone-specific interactions, the architecture calculates Euclidean hand-to-face proximity scalars. Finally, the pipeline derives first-order motion velocity descriptors through finite-difference operations, ensuring that both static grammatical postures and rapid articulatory transitions are accurately encoded.

**Multi-Branch Neural Network Architecture:**
The foundational recognition engine (`src/training/model.py`) utilizes the `SignLanguageGRU` model to orchestrate a sophisticated multi-stream processing lifecycle. By integrating a `LightweightSpatialGNN` module to map the physiological topology of hand landmarks alongside a Conv1D frontend for localized spatial feature extraction, the architecture condenses high-dimensional inputs into dense representations. These parallel feature streams are subsequently fused and passed to a robust Bidirectional GRU layer, which is specifically engineered to resolve long-term temporal dependencies across fluid gesture sequences. 

To fortify the model against multi-signer variability, the pipeline incorporates a Domain-Adversarial Neural Network (DANN) framework, facilitating the acquisition of strictly domain-invariant features. At the culmination of the network, a Multilayer Perceptron (MLP) classification head—reinforced with aggressive dropout regularization—synthesizes these spatiotemporal embeddings into a precise probability distribution across the expanded **300 distinct ISL gloss categories**.

```python
class SignLanguageGRU(nn.Module):
    def __init__(self, num_classes=300, config=None):
        super(SignLanguageGRU, self).__init__()

        self.spatial_gnn = LightweightSpatialGNN(hidden_dim=16, output_dim=8)
        self.pointwise_conv = nn.Conv1d(506, 128, kernel_size=1)
        self.temporal_conv = nn.Conv1d(128, 128, kernel_size=3, padding=1, groups=128)

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.30
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, num_classes)
        )
```

**Physiological Topology via Graph-Based Representations:**
The structural interdependencies of anatomical joints are modeled through the `LightweightSpatialGNN` and `GraphConvLayer` modules. By utilizing an adjacency matrix derived from the non-Euclidean connectivity of hand landmarks, the architecture explicitly captures biomechanical constraints. During the forward pass, feature propagation is governed by the following graph operation:

$$H' = \sigma(AHW)$$

where $A$ represents the adjacency matrix, $H$ denotes input node features, $W$ signifies the trainable weight parameters, and $\sigma$ represents the non-linear activation function.

```python
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_channels):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output
```

**Hybrid Proximity-Aware Attention Mechanism:**
To prioritize frames where articulatory gestures interact near facial reference points—critical zones for semantic signaling in SLT—the architecture utilizes a proximity-based weighting strategy. This Hybrid Attention layer calculates a Gaussian attention bias derived from the real-time hand-to-face proximity scalar ($d_t$):

$$\text{log\_bias}_t = -\frac{d_t^2}{2\sigma^2}$$

The normalized attention weights are subsequently formulated as:

$$\alpha_t = \frac{\exp(e_t+\text{log\_bias}_t)}{\sum_j \exp(e_j+\text{log\_bias}_j)}$$

In this optimization, $e_t$ signifies the baseline attention score, while $\alpha_t$ represents the resulting weighted distribution used for spatiotemporal feature fusion.

**Runtime Inference and Fault Tolerance:**
The production inference lifecycle is orchestrated via the `ONNXModelWrapper` within `src/inference/onnx_inference.py`. Before execution, the engine performs rigorous dimensional validation to ensure incoming feature tensors adhere to the strict 506-dimensional schema. To resolve topological shape mismatches caused by legacy interfaces, the pipeline applies deterministic alignment and padding routines to stabilize the input structure. 

To fortify the system against runtime failures, the workflow integrates aggressive exception handling. In the event of an ONNX execution error, the architecture automatically triggers a stateful PyTorch-based fallback mechanism, sustaining uninterrupted real-time service and bridging the gap between high-performance inference and operational reliability.

**Structural Topology Normalization and Legacy Integration:**
The architectural migration from 253-dimensional vectors to a high-fidelity 506-dimensional feature space introduced significant synchronization challenges with existing datasets and established client interfaces. To maintain system stability, the `ONNXModelWrapper` utilizes an automated input alignment layer. This mechanism dynamically intercepts incoming tensors that deviate from the target dimensional schema, applying deterministic padding and reshaping operations prior to inference. Such a strategy ensures seamless backward compatibility and prevents catastrophic runtime failures induced by topological shape mismatches.

**Temporal Output Stabilization and Linguistic Parsing:**
To suppress high-frequency prediction jitter, the system implements a `ConfidenceSmoother` that calculates an exponentially weighted moving average across an eight-frame sliding window. This mathematical smoothing heavily reduces articulatory noise during transitional states. Subsequently, the `NLPPostProcessor` maps these stabilized glosses into grammatically fluid text by executing token deduplication, resolving subject–verb agreement, and applying localized morphological transformations to bridge the syntactic modality gap.

**Heuristic Confidence Calibration for Confusable Pairs:**
The `SentenceBuilder` module employs an adaptive thresholding mechanism to resolve semantic ambiguity between visually analogous signs, such as “nice” and “good”. By referencing a curated registry in `similar_signs.json`, the architecture identifies high-risk pairs and programmatically escalates the commit threshold by 8%. This rigorous validation phase fortifies the pipeline against unstable transitions and drastically improves the precision of fine-grained articulatory identification.

**Motion-Gating Efficiency and Evaluative Trade-offs:**
Initial architectural iterations utilized a motion-gating heuristic to suppress model execution during idle intervals, aiming to minimize computational overhead. However, empirical analysis demonstrated that this gating logic inadvertently penalized static grammatical holds—critical semantic pauses where a signer maintains a fixed posture. To preserve the linguistic integrity of the recognition engine, the motion-gating module was formally deprecated in favor of continuous temporal tracking in the final production deployment.

**Continual Learning and Domain Adaptation:**
To dynamically adapt to user-specific articulatory drift and unique camera setups, the system features a RESTful `/feedback` endpoint. When a user corrects a misclassified sign, the API extracts the historical 20-frame spatial buffer and commits it to a local pseudo-data repository. Once a sufficient batch of corrections is accumulated, the architecture asynchronously trains a lightweight neural Adapter Layer (a non-linear MLP mapping). To mathematically prevent catastrophic forgetting of the foundational vocabulary, the adapter training process blends user-provided labels with soft-pseudo labels generated by the base ONNX ensemble, yielding a personalized classification head without corrupting the primary weights.

**Deterministic Validation and Infrastructure Monitoring:**
Quality assurance is sustained via a comprehensive `pytest` suite, integrated into a GitHub Actions CI/CD pipeline. This framework executes automated unit tests to verify mathematical normalization and feature integrity, alongside end-to-end integration tests that profile full-pipeline throughput. Continuous performance monitoring ensures that the optimized inference engine consistently operates within the strict real-time latency parameters required for extreme edge-device viability.

*(Note for formatting: Insert SCREENSHOT OF END-TO-END EXECUTION FLOW and Graph convolution execution process diagrams here)*

**Continuous Sign Language Training Pipeline:**
The repository features a specialized implementation within `src/train_continuous.py` that facilitates the transition from isolated gesture identification to the interpretation of continuous sign sequences. By utilizing the `ContinuousDataset` class to wrap the foundational `ISLDataset`, the architecture integrates boundary-noise augmentation alongside synthetic transition modeling. This optimization workflow is executed in two distinct phases: an initial stage focused on training with high-fidelity isolated samples, followed by a secondary fine-tuning phase that leverages historical archives and synthetic transitional data to fortify model robustness during fluid, real-world signing.

**Real-Time User Interface Rendering:**
To sustain high-fidelity visual feedback during data acquisition and live inference, a dedicated rendering infrastructure is provided in `src/ui/renderer.py`. Leveraging the OpenCV framework, this module projects extracted landmark coordinates and skeletal topologies directly onto the primary video frames. This real-time overlay enables immediate visual verification of joint tracking accuracy and substantially enhances the interactivity of the live perception pipeline.

**Command-Line Interface (CLI) Orchestrator:**
The system entry point is managed by `main.py`, which functions as a unified command-line routing hub for the various architectural subsystems. The CLI exposes specific flags for executing live webcam inference (`--webcam`), launching K-fold cross-validation (`--kfold`), initiating preprocessing routines (`--preprocess`), and orchestrating the collection of novel training benchmarks (`--collect`). By centralizing these operations, the interface streamlines pipeline management and significantly improves the accessibility of the underlying engineering modules.

---

### 6.3.2 REST API and Asynchronous Infrastructure
The REST API architecture, implemented within `api/app.py`, establishes a high-concurrency ASGI backend designed to facilitate real-time WebSocket communication. By leveraging the FastAPI framework, the system maintains persistent, full-duplex connections that support ultra-low-latency inference while enforcing rigorous validation protocols on all incoming feature payloads.

The architectural implementation of the API and its concurrency strategy are structured as follows:

**Persistent WebSocket Orchestration and Task Delegation:**
The primary `/ws/translate` gateway manages stateful sessions, appending 506-dimensional landmark vectors to a rolling `collections.deque` buffer. To preserve the exact temporal context of a 20-frame sliding window without blocking the asynchronous event loop, the system offloads intensive ONNX inference operations to a background `ThreadPoolExecutor`.

To mitigate I/O bottlenecks and sustain real-time performance during high-throughput transmission, the architecture employs a proactive request-throttling mechanism. By dynamically monitoring the `pending_tasks` registry, the system enforces a strict concurrency ceiling; once the threshold of two simultaneous operations is reached, incoming frame packets are programmatically discarded (load-shedded) to ensure the stability of the active inference stream.

```python
@app.websocket("/ws/translate")
async def websocket_translation_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Session-specific state containers
    sequence_deque = collections.deque(maxlen=20)
    sentence_builder = SentenceBuilder()
    pending_tasks = set()

    try:
        while True:
            data = await websocket.receive_json()
            payload = LandmarkFrameSchema(**data)

            sequence_deque.append(payload.features)

            # Request control mechanism (Load Shedding)
            if len(pending_tasks) >= 2:
                continue

            if len(sequence_deque) == 20:
                input_array = np.array(list(sequence_deque), dtype=np.float32)

                # Execute inference asynchronously
                loop = asyncio.get_running_loop()
                task = loop.run_in_executor(
                    thread_pool,
                    model_wrapper.predict,
                    input_array
                )

                pending_tasks.add(task)

                def handle_result(fut):
                    pending_tasks.discard(fut)

                task.add_done_callback(handle_result)

    finally:
        await websocket.close()
```

**Pydantic Serialization and Schema Validation:**
To guarantee the mathematical integrity of data entering the inference engine, the `api/schemas.py` module formalizes a strict `LandmarkFrameSchema`. This structural contract mandates exactly 506 floating-point features per packet, utilizing Pydantic validation to forcefully reject malformed or incomplete JSON payloads. This preventive ingress strategy ensures that semantic interpretation only occurs on structurally valid inputs, wholly isolating the downstream neural architecture from transmission-level noise.

**Latency Profiling and Throughput Metrics:**
The performance of the API was rigorously evaluated through client telemetry and simulated concurrency stress testing. Experimental data validates that WebSocket packet processing incurs less than 1 ms of latency, while the overhead of thread-pool delegation remains stabilized at roughly 0.35 ms per cycle. Under simulated workloads of highly concurrent requests, the system sustains a fluid, real-time throughput. Ultimately, driven by the highly optimized 6.22 ms ONNX inference core, the backend successfully bridges the gap between theoretical laboratory accuracy and robust, real-time edge viability.
