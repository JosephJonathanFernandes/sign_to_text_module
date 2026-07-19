## 4 SIGN LANGUAGE RECOGNITION SYSTEM DESIGN

### 4.1 System Architecture and Data Flow
The Sign Language Recognition Module functions as a high-fidelity spatiotemporal perception layer, engineered to facilitate the real-time classification of continuous Indian Sign Language (ISL) gestures via standard RGB video streams. To attain real-world viability, the architecture is rigorously optimized to satisfy strict sub-10ms latency benchmarks on standard consumer-grade CPU hardware. This module encapsulates the comprehensive interpretation lifecycle, bridging the gap between raw visual ingestion and the synthesis of stabilized, grammatically parsed linguistic tokens.

The internal architectural pipeline and data flow governing the recognition framework are organized according to the following functional stages:

**Feature Extraction and Adaptive Preprocessing:** 
The system ingests video at 30 FPS and utilizes the MediaPipe Holistic framework to extract high-precision 3D hand and facial landmarks. To maximize computational efficiency, resource-heavy HOG-based person detection is suppressed, while an adaptive interval mechanism throttles processing during static physiological states. Extracted coordinates undergo regional reference-based normalization (e.g., anchoring to central nodes) and are concatenated with finite-difference velocity descriptors. This condenses the spatial data into an abstract, posture-invariant 506-dimensional feature vector.

**Continuous Sequence Buffering:** 
To preserve the essential temporal context of a gesture lifecycle, incoming vectors are dynamically appended to a stateful FIFO circular buffer. This sliding window maintains a strict 20-frame temporal context, strictly managing memory by automatically evicting historical data to prevent prohibitive garbage collection pauses during live, uninterrupted inference.

**Hybrid Model Inference:** 
Upon reaching buffer capacity, the 20-frame sequence is dispatched to an aggressively quantized ONNX inference engine. The underlying mathematical architecture integrates a Spatial Graph Convolutional Network (GNN) to explicitly map physiological joint topologies, operating in tandem with a Bidirectional GRU (Bi-GRU) for tracking temporal dependencies. A hybrid proximity-aware attention mechanism assigns semantic weight to specific transition frames, ultimately generating a logit distribution across the expanded **300 ISL gloss categories**.

**Temporal Post-Processing and Output Generation:** 
To mathematically suppress high-frequency prediction jitter (motion epenthesis), a momentum-based state machine applies temporal hysteresis to the model outputs. Tokens are strictly committed to the sentence builder only after achieving majority consensus across recent frame windows and exceeding a dynamic confidence threshold engineered to resolve visually confusable sign pairs.

*(Note for formatting: Insert RECOGNITION PIPELINE DATA FLOW DIAGRAM, MODEL ARCHITECTURE DIAGRAM, and SENTENCE BUILDER STATE MACHINE DIAGRAM here)*

### 4.2 Algorithmic Design and Optimization
To sustain high predictive accuracy across diverse user morphologies while optimizing for extreme edge-device deployment, the recognition engine utilizes highly specific deterministic and stochastic algorithms. These mechanisms are deeply integrated across the preprocessing, inference, and post-processing layers to guarantee structural robustness and sub-millisecond computational efficiency.

The core algorithmic foundations of the module are organized as follows:

**Spatial Graph Convolutions and Temporal Attention:** 
Diverging from traditional flat vector-based models, the Lightweight Spatial GNN module employs normalized Laplacian matrix multiplication to extract features that preserve exact anatomical topology. Temporal dependencies are captured by the Bi-GRU and enhanced by a Hybrid Proximity-Aware Attention algorithm, which applies a Gaussian bias derived from hand-to-face proximity to prioritize critical articulatory frames over transitional noise.

**Velocity Encoding and Spatial Normalization:** 
To isolate articulatory dynamics from static handshapes, the pipeline implements Finite Difference Velocity Encoding. Wrist-anchored normalization stabilizes the input geometry by anchoring the skeletal structure to the origin and scaling coordinates by maximum Euclidean distance. This ensures strict scale and translation invariance across varying camera depths and signer proportions.

**Ensemble Averaging and Quantization Compression:** 
Inference overhead is drastically reduced via ONNX dynamic quantization, mathematically compressing the neural weights from FP32 to INT8 to minimize the RAM footprint (yielding a ~4x reduction). During ensemble evaluation, the system utilizes Logit Averaging to strictly preserve prediction rank-order while maintaining high-throughput computational efficiency (achieving 160+ FPS).

**Momentum-Based Hysteresis and Grammar Mapping:** 
The post-processing layer employs a temporal hysteresis algorithm to stabilize inter-sign transitions, utilizing a mathematically defined stability gate (`momentum_commit_count`) to ensure majority consensus. Once committed, a rule-based NLP processor applies heuristic mappings to bridge the syntactic modality gap locally, circumventing the latency overhead associated with external cloud service calls.

*(Note for formatting: Insert FEATURE EXTRACTION AND PREPROCESSING DIAGRAM here)*

### 4.3 API Infrastructure and Concurrency Strategy
The REST API module fundamentally decouples the computationally intensive machine learning pipeline from the client application, providing a highly scalable, platform-agnostic WebSocket interface. This infrastructure supports multiple concurrent sessions while strictly protecting the real-time inference loop from network-induced performance degradation or catastrophic operational failures.

The concurrency and infrastructure strategies are organized as follows:

**Asynchronous Task Delegation:** 
To prevent blocking the FastAPI event loop, the system leverages a `ThreadPoolExecutor` to optimally offload heavy prediction tasks to isolated background threads. This concurrent strategy ensures the WebSocket connection remains highly responsive to incoming network packets without incurring dropped frames or pipeline stalling.

**Request Flood Protection and State Management:** 
A drop-frame logic layer proactively monitors the inference queue depth; if processing thresholds are exceeded, incoming packets are programmatically discarded (load-shedding) to maintain strict real-time synchronization. The API securely isolates sliding windows using asynchronous session tracking, ensuring safe horizontal scalability across multiple distinct clients.

**Pydantic Schema Contract Validation:** 
Strict serialization contracts structurally enforce data integrity, ensuring incoming JSON payloads contain the mathematically exact 506-dimensional vector required for interpretation. Malformed requests are immediately rejected at the ingress layer with descriptive HTTP error codes to actively prevent downstream neural pipeline instability.

*(Note for formatting: Insert REST API & WEBSOCKET LIFECYCLE DIAGRAM and API REQUEST FLOW SEQUENCE DIAGRAM here)*
