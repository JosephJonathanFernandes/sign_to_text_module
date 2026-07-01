# Architecture Decision Records (ADR)

This document tracks significant architectural decisions made during the development of the ISL Sign-to-Text module.

## ADR-001: Client-Side MediaPipe Extraction
**Date:** 2026-06-10

**Context:** The API needs to receive video data and convert it into 3D landmarks for inference.
**Decision:** We will execute Google MediaPipe purely on the client-side (browser via WebAssembly) and transmit only the extracted coordinate vectors (`[x, y, z]`) over the WebSocket to the API.
**Reason:** 
- Drastically reduces network bandwidth (sending kilobytes of JSON rather than megabytes of video frames).
- Eliminates heavy computer vision processing from the server, allowing the backend to scale and focus purely on the PyTorch inference bottleneck.

## ADR-002: WebSocket Streaming for Inference
**Date:** 2026-06-15

**Context:** Sign language requires continuous, sequential frame analysis. HTTP polling is too slow and introduces massive overhead.
**Decision:** Use a long-lived WebSocket connection (`/ws/translate`) for real-time inference.
**Reason:** 
- Supports continuous streaming.
- Allows the backend to maintain a stateful temporal buffer (sliding window) per user session without needing a complex caching layer like Redis.

## ADR-003: HDF5 Storage Alongside NPY (Additive)
**Date:** 2026-06-25

**Context:** The dataset contains millions of individual `.npy` files. Filesystem traversal was taking 71+ seconds just to initialize the PyTorch DataLoader, bottlenecking GPU training.
**Decision:** Compile the dataset into a single `dataset.h5` file using `h5py`, but do not delete or overwrite the existing `.npy` workflow.
**Reason:** 
- HDF5 reduced epoch times by 5.4× by eliminating file-open overhead and OS traversal.
- Preserving the `.npy` fallback guarantees backward compatibility for existing custom tools that expect the old file structure.

## ADR-004: Centralized Config Dataclass
**Date:** 2026-06-26

**Context:** Spatial dimensions (253 vs 506) and sequence lengths (20 vs 30) were hardcoded across `dataset.py`, `model.py`, and `app.py`.
**Decision:** Extract all constants into a strict Pydantic/Dataclass-style configuration file (`src/core/config.py`).
**Reason:** 
- Changing a hyperparameter in one place guarantees it propagates everywhere, preventing shape mismatch errors (`E001`) during inference.

## ADR-005: Soft Heuristic Adjustment Layer
**Date:** 2026-07-01
**Context:** The GRU model sometimes predicts anatomically impossible signs (e.g., predicting a two-handed sign when only one hand is visible) due to out-of-distribution noise or motion blur. Hard-filtering candidates causes cascading errors if a hand is momentarily occluded.
**Decision:** We implemented a Soft Heuristic Adjustment Layer that applies multiplicative penalties to raw model probabilities based on real-time observations vs JSON metadata (`data/hand_sign_classification.json`).
**Reason:** 
- Multiplicative adjustment allows the GRU to remain dominant while safely down-weighting impossible classes.
- Confidence gating (e.g., `> 0.7`) prevents noisy heuristic misdetections from ruining valid predictions.

## ADR-006: Rejection of Generative Adversarial Networks (GANs)
**Status:** Accepted
**Date:** 2026-07-01
**Context:** The dataset is limited in size for specific minority classes. We investigated whether a GAN (or TimeGAN) should be introduced to generate synthetic time-series landmark data to expand the dataset.
**Decision:** We will NOT use Generative Adversarial Networks for data augmentation. Instead, we rely on deterministic mathematical perturbations (scaling, rotation, translation, temporal masking, scattered dropout) and Phase 2 noise injection.
**Consequences:** 
- The project avoids the massive computational overhead and complexity of training a sequence GAN.
- Temporal patterns remain strictly anchored to human-recorded motion, preventing the GRU from learning synthetic distribution artifacts.
**Alternatives considered:** 
- TimeGAN/VRAE: Overly complex for skeletal data, high risk of generating temporally inconsistent signs.
- Hardcoded interpolation: Can lead to cheating if the model learns the interpolation algorithm.

## ADR-007: SentenceBuilder State Machine
**Date:** 2026-07-01
**Context:** Continuous live inference frequently produced duplicate words or falsely triggered `__transition__` classes as independent words during the chaotic movement between signs.
**Decision:** We implemented a strict debouncing state machine (`separator_counter`) inside `SentenceBuilder` that requires `__transition__` or `__reject__` to be stable for at least 3 frames before they are permitted to "break" a continuous sign block.
**Reason:** 
- Prevents stuttering (e.g., `HELLO HELLO`).
- Suppresses mid-air transition noise while preserving the ability to recognize rapid, distinct signs.

## ADR-008: Adapter Model Safety Safeguards
**Date:** 2026-07-01
**Context:** The continuous learning `AdapterModel` trains on live pseudo-labels gathered during inference, which poses a high risk of catastrophic confirmation bias if the base model hallucinates.
**Decision:** We introduced strict thresholds (`adapter_min_saved_samples = 40`, `adapter_min_classes = 3`) that constrain when adaptation is allowed during live operation.
**Reason:** 
- Reduces unstable adaptation from pseudo-labeled data.

## ADR-009: HOG Person Detection Disabled
**Date:** 2026-07-01
**Context:** Real-time webcam inference needed additional CPU headroom to maintain stable 30 FPS.
**Decision:** We intentionally disabled HOG-based person detection (`disable_hog_detection = True`).
**Reason:** 
- Shaves off ~8ms of latency per frame. We accepted a trade-off between lower latency and reduced person-aware filtering capability, assuming the background will mostly have a single signer.

## ADR-010: Spatial GNN Integration
**Date:** 2026-07-01
**Context:** The standard BiGRU model struggled to capture complex joint-to-joint spatial topologies (e.g., specific finger curls).
**Decision:** We introduced a lightweight Spatial Graph Neural Network (GNN) branch that processes explicit finger-joint connectivity.
**Reason:** 
- Improves accuracy for topologically complex signs while maintaining a relatively small parameter footprint.
