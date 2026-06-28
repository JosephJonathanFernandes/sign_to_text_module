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
