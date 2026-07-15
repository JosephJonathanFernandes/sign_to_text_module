# Architecture Decision Records (ADRs)

This document tracks the major engineering and architectural decisions made during the development of the Sign Language Recognition backend. It explicitly lists what was implemented, what was rejected, and the technical reasoning behind those choices.

---

## 1. Decisions Taken (Implemented)

### 1.1 Using ONNX Runtime instead of PyTorch in Production
* **Context**: The Bi-GRU sequence model was trained in PyTorch.
* **Decision**: Export the model to ONNX and use `onnxruntime` for inference in the FastAPI backend.
* **Why**: PyTorch has a massive memory footprint and slower CPU inference overhead. ONNX reduced inference time to ~9–12ms per frame and stripped out training dependencies, making the deployment package vastly lighter and more robust for real-time CPU environments.

### 1.2 Thread-Pool Execution for Inference
* **Context**: ONNX inference is a blocking, CPU-bound operation.
* **Decision**: Wrap the ONNX `predict` calls in `await loop.run_in_executor(None, ...)`.
* **Why**: Running CPU-bound tasks directly inside a FastAPI (asyncio) endpoint blocks the event loop, freezing all other WebSocket connections and health-check endpoints. The executor offloads this to a background thread, preserving the async loop's responsiveness.

### 1.3 FastAPI Lifespan for Model Loading
* **Context**: The ONNX engine and NLP post-processors require heavy file I/O to load into memory.
* **Decision**: Load these assets once using FastAPI's `@asynccontextmanager lifespan` and store them in `app.state`.
* **Why**: Avoids loading models per-request. Additionally, we run a "warm-up" inference inside the lifespan. Neural network libraries (PyTorch/ONNX) lazy-load execution graphs, causing the *first* inference to take significantly longer (~40ms vs ~10ms). The warm-up ensures the very first user interaction is instantly responsive.

### 1.4 Isolated Client Sessions via UUIDs
* **Context**: WebSockets stream a continuous series of frames from multiple concurrent users.
* **Decision**: Create a `Session` dictionary keyed by UUID, instantiating an independent `InferenceSession` (containing its own Ring Buffer, Temporal Smoother, and Sentence Builder) for each connection.
* **Why**: Prevents state-bleeding. If sessions weren't perfectly isolated, User A's hand movements would corrupt User B's prediction sequence.

### 1.5 Pre-allocated Circular Ring Buffer (NumPy)
* **Context**: The model requires a sliding window of the last 20 frames (shape: `20, 506`).
* **Decision**: Initialize a `np.zeros((20, 506))` array and use a `write_idx` modulo operator to overwrite the oldest frame.
* **Why**: `deque.append()` combined with `np.array(list(deque))` requires re-allocating memory and copying the entire array every 33 milliseconds. The pre-allocated circular buffer achieves zero-copy insertion, minimizing Python garbage collection overhead in the hot loop.

### 1.6 Temporal Hysteresis & Exponential Smoothing
* **Context**: Model confidence flutters wildly between similar signs (e.g., "M" vs "N") across sequential frames.
* **Decision**: Implement a `TemporalPostProcessor` using exponential moving averages and a "patience" threshold (requiring a class to maintain top probability for $N$ consecutive frames before emitting).
* **Why**: Drastically reduces false positives and UI flickering, producing a significantly smoother UX at the cost of a microscopic delay in word emission.

### 1.7 Motion-Triggered Inference (Idle State)
* **Context**: When a user sits still, the model hallucinates signs because it is forced to classify static noise.
* **Decision**: Calculate the frame-to-frame velocity delta. If the sum is below `IDLE_VELOCITY_THRESHOLD (0.15)`, force the model output to an explicit `idle/none` class.
* **Why**: Saves CPU cycles by bypassing inference during rest, and mathematically guarantees the sentence builder won't accumulate garbage predictions while the user isn't signing.

### 1.8 Dropping Corrupted Frames (Fault Tolerance)
* **Context**: Network glitches or MediaPipe failures can result in `NaN` (Not a Number) coordinates.
* **Decision**: Validate `np.isfinite(frame).all()`. If false, log a warning and `continue` (drop the frame) instead of raising an exception.
* **Why**: Dropping a single frame is barely noticeable. Throwing an exception tears down the user's entire WebSocket connection and wipes their current sentence context.

### 1.9 Continual Learning Adapter Architecture
* **Context**: The base model degrades under unfamiliar lighting or novel user body proportions.
* **Decision**: Expose a `/feedback` endpoint to capture corrections, and train a lightweight PyTorch linear layer that sits *on top* of the frozen ONNX model.
* **Why**: Re-training a Bi-GRU on the fly is too computationally expensive and requires vast amounts of historical data to prevent catastrophic forgetting. The adapter approach allows instant, cheap personalization.

---

## 2. Decisions Rejected (Not Taken)

### 2.1 Rejected: `asyncio.Queue` Producer-Consumer Architecture
* **Context**: A standard pattern to prevent network flood is decoupling WebSocket reads from processing using a queue.
* **Decision NOT Taken**: We decided against refactoring `ws_translate` into a threaded producer-consumer model.
* **Why**: 
  1. **Math**: Our benchmarked inference time is ~10ms. The frame arrival rate at 30 FPS is 33.3ms. Because the backend is 3x faster than ingestion, the OS buffer is drained instantly. There is mathematically zero lag accumulation.
  2. **Complexity**: Introducing background tasks risks race conditions updating the stateful `SentenceBuilder`, unhandled background exceptions, and complex disconnect teardown logic. The current synchronous design guarantees ordered execution without the risk.

### 2.2 Rejected: Dynamic Missing-Frame Interpolation
* **Context**: Network latency can cause frames to arrive clustered or out-of-rhythm.
* **Decision NOT Taken**: We chose not to parse frontend `timestamps` to mathematically interpolate missing coordinate frames on the fly.
* **Why**: The `TemporalPostProcessor` and Bi-GRU are robust enough to handle 1-2 dropped frames natively. Real-time interpolation adds heavy CPU overhead for a negligible accuracy gain in an academic prototype.

### 2.3 Rejected: Relational Database for Active Learning
* **Context**: The system collects user corrections for fine-tuning.
* **Decision NOT Taken**: We did not implement PostgreSQL or MongoDB to store feedback.
* **Why**: Writing raw `.npy` arrays and `.json` metadata to a flat file structure (`data/feedback/`) is vastly simpler, requires zero infrastructure overhead, and allows the PyTorch `Dataset` class to load the data directly without costly ORM translation.

### 2.4 Rejected: Stride-Tricked Zero-Copy Extraction
* **Context**: Extracting the sequential sequence from the circular buffer currently uses `np.concatenate`, which executes a memory copy.
* **Decision NOT Taken**: We did not use NumPy `as_strided` to create a zero-copy virtual view of the buffer.
* **Why**: While technically more optimal, the array is incredibly small (20 × 506 floats = ~40KB). The copy takes less than 10 microseconds. Stride tricks are notoriously difficult to debug and maintain, violating the principle of premature optimization.
