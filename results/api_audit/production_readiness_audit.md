# Backend & ML Pipeline Production-Readiness Audit

**Date:** July 2026
**Role:** Senior ML Systems Engineer
**Scope:** Backend API (`api/`) & ML Inference Pipeline (`src/inference/`)

---

## 1. Model Lifecycle
- **Model loaded only once at startup:** ✅ Implemented
- **ONNX warm-up inference:** ✅ Implemented
- **Model reuse across requests:** ✅ Implemented
- **Proper session management:** ✅ Implemented
- **Thread safety:** ✅ Implemented

**Analysis**: Excellent implementation via FastAPI's `@asynccontextmanager lifespan`. The warmup inference effectively burns off PyTorch/ONNX initialization overhead. Model memory is shared across all concurrent sessions.

## 2. Input Validation
- **Shape validation:** ✅ Implemented
- **Data type validation:** ✅ Implemented
- **NaN checking:** ⚠️ Partially implemented
- **Infinity checking:** ⚠️ Partially implemented
- **Empty sequence handling:** ✅ Implemented
- **Invalid landmark handling:** ❌ Missing

**Analysis**: While `/validate_features` properly checks for NaNs and out-of-bounds coordinates, the high-throughput WebSocket stream in `ws_translate` **does not check for NaN/Inf** before appending to the buffer.
**Risk**: If MediaPipe glitches and sends `NaN`, `np.array(features)` will accept it, silently poisoning the ring buffer and returning junk predictions for the next 20 frames.
**Fix**: Add an explicit `if np.isnan(frame).any():` check in `ws_translate`. *(Priority: High)*

## 3. Temporal Pipeline
- **Sliding window correctness:** ✅ Implemented
- **Frame ordering:** ✅ Implemented
- **Missing frame handling:** ❌ Missing
- **Duplicate frame handling:** ⚠️ Partially implemented
- **Timestamp consistency:** ❌ Missing
- **Buffer reset logic:** ✅ Implemented

**Analysis**: The pre-allocated circular numpy buffer is very efficient. However, the backend entirely ignores the `timestamp` field in the WS payload, assuming the frontend guarantees a perfect 30 FPS. 
**Risk**: Network jitter causing frames to arrive out-of-rhythm will artificially alter the calculated velocity features.
**Fix**: In the future, track the frontend `timestamp` to interpolate missing frames. *(Priority: Medium)*

## 4. Inference Pipeline
- **Exception handling:** ✅ Implemented
- **Confidence thresholding:** ✅ Implemented
- **Reject/OOD logic:** ✅ Implemented
- **Softmax correctness:** ✅ Implemented
- **Label mapping:** ✅ Implemented
- **Invalid prediction handling:** ✅ Implemented

**Analysis**: The pipeline gracefully handles OOD concepts via idle logic and the ensemble. The sentence builder effectively gates low-confidence predictions.

## 5. WebSocket Handling
- **Connection lifecycle:** ✅ Implemented
- **Graceful disconnect:** ✅ Implemented
- **Exception handling:** ✅ Implemented
- **Multiple clients:** ✅ Implemented
- **Memory leaks:** ✅ Implemented
- **Queue management:** ❌ Incorrect implementation
- **Frame dropping strategy:** ❌ Incorrect implementation

**Analysis**: **This is the most critical bug in the system.** The flood protection logic uses `session.pending_count > MAX_PENDING`. However, the inference execution is awaited directly in the `while True:` loop (`await loop.run_in_executor(...)`). 
**Risk**: Because of the `await`, the event loop pauses reading from the WebSocket until inference completes. `pending_count` will therefore *never* exceed 1. Frames will pile up in the OS TCP buffer instead of being dropped by your application logic, causing massive unbounded latency (lag) during network spikes.
**Fix**: Inference must be spawned as an independent background task (`asyncio.create_task()`) so the `while True:` loop can continue to drain the WebSocket buffer instantly. *(Priority: Critical)*

## 6. Performance
- **Unnecessary copies:** ⚠️ Partially implemented
- **NumPy optimization:** ✅ Implemented
- **ONNX optimization:** ✅ Implemented
- **Memory allocations:** ✅ Implemented
- **Logging overhead:** ✅ Implemented
- **Blocking operations:** ❌ Incorrect implementation (See section 5)
- **Latency bottlenecks:** ✅ Implemented

**Analysis**: Pre-allocating the `(20, 506)` array avoids memory fragmentation. `np.concatenate` does create a copy when fetching the sequence, but for 20 frames, this is negligible. The primary bottleneck is the asyncio blocking mentioned above.

## 7. API Design
- **Proper response format:** ✅ Implemented
- **Error responses:** ✅ Implemented
- **Validation errors:** ✅ Implemented
- **Status codes:** ✅ Implemented
- **Health endpoint:** ✅ Implemented
- **Startup events:** ✅ Implemented

**Analysis**: Clean and idiomatic FastAPI design. Returning `feature_dimension` in `/health` is a great touch for dynamic frontend configuration.

## 8. Configuration
- **Environment variables:** ✅ Implemented
- **Hardcoded paths:** ⚠️ Partially implemented
- **Hardcoded model names:** ✅ Implemented
- **Hardcoded thresholds:** ⚠️ Partially implemented
- **Config management:** ✅ Implemented

**Analysis**: Almost all variables come from `get_config()`, but a few strings (e.g., `adapter_weights`, `assets/model_metadata.json` in metrics, and the `0.15` velocity threshold) are hardcoded in `api/app.py`.
**Fix**: Move these to `config.yaml`. *(Priority: Low)*

## 9. Robustness
- **Unexpected exceptions:** ✅ Implemented
- **Invalid websocket messages:** ✅ Implemented
- **Oversized payloads:** ❌ Missing
- **Corrupted input:** ⚠️ Partially implemented
- **Malformed JSON:** ✅ Implemented
- **Resource cleanup:** ✅ Implemented

**Analysis**: There is no explicit byte-limit on `websocket.receive_text()`. Starlette's defaults are high, making it susceptible to simple OOM attacks.
**Fix**: Check payload length or add a middleware rate-limiter. *(Priority: Low)*

## 10. Security
- **CORS:** ✅ Implemented (dynamically via `ALLOWED_ORIGINS`)
- **Input sanitization:** ⚠️ Partially implemented
- **Payload limits:** ❌ Missing
- **Secret management:** N/A

## 11. Code Quality
- **Duplicate logic:** ✅ Implemented
- **Large functions:** ⚠️ Partially implemented
- **Naming consistency:** ✅ Implemented
- **Modularity:** ✅ Implemented

**Analysis**: `ws_translate` is nearly 250 lines long. It handles JSON parsing, validation, buffer management, triggering inference, post-processing, and socket management.
**Fix**: Extract the buffer-update and inference-triggering logic into `session.py`. *(Priority: Medium)*

## 12. Production Readiness
- **Logging:** ✅ Implemented
- **Monitoring hooks:** ✅ Implemented (`/metrics` is excellent)
- **Health checks:** ✅ Implemented
- **Graceful shutdown:** ✅ Implemented
- **Configurable thresholds:** ⚠️ Partially implemented

---

# Final Assessment

### 1. Production Readiness Score: **88 / 100**
The architecture is fundamentally sound, resilient, and highly optimized. It loses points purely due to the asyncio blocking bug in the WebSocket loop, which defeats the intended flood protection.

### 2. Top Issues to Fix (Pre-Submission)
1. **Critical:** Change `await loop.run_in_executor` to `asyncio.create_task()` in `ws_translate` to prevent WebSocket buffer queueing and enable true frame-dropping.
2. **High:** Add `np.isnan(frame).any()` to the WebSocket loop.
3. **Medium:** Move `0.15` velocity threshold to `config.yaml`.
4. **Medium:** Refactor `ws_translate` to reduce its monolithic size.
5. **Low:** Fix hardcoded `adapter_weights` path in `app.py`.

### 3. Hidden Bugs & Edge Cases
- **The "Pending Count" Illusion**: As explained in Section 5, `session.pending_count` never actually exceeds 1.
- **Velocity Poisoning**: If a user's camera freezes and drops frames, then suddenly catches up, the MediaPipe coordinates will jump massively. The pipeline will interpret this as extreme velocity/motion rather than a camera glitch.

### 4. Performance Optimizations (No Retraining)
- **Zero-Copy Sequence Retrieval**: Instead of `np.concatenate` in `get_sequence()`, you could theoretically pass a stride-tricked view to ONNX if the runtime supports non-contiguous memory, saving a small memory copy. (Probably not worth the complexity).
- **Asynchronous Logging**: Move `logger.info()` calls that serialize large dictionaries to an async queue to save micro-seconds on the main thread.

### 5. Reliability Improvements
- **Automatic Session Timeout**: Sessions that remain connected but send no frames for >10 minutes should be actively severed by the backend to save memory.
- **Maximum Payload Size Check**: Add `if len(raw) > 5000: await websocket.close()` to prevent memory exhaustion from malicious JSON arrays.

### 6. Code Smells
- **Mixing I/O with Logic**: `ws_translate` contains deep business logic (idle threshold math). That logic belongs entirely inside `InferenceSession` or `TemporalPostProcessor`. The router should *only* handle socket reading/writing.
