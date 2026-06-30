# ISL Sign-to-Text Frontend Integration Handoff

**Backend status:** Ready for frontend integration testing.

## 0. Model File Architecture (Who needs what?)

**Clarification:** You **DO NOT** need the PyTorch `.pth` files. 

The architecture is strictly separated:
- **Frontend (Browser):** You only need the standard Google MediaPipe `.task` files (e.g., `hand_landmarker.task`, `face_landmarker.task`) to run the JS extraction.
- **Backend (API):** The server holds the `.pth` model and executes the inference. You simply stream the JSON numbers to it and receive text back.

---

## Required Backend Endpoints

### 1. Health check
`GET /health`

**Expected use:**
* Call before opening WebSocket
* Verify compatibility

**Expected response:**
```json
{
  "status": "healthy",
  "schema_version": "1.0",
  "feature_dimension": 506,
  "sequence_length": 20,
  "model_loaded": true
}
```

### 2. Feature validation
`POST /validate_features`

**Purpose:**
* Verify frontend MediaPipe preprocessing exactly matches backend expectations
* Compare frontend-generated features against backend ground truth

### 3. Real-time inference
**WebSocket:**
`/ws/translate`

**Payload format:**
```json
{
  "type": "landmarks",
  "schema_version": "1.0",
  "feature_dimension": 506,
  "sequence_length": 20,
  "features": [/* 506 floats */],
  "timestamp": 1698765432000
}
```

## Frontend Workflow

1. Camera
2. ↓ MediaPipe JS
3. ↓ Generate landmarks
4. ↓ Apply `FEATURE_CONTRACT.md` transformations
5. ↓ Generate 506-dimensional vector
6. ↓ Optional: validate via `/validate_features`
7. ↓ Connect to `/ws/translate`
8. ↓ Stream features continuously

## Important Constraints

* **Do not send image frames**
* **Do not send base64 images**
* **Do not change feature ordering**
* **Do not change normalization logic**
* **Missing landmarks** → fill with zeros
* **Velocity** must be frame-to-frame only (not cumulative)

## Reference Documents
* `FEATURE_CONTRACT.md` (root directory)
* `README.md`
* `api/simulate_frontend.py` (example integration script)

## Debugging the Stream

If you encounter issues where the model gets "stuck" on a single prediction or fails to detect new signs, you can run the API server in debug mode to see exactly what the frontend is sending:

```bash
LOG_LEVEL=DEBUG python run_api.py
```

This will output real-time `temporal_debug` logs containing:
- `sequence_hash`: A jitter-resistant hash of the 20-frame buffer you are sending. If this stays identical, your frontend stream is frozen.
- `frame_delta`: The numeric difference between the last two frames. If this is `0.0`, the landmarks are static.
- `raw_prediction`: What the ML model sees based on your landmarks.
- `stable_prediction`: What the API will actually return to you (after temporal smoothing).

See the **Debugging & Observability** section in `api/README.md` for a full diagnostic matrix on how to interpret these logs.
