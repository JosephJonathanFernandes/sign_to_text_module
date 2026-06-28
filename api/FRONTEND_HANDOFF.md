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
