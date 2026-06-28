# ISL Sign-to-Text Feature Contract

This document defines the exact data structures and normalizations required when integrating a frontend (e.g., MediaPipe JS in the browser) with the `sign_to_text` API.

**Failure to perfectly match this specification will result in the backend evaluating out-of-distribution data, causing silent inference failures.**

---

## 1. Schema Versioning
All WebSocket JSON payloads must include a schema version to prevent breaking changes over time.

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

### 1.1 Frontend Capability Check (Handshake)

Before initiating the WebSocket stream, the frontend **must** verify backend compatibility by querying the health endpoint.

**Endpoint:** `GET /health`

**Response:**
```json
{
   "status": "healthy",
   "schema_version": "1.0",
   "feature_dimension": 506,
   "sequence_length": 20,
   "model_loaded": true,
   ...
}
```

**Frontend Logic:**
```javascript
const response = await fetch("http://API_HOST/health");
const backend = await response.json();

if (backend.schema_version !== "1.0" || backend.feature_dimension !== 506) {
    showError("Incompatible backend version. Expected schema 1.0 and 506 dimensions.");
    return;
}
// Proceed with WebSocket stream...
```

---

## 2. Feature Vector Specification

The backend expects exactly **506** floating point dimensions per frame.

### 2.1 Spatial Features (Indices 0–252)

| Indices | Description | Details |
|---------|-------------|---------|
| `0–62` | **Left Hand (Normalized)** | 21 landmarks × 3 coordinates (x,y,z). Centered on wrist and scaled by hand size. |
| `63–125` | **Right Hand (Normalized)** | 21 landmarks × 3 coordinates (x,y,z). Centered on wrist and scaled by hand size. |
| `126–188` | **Left Hand (Face-Relative)** | 21 landmarks × 3 coordinates (x,y,z). Coordinates relative to face center. |
| `189–251` | **Right Hand (Face-Relative)** | 21 landmarks × 3 coordinates (x,y,z). Coordinates relative to face center. |
| `252` | **Proximity** | `min(norm(left_face_relative), norm(right_face_relative))`. Distance from hands to face. |

### 2.2 Temporal Velocity (Indices 253–505)

The second half of the vector (253 dimensions) represents the **frame-to-frame velocity** for all spatial features described above.

- Calculation: `velocity = current_frame_spatial - previous_frame_spatial`
- The first frame in a stream must send velocity as all `0.0`.

---

## 3. Normalization Rules

The frontend must extract coordinates from MediaPipe and apply these exact transformations BEFORE sending data to the WebSocket.

### 3.1 Hand Normalization (Indices 0–125)
For both the left and right hand:
1. Extract 21 landmarks as an array `[x1, y1, z1, x2, y2, z2 ...]`
2. Find the wrist (landmark `0`).
3. Subtract the wrist from all 21 landmarks (translation invariant).
4. Compute the maximum Euclidean distance from the wrist to any other landmark.
5. Divide all coordinates by this maximum distance (scale invariant).

### 3.2 Face-Relative Normalization (Indices 126–251)
1. Extract the face center (Nose tip, MediaPipe Face index `1`).
2. Calculate the face scale as the Euclidean distance between Left Eye (`33`) and Right Eye (`263`).
3. For both the left and right hand:
   - Subtract the face center from all 21 raw landmarks.
   - Divide all coordinates by the face scale.

### 3.3 Handling Missing Data
- If a hand is missing, output `0.0` for all 63 dimensions of that hand.
- If the face is missing, output `0.0` for face-relative features and set proximity to `1.0`.
- Do **not** send `NaN` or `Infinity`. Reject or fill with `0.0`.

---

## 4. Validation Tolerances

Before deploying, developers can validate their frontend Javascript pipeline against the Python ground-truth by sending RAW landmarks to the validation endpoint.

**Endpoint:** `POST /validate_features`

**Request:**
```json
{
  "schema_version": "1.0",
  "raw_landmarks": {
    "left_hand": [0.1, 0.2, ... 63 total],
    "right_hand": null,
    "face": [ ... 1434 total from MediaPipe ]
  },
  "features": [ ... 253 transformed features (without velocity) ]
}
```

**Tolerances:**
- **Datatype**: `float32` (or JSON standard floats).
- **Coordinate Range**: Valid normalized coordinates typically fall in `[-3.0, 3.0]`.
- **Maximum Acceptable Error (MAE)**: Frontend implementation must have an MAE of `< 1e-5` when compared to the backend `src.shared.feature_extractor`.
