# ISL Sign-to-Text API — Frontend Integration Guide

> **For Akaash** — everything you need to integrate the Next.js frontend with the backend is in this file.

---

## Quick Start

```bash
# Start the backend (run from project root)
python run_api.py

# With debug mode (shows top-5 probabilities per frame)
DEBUG=true python run_api.py
```

API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Check model status + get config dimensions |
| `POST` | `/predict` | Single-shot stateless inference (testing only) |
| `POST` | `/validate_features` | Validate your JS feature extraction matches backend |
| `WS` | `/ws/translate` | **Real-time streaming translation** (main endpoint) |

---

## Architecture

```
Browser (Next.js)
    │
    │  You only need MediaPipe .task files (hand_landmarker, face_landmarker)
    │  You do NOT need the .pth model files — those stay on the backend
    │
    ├─► GET /health           (once at startup — verify compatibility)
    │
    └─► WS /ws/translate      (continuous — stream one frame per camera tick)
```

---

## Step 1 — Health Check Before Opening WebSocket

Always verify backend compatibility before streaming.

```javascript
const res = await fetch("http://localhost:8000/health");
const backend = await res.json();

if (backend.schema_version !== "1.0" || backend.feature_dimension !== 506) {
    showError("Backend version mismatch — expected schema 1.0, 506 dims");
    return;
}
// Safe to open WebSocket
```

**Response shape:**
```json
{
  "status": "healthy",
  "schema_version": "1.0",
  "feature_dimension": 506,
  "sequence_length": 20,
  "model_loaded": true,
  "num_classes": 153,
  "device": "cpu"
}
```

---

## Step 2 — Build the 506-Dimensional Feature Vector

Each frame you send must be a flat array of **506 floats**. Split as:

| Indices | Content |
|---------|---------|
| `0–62` | Left hand normalized (21 landmarks × 3 coords) |
| `63–125` | Right hand normalized (21 landmarks × 3 coords) |
| `126–188` | Left hand face-relative (21 × 3) |
| `189–251` | Right hand face-relative (21 × 3) |
| `252` | Proximity (min hand-to-face distance) |
| `253–505` | Velocity — frame-to-frame delta of indices 0–252 |

### Hand normalization (indices 0–125)
```javascript
function normalizeHand(landmarks) {
    const wrist = landmarks[0];                          // landmark 0
    const centered = landmarks.map(lm => ({
        x: lm.x - wrist.x,
        y: lm.y - wrist.y,
        z: lm.z - wrist.z,
    }));
    const scale = Math.max(...centered.map(lm =>
        Math.sqrt(lm.x**2 + lm.y**2 + lm.z**2)
    )) || 1.0;
    return centered.flatMap(lm => [lm.x/scale, lm.y/scale, lm.z/scale]);
}
```

### Face-relative normalization (indices 126–251)
```javascript
function faceRelative(handLandmarks, faceLandmarks) {
    const nose = faceLandmarks[1];                       // nose tip
    const leftEye = faceLandmarks[33];
    const rightEye = faceLandmarks[263];
    const faceScale = Math.sqrt(
        (leftEye.x - rightEye.x)**2 +
        (leftEye.y - rightEye.y)**2
    ) || 1.0;
    return handLandmarks.flatMap(lm => [
        (lm.x - nose.x) / faceScale,
        (lm.y - nose.y) / faceScale,
        (lm.z - nose.z) / faceScale,
    ]);
}
```

### Handling missing data
- Hand not detected → fill its 63 values with `0.0`
- Face not detected → fill face-relative values with `0.0`, set proximity to `1.0`
- **Never send `NaN` or `Infinity`**

### Velocity (indices 253–505)
```javascript
// First frame: send zeros
let prevSpatial = new Array(253).fill(0.0);

function getVelocity(currentSpatial) {
    const velocity = currentSpatial.map((v, i) => v - prevSpatial[i]);
    prevSpatial = [...currentSpatial];
    return velocity;
}
```

---

## Step 3 — WebSocket Streaming

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/translate");

// Send one frame per MediaPipe result callback
function onMediaPipeResult(result) {
    const spatial  = buildSpatialFeatures(result);   // indices 0–252
    const velocity = getVelocity(spatial);            // indices 253–505
    const features = [...spatial, ...velocity];       // 506 floats total

    ws.send(JSON.stringify({
        type: "landmarks",
        schema_version: "1.0",
        feature_dimension: 506,
        sequence_length: 20,
        features: features,
        timestamp: Date.now(),
    }));
}

// When user finishes signing
function stopSigning() {
    ws.send(JSON.stringify({ type: "stop" }));
}

// Clear and start again
function clearSession() {
    ws.send(JSON.stringify({ type: "clear" }));
}
```

---

## Step 4 — Handle Backend Responses

```javascript
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    switch (msg.type) {

        case "prediction":
            // Called on every frame once buffer is full (after first 20 frames)
            // msg.word is null if confidence is below threshold
            if (msg.word) updateSubtitle(msg.word);
            updateConfidenceBar(msg.confidence);
            updateSentenceSoFar(msg.sentence_so_far);
            break;

        case "translation":
            // Called when user sends "stop" — final NLP-processed sentence
            showFinalSentence(msg.text);   // e.g. "Please help me"
            console.log("Words:", msg.words);
            break;

        case "emergency_alert":
            // Called when a high-confidence emergency sign is detected
            // AFTER temporal smoothing — reliable, not jittery
            handleEmergency(msg);
            break;

        case "cleared":
            resetUI();
            break;

        case "error":
            console.error("Backend error:", msg.message);
            break;
    }
};
```

### Emergency alert payload
```json
{
  "type": "emergency_alert",
  "word": "HELP",
  "confidence": 0.9134,
  "severity": "critical",
  "timestamp": 1720936400000,
  "session_id": "a3f9c1d2..."
}
```

```javascript
function handleEmergency(msg) {
    const isCritical = msg.severity === "critical";

    // Show banner — red for critical, amber for warning
    showEmergencyBanner(msg.word, isCritical ? "red" : "amber");

    // Vibrate if supported (works on Android Chrome)
    if ("vibrate" in navigator) {
        navigator.vibrate(
            isCritical ? [400, 100, 400, 100, 400] : [200, 100, 200]
        );
    }

    // Log to session-local history table
    alertHistory.push({
        time: new Date(msg.timestamp).toLocaleTimeString(),
        word: msg.word,
        confidence: (msg.confidence * 100).toFixed(1) + "%",
        severity: msg.severity,
    });
    renderAlertHistory();
}
```

**Severity values:**

| Severity | Signs | Suggested Color |
|----------|-------|----------------|
| `critical` | help, fire, danger, emergency, police, accident, ambulance, earthquake, flood, tsunami, cyclone, cpr | Red `#DC2626` |
| `warning` | stop, doctor, hospital, injury, pain, exit, fire_extinguisher | Amber `#D97706` |
| `info` | safe | Green `#16A34A` |

---

## Step 5 — Validate Your Feature Extraction (Do This Once)

Before going live, confirm your JS extraction matches the Python backend exactly.

```javascript
const res = await fetch("http://localhost:8000/validate_features", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        schema_version: "1.0",
        raw_landmarks: {
            left_hand:  [...63 raw MediaPipe floats...],
            right_hand: null,
            face:       [...1404 raw MediaPipe floats...],
        },
        features: [...your 253 spatial features (no velocity)...],
    }),
});
const validation = await res.json();
// validation.mae should be < 1e-5
// validation.valid should be true
```

---

## Debugging

Run backend in debug mode:
```bash
DEBUG=true python run_api.py
```

This adds a `debug` object to every `prediction` message:
```json
{
  "type": "prediction",
  "word": "HELLO",
  "confidence": 0.94,
  "debug": {
    "top5": [{"word": "HELLO", "confidence": 0.94}, ...],
    "raw_confidence": 0.91,
    "stable_class": 46
  }
}
```

### Diagnostic matrix

| `sequence_hash` | `frame_delta` | Meaning | Fix |
|---|---|---|---|
| identical | `0` | Frontend frozen / WebSocket stalled | Check camera loop is running |
| changes | `≈ 0` | Landmarks static / sensor jitter | Move hands more |
| changes | `> 0` | ✅ Data flowing normally | — |
| changes | `> 0` + `raw_prediction` stuck | Model not recognising gesture | Check normalization |
| changes | `> 0` + `stable_prediction` stuck | Temporal smoother locked | Check `switch_debug` logs |
| `stable_prediction` changes | — + sentence repeats | SentenceBuilder cooldown | Pause between signs |

Enable full debug logs:
```bash
LOG_LEVEL=DEBUG python run_api.py
```

---

## Constraints — Do Not Change These

- ❌ Do not send image frames or base64
- ❌ Do not change feature ordering or normalization
- ❌ Do not send `NaN` or `Infinity`
- ❌ Do not send cumulative velocity (frame-to-frame only)
- ✅ Missing hand/face → fill with `0.0`
- ✅ First frame velocity → all `0.0`
- ✅ Always call `/health` before opening the WebSocket
