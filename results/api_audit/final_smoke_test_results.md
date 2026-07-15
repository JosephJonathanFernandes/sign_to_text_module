# Final API Smoke Test Results

**Date:** July 2026
**Target:** `v2.0.0-final` Feature-Frozen Backend API
**Script:** `scratch/test_ws_smoke.py`

---

## 1. Normal Use
**Test:** Connect, send 30 valid frames at 30 FPS, await prediction, send stop command, verify final text.
**Result:** ✅ Passed
**Log:**
```
Received: prediction
Normal Use Passed!
Stop response: prediction
```

## 2. Disconnect Handing
**Test:** Send valid frame and abruptly close websocket connection.
**Result:** ✅ Passed (Server logged clean teardown without exceptions).

## 3. NaN/Corrupted Payload Validation
**Test:** Send frame containing explicit `NaN` values.
**Result:** ✅ Passed (Backend intercepted, logged warning, dropped the frame, and kept connection alive).

## 4. Oversized Payload Limit
**Test:** Send a massive 60,000-byte string (exceeds `MAX_PAYLOAD_SIZE` of 50KB).
**Result:** ✅ Passed
**Log:**
```
Connection closed as expected with code: 1008, reason: Policy Violation: Invalid Data
```

## 5. Rapid Burst (Stress)
**Test:** Send 100 frames instantly without waiting.
**Result:** ✅ Passed
**Log:**
```
Burst sent. Waiting for predictions to drain...
Received 81 predictions from burst.
```
*Note: As expected, the flood protection dropped excess frames while processing the maximum capacity safely without causing OOM or freezing the Uvicorn loop.*

---

**Status:** The API demonstrates rock-solid stability and handles malicious/corrupted inputs exactly per the latest configuration. 
