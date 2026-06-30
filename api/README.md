# ISL Sign-to-Text API

This directory contains the production-ready inference API wrapper for the existing ISL model pipeline.

## Features

- **No ML Code Changes**: Wraps existing `ensemble_predict`, `TemporalPostProcessor`, and `SentenceBuilder` without modifications.
- **Dynamic Shapes**: Sequence length (`20`) and feature dimensions (`506`) are loaded dynamically from `config.py`.
- **FastAPI + WebSockets**: Built for low-latency streaming and isolated per-connection sessions.
- **Flood Protection**: Safely discards incoming frames if inference starts lagging, ensuring real-time responsiveness.
- **Warmup**: Automatically burns off PyTorch startup overhead at server launch.

---

## 🚀 Running the API

```bash
# Basic run
python run_api.py

# Run with debug top-5 probabilities included in response
DEBUG=true python run_api.py
```

---

## 🧪 Testing

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Quick Smoke Test (Python)
We've included a script to test both HTTP `/predict` and WebSocket `/ws/translate` locally.
```bash
python api/test_api.py
```

### 3. Manual WebSocket Test (wscat)
```bash
wscat -c ws://localhost:8000/ws/translate

# Send a frame (replace with actual 506 floats)
> {"type": "landmarks", "features": [0.0, ... 506 floats total]}

# Send stop command to finalize sentence
> {"type": "stop"}
```

---

## 🔍 Debugging & Observability (For Akash)

If you are experiencing issues where the API gets "stuck" on a single prediction or fails to recognize new gestures, run the API with the debug log level to activate the diagnostic toolchain:

```bash
LOG_LEVEL=DEBUG python run_api.py
```

This will enable two high-frequency loggers:
1. **`temporal_debug`**: Tracks the exact sequence hash, frame delta, and confidence scores at each step of the pipeline.
2. **`switch_debug`**: Tracks the dynamic hysteresis threshold used to transition between signs.

### How to read the Diagnostic Matrix

Observe the `temporal_debug` logs during a session. You can isolate the exact point of failure using this matrix:

- **`sequence_hash` identical** AND **`frame_delta = 0`**
  ➡️ **Pipeline frozen**: The frontend has stopped sending frames, or the WebSocket queue is stalled.
- **`sequence_hash` changes** AND **`frame_delta ≈ 0`**
  ➡️ **Sensor jitter**: The landmarks are effectively static; no real movement is happening.
- **`sequence_hash` changes** AND **`frame_delta > 0`**
  ➡️ **Real movement**: The data is flowing correctly from the camera to the model.
- **Real movement** AND **`raw_prediction` changes** AND **`stable_prediction` stuck**
  ➡️ **Postprocessor lock**: The temporal smoother is refusing to transition. Check `switch_debug` logs to see if the dynamic hysteresis threshold is too tight.
- **Real movement** AND **`raw_prediction` stuck**
  ➡️ **Model issue**: The ML model itself is failing to recognize the new gesture features.
- **`stable_prediction` changes** BUT **sentence repeats the same word**
  ➡️ **Sentence Builder issue**: Check the cooldown or transition logic in `SentenceBuilder`.

