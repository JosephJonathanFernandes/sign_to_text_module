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
