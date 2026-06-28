# ISL Sign-to-Text System

[![CI Build](https://github.com/JosephJonathanFernandes/sign_to_text_module/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephJonathanFernandes/sign_to_text_module/actions/workflows/ci.yml)
[![Test Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

A real-time Indian Sign Language (ISL) recognition system. This module extracts 3D skeletal landmarks from video using Google MediaPipe and classifies continuous gestures using a hybrid BiGRU + Spatial GNN deep learning architecture.

This project is part of a Final Year Project (FYP) focused on low-latency, real-time edge inference for sign language translation.

## 🚀 Features

- **Real-Time Inference:** Sustained 60+ FPS processing on standard CPU hardware (12-15ms latency per frame).
- **Temporal & Spatial Modeling:** Combines Graph Neural Networks (GNN) for spatial hand dynamics and Bidirectional GRUs for temporal sequence modeling.
- **Robust Feature Extraction:** 253-dimension feature vector capturing relative face-hand proximity and normalized joint coordinates.
- **WebSocket Streaming:** Native continuous sign-to-text API for seamless frontend integration.
- **Optimized Data Pipeline:** HDF5-backed dataset storage yielding a 5.4× faster epoch execution and 209× faster initialization compared to legacy filesystems.

---

## 🏗 Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed diagrams and pipeline explanations.

The system is composed of four main layers:
1. **Frontend / Camera:** Captures video and optionally extracts MediaPipe landmarks.
2. **API (FastAPI):** WebSocket and REST endpoints handling incoming landmark frames.
3. **Inference Engine:** A stateful pseudo-buffer processing temporal sequences via ONNX-accelerated PyTorch models.
4. **Post-Processing:** Sentence building, confidence smoothing, and natural language correction.

---

## ⏱️ Measured Results (test environment)

*Evaluated on: Intel Core i7 (11th Gen), CPU-only inference.*

| Metric | Measurement |
|--------|-------------|
| **Inference latency:** | `~12 ms` per frame sequence |
| **Dataset initialization:** | `71.14 s → 0.18 s` (≈391× improvement via HDF5) |
| **Epoch execution time:** | `98.58 s → 18.28 s` (≈5.4× improvement via HDF5) |
| **Frame processing:** | `Up to 60 FPS` under evaluated hardware |

---

## ⚙️ Quick Start

### 1. Developer Setup
Clone the repository and run the setup script to install dependencies, virtual environments, and pre-commit hooks.

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
venv\Scripts\activate
```

**Linux / macOS:**
```bash
bash scripts/setup.sh
source venv/bin/activate
```

### 2. Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```
Ensure `ALLOWED_ORIGINS` is configured correctly for your frontend (default is `http://localhost:3000,http://localhost:5173` for dev).

### 3. Run the API Server
Start the FastAPI server:
```bash
python run_api.py
```
The server will start on `http://localhost:8000`. WebSocket translation is available at `ws://localhost:8000/ws/translate`.

---

## 🧪 Testing

The repository uses `pytest` for all unit and API testing.

```bash
# Run unit tests (no model dependencies)
pytest tests/unit/

# Run API endpoint schemas tests (mocked models)
pytest tests/api/

# Run all tests with coverage
pytest tests/ -v --cov=src --cov=api
```

---

## 🛠 Project Structure

The project has been refactored into a modular, production-grade structure.

- `src/` — Core machine learning logic (config, inference, training, preprocessing)
- `api/` — FastAPI application and WebSocket endpoints
- `tests/` — Test suites (unit, api, integration, e2e)
- `tools/` / `scripts/` — Development and CI utilities
- `assets/` — (Gitignored) Compiled models and HDF5 datasets
- `docs/` — Technical documentation

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, pre-commit hooks, and the pull request process.

Please read the [SECURITY.md](SECURITY.md) before reporting any vulnerabilities.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
