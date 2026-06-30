# 1. Repository Overview

This repository contains the full pipeline for the ISL Sign-to-Text Recognition system. The core approach utilizes a BiGRU neural network trained on MediaPipe hand/face landmarks extracted from video inputs.

### Folder Structure
* `api/` – Scripts for running the system as an API server (FastAPI).
* `assets/` – Static files and resources (compiled models and HDF5 datasets).
* `data/` – Usually contains the raw video datasets (may need to be created manually if `.gitignore` is used).
* `docs/` – Technical documentation and architecture diagrams.
* `experimental/` – Experimental scripts and notebooks.
* `logs/` – Execution and training logs.
* `models/` – Contains saved PyTorch checkpoints and ONNX model files.
* `pseudo_data/` – Synthetic or temporary data storage.
* `scripts/` – Utility scripts for augmentations, quantization, evaluation, and exporting.
* `src/` – The core application codebase.
  * `src/augmentations/` – Specialized scripts for continuous sign training data generation (`boundary_noise.py`, `transition_generator.py`).
  * `src/config/` – Configuration settings.
  * `src/core/` – Central logic, model configuration, and pipeline integrations. `main.py` here handles parsing CLI arguments.
  * `src/inference/` – Post-processing, ensemble logic, ONNX loading.
  * `src/preprocessing/` – MediaPipe landmark extraction, data loaders, cleaning.
  * `src/shared/` – Shared utilities and feature extractors.
  * `src/tools/` – Development utilities inside the source package.
  * `src/training/` – Model architecture (`model.py`), loss functions, training loops.
  * `src/ui/` – Any UI components.
  * `src/utils/` – Logging and helpers.
* `tests/` – Unit, API, integration, and e2e test suites.
* `tools/` – Root-level development and CI utilities.
* `venv/` – Python Virtual Environment.

### Data Flow Through The Project
```text
raw_videos/
↓ (preprocessing)
landmark extraction (saved as .npy arrays in processed/)
↓ (augmentations)
augmented dataset (multiple merge variants, spatial warps)
↓ (training)
model train (BiGRU + Attention)
↓ (export)
model save (.pth / .onnx / quantized)
↓ (live inference)
webcam loop / video prediction
```

---

# 2. Environment Setup (with venv)

Use the following commands to create a clean environment and install required dependencies.

## Create venv

**Windows:**
```powershell
python -m venv venv
```
**Linux/Mac:**
```bash
python3 -m venv venv
```

## Activate venv

**Windows CMD:**
```cmd
venv\Scripts\activate
```
**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```
**Linux/Mac:**
```bash
source venv/bin/activate
```

## Install dependencies

Ensure you're using the virtual environment (your terminal prompt should show `(venv)`).

```powershell
pip install -r requirements.txt
```
To install dependencies for development (testing/linting):
```powershell
pip install -r requirements-dev.txt
```

## Verify installation
You can verify the environment by invoking the help menu of the main script:
```powershell
python main.py -h
```

---

# 3. Repository Startup Commands

The core entry point of the project is `main.py` in the root directory.

### Help Menu
**Command:** `python main.py -h`
**Purpose:** Displays all available CLI arguments and options.
**Expected output:** A list of all arguments like `--train`, `--preprocess`, etc.

### API Server Startup
**Command:** `python run_api.py`
**Purpose:** Starts the backend server to serve predictions via API.
**Expected output:** Uvicorn startup logs indicating the server is running on a local port.

---

# 4. Dataset Preparation Commands

The preprocessing pipeline converts raw `.mp4` or video files into coordinate data (`.npy` files) using MediaPipe.

### Data Collection (Interactive)
**Command:** `python main.py --collect`
**Purpose:** Start webcam interface to record your own dataset.
**Expected output:** Live webcam feed UI.
**Expected generated files:** New video files saved to dataset folders.

### Data Collection (Specific Class)
**Command:** `python main.py --collect --cls hello --n 10`
**Purpose:** Automatically record 10 samples for the class "hello".

### Video to Landmark Preprocessing
**Command:** `python main.py --preprocess`
**Purpose:** Reads raw videos (from `dataset/`) and runs MediaPipe extraction.
**Input:** Raw video files.
**Output:** `.npy` files containing temporal landmark coordinates.
**Expected generated files:** Populated `processed/` directory.

### Dataset Cleanup
**Command:** `python -m src.preprocessing.cleanup_dataset_npy`
**Purpose:** Scans the dataset directories and standardizes/cleans up the `.npy` files. Can also be invoked via `main.py --cleanup`.

### Dataset Quality & Diversity Filtering
**Command:** `python -m src.preprocessing.quality_filter_hybrid`
**Purpose:** Deep-learning powered filter that evaluates sample quality (hand visibility, motion) and diversity. Automatically moves low-quality and redundant samples into a `processed_del` archive folder.
**Note:** To run on a specific class, append `--class [CLASS_NAME]`.

### Dataset Downsampling
**Command:** `python -m src.preprocessing.random_downsample_processed`
**Purpose:** Randomly downsamples class folders inside the processed directory to a fixed threshold to prevent large class imbalances.

### Dataset Balancing
**Command:** `python -m src.preprocessing.balance_processed_dataset`
**Purpose:** Balances the processed class folders to a fixed target count, prioritizing webcam captures when duplicating.

---

# 5. Augmentation Commands

Data augmentation generates synthetic variation to improve model robustness.

### Augment Processed Landmarks
**Command:** `python main.py --augment-landmarks --augment-landmarks-n 3`
**Purpose:** Applies noise, dropout, and spatial rotation on existing `.npy` files.
**Inputs:** Existing `processed/` `.npy` files.
**Generated outputs:** New `.npy` files with suffix `_aug`.
**When to use:** To artificially increase dataset variation and prevent overfitting.

### Merge Mode Augmentation (Splicing)
**Command:** `python main.py --merge --merge-mode crossfade_splice --merge-n 2`
**Purpose:** Creates synthetic samples by merging frames from two different samples in the same class.
**When to use:** When you need temporal variations.
**When NOT to use:** When dealing with very short, sudden signs where splicing might destroy the gesture meaning.

### Raw Video Augmentation
**Command:** `python main.py --augment-videos --augment-max-per-video 4`
**Purpose:** Uses OpenCV to generate variations of raw videos (brightness, crop, blur) before MediaPipe extraction.

### Full Pipeline Scripts
**Command:** `python -m src.preprocessing.augment_pipeline`
**Purpose:** Orchestrates the entire augmentation flow (landmark aug -> merge aug -> cleanup). Run this as a module from the project root.

### Full Video Augmentation Pipeline
**Command:** `python -m src.preprocessing.augment_video_pipeline`
**Purpose:** Orchestrates systematic video augmentations covering spatial crops and visual effects. Run this as a module from the project root.

---

# 6. Training Commands

### Existing Isolated-Word Training

**Command:** `python main.py --train`
**Purpose:** Trains a single BiGRU model on the isolated-word processed dataset.
**Output model path:** Checkpoints saved to `models/` directory (e.g., `sign_language_model.pth`).
**Approximate runtime:** 10 mins to a few hours (depending on dataset size & GPU/CPU).

### Ensemble K-Fold Training
**Command:** `python main.py --kfold`
**Purpose:** Trains multiple models on different splits of the dataset for an ensemble strategy.

### Continuous-Sign Extension Training

**Command:** `python src/train_continuous.py --archived-weight 0.25`
**Purpose:** Trains the continuous signing variant using the boundary noise and synthetic transition features. It runs in two phases: Phase 1 trains on clean data, and Phase 2 fine-tunes on the archived data from `processed_del/` using the specified weight.
**Output model path:** Usually saved as `models/sign_language_continuous.pth`.
**Approximate runtime:** Same as normal training, slightly longer due to dynamic noise injection.

---

# 7. Optimization & Export Commands

To optimize your trained PyTorch models for faster CPU inference, you can export them to ONNX or apply INT8 Dynamic Quantization.

### PyTorch Native Quantization
**Command:** `python -m src.inference.quantize_model --ensemble-dir "models/ensemble" --output "models/ensemble_quantized"`
**Purpose:** Shrinks the PyTorch model size and improves CPU speed using PyTorch's native INT8 Dynamic Quantization. You can also quantize a single model by passing `--checkpoint` instead.

### Export to ONNX (Recommended)
**Command:** `python -m src.inference.export_onnx --checkpoint "models/ensemble/model_0.pth" --output "models/ensemble_onnx/model_0.onnx"`
**Purpose:** Exports the PyTorch model to ONNX format, which runs significantly faster by stripping Python overhead and using a highly optimized C++ backend. *(Run this for each model in your ensemble).*

### ONNX INT8 Quantization
**Command:** `python -m src.inference.quantize_onnx --model "models/ensemble_onnx/model_0.onnx" --output "models/ensemble_onnx/model_0_int8.onnx"`
**Purpose:** Takes an exported ONNX model and compresses it to INT8 for the absolute smallest file size and fastest inference.

---

# 8. Evaluation Commands

### Inference Benchmarking
**Command:** `python scripts\benchmark_inference.py`
**Purpose:** Benchmarks the current ensemble/model for throughput (FPS) and latency.

### Evaluate Quantized Model
**Command:** `python scripts\evaluate_quantized_model.py`
**Purpose:** Runs validation checks against a quantized Int8 ONNX model to verify accuracy hasn't degraded during quantization.

### Video Evaluation
**Command:** `python main.py --predict path\to\test_video.mp4`
**Purpose:** Runs the entire pipeline (extraction -> inference) on a single test video and prints probabilities.

---

# 9. Testing Commands

The project uses `pytest` for running unit, API, integration, and end-to-end tests.

### Run Unit Tests
**Command:** `pytest tests/unit/`
**Purpose:** Run lightweight unit tests that don't depend on network or model files.

### Run API Tests
**Command:** `pytest tests/api/`
**Purpose:** Run tests for API endpoint schemas (uses mocked models).

### Run All Tests with Coverage
**Command:** `pytest tests/ -v --cov=src --cov=api`
**Purpose:** Execute the entire test suite and generate a coverage report for the `src` and `api` directories.

---

# 10. Live Inference Commands

### Webcam Inference (Standard)
**Command:** `python main.py --webcam`
**Purpose:** Boots up the OpenCV webcam feed with the live isolated-word detection loop.
**Expected behavior:** Window opens, hand wireframes are drawn, and predictions are printed on screen when hand motion stops.

### Quantized Webcam Inference
**Command:** `python main.py --webcam --quantized`
**Purpose:** Runs webcam inference using the INT8 quantized ONNX model bundle for better CPU performance.

---

# 11. Continuous Signing Extension Commands

The continuous extension generates synthetic "transition" noise and boundary frames to help the model ignore the space between signs.

**Step 1:** Create or ensure your base processed dataset exists.
```powershell
python main.py --preprocess
```

**Step 2:** Ensure you are in the project root directory.
```powershell
cd C:\Users\Joseph\Desktop\projects\sign_to_text
```

**Step 3:** Run the continuous training script directly. (The dataset augmentations like `boundary_noise.py` and `transition_generator.py` are loaded dynamically by the `ContinuousDataset` loader during execution).
```powershell
python src\train_continuous.py
```

---

# 12. Full Beginner Workflow

If you just cloned the repository and want to train and test a model from scratch:

**Step 1:** Create venv
```powershell
python -m venv venv
```
**Step 2:** Activate venv
```powershell
.\venv\Scripts\Activate.ps1
```
**Step 3:** Install dependencies
```powershell
pip install -r requirements.txt
```
**Step 4:** Collect sample videos (interactive UI)
```powershell
python main.py --collect
```
**Step 5:** Preprocess videos into landmarks
```powershell
python main.py --preprocess
```
**Step 6:** Augment landmarks for variation
```powershell
python main.py --augment-landmarks
```
**Step 7:** Train model
```powershell
python main.py --train
```
**Step 8:** Run live webcam inference
```powershell
python main.py --webcam
```

---

# 13. Common Errors + Fixes

**Error:** `ModuleNotFoundError: No module named 'src'`
**Cause:** Attempting to run a script inside a subfolder (like `src/augmentations/`) where python cannot resolve the root package structure.
**Fix command:** Always run scripts from the project root: `python src/train_continuous.py`

**Error:** `ValueError: num_samples should be a positive integer value, but got num_samples=0`
**Cause:** The dataset loader failed to find any processed `.npy` files or failed to extract from the HDF5 Fast-Path cache.
**Fix command:** Run `python main.py --preprocess` to generate the processed dataset first.

**Error:** `Command 'python' not found` or `python is not recognized`
**Cause:** Python is not installed or not added to your system PATH.
**Fix command:** Install Python from python.org and check "Add to PATH" during installation.

**Error:** Cannot open webcam (cv2 errors)
**Cause:** Another application is using your webcam, or you lack camera permissions.
**Fix command:** Close Zoom/Teams/OBS. Unplug and replug the camera.

---

# 14. Quick Cheat Sheet

**SETUP**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**TRAIN**
```powershell
python main.py --preprocess
python main.py --train
python src\train_continuous.py
python main.py --kfold
```

**AUGMENT**
```powershell
python main.py --augment-landmarks
python main.py --merge
python -m src.preprocessing.augment_pipeline
python -m src.preprocessing.augment_video_pipeline
```

**OPTIMIZE & EXPORT**
```powershell
python -m src.inference.quantize_model --ensemble-dir "models/ensemble" --output "models/ensemble_quantized"
python -m src.inference.export_onnx --checkpoint "models/ensemble/model_0.pth" --output "models/ensemble_onnx/model_0.onnx"
python -m src.inference.quantize_onnx --model "models/ensemble_onnx/model_0.onnx" --output "models/ensemble_onnx/model_0_int8.onnx"
```

**TESTING**
```powershell
pytest tests/unit/
pytest tests/ -v --cov=src --cov=api
```

**EVALUATE**
```powershell
python main.py --predict "data\my_video.mp4"
python scripts\benchmark_inference.py
python scripts\evaluate_quantized_model.py
```

**LIVE INFERENCE**
```powershell
python main.py --webcam
python main.py --webcam --quantized
```
