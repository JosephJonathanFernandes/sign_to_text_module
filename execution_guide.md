# 1. Repository Overview

This repository contains the full pipeline for the ISL Sign-to-Text Recognition system. The core approach utilizes a BiGRU neural network trained on MediaPipe hand/face landmarks extracted from video inputs.

### Folder Structure
* `api/` – Scripts for running the system as an API server.
* `assets/` – Static files and resources.
* `data/` – Usually contains the raw video datasets (may need to be created manually if `.gitignore` is used).
* `models/` – Contains saved PyTorch checkpoints and ONNX model files.
* `scripts/` – Utility scripts for augmentations, quantization, evaluation, and exporting.
* `src/` – The core application codebase.
  * `src/augmentations/` – Specialized scripts for continuous sign training data generation (`boundary_noise.py`, `transition_generator.py`).
  * `src/config/` – Configuration settings.
  * `src/core/` – Central logic, model configuration, and pipeline integrations. `main.py` here handles parsing CLI arguments.
  * `src/inference/` – Post-processing, ensemble logic, ONNX loading.
  * `src/preprocessing/` – MediaPipe landmark extraction, data loaders, cleaning.
  * `src/training/` – Model architecture (`model.py`), loss functions, training loops.
  * `src/ui/` – Any UI components.
  * `src/utils/` – Logging and helpers.
* `tests/` – Unit tests for the pipeline.
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
**Command:** `python main.py --cleanup --cleanup-max-aug 50`
**Purpose:** Scans the `processed/` directory and removes duplicate or excessive augmentations to balance dataset size.

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
**Command:** `python scripts\augment_pipeline.py`
**Purpose:** Orchestrates the entire augmentation flow (landmark aug -> merge aug -> cleanup).

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

**Command:** `python src\train_continuous.py`
**Purpose:** Trains the continuous signing variant using the boundary noise and synthetic transition features. Note that this script is nested inside `src/`.
**Output model path:** Usually saved as `models/sign_language_continuous.pth`.
**Approximate runtime:** Same as normal training, slightly longer due to dynamic noise injection.

---

# 7. Evaluation Commands

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

# 8. Live Inference Commands

### Webcam Inference (Standard)
**Command:** `python main.py --webcam`
**Purpose:** Boots up the OpenCV webcam feed with the live isolated-word detection loop.
**Expected behavior:** Window opens, hand wireframes are drawn, and predictions are printed on screen when hand motion stops.

### Quantized Webcam Inference
**Command:** `python main.py --webcam --quantized`
**Purpose:** Runs webcam inference using the INT8 quantized ONNX model bundle for better CPU performance.

---

# 9. Continuous Signing Extension Commands

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

# 10. Full Beginner Workflow

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

# 11. Common Errors + Fixes

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

# 12. Quick Cheat Sheet

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
python scripts\augment_pipeline.py
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
