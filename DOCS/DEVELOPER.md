# Developer Guide

This file documents developer-facing utilities, debug scripts, and workflows present in the repository.

Contents
- Quick verification commands
- Checkpoints and naming conventions
- K-Fold full fine-tuning
- Data quality and maintenance scripts
- Profiling, debugging and evaluation
- Pseudo-labeling and adapter utilities
- Useful scripts and tools

Quick verification
- Run a smoke forward pass (synthetic) to validate model import and shapes:
  ```bash
  python scripts/smoke_gnn_test.py
  ```
- Run a short benchmark (synthetic or webcam) with the benchmarking harness:
  ```bash
  python scripts/benchmark_gnn.py --mode synthetic --use_gnn 1 --iters 100
  python scripts/benchmark_gnn.py --mode webcam --duration 10
  ```

Checkpoints & naming
- Canonical single model: `model.pth`
- K-fold checkpoints: `ensemble/fold_{n}.pth`
- Adapter weights live in `adapter_weights/` and are named by timestamp.

K-Fold full fine-tuning
- Use `python train.py --kfold N --epochs X --lr Y`.
- Each fold trains the full model directly (no warmup stage).

Data QC & maintenance
- `find_corrupt_npy.py` — locate corrupt or unreadable `.npy` sequence files.
- `validate_npy_files.py` — validate shape and expected feature layout.
- `cleanup_dataset_npy.py` — helper to remove or move invalid samples.
- `random_downsample_processed.py` — downsample processed dataset per-class.

Profiling & debugging
- `profiling.py` — lightweight profiler for inference/training segments.
- `debug_model.py` — harness to run model-specific debug cases and check forward/backward pass behaviors.
- Logging: pipeline events are emitted via `pipeline_logger.py` and written to `logs/`.

Evaluation utilities
- `eval_per_class.py` — compute per-class metrics and confusion matrices.

Pseudo-labeling and adapters
- `pseudo_buffer.py`, `pseudo_utilities.py`, and `pseudo_data/` provide primitives for generating and storing pseudo-labeled samples.
- Adapter training scripts: `adapter_training.py`, `adapter_model.py`.

Developer scripts & helpers
- `scripts/` contains small dev tools like `smoke_gnn_test.py` and `benchmark_gnn.py`.
- `modify_train_kfold.py` and `train_kfold_resume.py` help orchestrate multi-run experiments.
- `tools/` includes ad-hoc helpers; inspect to see utilities used for dataset ops.

Reproducible experiment scripts
- `scripts/run_kfold_gnn.sh` — POSIX shell script to run K-fold full fine-tuning, install deps, and save logs.
- `scripts/run_kfold_gnn.ps1` — PowerShell equivalent for Windows. Usage examples:

  Bash (Linux/macOS/WSL):
  ```bash
  ./scripts/run_kfold_gnn.sh 5 8 1e-4 logs/exp01
  ```

  PowerShell (Windows):
  ```powershell
  .\scripts\run_kfold_gnn.ps1 -Folds 5 -Epochs 8 -Lr 0.0001 -OutDir logs\exp01
  ```

Environment setup (recommended)
- Create and activate a virtual environment:
  - Bash/macOS/WSL:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
  - PowerShell (Windows):
    ```powershell
    python -m venv venv
    & .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```

Notes
- The scripts assume the repo root is the current working directory. They will create an output folder under `logs/` by default and tee both run metadata and full training logs there.
- If you prefer conda, activate your conda environment before invoking the scripts.

Notes
- The repo includes `face_landmarker.task` and `hand_landmarker.task` (MediaPipe task definitions) used by `preprocess.py`.
- Keep virtual environment and build artifacts out of source control (`venv/`, `__pycache__/`) — `.gitignore` should already exclude them.

If you'd like, I can expand any section into runnable examples or add one-click scripts for reproducible experiments (e.g., `run_kfold_gnn.sh`).
