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

Data QC, Generation, and Augmentation Pipeline
- [cvae_landmarks.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/cvae_landmarks.py) — Defines configuration parameters (for network, loss, and quality filters), label encoders, sequence padding/truncation utilities, and the core LandmarkCVAE model. Custom training losses are computed combining reconstruction MSE, KL divergence, and velocity consistency.
- [train_cvae.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/train_cvae.py) — Trains the LandmarkCVAE model on processed landmark sequences. Implements stratified per-class validation splits, TensorBoard logging with PCA projection of the latent space, and early stopping.
- [generate_cvae_samples.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/generate_cvae_samples.py) — Generates class-balanced synthetic sequences by sampling from the latent space using per-class stats (means/stds) and passing them through the decoder. Applies heuristic filters to reject invalid sequences before saving them.
- [visualize_latent_space.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/visualize_latent_space.py) — Computes and plots 2D visualizations (PCA or t-SNE) of the latent space clusters for trained CVAE embeddings.
- [quality_discriminator.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/quality_discriminator.py) — Implements a lightweight GRU-based quality classifier that scores sequence realism $P(\text{sample is real})$. Includes additional heuristic checks (motion variance, joint drift, active ratio).
- [train_quality_discriminator.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/train_quality_discriminator.py) — Trains the quality discriminator. Utilizes **hard negative mining** to isolate synthetic samples that trick the model and feeds them back into training for dedicated fine-tuning.
- [augmentations.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/augmentations.py) — Implements 20 deterministic landmark-level augmentations:
  - *Spatial/Geometric:* 3D rotation, scaling, translation, horizontal flipping (swapping left/right hand channels and negating X coordinates).
  - *Temporal:* speed variation (stretching and resampling), time shifting, frame dropping.
  - *Occlusion/Noise:* fog noise, pixel/coarse dropouts, finger articulation scaling, and wrist trajectory drift.
- [augment_pipeline.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/augment_pipeline.py) — Pipeline orchestrator that runs landmark augmentation, merge augmentation, and diversity cleanup in sequence.
- [merge_augmentations.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/merge_augmentations.py) — Generates merged samples by splicing and blending contiguous frame ranges from peer sequences of the same class (splicing, crossfading, tempo-aligned warping, hand swapping, and style blending).
- [balance_processed_dataset.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/balance_processed_dataset.py) — Balances class folders to a fixed target count by duplicating original webcam captures for underrepresented classes or removing excess files for overrepresented classes.
- [random_downsample_processed.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/random_downsample_processed.py) — Downsamples class folders to a fixed threshold by randomly deleting non-webcam files while fully protecting original webcam captures.
- [cleanup_dataset_npy.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/cleanup_dataset_npy.py) — Deletes near-duplicates and selects the top-K diverse augmented/merged files per class using Farthest Point Sampling (FPS) on L2-normalized flattened sequences.
- `find_corrupt_npy.py` — Locates corrupt or unreadable `.npy` sequence files.
- `validate_npy_files.py` — Validates sequence shape and expected feature layout.

Profiling & debugging
- [profiling.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/profiling.py) — Lightweight profiler for inference/training segments.
- [debug_model.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/debug_model.py) — Harness to run model-specific debug cases and check forward/backward pass behaviors.
- [pipeline_logger.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/pipeline_logger.py) — Emits pipeline log events to the `logs/` directory.

Evaluation utilities
- `eval_per_class.py` — Computes per-class metrics and confusion matrices.

Pseudo-labeling and User-Specific Live Adapter
- [pseudo_buffer.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/pseudo_buffer.py), [pseudo_utilities.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/pseudo_utilities.py), and `pseudo_data/` — Provide primitives for generating, buffering, and storing high-confidence pseudo-labeled samples.
- [adapter_model.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/adapter_model.py) — Implements `AdapterModel` and `AdapterTrainer`:
  - `AdapterModel`: A lightweight MLP (Dense -> ReLU -> Dense) with a residual connection (`adapted_logits = logp + delta`) that corrects ensemble probability vectors in log-probability space without modifying the base models.
  - `AdapterTrainer`: Trains the adapter model using CrossEntropyLoss on logits, handles class weights, and evaluates pre/post-adaptation average confidence.
- [adapter_training.py](file:///c:/Users/Joseph/Desktop/projects/sign_to_text/adapter_training.py) — Implements `AdapterTrainingManager`:
  - *Threaded Asynchronous Training:* Performs training in a background thread to prevent blocking the webcam loop.
  - *Class Balancing:* Downsamples training data to the smallest class count to prevent dominant class bias.
  - *Inverse-Frequency Class Weights:* Computes normalized, clipped class weights to reduce bias from skewed pseudo-data.
  - *Performance Validation:* Validates that adaptation does not degrade ensemble confidence; reverts to weight backups if validation fails.

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
