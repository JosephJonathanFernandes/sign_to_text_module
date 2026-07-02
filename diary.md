# Engineering Log (January 1 - July 11, 2026)

**January 2026**

* **Jan 1-5**: Started looking into building a real-time sign language translation system using webcam feeds. Spent the first few days comparing OpenPose and MediaPipe. Decided to go with MediaPipe Holistic because it's much lighter and runs reasonably well out of the box on CPU.
* **Jan 8-12**: Wrote some messy local test scripts to hook OpenCV into MediaPipe. Had an annoying issue where the OpenCV video capture thread kept blocking the main loop, making the feed stutter terribly. Ended up moving the frame reading into a background daemon thread, which mostly fixed it.
* **Jan 15-20**: Got stuck on how to handle the landmarks. The coordinates change drastically depending on how close I sit to the webcam. Spent a few days experimenting with bounding box normalization to scale the landmarks so the model won't care about distance.
* **Jan 21-25**: Realized that parsing raw video during training is way too slow and eats up disk space. Wrote a quick local script to extract all landmarks upfront and save them as `.npy` arrays. Also started prototyping PyTorch dataset classes to load these dynamically.
* **Jan 27-30**: Started experimenting with `src/training/model.py` and tested different sequence-processing approaches. Initially considered a simpler GRU-only setup, but sequence behavior felt too weak for local temporal patterns (like a quick finger flick), so I added a 1D convolution block before a Bidirectional GRU (Bi-GRU). Most of this period was spent getting tensor dimensions aligned and debugging shape mismatches during the forward pass.

**February 2026**

* **Feb 4-6**: Was away from the project for a bit. When I got back, I ran some local training tests and noticed that a basic CrossEntropy loss struggles to differentiate similar signs (like 'good' vs 'bad'). Wrote down a note to look into Focal Loss later.
* **Feb 10-13**: Spent most of the week setting up the actual training loop (`src/training/train.py`) and moving paths/hyperparameters into `src/core/config.py` so I don't have to hardcode everything. Ran a dry run with dummy tensors. The loss decreased, which is a good sign.
* **Feb 17-20**: Hooked up the `src/preprocessing/preprocess.py` pipeline. Spent hours debugging inconsistent landmark dimensions caused by frames where MediaPipe tracking temporarily failed due to motion blur. Ended up adding a simple zero-padding strategy to handle missing frames.
* **Feb 21-22**: Finally initialized the Git repository and pushed the core files (`main.py`, `model.py`, `train.py`, `dataset.py`, `preprocess.py`, `.gitignore`, `config.py`). Wrote the first commit message ("first commit").
* **Feb 25-26**: Minor bug fixing. Realized the validation split was incorrectly handled in `train.py`, so I pushed a fix. Also tweaked some data collection paths to match the DataLoader.
* **Feb 28**: Pushed the last of the initial structure updates. The base code is there, but there’s no real data yet.

**March 2026**

* **March**: There were no code changes during this period. I spent time reading about sequence-based sign language models and alternatives such as ST-GCNs. I also stepped away from implementation for parts of the month while deciding whether the current Conv-BiGRU approach would scale for complex face/hand interactions.

**April 2026**

* **April 1-2**: Back at it. Integrated a face-proximity attention module to focus the network on hands that are near the face, which seemed to help local tests. Also started drafting `src/inference/nlp_postprocessor.py` and `sentence_builder.py` to turn the raw predicted words into somewhat readable sentences.
* **April 4**: Prediction smoothing was too jittery. Implemented motion gating to ignore frames where hands are mostly still. Added dynamic thresholds, frame cropping, and some transition logic in `src/core/webcam.py` to try and stabilize the local UI output.
* **April 7**: Added a `train_kfold` script for hyperparameter tuning. Finally got around to implementing Focal Loss and Mixup to help with class imbalances.
* **April 9-11**: Spent three exhausting days recording base vocabulary in front of the webcam. Added massive batches of `.npy` files for testing.
* **April 14-17**: The model was failing on common words like "hello", "I", "you", "thin", and "thick". Recorded targeted data for these and pushed them. Also spent time renaming directories because the dataset structure was getting messy and hard to manage.
* **April 22-24**: Prediction smoothing was causing the model to stay stuck on previous signs longer than expected (the "sign lock-in" bug). I spent time replaying webcam recordings and noticed that the stabilization logic was over-correcting.
* **April 27**: Removed the hard lock system and replaced it with a confidence-based `TemporalPostProcessor` in `src/inference/`, which felt much more responsive during testing. Also pushed `HandSelector` fixes to properly convert MediaPipe landmarks to numpy arrays. Updated the README.
* **April 29-30**: Recorded and pushed more `.npy` files for pronouns (they, we, you_all, she, he), adjectives (loose, tight, cheap, good), and greetings (good_morning, good_night). Deleted some old, unwanted files.

**May 2026**

* **May 1-4**: Getting tired of manual recording, so I spent the weekend writing an augmentation pipeline (`src/preprocessing/augmentations.py`) to simulate hand-proportion changes and face-anchor shifts. Used this to generate over 5,600 `.npy` sequences. Refined the train/val splits to be source-aware so augmented data doesn't leak into validation.
* **May 6-8**: More data collection. Focused on spatial words (wide, tall, short, long), opposites (hot/cold, wet/bad, new/old), and speed modifiers (fast/slow). Added `.npy` files and augmented merges for all of these.
* **May 9-10**: No major coding this weekend. Wrote a technical doc and updated the README to reflect all the recent architectural changes.
* **May 12-13**: Decided to actually test the GNN idea from March. Drafted `src/training/spatial_gnn.py` to trace skeletal shapes. After testing it out, the added complexity and latency didn't seem worth it for real-time performance, so I stayed with the Conv-BiGRU. Added final batches of adjectives (weak, strong, alive, dead).
* **May 18-20**: The repo was getting way too big and pulling it was taking forever. Did a major cleanup: removed `processed` and `pseudo_data` directories from git tracking, updated `.gitignore` to block model weights and `.pth` files. Also pushed some tweaks to improve local OpenCV FPS in `webcam.py`.

**June 2026**

* **June 2-5**: Decided to try exporting the PyTorch model to ONNX to see if it would run faster on my laptop. Created `src/inference/export_onnx.py` and `onnx_inference.py`. Had repeated issues with ONNX export because some custom tensor operations weren't supported. Spent a few days rewriting that math to use standard operations.
* **June 9-12**: Got ONNX working locally. CPU usage dropped significantly. Also spent time experimenting with `quantize_onnx.py` and checking the results in `evaluate_quantized_model.py`.
* **June 16-19**: The project scope expanded to deploying this as a backend service for a separate frontend team. Started laying out the `api/` folder. Spent a few days setting up FastAPI in `api/app.py` and defining Pydantic models in `schemas.py` to ensure the frontend sends the exact 506-dimensional feature vector we expect.
* **June 24-26**: Ran into major headaches trying to sync with the frontend team. They were sending raw image frames over the socket, which was way too slow. Drafted `FEATURE_CONTRACT.md` and `FRONTEND_HANDOFF.md` to explicitly document that MediaPipe extraction happens on the client, and they only stream the normalized arrays to the backend's WebSocket endpoint (`/ws/translate`).

**July 2026**

* **July 1-3**: Debugged a weird issue where the model got "stuck" on a single prediction over the WebSocket. Turned out the frontend was sending frozen landmarks. Added a `LOG_LEVEL=DEBUG` mode to `run_api.py` that calculates a jitter-resistant hash of the sequence buffer to detect frozen streams in `api/session.py`.
* **July 6-8**: Did some boring maintenance work. Cleaned up old preprocessing scripts (`cleanup_dataset_npy.py`), formatted the codebase with Flake8, separated the API dependencies into `requirements_api.txt`, and pinned versions.
* **July 10-11**: Ran an end-to-end test with a simulated frontend script. Streamed a full minute of landmarks over the WebSocket without crashing or dropping frames. Wrote the `final_validation_report.md` in the `api/` folder. Everything looks stable, so I wrapped up documentation and considered this phase complete.

---

## Repository Work Index (Cumulative)

### Core Training & Models

* `src/training/model.py`
  * Implemented hybrid Conv1D + Bi-GRU architecture.
  * Integrated specialized face-proximity attention module.
  * Resolved tensor dimension and shape mismatch logic.
* `src/training/adapter_model.py`
  * Prototyped modular fine-tuning adapter architectures.
* `src/training/train.py`
  * Implemented primary training loop with epoch management.
  * Corrected validation split logic.
  * Configured Focal Loss and Mixup augmentation to handle class imbalances.
* `src/train_continuous.py`
  * Built continuous learning and streaming loop implementation.
* `src/training/train_kfold.py` & `src/training/train_kfold_resume.py`
  * Integrated K-Fold cross-validation for robust hyperparameter tuning.
  * Hardened training pipelines with fault-tolerant resume capabilities.
* `src/training/spatial_gnn.py`
  * Evaluated experimental skeletal spatial graph neural networks (ST-GCN variants).
* `src/core/config.py`
  * Extracted all hardcoded paths and hyperparameters into centralized configs.

### Experimental Generative Modeling (CVAE)

* `experimental/train_cvae.py` & `experimental/cvae_landmarks.py`
  * Developed a Conditional Variational Autoencoder (CVAE) for synthetic landmark sequence generation.
* `experimental/train_quality_discriminator.py` & `experimental/quality_discriminator.py`
  * Implemented an adversarial quality discriminator to score and validate synthetic sequences.
* `experimental/visualize_latent_space.py` & `experimental/visualize_quality_scores.py`
  * Built visualization tools to map the CVAE latent space and analyze sample quality.
* `experimental/generate_cvae_samples.py` & `experimental/filter_synthetic_samples.py`
  * Automated high-volume synthetic sample generation and strict filtration pipelines.

### Preprocessing & Dataset Pipeline

* `src/preprocessing/preprocess.py`
  * Built real-time landmark extraction logic via MediaPipe Holistic.
  * Engineered zero-padding strategy for missing frames (motion blur recovery).
* `src/preprocessing/augmentations.py` & `src/preprocessing/merge_augmentations.py`
  * Engineered synthetic data generation scaling over 5,600 unique sequences.
  * Simulated face-anchor shifts and dynamic hand-proportion transformations.
* Dataset Balancing & Filtering
  * `src/preprocessing/quality_filter_hybrid.py`: Stripped corrupted samples from the dataset.
  * `src/preprocessing/balance_processed_dataset.py` & `random_downsample_processed.py`: Enforced strict class distribution equity.
* General Dataset Operations
  * Enforced bounding-box normalization to nullify camera distance variance.
  * Automated `.npy` extraction to eliminate I/O bottlenecks during PyTorch training.
  * Enforced source-aware train/validation splits to prevent data leakage.
* Advanced Data Tooling (`src/tools/`)
  * `compile_hdf5.py`: Compiled massive dataset archives into HDF5 for high-throughput I/O.
  * `generate_dataset_heuristics.py`: Developed rule-based dataset heuristic generators.
  * `generate_negative_root.py`: Synthesized negative action frames for robust background rejection.

### Performance Profiling & Benchmarking

* `src/utils/profiling.py`
  * Engineered deep-level latency profiling logic for the inference pipeline.
* `src/utils/pipeline_logger.py`
  * Constructed a robust, centralized structured logger.
* Benchmarking Scripts (`src/tools/`)
  * `benchmark_inference.py`: Automated performance testing on the real-time model.
  * `benchmark_dataset.py`: Dataloader throughput validation.

### Temporal Inference & Post-Processing

* `src/inference/nlp_postprocessor.py` & `src/inference/sentence_builder.py`
  * Designed an advanced Natural Language Processing pipeline.
  * Translated raw predictive tokens into grammatically coherent output sentences.
* `src/inference/pseudo_buffer.py`
  * Managed sliding temporal sequences.
* `TemporalPostProcessor`
  * Overhauled "hard lock-in" bugs by implementing dynamic confidence thresholds.
  * Implemented sliding-window stabilization for fluid transitions.
* `src/core/webcam.py` & `src/core/main.py`
  * Built dual-tier local OpenCV inference engine.
  * Integrated motion gating and dynamic frame cropping to minimize static jitter.
* Peripheral Components
  * `src/shared/feature_extractor.py`: Handled standalone landmark routing.
  * `src/ui/renderer.py`: Abstracted OpenCV bounding box/text rendering logic.

### ONNX & Ensemble Optimization

* `src/inference/export_onnx.py` & `src/inference/onnx_inference.py`
  * Exported PyTorch state dictionaries to highly optimized ONNX runtimes.
  * Rewrote complex mathematical operations to align with standard ONNX tensors.
* `src/inference/quantize_onnx.py` & `src/inference/evaluate_quantized_model.py`
  * Built rigorous model quantization pipelines to accelerate local CPU-bound inference.
* `src/inference/ensemble.py` & `src/inference/onnx_ensemble.py`
  * Integrated multi-checkpoint prediction blending for complex motion sequences.

### API Backend & WebSocket Architecture

* `api/app.py`
  * Deployed a decoupled, real-time FastAPI backend service.
  * Implemented the `/ws/translate` persistent WebSocket endpoint.
* `api/schemas.py`
  * Enforced strict Pydantic payload validation for the 506-dimensional normalized vectors.
* `api/session.py`
  * Handled stateful session management.
  * Built jitter-resistant sequence hashing to dynamically detect and recover from frozen streams.
* Integration Contracts
  * `api/FRONTEND_HANDOFF.md` & `api/FEATURE_CONTRACT.md`
  * `api/final_validation_report.md`

### Testing, CI/CD & DevOps

* Automated Testing Matrix (`tests/`)
  * Engineered a comprehensive pytest suite spanning `unit`, `integration`, `e2e`, and `api` tests.
  * Implemented `.coverage` tracking.
* Continuous Integration (`.github/workflows/ci.yml`)
  * Configured GitHub Actions to automatically enforce build integrity and run test matrices on push.
* Environment Bootstrapping
  * Authored automated setup scripts (`scripts/setup.ps1`, `scripts/setup.sh`).
  * Engineered a rigorous `scripts/verify_repo.py` system integrity checker.

### Repository Maintenance & Code Quality

* Code Quality Standards
  * `pyproject.toml` & `.pre-commit-config.yaml`
  * Enforced rigorous linting and formatting via Flake8, Ruff, and Black.
* Git Tree Optimization
  * `.gitignore`: Aggressively purged tracked model weights (`.pth`) and raw `processed/` data directories.
  * `scripts/cleanup_dataset_npy.py`: Scripted automated purges of outdated or corrupted dataset samples.
* Dependency Management
  * Explicitly pinned staging and production environments via `requirements.txt`, `requirements_api.txt`, and `requirements-dev.txt`.
* Documentation Library (`docs/`)
  * `ARCHITECTURE.md`: High-level system design.
  * `DECISIONS.md`: Logged architecture decision records (ADRs).
  * `PROFILING_LATENCY_REPORT.md`: System latency and performance bottleneck breakdown.
  * `dataset.md`, `inference_pipeline.md`, `model_architecture.md`, `training_pipeline.md`: Deep-dive component manuals.
  * `execution_guide.md`: End-to-end local deployment guide.
  * `README.md`, `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTING.md`: Core repository onboarding docs.

### Repository Artifacts & Data Stores

* `models/` directory
  * Final deployed weights (`model.pth`, `model.onnx`, `model.onnx.data`).
  * MediaPipe standalone task assets (`face_landmarker.task`, `hand_landmarker.task`).
  * Dedicated directories for `adapter_weights/` and `ensemble/` checkpoints.
  * Massive `model_metadata.json` for architectural reference.
* `assets/` directory
  * Centralized telemetry and statistical maps (`confidence_statistics.json`, `candidate_map.json`).
  * Core vocabularies (`sign_categories.json`, `similar_signs.json`).
  * Large-scale keypoint backups (`keypoints.csv`).
* `data/` directory
  * Consolidated `dataset.h5` container.
  * Pipeline output verification via `validation_report.json`.
* `logs/` directory
  * Comprehensive telemetry collection for data collection (`collect_*.log`) and real-time testing (`inference_*.log`).
* `archive/` directory
  * Retained legacy scripts such as `audit_api.py`.

---

## MY WORK

### 1. Executive Summary & System Overview

This report details the architectural design, engineering progression, and final implementation of the Real-Time Sign Language Translation System. Designed to bridge the communication gap by translating continuous sign language into readable text in real-time, the system operates on a highly flexible dual-tier deployment model. The architecture supports both a low-latency local inference engine (`src/core/main.py` and `webcam.py`) for offline desktop use, and a decoupled FastAPI WebSocket service (`api/app.py`) for seamless frontend integration. This dual approach maximizes utility, allowing for rapid edge-device execution while maintaining robust, platform-agnostic cloud API capabilities.

### 2. Data Engineering & Preprocessing Pipeline

The foundation of the system relies on highly consistent, normalized spatial data.

* **Landmark Extraction Strategy:** Early evaluations compared OpenPose and MediaPipe. MediaPipe Holistic was selected due to its significantly lower computational overhead and ability to run efficiently on CPU environments, making it the ideal candidate for edge-device frontend extraction.
* **Normalization & Augmentation:** Raw coordinate data is highly sensitive to subject distance and camera positioning. To mitigate this, bounding box normalization was implemented. Furthermore, a comprehensive data augmentation pipeline (`src/preprocessing/augmentations.py`, `merge_augmentations.py`) was engineered to simulate face-anchor shifts and hand-proportion changes, ensuring model robustness against varying user anatomies and camera angles.
* **Storage and Loading Optimizations:** Processing raw video during model training proved to be a severe I/O bottleneck. The pipeline was overhauled to extract all landmarks upfront and serialize them as `.npy` arrays, drastically accelerating training iterations.
* **Advanced Dataset Balancing:** To ensure equitable class distribution and high-quality inputs before model training, the dataset undergoes strict filtering and balancing protocols utilizing `quality_filter_hybrid.py`, `balance_processed_dataset.py`, and `random_downsample_processed.py`.
* **Handling Data Anomalies:** Real-world webcam feeds are prone to motion blur, occasionally causing the MediaPipe tracker to drop frames. A zero-padding strategy gracefully handles missing frames within the preprocessing pipeline, maintaining tensor dimensionality without corrupting the temporal sequence.

### 3. Core Deep Learning Architecture

The predictive engine was iteratively refined to capture both local gestures and global temporal context.

* **Evolution to Conv-BiGRU:** Initial prototyping explored a standard LSTM architecture. However, LSTMs struggled to capture rapid, highly localized temporal patterns (e.g., quick finger flicks). The architecture was subsequently upgraded to a 1D Convolutional block preceding a Bidirectional GRU (Bi-GRU) (`src/training/model.py`). The Conv1D layer extracts local temporal features, while the Bi-GRU captures broader sequential context.
* **Alternative Architectural Explorations:** The engineering phase included deep explorations into Spatial Graph Neural Networks (`src/training/spatial_gnn.py`) to trace skeletal topologies. While yielding valuable insights, the Conv-BiGRU was ultimately favored for its superior real-time latency profile. Additional flexibility was added via modular fine-tuning adapters (`src/training/adapter_model.py`).
* **Specialized Attention Modules:** To prioritize critical semantic information, a face-proximity attention module dynamically weights the network's focus toward hands operating in close proximity to the face.
* **Loss Functions and Training Optimizations:** Addressing class imbalance and nuanced similarities between signs, the training regime transitioned from standard CrossEntropy to Focal Loss, combined with Mixup augmentation to stabilize decision boundaries across the vocabulary.

### 4. Temporal Inference & Post-Processing

Translating continuous streams of landmarks into coherent text required sophisticated stabilization logic.

* **ONNX & Quantization Pipeline:** To maximize local CPU performance, the PyTorch models undergo rigorous export and optimization. The pipeline leverages ONNX Runtime (`src/inference/export_onnx.py`, `onnx_inference.py`) alongside strict model quantization (`quantize_onnx.py`, `evaluate_quantized_model.py`) to slash inference latency.
* **Ensemble Modeling:** For maximum accuracy on complex sequences, the inference engine supports blending multiple model checkpoints via `ensemble.py` and `onnx_ensemble.py`.
* **Motion Gating & Temporal Post-Processing:** Early iterations suffered from prediction jitter during static frames. Motion gating was implemented to dynamically ignore frames where the hands remain predominantly stationary. A confidence-based `TemporalPostProcessor` utilizes sliding window smoothing and dynamic confidence thresholds to ensure responsive, yet highly stable, transitions between signs.
* **NLP Translation Layer:** Raw predicted tokens are not immediately presentable to users. An advanced Natural Language Processing pipeline (`src/inference/nlp_postprocessor.py`, `sentence_builder.py`) structurally converts these raw tokens into grammatically coherent, readable sentences.

### 5. API Architecture & Deployment (FastAPI & WebSockets)

To support a decoupled frontend, the system was packaged into a scalable, real-time backend service.

* **WebSocket Streaming Architecture:** The core inference loop is exposed via a persistent WebSocket connection (`/ws/translate`). This allows the frontend to continuously stream lightweight landmark vectors rather than bandwidth-heavy video frames, achieving near-zero latency inference.
* **Frontend/Backend Handoff Contract:** To ensure absolute synchronization between the client extraction and backend inference, a strict `FEATURE_CONTRACT.md` and `FRONTEND_HANDOFF.md` were established. The FastAPI service utilizes Pydantic validation (`api/schemas.py`) to strictly enforce that the frontend delivers exactly 506-dimensional normalized feature vectors per frame.
* **Session Management & Jitter Resistance:** A significant hurdle involved handling frozen streams where the frontend inadvertently transmitted static landmarks. This was mitigated by implementing stateful session management (`api/session.py`) and a jitter-resistant hashing mechanism across the sequence buffer.

### 6. Testing, CI/CD & DevOps

A hallmark of the system's production readiness is its robust testing and deployment infrastructure, designed to eliminate regressions and ensure environmental parity.

* **Comprehensive Testing Matrix:** The repository contains an extensive, multi-layered testing suite (`tests/`) driven by `pytest`. This includes unit tests for isolated component logic, integration tests for cross-module workflows, end-to-end (E2E) pipeline validations, and dedicated API endpoint testing.
* **Continuous Integration (CI):** To strictly enforce build integrity, a GitHub Actions CI pipeline (`.github/workflows/ci.yml`) is configured to automatically trigger the test suite, linting, and coverage generation on every repository push or pull request.
* **Code Quality & Standardization:** Strict formatting and linting rules are enforced via `pyproject.toml` and `.pre-commit-config.yaml`. By integrating tools like Ruff and Black as pre-commit hooks, the repository maintains consistent, professional code hygiene.
* **Automated Environment Bootstrapping:** To eliminate the "it works on my machine" anti-pattern, cross-platform setup scripts (`scripts/setup.ps1`, `setup.sh`) and a dedicated repository verification tool (`scripts/verify_repo.py`) guarantee deterministic environment creation and dependency alignment.
* **Extensive Documentation:** Critical deployment contracts and system requirements are persistently documented within `docs/` and `execution_guide.md`, enabling seamless developer onboarding.

### 7. Technical Debt & Repository Management

Ensuring the repository remained maintainable and deployment-ready was a continuous priority.

* **Fault-Tolerant Training:** The training pipeline was hardened with K-Fold resume capabilities (`src/training/train_kfold_resume.py`), ensuring that extensive, long-running training epochs could recover safely from interruptions or environment failures.
* **Git Tree Optimization:** As the dataset grew, repository cloning became a bottleneck. A strict audit removed legacy processing scripts, purged corrupted data blocks, and enforced `.gitignore` policies to exclude heavy `.pth` model weights and raw `processed/` directories from version control.
* **Deployment Readiness:** The backend was hardened for production deployment. Code was rigorously formatted to Flake8 standards, dependencies were explicitly isolated and pinned in `requirements_api.txt`, and a top-level `uvicorn` launcher (`run_api.py`) was established to streamline server-based execution.
