# Engineering Log (February 21 - July 15, 2026)

> **Note:** The project started in **February 2026**. The Git repository was initialized on **February 21, 2026**. All entries from that date onward are backed by commit history. March entries cover offline experimental work with no commits pushed.

**February 2026**

* **Feb 21** *(first commit: `8c838b8e6`)* : Initialized the Git repository. Pushed the foundational codebase: core preprocessing pipeline (`src/preprocessing/preprocess.py`), training loop (`train.py`, `train_ablation.py`), and initial Conv1D + Bidirectional GRU model architecture. Included bounding-box normalization, `.npy` landmark extraction pipeline, zero-padding for missing frames, and PyTorch `DataLoader` with validation split logic.
* **Feb 22-28** *(commits: `57df43514` → `15dcdfd63`)* : Debugged validation leakage in the PyTorch `DataLoader`. Pushed incremental fixes stabilizing the base training pipeline.

**March 2026 (No commits — offline experimental work)**

* **March 3-10**: Pivoted focus toward experimental machine learning architectures. Researched Spatial Temporal Graph Convolutional Networks (ST-GCNs) for skeletal mapping — evaluated feasibility against the Conv-BiGRU baseline.
* **March 15-25**: Developed a Conditional Variational Autoencoder (CVAE) (`experimental/cvae_landmarks.py`, `train_cvae.py`) to generate synthetic landmark sequences and augment underrepresented sign classes.
* **March 26-31**: Built an adversarial Quality Discriminator (`quality_discriminator.py`, `train_quality_discriminator.py`) to validate CVAE outputs, and latent space visualization tools (`visualize_latent_space.py`) to inspect the generation quality.

**April 2026**

* **April 1-4** *(commit: `2a66e7026`, `aec2df267`)* : Returned to the core Conv-BiGRU model. Added automatic sentence translation and dataset integration. Integrated motion gating, dynamic confidence thresholds, transition logic, NLP post-processing, and dynamic frame cropping into the live inference path.
* **April 7-11** *(commits: `da63625ee` → `06985fb44`)* : Added Focal Loss, configurable Mixup augmentation, extended training epochs, and a comprehensive hyperparameter tuning script. Fixed model loading to eliminate feature dimension mismatch errors. Expanded vocabulary — committed `.npy` files for 20+ new sign classes across multiple recording sessions.
* **April 16-17** *(commits: `cf7732cdb` → `30c624af9`)* : Major recording sprint — committed landmark sequences for: `hello`, `I`, `you`, `thin`, `thick`, `mean`, `how_are_you`, `thank_you`, `expensive`. Updated model files with latest checkpoint.
* **April 27** *(commits: `a63d81884` → `d93f0bd45`, `6ff84e8c6`)* : Integrated `TemporalPostProcessor` and `HandSelector` into the webcam inference pipeline. Solved the critical **"sign lock-in" bug** — replaced hard-lock logic with a sliding-window exponential smoother using dynamic confidence thresholds. Reduced smoothing window size for faster sign transitions. Committed `.npy` files for: `cheap`, `good`, `idle`, `male`, `female`.
* **April 29-30** *(commits: `6ae01e630` → `8d504c55a`)* : Major dataset recording weekend — committed landmark sequences for: `tight`, `loose`, `he`, `she`, `it`, `you_all`, `we`, `they`, `alright`, `good_morning`, `morning`, `good_afternoon`, `good_evening`, `good_night`, `pleased`. Pushed updated model checkpoint.

**May 2026**

* **May 1-4** *(commits: `5ff8468f6` → `551fb8a73`)* : Intensive dataset recording sprint — committed `.npy` sequences for number signs 1–22 (marks/numerals) and expanded vocabulary from 42 to 79 classes (all adjective signs completed). Added source-aware train/validation splits.
* **May 6-13** *(commits: `b66908976` → `dd2dab01b`)* : Continued dataset expansion — recorded and committed sequences for: `old`, `bad`, `wet`, `hot`, `cold`, `warm`, `cool`, `new`, `narrow`, `big_large`, `small_little`, `slow`, `fast`, `healthy`, `high`, `deep`, `shallow`, `clean`, `dirty`, `strong`, `weak`, `dead`, `alive`, `heavy`. Updated augmentation pipeline and merge pipeline to align with methodology from academic papers.
* **May 12** *(commits: `fc6f3c208`, `8e48c5157`)* : Hardened training and inference pipeline — added safer Conv frontend, smoke checks, and live inference momentum improvements. Conducted shape trace and GNN feasibility analysis.
* **May 20** *(commits: `999fafe34`, `bcc5e5b7f`)* : Major `.gitignore` cleanup — removed model weights (`.pth`), `processed/` dataset directories, and pseudo data from Git tracking to prevent repository bloat.
* **May 25-31** *(commits: `8c01084ff` → `68391e782`)* : Added ONNX tooling scaffolding and INT8 quantization pipeline. Implemented two-phase training architecture: Phase 1 on processed-only data; Phase 2 archived fine-tune with `--archived-weight` CLI flag. Added K-fold fine-tune for all folds. Fixed weighted sampler unpacking bug in `_BalancedAugSubset`. Updated README with CVAE, quality discriminator, and ONNX integration documentation.

**June 2026**

* **June 1-5** *(commits: `97a036f95` → `0d210ef8c`)* : Comprehensive documentation and cleanup sprint. Compiled FYP dissertation chapters, Viva preparation guide, consolidated sign-to-text report (Section 4), and advanced pipeline optimization documentation. Cleaned the repository structure.
* **June 28** *(commits: `e122003be` → `46570c44f`)* : **Repository Transformation v2.0** — the largest single-day commit set. Architected the production FastAPI backend (`api/app.py`) with the `/ws/translate` persistent WebSocket endpoint. Defined strict Pydantic payload validation (`FEATURE_CONTRACT.md`, `FRONTEND_HANDOFF.md`). Deployed GitHub Actions CI (`.github/workflows/ci.yml`), reorganized a comprehensive `pytest` matrix (unit, integration, e2e, API). Integrated Domain Adversarial Neural Network (DANN) via Gradient Reversal Layer (GRL) for domain adaptation. Fixed API dimension validation, reorganized scripts into proper `src` domains, and added production-grade metrics, logging, and ADR documentation.
* **June 29-30** *(commits: `3024ea31f` → `2fa459eff`)* : **Massive Data Engineering Overhaul.** Replaced manual sign classifications with dataset-derived heuristics (`keypoints.csv` — 1.3GB, added to `.gitignore`). Built `generate_dataset_heuristics.py` for automated `npy`-based landmark feature extraction. Implemented the hybrid quality-and-diversity filtering pipeline (`quality_filter_hybrid.py`). Built ONNX inference wrapper with PyTorch fallback (`onnx_inference.py`), performance benchmarking utilities, and ONNX input validation diagnostics (`debug_onnx_input_check.py`). Implemented `AdapterTrainingManager` for asynchronous model adaptation with data balancing.

**July 2026**

* **July 1-2** *(commits: `8869442fd` → `d96d6069f`)* : Inference and NLP Upgrades. Implemented `SentenceBuilder` with robust frame-based transition logic and NLP post-processing. Built lightweight MLP adapter for ensemble output correction. Added `AdapterTrainingManager` for asynchronous, balanced adapter training with safety validation. Added `similar_signs.json` vocabulary for sign interpretation disambiguation. Published model evaluation script generating accuracy, F1, and confusion analysis. Added FYP dissertation documentation, exhaustive technical audits, and profiling reports.
* **July 3-4** *(commits: `3681f45bf` → `540f47ff7`)* : Architectural Expansion. Implemented the hybrid quality-and-diversity filtering pipeline. Built dataset balance scripts (duplication and random downsampling to fixed thresholds). Engineered a lightweight Spatial GNN for MediaPipe hand topology analysis. Built inference benchmarking tool and documented performance metrics in `PROFILING_LATENCY_REPORT.md`.
* **July 5-12** *(commits: `8dc1fb983` → `ab93467fd`)* : High-intensity vocabulary recording and pipeline automation week. Created `process_class.py` — a single-command pipeline runner that executes augmentation, quality filtering, merge, and heuristics regeneration for any newly recorded sign class. Committed landmark data for 50+ additional classes (S–Z alphabet set, adjectives, common phrases). Fixed HDF5 compile trigger in training scripts and exact class matching in preprocessing. Built live inference engine with ensemble support. Committed similar-signs dataset and augmentation utilities for sequence preprocessing. Updated classification metadata (`candidate_map.json`, `confidence_statistics.json`, `hand_sign_classification.json`) iteratively as new classes were registered.
* **July 12** *(additional context)*: Identified a critical flaw in Out-of-Distribution (OOD) handling — the model was forcing noise and idle states into valid sign predictions (high False Acceptance Rate). Began **Phase 2 Robustness**, explicitly collecting 3,000+ negative sequences (idle, transitions, random hand noise) to train a dedicated `__reject__` class.
* **July 13**: Finalized robustness evaluation. Implemented a confidence threshold sweep (ROC analysis) — discovered the model learns `__reject__` through slashed confidence (median 0.37 vs. 0.97 for valid signs). Established optimal 0.5 operating threshold yielding a 0.74% False Rejection Rate. Consolidated fragmented documentation and performed a massive repository cleanup.
* **July 14** *(commits: `a502de1aa` → `1c4e2a2be`)* : Full-day engineering sprint on evaluation, active learning, and emergency detection. Built and wired the complete feedback loop — `/feedback` endpoint captures user corrections triggering background `AdapterTrainingManager` training for signer personalization without retraining the base Bi-GRU. Implemented UUID-keyed `InferenceSession` state isolation. Architected the Emergency Sign Detection module (`api/emergency.py`) with configurable keyword monitoring and ntfy push notification alerts. Built a comprehensive WebSocket evaluation suite producing empirical benchmarks: **105 FPS / 9.1ms mean ONNX inference latency**. Fixed multiple bugs: Pydantic 422 payload errors, JSON format mismatches, `pending_count` attribute initialization, `AdapterModel` state dict unpacking. Regenerated all dataset heuristics for 152-class vocabulary.
* **July 15** *(commits: `746e59e6d` → `9d22ba7a9`)* : Focused API hardening and robustness engineering. Conducted 12-point production-readiness audit. Committed three immediate `chore(api)` fixes: non-fatal `np.isfinite()` NaN validation, `MAX_PAYLOAD_SIZE = 50KB` payload guard (WS code 1008), and magic-number extraction to named constants. All four smoke tests passed. Conducted definitive asyncio.Queue architecture analysis — proved sequential `await run_in_executor` design is correct given 10ms inference vs. 33ms frame interval. Compiled Architecture Decision Records (`docs/ARCHITECTURE_DECISIONS.md`) with 9 implemented and 4 rejected decisions. Implemented skeleton quality gate: zero-ratio check and landmark jump detector with session-level counters. Fixed `prev_frame` logic bug (must reference last *accepted* frame). Added `calibrate_jump_threshold.py` for empirical threshold validation. Processed new `better` and `broken` vocabulary classes through the full pipeline.
* **July 16-17 (Planned)**: Empirical jump threshold calibration from real signing sessions. Final Presentation (PPT) preparation. Validate model against Goa Board ISL videos and Muskaan Ma'am's ISL book.

---

## Repository Work Index (Cumulative)

### Core Training & Models

* Root Entry Points & Configs
  * `main.py`, `train.py`, `model.py`, `webcam.py`, `config.py`: Top-level aliases and script launchers.
* `src/training/model.py`
  * Implemented hybrid Conv1D + Bi-GRU architecture.
  * Integrated specialized face-proximity attention module.
* `src/training/adapter_model.py` & `adapter_training.py`
  * Prototyped modular fine-tuning adapter architectures and training loops.
* `src/training/train.py` & `train_ablation.py`
  * Implemented primary training loop with epoch management.
  * Configured Focal Loss and Mixup augmentation to handle class imbalances.
  * Developed ablation study pipelines.
* `src/train_continuous.py` & `src/config/continuous_signing.py`
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

* `src/preprocessing/preprocess.py` & `collect_data.py`
  * Built real-time landmark extraction logic via MediaPipe Holistic and data collection utilities.
  * Engineered zero-padding strategy for missing frames (motion blur recovery).
* `src/preprocessing/augment_pipeline.py`, `augment_video_pipeline.py`, `augmentations.py` & `merge_augmentations.py`
  * Engineered synthetic data generation scaling over 5,600 unique sequences.
  * Simulated face-anchor shifts and dynamic hand-proportion transformations.
* `src/preprocessing/dataset.py`
  * Implemented `ISLDataset` featuring hybrid HDF5 and file-based loaders.
* `src/preprocessing/quality_filter_hybrid.py` & `balance_processed_dataset.py`
  * Added automated curation pipelines for duplicate suppression, quality checking, and class balancing.

### API & Production Backend

* `api/app.py`, `api/schemas.py`, & `api/session.py`
  * Architected the FastAPI WebSocket backend for real-time edge streaming.
  * Enforced strict Pydantic payload contracts and jitter-resistant session hashing.

### Temporal Inference & NLP

* `src/inference/sentence_builder.py` & `nlp_postprocessor.py`
  * Engineered state-based transitions and robust NLP logic to turn raw tokens into grammatical sentences.
* `src/inference/ensemble.py` & `onnx_ensemble_integration.py`
  * Integrated multi-checkpoint blending to maximize sequence accuracy.
* `src/inference/export_onnx.py` & `quantize_onnx.py`
  * Built the export and INT8 quantization pipelines for CPU optimization.

### Tools, Testing & Evaluation

* `src/tools/benchmark_inference.py` & `src/utils/profiling.py`
  * Developed deep latency benchmarking and hardware profiling utilities.
* `src/tools/evaluate_robustness.py`
  * Created evaluation scripts for Out-of-Distribution (OOD) testing and ROC threshold sweeping.
* `tests/`
  * Engineered a comprehensive `pytest` matrix covering unit, integration, e2e, and API boundaries.
* Dataset Balancing & Filtering
  * `src/preprocessing/quality_filter_hybrid.py`: Stripped corrupted samples from the dataset.
  * `src/preprocessing/balance_processed_dataset.py` & `random_downsample_processed.py`: Enforced strict class distribution equity.
* Advanced Data Tooling (`src/tools/`) & Transitions (`src/augmentations1/`)
  * `compile_hdf5.py`: Compiled massive dataset archives into HDF5 for high-throughput I/O.
  * `generate_dataset_heuristics.py`: Developed rule-based dataset heuristic generators.
  * `generate_negative_root.py`: Synthesized negative action frames for robust background rejection.
  * `transition_generator.py` & `boundary_noise.py`: Simulated fluid continuous signing transitions.

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
* `src/core/webcam.py`, `src/core/main.py`, `camera_manager.py`, `inference_engine.py`, `landmark_processor.py`, `motion_tracker.py`
  * Architected modular, dual-tier local OpenCV inference engine.
  * Integrated motion gating and dynamic frame cropping to minimize static jitter.
* Peripheral Components
  * `src/inference/hand_selector.py` & `src/shared/feature_extractor.py`: Handled standalone hand landmark routing.
  * `src/ui/renderer.py`: Abstracted OpenCV bounding box/text rendering logic.

### ONNX & Ensemble Optimization

* `src/inference/export_onnx.py` & `src/inference/onnx_inference.py`
  * Exported PyTorch state dictionaries to highly optimized ONNX runtimes.
  * Rewrote complex mathematical operations to align with standard ONNX tensors.
* `src/inference/quantize_onnx.py`, `quantize_model.py` & `evaluate_quantized_model.py`
  * Built rigorous model quantization pipelines to accelerate local CPU-bound inference.
* `src/inference/ensemble.py`, `onnx_ensemble.py` & `onnx_ensemble_integration.py`
  * Integrated multi-checkpoint prediction blending for complex motion sequences.

### API Backend & WebSocket Architecture

* `run_api.py`
  * Deployed top-level `uvicorn` ASGI server launcher.
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
  * `tests/conftest.py`: Centralized test fixtures and configuration.
  * `tests/unit/`: `test_config.py`, `test_feature_extractor.py`, `test_hdf5.py`, `test_velocity.py`.
  * `tests/integration/`: `smoke_test_api.py`, `verify_refactor.py`.
  * `tests/api/`: `test_endpoints.py`.
  * `tests/e2e/`: `simulate_frontend.py`.
  * Implemented rigorous `.coverage` tracking across all test domains.
* Continuous Integration (`.github/workflows/ci.yml`)
  * Configured GitHub Actions to automatically enforce build integrity and run test matrices on push.
* Environment Bootstrapping
  * Authored automated setup scripts (`scripts/setup.ps1`, `scripts/setup.sh`).
  * Engineered a rigorous `scripts/verify_repo.py` system integrity checker.

### Repository Maintenance & Code Quality

* Environment Configs & Debugging
  * `scripts/debug_model.py`: Local tensor/weight debugging tool.
  * `.env.example`, `.vscode/settings.json`, `powershell_disabled.cmd`: IDE and environment configurations.
  * `LICENSE`: Open-source licensing.
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
  * `docs/Project_Report_Help/` (Academic & FYP Subdirectory)
    * Prepared comprehensive dissertation chapters (`REPORT_SECTION_4_SIGN_TO_TEXT.md`, `report_section_6.3.md`, `exhaustive_report_section_6.3.md`, `final_consolidated_dissertation_chapter.md`).
    * Generated architectural and forensic audits (`system_design.md`, `forensic_implementation_analysis.md`, `final_dissertation_audit.md`).
    * Structured final academic deliverables (`FYP_REPORT_STRUCTURE.md`, `VIVA_PREPARATION_GUIDE.md`).

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

## Jira Issue Tracking (July 2026)

| ID | Type | Summary | Assignee | Completion Date |
|---|---|---|---|---|
| KAN-44 | Feature | Train new model with robustness more reject classed | josephfernandes273 | 13 Jul 2026 |
| KAN-40 | Task | Train model today | josephfernandes273 | 12 Jul 2026 |
| KAN-39 | Story | Robustness methods additions if any | josephfernandes273 | 12 Jul 2026 |
| KAN-36 | Task | appendix make concise | josephfernandes273 | 10 Jul 2026 |
| KAN-34 | Feature | Need to record dataset alphabet S onwards till Z | josephfernandes273 | 11 Jul 2026 |
| KAN-33 | Subtask | Reduce the Figure Caption | G. Akaash Samson | 10 Jul 2026 |
| KAN-32 | Subtask | Model Architecture in Design (4.4.1) | G. Akaash Samson | 10 Jul 2026 |
| KAN-30 | Subtask | Technology Part Revision | G. Akaash Samson | 10 Jul 2026 |
| KAN-27 | Subtask | Decide to Replace sys architecture content with shorter paras | G. Akaash Samson | 10 Jul 2026 |
| KAN-23 | Subtask | Adding the Revised Literature Survey | G. Akaash Samson | 10 Jul 2026 |
| KAN-16 | Subtask | Revise Conclusion | G. Akaash Samson | 10 Jul 2026 |
| KAN-15 | Subtask | Revise Future Scope | G. Akaash Samson | 10 Jul 2026 |
| KAN-14 | Subtask | Revise Challenges and Limitation | G. Akaash Samson | 10 Jul 2026 |
| KAN-13 | Subtask | Shortening Introduction | G. Akaash Samson | 10 Jul 2026 |
| KAN-12 | Subtask | Synopsis Expansion | G. Akaash Samson | 10 Jul 2026 |
| KAN-11 | Subtask | Add Resource Links in Technology section | G. Akaash Samson | 10 Jul 2026 |
| KAN-6 | Subtask | Johnny's Implementation Part Revision | G. Akaash Samson | 10 Jul 2026 |
| KAN-5 | Subtask | Bibliography; confirm with Ma'am if Alphabetical or in the order of reference | G. Akaash Samson | 10 Jul 2026 |
| KAN-3 | Task | Report Formatting and Revisions | G. Akaash Samson | 10 Jul 2026 |
| KAN-1 | Task | Implementation Section Revisions | G. Akaash Samson | 12 Jul 2026 |
