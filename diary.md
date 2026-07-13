# Engineering Log (January 1 - July 11, 2026)

**January 2026**

* **Jan 1-5**: Initiated the real-time sign language translation project. Evaluated OpenPose vs. MediaPipe, ultimately selecting MediaPipe Holistic for its superior CPU inference efficiency.
* **Jan 8-12**: Began modularizing the core engine (`src/core/`). Built the initial `camera_manager.py` and `landmark_processor.py`. Solved severe OpenCV I/O blocking by offloading the video capture into a background daemon thread.
* **Jan 15-20**: Encountered significant spatial variance issues due to camera distance. Engineered a robust bounding-box normalization strategy to ensure spatial consistency.
* **Jan 21-25**: Discovered that parsing raw video during PyTorch training was a severe I/O bottleneck. Wrote extraction scripts to process landmarks upfront into `.npy` arrays, and prototyped dynamic PyTorch dataset loaders.
* **Jan 27-30**: Prototyped the predictive architecture in `src/training/model.py`. Shifted from a pure GRU approach to a hybrid Conv1D + Bidirectional GRU (Bi-GRU) to capture both micro-temporal flicks and macro-sequence context.

**February 2026**

* **Feb 4-6**: Addressed early classification failures on visually similar signs by researching and integrating Focal Loss.
* **Feb 10-13**: Centralized all hyperparameters and paths into `src/core/config.py`. Built the primary training loop in `src/training/train.py`, incorporating validation split logic and `train_ablation.py` for structured ablation studies.
* **Feb 17-20**: Overhauled the preprocessing pipeline (`src/preprocessing/preprocess.py`). Encountered MediaPipe frame drops due to motion blur; engineered a zero-padding strategy to maintain sequence dimensionality.
* **Feb 21-26**: Initialized the Git repository. Debugged validation leakage in the PyTorch `DataLoader` and committed fixes.
* **Feb 28**: Pushed the foundational codebase.

**March 2026**

* **March 3-10**: Pivoted focus toward experimental machine learning architectures. Researched Spatial Temporal Graph Convolutional Networks (ST-GCNs) for skeletal mapping.
* **March 15-25**: Developed a Conditional Variational Autoencoder (CVAE) (`experimental/cvae_landmarks.py`, `train_cvae.py`) to generate synthetic landmark sequences. 
* **March 26-31**: To validate the synthetic CVAE outputs, built an adversarial Quality Discriminator (`quality_discriminator.py`, `train_quality_discriminator.py`) and latent space visualization tools (`visualize_latent_space.py`).

**April 2026**

* **April 1-4**: Returned to the core Conv-BiGRU model. Integrated a specialized face-proximity attention module. To handle static frames, built `motion_tracker.py` and implemented motion gating.
* **April 7-11**: Expanded vocabulary recording. Integrated `src/inference/hand_selector.py` to route standalone hand landmarks when facial data was obscured.
* **April 14-17**: Tackled continuous signing issues. Built `src/augmentations1/transition_generator.py` and `boundary_noise.py` to simulate realistic fluid transitions between discrete signs.
* **April 22-27**: Solved the "sign lock-in" bug (where the model hung on past predictions). Replaced hard-lock logic with a sliding-window `TemporalPostProcessor` using dynamic confidence thresholds.
* **April 29-30**: Drafted the Natural Language Processing pipeline (`src/inference/nlp_postprocessor.py`, `sentence_builder.py`) to translate raw predictive tokens into grammatically structured sentences.

**May 2026**

* **May 1-4**: Overhauled data augmentation. Built `src/preprocessing/augment_pipeline.py` and `merge_augmentations.py` to scale synthetic generation to over 5,600 unique sequences using face-anchor shifts.
* **May 6-10**: Engineered advanced dataset tooling. Wrote `quality_filter_hybrid.py` to strip corrupted samples and `balance_processed_dataset.py` to enforce strict class distribution equity.
* **May 12-16**: To resolve PyTorch dataloader bottlenecks on the massive new dataset, wrote `src/tools/compile_hdf5.py` to compile all sequences into a single high-throughput `dataset.h5` container.
* **May 18-20**: Prototyped modular fine-tuning strategies (`src/training/adapter_training.py`). Developed `tools/generate_dataset_heuristics.py` to synthesize negative action frames for robust background rejection.
* **May 25-28**: Hardened the training pipeline for long-running epochs by implementing K-Fold cross-validation with fault-tolerant resume capabilities (`train_kfold_resume.py`).

**June 2026**

* **June 2-5**: Transitioned from PyTorch to ONNX to maximize local CPU performance (`src/inference/export_onnx.py`). Rewrote complex tensor operations to achieve ONNX compatibility.
* **June 9-12**: Built a rigorous model quantization pipeline (`quantize_onnx.py`, `quantize_model.py`) to aggressively reduce inference latency.
* **June 14-18**: Integrated multi-checkpoint prediction blending via `src/inference/ensemble.py` and `onnx_ensemble_integration.py` to maximize accuracy on complex continuous sequences.
* **June 20-23**: Engineered deep latency profiling logic (`src/utils/profiling.py`) and benchmarked the inference engine (`benchmark_inference.py`), compiling the findings into `PROFILING_LATENCY_REPORT.md`.
* **June 24-28**: Repository Transformation v2.0. Architected the backend service (`api/app.py`) with a persistent FastAPI WebSocket (`/ws/translate`). Defined strict Pydantic payload schemas (`FEATURE_CONTRACT.md`). Deployed DevOps infrastructure including GitHub Actions CI (`.github/workflows/ci.yml`) and reorganized a comprehensive `pytest` matrix. Integrated Domain Adversarial Neural Network (DANN) via Gradient Reversal Layer (GRL).
* **June 29-30**: Massive Data Engineering Overhaul. Replaced manual sign classifications with dataset-derived heuristics (`keypoints.csv`). Built an automated spatial augmentation pipeline. Implemented a hybrid dataset curation pipeline featuring quality filtering, diversity embeddings, and duplicate suppression. Built performance benchmarking and ONNX validation tools.
* **July 1-2**: Inference & NLP Upgrades. Implemented `ISLDataset` with HDF5 support for 200x faster I/O. Engineered the `SentenceBuilder` for real-time sign recognition, handling state-based word transitions and NLP post-processing. Created a lightweight MLP adapter (`AdapterTrainingManager`) for ensemble output correction and domain adaptation.
* **July 3-4**: Architectural Expansion. Engineered a lightweight Spatial GNN (Graph Neural Network) to learn structural relationships across the 21 MediaPipe hand nodes. Wrote dataset balancing scripts (pruning, duplication, random downsampling) to enforce class equilibrium.
* **July 5-11**: Documentation, Backups & Finalization. Executed critical pendrive storage operations to securely backup the massive dataset, model checkpoints, and repository states. Synchronized core documentation (`ARCHITECTURE_AND_DESIGN.md`). Compiled exhaustive Final Year Project (FYP) dissertation chapters, technical audits, and Viva preparation materials. Ran end-to-end WebSocket simulations.
* **July 12**: Identified a critical flaw in real-world Out-of-Distribution (OOD) data handling where the model forced random noise and idle states into valid signs (high False Acceptance Rate). Began **Phase 2 Robustness**, explicitly collecting over 3,000 negative sequences (idle, transitions, random hand movements) to train a dedicated `__reject__` class.
* **July 13**: Finalized the robustness evaluation. Implemented a confidence threshold sweep (ROC analysis) and discovered that the model learns the `__reject__` concept primarily through slashed confidence (median 0.37 vs 0.97 for valid signs). Established an optimal 0.5 operating threshold yielding a 0.74% False Rejection Rate. Consolidated all fragmented documentation and performed a massive repository cleanup, deleting cache folders and legacy scripts.
* **July 14-17 (Planned)**: Presentation (PPT) preparation and massive model vocabulary expansion. Successfully scaled the dataset vocabulary from 89 to 148 classes (totaling over 58,344 processed sequence samples). Will continue recording additional words and conducting rigorous testing. Planned roadmap includes exploring deafblind accessibility features, further hardening the API, and officially validating the model's sign predictions against Goa Board videos and Muskaan Ma'am's ISL book.

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
