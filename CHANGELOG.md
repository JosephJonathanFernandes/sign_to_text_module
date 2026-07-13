# Changelog

All notable changes to this project will be documented in this file.

## [2.2.0] - 2026-07-01

### Added
- **Soft Heuristic Adjustment Layer:** Introduced multiplicative penalty system to dynamically down-weight anatomically impossible signs based on live visual confidence, replacing unstable hierarchical classifiers.
- **Advanced Temporal Augmentations:** Added `TimeMasking` (simulates contiguous frame dropping / webcam lag) and `Scattered Dropout` to prevent the model from learning artificial interpolation logic.
- **Spatial GNN Branch:** Integrated a lightweight Spatial Graph Neural Network to process explicit finger-joint connectivity for complex topologies.
- **Adapter Safety Safeguards:** Introduced strict thresholds (`>40 samples`, `>3 classes`) constraining when adaptation is allowed during live inference to prevent confirmation bias.
- **Experimental CVAE Pipeline:** Added research scripts for Conditional Variational Autoencoder synthetic landmark generation.

### Changed
- **State Machine Debouncing:** Upgraded `SentenceBuilder` with a strict 3-frame `separator_counter` and an aggressive 45-frame identical-word cooldown (~1.5s) to suppress noise and stuttering.
- **Augmentation Strategy:** Formally rejected Generative Adversarial Networks (GANs) for sequence augmentation in favor of deterministic mathematical perturbations, ensuring temporal dynamics remain strictly anchored to human motion.
- **Latency Optimization:** Intentionally disabled HOG-based person detection (`disable_hog_detection = True`), shaving ~8ms of latency per frame at the acceptable trade-off of reduced person-aware filtering.
- **Hyperparameter Evolution:** Reduced learning rate to `3e-4` and early stopping patience to `10` to ensure stable convergence on current datasets.
- **Low Confidence Regime:** Standardized baseline acceptance confidence threshold to `~0.12`, relying on temporal momentum and state logic over absolute confidence peaks.

## [2.1.0] - 2026-06-28

### Added
- **HDF5 Storage Engine:** Introduced `compile_hdf5.py` and `assets/dataset.h5` support, reducing dataset initialization time by 209× (43.2s → 0.2s) and first-epoch execution time by 5.4× (98.5s → 18.2s).
- **Test Infrastructure:** Added robust `pytest` suite covering unit and API integration scenarios (`tests/unit`, `tests/api`).
- **CI/CD Pipeline:** Added GitHub Actions workflow (`.github/workflows/ci.yml`) for automated linting, security scanning, and unit testing on PRs.
- **Developer Tooling:** Added `black`, `isort`, `ruff`, `mypy`, and `bandit` configurations with `pre-commit` hooks.
- **Setup Scripts:** Added `scripts/setup.ps1` and `scripts/setup.sh` for one-click developer onboarding.
- **Documentation:** Added `docs/ARCHITECTURE.md`, `SECURITY.md`, and a comprehensive `.env.example` file.

### Changed
- **API Security:** Migrated wildcard CORS policy (`allow_origins=["*"]`) to an environment variable (`ALLOWED_ORIGINS`) for production safety.
- **Repository Structure:** Migrated root-level utility scripts (`tools/*`) to `src/tools/`, leaving lightweight deprecation wrappers.
- **Dependency Management:** Segregated development dependencies into `requirements-dev.txt`.
- **Git Ignore:** Expanded `.gitignore` to explicitly exclude all compiled models (`.pth`, `.ckpt`), caches, and environment secrets.

### Fixed
- Duplicate function definitions in `api/app.py`.
- Hardcoded spatial dimensions in API `LandmarkFrame` Pydantic models.

---

## [2.0.0] - 2026-06-25

### Added
- Initial modular `src/` layout.
- Config-driven deep learning architecture.
- Real-time FastAPI WebSocket streaming.
