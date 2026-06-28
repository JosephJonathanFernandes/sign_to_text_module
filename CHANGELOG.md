# Changelog

All notable changes to this project will be documented in this file.

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
