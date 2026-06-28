# Repository Audit — ISL Sign-to-Text

> Generated: 2026-06-28 | Auditor: Principal Engineer Review

---

## 1. Repository Tree

```
sign_to_text/
├── api/                          ← FastAPI application layer
│   ├── app.py                    ← Main ASGI app, endpoints, WebSocket
│   ├── audit_api.py              ← [UNUSED] Not imported by any module
│   ├── inference.py              ← run_predict() wrapper
│   ├── schemas.py                ← Pydantic request/response models
│   ├── session.py                ← UUID session state management
│   ├── FRONTEND_HANDOFF.md
│   └── README.md
│
├── src/                          ← Core source code
│   ├── core/
│   │   ├── config.py             ← Master config (1134 lines, dataclasses)
│   │   ├── main.py               ← CLI pipeline orchestration
│   │   ├── webcam.py             ← Real-time inference loop
│   │   ├── camera_manager.py     ← Camera init helper
│   │   ├── inference_engine.py   ← Inference session wrapper
│   │   ├── landmark_processor.py ← MediaPipe landmark math
│   │   └── motion_tracker.py     ← Frame-to-frame motion estimation
│   │
│   ├── preprocessing/
│   │   ├── preprocess.py         ← Video → .npy extraction
│   │   ├── dataset.py            ← ISLDataset (HDF5 + .npy dual-path)
│   │   ├── augmentations.py      ← 20 landmark augmentations
│   │   ├── merge_augmentations.py← Frame-splicing augmentation
│   │   ├── collect_data.py       ← Webcam data collection
│   │   └── cleanup_dataset_npy.py← Near-duplicate removal
│   │
│   ├── training/
│   │   ├── model.py              ← SignLanguageGRU (BiGRU+GNN+Attention)
│   │   ├── spatial_gnn.py        ← Spatial GCN over hand skeleton
│   │   ├── train.py              ← Training loop, K-fold CV
│   │   ├── adapter_model.py      ← Residual log-prob adapter MLP
│   │   └── adapter_training.py   ← Async background adapter trainer
│   │
│   ├── inference/
│   │   ├── ensemble.py           ← Ensemble load + predict
│   │   ├── onnx_inference.py     ← ONNX Runtime wrapper
│   │   ├── onnx_ensemble.py      ← Mixed ONNX+PyTorch ensemble
│   │   ├── onnx_ensemble_integration.py ← Drop-in ensemble replacement
│   │   ├── temporal_postprocessor.py    ← ConfidenceSmoother + Stable
│   │   ├── sentence_builder.py   ← Continuous sign-to-text assembly
│   │   ├── nlp_postprocessor.py  ← Grammar/punctuation cleanup
│   │   ├── hand_selector.py      ← Multi-signer hand assignment
│   │   └── pseudo_buffer.py      ← Pseudo-label buffering
│   │
│   ├── shared/
│   │   └── feature_extractor.py  ← Single source of truth for features
│   │
│   ├── tools/
│   │   ├── compile_hdf5.py       ← HDF5 compiler
│   │   └── benchmark_dataset.py  ← NPY vs HDF5 benchmark
│   │
│   ├── utils/
│   │   ├── pipeline_logger.py    ← Structured logging
│   │   ├── profiling.py          ← Latency profiler
│   │   ├── pseudo_utilities.py   ← Pseudo-label utilities
│   │   └── quantization_utils.py ← Checkpoint quantization
│   │
│   └── ui/
│       └── renderer.py           ← OpenCV overlay rendering
│
├── scripts/                      ← Data pipeline scripts
│   ├── export_onnx.py            ← PyTorch → ONNX export
│   ├── quantize_onnx.py          ← FP32 → INT8 quantization
│   ├── augment_pipeline.py       ← Landmark augmentation runner
│   ├── augment_video_pipeline.py ← Video augmentation runner
│   ├── balance_processed_dataset.py
│   ├── random_downsample_processed.py
│   ├── quality_filter_hybrid.py  ← Near-duplicate filter (74KB — large)
│   ├── train_kfold_resume.py     ← K-fold orchestration
│   ├── debug_model.py            ← Model shape trace
│   ├── evaluate_quantized_model.py
│   ├── quantize_model.py
│   └── update_hand_classification.py
│
├── tools/                        ← Developer utilities (root-level)
│   ├── validate_npy.py
│   ├── verify_imports.py
│   ├── dependency_analyzer.py
│   ├── generate_mermaid.py
│   ├── build_weighted_filelist.py
│   ├── debug_onnx_input_check.py
│   ├── generate_negative_root.py
│   └── grid_search_archived.py
│
├── tests/
│   ├── unit/
│   │   ├── test_velocity.py
│   │   ├── test_config.py        ← [NEW]
│   │   ├── test_feature_extractor.py ← [NEW]
│   │   └── test_hdf5.py          ← [NEW]
│   ├── integration/
│   │   ├── test_api.py
│   │   └── verify_refactor.py
│   ├── api/
│   │   └── test_endpoints.py     ← [NEW]
│   ├── e2e/
│   │   └── simulate_frontend.py
│   ├── fixtures/
│   └── conftest.py               ← [NEW]
│
├── experimental/                 ← CVAE research experiments
├── docs/                         ← Technical documentation
├── data/                         ← JSON configs
├── Paper/                        ← FYP paper scripts
├── archive/                      ← Archived/deprecated files [NEW]
│
├── main.py                       ← Root shim (backward compat)
├── config.py                     ← Root shim (backward compat)
├── model.py                      ← Root shim (backward compat)
├── train.py                      ← Root shim (backward compat)
├── webcam.py                     ← Root shim (backward compat)
├── run_api.py                    ← API launcher
├── requirements.txt
├── requirements-dev.txt          ← [NEW]
├── pyproject.toml
└── .github/workflows/ci.yml      ← [NEW]
```

---

## 2. Dependency Graph (Core Modules)

```
api/app.py
  ├── src.core.config          (get_config)
  ├── src.inference.ensemble   (load_ensemble, ensemble_predict)
  ├── api.inference            (run_predict)
  ├── api.schemas              (HealthResponse, PredictRequest, ...)
  ├── api.session              (InferenceSession, create_session)
  └── src.shared.feature_extractor (build_single_frame_features)

src/core/main.py
  ├── src.core.config
  ├── src.preprocessing.preprocess
  ├── src.preprocessing.dataset
  ├── src.training.train
  ├── src.inference.ensemble
  └── src.core.webcam

src/preprocessing/dataset.py
  ├── config (root shim → src.core.config)
  └── h5py (optional, HDF5 fast-path)

src/training/model.py
  ├── src.core.config
  └── src.training.spatial_gnn

src/shared/feature_extractor.py
  └── numpy (no internal dependencies — intentionally isolated)
```

---

## 3. Import Graph — Circularity Check

| Module Pair | Circular? | Notes |
|-------------|-----------|-------|
| `config` ↔ `dataset` | ✅ No | One-directional |
| `config` ↔ `model` | ✅ No | One-directional |
| `api.app` ↔ `api.inference` | ✅ No | One-directional |
| `feature_extractor` ↔ `*` | ✅ No | Leaf module, no internal imports |
| `ensemble` ↔ `model` | ✅ No | One-directional |

**No circular imports detected.**

---

## 4. Unused Files

| File | Status | Recommendation |
|------|--------|----------------|
| `api/audit_api.py` | Not imported by any module | Move to `archive/` |
| `tools/grid_search_archived.py` | Historical reference | Keep or archive |
| `tools/generate_negative_root.py` | Standalone script | Keep |

---

## 5. Duplicate Logic

| Area | Files | Overlap |
|------|-------|---------|
| Velocity computation | `src/preprocessing/preprocess.py` + `src/shared/feature_extractor.py` | **Intentional** — extractor is verified zero-drift copy |
| Config access | Root `config.py` shim + `src/core/config.py` | **Intentional** — backward compat |

---

## 6. Security Findings

| Severity | Finding | File | Action |
|----------|---------|------|--------|
| 🟡 Medium | `allow_origins=["*"]` with no env-var fallback | `api/app.py:120` | Fix: production-safe env-var split |
| 🟡 Medium | No `.env` / secret documentation | Root | Add `.env.example` + `SECURITY.md` |
| 🟢 Low | Debug mode via env var — correct pattern | `api/app.py:58` | No action needed |
| 🟢 Low | No hardcoded secrets, API keys, or tokens | All files | Confirmed clean |
| 🟢 Low | Model paths from config dataclass — no shell injection surface | Config | No action needed |

---

## 7. Configuration Issues

| Issue | Location | Severity |
|-------|----------|----------|
| Duplicate `health()` function definition (lines 132–133) | `api/app.py` | 🔴 Bug — dead code, one definition shadows the other |
| `LandmarkFrame` schema has hardcoded `feature_dimension=506`, `sequence_length=20` | `api/schemas.py:95–96` | 🟡 Medium — breaks silently if config changes |

---

## 8. Technical Debt

| Item | Priority | Notes |
|------|----------|-------|
| `scripts/quality_filter_hybrid.py` is 74KB (single file) | Low | Works, but difficult to maintain |
| `src/core/config.py` is 1134 lines | Low | Well-structured but could split into domain sub-configs |
| No `requirements-lock.txt` / `uv.lock` | Medium | Dependency drift risk in CI |
| No pre-commit hooks | Medium | Manual lint discipline required |
| No CI/CD | High | No automated quality gates |
| `tools/` duplicates `src/tools/` purpose | Low | Consolidate via wrappers |

---

## 9. Risk Level Per Module

| Module | Risk | Reason |
|--------|------|--------|
| `src/core/config.py` | 🟢 Low | Dataclass-based, self-validating |
| `src/shared/feature_extractor.py` | 🟢 Low | Leaf module, well-tested |
| `api/app.py` | 🟡 Medium | Has duplicate function definition |
| `api/schemas.py` | 🟡 Medium | Hardcoded fallback dimensions |
| `src/preprocessing/dataset.py` | 🟢 Low | HDF5 + fallback both tested |
| `src/training/model.py` | 🟢 Low | Stable, no external side-effects |
| `scripts/quality_filter_hybrid.py` | 🟡 Medium | Very large file, hard to test |
| `experimental/` | 🟢 Low | Research-only, not in production path |

---

## 10. Module Classification

| Category | Modules |
|----------|---------|
| **Core ML** | `src/training/model.py`, `src/training/spatial_gnn.py`, `src/training/train.py` |
| **Inference** | `src/inference/ensemble.py`, `src/inference/onnx_inference.py`, `src/inference/temporal_postprocessor.py`, `src/inference/sentence_builder.py` |
| **API** | `api/app.py`, `api/schemas.py`, `api/session.py`, `api/inference.py` |
| **Preprocessing** | `src/preprocessing/preprocess.py`, `src/preprocessing/dataset.py`, `src/preprocessing/augmentations.py` |
| **Shared** | `src/shared/feature_extractor.py` |
| **Config** | `src/core/config.py` |
| **Scripts** | `scripts/*.py` |
| **Tools** | `tools/*.py`, `src/tools/*.py` |
| **Assets** | `assets/processed/`, `assets/ensemble/`, `assets/dataset.h5` |
| **Docs** | `docs/`, `README.md`, `CHANGELOG.md`, `FEATURE_CONTRACT.md` |
| **Tests** | `tests/unit/`, `tests/integration/`, `tests/api/`, `tests/e2e/` |
| **Experimental** | `experimental/*.py`, `Paper/*.py` |
