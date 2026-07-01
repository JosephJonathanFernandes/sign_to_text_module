# Missing Knowledge Report & Repository Audit

This report surfaces hidden project intelligence, undocumented heuristics, failed experiments, and hyperparameter evolutions discovered during a deep repository audit of memory files, configurations, git history, and experimental code paths.

---

## Part 1: Discovered Hidden Knowledge Items

### Item 1: Severe Confidence Threshold Deflation
* **Knowledge discovered**: The system operates with a relatively low confidence threshold (~0.12). Observed inference confidence values commonly fall in the ~0.1-0.2 range, so prediction stability relies more on temporal consistency and state logic than on high absolute confidence values.
* **Evidence source**: Code comments in `src/core/config.py` (`InferenceConfig`).
* **Affected files**: `src/core/config.py`, `src/inference/sentence_builder.py`
* **Suggested documentation update**: Update `docs/inference_pipeline.md` to explicitly mention the reliance on momentum over absolute confidence.
* **Confidence**: High (Config values) / Medium (Inferred motivation)

### Item 2: Aggressive Sentence Builder Debouncing (The "Same Word Cooldown")
* **Knowledge discovered**: The sentence builder includes an aggressive cooldown mechanism intended to reduce repeated-word stuttering during continuous inference. A massive 45-frame cooldown (~1.5 seconds at 30fps) is enforced before the same word can be predicted again consecutively (`same_word_cooldown_frames = 45`). Additionally, 3 consecutive frames of noise (`__transition__`, `...`) are required to break a sign (`separator_counter >= 3`).
* **Evidence source**: Hardcoded logic and comments in `src/inference/sentence_builder.py`.
* **Affected files**: `src/inference/sentence_builder.py`
* **Suggested documentation update**: Add a section in `docs/inference_pipeline.md` detailing the state machine's anti-stutter mechanics and the identical-word cooldown.
* **Confidence**: High (Cooldown values)

### Item 3: Confusable Pair Strictness Multiplier
* **Knowledge discovered**: The system applies a stricter transition requirement for known confusable sign pairs in order to reduce false transitions between visually similar classes during live inference. It applies a `1.3x` strictness multiplier to the confidence threshold if the current sign transitions into a known "similar sign" (e.g., pairs in `similar_signs.json`).
* **Evidence source**: `is_confusable_pair` and `get_transition_requirement` methods in `src/inference/sentence_builder.py`.
* **Affected files**: `src/inference/sentence_builder.py`, `data/similar_signs.json`
* **Suggested documentation update**: Document the "Dynamic Confusable Pair Thresholding" logic in `docs/DECISIONS.md`.
* **Confidence**: High (Threshold values)

### Item 4: HOG Person Detection Disabled for Latency
* **Knowledge discovered**: HOG-based person detection was intentionally disabled (`disable_hog_detection = True`) to shave off ~8ms of latency per frame. The team accepted the trade-off of losing person-aware filtering, assuming the background will mostly have a single signer.
* **Evidence source**: Comments in `src/core/config.py` (`PreprocessingConfig`).
* **Affected files**: `src/core/config.py`, `src/preprocessing/preprocess.py`
* **Suggested documentation update**: Add a "Latency vs. Accuracy Trade-offs" section in `docs/ARCHITECTURE.md` documenting the removal of HOG.
* **Confidence**: High (HOG disabled flag)

### Item 5: Adapter Model Safety Measures
* **Knowledge discovered**: The continuous learning `AdapterModel` trains on live pseudo-labels. Strict safety thresholds (`adapter_min_saved_samples = 40`, `adapter_min_classes = 3`) were added. These thresholds act as safeguards intended to reduce the risk of unstable adaptation from pseudo-labeled live data.
* **Evidence source**: `.internal/codebase_audit_report.md` (Phase 3) and `src/core/config.py`.
* **Affected files**: `src/training/adapter_training.py`, `src/core/config.py`
* **Suggested documentation update**: Add a section to the Continuous Learning documentation highlighting the conditions for live adaptation.
* **Confidence**: High (Config values) / Medium (Expected effects)

### Item 6: The Spatial GNN Addition
* **Knowledge discovered**: A lightweight Spatial GNN branch was introduced to model explicit finger-joint connectivity while maintaining a small parameter footprint (<2K additional parameters).
* **Evidence source**: `src/core/config.py` (`use_gnn = True` and documentation).
* **Affected files**: `src/core/config.py`, `src/training/spatial_gnn.py`, `src/training/model.py`
* **Suggested documentation update**: Explicitly document the hybrid BiGRU + Spatial GNN topology in `docs/model_architecture.md`.
* **Confidence**: High (Parameter values)

### Item 7: CVAE and Synthetic Data Evolution
* **Knowledge discovered**: GAN-based approaches were rejected because skeletal landmark sequence generation was assessed as high-risk and high-complexity relative to deterministic augmentation methods already used in the pipeline. Experimental CVAE work appears to investigate synthetic landmark generation as a research direction rather than as a production requirement.
* **Evidence source**: `CHANGELOG.md` and the `experimental/` directory structure (`cvae_landmarks.py`, `train_cvae.py`).
* **Affected files**: `experimental/*`
* **Suggested documentation update**: Document the ongoing CVAE research in `docs/dataset.md`.
* **Confidence**: Medium (Historical motivations)

### Item 8: Hyperparameter Retraction
* **Knowledge discovered**: The team adjusted hyperparameters for smaller datasets, reducing the learning rate from `5e-4` to `3e-4`, increasing `weight_decay` to `5e-4`, and reducing early stopping `patience` from 20 to 10. `use_focal_loss` is currently disabled.
* **Evidence source**: Inline comments in `src/core/config.py` (`TrainingConfig`).
* **Affected files**: `src/core/config.py`
* **Suggested documentation update**: Add a "Training Stability Findings" section in `docs/training_pipeline.md` capturing these hyperparameter choices.
* **Confidence**: High (Config values) / Medium (Inferred motivation)

---

## Part 2: Missing Knowledge Summary

### Critical Undocumented Decisions
1. **The Sub-15% Confidence Regime:** The system operates with a relatively low confidence threshold (~0.12). Observed inference confidence values commonly fall in the ~0.1-0.2 range, so prediction stability relies more on temporal consistency and state logic than on high absolute confidence values.
2. **The 45-Frame Stutter Lock:** The sentence builder employs a strict cooldown to reduce repeated-word stuttering, which limits how quickly consecutive identical words can be registered.
3. **Hard-coded Confusable Penalties:** Similar signs (e.g., from `similar_signs.json`) are given a 30% stricter transition threshold to reduce false positive jumps between similar classes.

### Medium-Priority Missing Context
1. **HOG Removal:** The removal of person-aware filtering means the webcam pipeline optimizes for speed but is potentially more sensitive to background motion.
2. **Hyperparameter Deflation:** Learning rates and early stopping patience were scaled back to ensure stable convergence on current dataset boundaries.
3. **Phase 2 Archived Fine-Tuning:** The system uses a `processed_del` archive of hard/rejected samples for a Phase 2 fine-tuning pass.

### Historical Insights Worth Preserving
1. **The GAN Rejection:** GAN-based approaches were rejected because skeletal landmark sequence generation was assessed as high-risk and high-complexity relative to deterministic augmentation methods already used in the pipeline.
2. **Adapter Flow:** The strict thresholds (`>40 samples`, `>3 classes`) act as safeguards intended to reduce the risk of unstable adaptation from pseudo-labeled live data.

### Knowledge that only exists in memory files or git history
1. **"Soft Heuristic Adjustment Layer" (ADR-005):** The git history (CHANGELOG) explicitly mentions this replaced unstable hierarchical classifiers.
2. **Diversity Embeddings / Duplicate Suppression:** A hybrid quality filter pipeline (`quality_filter_hybrid.py`) was introduced to manage dataset sample variety.
3. **ONNX INT8 Quantization Pivot:** The team optimized for CPU edge devices via INT8 quantization and disabling HOG entirely to maintain real-time FPS.
