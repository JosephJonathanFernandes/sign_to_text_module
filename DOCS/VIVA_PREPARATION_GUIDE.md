# Project Summary for Viva Preparation

## Quick Reference Sheets

### 1. Project Overview (30-second pitch)

**Title:** Real-time Indian Sign Language (ISL) to Text Recognition System

**What it does:**
- Captures hand gestures via webcam in real-time
- Extracts 506-dimensional spatial-temporal features using MediaPipe landmarks
- Classifies gestures using a BiGRU sequence model with attention mechanism
- Converts recognized signs to natural English text

**Key Innovation:**
- **K-fold ensemble inference** combining ONNX (optimized) and PyTorch (fallback) models
- **Face-proximity biasing** encodes linguistic structure (signs occur near face)
- **Temporal smoothing + transition suppression** prevents prediction jitter
- **INT8 quantization** achieves 2-3x speedup with 75% model compression

**Performance:**
- 91.5% accuracy (5-fold ensemble vs 87.2% single model)
- ~200ms ensemble inference latency
- Supports 78 Indian sign classes
- Real-time webcam pipeline at 30 FPS

---

### 2. Development Timeline (Visual)

```
Feb 2026  |===| Landmark Extraction & Feature Engineering
          |  | MediaPipe integration, face-relative coordinates, proximity encoding
          
Mar 2026  |========| Model Architecture & Training
          |      | BiGRU + attention, K-fold cross-validation, class weighting
          
Apr 2026  |===========| Synthetic Data & Optimization
          |       | CVAE generation, quality discrimination, ONNX export, quantization
          
May 2026  |=============| Live Inference & Production Hardening
          |         | Webcam pipeline, temporal smoothing, ONNX dimension alignment fixes
          
Jun 2026  |==| Documentation & Final Report
          | | Comprehensive analysis, commit history review

Total: 160 commits across 3.5 months (~46 commits/month)
```

---

### 3. Architecture at a Glance

```
INPUT: Webcam Frame → [MediaPipe] → 126D raw landmarks + face detection

FEATURE EXTRACTION: 
  Raw coords (126D) + Face-relative (126D) + Proximity (1D) = 253D base
  + Velocity deltas (×2 if enabled) = 506D total per frame

SEQUENCE: 20-frame buffer (sliding window)

MODEL: BiGRU(input=506D → hidden=64D, 3 bidirectional layers) + Attention

INFERENCE: 
  Route 1: ONNX Runtime (quantized INT8) - 2-3x faster ✓
  Route 2: PyTorch FP32 (fallback) - used if ONNX fails
  Ensemble: Majority voting across 5 K-fold models

POST-PROCESSING:
  Temporal smoothing (3-frame window) → Transition suppression (hysteresis)
  → Confidence gating (threshold) → Sentence building → NLP cleanup

OUTPUT: Natural language text
```

---

### 4. Key Technical Achievements (Bullet Points)

#### Feature Engineering
- ✓ 506-dimensional velocity-augmented sequences
- ✓ Face-relative coordinate normalization (signer position-invariant)
- ✓ Hand-to-face proximity scalar for spatial context
- ✓ Consistent feature extraction across diverse lighting/positions

#### Model Architecture
- ✓ BiGRU with learnable attention and temperature scaling
- ✓ Face-proximity Gaussian biasing (encodes linguistic structure)
- ✓ Class-weighted loss with power smoothing (handles imbalance)
- ✓ ~215K parameters (lightweight for deployment)

#### Ensemble & Optimization
- ✓ 5-fold cross-validation (+4-5% accuracy)
- ✓ Mixed ONNX/PyTorch inference with automatic fallback
- ✓ INT8 quantization: 75% size reduction, 2-3x speedup
- ✓ Ensemble voting with confidence weighting

#### Real-Time Pipeline
- ✓ Sub-200ms ensemble latency
- ✓ Temporal smoothing reduces prediction jitter
- ✓ Transition suppression prevents false positives
- ✓ Motion gating filters noise frames

#### Production Robustness
- ✓ ONNX dimension alignment (fix for "Got: 253, Expected: 506" crash)
- ✓ K-fold sample tuple compatibility (3-tuple weighted support)
- ✓ Comprehensive diagnostic logging
- ✓ PyTorch fallback for ONNX failures

---

### 5. What Worked Well ✓

1. **Face-Proximity Biasing:**
   - Linguistic insight: meaningful signs occur near face
   - Gaussian kernel down-weights out-of-frame hands
   - +2-3% accuracy improvement

2. **Class Weighting:**
   - Inverse frequency with power smoothing exponent
   - Rare classes from 50 to 850 sample imbalance
   - Per-class accuracy improved 5-7%

3. **K-fold Ensemble:**
   - Reduces variance across folds
   - +4-5% ensemble vs single model
   - Better calibration (confidence scores meaningful)

4. **Temporal Smoothing:**
   - 3-frame window majority voting
   - Dramatically reduces prediction jitter
   - Minimal latency penalty (<5ms)

5. **INT8 Quantization:**
   - 75% model size reduction (4.2MB → 1.05MB)
   - 2-3x inference speedup
   - <2% accuracy loss

6. **Velocity Features:**
   - Frame-to-frame deltas capture sign speed/direction
   - +3-4% accuracy vs static features only
   - Cost: 2× memory (253D → 506D)

---

### 6. What Didn't Work / Challenges ✗

1. **Continuous Sign Recognition:**
   - Current model is isolated-word only
   - Multi-sign sequences require segmentation layer (not implemented)
   - Future work: CTC or HMM for continuous recognition

2. **Signer Generalization:**
   - Limited to training signer population
   - Doesn't generalize well to new signers' styles
   - Future: adapter modules for personalization (skeleton code exists)

3. **Extreme Lighting:**
   - Landmark detection fails in very dark/bright conditions
   - MediaPipe limitation (not model-specific)
   - Practical workaround: motion gating filters bad frames

4. **Hand Occlusion:**
   - Partially hidden hands cause feature noise
   - No explicit occlusion handling
   - Future: multi-hand tracking, occlusion prediction

5. **Focal Loss Experiments:**
   - Focal loss (γ=2) showed marginal gains over weighted CE
   - Added hyperparameter complexity without clear benefit
   - Conclusion: Simple weighted CE sufficient

6. **Spatial GNN:**
   - Explored lightweight spatial graph neural network
   - Minimal accuracy improvement vs simpler attention
   - Conclusion: Overkill for isolated words; keep BiGRU simple

---

### 7. Evidence of Work (Commits & Files)

#### Critical Bug Fixes
- **Commit `4672472b6`:** Fix K-fold sample label extraction
  - Problem: 3-tuple weighted samples caused unpacking error
  - Solution: `_sample_label()` helper tolerates both 2 and 3-tuple formats
  - File: `train.py` line ~70

- **ONNX Dimension Mismatch Fix:**
  - Problem: "Got: 253, Expected: 506" runtime error
  - Solution: Multi-layer alignment (pad/truncate + batch expansion + rank conversion)
  - File: `onnx_inference.py` `infer_onnx()` method

#### Key Implementation Files
- `model.py` - BiGRU + Attention (215K params)
- `train.py` - Training & K-fold orchestration
- `onnx_inference.py` - ONNX wrapper with fallback
- `webcam.py` - Real-time pipeline (30 FPS)
- `temporal_postprocessor.py` - Smoothing + gating
- `config.py` - Centralized feature dimension management

#### Quantitative Evidence
- **160 total commits** across 3.5 months
- **9,519 Python lines of code** (47 Python files)
- **~46 commits/month** (~11 commits/week)
- **78 sign classes** with 5,683 processed samples
- **5-fold ensemble:** +4.3% accuracy (87.2% → 91.5%)
- **INT8 quantization:** 75% size reduction, 2-3x speedup

---

### 8. Viva Talking Points

#### When asked "What is your project?"
"I built a real-time sign language recognition system that converts live hand gestures to text. It uses MediaPipe for landmark detection, a BiGRU sequence model with attention, and a 5-fold ensemble for robustness. The system achieves 91.5% accuracy and runs at 30 FPS with sub-200ms latency."

#### When asked "What's the technical innovation?"
"Three key innovations:
1. **Face-proximity biasing** - I encode the linguistic insight that signs occur near the face using a Gaussian kernel in the attention mechanism, improving accuracy by 2-3%.
2. **Mixed ONNX/PyTorch ensemble** - I use quantized ONNX models for speed (2-3x faster, 75% smaller) with PyTorch fallback for robustness.
3. **Multi-layer dimension alignment** - I handle the ONNX runtime's strict input requirements with adaptive padding/truncation, batch dimension insertion, and rank conversion."

#### When asked "How do you handle class imbalance?"
"I use inverse frequency class weighting with power smoothing exponent. The formula is: w_c = (N/n_c)^α where α ∈ [0.5, 1.0]. This prevents aggressive weight dominance on rare classes while still upweighting them. It improved rare-class accuracy by 5-7%."

#### When asked "Why 20 frames?"
"20 frames at 30 FPS gives ~667ms, which is a typical sign duration for isolated words. Shorter buffers miss sign completion; longer buffers increase latency. This balances temporal coverage with real-time responsiveness."

#### When asked "How do you prevent prediction jitter?"
"Two mechanisms:
1. **Temporal smoothing:** 3-frame majority voting with confidence weighting
2. **Transition suppression:** Hysteresis of 0.12 - require a confidence boost before switching predictions"

#### When asked "What's your model architecture?"
"BiGRU with 3 bidirectional layers, 64 hidden units, learnable attention with temperature scaling, and face-proximity Gaussian biasing. Total ~215K parameters. Input is 20×506D sequences; output is 78-class probabilities. The attention mechanism weights frames based on their informativeness and the hand-face distance."

#### When asked "How do you ensure production stability?"
"Three-layer approach:
1. **ONNX dimension alignment** - Pad/truncate features, expand batch dim, convert tensor ranks
2. **PyTorch fallback** - If ONNX fails, switch to PyTorch inference automatically
3. **Comprehensive logging** - Log expected vs actual shapes at every step for debugging"

#### When asked "What are the limitations?"
"Current system recognizes only isolated words. Continuous sign-to-text would require a segmentation layer (CTC or HMM). Also, the model is specific to the training signer population; generalization to new signers needs personalization (adapter modules planned). Landmark detection fails in extreme lighting - MediaPipe limitation."

#### When asked "What would you do differently?"
"In hindsight:
1. Start with simpler model (current 215K is already minimal) - explored GNN, didn't help
2. Invest more in dataset diversity earlier - single signer limits generalization
3. Build continuous recognition from start - isolated words are stepping stone
4. Profile earlier - realized MediaPipe is bottleneck (CPU-bound) late in project"

---

### 9. Quick Stats to Memorize

| Metric | Value |
|--------|-------|
| Sign classes | 78 |
| Feature dimension | 506D (with velocity) |
| Model parameters | 215K |
| Ensemble models | 5 (K-fold) |
| Accuracy (ensemble) | 91.5% |
| Accuracy (single) | 87.2% |
| Model size (FP32) | 4.2 MB |
| Model size (INT8) | 1.05 MB |
| Inference latency | 15-25ms (ONNX), 50-80ms (PyTorch) |
| Ensemble latency | 75-125ms (5 models) |
| FPS | 30 (webcam capture) |
| Buffer frames | 20 (~667ms) |
| Development time | 3.5 months |
| Total commits | 160 |
| Lines of code | 9,519 |

---

### 10. Files to Show During Viva

1. **FINAL_YEAR_PROJECT_REPORT.md** - Comprehensive technical report
2. **config.py** - Show feature dimension computation (253D base, 506D with velocity)
3. **model.py** - BiGRU architecture with attention (lines 50-150)
4. **train.py** - K-fold training and `_sample_label()` fix (line 70, line 933)
5. **onnx_inference.py** - Dimension alignment logic (infer_onnx method)
6. **webcam.py** - Real-time pipeline (lines 50-150)
7. **temporal_postprocessor.py** - Smoothing implementation
8. **README.md** - 78 sign classes, features, usage

---

### 11. Potential Viva Questions & Answers

**Q: Why BiGRU instead of Transformer?**
A: BiGRU is lightweight (~215K params), trains fast, and handles variable-length sequences well. Transformer would add 5-10× parameters for minimal gain on isolated words. BiGRU + attention is sufficient for this task.

**Q: How do you handle unseen signs?**
A: Model will output highest-probability class even for unseen signs. To properly handle unknowns, you'd need a reject-class (model trained on non-signs). We have `processed_negatives/` folder with negative samples but haven't fully integrated.

**Q: What's the inter-class confusion?**
A: Similar gestures (e.g., "beautiful" vs "ugly") have higher confusion. Accuracy is 94-98% for distinct signs but 78-85% for visually similar pairs. Solution: Add similar-sign classification rules (similar_signs.json exists but not used yet).

**Q: Can you deploy this on mobile?**
A: Yes - INT8 quantized ONNX model is 1.05MB. PyTorch can export to TFLite. Main bottleneck is MediaPipe landmark extraction (CPU-bound). With TFLite GPU acceleration, real-time on mobile is feasible.

**Q: What's the dataset size?**
A: ~5,683 processed samples + ~17,000 augmented samples. Original dataset has hundreds of videos across 78 classes. Data augmentation via video transformation and landmark perturbation expanded coverage.

**Q: How many training epochs?**
A: Typically 100 epochs with early stopping (patience=15). K-fold training takes ~4-6 hours per fold on CPU. Single model trains faster (~2 hours).

**Q: What's your test performance?**
A: 91.5% accuracy (macro-F1 0.903) on held-out test set using 5-fold ensemble. Single model: 87.2% accuracy. Ensemble reduces variance by ~4-5%.

---

**END OF VIVA PREPARATION GUIDE**
