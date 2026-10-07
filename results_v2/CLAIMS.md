# Verification of Paper Claims

| Claim | Source File | Supporting Script/File | Verdict | Notes |
|-------|-------------|------------------------|---------|-------|
| **94.13% test accuracy** | `Paper/improved_paper_draft.md` (and Abstract) | `src/tools/evaluate_robustness.py` | **Unsupported** | 70% of evaluated samples were in the training set. True held-out accuracy is ~93.4%, but still suffers from ~31% near-duplicate leakage. |
| **98.33% rejection precision** | `Paper/improved_paper_draft.md` | `src/tools/generate_report_metrics.py` | **Relabel** | Mislabeled. 98.33% is the accuracy on the `webcam` subset. Actual reject precision is 95.0% but recall is extremely poor (1.62%). |
| **Model suppresses epenthesis via reject class** | Abstract | `src/tools/evaluate_robustness.py` | **Unsupported** | The model misclassifies 98.4% of transition/noise samples as valid signs. Epenthesis is not suppressed. |
| **ONNX INT8 dynamic quantization, compressing storage by 75%** | Abstract | `models/` & `benchmark_inference.py` | **Unsupported** | INT8 actually increases size (1.29 MB) and latency (3.28 ms). The standard ONNX FP32 model provides a true 90% compression (4.2 MB -> 0.41 MB) and runs at 1.82 ms. |
| **6.22 ms execution latency on standard CPUs** | Abstract | `src/tools/benchmark_inference.py` | **Relabel** | PyTorch takes 22.86 ms on this machine, while ONNX FP32 takes 1.82 ms. 6.22 ms is likely an older PyTorch benchmark. Replace with sub-2ms ONNX FP32 claim. |
| **Test-Time Adaptation via entropy minimization** | Abstract & `Paper/improved_literature_survey.md` | `src/inference/ensemble.py` | **Unsupported** | The code only measures entropy and logs warnings; it does not minimize entropy via test-time backpropagation/parameter updates. |
| **Shoulder-normalized features** | Abstract | `src/preprocessing/preprocess.py` | **Relabel** | Features are actually *wrist-normalized*. Shoulder landmarks were explicitly excluded to remove noise. |
| **Statistical sequence alignment** | Abstract | `DOCS/DISSERTATION_HELP_ARCHIVE.md` | **Unsupported** | No sequence alignment (CTC or DTW) is implemented. The docs list CTC as a "future enhancement" and explicitly state DTW is absent. |
| **144-dimensional spatiotemporal manifold** | Abstract | `src/training/model.py` | **Supported** | Confirmed. The 128-D Conv1D output and 16-D GNN output are correctly concatenated. |
| **Finite-difference kinematic trajectories** | Abstract | `src/preprocessing/preprocess.py` | **Supported** | Confirmed. `_add_velocity` computes exact frame-to-frame deltas. |
