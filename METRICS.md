PS C:\Users\Joseph\Desktop\projects\sign_to_text> python -m src.tools.generate_report_metrics

╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ISL PIPELINE CONFIGURATION SUMMARY                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

[Version & Reproducibility]
  Config Version: 2.0.0
  Config Hash: 74025fa6
  Debug Mode: False

[Feature Dimensions per Frame]
  Landmark features (raw): 126
    └─ (21 landmarks × 3 coords × 2 hands)
  Spatial relative features: 126
    └─ Face-relative: True | Distance matrix: False
  Proximity features: 1
  ─────────────────────────────────
  Total per frame: 253

[Sequence Input]
  Frames: 20
  Use velocity: True
  Input dimension: 506
    ➜ Sequence shape: (batch, 20, 506)

[Motion Detection (Resolution-Independent)]
  Enabled: False
  Frame resolution: 640×480
  Motion threshold (normalized): 0.0150 × diagonal
  Motion threshold (pixels): 12.00
  Idle confidence threshold: 0.7

[Model Architecture]
  Recurrent type: LSTM/GRU
  Hidden size: 64
  Layers: 3 (bidirectional: True)
  Dropout: 0.3
  Proximity attention: True

[Training]
  Batch size: 8
  Learning rate: 0.0003
  Epochs: 50
  Early stopping patience: 10
  Label smoothing: 0.05
  Class weighting: True (power=1.0)
    Adapter weighting: True (power=0.5, clip=0.5-3.0)

[Inference]
  Confidence threshold: 0.12
  Smoothing window: 2
  Transition hysteresis: 0.1

[Hardware]
  Device: CPU
  CPU threads: 11

╔═══════════════════════════════════════════════════════════════════════════════╗

[Dataset] HDF5 loaded: 37960 samples, 89 classes, 4 domains (augment=False)
Evaluating canonical model: models/model.pth (Trained on 90 classes)
C:\Users\Joseph\AppData\Roaming\Python\Python314\site-packages\sklearn\metrics\_classification.py:2924: UserWarning: y_pred contains classes not in y_true
  warnings.warn("y_pred contains classes not in y_true")

================================================================================
FINAL MODEL RESULTS
================================================================================

Validation Accuracy:  91.34%
Unseen Data Accuracy: 98.13%
Macro F1 Score:       96.61%
Weighted F1 Score:    97.73%
Balanced Accuracy:    98.20%

================================================================================
TOP-10 CONFUSION PAIRS (Unseen Data)
================================================================================

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\Joseph\Desktop\projects\sign_to_text\src\tools\generate_report_metrics.py", line 122, in <module>
    generate_metrics()
    ~~~~~~~~~~~~~~~~^^
  File "C:\Users\Joseph\Desktop\projects\sign_to_text\src\tools\generate_report_metrics.py", line 111, in generate_metrics
    if i != j and cm[i, j] > 0:
                  ~~^^^^^^
IndexError: index 86 is out of bounds for axis 1 with size 86
PS C:\Users\Joseph\Desktop\projects\sign_to_text>
