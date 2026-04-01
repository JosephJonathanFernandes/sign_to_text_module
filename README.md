# ISL Sign To Text

Real-time Indian Sign Language word recognition using hand landmarks and a BiGRU-based classifier.

## Features

- Video preprocessing into landmark sequences
- Single model training and K-fold ensemble training
- Live webcam inference
- Webcam data collection for new samples
- Runtime signer validation for webcam mode:
  - Supports both one-hand and two-hand signs
  - Tries to ensure two detected hands belong to the same person
  - Shows on-screen status and bounding boxes for debugging

## Current Classes

- loud
- quiet
- happy
- sad
- beautiful
- ugly
- deaf
- blind

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Minimal packages in requirements:

- torch
- numpy
- opencv-python
- mediapipe

## Project Layout

- main.py: Main CLI entry
- preprocess.py: Video to landmark preprocessing
- dataset.py: Dataset loading utilities
- model.py: Model architecture
- train.py: Training routines
- ensemble.py: Ensemble loading/inference
- webcam.py: Live webcam prediction
- collect_data.py: Webcam sample collection
- config.py: Settings and constants

## Quick Start

### 1) Preprocess and Train

```bash
python main.py --preprocess
python main.py --train
```

### 2) Optional: K-fold Ensemble

```bash
python main.py --kfold
```

### 3) Predict From a Video

```bash
python main.py --predict path/to/video.mp4
```

### 4) Run Live Webcam

```bash
python main.py --webcam
```

Press Q or ESC to quit.

### 5) Collect New Webcam Samples

Interactive mode:

```bash
python main.py --collect
```

Direct class mode:

```bash
python main.py --collect --cls happy --n 10
```

## Webcam Signer Validation (What You Will See)

In webcam mode, the app draws:

- Person boxes (if detected)
- Hand boxes with Left/Right labels
- Assignment label per hand
- Status text:
  - Single-hand sign mode
  - Same person: YES
  - Same person: NO
  - Same person: waiting

Prediction update logic:

- One-hand sign visible: prediction is allowed
- Two-hand sign visible: prediction is allowed when pair is validated as same signer
- Invalid pair/no hand for a short period: rolling window resets

## Notes

- The same-person check is a runtime safety gate. It does not require re-recording the dataset.
- If person detection is unstable in a specific environment, use better lighting and a clean background for best results.

## Troubleshooting

- Webcam not opening:
  - Close apps using the camera
  - Check camera permissions in Windows privacy settings

- No model found warning:
  - Run training first or ensure ensemble weights exist in the ensemble folder

- Low confidence predictions:
  - Improve lighting
  - Keep hand signs centered and steady
  - Collect more balanced samples per class

## License

For academic and learning use unless a separate license is added.
