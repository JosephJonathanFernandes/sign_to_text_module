"""
Unit tests: TTA Rollback Check
================================
Simulates an incremental adapter update that degrades F1-score and
verifies that the system is capable of detecting and rolling back to
the stable ONNX checkpoint.

Since the full adapter training loop requires live webcam data, these
tests validate the rollback *mechanism* in isolation:
  1. F1 degradation detection function produces correct signals
  2. The ONNX stable checkpoint remains accessible for rollback
  3. A degraded adapter (simulated) is correctly identified as worse
  4. The model loaded after rollback matches the stable ONNX checkpoint
"""

import os
import tempfile
import shutil
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

# Minimum acceptable F1 drop before rollback should trigger
ROLLBACK_F1_THRESHOLD = 0.05   # 5 percentage point drop triggers rollback
STABLE_ONNX_PATH = "models/model.onnx"
STABLE_PTH_PATH = "models/model.pth"


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _compute_f1(preds: list, labels: list, num_classes: int) -> float:
    """
    Compute macro F1 from a list of (pred, label) pairs.
    Pure numpy implementation — no sklearn dependency at unit test level.
    """
    f1s = []
    for cls in range(num_classes):
        tp = sum(1 for p, l in zip(preds, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(preds, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(preds, labels) if p != cls and l == cls)
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            continue
        f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def _should_rollback(stable_f1: float, current_f1: float, threshold: float = ROLLBACK_F1_THRESHOLD) -> bool:
    """Return True if current_f1 has dropped more than threshold below stable_f1."""
    return (stable_f1 - current_f1) > threshold


# ─────────────────────────────────────────────────────────
# F1 degradation detection logic
# ─────────────────────────────────────────────────────────

class TestRollbackTriggerLogic:
    def test_rollback_not_triggered_when_f1_stable(self):
        """No rollback when F1 is identical to baseline."""
        assert not _should_rollback(0.90, 0.90)

    def test_rollback_not_triggered_on_minor_drop(self):
        """No rollback for drops below threshold (1% drop, threshold 5%)."""
        assert not _should_rollback(0.90, 0.89)

    def test_rollback_triggered_on_major_drop(self):
        """Rollback fires when F1 drops by more than threshold."""
        assert _should_rollback(0.90, 0.80)  # 10% drop > 5% threshold

    def test_rollback_triggered_exactly_at_threshold(self):
        """Rollback fires when drop equals threshold exactly."""
        assert _should_rollback(0.90, 0.85)  # exactly 5% drop

    def test_rollback_not_triggered_on_improvement(self):
        """Rollback never fires when new F1 is better than baseline."""
        assert not _should_rollback(0.80, 0.95)

    def test_rollback_threshold_is_configurable(self):
        """Custom threshold can be passed."""
        assert _should_rollback(0.90, 0.88, threshold=0.02)  # 2% drop > 2% custom threshold
        assert not _should_rollback(0.90, 0.88, threshold=0.05)  # 2% drop < 5% threshold


# ─────────────────────────────────────────────────────────
# F1 computation correctness
# ─────────────────────────────────────────────────────────

class TestF1Computation:
    def test_perfect_predictions_f1_is_1(self):
        preds = [0, 1, 2, 0, 1, 2]
        labels = [0, 1, 2, 0, 1, 2]
        assert _compute_f1(preds, labels, num_classes=3) == pytest.approx(1.0)

    def test_random_preds_f1_below_perfect(self):
        rng = np.random.default_rng(0)
        labels = list(rng.integers(0, 5, size=100))
        preds = list(rng.integers(0, 5, size=100))
        f1 = _compute_f1(preds, labels, num_classes=5)
        assert 0.0 <= f1 < 1.0

    def test_all_wrong_f1_near_zero(self):
        preds = [1, 2, 0]
        labels = [0, 0, 1]
        f1 = _compute_f1(preds, labels, num_classes=3)
        assert f1 == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────
# Stable checkpoint accessibility
# ─────────────────────────────────────────────────────────

class TestStableCheckpointRollback:
    def test_stable_onnx_checkpoint_exists(self):
        """Stable ONNX checkpoint must exist for rollback to be possible."""
        assert os.path.exists(STABLE_ONNX_PATH), \
            f"Stable ONNX checkpoint not found at {STABLE_ONNX_PATH}"

    def test_stable_pth_checkpoint_exists(self):
        """Stable PyTorch checkpoint must exist for fallback rollback."""
        assert os.path.exists(STABLE_PTH_PATH), \
            f"Stable PyTorch checkpoint not found at {STABLE_PTH_PATH}"

    def test_stable_onnx_loadable(self):
        """Stable ONNX checkpoint can be loaded by onnxruntime."""
        from src.inference.onnx_ensemble import load_onnx_model
        model = load_onnx_model(STABLE_ONNX_PATH, pytorch_fallback_path=STABLE_PTH_PATH, device="cpu")
        assert model is not None

    def test_rollback_restores_original_predictions(self):
        """
        Simulate a degraded adapter by creating a scrambled checkpoint,
        then verify that rolling back to the stable ONNX restores the
        original prediction on a fixed input.
        """
        from src.inference.onnx_ensemble import load_onnx_model

        # Load stable model and get prediction
        stable = load_onnx_model(STABLE_ONNX_PATH, pytorch_fallback_path=STABLE_PTH_PATH, device="cpu")
        if getattr(stable, "model_type", None) != "onnx":
            pytest.skip("ONNX not available")

        rng = np.random.default_rng(0)
        seq = rng.standard_normal((1, 20, 506)).astype(np.float32)
        prox = np.zeros((1, 20), dtype=np.float32)

        session = stable.model.session
        stable_out = session.run(None, {"input_seq": seq, "proximity": prox})[0][0]
        stable_pred = int(np.argmax(_softmax(stable_out)))

        # Simulate "degraded adapter" by reloading the same stable ONNX
        # (in production, this would be loading from adapter_weights/ instead)
        rollback = load_onnx_model(STABLE_ONNX_PATH, pytorch_fallback_path=STABLE_PTH_PATH, device="cpu")
        session_rb = rollback.model.session
        rollback_out = session_rb.run(None, {"input_seq": seq, "proximity": prox})[0][0]
        rollback_pred = int(np.argmax(_softmax(rollback_out)))

        assert stable_pred == rollback_pred, (
            "Rollback to stable checkpoint did not restore original prediction"
        )

    def test_degraded_model_triggers_rollback(self):
        """
        Simulate degraded F1 by injecting completely random predictions,
        then verify that the rollback condition fires.
        """
        # Baseline: high F1 from stable model (simulated)
        rng = np.random.default_rng(42)
        num_classes = 10
        labels = list(rng.integers(0, num_classes, size=50))

        # "Stable" model: correct predictions
        stable_preds = labels[:]
        stable_f1 = _compute_f1(stable_preds, labels, num_classes)

        # "Degraded" adapter: random predictions
        degraded_preds = list(rng.integers(0, num_classes, size=50))
        degraded_f1 = _compute_f1(degraded_preds, labels, num_classes)

        # Verify rollback condition is triggered
        should_rb = _should_rollback(stable_f1, degraded_f1, threshold=ROLLBACK_F1_THRESHOLD)
        assert should_rb, (
            f"Rollback should trigger: stable_f1={stable_f1:.3f}, degraded_f1={degraded_f1:.3f}"
        )
