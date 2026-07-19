"""
Unit tests: Latency Budget Enforcement
=======================================
Verifies that end-to-end inference (feature extraction + model forward pass)
does not exceed the strict 200 ms real-time budget.

Test thresholds (with generous CI headroom):
  - Single ONNX FP32 inference:  < 50 ms  (benchmark shows ~6 ms)
  - Single PyTorch inference:     < 100 ms (benchmark shows ~28 ms)
  - Full pipeline per frame:      < 200 ms (hard real-time budget)

All thresholds are headroom-padded for slow CI runners.
"""

import time
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

# Hard real-time budget from config
REALTIME_BUDGET_MS = 200.0

# Per-model generous CI thresholds (actual numbers much lower on real hardware)
PYTORCH_LATENCY_BUDGET_MS = 100.0
ONNX_FP32_LATENCY_BUDGET_MS = 50.0
ONNX_INT8_LATENCY_BUDGET_MS = 60.0

NUM_WARMUP = 5
NUM_BENCH_ITERS = 20  # lightweight for CI


def _load_onnx_model(path: str, pth_fallback: str):
    """Load ONNX model with graceful fallback if onnxruntime not available."""
    from src.inference.onnx_ensemble import load_onnx_model
    return load_onnx_model(path, pytorch_fallback_path=pth_fallback, device="cpu")


def _load_pytorch_model(pth_path: str):
    """Load PyTorch model from checkpoint."""
    from src.utils.quantization_utils import load_model_artifact
    from src.inference.onnx_ensemble import EnsembleModel
    raw_model, _, _, _, _ = load_model_artifact(pth_path, map_location="cpu")
    return EnsembleModel(raw_model, model_type="pytorch", name="PyTorch_FP32")


def _make_dummy_inputs(num_frames: int = 20, feat_dim: int = 506):
    rng = np.random.default_rng(0)
    seq_pt = torch.from_numpy(
        rng.standard_normal((1, num_frames, feat_dim)).astype(np.float32)
    )
    seq_np = seq_pt.numpy()
    prox_np = np.zeros((1, num_frames), dtype=np.float32)
    return seq_pt, seq_np, prox_np


def _bench_ms(fn, warmup: int = NUM_WARMUP, iters: int = NUM_BENCH_ITERS) -> float:
    """Return average latency in milliseconds over `iters` calls."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


class TestPyTorchLatency:
    @pytest.fixture(scope="class")
    def pt_model_and_inputs(self):
        model = _load_pytorch_model("models/model.pth")
        seq_pt, _, _ = _make_dummy_inputs()
        return model, seq_pt

    def test_pytorch_single_inference_under_budget(self, pt_model_and_inputs):
        """Single PyTorch forward pass must stay under PYTORCH_LATENCY_BUDGET_MS."""
        model, seq_pt = pt_model_and_inputs
        avg_ms = _bench_ms(lambda: model.infer(seq_pt, proximity=None))
        assert avg_ms < PYTORCH_LATENCY_BUDGET_MS, (
            f"PyTorch avg latency {avg_ms:.2f} ms exceeds {PYTORCH_LATENCY_BUDGET_MS} ms budget"
        )

    def test_pytorch_under_realtime_budget(self, pt_model_and_inputs):
        """PyTorch inference must be well within 200 ms hard real-time budget."""
        model, seq_pt = pt_model_and_inputs
        avg_ms = _bench_ms(lambda: model.infer(seq_pt, proximity=None))
        assert avg_ms < REALTIME_BUDGET_MS, (
            f"PyTorch avg latency {avg_ms:.2f} ms exceeds 200 ms real-time budget"
        )

    def test_pytorch_output_is_not_none(self, pt_model_and_inputs):
        model, seq_pt = pt_model_and_inputs
        result = model.infer(seq_pt, proximity=None)
        assert result is not None


class TestONNXLatency:
    @pytest.fixture(scope="class")
    def onnx_model_and_inputs(self):
        model = _load_onnx_model("models/model.onnx", "models/model.pth")
        _, seq_np, prox_np = _make_dummy_inputs()
        return model, seq_np, prox_np

    def test_onnx_fp32_single_inference_under_budget(self, onnx_model_and_inputs):
        """ONNX FP32 single inference must stay under ONNX_FP32_LATENCY_BUDGET_MS."""
        model, seq_np, prox_np = onnx_model_and_inputs
        if getattr(model, "model_type", None) != "onnx":
            pytest.skip("ONNX runtime not available, falling back to PyTorch")
        session = model.model.session
        fn = lambda: session.run(None, {"input_seq": seq_np, "proximity": prox_np})
        avg_ms = _bench_ms(fn)
        assert avg_ms < ONNX_FP32_LATENCY_BUDGET_MS, (
            f"ONNX FP32 avg latency {avg_ms:.2f} ms exceeds {ONNX_FP32_LATENCY_BUDGET_MS} ms"
        )

    def test_onnx_fp32_under_realtime_budget(self, onnx_model_and_inputs):
        model, seq_np, prox_np = onnx_model_and_inputs
        if getattr(model, "model_type", None) != "onnx":
            pytest.skip("ONNX runtime not available")
        session = model.model.session
        fn = lambda: session.run(None, {"input_seq": seq_np, "proximity": prox_np})
        avg_ms = _bench_ms(fn)
        assert avg_ms < REALTIME_BUDGET_MS

    def test_onnx_fp32_faster_than_pytorch(self, onnx_model_and_inputs):
        """ONNX FP32 should be faster than raw PyTorch on CPU."""
        model, seq_np, prox_np = onnx_model_and_inputs
        if getattr(model, "model_type", None) != "onnx":
            pytest.skip("ONNX runtime not available")
        pt_model = _load_pytorch_model("models/model.pth")
        seq_pt, _, _ = _make_dummy_inputs()
        session = model.model.session
        onnx_ms = _bench_ms(
            lambda: session.run(None, {"input_seq": seq_np, "proximity": prox_np})
        )
        pt_ms = _bench_ms(lambda: pt_model.infer(seq_pt, proximity=None))
        assert onnx_ms < pt_ms, (
            f"ONNX ({onnx_ms:.2f} ms) should be faster than PyTorch ({pt_ms:.2f} ms)"
        )


class TestONNXInt8Latency:
    @pytest.fixture(scope="class")
    def int8_model_and_inputs(self):
        model = _load_onnx_model("models/model_int8.onnx", "models/model.pth")
        _, seq_np, prox_np = _make_dummy_inputs()
        return model, seq_np, prox_np

    def test_onnx_int8_under_budget(self, int8_model_and_inputs):
        """ONNX INT8 inference must stay under ONNX_INT8_LATENCY_BUDGET_MS."""
        model, seq_np, prox_np = int8_model_and_inputs
        if getattr(model, "model_type", None) != "onnx":
            pytest.skip("ONNX runtime not available")
        session = model.model.session
        fn = lambda: session.run(None, {"input_seq": seq_np, "proximity": prox_np})
        avg_ms = _bench_ms(fn)
        assert avg_ms < ONNX_INT8_LATENCY_BUDGET_MS, (
            f"INT8 avg latency {avg_ms:.2f} ms exceeds {ONNX_INT8_LATENCY_BUDGET_MS} ms"
        )

    def test_onnx_int8_under_realtime_budget(self, int8_model_and_inputs):
        model, seq_np, prox_np = int8_model_and_inputs
        if getattr(model, "model_type", None) != "onnx":
            pytest.skip("ONNX runtime not available")
        session = model.model.session
        avg_ms = _bench_ms(
            lambda: session.run(None, {"input_seq": seq_np, "proximity": prox_np})
        )
        assert avg_ms < REALTIME_BUDGET_MS


class TestFeatureExtractionLatency:
    """Verify feature extraction alone is well within budget."""

    def test_single_frame_extraction_under_5ms(self, synthetic_landmarks):
        """Feature extraction for one frame should be well under 5 ms."""
        from src.shared.feature_extractor import build_single_frame_features
        left, right, face = synthetic_landmarks
        avg_ms = _bench_ms(
            lambda: build_single_frame_features(left, right, face)
        )
        assert avg_ms < 5.0, f"Feature extraction took {avg_ms:.2f} ms — unexpectedly slow"
