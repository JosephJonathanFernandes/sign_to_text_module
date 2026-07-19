"""
Unit tests: Model Quantization Parity
======================================
Compares output logits of the PyTorch FP32 baseline against the
ONNX INT8 dynamic-quantized model.

Acceptance criteria (derived from benchmark results):
  - Max probability deviation: < 0.05 on real sign sequences
  - Top-1 prediction must agree on at least 90% of test sequences
  - INT8 model file size must be ≥ 60% smaller than FP32 model
    (benchmark showed 75% compression: 4.20 MB → 1.05 MB)

Note: On random noise inputs, deviation can be higher (as seen in
benchmark), so tests use structured synthetic sign-like sequences.
"""

import os
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

# Acceptable max probability deviation on structured inputs
MAX_PROB_DEVIATION = 0.05
# Minimum top-1 agreement rate across multiple inputs
MIN_TOP1_AGREEMENT_RATE = 0.90
# Minimum compression ratio (INT8 must be < FP32 × this factor in size)
MAX_INT8_SIZE_RATIO = 0.40  # INT8 should be < 40% of FP32 size


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _make_structured_sequences(n: int = 20, num_frames: int = 20, feat_dim: int = 506):
    """
    Generate structured (non-random) sequences that resemble real sign data:
    slowly-varying landmark trajectories rather than white noise.
    """
    rng = np.random.default_rng(42)
    seqs = []
    for _ in range(n):
        base = rng.standard_normal((1, feat_dim)).astype(np.float32)
        noise = rng.standard_normal((num_frames, feat_dim)).astype(np.float32) * 0.05
        seq = np.tile(base, (num_frames, 1)) + noise
        seqs.append(seq.astype(np.float32))
    return seqs


@pytest.fixture(scope="module")
def models_and_inputs():
    """Load both PyTorch and ONNX INT8 models with structured test sequences."""
    from src.utils.quantization_utils import load_model_artifact
    from src.inference.onnx_ensemble import EnsembleModel, load_onnx_model

    pt_raw, _, _, _, _ = load_model_artifact("models/model.pth", map_location="cpu")
    pt_model = EnsembleModel(pt_raw, model_type="pytorch", name="PyTorch_FP32")
    int8_model = load_onnx_model("models/model_int8.onnx", pytorch_fallback_path="models/model.pth", device="cpu")

    sequences = _make_structured_sequences(n=20)
    return pt_model, int8_model, sequences


class TestQuantizationParity:
    def test_top1_agreement_rate(self, models_and_inputs):
        """Top-1 class must agree between PyTorch and INT8 on ≥ 90% of inputs."""
        pt_model, int8_model, sequences = models_and_inputs
        if getattr(int8_model, "model_type", None) != "onnx":
            pytest.skip("ONNX INT8 not available")

        session = int8_model.model.session
        agreements = 0
        for seq in sequences:
            seq_pt = torch.from_numpy(seq[np.newaxis])  # (1, 20, 506)
            seq_np = seq[np.newaxis]
            prox_np = np.zeros((1, seq.shape[0]), dtype=np.float32)

            pt_logits = np.array(pt_model.infer(seq_pt, proximity=None))
            int8_raw = session.run(None, {"input_seq": seq_np, "proximity": prox_np})[0][0]

            pt_pred = int(np.argmax(_softmax(pt_logits)))
            int8_pred = int(np.argmax(_softmax(int8_raw)))
            if pt_pred == int8_pred:
                agreements += 1

        rate = agreements / len(sequences)
        assert rate >= MIN_TOP1_AGREEMENT_RATE, (
            f"Top-1 agreement rate {rate:.2%} < {MIN_TOP1_AGREEMENT_RATE:.0%} threshold"
        )

    def test_max_prob_deviation_on_structured_inputs(self, models_and_inputs):
        """Max prob deviation between FP32 and INT8 must be < 0.05 on structured seqs."""
        pt_model, int8_model, sequences = models_and_inputs
        if getattr(int8_model, "model_type", None) != "onnx":
            pytest.skip("ONNX INT8 not available")

        session = int8_model.model.session
        max_dev = 0.0
        for seq in sequences:
            seq_pt = torch.from_numpy(seq[np.newaxis])
            seq_np = seq[np.newaxis]
            prox_np = np.zeros((1, seq.shape[0]), dtype=np.float32)

            pt_prob = _softmax(np.array(pt_model.infer(seq_pt, proximity=None)))
            int8_prob = _softmax(
                session.run(None, {"input_seq": seq_np, "proximity": prox_np})[0][0]
            )
            max_dev = max(max_dev, float(np.max(np.abs(pt_prob - int8_prob))))

        assert max_dev < MAX_PROB_DEVIATION, (
            f"Max prob deviation {max_dev:.4f} exceeds {MAX_PROB_DEVIATION} threshold"
        )

    def test_int8_model_is_smaller_than_fp32(self):
        """INT8 model must be significantly smaller than FP32 (≥ 60% compression)."""
        fp32_size = os.path.getsize("models/model.onnx")
        int8_size = os.path.getsize("models/model_int8.onnx")
        ratio = int8_size / fp32_size
        assert ratio < MAX_INT8_SIZE_RATIO, (
            f"INT8/FP32 size ratio {ratio:.2%} not below {MAX_INT8_SIZE_RATIO:.0%} — "
            f"INT8={int8_size/1024:.0f}KB, FP32={fp32_size/1024:.0f}KB"
        )

    def test_int8_model_file_exists(self):
        assert os.path.exists("models/model_int8.onnx"), \
            "model_int8.onnx not found — run quantization export first"

    def test_fp32_model_file_exists(self):
        assert os.path.exists("models/model.onnx"), \
            "model.onnx not found"

    def test_both_models_produce_301_class_output(self, models_and_inputs):
        """Both models must output logits for exactly 301 classes (300 signs + clean)."""
        pt_model, int8_model, sequences = models_and_inputs
        seq = sequences[0]
        seq_pt = torch.from_numpy(seq[np.newaxis])
        seq_np = seq[np.newaxis]
        prox_np = np.zeros((1, seq.shape[0]), dtype=np.float32)

        pt_out = np.array(pt_model.infer(seq_pt, proximity=None))
        assert pt_out.shape[-1] == 301, f"PyTorch output has {pt_out.shape[-1]} classes, expected 301"

        if getattr(int8_model, "model_type", None) == "onnx":
            session = int8_model.model.session
            int8_out = session.run(None, {"input_seq": seq_np, "proximity": prox_np})[0][0]
            assert int8_out.shape[-1] == 301, f"INT8 output has {int8_out.shape[-1]} classes, expected 301"
