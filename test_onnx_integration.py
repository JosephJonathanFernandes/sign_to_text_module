"""Test suite for ONNX integration components.

Validates:
- ONNX export functionality
- ONNX inference wrapper
- Mixed ensemble detection and loading
- Numeric parity between PyTorch and ONNX
- Quantization support
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from model import ISLModel


def create_dummy_model(output_path: str) -> None:
    """Create a dummy ISL model for testing."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model = ISLModel()
    torch.save(
        {"model_state_dict": model.state_dict()},
        output_path,
    )
    print(f"✓ Created dummy model: {output_path}")


def test_export_onnx() -> bool:
    """Test ONNX export functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: ONNX Export")
    print("=" * 80)

    try:
        from export_onnx import export_to_onnx, load_pytorch_model

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "model.pth")
            output_onnx = os.path.join(tmpdir, "model.onnx")

            create_dummy_model(checkpoint)

            print(f"Loading PyTorch model...")
            model, _ = load_pytorch_model(checkpoint)

            print(f"Exporting to ONNX...")
            meta = export_to_onnx(
                model,
                output_onnx,
                seq_len=20,
                feature_dim=506,
            )

            assert os.path.isfile(output_onnx), "ONNX file not created"
            assert meta["model_size_bytes"] > 0, "ONNX file is empty"

            print(f"✓ ONNX export successful")
            print(f"  Size: {meta['model_size_bytes'] / 1024 / 1024:.2f} MB")
            return True

    except Exception as e:
        print(f"✗ ONNX export failed: {str(e)}")
        return False


def test_onnx_inference() -> bool:
    """Test ONNX inference wrapper."""
    print("\n" + "=" * 80)
    print("TEST 2: ONNX Inference")
    print("=" * 80)

    try:
        from export_onnx import export_to_onnx, load_pytorch_model
        from onnx_inference import ONNXModelWrapper

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "model.pth")
            output_onnx = os.path.join(tmpdir, "model.onnx")

            create_dummy_model(checkpoint)

            print(f"Exporting model to ONNX...")
            model, _ = load_pytorch_model(checkpoint)
            export_to_onnx(model, output_onnx, seq_len=20, feature_dim=506)

            print(f"Loading ONNX wrapper...")
            wrapper = ONNXModelWrapper(output_onnx, device="cpu")

            print(f"Running inference...")
            input_seq = np.random.randn(1, 20, 506).astype(np.float32)
            proximity = np.random.randn(1, 20).astype(np.float32)

            output = wrapper(input_seq, proximity)

            assert output is not None, "Inference returned None"
            assert output.shape == (1, 62), f"Unexpected output shape: {output.shape}"
            assert np.all(np.isfinite(output)), "Output contains NaN/Inf"

            stats = wrapper.get_stats()
            assert stats["total_calls"] == 1, "Stats not updated"

            print(f"✓ ONNX inference successful")
            print(f"  Output shape: {output.shape}")
            print(f"  Stats: {stats}")
            return True

    except Exception as e:
        print(f"✗ ONNX inference failed: {str(e)}")
        return False


def test_onnx_fallback() -> bool:
    """Test automatic fallback to PyTorch."""
    print("\n" + "=" * 80)
    print("TEST 3: Fallback to PyTorch")
    print("=" * 80)

    try:
        from onnx_inference import ONNXModelWrapper

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "model.pth")

            create_dummy_model(checkpoint)

            print(f"Loading with missing ONNX path...")
            wrapper = ONNXModelWrapper(
                onnx_path="nonexistent.onnx",
                pytorch_checkpoint=checkpoint,
                fallback_to_pytorch=True,
            )

            print(f"Running inference with fallback...")
            input_seq = np.random.randn(1, 20, 506).astype(np.float32)
            proximity = np.random.randn(1, 20).astype(np.float32)

            output = wrapper(input_seq, proximity)

            assert output is not None, "Fallback inference failed"
            assert output.shape == (1, 62), f"Unexpected output shape: {output.shape}"

            stats = wrapper.get_stats()
            assert stats["fallback_count"] > 0, "Fallback not triggered"

            print(f"✓ Fallback to PyTorch successful")
            print(f"  Fallback triggered: {stats['fallback_count']} times")
            return True

    except Exception as e:
        print(f"✗ Fallback test failed: {str(e)}")
        return False


def test_quantization() -> bool:
    """Test ONNX quantization."""
    print("\n" + "=" * 80)
    print("TEST 4: ONNX Quantization")
    print("=" * 80)

    try:
        from export_onnx import export_to_onnx, load_pytorch_model
        from quantize_onnx import quantize_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "model.pth")
            onnx_fp32 = os.path.join(tmpdir, "model_fp32.onnx")
            onnx_int8 = os.path.join(tmpdir, "model_int8.onnx")

            create_dummy_model(checkpoint)

            print(f"Exporting to FP32 ONNX...")
            model, _ = load_pytorch_model(checkpoint)
            export_to_onnx(model, onnx_fp32, seq_len=20, feature_dim=506)

            print(f"Quantizing to INT8...")
            result = quantize_onnx(onnx_fp32, onnx_int8)

            assert os.path.isfile(onnx_int8), "Quantized model not created"
            assert result["int8_size_bytes"] < result["fp32_size_bytes"], "INT8 not smaller"
            assert result["reduction_percent"] > 0, "No size reduction"

            print(f"✓ Quantization successful")
            print(f"  FP32: {result['fp32_size_bytes'] / 1024 / 1024:.2f} MB")
            print(f"  INT8: {result['int8_size_bytes'] / 1024 / 1024:.2f} MB")
            print(f"  Reduction: {result['reduction_percent']:.1f}%")
            return True

    except Exception as e:
        print(f"✗ Quantization test failed: {str(e)}")
        return False


def test_mixed_ensemble() -> bool:
    """Test mixed ONNX/PyTorch ensemble loading."""
    print("\n" + "=" * 80)
    print("TEST 5: Mixed Ensemble Detection")
    print("=" * 80)

    try:
        from export_onnx import export_to_onnx, load_pytorch_model
        from onnx_ensemble import detect_and_load_models

        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble_dir = os.path.join(tmpdir, "ensemble")
            os.makedirs(ensemble_dir)

            print(f"Creating mixed ensemble...")

            for i in range(3):
                checkpoint = os.path.join(ensemble_dir, f"fold_{i}.pth")
                onnx_model = os.path.join(ensemble_dir, f"fold_{i}_fp32.onnx")

                create_dummy_model(checkpoint)

                model, _ = load_pytorch_model(checkpoint)
                export_to_onnx(model, onnx_model, seq_len=20, feature_dim=506)

            print(f"Loading ensemble...")
            models, meta = detect_and_load_models(ensemble_dir, max_models=10)

            assert len(models) == 6, f"Expected 6 models, got {len(models)}"
            assert meta["pytorch_models"] == 3, "PyTorch models not detected"
            assert meta["onnx_models"] == 3, "ONNX models not detected"

            print(f"✓ Mixed ensemble loaded successfully")
            print(f"  PyTorch: {meta['pytorch_models']}")
            print(f"  ONNX: {meta['onnx_models']}")
            print(f"  Total: {len(models)}")
            return True

    except Exception as e:
        print(f"✗ Mixed ensemble test failed: {str(e)}")
        return False


def test_numeric_parity() -> bool:
    """Test numeric parity between PyTorch and ONNX."""
    print("\n" + "=" * 80)
    print("TEST 6: Numeric Parity")
    print("=" * 80)

    try:
        from export_onnx import export_to_onnx, load_pytorch_model
        from onnx_inference import ONNXModelWrapper
        import torch.nn.functional as F

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "model.pth")
            onnx_model = os.path.join(tmpdir, "model.onnx")

            create_dummy_model(checkpoint)

            print(f"Setting up models...")
            pytorch_model, _ = load_pytorch_model(checkpoint)
            export_to_onnx(pytorch_model, onnx_model, seq_len=20, feature_dim=506)

            onnx_wrapper = ONNXModelWrapper(onnx_model, device="cpu")

            print(f"Running parity test...")
            num_samples = 10
            max_diff = 0.0
            prediction_agreement = 0

            with torch.no_grad():
                for i in range(num_samples):
                    input_seq = np.random.randn(1, 20, 506).astype(np.float32)
                    proximity = np.random.randn(1, 20).astype(np.float32)

                    pytorch_out = onnx_wrapper.infer_pytorch(
                        input_seq,
                        torch.from_numpy(proximity),
                    )
                    onnx_out = onnx_wrapper.infer_onnx(input_seq, proximity)

                    diff = np.abs(pytorch_out - onnx_out)
                    max_diff = max(max_diff, np.max(diff))

                    pytorch_pred = np.argmax(pytorch_out)
                    onnx_pred = np.argmax(onnx_out)
                    if pytorch_pred == onnx_pred:
                        prediction_agreement += 1

            prediction_agreement = (prediction_agreement / num_samples) * 100

            assert prediction_agreement >= 95, f"Agreement too low: {prediction_agreement:.1f}%"

            print(f"✓ Numeric parity test passed")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Prediction agreement: {prediction_agreement:.1f}%")
            return True

    except Exception as e:
        print(f"✗ Numeric parity test failed: {str(e)}")
        return False


def run_all_tests() -> bool:
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print("ONNX Integration Test Suite")
    print("=" * 80)

    tests = [
        ("ONNX Export", test_export_onnx),
        ("ONNX Inference", test_onnx_inference),
        ("PyTorch Fallback", test_onnx_fallback),
        ("ONNX Quantization", test_quantization),
        ("Mixed Ensemble", test_mixed_ensemble),
        ("Numeric Parity", test_numeric_parity),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} crashed: {str(e)}")
            results.append((name, False))

    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print("-" * 80)
    print(f"Total: {passed_count}/{total_count} tests passed")
    print("=" * 80)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
