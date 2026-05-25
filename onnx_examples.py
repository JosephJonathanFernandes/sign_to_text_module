"""Complete workflow examples for ONNX integration.

Demonstrates:
1. Exporting PyTorch models to ONNX
2. Inferencing with ONNX Runtime
3. Quantizing ONNX models
4. Benchmarking all backends
5. Validating numeric parity
6. Using mixed ONNX/PyTorch ensembles
"""

import os
import subprocess
import sys


def run_command(cmd: str, description: str) -> int:
    """Run a shell command and report status."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    return subprocess.call(cmd, shell=True)


def export_single_model():
    """Export trained model.pth to ONNX."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Export Single Model to ONNX")
    print("=" * 80)

    cmd = "python export_onnx.py --checkpoint model.pth --output models/model_fp32.onnx"
    run_command(cmd, "Exporting model.pth to ONNX FP32")


def export_ensemble_models():
    """Export all K-fold ensemble models to ONNX."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Export Ensemble Models to ONNX")
    print("=" * 80)

    ensemble_dir = "ensemble"
    output_dir = "models"

    if not os.path.exists(ensemble_dir):
        print(f"Ensemble directory not found: {ensemble_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(ensemble_dir):
        if fname.endswith(".pth"):
            input_path = os.path.join(ensemble_dir, fname)
            output_name = fname.replace(".pth", "_fp32.onnx")
            output_path = os.path.join(output_dir, output_name)

            cmd = f"python export_onnx.py --checkpoint {input_path} --output {output_path}"
            run_command(cmd, f"Exporting {fname}")


def quantize_onnx_models():
    """Convert FP32 ONNX models to INT8 quantized versions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Quantize ONNX Models")
    print("=" * 80)

    models_dir = "models"

    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return

    for fname in os.listdir(models_dir):
        if fname.endswith("_fp32.onnx"):
            input_path = os.path.join(models_dir, fname)
            output_name = fname.replace("_fp32.onnx", "_int8.onnx")
            output_path = os.path.join(models_dir, output_name)

            cmd = f"python quantize_onnx.py --model {input_path} --output {output_path}"
            run_command(cmd, f"Quantizing {fname}")


def validate_onnx_parity():
    """Validate ONNX outputs match PyTorch outputs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Validate ONNX Numeric Parity")
    print("=" * 80)

    pytorch_checkpoint = "model.pth"
    onnx_model = "models/model_fp32.onnx"

    if not os.path.exists(pytorch_checkpoint):
        print(f"PyTorch checkpoint not found: {pytorch_checkpoint}")
        return

    if not os.path.exists(onnx_model):
        print(f"ONNX model not found: {onnx_model}")
        return

    cmd = (
        f"python validate_onnx.py "
        f"--pytorch-checkpoint {pytorch_checkpoint} "
        f"--onnx-model {onnx_model} "
        f"--num-samples 100 "
        f"--output validation_report.json"
    )
    run_command(cmd, "Validating ONNX vs PyTorch")


def benchmark_all_backends():
    """Benchmark all inference backends (PyTorch FP32, PyTorch Quantized, ONNX FP32, ONNX INT8)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Benchmark All Backends")
    print("=" * 80)

    pytorch_checkpoint = "model.pth"
    pytorch_quantized = "models/model_pt_quantized.pth"
    onnx_fp32 = "models/model_fp32.onnx"
    onnx_int8 = "models/model_int8.onnx"

    cmd = "python benchmark_onnx.py"

    if os.path.exists(pytorch_checkpoint):
        cmd += f" --pytorch-checkpoint {pytorch_checkpoint}"

    if os.path.exists(pytorch_quantized):
        cmd += f" --pytorch-quantized {pytorch_quantized}"

    if os.path.exists(onnx_fp32):
        cmd += f" --onnx-fp32 {onnx_fp32}"

    if os.path.exists(onnx_int8):
        cmd += f" --onnx-int8 {onnx_int8}"

    cmd += " --num-iterations 1000 --output benchmark_results.json"

    run_command(cmd, "Benchmarking all backends (1000 iterations)")


def test_onnx_inference():
    """Test ONNX inference with fallback."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Test ONNX Inference with Fallback")
    print("=" * 80)

    import numpy as np
    from onnx_inference import ONNXModelWrapper

    onnx_path = "models/model_fp32.onnx"
    pytorch_path = "model.pth"

    if not os.path.exists(onnx_path):
        print(f"ONNX model not found: {onnx_path}")
        return

    print(f"Loading ONNX model: {onnx_path}")
    wrapper = ONNXModelWrapper(
        onnx_path,
        pytorch_checkpoint=pytorch_path if os.path.exists(pytorch_path) else None,
        device="cpu",
        fallback_to_pytorch=True,
        enable_profiling=True,
    )

    print(f"Running 10 inference passes...")
    for i in range(10):
        input_seq = np.random.randn(1, 20, 506).astype(np.float32)
        proximity = np.random.randn(1, 20).astype(np.float32)

        output = wrapper(input_seq, proximity)
        pred_class = int(np.argmax(output[0]))
        confidence = float(np.max(output[0]))

        print(f"  Pass {i+1}: class={pred_class}, confidence={confidence:.4f}")

    print(f"\nProfiling stats: {wrapper.get_stats()}")


def test_mixed_ensemble():
    """Test mixed ONNX/PyTorch ensemble."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Test Mixed ONNX/PyTorch Ensemble")
    print("=" * 80)

    import numpy as np
    from onnx_ensemble import detect_and_load_models, ensemble_predict_mixed

    ensemble_dir = "ensemble"

    if not os.path.exists(ensemble_dir):
        print(f"Ensemble directory not found: {ensemble_dir}")
        return

    print(f"Loading models from {ensemble_dir}...")
    models, metadata = detect_and_load_models(ensemble_dir, max_models=3, device="cpu")

    print(f"Loaded models: {metadata}")

    if not models:
        print("No models loaded.")
        return

    print(f"Running ensemble inference...")
    input_seq = np.random.randn(20, 506).astype(np.float32)

    try:
        pred_idx, confidence, probs = ensemble_predict_mixed(models, input_seq, device="cpu")
        print(f"Prediction: class={pred_idx}, confidence={confidence:.4f}")
        print(f"Top-3 classes: {np.argsort(-probs)[:3]}")
    except Exception as e:
        print(f"Ensemble inference failed: {str(e)}")


def main():
    """Run example workflows."""
    print("=" * 80)
    print("ONNX Integration Workflow Examples")
    print("=" * 80)

    examples = [
        ("1", "Export single model", export_single_model),
        ("2", "Export ensemble models", export_ensemble_models),
        ("3", "Quantize ONNX models", quantize_onnx_models),
        ("4", "Validate ONNX parity", validate_onnx_parity),
        ("5", "Benchmark all backends", benchmark_all_backends),
        ("6", "Test ONNX inference", test_onnx_inference),
        ("7", "Test mixed ensemble", test_mixed_ensemble),
    ]

    print("\nAvailable examples:")
    for idx, name, _ in examples:
        print(f"  {idx}: {name}")

    print(f"  all: Run all examples")
    print(f"  0: Exit")

    choice = input("\nSelect example: ").strip().lower()

    if choice == "0":
        return
    elif choice == "all":
        for _, _, func in examples:
            try:
                func()
            except Exception as e:
                print(f"Error: {str(e)}")
    else:
        for idx, _, func in examples:
            if choice == idx:
                try:
                    func()
                except Exception as e:
                    print(f"Error: {str(e)}")
                return

        print("Invalid selection")


if __name__ == "__main__":
    main()
