import os
import sys
import time
import numpy as np
import torch

# Add the project root to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.inference.onnx_ensemble import load_onnx_model, EnsembleModel
from src.utils.quantization_utils import load_model_artifact
from src.core.config import get_config

def benchmark_latency_pt(model, model_name, sequence, iterations=500):
    print(f"\n--- Benchmarking {model_name} ---")
    for _ in range(10):
        model.infer(sequence, proximity=None)
    start_time = time.time()
    for _ in range(iterations):
        model.infer(sequence, proximity=None)
    end_time = time.time()
    avg_latency_ms = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_latency_ms
    print(f"Average Latency: {avg_latency_ms:.2f} ms per inference")
    print(f"Throughput:      {fps:.1f} inferences/second")
    return avg_latency_ms

def benchmark_latency_onnx(model, model_name, sequence, proximity, iterations=500):
    print(f"\n--- Benchmarking {model_name} ---")
    if getattr(model, "model_type", None) != "onnx":
        print("ONNX model is missing. Falling back to PyTorch timing...")
        return benchmark_latency_pt(model, model_name, sequence, iterations)
        
    session = model.model.session
    inputs = {"input_seq": sequence, "proximity": proximity}
    for _ in range(10):
        session.run(None, inputs)
    start_time = time.time()
    for _ in range(iterations):
        session.run(None, inputs)
    end_time = time.time()
    avg_latency_ms = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_latency_ms
    print(f"Average Latency: {avg_latency_ms:.2f} ms per inference")
    print(f"Throughput:      {fps:.1f} inferences/second")
    return avg_latency_ms

def main():
    cfg = get_config()
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    
    pth_path = os.path.join(model_dir, "model.pth")
    onnx_fp32_path = os.path.join(model_dir, "model.onnx")
    onnx_int8_path = os.path.join(model_dir, "model_int8.onnx")
    
    device = "cpu"
    
    print("=========================================")
    print("   Inference Latency & Precision Test")
    print("=========================================\n")
    print("Loading models...")
    
    try:
        pt_model_raw, _, _, _, _ = load_model_artifact(pth_path, map_location=device)
        pt_model = EnsembleModel(pt_model_raw, model_type="pytorch", name="PyTorch_FP32")
        onnx_fp32_model = load_onnx_model(onnx_fp32_path, pytorch_fallback_path=pth_path, device=device)
        onnx_int8_model = load_onnx_model(onnx_int8_path, pytorch_fallback_path=pth_path, device=device)
    except Exception as e:
        print(f"Failed to load a model: {e}")
        return

    num_frames = cfg.preprocessing.num_frames
    feat_dim = cfg.frame_features.input_sequence_dim
    
    dummy_sequence_pt = torch.randn(1, num_frames, feat_dim, dtype=torch.float32).to(device)
    dummy_sequence_np = dummy_sequence_pt.cpu().numpy().astype(np.float32)
    # The models were exported expecting a 2D proximity tensor (Batch, Frames)
    dummy_proximity_np = np.zeros((1, num_frames), dtype=np.float32)
    
    print("\nStarting Latency Profiling (500 iterations each)...")
    benchmark_latency_pt(pt_model, "PyTorch Baseline (.pth)", dummy_sequence_pt, iterations=500)
    benchmark_latency_onnx(onnx_fp32_model, "ONNX FP32 (.onnx)", dummy_sequence_np, dummy_proximity_np, iterations=500)
    benchmark_latency_onnx(onnx_int8_model, "ONNX INT8 Quantized (_int8.onnx)", dummy_sequence_np, dummy_proximity_np, iterations=500)
    
    print("\n=========================================")
    print("   Precision & Accuracy Deviation Test")
    print("=========================================\n")
    
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    pt_logits = pt_model.infer(dummy_sequence_pt, proximity=None)
    pt_prob = softmax(pt_logits)
    
    def get_onnx_prob(model_obj):
        if getattr(model_obj, "model_type", None) != "onnx":
            # If it fell back to PyTorch, just use .infer()
            logits = model_obj.infer(dummy_sequence_pt, proximity=None)
            return softmax(logits)
        else:
            raw = model_obj.model.session.run(None, {"input_seq": dummy_sequence_np, "proximity": dummy_proximity_np})[0]
            return softmax(raw[0])
            
    onnx_fp32_prob = get_onnx_prob(onnx_fp32_model)
    onnx_int8_prob = get_onnx_prob(onnx_int8_model)
    
    pt_pred = np.argmax(pt_prob)
    onnx_fp32_pred = np.argmax(onnx_fp32_prob)
    onnx_int8_pred = np.argmax(onnx_int8_prob)
    
    print(f"PyTorch prediction class:   {pt_pred} (confidence: {pt_prob[pt_pred]:.4f})")
    print(f"ONNX FP32 prediction class: {onnx_fp32_pred} (confidence: {onnx_fp32_prob[onnx_fp32_pred]:.4f})")
    print(f"ONNX INT8 prediction class: {onnx_int8_pred} (confidence: {onnx_int8_prob[onnx_int8_pred]:.4f})")
    
    mae_fp32 = np.max(np.abs(pt_prob - onnx_fp32_prob))
    mae_int8 = np.max(np.abs(pt_prob - onnx_int8_prob))
    
    print(f"\nMax Probability Deviation from PyTorch:")
    print(f"- ONNX FP32: {mae_fp32:.6f} (Expect very close to 0)")
    print(f"- ONNX INT8: {mae_int8:.6f} (Expect minor deviation due to quantization)")

if __name__ == "__main__":
    main()
