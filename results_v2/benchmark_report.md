# ONNX Inference Benchmark (20-Frame Window)

The evaluation was performed using `src/tools/benchmark_inference.py`, running on standard CPU with 500 iterations for latency profiling (after 10 warm-up runs).
All models take the standard (1, 20, 506) input manifold.

| Model Format | Execution Device | File Size | Average Latency | Throughput |
|--------------|------------------|-----------|-----------------|------------|
| PyTorch (.pth) Baseline | CPU | 4.20 MB | 22.86 ms | 43.7 FPS |
| ONNX FP32 (.onnx) | CPU (ORT) | 0.41 MB | 1.82 ms | 549.1 FPS |
| ONNX INT8 (_int8.onnx) | CPU (ORT) | 1.29 MB | 3.28 ms | 305.1 FPS |

### Regarding the 6.22 ms claim
The 6.22 ms execution latency claimed in the abstract is unsupported by the current ONNX models on this machine (they are significantly faster, at 1.82 ms and 3.28 ms respectively). It is likely that 6.22 ms refers to the **PyTorch baseline model executing on a significantly faster desktop-class processor** during earlier drafting, or an older, heavier iteration of the PyTorch model. 

The claim that INT8 dynamic quantization "compresses storage by 75%" is mathematically false when compared to the standard ONNX FP32 model. The ONNX FP32 model achieves a ~90% compression (4.20 MB to 0.41 MB) simply by stripping Python overhead and using the highly-optimized computational graph format, while dynamic INT8 actually bloats the file to 1.29 MB due to the structural overhead of storing quantization parameters (scales, zero-points, and cast operators).
