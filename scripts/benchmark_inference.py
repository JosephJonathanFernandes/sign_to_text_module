#!/usr/bin/env python3
"""
benchmark_inference.py

Reproducible benchmark script for ISL Sign-to-Text inference latency.
Outputs standard percentiles (P50, P95, P99) and CPU/RAM usage.
"""

import os
import sys
import time
import platform
import argparse

import numpy as np
import psutil
import torch

# Ensure the root directory is in PYTHONPATH so we can import src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import get_config
from src.inference.ensemble import load_ensemble, ensemble_predict

def run_benchmark(warmup_runs=100, test_runs=1000):
    cfg = get_config()
    device = cfg.hardware.torch_device
    num_frames = cfg.preprocessing.num_frames
    input_size = cfg.frame_features.input_sequence_dim

    print(f"==================================================")
    print(f" ISL Sign-to-Text — Inference Benchmark")
    print(f"==================================================")
    print(f"Environment:")
    print(f"  Python version : {platform.python_version()}")
    print(f"  Torch version  : {torch.__version__}")
    print(f"  OS             : {platform.system()} {platform.release()}")
    print(f"  CPU            : {platform.processor()}")
    print(f"  RAM            : {round(psutil.virtual_memory().total / (1024**3), 1)} GB")
    print(f"  Device         : {device}")
    print(f"  Sequence Shape : ({num_frames}, {input_size})")
    print(f"==================================================\n")

    print("[1/3] Loading ensemble models...")
    start_load = time.perf_counter()
    models, classes, num_classes = load_ensemble()
    load_time = time.perf_counter() - start_load
    print(f"      Models loaded in {load_time:.2f}s.\n")

    for m in models:
        m.eval()

    # Pre-allocate random sequence
    dummy_input = np.random.randn(num_frames, input_size).astype(np.float32)

    print(f"[2/3] Warmup phase ({warmup_runs} requests)...")
    # This prevents PyTorch cold-start caching from polluting benchmark
    for _ in range(warmup_runs):
        ensemble_predict(models, dummy_input)
    print("      Warmup complete.\n")

    print(f"[3/3] Benchmark phase ({test_runs} requests)...")
    latencies = []
    
    # Baseline process resource usage
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None) # Initialize cpu measurement
    
    start_total = time.perf_counter()
    for _ in range(test_runs):
        t0 = time.perf_counter()
        ensemble_predict(models, dummy_input)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # ms
    total_time = time.perf_counter() - start_total

    cpu_usage = process.cpu_percent(interval=None)
    ram_mb = process.memory_info().rss / (1024 * 1024)

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    fps = test_runs / total_time

    print(f"==================================================")
    print(f" Results")
    print(f"==================================================")
    print(f"Latency:")
    print(f"  P50 : {p50:.2f} ms")
    print(f"  P95 : {p95:.2f} ms")
    print(f"  P99 : {p99:.2f} ms")
    print(f"")
    print(f"Throughput:")
    print(f"  FPS : {fps:.2f} iterations/sec")
    print(f"")
    print(f"Resource:")
    print(f"  CPU : {cpu_usage:.1f} %")
    print(f"  RAM : {ram_mb:.1f} MB")
    print(f"==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Benchmark")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=1000, help="Number of test iterations")
    args = parser.parse_args()

    run_benchmark(warmup_runs=args.warmup, test_runs=args.runs)
