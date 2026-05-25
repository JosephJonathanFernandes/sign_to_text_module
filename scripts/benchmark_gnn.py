"""
Benchmark GNN vs no-GNN for synthetic and webcam modes.

Usage:
  python -u scripts/benchmark_gnn.py --mode synthetic --use_gnn 1 --iters 200
  python -u scripts/benchmark_gnn.py --mode webcam --use_gnn 1 --duration 30

Outputs JSON summary to stdout.
"""
import argparse
import os
import sys
import time
import statistics
import json

# Ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch

try:
    import psutil
except Exception:
    psutil = None

import config as config_module
from config import reset_config, get_config

from preprocess import (
    create_landmarker,
    create_face_landmarker,
    extract_landmarks_with_face_relative,
    _normalize_landmarks,
    _add_velocity,
)


def setup_config(use_gnn: bool):
    # Reset and inject custom config instance before importing model
    reset_config()
    cfg = config_module.Config()
    cfg.arch_improvements.use_gnn = bool(use_gnn)
    # assign as singleton
    config_module._config_instance = cfg
    # Now import model
    global SignLanguageGRU
    from model import SignLanguageGRU
    return cfg


def synthetic_benchmark(cfg, iters=200, batch=1):
    from model import SignLanguageGRU
    num_classes = 10
    model = SignLanguageGRU(num_classes=num_classes)
    model.eval()

    B = batch
    T = cfg.preprocessing.num_frames
    INPUT_SIZE = cfg.frame_features.input_sequence_dim
    x = torch.randn(B, T, INPUT_SIZE)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    times = []
    proc = psutil.Process(os.getpid()) if psutil else None
    cpu_samples = []
    mem_samples = []

    with torch.no_grad():
        for i in range(iters):
            t0 = time.time()
            _ = model(x)
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)
            if proc:
                cpu_samples.append(proc.cpu_percent(interval=None))
                mem_samples.append(proc.memory_info().rss / (1024 * 1024))

    summary = {
        'mode': 'synthetic',
        'use_gnn': cfg.arch_improvements.use_gnn,
        'iters': iters,
        'batch': batch,
        'avg_ms': statistics.mean(times),
        'p95_ms': statistics.quantiles(times, n=100)[94],
        'min_ms': min(times),
        'max_ms': max(times),
        'fps': 1000.0 / statistics.mean(times) if statistics.mean(times) > 0 else 0,
        'cpu_mean_percent': statistics.mean(cpu_samples) if cpu_samples else None,
        'mem_mean_mb': statistics.mean(mem_samples) if mem_samples else None,
    }
    return summary


def webcam_benchmark(cfg, duration=30):
    # Create model and landmarker
    from model import SignLanguageGRU
    num_classes = 10
    model = SignLanguageGRU(num_classes=num_classes)
    model.eval()

    landmarker = create_landmarker(num_hands=cfg.landmarks.num_hands, for_webcam=True)
    face_landmarker = create_face_landmarker(for_webcam=True)

    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.preprocessing.webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.preprocessing.webcam_height)

    from collections import deque
    seq_buf = deque(maxlen=cfg.preprocessing.num_frames)

    proc = psutil.Process(os.getpid()) if psutil else None
    cpu_samples = []
    mem_samples = []
    infer_times = []
    frame_times = []

    end_time = time.time() + duration
    frames = 0
    inferences = 0

    # warm-up
    warmup_target = 5

    while time.time() < end_time:
        t_frame0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        frames += 1
        # Extract features
        feat = extract_landmarks_with_face_relative(
            frame=frame,
            landmarker=landmarker,
            face_landmarker=face_landmarker,
        )
        seq_buf.append(feat)

        if len(seq_buf) == cfg.preprocessing.num_frames:
            # prepare sequence: normalize + velocity
            import numpy as np
            seq = np.stack(list(seq_buf), axis=0)
            seq = _normalize_landmarks(seq)
            if cfg.frame_features.use_velocity:
                seq = _add_velocity(seq)
            x = torch.from_numpy(seq).unsqueeze(0)  # (1, T, feat_dim)

            # run inference
            with torch.no_grad():
                t0 = time.time()
                _ = model(x)
                t1 = time.time()
            infer_times.append((t1 - t0) * 1000.0)
            inferences += 1

        t_frame1 = time.time()
        frame_times.append((t_frame1 - t_frame0) * 1000.0)
        if proc:
            cpu_samples.append(proc.cpu_percent(interval=None))
            mem_samples.append(proc.memory_info().rss / (1024 * 1024))

    cap.release()
    landmarker.close()
    if face_landmarker is not None:
        face_landmarker.close()

    summary = {
        'mode': 'webcam',
        'use_gnn': cfg.arch_improvements.use_gnn,
        'duration_s': duration,
        'frames_captured': frames,
        'inferences': inferences,
        'avg_frame_ms': statistics.mean(frame_times) if frame_times else None,
        'p95_frame_ms': statistics.quantiles(frame_times, n=100)[94] if len(frame_times) >= 100 else None,
        'avg_infer_ms': statistics.mean(infer_times) if infer_times else None,
        'p95_infer_ms': statistics.quantiles(infer_times, n=100)[94] if len(infer_times) >= 100 else None,
        'fps_capture': frames / duration,
        'fps_infer': inferences / duration,
        'cpu_mean_percent': statistics.mean(cpu_samples) if cpu_samples else None,
        'cpu_max_percent': max(cpu_samples) if cpu_samples else None,
        'mem_mean_mb': statistics.mean(mem_samples) if mem_samples else None,
        'mem_max_mb': max(mem_samples) if mem_samples else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('synthetic','webcam'), default='synthetic')
    parser.add_argument('--use_gnn', type=int, choices=(0,1), default=1)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()

    cfg = setup_config(args.use_gnn)

    if args.mode == 'synthetic':
        out = synthetic_benchmark(cfg, iters=args.iters, batch=args.batch)
    else:
        out = webcam_benchmark(cfg, duration=args.duration)

    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
