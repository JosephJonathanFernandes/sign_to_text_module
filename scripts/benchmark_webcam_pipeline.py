"""Benchmark the live webcam landmark pipeline for a fixed number of frames.

This measures:
- detection frequency (how often a fresh hand detection runs)
- average MediaPipe/extraction latency on detection frames
- re-detect count (periodic + drift-triggered refreshes)

Usage:
  python -u scripts/benchmark_webcam_pipeline.py --frames 120
  python -u scripts/benchmark_webcam_pipeline.py --frames 240 --warmup 20 --camera-index 0
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import deque

import cv2


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from config import get_config
from preprocess import create_face_landmarker, create_landmarker
from webcam import (
    HAND_FORCE_REDETECT_INTERVAL,
    _calculate_hand_motion,
    _detect_hand_drift,
    _extract_frame_landmarks,
    _get_adaptive_hand_detection_interval,
)


def run_benchmark(frames: int, warmup: int, camera_index: int) -> dict:
    cfg = get_config()

    landmarker = create_landmarker(num_hands=cfg.landmarks.num_hands, for_webcam=True)
    force_landmarker = create_landmarker(
        num_hands=cfg.landmarks.num_hands,
        for_webcam=True,
        force_image_mode=True,
    )
    face_landmarker = create_face_landmarker(for_webcam=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.preprocessing.webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.preprocessing.webcam_height)

    hand_cache = {}
    face_cache = {}
    wrist_history = deque(maxlen=3)
    motion_magnitude = 0.0

    detection_calls = 0
    drift_redetects = 0
    detection_times_ms = []

    frame_idx = 0

    for _ in range(max(0, warmup)):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        frame_timestamp_ms = int(time.perf_counter() * 1000)
        hand_detect_interval = _get_adaptive_hand_detection_interval(motion_magnitude)
        _extract_frame_landmarks(
            landmarker,
            force_landmarker,
            face_landmarker,
            frame,
            hand_cache,
            face_cache,
            frame_idx,
            frame_timestamp_ms=frame_timestamp_ms,
            hand_detect_interval=hand_detect_interval,
        )
        frame_idx += 1

    detection_calls = 0
    drift_redetects = 0
    detection_times_ms = []
    measured_frames = 0
    frame_idx = 0

    try:
        while measured_frames < frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            frame_timestamp_ms = int(time.perf_counter() * 1000)
            hand_detect_interval = _get_adaptive_hand_detection_interval(motion_magnitude)

            prev_hand_frame_idx = hand_cache.get("frame_idx")
            should_detect = (
                frame_idx % max(1, int(HAND_FORCE_REDETECT_INTERVAL)) == 0
                or frame_idx % max(1, int(hand_detect_interval)) == 0
                or hand_cache.get("result") is None
            )

            t0 = time.perf_counter()
            landmarks_vec, hand_infos, _, _ = _extract_frame_landmarks(
                landmarker,
                force_landmarker,
                face_landmarker,
                frame,
                hand_cache,
                face_cache,
                frame_idx,
                frame_timestamp_ms=frame_timestamp_ms,
                hand_detect_interval=hand_detect_interval,
            )
            detection_happened = should_detect and hand_cache.get("frame_idx") == frame_idx
            if detection_happened:
                detection_calls += 1
                detection_times_ms.append((time.perf_counter() - t0) * 1000.0)

            measured_frames += 1

            wrist_points = []
            for info in hand_infos:
                landmarks = info.get("landmarks")
                if landmarks:
                    wrist_points.append((landmarks[0].x * cfg.preprocessing.webcam_width, landmarks[0].y * cfg.preprocessing.webcam_height))

            if wrist_points:
                avg_wrist = (
                    sum(point[0] for point in wrist_points) / len(wrist_points),
                    sum(point[1] for point in wrist_points) / len(wrist_points),
                )
                motion_magnitude = _calculate_hand_motion(avg_wrist, wrist_history, motion_magnitude)

            if _detect_hand_drift(hand_infos, wrist_history, frame.shape):
                drift_redetects += 1
                t1 = time.perf_counter()
                _extract_frame_landmarks(
                    landmarker,
                    force_landmarker,
                    face_landmarker,
                    frame,
                    hand_cache,
                    face_cache,
                    frame_idx,
                    frame_timestamp_ms=frame_timestamp_ms,
                    hand_detect_interval=hand_detect_interval,
                    force_detection=True,
                )
                detection_calls += 1
                detection_times_ms.append((time.perf_counter() - t1) * 1000.0)

            if wrist_points:
                wrist_history.append(avg_wrist)

            frame_idx += 1

    finally:
        cap.release()
        landmarker.close()
        force_landmarker.close()
        if face_landmarker is not None:
            face_landmarker.close()

    avg_detection_ms = statistics.mean(detection_times_ms) if detection_times_ms else 0.0
    p95_detection_ms = statistics.quantiles(detection_times_ms, n=100)[94] if len(detection_times_ms) >= 100 else None

    return {
        "frames_measured": measured_frames,
        "warmup_frames": warmup,
        "detection_calls": detection_calls,
        "detection_frequency": (detection_calls / measured_frames) if measured_frames else 0.0,
        "avg_mediapipe_ms": avg_detection_ms,
        "p95_mediapipe_ms": p95_detection_ms,
        "redetect_count": max(0, detection_calls - 1),
        "drift_redetects": drift_redetects,
        "forced_redetect_interval": HAND_FORCE_REDETECT_INTERVAL,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the live webcam landmark pipeline")
    parser.add_argument("--frames", type=int, default=120, help="Number of measured frames to process")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup frames to skip from metrics")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to open")
    args = parser.parse_args()

    result = run_benchmark(frames=args.frames, warmup=args.warmup, camera_index=args.camera_index)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()