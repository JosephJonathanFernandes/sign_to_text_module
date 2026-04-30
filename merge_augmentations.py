"""Frame-splice merge augmentation for landmark sequences.

Creates merged samples by splicing contiguous frame ranges from one
sequence into another (same class). This preserves real gesture fragments
and is more realistic than global mixing.

Usage:
    python merge_augmentations.py processed --output_dir processed_merge --n 2
"""
from __future__ import annotations

import os
import uuid
import random
from typing import Optional

import numpy as np

from augmentations import _split_pos_vel, _recompute_velocity, _pack_seq, NUM_FRAMES, INPUT_SIZE


def _is_webcam_file(filename: str) -> bool:
    """Return True only for webcam-derived samples."""
    return "webcam" in filename.lower()


def frame_splice_merge(a: np.ndarray, b: np.ndarray, min_span: int = 3, max_span: int = 10) -> np.ndarray:
    """Splice a contiguous range of frames from `b` into `a` and return merged seq.

    Both `a` and `b` must have shape (NUM_FRAMES, INPUT_SIZE).
    """
    if a.shape != (NUM_FRAMES, INPUT_SIZE) or b.shape != (NUM_FRAMES, INPUT_SIZE):
        raise ValueError("Input sequences must have shape (NUM_FRAMES, INPUT_SIZE)")

    pos_a, vel_a = _split_pos_vel(a)
    pos_b, vel_b = _split_pos_vel(b)

    # Choose splice length and start
    L = random.randint(min_span, min(max_span, NUM_FRAMES - 1))
    start = random.randint(0, NUM_FRAMES - L)

    merged_pos = pos_a.copy()
    merged_pos[start:start + L] = pos_b[start:start + L]

    merged_vel = _recompute_velocity(merged_pos) if vel_a is not None else None
    return _pack_seq(merged_pos, merged_vel)


def merge_dataset(input_dir: str, output_dir: Optional[str] = None, per_sample: int = 1, min_span: int = 3, max_span: int = 10):
    """For each class folder under `input_dir`, create `per_sample` merged
    samples per original by splicing a random peer from same class.
    """
    if output_dir is None:
        output_dir = input_dir

    for cls in sorted(os.listdir(input_dir)):
        cls_in = os.path.join(input_dir, cls)
        cls_out = os.path.join(output_dir, cls)
        if not os.path.isdir(cls_in):
            continue
        os.makedirs(cls_out, exist_ok=True)

        npy_files = [
            f for f in sorted(os.listdir(cls_in))
            if f.endswith('.npy') and _is_webcam_file(f)
        ]
        if len(npy_files) < 2:
            print(f"[Merge] Skipping class '{cls}' (need >=2 webcam samples)")
            continue

        for fname in npy_files:
            a_path = os.path.join(cls_in, fname)
            try:
                a = np.load(a_path)
            except Exception:
                print(f"[WARN] Could not load {a_path}")
                continue

            for i in range(per_sample):
                # choose random peer different from current
                bname = random.choice(npy_files)
                while bname == fname and len(npy_files) > 1:
                    bname = random.choice(npy_files)
                b_path = os.path.join(cls_in, bname)
                try:
                    b = np.load(b_path)
                except Exception:
                    print(f"[WARN] Could not load peer {b_path}")
                    continue

                merged = frame_splice_merge(a, b, min_span=min_span, max_span=max_span)
                base, _ = os.path.splitext(fname)
                out_name = f"{base}_merge_{i}_{uuid.uuid4().hex[:8]}.npy"
                out_path = os.path.join(cls_out, out_name)
                np.save(out_path, merged)

        print(f"[Merge] Saved merged samples for class '{cls}' to '{cls_out}'")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create frame-splice merged samples')
    parser.add_argument('input_dir', nargs='?', default='processed')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--n', type=int, default=1, help='merged samples per original')
    parser.add_argument('--min_span', type=int, default=3)
    parser.add_argument('--max_span', type=int, default=8)
    args = parser.parse_args()

    merge_dataset(args.input_dir, args.output_dir, per_sample=args.n, min_span=args.min_span, max_span=args.max_span)
