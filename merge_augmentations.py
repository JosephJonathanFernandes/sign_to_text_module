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

LANDMARK_DIM = 63


def _is_webcam_file(filename: str) -> bool:
    """Return True only for webcam-derived samples."""
    return "webcam" in filename.lower()


def _is_base_webcam_file(filename: str) -> bool:
    """Return True only for original webcam samples, not generated derivatives."""
    name = filename.lower()
    return (
        "webcam" in name
        and "_aug_" not in name
        and "_merge_" not in name
        and "_mrg_" not in name
    )


def _compact_sample_stem(filename: str, max_len: int = 28) -> str:
    """Build a shorter, stable stem for output filenames."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = stem.replace("_aug_", "_").replace("_merge_", "_").replace("_mrg_", "_")
    stem = stem.replace("__", "_")
    if len(stem) > max_len:
        stem = stem[:max_len].rstrip("_")
    return stem


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


def weighted_blend_merge(a: np.ndarray, b: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
    """Blend two sequences to simulate a new signer style identity."""
    if a.shape != (NUM_FRAMES, INPUT_SIZE) or b.shape != (NUM_FRAMES, INPUT_SIZE):
        raise ValueError("Input sequences must have shape (NUM_FRAMES, INPUT_SIZE)")

    if alpha is None:
        alpha = float(np.random.uniform(0.35, 0.65))

    pos_a, vel_a = _split_pos_vel(a)
    pos_b, vel_b = _split_pos_vel(b)

    merged_pos = alpha * pos_a + (1.0 - alpha) * pos_b
    merged_vel = _recompute_velocity(merged_pos) if vel_a is not None else None
    return _pack_seq(merged_pos, merged_vel)


def hand_swap_merge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Swap left/right hand feature blocks between two signers."""
    if a.shape != (NUM_FRAMES, INPUT_SIZE) or b.shape != (NUM_FRAMES, INPUT_SIZE):
        raise ValueError("Input sequences must have shape (NUM_FRAMES, INPUT_SIZE)")

    pos_a, vel_a = _split_pos_vel(a)
    pos_b, vel_b = _split_pos_vel(b)

    merged_pos = pos_a.copy()
    total_dims = merged_pos.shape[1]

    # blocks: left_raw, right_raw, left_rel, right_rel
    l_raw = slice(0, LANDMARK_DIM)
    r_raw = slice(LANDMARK_DIM, 2 * LANDMARK_DIM)
    l_rel = slice(2 * LANDMARK_DIM, 3 * LANDMARK_DIM)
    r_rel = slice(3 * LANDMARK_DIM, 4 * LANDMARK_DIM)

    # randomly swap either left or right hand stream
    if random.random() < 0.5:
        merged_pos[:, l_raw] = pos_b[:, l_raw]
        if l_rel.stop <= total_dims:
            merged_pos[:, l_rel] = pos_b[:, l_rel]
    else:
        merged_pos[:, r_raw] = pos_b[:, r_raw]
        if r_rel.stop <= total_dims:
            merged_pos[:, r_rel] = pos_b[:, r_rel]

    merged_vel = _recompute_velocity(merged_pos) if vel_a is not None else None
    return _pack_seq(merged_pos, merged_vel)


def hybrid_merge(a: np.ndarray, b: np.ndarray, min_span: int = 3, max_span: int = 10) -> np.ndarray:
    """Apply two merge operations to create stronger signer simulation."""
    mode = random.choice(["splice_blend", "swap_splice", "swap_blend"])
    if mode == "splice_blend":
        s = frame_splice_merge(a, b, min_span=min_span, max_span=max_span)
        return weighted_blend_merge(s, b)
    if mode == "swap_splice":
        s = hand_swap_merge(a, b)
        return frame_splice_merge(s, b, min_span=min_span, max_span=max_span)
    s = hand_swap_merge(a, b)
    return weighted_blend_merge(s, b)


def merge_dataset(
    input_dir: str,
    output_dir: Optional[str] = None,
    per_sample: int = 1,
    min_span: int = 3,
    max_span: int = 10,
    mode: str = "hybrid",
):
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
            if f.endswith('.npy') and _is_base_webcam_file(f)
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

                if mode == "splice":
                    merged = frame_splice_merge(a, b, min_span=min_span, max_span=max_span)
                elif mode == "blend":
                    merged = weighted_blend_merge(a, b)
                elif mode == "hand_swap":
                    merged = hand_swap_merge(a, b)
                elif mode == "hybrid":
                    merged = hybrid_merge(a, b, min_span=min_span, max_span=max_span)
                else:
                    raise ValueError(f"Unsupported merge mode: {mode}")
                base = _compact_sample_stem(fname)
                out_name = f"{base}_{mode}_merge_{i}_{uuid.uuid4().hex[:8]}.npy"
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
    parser.add_argument('--mode', choices=['splice', 'blend', 'hand_swap', 'hybrid'], default='hybrid')
    args = parser.parse_args()

    merge_dataset(
        args.input_dir,
        args.output_dir,
        per_sample=args.n,
        min_span=args.min_span,
        max_span=args.max_span,
        mode=args.mode,
    )
