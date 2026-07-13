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

from src.preprocessing.augmentations import (
    _split_pos_vel,
    _recompute_velocity,
    _pack_seq,
    NUM_FRAMES,
    INPUT_SIZE,
    spatial_noise_injection,
)

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


def crossfade_splice_merge(
    a: np.ndarray,
    b: np.ndarray,
    min_span: int = 3,
    max_span: int = 10,
    ramp_frames: int = 2,
) -> np.ndarray:
    """Splice with soft boundary ramps to reduce hard transition artifacts."""
    if a.shape != (NUM_FRAMES, INPUT_SIZE) or b.shape != (NUM_FRAMES, INPUT_SIZE):
        raise ValueError("Input sequences must have shape (NUM_FRAMES, INPUT_SIZE)")

    pos_a, vel_a = _split_pos_vel(a)
    pos_b, vel_b = _split_pos_vel(b)

    L = random.randint(min_span, min(max_span, NUM_FRAMES - 1))
    start = random.randint(0, NUM_FRAMES - L)
    end = start + L

    merged_pos = pos_a.copy()
    host_seg = pos_a[start:end]
    peer_seg = pos_b[start:end]

    ramp = max(1, min(ramp_frames, L // 2))
    w = np.ones((L,), dtype=np.float32)
    if 2 * ramp < L:
        w[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=True, dtype=np.float32)
        w[-ramp:] = np.linspace(1.0, 0.0, ramp, endpoint=True, dtype=np.float32)
    else:
        mid = L // 2
        if mid > 0:
            w[:mid] = np.linspace(0.0, 1.0, mid, endpoint=False, dtype=np.float32)
            w[mid:] = np.linspace(1.0, 0.0, L - mid, endpoint=True, dtype=np.float32)

    merged_pos[start:end] = (1.0 - w[:, None]) * host_seg + w[:, None] * peer_seg
    merged_vel = _recompute_velocity(merged_pos) if vel_a is not None else None
    return _pack_seq(merged_pos, merged_vel)


def multi_splice_merge(
    a: np.ndarray,
    b: np.ndarray,
    min_span: int = 3,
    max_span: int = 8,
    min_segments: int = 2,
    max_segments: int = 3,
) -> np.ndarray:
    """Splice 2-3 non-overlapping peer segments into the host sequence."""
    if a.shape != (NUM_FRAMES, INPUT_SIZE) or b.shape != (NUM_FRAMES, INPUT_SIZE):
        raise ValueError("Input sequences must have shape (NUM_FRAMES, INPUT_SIZE)")

    pos_a, vel_a = _split_pos_vel(a)
    pos_b, vel_b = _split_pos_vel(b)

    merged_pos = pos_a.copy()
    used = np.zeros(NUM_FRAMES, dtype=bool)
    n_segments = random.randint(min_segments, max_segments)
    inserted = 0

    for _ in range(n_segments):
        placed = False
        for _attempt in range(30):
            L = random.randint(min_span, min(max_span, NUM_FRAMES - 1))
            start = random.randint(0, NUM_FRAMES - L)
            end = start + L
            if used[start:end].any():
                continue
            merged_pos[start:end] = pos_b[start:end]
            used[start:end] = True
            inserted += 1
            placed = True
            break
        if not placed:
            continue

    if inserted == 0:
        return frame_splice_merge(a, b, min_span=min_span, max_span=max_span)

    merged_vel = _recompute_velocity(merged_pos) if vel_a is not None else None
    return _pack_seq(merged_pos, merged_vel)


def _timewarp_segment_to_len(seg: np.ndarray, speed_factor: float, out_len: int) -> np.ndarray:
    """Resample a segment to out_len after speed scaling on its internal time axis."""
    if seg.shape[0] <= 1:
        return np.repeat(seg, out_len, axis=0)

    in_len = seg.shape[0]
    src_t = np.linspace(0.0, 1.0, in_len)
    warp_t = np.linspace(0.0, 1.0, out_len) * (1.0 / max(1e-6, speed_factor))
    warp_t = np.clip(warp_t, 0.0, 1.0)

    out = np.zeros((out_len, seg.shape[1]), dtype=np.float32)
    for d in range(seg.shape[1]):
        out[:, d] = np.interp(warp_t, src_t, seg[:, d])
    return out


def tempo_aligned_splice_merge(
    a: np.ndarray,
    b: np.ndarray,
    min_span: int = 3,
    max_span: int = 10,
) -> np.ndarray:
    """Time-warp peer segment before splicing to better match host motion speed."""
    if a.shape != (NUM_FRAMES, INPUT_SIZE) or b.shape != (NUM_FRAMES, INPUT_SIZE):
        raise ValueError("Input sequences must have shape (NUM_FRAMES, INPUT_SIZE)")

    pos_a, vel_a = _split_pos_vel(a)
    pos_b, vel_b = _split_pos_vel(b)

    L = random.randint(min_span, min(max_span, NUM_FRAMES - 1))
    start = random.randint(0, NUM_FRAMES - L)
    end = start + L

    host_seg = pos_a[start:end]
    peer_seg = pos_b[start:end]

    host_speed = float(np.linalg.norm(np.diff(host_seg, axis=0), axis=1).mean()) if L > 1 else 0.0
    peer_speed = float(np.linalg.norm(np.diff(peer_seg, axis=0), axis=1).mean()) if L > 1 else 0.0
    if peer_speed < 1e-6:
        speed_factor = 1.0
    else:
        speed_factor = float(np.clip(host_speed / peer_speed, 0.7, 1.4))

    warped_peer = _timewarp_segment_to_len(peer_seg, speed_factor=speed_factor, out_len=L)

    merged_pos = pos_a.copy()
    merged_pos[start:end] = warped_peer

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


def proximity_only_swap_merge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Swap face-relative and proximity-tail features while preserving raw hand tracks."""
    if a.shape != (NUM_FRAMES, INPUT_SIZE) or b.shape != (NUM_FRAMES, INPUT_SIZE):
        raise ValueError("Input sequences must have shape (NUM_FRAMES, INPUT_SIZE)")

    pos_a, vel_a = _split_pos_vel(a)
    pos_b, vel_b = _split_pos_vel(b)

    merged_pos = pos_a.copy()
    total_dims = merged_pos.shape[1]

    swap_start = min(2 * LANDMARK_DIM, total_dims)
    if swap_start < total_dims:
        merged_pos[:, swap_start:total_dims] = pos_b[:, swap_start:total_dims]

    merged_vel = _recompute_velocity(merged_pos) if vel_a is not None else None
    return _pack_seq(merged_pos, merged_vel)


def left_right_cross_swap_merge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross-combine streams: left hand from peer, right hand from host."""
    if a.shape != (NUM_FRAMES, INPUT_SIZE) or b.shape != (NUM_FRAMES, INPUT_SIZE):
        raise ValueError("Input sequences must have shape (NUM_FRAMES, INPUT_SIZE)")

    pos_a, vel_a = _split_pos_vel(a)
    pos_b, vel_b = _split_pos_vel(b)

    merged_pos = pos_a.copy()
    total_dims = merged_pos.shape[1]

    l_raw = slice(0, LANDMARK_DIM)
    l_rel = slice(2 * LANDMARK_DIM, 3 * LANDMARK_DIM)

    merged_pos[:, l_raw] = pos_b[:, l_raw]
    if l_rel.stop <= total_dims:
        merged_pos[:, l_rel] = pos_b[:, l_rel]

    merged_vel = _recompute_velocity(merged_pos) if vel_a is not None else None
    return _pack_seq(merged_pos, merged_vel)


def blend_then_noise_merge(a: np.ndarray, b: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
    """Blend first, then inject controlled landmark noise to avoid oversmoothing."""
    blended = weighted_blend_merge(a, b, alpha=alpha)
    sigma = float(np.random.uniform(0.005, 0.012))
    return spatial_noise_injection(blended, sigma=sigma)


def hybrid_merge(a: np.ndarray, b: np.ndarray, min_span: int = 3, max_span: int = 10) -> np.ndarray:
    """Apply two merge operations to create stronger signer simulation."""
    mode = random.choice([
        "splice_blend",
        "swap_splice",
        "swap_blend",
        "crossfade_blend",
        "tempo_splice",
        "proximity_blend_noise",
    ])
    if mode == "splice_blend":
        s = frame_splice_merge(a, b, min_span=min_span, max_span=max_span)
        return weighted_blend_merge(s, b)
    if mode == "swap_splice":
        s = hand_swap_merge(a, b)
        return frame_splice_merge(s, b, min_span=min_span, max_span=max_span)
    if mode == "swap_blend":
        s = hand_swap_merge(a, b)
        return weighted_blend_merge(s, b)
    if mode == "crossfade_blend":
        s = crossfade_splice_merge(a, b, min_span=min_span, max_span=max_span)
        return weighted_blend_merge(s, b)
    if mode == "tempo_splice":
        return tempo_aligned_splice_merge(a, b, min_span=min_span, max_span=max_span)
    s = proximity_only_swap_merge(a, b)
    return blend_then_noise_merge(s, b)


def merge_dataset(
    input_dir: str,
    output_dir: Optional[str] = None,
    per_sample: int = 1,
    min_span: int = 3,
    max_span: int = 10,
    mode: str = "hybrid",
    class_only: Optional[str] = None,
):
    """For each class folder under `input_dir`, create `per_sample` merged
    samples per original by splicing a random peer from same class.
    
    Args:
        class_only: If set, only process this specific class folder.
    """
    if output_dir is None:
        output_dir = input_dir

    classes_to_process = sorted(os.listdir(input_dir))
    if class_only:
        matching = [c for c in classes_to_process if c == class_only]
        if not matching:
            raise ValueError(f"Class '{class_only}' not found in {input_dir}. Available: {', '.join(classes_to_process[:5])}...")
        classes_to_process = matching

    for cls in classes_to_process:
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
                elif mode == "crossfade_splice":
                    merged = crossfade_splice_merge(a, b, min_span=min_span, max_span=max_span)
                elif mode == "multi_splice":
                    merged = multi_splice_merge(a, b, min_span=min_span, max_span=max_span)
                elif mode == "tempo_aligned_splice":
                    merged = tempo_aligned_splice_merge(a, b, min_span=min_span, max_span=max_span)
                elif mode == "blend":
                    merged = weighted_blend_merge(a, b)
                elif mode == "blend_then_noise":
                    merged = blend_then_noise_merge(a, b)
                elif mode == "hand_swap":
                    merged = hand_swap_merge(a, b)
                elif mode == "proximity_only_swap":
                    merged = proximity_only_swap_merge(a, b)
                elif mode == "left_right_cross_swap":
                    merged = left_right_cross_swap_merge(a, b)
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
    parser.add_argument(
        '--mode',
        choices=[
            'splice',
            'crossfade_splice',
            'multi_splice',
            'tempo_aligned_splice',
            'blend',
            'blend_then_noise',
            'hand_swap',
            'proximity_only_swap',
            'left_right_cross_swap',
            'hybrid',
        ],
        default='hybrid',
    )
    parser.add_argument('--class', dest='class_only', default=None, help='Only process this specific class')
    args = parser.parse_args()

    merge_dataset(
        args.input_dir,
        args.output_dir,
        per_sample=args.n,
        min_span=args.min_span,
        max_span=args.max_span,
        mode=args.mode,
        class_only=args.class_only,
    )
