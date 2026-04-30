"""
Augmentation utilities for landmark sequence data (.npy)

Provides several realistic augmentations operating directly on
landmark sequences (shape: (num_frames=20, feature_dim)).

Usage:
    from augmentations import augment_dataset
    augment_dataset('processed', 'processed', augment_per_sample=3)

All augmentations preserve the feature layout and final shape.
"""
from __future__ import annotations

import os
import uuid
from typing import List, Optional

import numpy as np
from scipy import interpolate

from config import get_config

cfg = get_config()

# Derived sizes from configuration to keep layout consistent
LANDMARK_DIM = cfg.landmarks.landmark_dim_per_hand  # 21*3 = 63
FRAME_FEAT_DIM = cfg.frame_features.frame_features_dim
INPUT_SIZE = cfg.frame_features.input_sequence_dim
NUM_FRAMES = cfg.preprocessing.num_frames
USE_VELOCITY = cfg.frame_features.use_velocity
PROXIMITY_DIM = cfg.spatial.proximity_dim


def _is_webcam_file(filename: str) -> bool:
    """Return True only for webcam-derived samples.

    We treat filenames containing 'webcam' as webcam data.
    """
    return "webcam" in filename.lower()


def _split_pos_vel(seq: np.ndarray):
    """Split sequence into position block and optional velocity block.

    Returns (pos, vel_or_none)
    pos shape: (T, FRAME_FEAT_DIM)
    vel shape: (T, FRAME_FEAT_DIM) if present (stored as second half), else None
    """
    feat_dim = seq.shape[1]
    if USE_VELOCITY and feat_dim >= FRAME_FEAT_DIM * 2:
        pos = seq[:, :FRAME_FEAT_DIM].astype(np.float32)
        vel = seq[:, FRAME_FEAT_DIM:FRAME_FEAT_DIM * 2].astype(np.float32)
        return pos, vel
    # No explicit velocity stored
    pos = seq[:, :FRAME_FEAT_DIM].astype(np.float32)
    return pos, None


def _recompute_velocity(pos: np.ndarray) -> np.ndarray:
    """Recompute velocity (frame-to-frame delta) consistent with pos.

    velocity[0] = 0, velocity[i] = pos[i] - pos[i-1]
    """
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    return vel


def _pack_seq(pos: np.ndarray, vel: Optional[np.ndarray]) -> np.ndarray:
    """Pack pos and optional vel back into (T, INPUT_SIZE) sequence.
    Pads to INPUT_SIZE if needed.
    """
    if vel is not None:
        out = np.concatenate([pos, vel], axis=1)
    else:
        out = pos
    # If original INPUT_SIZE larger due to config changes, pad zeros
    if out.shape[1] < INPUT_SIZE:
        pad = np.zeros((out.shape[0], INPUT_SIZE - out.shape[1]), dtype=np.float32)
        out = np.concatenate([out, pad], axis=1)
    elif out.shape[1] > INPUT_SIZE:
        out = out[:, :INPUT_SIZE]
    return out.astype(np.float32)


def temporal_speed_variation(seq: np.ndarray, speed_factor: float) -> np.ndarray:
    """Simulate faster/slower gesture by time-stretching then resampling to NUM_FRAMES.

    speed_factor > 1.0 -> faster (compress time), < 1.0 -> slower (stretch time)

    We implement by first interpolating the original features over a finer
    timeline after stretching, then resampling back to exactly NUM_FRAMES.
    """
    assert seq.shape[0] == NUM_FRAMES
    pos, vel = _split_pos_vel(seq)

    # Create intermediate length proportional to speed to keep temporal detail
    intermediate_len = max(4, int(round(NUM_FRAMES * max(0.6, speed_factor))))

    # Original time points
    t_orig = np.linspace(0.0, 1.0, NUM_FRAMES)

    # Stretched time axis: simulate compression/expansion
    t_stretched = np.linspace(0.0, 1.0, intermediate_len) * (1.0 / max(1e-6, speed_factor))
    # Clamp to [0,1]
    t_stretched = np.clip(t_stretched, 0.0, 1.0)

    # Interpolate position features across time for each feature dim
    f = interpolate.interp1d(t_orig, pos, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    stretched = f(t_stretched)

    # Now resample back to NUM_FRAMES
    t_new = np.linspace(0.0, 1.0, NUM_FRAMES)
    f2 = interpolate.interp1d(np.linspace(0.0, 1.0, stretched.shape[0]), stretched, axis=0, kind='linear')
    pos_new = f2(t_new)

    # Recompute velocity for consistency
    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def spatial_noise_injection(seq: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Add small Gaussian noise to position coordinates.

    Noise is applied to the positional block only. We then recompute velocity
    to keep features consistent. sigma is relative to normalized landmark coords.
    """
    pos, vel = _split_pos_vel(seq)
    noise = np.random.randn(*pos.shape).astype(np.float32) * float(sigma)
    pos_noisy = pos + noise
    vel_new = _recompute_velocity(pos_noisy) if vel is not None else None
    return _pack_seq(pos_noisy, vel_new)


def random_translation(seq: np.ndarray, tx: Optional[float] = None, ty: Optional[float] = None) -> np.ndarray:
    """Apply a small translation to x and y coordinates across all frames.

    If tx/ty not provided, pick random small offsets.
    Operates on the positional block. Velocity recomputed.
    """
    pos, vel = _split_pos_vel(seq)
    if tx is None:
        tx = np.random.uniform(-0.05, 0.05)
    if ty is None:
        ty = np.random.uniform(-0.05, 0.05)

    # x and y are every 3rd coordinate starting at offset 0 and 1 within each landmark block
    pos_t = pos.copy()
    # Determine number of landmark triples in pos
    triples = pos_t.shape[1] // 3
    # Create index arrays for x and y columns
    x_idxs = [i * 3 for i in range(triples)]
    y_idxs = [i * 3 + 1 for i in range(triples)]

    pos_t[:, x_idxs] += float(tx)
    pos_t[:, y_idxs] += float(ty)

    vel_new = _recompute_velocity(pos_t) if vel is not None else None
    return _pack_seq(pos_t, vel_new)


def random_scaling(seq: np.ndarray, scale: Optional[float] = None, center_mode: str = 'mean') -> np.ndarray:
    """Scale coordinates around a chosen center (mean or wrist) to simulate distance.

    center_mode: 'mean' uses per-frame mean across position dims; 'wrist' uses wrist landmark (index 0)
    """
    pos, vel = _split_pos_vel(seq)
    if scale is None:
        scale = np.random.uniform(0.90, 1.12)

    pos_s = pos.copy()
    T, D = pos_s.shape
    # Process per-frame
    for t in range(T):
        frame = pos_s[t]
        # reshape into (num_landmarks_total, 3) if divisible
        if D % 3 != 0:
            # Fallback: global mean
            center = frame.mean()
            pos_s[t] = (frame - center) * scale + center
            continue

        arr = frame.reshape(-1, 3)
        if center_mode == 'wrist' and arr.shape[0] >= 1:
            center_pt = arr[0].copy()
        else:
            center_pt = arr.mean(axis=0)

        arr = (arr - center_pt) * float(scale) + center_pt
        pos_s[t] = arr.flatten()

    vel_new = _recompute_velocity(pos_s) if vel is not None else None
    return _pack_seq(pos_s, vel_new)


def frame_drop_and_interpolate(seq: np.ndarray, n_drop: Optional[int] = None) -> np.ndarray:
    """Randomly drop (zero out) up to n_drop frames, then fill via linear interpolation.

    We explicitly set dropped frames to NaN then interpolate each feature across time.
    """
    pos, vel = _split_pos_vel(seq)
    T = pos.shape[0]
    if n_drop is None:
        n_drop = np.random.randint(1, min(4, max(2, T // 6)) + 1)

    drop_idx = np.random.choice(T, n_drop, replace=False)
    pos_d = pos.copy()
    pos_d[drop_idx] = np.nan

    # Interpolate NaNs per feature
    for dim in range(pos_d.shape[1]):
        col = pos_d[:, dim]
        nans = np.isnan(col)
        if nans.all():
            # If entire column NaN (unlikely), set zeros
            pos_d[:, dim] = 0.0
            continue
        if nans.any():
            valid_x = np.where(~nans)[0]
            valid_y = col[valid_x]
            f = interpolate.interp1d(valid_x, valid_y, kind='linear', bounds_error=False, fill_value='extrapolate')
            interp_vals = f(np.where(nans)[0])
            col[nans] = interp_vals
            pos_d[:, dim] = col

    vel_new = _recompute_velocity(pos_d) if vel is not None else None
    return _pack_seq(pos_d, vel_new)


def horizontal_flip(seq: np.ndarray) -> np.ndarray:
    """Flip left/right horizontally. Swaps left/right blocks and negates x coordinates.

    Assumes position block ordering used in `preprocess.py` when face-relative enabled:
        [left_raw, right_raw, left_rel, right_rel, proximity?]
    Works generically by detecting blocks using LANDMARK_DIM.
    """
    pos, vel = _split_pos_vel(seq)
    pos_f = pos.copy()
    D = pos_f.shape[1]

    # Number of full landmark blocks (each landmark block length = LANDMARK_DIM)
    # Some configs include relative blocks; we operate on blocks of LANDMARK_DIM
    blocks = D // LANDMARK_DIM
    if blocks >= 4:
        # common ordering: left_raw(0), right_raw(1), left_rel(2), right_rel(3)
        # swap 0<->1 and 2<->3
        pos_f[:, 0:LANDMARK_DIM], pos_f[:, LANDMARK_DIM:2 * LANDMARK_DIM] = (
            pos_f[:, LANDMARK_DIM:2 * LANDMARK_DIM].copy(),
            pos_f[:, 0:LANDMARK_DIM].copy(),
        )
        pos_f[:, 2 * LANDMARK_DIM:3 * LANDMARK_DIM], pos_f[:, 3 * LANDMARK_DIM:4 * LANDMARK_DIM] = (
            pos_f[:, 3 * LANDMARK_DIM:4 * LANDMARK_DIM].copy(),
            pos_f[:, 2 * LANDMARK_DIM:3 * LANDMARK_DIM].copy(),
        )
    elif blocks >= 2:
        # fallback: only raw blocks present
        pos_f[:, 0:LANDMARK_DIM], pos_f[:, LANDMARK_DIM:2 * LANDMARK_DIM] = (
            pos_f[:, LANDMARK_DIM:2 * LANDMARK_DIM].copy(),
            pos_f[:, 0:LANDMARK_DIM].copy(),
        )

    # Negate x coordinates (every 3rd starting at 0)
    triples = pos_f.shape[1] // 3
    x_idxs = [i * 3 for i in range(triples)]
    pos_f[:, x_idxs] = -pos_f[:, x_idxs]

    vel_new = _recompute_velocity(pos_f) if vel is not None else None
    return _pack_seq(pos_f, vel_new)


def simulate_new_person(seq: np.ndarray) -> np.ndarray:
    """Simulate a new person's hand proportions and posture.

    Scales each landmark triple by an independent factor and adds a small
    consistent bias to emulate different finger lengths and posture.
    """
    pos, vel = _split_pos_vel(seq)


    # Handle possible trailing non-landmark dims (e.g., proximity)
    total_dims = pos.shape[1]
    tail_dims = total_dims - PROXIMITY_DIM if PROXIMITY_DIM and total_dims >= PROXIMITY_DIM else total_dims
    # number of coords that correspond to landmark triples
    landmark_coords = (tail_dims // 3) * 3

    if landmark_coords <= 0:
        # Fallback: nothing to scale, just add small bias
        pos_new = pos.copy()
    else:
        n_landmarks = landmark_coords // 3
        scale_factors = np.random.uniform(0.85, 1.15, size=(n_landmarks,)).astype(np.float32)

        landmark_block = pos[:, :landmark_coords].copy().reshape(NUM_FRAMES, n_landmarks, 3)
        for i in range(n_landmarks):
            landmark_block[:, i, :] *= scale_factors[i]

        landmark_flat = landmark_block.reshape(NUM_FRAMES, -1)
        # reattach any tail columns (proximity etc.) unchanged
        if landmark_coords < total_dims:
            tail = pos[:, landmark_coords:]
            pos_new = np.concatenate([landmark_flat, tail], axis=1)
        else:
            pos_new = landmark_flat

    # Add slight consistent bias (like different posture)
    bias = np.random.uniform(-0.05, 0.05, size=pos_new.shape[1]).astype(np.float32)
    pos_new = pos_new + bias

    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def augment_sequence(sequence: np.ndarray, variants: int = 3) -> List[np.ndarray]:
    """Generate multiple augmented versions of a single sequence.

    The function composes random combinations of the elementary augmentations
    while keeping the output count equal to `variants`.
    """
    out = []
    rng = np.random.RandomState()

    attempts = 0
    # Define augmentation candidates with base selection probabilities
    aug_candidates = [
        (simulate_new_person, 0.7),
        (temporal_speed_variation, 0.5),
        (frame_drop_and_interpolate, 0.45),
        (spatial_noise_injection, 0.9),
        (random_translation, 0.6),
        (random_scaling, 0.6),
        (horizontal_flip, 0.15),
    ]

    while len(out) < variants and attempts < variants * 6:
        attempts += 1
        s = sequence.copy()

        # choose 2 or 3 augmentations to apply
        k = int(rng.choice([2, 3], p=[0.5, 0.5]))

        funcs = [f for f, p in aug_candidates]
        probs = np.array([p for f, p in aug_candidates], dtype=float)
        probs = probs / probs.sum()

        # sample without replacement using RNG
        try:
            idxs = rng.choice(len(funcs), size=k, replace=False, p=probs)
        except Exception:
            # fallback to uniform sample if weighted sampling fails
            idxs = rng.choice(len(funcs), size=k, replace=False)

        selected = list(idxs)
        rng.shuffle(selected)

        for idx in selected:
            fn = funcs[idx]
            if fn is temporal_speed_variation:
                factor = float(rng.uniform(0.82, 1.22))
                s = temporal_speed_variation(s, factor)
            elif fn is frame_drop_and_interpolate:
                # small chance to drop 1-3 frames
                n_drop = int(rng.randint(1, min(4, max(2, NUM_FRAMES // 6)) + 1))
                s = frame_drop_and_interpolate(s, n_drop=n_drop)
            elif fn is spatial_noise_injection:
                s = spatial_noise_injection(s, sigma=float(rng.uniform(0.004, 0.02)))
            elif fn is random_translation:
                s = random_translation(s)
            elif fn is random_scaling:
                s = random_scaling(s)
            elif fn is horizontal_flip:
                s = horizontal_flip(s)
            elif fn is simulate_new_person:
                s = simulate_new_person(s)

        # final dtype & shape check
        s = np.asarray(s, dtype=np.float32)
        if s.shape == (NUM_FRAMES, INPUT_SIZE):
            out.append(s)

    return out


def augment_dataset(input_dir: str, output_dir: Optional[str] = None, augment_per_sample: int = 3):
    """Augment all .npy files under `input_dir` class subfolders.

    Saves augmented files into same class folders under `output_dir` (defaults
    to `input_dir`). Original files are left untouched.
    """
    if output_dir is None:
        output_dir = input_dir

    for cls in sorted(os.listdir(input_dir)):
        cls_in = os.path.join(input_dir, cls)
        cls_out = os.path.join(output_dir, cls)
        if not os.path.isdir(cls_in):
            continue
        os.makedirs(cls_out, exist_ok=True)

        for fname in sorted(os.listdir(cls_in)):
            if not fname.endswith('.npy'):
                continue
            if not _is_webcam_file(fname):
                continue
            fpath = os.path.join(cls_in, fname)
            try:
                seq = np.load(fpath)
            except Exception:
                print(f"[WARN] Could not load: {fpath}")
                continue

            aug_list = augment_sequence(seq, variants=augment_per_sample)
            base, ext = os.path.splitext(fname)
            for i, aug in enumerate(aug_list):
                new_name = f"{base}_aug_{i}_{uuid.uuid4().hex[:8]}.npy"
                out_path = os.path.join(cls_out, new_name)
                np.save(out_path, aug)

        print(f"[Augment] Saved augmented samples for class '{cls}' to '{cls_out}'")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Augment processed landmark dataset')
    parser.add_argument('input_dir', nargs='?', default=cfg.paths.processed_dir)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--n', type=int, default=3, help='augmentations per sample')
    args = parser.parse_args()

    augment_dataset(args.input_dir, args.output_dir, augment_per_sample=args.n)
