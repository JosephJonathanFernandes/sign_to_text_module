"""
Augmentation utilities for landmark sequence data (.npy)

Provides several realistic augmentations operating directly on
landmark sequences (shape: (num_frames=20, feature_dim)).

Usage:
    from src.preprocessing.augmentations import augment_dataset
    augment_dataset('processed', 'processed', augment_per_sample=3)

All augmentations preserve the feature layout and final shape.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import uuid
from typing import List, Optional

import numpy as np
from scipy import interpolate

from src.core.config import get_config

cfg = get_config()

# Derived sizes from configuration to keep layout consistent
LANDMARK_DIM = cfg.landmarks.landmark_dim_per_hand  # 21*3 = 63
FRAME_FEAT_DIM = cfg.frame_features.frame_features_dim
INPUT_SIZE = cfg.frame_features.input_sequence_dim
NUM_FRAMES = cfg.preprocessing.num_frames
USE_VELOCITY = cfg.frame_features.use_velocity
PROXIMITY_DIM = cfg.spatial.proximity_dim
DEFAULT_AUGMENT_VARIANTS = 20


_FINGER_GROUPS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def _is_webcam_file(filename: str) -> bool:
    """Return True only for webcam-derived samples.

    We treat filenames containing 'webcam' as webcam data.
    """
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
    """Build a shorter, stable stem for output filenames.

    Keeps only the base filename without extension and trims it to a safe length.
    This avoids runaway filename growth when files are regenerated.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = stem.replace("_aug_", "_").replace("_merge_", "_").replace("_mrg_", "_")
    stem = stem.replace("__", "_")
    if len(stem) > max_len:
        stem = stem[:max_len].rstrip("_")
    return stem


def _get_block_slices(total_dims: int):
    """Return slices for [left_raw, right_raw, left_rel, right_rel, tail]."""
    s0 = slice(0, LANDMARK_DIM)
    s1 = slice(LANDMARK_DIM, 2 * LANDMARK_DIM)
    s2 = slice(2 * LANDMARK_DIM, 3 * LANDMARK_DIM)
    s3 = slice(3 * LANDMARK_DIM, 4 * LANDMARK_DIM)
    tail = slice(4 * LANDMARK_DIM, total_dims)
    return s0, s1, s2, s3, tail


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


def _variant_seed(sequence: np.ndarray, variant_name: str) -> int:
    payload = np.ascontiguousarray(sequence, dtype=np.float32).tobytes() + variant_name.encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


@contextlib.contextmanager
def _temporary_numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


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


def simulate_hand_proportions(seq: np.ndarray) -> np.ndarray:
    """Simulate different hand geometry by separate left/right scaling.

    Applies different isotropic scales to left/right hand landmark blocks,
    and also to their face-relative counterparts when present.
    """
    pos, vel = _split_pos_vel(seq)
    pos_new = pos.copy()
    total_dims = pos_new.shape[1]
    l_raw, r_raw, l_rel, r_rel, tail = _get_block_slices(total_dims)

    left_scale = float(np.random.uniform(0.82, 1.18))
    right_scale = float(np.random.uniform(0.82, 1.18))

    pos_new[:, l_raw] *= left_scale
    pos_new[:, r_raw] *= right_scale

    if l_rel.stop <= total_dims and r_rel.stop <= total_dims:
        pos_new[:, l_rel] *= left_scale
        pos_new[:, r_rel] *= right_scale

    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def simulate_face_anchor_shift(seq: np.ndarray) -> np.ndarray:
    """Simulate different face geometry/posture in face-relative features.

    This only modifies relative blocks and proximity-like tail dimensions,
    keeping raw hand coordinates intact.
    """
    pos, vel = _split_pos_vel(seq)
    pos_new = pos.copy()
    total_dims = pos_new.shape[1]
    l_raw, r_raw, l_rel, r_rel, tail = _get_block_slices(total_dims)

    if l_rel.stop <= total_dims and r_rel.stop <= total_dims:
        rel_scale = float(np.random.uniform(0.9, 1.15))
        rel_bias = np.random.uniform(-0.04, 0.04, size=LANDMARK_DIM).astype(np.float32)
        pos_new[:, l_rel] = pos_new[:, l_rel] * rel_scale + rel_bias
        pos_new[:, r_rel] = pos_new[:, r_rel] * rel_scale + rel_bias

    # Slightly perturb proximity (if present as trailing dims)
    if tail.start < total_dims:
        prox_noise = np.random.uniform(-0.05, 0.05, size=(NUM_FRAMES, total_dims - tail.start)).astype(np.float32)
        pos_new[:, tail] = pos_new[:, tail] + prox_noise

    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def landmark_3d_rotation(seq: np.ndarray, axis: str = 'z', angle: Optional[float] = None) -> np.ndarray:
    """Apply 3D rotation to landmark coordinates around specified axis (x, y, or z).
    
    Simulates camera angle changes or hand rotation in 3D space.
    axis: 'x', 'y', or 'z' (random if not specified)
    angle: rotation angle in degrees (random in ±15 degrees if not specified)
    """
    pos, vel = _split_pos_vel(seq)
    
    if axis is None or not isinstance(axis, str) or axis.lower() not in ['x', 'y', 'z']:
        axis = np.random.choice(['x', 'y', 'z'])
    else:
        axis = axis.lower()
    
    if angle is None:
        angle = np.random.uniform(-15, 15)
    
    angle_rad = np.radians(float(angle))
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    total_dims = pos.shape[1]
    landmark_coords = (total_dims // 3) * 3
    
    if landmark_coords <= 0:
        pos_new = pos.copy()
    else:
        n_landmarks = landmark_coords // 3
        landmark_block = pos[:, :landmark_coords].copy().reshape(NUM_FRAMES, n_landmarks, 3)
        
        # Apply rotation matrix
        for t in range(NUM_FRAMES):
            for i in range(n_landmarks):
                x, y, z = landmark_block[t, i]
                
                if axis == 'x':
                    # Rotation around X axis
                    y_new = y * cos_a - z * sin_a
                    z_new = y * sin_a + z * cos_a
                    landmark_block[t, i] = [x, y_new, z_new]
                elif axis == 'y':
                    # Rotation around Y axis
                    x_new = x * cos_a + z * sin_a
                    z_new = -x * sin_a + z * cos_a
                    landmark_block[t, i] = [x_new, y, z_new]
                elif axis == 'z':
                    # Rotation around Z axis (most common)
                    x_new = x * cos_a - y * sin_a
                    y_new = x * sin_a + y * cos_a
                    landmark_block[t, i] = [x_new, y_new, z]
        
        landmark_flat = landmark_block.reshape(NUM_FRAMES, -1)
        if landmark_coords < total_dims:
            tail = pos[:, landmark_coords:]
            pos_new = np.concatenate([landmark_flat, tail], axis=1)
        else:
            pos_new = landmark_flat
    
    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def landmark_pixel_dropout(seq: np.ndarray, dropout_rate: Optional[float] = None) -> np.ndarray:
    """Randomly drop (zero out) individual landmarks, simulating missing detection.
    
    dropout_rate: fraction of landmarks to drop (default: random 0.05-0.15)
    """
    pos, vel = _split_pos_vel(seq)
    
    if dropout_rate is None:
        dropout_rate = np.random.uniform(0.05, 0.15)
    
    total_dims = pos.shape[1]
    landmark_coords = (total_dims // 3) * 3
    
    if landmark_coords <= 0:
        pos_new = pos.copy()
    else:
        n_landmarks = landmark_coords // 3
        num_to_drop = max(1, int(np.ceil(n_landmarks * dropout_rate)))
        drop_indices = np.random.choice(n_landmarks, num_to_drop, replace=False)
        
        landmark_block = pos[:, :landmark_coords].copy().reshape(NUM_FRAMES, n_landmarks, 3)
        for idx in drop_indices:
            landmark_block[:, idx, :] = 0.0  # Zero out the landmark
        
        landmark_flat = landmark_block.reshape(NUM_FRAMES, -1)
        if landmark_coords < total_dims:
            tail = pos[:, landmark_coords:]
            pos_new = np.concatenate([landmark_flat, tail], axis=1)
        else:
            pos_new = landmark_flat
    
    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def landmark_coarse_dropout(seq: np.ndarray, block_size: Optional[int] = None) -> np.ndarray:
    """Drop a contiguous block of landmarks (like DropBlock for sequences).
    
    Simulates occlusion or partial hand loss. Zeros out a random range of landmarks.
    block_size: number of consecutive landmarks to drop (default: random 2-6)
    """
    pos, vel = _split_pos_vel(seq)
    
    if block_size is None:
        block_size = np.random.randint(2, 7)
    
    total_dims = pos.shape[1]
    landmark_coords = (total_dims // 3) * 3
    
    if landmark_coords <= 0:
        pos_new = pos.copy()
    else:
        n_landmarks = landmark_coords // 3
        if block_size >= n_landmarks:
            # Drop all landmarks
            pos_new = pos.copy()
            pos_new[:, :landmark_coords] = 0.0
        else:
            start_idx = np.random.randint(0, n_landmarks - block_size + 1)
            end_idx = start_idx + block_size
            
            landmark_block = pos[:, :landmark_coords].copy().reshape(NUM_FRAMES, n_landmarks, 3)
            landmark_block[:, start_idx:end_idx, :] = 0.0  # Zero out block
            
            landmark_flat = landmark_block.reshape(NUM_FRAMES, -1)
            if landmark_coords < total_dims:
                tail = pos[:, landmark_coords:]
                pos_new = np.concatenate([landmark_flat, tail], axis=1)
            else:
                pos_new = landmark_flat
    
    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def landmark_fog_noise(seq: np.ndarray, intensity: Optional[float] = None) -> np.ndarray:
    """Add enhanced noise to simulate poor visibility (fog-like effect).
    
    intensity: noise standard deviation (default: random 0.03-0.08)
    """
    pos, vel = _split_pos_vel(seq)
    
    if intensity is None:
        intensity = np.random.uniform(0.03, 0.08)
    
    # Add heavier Gaussian noise
    noise = np.random.randn(*pos.shape).astype(np.float32) * float(intensity)
    pos_noisy = pos + noise
    
    vel_new = _recompute_velocity(pos_noisy) if vel is not None else None
    return _pack_seq(pos_noisy, vel_new)


def landmark_brightness_contrast(seq: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
    """Scale landmark magnitudes to simulate brightness/contrast changes.
    
    Increases or decreases the spread of landmark values around their mean.
    scale: multiplier for contrast (default: random 0.85-1.15)
    """
    pos, vel = _split_pos_vel(seq)
    
    if scale is None:
        scale = np.random.uniform(0.85, 1.15)
    
    # Compute per-frame mean
    pos_new = pos.copy()
    for t in range(NUM_FRAMES):
        frame_mean = pos_new[t].mean()
        pos_new[t] = (pos_new[t] - frame_mean) * float(scale) + frame_mean
    
    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def time_shift_with_wrap_or_pad(seq: np.ndarray, max_shift: int = 3, mode: Optional[str] = None) -> np.ndarray:
    """Shift sequence in time using wrap-around or zero-padding.

    mode:
        - 'wrap': circular shift
        - 'pad': shift with zero-fill on exposed side
        - None: randomly chooses between wrap and pad
    """
    pos, vel = _split_pos_vel(seq)
    T = pos.shape[0]
    if T <= 1:
        return seq.astype(np.float32)

    max_shift = max(1, min(int(max_shift), T - 1))
    shift = int(np.random.randint(-max_shift, max_shift + 1))
    if shift == 0:
        shift = 1

    if mode is None:
        mode = str(np.random.choice(["wrap", "pad"]))
    mode = mode.lower()

    if mode == "wrap":
        pos_new = np.roll(pos, shift=shift, axis=0)
    else:
        pos_new = np.zeros_like(pos)
        if shift > 0:
            pos_new[shift:] = pos[:-shift]
        else:
            k = -shift
            pos_new[:T - k] = pos[k:]

    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def piecewise_temporal_warp(seq: np.ndarray, n_segments: int = 3,
                            speed_min: float = 0.72, speed_max: float = 1.35) -> np.ndarray:
    """Apply piecewise speed changes over different temporal segments.

    Each segment is stretched/compressed independently, then the result is
    resampled back to NUM_FRAMES.
    """
    pos, vel = _split_pos_vel(seq)
    T = pos.shape[0]
    if T <= 3:
        return seq.astype(np.float32)

    n_segments = int(max(2, min(n_segments, max(2, T // 3))))
    candidates = np.arange(1, T - 1)
    if n_segments - 1 > candidates.size:
        cut_points = np.array([], dtype=np.int32)
    else:
        cut_points = np.sort(np.random.choice(candidates, n_segments - 1, replace=False))
    bounds = np.concatenate(([0], cut_points, [T - 1]))

    warped_parts: List[np.ndarray] = []
    for i in range(len(bounds) - 1):
        start = int(bounds[i])
        end = int(bounds[i + 1])
        seg = pos[start:end + 1]
        seg_len = seg.shape[0]
        if seg_len <= 1:
            continue

        speed = float(np.random.uniform(speed_min, speed_max))
        warped_len = max(2, int(round(seg_len / speed)))

        t_seg = np.linspace(0.0, 1.0, seg_len)
        t_warp = np.linspace(0.0, 1.0, warped_len)
        interp_fn = interpolate.interp1d(
            t_seg, seg, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        seg_warped = interp_fn(t_warp)

        if i > 0 and warped_parts:
            seg_warped = seg_warped[1:]
        warped_parts.append(seg_warped)

    if not warped_parts:
        pos_new = pos.copy()
    else:
        concat = np.concatenate(warped_parts, axis=0)
        if concat.shape[0] < 2:
            pos_new = pos.copy()
        else:
            t_concat = np.linspace(0.0, 1.0, concat.shape[0])
            t_final = np.linspace(0.0, 1.0, T)
            interp_final = interpolate.interp1d(t_concat, concat, axis=0, kind="linear")
            pos_new = interp_final(t_final)

    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def _scale_finger_group(block: np.ndarray, finger_scales: dict) -> np.ndarray:
    """Scale finger articulation per finger group in one hand block (T, 63)."""
    arr = block.reshape(block.shape[0], 21, 3).copy()
    for finger_name, idxs in _FINGER_GROUPS.items():
        root_idx = idxs[0]
        scale = float(finger_scales[finger_name])
        root = arr[:, root_idx:root_idx + 1, :]
        arr[:, idxs, :] = root + (arr[:, idxs, :] - root) * scale
    return arr.reshape(block.shape[0], -1)


def per_finger_articulation_scaling(seq: np.ndarray) -> np.ndarray:
    """Scale articulation per finger (thumb/index/middle/ring/pinky) per hand.

    Uses different factors for each finger instead of global hand scaling.
    """
    pos, vel = _split_pos_vel(seq)
    pos_new = pos.copy()
    total_dims = pos_new.shape[1]
    l_raw, r_raw, l_rel, r_rel, _ = _get_block_slices(total_dims)

    left_scales = {k: np.random.uniform(0.82, 1.22) for k in _FINGER_GROUPS}
    right_scales = {k: np.random.uniform(0.82, 1.22) for k in _FINGER_GROUPS}

    if l_raw.stop <= total_dims:
        pos_new[:, l_raw] = _scale_finger_group(pos_new[:, l_raw], left_scales)
    if r_raw.stop <= total_dims:
        pos_new[:, r_raw] = _scale_finger_group(pos_new[:, r_raw], right_scales)
    if l_rel.stop <= total_dims:
        pos_new[:, l_rel] = _scale_finger_group(pos_new[:, l_rel], left_scales)
    if r_rel.stop <= total_dims:
        pos_new[:, r_rel] = _scale_finger_group(pos_new[:, r_rel], right_scales)

    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def wrist_trajectory_drift(seq: np.ndarray, max_drift_xy: float = 0.03, max_drift_z: float = 0.01) -> np.ndarray:
    """Add smooth wrist-centered trajectory drift over time.

    A small low-frequency drift is applied to all landmarks, approximating
    subtle camera/subject movement.
    """
    pos, vel = _split_pos_vel(seq)
    pos_new = pos.copy()
    T = pos_new.shape[0]

    steps = np.random.normal(loc=0.0, scale=0.004, size=(T, 3)).astype(np.float32)
    steps[:, 2] *= 0.5
    drift = np.cumsum(steps, axis=0)

    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    for d in range(3):
        drift[:, d] = np.convolve(drift[:, d], kernel, mode="same")

    drift[:, 0] = np.clip(drift[:, 0], -max_drift_xy, max_drift_xy)
    drift[:, 1] = np.clip(drift[:, 1], -max_drift_xy, max_drift_xy)
    drift[:, 2] = np.clip(drift[:, 2], -max_drift_z, max_drift_z)

    triples = pos_new.shape[1] // 3
    if triples > 0:
        xyz = pos_new[:, :triples * 3].reshape(T, triples, 3)
        xyz += drift[:, None, :]
        pos_new[:, :triples * 3] = xyz.reshape(T, -1)

    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def _mask_landmarks_in_block(block: np.ndarray, landmark_indices: List[int], time_indices: np.ndarray) -> np.ndarray:
    """Mask selected landmarks in a hand block (T, 63) at selected timesteps."""
    arr = block.reshape(block.shape[0], 21, 3).copy()
    arr[np.asarray(time_indices, dtype=np.int32)[:, None], np.asarray(landmark_indices, dtype=np.int32), :] = 0.0
    return arr.reshape(block.shape[0], -1)


def landmark_confidence_masking(seq: np.ndarray) -> np.ndarray:
    """Apply structured missingness patterns to simulate confidence failures.

    Patterns include finger-level temporal dropouts and hand-level temporal
    masking with contiguous windows.
    """
    pos, vel = _split_pos_vel(seq)
    pos_new = pos.copy()
    total_dims = pos_new.shape[1]
    T = pos_new.shape[0]
    l_raw, r_raw, l_rel, r_rel, _ = _get_block_slices(total_dims)

    if T <= 2:
        vel_new = _recompute_velocity(pos_new) if vel is not None else None
        return _pack_seq(pos_new, vel_new)

    pattern = str(np.random.choice(["finger_window", "hand_window", "staggered_fingers"]))
    win_len = int(np.random.randint(3, min(8, T) + 1))
    start = int(np.random.randint(0, max(1, T - win_len + 1)))
    time_window = np.arange(start, start + win_len, dtype=np.int32)

    hand_blocks = []
    if l_raw.stop <= total_dims:
        hand_blocks.append(("left", l_raw, l_rel if l_rel.stop <= total_dims else None))
    if r_raw.stop <= total_dims:
        hand_blocks.append(("right", r_raw, r_rel if r_rel.stop <= total_dims else None))

    if not hand_blocks:
        vel_new = _recompute_velocity(pos_new) if vel is not None else None
        return _pack_seq(pos_new, vel_new)

    if pattern == "hand_window":
        hand_name, raw_slice, rel_slice = hand_blocks[np.random.randint(0, len(hand_blocks))]
        _ = hand_name
        all_landmarks = list(range(21))
        pos_new[:, raw_slice] = _mask_landmarks_in_block(pos_new[:, raw_slice], all_landmarks, time_window)
        if rel_slice is not None:
            pos_new[:, rel_slice] = _mask_landmarks_in_block(pos_new[:, rel_slice], all_landmarks, time_window)

    elif pattern == "staggered_fingers":
        selected_hands = hand_blocks if np.random.rand() < 0.4 else [hand_blocks[np.random.randint(0, len(hand_blocks))]]
        finger_names = list(_FINGER_GROUPS.keys())
        chosen = np.random.choice(finger_names, size=2, replace=False)
        for _, raw_slice, rel_slice in selected_hands:
            for j, fname in enumerate(chosen):
                idxs = _FINGER_GROUPS[str(fname)]
                stagger = time_window[j::2]
                if stagger.size == 0:
                    stagger = time_window
                pos_new[:, raw_slice] = _mask_landmarks_in_block(pos_new[:, raw_slice], idxs, stagger)
                if rel_slice is not None:
                    pos_new[:, rel_slice] = _mask_landmarks_in_block(pos_new[:, rel_slice], idxs, stagger)

    else:
        _, raw_slice, rel_slice = hand_blocks[np.random.randint(0, len(hand_blocks))]
        fname = str(np.random.choice(list(_FINGER_GROUPS.keys())))
        idxs = _FINGER_GROUPS[fname]
        pos_new[:, raw_slice] = _mask_landmarks_in_block(pos_new[:, raw_slice], idxs, time_window)
        if rel_slice is not None:
            pos_new[:, rel_slice] = _mask_landmarks_in_block(pos_new[:, rel_slice], idxs, time_window)

    vel_new = _recompute_velocity(pos_new) if vel is not None else None
    return _pack_seq(pos_new, vel_new)


def augment_sequence(sequence: np.ndarray, variants: int = DEFAULT_AUGMENT_VARIANTS) -> List[np.ndarray]:
    """Generate a fixed ordered set of landmark augmentations.

    This is deterministic in variant selection: each output corresponds to one
    predefined augmentation variant instead of a random weighted combination.
    """
    fixed_variants = [
        ("aug1", simulate_new_person),
        ("aug2", simulate_hand_proportions),
        ("aug3", simulate_face_anchor_shift),
        ("aug4", lambda s: temporal_speed_variation(s, speed_factor=0.88)),
        ("aug5", lambda s: frame_drop_and_interpolate(s, n_drop=2)),
        ("aug6", lambda s: spatial_noise_injection(s, sigma=0.01)),
        ("aug7", lambda s: random_translation(s, tx=0.03, ty=-0.03)),
        ("aug8", lambda s: random_scaling(s, scale=1.08, center_mode="mean")),
        ("aug9", horizontal_flip),
        ("aug10", lambda s: landmark_3d_rotation(s, axis="z", angle=12.0)),
        ("aug11", lambda s: landmark_pixel_dropout(s, dropout_rate=0.10)),
        ("aug12", lambda s: landmark_coarse_dropout(s, block_size=4)),
        ("aug13", lambda s: landmark_fog_noise(s, intensity=0.05)),
        ("aug14", lambda s: landmark_brightness_contrast(s, scale=1.10)),
        ("aug15", lambda s: time_shift_with_wrap_or_pad(s, max_shift=3, mode="wrap")),
        ("aug16", lambda s: time_shift_with_wrap_or_pad(s, max_shift=2, mode="pad")),
        ("aug17", lambda s: piecewise_temporal_warp(s, n_segments=3, speed_min=0.75, speed_max=1.30)),
        ("aug18", per_finger_articulation_scaling),
        ("aug19", wrist_trajectory_drift),
        ("aug20", landmark_confidence_masking),
    ]

    limit = len(fixed_variants) if variants is None or variants <= 0 else min(int(variants), len(fixed_variants))
    out: List[np.ndarray] = []

    for variant_name, fn in fixed_variants[:limit]:
        s = sequence.copy()
        with _temporary_numpy_seed(_variant_seed(sequence, variant_name)):
            s = fn(s)
        s = np.asarray(s, dtype=np.float32)
        if s.shape == (NUM_FRAMES, INPUT_SIZE):
            out.append(s)

    return out


def augment_dataset(input_dir: str, output_dir: Optional[str] = None, augment_per_sample: int = 3, class_only: Optional[str] = None):
    """Augment all .npy files under `input_dir` class subfolders.

    Saves augmented files into same class folders under `output_dir` (defaults
    to `input_dir`). Original files are left untouched.
    
    Args:
        class_only: If set, only process this specific class folder (supports partial matching).
    """
    if output_dir is None:
        output_dir = input_dir

    all_classes = sorted(os.listdir(input_dir))
    
    if class_only:
        matching = [c for c in all_classes if c == class_only]
        if not matching:
            raise ValueError(f"Class '{class_only}' not found in {input_dir}. Available: {', '.join(all_classes[:5])}...")
        classes_to_process = matching
    else:
        classes_to_process = all_classes

    for cls in classes_to_process:
        cls_in = os.path.join(input_dir, cls)
        cls_out = os.path.join(output_dir, cls)
        if not os.path.isdir(cls_in):
            continue
        os.makedirs(cls_out, exist_ok=True)

        for fname in sorted(os.listdir(cls_in)):
            if not fname.endswith('.npy'):
                continue
            if not _is_base_webcam_file(fname):
                continue
            fpath = os.path.join(cls_in, fname)
            try:
                seq = np.load(fpath)
            except Exception:
                print(f"[WARN] Could not load: {fpath}")
                continue

            aug_list = augment_sequence(seq, variants=augment_per_sample)
            base = _compact_sample_stem(fname)
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
    parser.add_argument('--n', type=int, default=DEFAULT_AUGMENT_VARIANTS, help='augmentations per sample (fixed ordered variants)')
    parser.add_argument('--class', dest='class_only', default=None, help='only augment this specific class (supports partial name matching)')
    args = parser.parse_args()

    augment_dataset(args.input_dir, args.output_dir, augment_per_sample=args.n, class_only=args.class_only)
