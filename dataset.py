"""
Custom PyTorch Dataset for loading preprocessed .npy landmark sequences.
Includes data augmentation and balanced oversampling for training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import (
    PROCESSED_DIR,
    LANDMARK_DIM,
    RAW_FRAME_FEAT_DIM,
    INPUT_SIZE,
    FRAME_FEAT_DIM,
    PROXIMITY_FEAT_DIM,
    PROXIMITY_INDEX,
    USE_VELOCITY,
)


class ISLDataset(Dataset):
    """
    Loads .npy files from processed/ directory.
    Each .npy file is a (NUM_FRAMES, feat_dim) array of hand landmarks.

    Supports:
      - On-the-fly augmentation (noise, scale, rotate, warp, dropout)
      - Balanced oversampling (repeat minority classes to match majority)
    """

    def __init__(
        self,
        root_dir: str = PROCESSED_DIR,
        augment: bool = False,
        min_samples: int = 1,
        oversample: bool = False,
    ):
        """
        Args:
            root_dir: Path to the processed/ directory.
            augment: Whether to apply data augmentation.
            min_samples: Minimum samples per class to include.
            oversample: Whether to oversample minority classes to
                        balance the dataset (repeat samples).
        """
        self.augment = augment
        self.samples = []   # List of (file_path, label_index)
        self.classes = []    # Sorted list of class names
        self.class_to_idx = {}

        # Discover classes and count samples
        class_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if not class_dirs:
            raise FileNotFoundError(
                f"No class folders in {root_dir}. "
                "Run preprocess.py first."
            )

        # Filter classes by minimum sample count
        filtered_dirs = []
        for cls_name in class_dirs:
            cls_dir = os.path.join(root_dir, cls_name)
            npy_count = len([
                f for f in os.listdir(cls_dir) if f.endswith(".npy")
            ])
            if npy_count >= min_samples:
                filtered_dirs.append(cls_name)

        if not filtered_dirs:
            raise ValueError(
                f"No classes have >= {min_samples} samples."
            )

        if len(filtered_dirs) < len(class_dirs):
            print(
                f"[Dataset] Filtered: {len(class_dirs)} -> "
                f"{len(filtered_dirs)} classes "
                f"(min_samples={min_samples})"
            )

        self.classes = filtered_dirs
        self.class_to_idx = {
            cls: i for i, cls in enumerate(filtered_dirs)
        }

        # Collect all .npy file paths with labels, grouped by class
        class_samples = {i: [] for i in range(len(filtered_dirs))}
        for cls_name in filtered_dirs:
            cls_dir = os.path.join(root_dir, cls_name)
            cls_idx = self.class_to_idx[cls_name]
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    fpath = os.path.join(cls_dir, fname)
                    class_samples[cls_idx].append((fpath, cls_idx))

        # Balanced oversampling: repeat minority class samples
        if oversample and class_samples:
            max_count = max(len(v) for v in class_samples.values())
            for cls_idx, items in class_samples.items():
                if not items:
                    continue
                n = len(items)
                if n < max_count:
                    # Repeat samples to reach max_count
                    repeats = (max_count // n)
                    remainder = max_count % n
                    oversampled = items * repeats + items[:remainder]
                    class_samples[cls_idx] = oversampled

        # Flatten to single list
        for cls_idx in sorted(class_samples.keys()):
            self.samples.extend(class_samples[cls_idx])

        # Print distribution
        label_counts = {}
        for _, lbl in self.samples:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        print(
            f"[Dataset] {len(self.samples)} samples, "
            f"{len(self.classes)} classes "
            f"(augment={self.augment}, oversample={oversample})"
        )
        dist = ", ".join(
            f"{self.classes[i]}={label_counts.get(i, 0)}"
            for i in range(len(self.classes))
        )
        print(f"[Dataset] Distribution: {dist}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            sequence: FloatTensor (NUM_FRAMES, INPUT_SIZE)
            proximity: FloatTensor (NUM_FRAMES,)
            label:    LongTensor scalar
        """
        fpath, label = self.samples[idx]
        seq = np.load(fpath).astype(np.float32)
        seq, proximity = self._prepare_sequence(
            seq,
            augment=self.augment,
        )

        seq_t = torch.from_numpy(seq)
        prox_t = torch.from_numpy(proximity)
        lbl_t = torch.tensor(label, dtype=torch.long)
        return seq_t, prox_t, lbl_t

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    # ── Augmentation methods ─────────────────────────

    @staticmethod
    def _augment(seq: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a sequence.
        Works with all configured feature sizes.
        """
        seq = seq.copy()
        num_frames, feat_dim = seq.shape

        # 1) Gaussian noise (70% chance)
        if np.random.rand() < 0.7:
            noise = np.random.randn(*seq.shape) * 0.015
            seq = seq + noise.astype(np.float32)

        # 2) Random scaling (60% chance)
        if np.random.rand() < 0.6:
            scale = np.random.uniform(0.88, 1.12)
            seq = seq * scale

        # 3) Temporal shift -- roll frames (50% chance)
        if np.random.rand() < 0.5:
            shift = np.random.randint(-3, 4)
            seq = np.roll(seq, shift, axis=0)

        # 4) Random frame dropout (30% chance)
        #    Zero out 1-3 random frames
        if np.random.rand() < 0.3:
            n_drop = np.random.randint(1, 4)
            drop_idx = np.random.choice(num_frames, n_drop, replace=False)
            seq[drop_idx] = 0.0

        # 5) XY rotation of RAW hand landmarks (40% chance)
        #    Rotate each raw hand block in the XY plane by a small angle.
        if np.random.rand() < 0.4:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for f in range(num_frames):
                raw_dim = min(feat_dim, RAW_FRAME_FEAT_DIM)
                hand_blocks = raw_dim // LANDMARK_DIM
                for b in range(hand_blocks):
                    start = b * LANDMARK_DIM
                    end = start + LANDMARK_DIM
                    pos = seq[f, start:end].reshape(-1, 3)
                    x_rot = pos[:, 0] * cos_a - pos[:, 1] * sin_a
                    y_rot = pos[:, 0] * sin_a + pos[:, 1] * cos_a
                    pos[:, 0] = x_rot
                    pos[:, 1] = y_rot
                    seq[f, start:end] = pos.flatten()

        # 6) Time warping (30% chance)
        #    Slightly speed up or slow down by resampling frames
        if np.random.rand() < 0.3:
            warp = np.random.uniform(0.8, 1.2)
            new_len = max(int(num_frames * warp), num_frames)
            indices = np.linspace(0, num_frames - 1, new_len, dtype=int)
            warped = seq[indices]
            # Resample back to original length
            re_idx = np.linspace(0, len(warped) - 1, num_frames, dtype=int)
            seq = warped[re_idx]

        seq = ISLDataset._recompute_proximity(seq)
        return seq

    @staticmethod
    def _align_input_size(seq: np.ndarray) -> np.ndarray:
        """Pad/truncate feature dim to match current INPUT_SIZE."""
        feat_dim = seq.shape[1]
        if feat_dim == INPUT_SIZE:
            return seq
        if feat_dim > INPUT_SIZE:
            return seq[:, :INPUT_SIZE]

        pad = np.zeros(
            (seq.shape[0], INPUT_SIZE - feat_dim),
            dtype=np.float32,
        )
        return np.concatenate([seq, pad], axis=1)

    @staticmethod
    def _prepare_sequence(
        seq: np.ndarray,
        augment: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align shape, apply optional augmentation, and produce proximity."""
        seq = ISLDataset._align_input_size(seq.astype(np.float32, copy=False))
        if augment:
            seq = ISLDataset._augment(seq)
        else:
            seq = ISLDataset._recompute_proximity(seq)
        proximity = ISLDataset._extract_proximity(seq)
        return seq, proximity

    @staticmethod
    def _extract_proximity(seq: np.ndarray) -> np.ndarray:
        """Read per-frame proximity from the position block."""
        num_frames = seq.shape[0]
        if PROXIMITY_FEAT_DIM <= 0 or seq.shape[1] < FRAME_FEAT_DIM:
            return np.zeros(num_frames, dtype=np.float32)

        proximity = seq[:, PROXIMITY_INDEX].astype(np.float32)
        if np.allclose(proximity, 0.0):
            # Legacy files may not contain proximity; infer from relatives.
            seq = ISLDataset._recompute_proximity(seq)
            proximity = seq[:, PROXIMITY_INDEX].astype(np.float32)
        return proximity

    @staticmethod
    def _recompute_proximity(seq: np.ndarray) -> np.ndarray:
        """
        Keep proximity coherent with augmented relative features.
        Proximity uses min(||left_relative||, ||right_relative||) per frame.
        """
        if PROXIMITY_FEAT_DIM <= 0 or seq.shape[1] < FRAME_FEAT_DIM:
            return seq

        pos = seq[:, :FRAME_FEAT_DIM]
        left_raw = pos[:, :LANDMARK_DIM]
        right_raw = pos[:, LANDMARK_DIM:2 * LANDMARK_DIM]
        left_rel = pos[:, 2 * LANDMARK_DIM:3 * LANDMARK_DIM]
        right_rel = pos[:, 3 * LANDMARK_DIM:4 * LANDMARK_DIM]

        left_present = np.any(left_raw != 0.0, axis=1)
        right_present = np.any(right_raw != 0.0, axis=1)

        d_left = np.linalg.norm(left_rel, axis=1)
        d_right = np.linalg.norm(right_rel, axis=1)

        proximity = np.ones(pos.shape[0], dtype=np.float32)
        for i in range(pos.shape[0]):
            vals = []
            if left_present[i]:
                vals.append(float(d_left[i]))
            if right_present[i]:
                vals.append(float(d_right[i]))
            if vals:
                proximity[i] = min(vals)

        pos[:, PROXIMITY_INDEX] = proximity
        seq[:, :FRAME_FEAT_DIM] = pos

        if USE_VELOCITY and seq.shape[1] >= FRAME_FEAT_DIM * 2:
            vel_col = FRAME_FEAT_DIM + PROXIMITY_INDEX
            seq[0, vel_col] = 0.0
            seq[1:, vel_col] = proximity[1:] - proximity[:-1]

        return seq
