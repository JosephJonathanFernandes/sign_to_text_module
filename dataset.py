"""
Custom PyTorch Dataset for loading preprocessed .npy landmark sequences.
Includes data augmentation for training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import PROCESSED_DIR


class ISLDataset(Dataset):
    """
    Loads .npy files from processed/ directory.
    Each .npy file is a (NUM_FRAMES, 63) array of hand landmarks.

    Supports on-the-fly augmentation for training:
      - Gaussian noise on landmarks
      - Random scaling
      - Temporal shift (roll along time axis)
      - Random frame dropout (zero out some frames)
    """

    def __init__(
        self,
        root_dir: str = PROCESSED_DIR,
        augment: bool = False,
        min_samples: int = 1,
    ):
        """
        Args:
            root_dir: Path to the processed/ directory.
            augment: Whether to apply data augmentation.
            min_samples: Minimum samples per class to include (default 1).
                         Set higher (e.g. 3) to filter out single-sample classes.
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

        # Collect all .npy file paths with labels
        for cls_name in filtered_dirs:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    fpath = os.path.join(cls_dir, fname)
                    self.samples.append(
                        (fpath, self.class_to_idx[cls_name])
                    )

        print(
            f"[Dataset] {len(self.samples)} samples, "
            f"{len(self.classes)} classes "
            f"(augment={self.augment})"
        )
        print(f"[Dataset] Classes: {self.classes}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            sequence: FloatTensor (NUM_FRAMES, 63)
            label:    LongTensor scalar
        """
        fpath, label = self.samples[idx]
        seq = np.load(fpath).astype(np.float32)

        if self.augment:
            seq = self._augment(seq)

        seq_t = torch.from_numpy(seq)
        lbl_t = torch.tensor(label, dtype=torch.long)
        return seq_t, lbl_t

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    # ── Augmentation methods ─────────────────────────

    @staticmethod
    def _augment(seq: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a sequence.
        Works with both (T, 63) and (T, 126) feature dims.
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

        # 5) XY rotation of landmarks (40% chance)
        #    Rotate hand in the XY plane by a small angle
        if np.random.rand() < 0.4:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # Apply to position coords (first 63 dims = 21 landmarks x 3)
            for f in range(num_frames):
                lm_dim = min(feat_dim, 63)
                pos = seq[f, :lm_dim].reshape(-1, 3)
                x_rot = pos[:, 0] * cos_a - pos[:, 1] * sin_a
                y_rot = pos[:, 0] * sin_a + pos[:, 1] * cos_a
                pos[:, 0] = x_rot
                pos[:, 1] = y_rot
                seq[f, :lm_dim] = pos.flatten()

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

        return seq
