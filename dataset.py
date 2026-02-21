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
    ):
        """
        Args:
            root_dir: Path to the processed/ directory.
            augment: Whether to apply data augmentation.
        """
        self.augment = augment
        self.samples = []   # List of (file_path, label_index)
        self.classes = []    # Sorted list of class names
        self.class_to_idx = {}

        # Discover classes
        class_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if not class_dirs:
            raise FileNotFoundError(
                f"No class folders in {root_dir}. "
                "Run preprocess.py first."
            )

        self.classes = class_dirs
        self.class_to_idx = {
            cls: i for i, cls in enumerate(class_dirs)
        }

        # Collect all .npy file paths with labels
        for cls_name in class_dirs:
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
        """Apply random augmentations to a sequence."""
        seq = seq.copy()

        # 1) Gaussian noise (70% chance)
        if np.random.rand() < 0.7:
            noise = np.random.randn(*seq.shape) * 0.02
            seq = seq + noise.astype(np.float32)

        # 2) Random scaling (60% chance)
        if np.random.rand() < 0.6:
            scale = np.random.uniform(0.85, 1.15)
            seq = seq * scale

        # 3) Temporal shift — roll frames (50% chance)
        if np.random.rand() < 0.5:
            shift = np.random.randint(-3, 4)
            seq = np.roll(seq, shift, axis=0)

        # 4) Random frame dropout (30% chance)
        #    Zero out 1-3 random frames
        if np.random.rand() < 0.3:
            n_drop = np.random.randint(1, 4)
            drop_idx = np.random.choice(
                seq.shape[0], n_drop, replace=False
            )
            seq[drop_idx] = 0.0

        return seq
