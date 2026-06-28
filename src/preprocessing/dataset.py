"""
Custom PyTorch Dataset for loading preprocessed .npy landmark sequences.
Includes data augmentation and balanced oversampling for training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import get_config

cfg = get_config()

# Convenience references for dataset operations
PROCESSED_DIR = cfg.paths.processed_dir
LANDMARK_DIM = cfg.landmarks.landmark_dim_per_hand
RAW_FRAME_FEAT_DIM = cfg.landmarks.raw_frame_features_dim
INPUT_SIZE = cfg.frame_features.input_sequence_dim
FRAME_FEAT_DIM = cfg.frame_features.frame_features_dim
PROXIMITY_FEAT_DIM = cfg.spatial.proximity_dim
PROXIMITY_INDEX = cfg.frame_features.proximity_index
USE_VELOCITY = cfg.frame_features.use_velocity


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
        neg_root: str | None = None,
        neg_label: str = "__reject__",
        archived_root: str | None = None,
        archived_weight: float = 0.25,
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
        # Each sample is a tuple: (file_path, label_index, sample_weight)
        self.samples = []   # List of (file_path, label_index, weight)
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

        # Optionally include negatives root as a single reject class
        self.neg_root = None
        self.neg_label = neg_label
        if neg_root and os.path.isdir(neg_root):
            # Count .npy files under neg_root (recursive)
            neg_count = 0
            for _root, _dirs, files in os.walk(neg_root):
                for fn in files:
                    if fn.endswith(".npy"):
                        neg_count += 1
            if neg_count >= min_samples:
                # Append reject class at the end
                self.neg_root = neg_root
                if neg_label not in self.classes:
                    self.class_to_idx[neg_label] = len(self.classes)
                    self.classes.append(neg_label)

        # Collect all .npy file paths with labels, grouped by class
        # Validate files during collection to skip corrupt ones
        class_samples = {i: [] for i in range(len(self.classes))}
        corrupt_files = []
        for cls_name in self.classes:
            if cls_name == self.neg_label and self.neg_root:
                # Collect negatives recursively from neg_root
                cls_idx = self.class_to_idx[cls_name]
                for _root, _dirs, files in os.walk(self.neg_root):
                    for fname in files:
                        if fname.endswith(".npy"):
                            fpath = os.path.join(_root, fname)
                            try:
                                test_data = np.load(fpath)
                                if test_data.size > 0:
                                    class_samples[cls_idx].append((fpath, cls_idx, 1.0))
                                else:
                                    corrupt_files.append((fpath, "Empty file"))
                            except Exception as e:
                                corrupt_files.append((fpath, str(e)))
            else:
                cls_dir = os.path.join(root_dir, cls_name)
                cls_idx = self.class_to_idx[cls_name]
                for fname in os.listdir(cls_dir):
                    if fname.endswith(".npy"):
                        fpath = os.path.join(cls_dir, fname)
                        # Quick validation: try to load file
                        try:
                            test_data = np.load(fpath)
                            if test_data.size > 0:
                                class_samples[cls_idx].append((fpath, cls_idx, 1.0))
                            else:
                                corrupt_files.append((fpath, "Empty file"))
                        except Exception as e:
                            corrupt_files.append((fpath, str(e)))
                # Optionally include archived samples for this class (lower weight)
                if archived_root and os.path.isdir(archived_root):
                    archived_cls_dir = os.path.join(archived_root, cls_name)
                    if os.path.isdir(archived_cls_dir):
                        for af in os.listdir(archived_cls_dir):
                            if af.endswith(".npy"):
                                afpath = os.path.join(archived_cls_dir, af)
                                try:
                                    test_data = np.load(afpath)
                                    if test_data.size > 0:
                                        class_samples[cls_idx].append((afpath, cls_idx, float(archived_weight)))
                                except Exception:
                                    # skip corrupt archived files silently
                                    pass
        
        if corrupt_files:
            print(f"[Dataset] WARNING: Found {len(corrupt_files)} corrupt files:")
            for fpath, err in corrupt_files[:5]:
                print(f"  - {os.path.basename(fpath)}: {err}")
            if len(corrupt_files) > 5:
                print(f"  ... and {len(corrupt_files) - 5} more")
            print("[Dataset] Corrupt files will be skipped.")

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
            # Each item already has (fpath, idx, weight)
            self.samples.extend(class_samples[cls_idx])

        # Print distribution
        label_counts = {}
        for _, lbl, _ in self.samples:
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
        
        Handles corrupt files by retrying with different samples or raising informative error.
        """
        import sys
        
        fpath, label, weight = self.samples[idx]
        
        # Try to load the file with error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                seq = np.load(fpath).astype(np.float32)
                if seq.size == 0:
                    raise ValueError("Empty file")
                break
            except (OSError, ValueError, RuntimeError) as e:
                if attempt == max_retries - 1:
                    # Last attempt failed - provide detailed error
                    error_msg = (
                        f"\n[Dataset] ❌ CORRUPT FILE DETECTED:\n"
                        f"  Path: {fpath}\n"
                        f"  Error: {str(e)}\n"
                        f"  Index: {idx}\n"
                        f"\nTo fix this issue:\n"
                        f"  1. Delete the file: rm \"{fpath}\"\n"
                        f"  2. Re-run training\n"
                        f"\nOr clean all corrupt files:\n"
                        f"  python cleanup_dataset_npy.py\n"
                    )
                    print(error_msg, file=sys.stderr)
                    raise RuntimeError(error_msg) from e
                # Try again
                import time
                time.sleep(0.01 * (attempt + 1))
        
        seq, proximity = self._prepare_sequence(
            seq,
            augment=self.augment,
        )

        seq_t = torch.from_numpy(seq)
        prox_t = torch.from_numpy(proximity)
        lbl_t = torch.tensor(label, dtype=torch.long)
        weight_t = torch.tensor(weight, dtype=torch.float32)
        return seq_t, prox_t, lbl_t, weight_t

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

        # 6) Time warping (40% chance - increased from 30%)
        #    Slightly speed up or slow down by resampling frames
        if np.random.rand() < 0.4:
            warp = np.random.uniform(0.75, 1.25)  # Increased range
            new_len = max(int(num_frames * warp), num_frames)
            indices = np.linspace(0, num_frames - 1, new_len, dtype=int)
            warped = seq[indices]
            re_idx = np.linspace(0, len(warped) - 1, num_frames, dtype=int)
            seq = warped[re_idx]

        # 7) Per-hand coordinate dropout (20% chance) - NEW
        #    Randomly zero out entire hand for some frames
        if np.random.rand() < 0.2:
            raw_dim = min(feat_dim, RAW_FRAME_FEAT_DIM)
            hand_blocks = raw_dim // LANDMARK_DIM
            for _ in range(np.random.randint(1, 2)):
                hand_to_drop = np.random.randint(0, hand_blocks)
                start = hand_to_drop * LANDMARK_DIM
                end = start + LANDMARK_DIM
                drop_frames = np.random.choice(num_frames, 
                                              max(1, num_frames // 3), 
                                              replace=False)
                seq[drop_frames, start:end] = 0.0

        # 8) Stronger noise on specific frames (25% chance) - NEW
        if np.random.rand() < 0.25:
            noise_frames = np.random.choice(num_frames, 
                                           max(1, num_frames // 4), 
                                           replace=False)
            seq[noise_frames] += np.random.randn(len(noise_frames), feat_dim) * 0.03

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

    @staticmethod
    def mixup(seq1: np.ndarray, seq2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """
        Mixup augmentation: blend two sequences with random weight.
        Creates synthetic training samples from convex combinations.
        Applied during training with labels soft-mixed accordingly.
        """
        lam = np.random.beta(alpha, alpha)
        mixed = lam * seq1 + (1 - lam) * seq2
        return mixed.astype(np.float32)

    @staticmethod
    def cutmix(seq1: np.ndarray, seq2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """
        CutMix augmentation: replace temporal region of seq1 with seq2.
        Preserves temporal structure while mixing sequences.
        """
        num_frames = seq1.shape[0]
        lam = np.random.beta(alpha, alpha)
        cut_ratio = np.sqrt(1 - lam)
        num_cut = max(1, int(num_frames * cut_ratio))
        
        cut_start = np.random.randint(0, num_frames - num_cut) if num_frames > num_cut else 0
        cut_end = cut_start + num_cut
        
        mixed = seq1.copy()
        mixed[cut_start:cut_end] = seq2[cut_start:cut_end]
        return mixed.astype(np.float32)
