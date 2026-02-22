"""
Dataset class for image-based ISL letter/number recognition.
Merges data from archive/, archive (1)/ and archive (2)/ into a unified dataset.
Supports on-the-fly augmentation for training.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config_image import (
    ARCHIVE1_DIR, ARCHIVE2_DIR, ARCHIVE3_DIR, IMG_SIZE,
)


# Unified class list: A-Z then 0-9
LETTER_CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
DIGIT_CLASSES = [str(i) for i in range(0, 10)]
ALL_CLASSES = sorted(LETTER_CLASSES + DIGIT_CLASSES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}


class ISLImageDataset(Dataset):
    """
    Loads static hand gesture images from archive (1) and (2).
    Each image is a 128x128 RGB JPG of a single hand sign.

    Classes: A-Z (26 letters) + 0-9 (10 digits) = 36 classes.
    """

    def __init__(self, augment: bool = False):
        self.augment = augment
        self.samples = []       # (file_path, label_index)
        self.classes = ALL_CLASSES
        self.class_to_idx = CLASS_TO_IDX

        self._scan_archive(ARCHIVE1_DIR)
        self._scan_archive(ARCHIVE2_DIR)
        self._scan_archive(ARCHIVE3_DIR)

        print(
            f"[ImageDataset] {len(self.samples)} images, "
            f"{len(self.classes)} classes (augment={augment})"
        )

    def _scan_archive(self, root: str):
        """Scan an archive directory for class folders."""
        if not os.path.isdir(root):
            print(f"[ImageDataset] Warning: {root} not found")
            return

        for folder in os.listdir(root):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue

            # Map folder name to class
            cls_name = folder.upper().strip()
            if cls_name not in CLASS_TO_IDX:
                continue

            label = CLASS_TO_IDX[cls_name]
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fpath = os.path.join(folder_path, fname)
                    self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = cv2.imread(fpath)

        if img is None:
            # Return black image on read failure
            img = np.zeros(
                (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
            )

        # Resize if needed
        if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            img = self._augment(img)

        # Normalize to [0, 1] and convert to (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        return (
            torch.from_numpy(img),
            torch.tensor(label, dtype=torch.long),
        )

    @property
    def num_classes(self):
        return len(self.classes)

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        """Apply random augmentations to an image."""
        img = img.copy()
        h, w = img.shape[:2]

        # 1) Random horizontal flip (50%)
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)

        # 2) Random rotation +/- 15 degrees (60%)
        if np.random.rand() < 0.6:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        # 3) Random brightness/contrast (50%)
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.randint(-20, 21)     # brightness
            img = np.clip(
                alpha * img.astype(np.float32) + beta,
                0, 255,
            ).astype(np.uint8)

        # 4) Random crop + resize (40%)
        if np.random.rand() < 0.4:
            pad = np.random.randint(5, 15)
            x1 = np.random.randint(0, pad + 1)
            y1 = np.random.randint(0, pad + 1)
            x2 = w - np.random.randint(0, pad + 1)
            y2 = h - np.random.randint(0, pad + 1)
            if x2 > x1 + 10 and y2 > y1 + 10:
                img = img[y1:y2, x1:x2]
                img = cv2.resize(img, (w, h))

        # 5) Gaussian noise (30%)
        if np.random.rand() < 0.3:
            noise = np.random.randn(*img.shape) * 10
            img = np.clip(
                img.astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)

        return img
