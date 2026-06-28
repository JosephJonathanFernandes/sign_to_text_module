"""Lightweight quality discriminator for CVAE-generated landmark sequences.

This is a post-generation filter, not an adversarial GAN discriminator.
It scores sequences with P(sample_is_real) and can optionally condition on class.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class DiscriminatorConfig:
    seq_len: int = 20
    feature_dim: int = 506
    hidden_size: int = 64
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.15
    class_embed_dim: int = 16
    class_aware: bool = False


@dataclass
class FilterHeuristicConfig:
    enabled: bool = True
    min_motion_variance: float = 1e-5
    min_feature_std: float = 0.002
    max_frame_jump: float = 2.5
    max_frame_drift: float = 4.0
    min_active_ratio: float = 0.05


@dataclass
class LabelEncoderState:
    classes: list[str]

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.classes)}


class LandmarkLabelEncoder:
    def __init__(self, classes: Iterable[str]):
        classes_sorted = sorted({c.strip() for c in classes if c and c.strip()})
        if not classes_sorted:
            raise ValueError("Quality discriminator requires at least one class")
        self._state = LabelEncoderState(classes=classes_sorted)
        self._class_to_idx = self._state.class_to_idx

    @property
    def classes(self) -> list[str]:
        return list(self._state.classes)

    @property
    def num_classes(self) -> int:
        return len(self._state.classes)

    def encode(self, class_name: str) -> int:
        return self._class_to_idx[class_name]

    def decode(self, idx: int) -> str:
        return self._state.classes[idx]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self._state)

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "LandmarkLabelEncoder":
        return cls(classes=state["classes"])


class SequenceQualityDataset(Dataset):
    """Dataset over real and synthetic landmark sequences.

    Real samples come from processed/<class_name>/*.npy.
    Synthetic samples can come from generated/<class_name>/*.npy or
    processed/<class_name>/cvae_*.npy.
    """

    def __init__(
        self,
        real_root: str = "processed",
        synthetic_root: str | None = "generated",
        *,
        seq_len: int,
        feature_dim: int | None = None,
        class_aware: bool = False,
        include_classes: set[str] | None = None,
        max_real_per_class: int | None = None,
        max_fake_per_class: int | None = None,
        real_exclude_prefixes: tuple[str, ...] = ("cvae_",),
        synthetic_prefixes: tuple[str, ...] | None = None,
    ) -> None:
        self.real_root = real_root
        self.synthetic_root = synthetic_root
        self.seq_len = int(seq_len)
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self.class_aware = class_aware
        self.include_classes = include_classes
        self.real_exclude_prefixes = tuple(real_exclude_prefixes)
        self.synthetic_prefixes = tuple(synthetic_prefixes) if synthetic_prefixes is not None else None
        self.max_real_per_class = max_real_per_class
        self.max_fake_per_class = max_fake_per_class

        if not os.path.isdir(real_root):
            raise FileNotFoundError(f"Real root not found: {os.path.abspath(real_root)}")

        self.class_names = self._discover_classes(real_root)
        if synthetic_root and os.path.isdir(synthetic_root):
            for class_name in self._discover_classes(synthetic_root):
                if class_name not in self.class_names:
                    self.class_names.append(class_name)
            self.class_names = sorted(set(self.class_names))

        self.label_encoder = LandmarkLabelEncoder(self.class_names)
        self.samples: list[dict[str, Any]] = []
        self._scan_sources()

        if not self.samples:
            raise RuntimeError("No samples found for SequenceQualityDataset")

        if self.feature_dim is None:
            seq = np.load(self.samples[0]["path"], allow_pickle=False)
            if seq.ndim != 2:
                raise ValueError(f"Expected 2D sequence, got {seq.shape} in {self.samples[0]['path']}")
            self.feature_dim = int(seq.shape[1])

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r"^\d+\.?\s*", "", s)
        s = s.replace(" ", "_")
        return s

    def _allow_class(self, class_name: str) -> bool:
        if not self.include_classes:
            return True
        return self._normalize_class_name(class_name) in {self._normalize_class_name(c) for c in self.include_classes}

    def _discover_classes(self, root: str) -> list[str]:
        out: list[str] = []
        for name in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, name)) and self._allow_class(name):
                out.append(name)
        return out

    def _allow_filename(self, fname: str, *, synthetic: bool) -> bool:
        lower = fname.lower()
        if not lower.endswith(".npy"):
            return False
        if synthetic:
            if self.synthetic_prefixes is None:
                return True
            return lower.startswith(self.synthetic_prefixes)
        return not lower.startswith(self.real_exclude_prefixes)

    def _scan_one_root(self, root: str, is_real: bool) -> None:
        for class_name in self.class_names:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue

            files = []
            for fname in sorted(os.listdir(class_dir)):
                if self._allow_filename(fname, synthetic=not is_real):
                    files.append(fname)

            if is_real and self.max_real_per_class is not None:
                files = files[: self.max_real_per_class]
            if not is_real and self.max_fake_per_class is not None:
                files = files[: self.max_fake_per_class]

            for fname in files:
                path = os.path.join(class_dir, fname)
                if os.path.isfile(path):
                    self.samples.append(
                        {
                            "path": path,
                            "class_name": class_name,
                            "class_idx": self.label_encoder.encode(class_name),
                            "is_real": is_real,
                        }
                    )

    def _scan_sources(self) -> None:
        self._scan_one_root(self.real_root, is_real=True)
        if self.synthetic_root and os.path.isdir(self.synthetic_root):
            self._scan_one_root(self.synthetic_root, is_real=False)

        if not self.class_names:
            raise RuntimeError("No classes found for quality discriminator dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        seq = np.load(sample["path"], allow_pickle=False).astype(np.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got {seq.shape} in {sample['path']}")
        seq = pad_or_truncate_sequence(seq, self.seq_len, self.feature_dim)
        return {
            "x": torch.from_numpy(seq),
            "y": torch.tensor(sample["class_idx"], dtype=torch.long),
            "target": torch.tensor(1.0 if sample["is_real"] else 0.0, dtype=torch.float32),
            "class_name": sample["class_name"],
            "path": sample["path"],
            "is_real": sample["is_real"],
        }


class QualityDiscriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig) -> None:
        super().__init__()
        self.cfg = config
        self.use_class_conditioning = config.class_aware
        if self.use_class_conditioning:
            self.class_embedding = nn.Embedding(512, config.class_embed_dim)

        input_size = config.feature_dim
        if self.use_class_conditioning:
            input_size += config.class_embed_dim

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size * (2 if config.bidirectional else 1) * 2, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_class_conditioning:
            if y is None:
                raise ValueError("Class-aware discriminator requires class labels")
            y_embed = self.class_embedding(y)
            y_expand = y_embed.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, y_expand], dim=-1)

        enc_out, h_n = self.encoder(x)
        if self.cfg.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]

        pooled_mean = enc_out.mean(dim=1)
        pooled_max = enc_out.max(dim=1).values
        features = torch.cat([pooled_mean, pooled_max], dim=-1)
        logits = self.head(features).squeeze(-1)
        return logits

    @torch.no_grad()
    def score(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, y))


def pad_or_truncate_sequence(seq: np.ndarray, seq_len: int, feature_dim: int) -> np.ndarray:
    if seq.shape[1] > feature_dim:
        seq = seq[:, :feature_dim]
    elif seq.shape[1] < feature_dim:
        pad_f = np.zeros((seq.shape[0], feature_dim - seq.shape[1]), dtype=np.float32)
        seq = np.concatenate([seq, pad_f], axis=1)

    if seq.shape[0] > seq_len:
        seq = seq[:seq_len]
    elif seq.shape[0] < seq_len:
        pad_t = np.zeros((seq_len - seq.shape[0], feature_dim), dtype=np.float32)
        seq = np.concatenate([seq, pad_t], axis=0)

    return seq.astype(np.float32, copy=False)


def quality_heuristic_score(seq: np.ndarray, cfg: FilterHeuristicConfig) -> tuple[bool, str]:
    if not cfg.enabled:
        return True, "disabled"

    if seq.ndim != 2 or seq.shape[0] < 2:
        return False, "invalid_shape"

    if np.isnan(seq).any() or np.isinf(seq).any():
        return False, "nan_or_inf"

    if np.std(seq) < cfg.min_feature_std:
        return False, "low_feature_std"

    velocity = np.diff(seq, axis=0)
    motion_var = float(np.var(velocity))
    if motion_var < cfg.min_motion_variance:
        return False, "low_motion_variance"

    jump_mag = float(np.max(np.linalg.norm(velocity, axis=1)))
    if jump_mag > cfg.max_frame_jump:
        return False, "large_frame_jump"

    drift = float(np.linalg.norm(seq[-1] - seq[0]))
    if drift > cfg.max_frame_drift:
        return False, "excessive_drift"

    active_ratio = float(np.mean(np.abs(seq) > 1e-6))
    if active_ratio < cfg.min_active_ratio:
        return False, "frozen_landmarks"

    return True, "ok"


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)
    y_pred = (y_score >= threshold).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    far = fp / max(1, int(np.sum(y_true == 0)))
    rejection_rate = tn / max(1, int(np.sum(y_true == 0)))

    auc = roc_auc_score_binary(y_true, y_score)

    return {
        "accuracy": float(accuracy),
        "roc_auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "false_accept_rate": float(far),
        "synthetic_rejection_rate": float(rejection_rate),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def save_json(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
