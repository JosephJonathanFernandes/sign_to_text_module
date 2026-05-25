"""CVAE utilities for class-conditioned landmark sequence modeling.

This module provides:
- Dataset scanning for processed landmark sequences (.npy)
- Class label encoding and metadata helpers
- Conditional VAE model for sequence reconstruction/generation
- Loss computation (reconstruction + KL + velocity consistency)
- Lightweight quality checks for synthetic outputs
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
class CVAEModelConfig:
    seq_len: int = 20
    feature_dim: int = 506
    num_classes: int = 1
    z_dim: int = 64
    class_embed_dim: int = 32
    encoder_hidden_dim: int = 256
    encoder_layers: int = 2
    encoder_bidirectional: bool = True
    decoder_hidden_dim: int = 256
    decoder_layers: int = 2
    dropout: float = 0.1


@dataclass
class LossConfig:
    beta_kl: float = 1e-3
    velocity_weight: float = 0.5


@dataclass
class QualityFilterConfig:
    enabled: bool = True
    min_motion_magnitude: float = 0.002
    min_feature_std: float = 0.002
    max_frame_jump: float = 2.5


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
            raise ValueError("LabelEncoder requires at least one class")
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


class ProcessedLandmarkDataset(Dataset):
    """Dataset over processed/<class_name>/*.npy with fixed seq_len padding/truncation."""

    def __init__(
        self,
        root_dir: str,
        *,
        seq_len: int,
        feature_dim: int | None = None,
        include_classes: set[str] | None = None,
        include_prefixes: tuple[str, ...] | None = None,
        exclude_prefixes: tuple[str, ...] | None = None,
        synthetic_root: str | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.seq_len = int(seq_len)
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self.include_classes = include_classes
        self.include_classes_normalized = (
            {self._normalize_class_name(c) for c in include_classes} if include_classes else None
        )
        self.include_prefixes = tuple(include_prefixes) if include_prefixes else None
        self.exclude_prefixes = tuple(exclude_prefixes) if exclude_prefixes else None
        self.synthetic_root = synthetic_root

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Processed root not found: {os.path.abspath(root_dir)}")

        self.samples: list[tuple[str, str, bool]] = []
        self.class_names = self._discover_classes(root_dir)
        if synthetic_root and os.path.isdir(synthetic_root):
            for class_name in self._discover_classes(synthetic_root):
                if class_name not in self.class_names:
                    self.class_names.append(class_name)
            self.class_names = sorted(set(self.class_names))

        self.label_encoder = LandmarkLabelEncoder(self.class_names)
        self._scan_sources()

        if not self.samples:
            raise RuntimeError("No samples found for ProcessedLandmarkDataset")

        # Infer feature dim if not supplied.
        if self.feature_dim is None:
            seq = np.load(self.samples[0][1], allow_pickle=False)
            if seq.ndim != 2:
                raise ValueError(f"Expected 2D sequence, got shape={seq.shape} in {self.samples[0][1]}")
            self.feature_dim = int(seq.shape[1])

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        # Supports both legacy names like "58. Idle" and normalized names like "idle".
        s = name.strip().lower()
        s = re.sub(r"^\d+\.?\s*", "", s)
        s = s.replace(" ", "_")
        return s

    def _discover_classes(self, root: str) -> list[str]:
        out: list[str] = []
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                if self.include_classes_normalized:
                    nname = self._normalize_class_name(name)
                    if nname not in self.include_classes_normalized:
                        continue
                if self.include_classes and name in self.include_classes:
                    out.append(name)
                    continue
                if self.include_classes_normalized:
                    out.append(name)
                    continue
                out.append(name)
        return out

    def _allow_filename(self, fname: str) -> bool:
        lower = fname.lower()
        if not lower.endswith(".npy"):
            return False
        if self.include_prefixes and not lower.startswith(self.include_prefixes):
            return False
        if self.exclude_prefixes and lower.startswith(self.exclude_prefixes):
            return False
        return True

    def _scan_one_root(self, root: str, is_synthetic: bool) -> None:
        for class_name in self.class_names:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not self._allow_filename(fname):
                    continue
                path = os.path.join(class_dir, fname)
                if os.path.isfile(path):
                    self.samples.append((class_name, path, is_synthetic))

    def _scan_sources(self) -> None:
        self._scan_one_root(self.root_dir, is_synthetic=False)
        if self.synthetic_root:
            self._scan_one_root(self.synthetic_root, is_synthetic=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        class_name, path, is_synthetic = self.samples[idx]
        seq = np.load(path, allow_pickle=False).astype(np.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got {seq.shape} in {path}")

        seq = pad_or_truncate_sequence(seq, self.seq_len, self.feature_dim)
        return {
            "x": torch.from_numpy(seq),
            "y": torch.tensor(self.label_encoder.encode(class_name), dtype=torch.long),
            "class_name": class_name,
            "path": path,
            "is_synthetic": is_synthetic,
        }


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


class LandmarkCVAE(nn.Module):
    def __init__(self, config: CVAEModelConfig) -> None:
        super().__init__()
        self.cfg = config

        self.class_embedding = nn.Embedding(config.num_classes, config.class_embed_dim)

        encoder_in_dim = config.feature_dim + config.class_embed_dim
        self.encoder_gru = nn.GRU(
            input_size=encoder_in_dim,
            hidden_size=config.encoder_hidden_dim,
            num_layers=config.encoder_layers,
            dropout=config.dropout if config.encoder_layers > 1 else 0.0,
            bidirectional=config.encoder_bidirectional,
            batch_first=True,
        )
        enc_dir = 2 if config.encoder_bidirectional else 1
        encoder_out_dim = config.encoder_hidden_dim * enc_dir
        self.mu_head = nn.Linear(encoder_out_dim, config.z_dim)
        self.logvar_head = nn.Linear(encoder_out_dim, config.z_dim)

        decoder_in_dim = config.z_dim + config.class_embed_dim
        self.decoder_gru = nn.GRU(
            input_size=decoder_in_dim,
            hidden_size=config.decoder_hidden_dim,
            num_layers=config.decoder_layers,
            dropout=config.dropout if config.decoder_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder_out = nn.Linear(config.decoder_hidden_dim, config.feature_dim)

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F), y: (B,)
        y_embed = self.class_embedding(y)  # (B, E)
        y_expand = y_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        enc_in = torch.cat([x, y_expand], dim=-1)
        _, h_n = self.encoder_gru(enc_in)

        # Use the last layer outputs from both directions if bidirectional.
        if self.cfg.encoder_bidirectional:
            h_last_fw = h_n[-2]
            h_last_bw = h_n[-1]
            h_last = torch.cat([h_last_fw, h_last_bw], dim=-1)
        else:
            h_last = h_n[-1]

        mu = self.mu_head(h_last)
        logvar = self.logvar_head(h_last)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        seq_len = self.cfg.seq_len if seq_len is None else int(seq_len)
        y_embed = self.class_embedding(y)
        dec_token = torch.cat([z, y_embed], dim=-1)  # (B, z+E)
        dec_in = dec_token.unsqueeze(1).expand(-1, seq_len, -1)
        dec_out, _ = self.decoder_gru(dec_in)
        return self.decoder_out(dec_out)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y, seq_len=x.size(1))
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, y: torch.Tensor, z: torch.Tensor | None = None, seq_len: int | None = None) -> torch.Tensor:
        if z is None:
            z = torch.randn((y.size(0), self.cfg.z_dim), device=y.device)
        return self.decode(z, y, seq_len=seq_len)


def cvae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta_kl: float,
    velocity_weight: float,
) -> dict[str, torch.Tensor]:
    recon_loss = F.mse_loss(recon, x)

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl.mean()

    real_v = x[:, 1:] - x[:, :-1]
    fake_v = recon[:, 1:] - recon[:, :-1]
    velocity_loss = F.mse_loss(fake_v, real_v)

    total = recon_loss + beta_kl * kl_loss + velocity_weight * velocity_loss
    return {
        "total": total,
        "recon": recon_loss,
        "kl": kl_loss,
        "velocity": velocity_loss,
    }


def filter_synthetic_sequence(seq: np.ndarray, cfg: QualityFilterConfig) -> tuple[bool, str]:
    if not cfg.enabled:
        return True, "disabled"

    if np.isnan(seq).any() or np.isinf(seq).any():
        return False, "nan_or_inf"

    if seq.ndim != 2 or seq.shape[0] < 2:
        return False, "invalid_shape"

    feat_std = float(np.std(seq))
    if feat_std < cfg.min_feature_std:
        return False, "low_variance"

    velocity = seq[1:] - seq[:-1]
    motion_mag = float(np.mean(np.linalg.norm(velocity, axis=1)))
    if motion_mag < cfg.min_motion_magnitude:
        return False, "low_motion"

    jump_mag = float(np.max(np.linalg.norm(velocity, axis=1)))
    if jump_mag > cfg.max_frame_jump:
        return False, "large_jump"

    return True, "ok"


def per_class_stratified_split(
    labels: list[int],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for class_id in sorted(set(labels)):
        idx = np.where(labels_arr == class_id)[0]
        rng.shuffle(idx)
        n_val = max(1, int(math.ceil(len(idx) * val_ratio)))
        class_val = idx[:n_val].tolist()
        class_train = idx[n_val:].tolist()

        # Ensure at least one train sample when possible.
        if not class_train and class_val:
            class_train.append(class_val.pop())

        train_idx.extend(class_train)
        val_idx.extend(class_val)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def save_json(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
