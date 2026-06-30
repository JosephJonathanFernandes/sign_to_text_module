"""Hybrid quality + diversity curation for processed sign-language sequences.

The pipeline is per-class and production-oriented:
- compute per-sample quality scores
- keep the quality-curve budget per class
- build diversity embeddings from a trained classifier when available
- otherwise fall back to normalized landmark trajectory statistics
- suppress near-duplicates with cosine similarity
- greedily pick diverse high-quality samples within a shortlist
- export JSON and CSV metadata for the kept samples

This module is designed to be called from the existing quality_filter_npy.py
entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch is expected but optional
    torch = None
    nn = None


try:
    from src.training.model import SignLanguageGRU
except Exception as e:  # pragma: no cover
    import sys
    print("=" * 80, file=sys.stderr)
    print("[WARNING] Failed to import SignLanguageGRU from src.training.model!", file=sys.stderr)
    print("          You are probably not running this script as a module from the root directory.", file=sys.stderr)
    print("          The script will fall back to basic statistical features instead of deep embeddings.", file=sys.stderr)
    print("          To fix this, run: python -m src.preprocessing.quality_filter_hybrid", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    SignLanguageGRU = None

ROOT_DIR = os.path.join("assets", "processed")
DEFAULT_QUALITY_COVERAGE = 0.85
DEFAULT_SHORTLIST_MULTIPLIER = 3.0
DEFAULT_DUPLICATE_MODE = "relaxed"
DEFAULT_DUPLICATE_THRESHOLD = 0.985
DEFAULT_DUPLICATE_THRESHOLDS = {
    "strict": 0.975,
    "balanced": 0.985,
    "relaxed": 0.992,
}
DEFAULT_QUALITY_POWER = 1.5
DEFAULT_MIN_FILL_RATIO = 0.75
DEFAULT_MIN_SCORE = 0.0
DEFAULT_SEED = 42
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_REPORT_DIR = os.path.join("logs", "quality_filter")
DEFAULT_REPORT_PREFIX = "hybrid_quality_diversity"
DEFAULT_CACHE_DIR = os.path.join("logs", "cache", "quality_filter")
DEFAULT_CACHE_MAX_AGE_HOURS = 24
EMBEDDING_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0, "stale": 0, "saves": 0}
DEFAULT_MIN_CLASS_SAMPLES = 20
EPS = 1e-6
FRAME_FEAT_DIM = 253
RAW_FRAME_FEAT_DIM = 126
LEFT_RAW_DIM = 63
RIGHT_RAW_DIM = 63
REL_DIM = 126
VEL_OFFSET = FRAME_FEAT_DIM


@dataclass
class SampleCandidate:
    original_index: int
    path: str
    sequence: np.ndarray
    quality_score: float
    raw_coverage: float
    rel_coverage: float
    left_coverage: float
    right_coverage: float
    signal_strength: float
    motion_energy: float
    jerk: float
    zero_ratio: float


@dataclass
class SelectedSampleRecord:
    class_name: str
    original_index: int
    path: str
    quality_score: float
    novelty_score: float
    final_score: float
    suppressed_duplicate_count: int
    nearest_similarity: float
    selected_rank: int
    curve_budget: int
    shortlist_rank: int
    shortlist_size: int
    curve_threshold: float
    effective_threshold: float
    effective_duplicate_threshold: float
    embedding_cache_key: str | None = None
    pca_x: float | None = None
    pca_y: float | None = None


@dataclass
class ClassSummary:
    class_name: str
    total_samples: int
    adaptive_budget: int
    curve_threshold: float
    shortlist_size: int
    duplicates_removed: int
    final_kept: int
    average_quality: float
    average_novelty: float
    effective_threshold: float
    redundancy_ratio: float
    average_pairwise_similarity: float
    duplicate_suppression_pct: float
    novelty_p10: float
    novelty_p50: float
    novelty_p90: float
    keep_ratio: float
    health_score: float
    dry_run: bool
    embedding_mode: str
    duplicate_mode: str
    duplicate_threshold: float
    min_fill_ratio: float
    fill_ratio: float
    avg_final_similarity: float
    effective_duplicate_threshold: float
    novelty_weight: float
    adaptive_relaxed: bool
    # Removal breakdown
    removed_by_quality: int
    removed_by_duplicate: int
    removed_by_novelty: int
    # Collapse detection flags (lightweight)
    collapse_signer: bool
    collapse_motion: bool
    collapse_overcompression: bool
    collapse_reason: str


class EmbeddingExtractor:
    """Legacy extraction wrapper."""
    def __init__(self, checkpoint_path: str | None = None, batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE):
        self.batch_size = max(1, int(batch_size))
        self.checkpoint_path = checkpoint_path
        self.model: Any = None
        self.embedding_mode = "fallback"
        self.device = None
        self._try_load_model()

    def _try_load_model(self) -> None:
        if torch is None or SignLanguageGRU is None:
            return

        checkpoint_path = self.checkpoint_path or _resolve_default_checkpoint()
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = _extract_state_dict(checkpoint)
            num_classes = _infer_num_classes(state_dict)
            if num_classes is None:
                return

            model = SignLanguageGRU(num_classes=num_classes)
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model.load_state_dict(state_dict, strict=False)

            model.eval()
            self.model = model
            self.device = torch.device("cpu")
            self.embedding_mode = f"model:{os.path.basename(checkpoint_path)}"
        except Exception as exc:
            print(f"[Embed] Falling back to handcrafted embeddings: {exc}")
            self.model = None
            self.device = None
            self.embedding_mode = "fallback"

    def encode(self, sequences: np.ndarray) -> np.ndarray:
        if sequences.ndim != 3:
            raise ValueError(f"Expected 3D batch (N, T, D), got shape {sequences.shape}")

        if self.model is not None and torch is not None:
            try:
                return self._encode_with_model(sequences)
            except Exception as exc:
                print(f"[Embed] Model embeddings failed; using fallback features: {exc}")
                self.model = None
                self.embedding_mode = "fallback"

        return _fallback_embeddings(sequences)

    def _encode_with_model(self, sequences: np.ndarray) -> np.ndarray:
        assert torch is not None
        assert self.model is not None

        capture: list[np.ndarray] = []

        def _capture_pre_fc(_module, inputs):
            capture.append(inputs[0].detach().cpu().numpy())

        handle = None
        if hasattr(self.model, "fc") and len(self.model.fc) > 0:
            handle = self.model.fc[0].register_forward_pre_hook(_capture_pre_fc)

        self.model.to(self.device)
        batch_list = []
        for start in range(0, len(sequences), self.batch_size):
            end = min(start + self.batch_size, len(sequences))
            batch = torch.from_numpy(np.asarray(sequences[start:end], dtype=np.float32)).to(self.device)
            with torch.inference_mode():
                _ = self.model(batch)
            batch_list.append(capture.pop(0))

        if handle is not None:
            handle.remove()

        if not batch_list:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings = np.concatenate(batch_list, axis=0).astype(np.float32)
        return _normalize_embeddings(embeddings)


def _resolve_default_checkpoint() -> str | None:
    candidates = [
        os.path.abspath("model.pth"),
        os.path.abspath(os.path.join("ensemble", "model.pth")),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def _infer_num_classes(state_dict: dict[str, Any]) -> int | None:
    key_candidates = ["fc.3.weight", "classifier.weight", "head.weight", "logits.weight"]
    for key in key_candidates:
        value = state_dict.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    for key, value in state_dict.items():
        if key.endswith("weight") and hasattr(value, "shape") and len(value.shape) == 2 and value.shape[0] > 1:
            if "fc.3" in key or "classifier" in key or "head" in key or "logits" in key:
                return int(value.shape[0])
    return None


def _list_class_dirs(root_dir: str) -> list[str]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {os.path.abspath(root_dir)}")
    return sorted(
        os.path.join(root_dir, entry)
        for entry in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, entry))
    )


def _list_npy_files(class_dir: str) -> list[str]:
    return [
        os.path.join(class_dir, entry)
        for entry in sorted(os.listdir(class_dir))
        if entry.lower().endswith(".npy") and os.path.isfile(os.path.join(class_dir, entry))
    ]


def _normalize_class_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _safe_delete(path: str, class_dir: str, dry_run: bool, archive_root: str | None = None) -> bool:
    path_abs = os.path.normcase(os.path.abspath(path))
    class_abs = os.path.normcase(os.path.abspath(class_dir))

    if not path_abs.lower().endswith(".npy"):
        return False
    if os.path.normcase(os.path.dirname(path_abs)) != class_abs:
        return False
    if not os.path.isfile(path_abs):
        return False

    if not dry_run:
        try:
            if archive_root:
                archive_class_dir = os.path.join(os.path.abspath(archive_root), os.path.basename(class_dir))
                os.makedirs(archive_class_dir, exist_ok=True)
                archive_path = os.path.join(archive_class_dir, os.path.basename(path_abs))
                shutil.copy2(path_abs, archive_path)
            os.remove(path_abs)
        except OSError as e:
            print(f"    [WARN] Failed to delete or archive {path_abs}: {e}")
            return False
    return True


def _load_sequence(path: str) -> np.ndarray:
    seq = np.load(path, allow_pickle=False).astype(np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got shape {seq.shape}")
    if seq.shape[1] < FRAME_FEAT_DIM:
        raise ValueError(f"Expected at least {FRAME_FEAT_DIM} feature dims, got {seq.shape[1]}")
    return np.nan_to_num(seq, copy=False)


def _split_features(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = seq[:, :FRAME_FEAT_DIM]
    vel = seq[:, VEL_OFFSET:VEL_OFFSET + FRAME_FEAT_DIM] if seq.shape[1] >= FRAME_FEAT_DIM * 2 else np.zeros_like(base)
    return base, vel


def _frame_norms(block: np.ndarray) -> np.ndarray:
    return np.linalg.norm(block, axis=1)


def _compute_metrics(seq: np.ndarray) -> dict[str, float]:
    base, vel = _split_features(seq)

    raw = base[:, :RAW_FRAME_FEAT_DIM]
    rel = base[:, RAW_FRAME_FEAT_DIM:RAW_FRAME_FEAT_DIM + REL_DIM]
    raw_left = raw[:, :LEFT_RAW_DIM]
    raw_right = raw[:, LEFT_RAW_DIM:LEFT_RAW_DIM + RIGHT_RAW_DIM]

    raw_frame_norm = _frame_norms(raw)
    rel_frame_norm = _frame_norms(rel)
    left_frame_norm = _frame_norms(raw_left)
    right_frame_norm = _frame_norms(raw_right)
    vel_frame_norm = _frame_norms(vel)

    if seq.shape[0] > 1:
        vel_jitter = np.linalg.norm(np.diff(vel, axis=0), axis=1)
        jerk = float(np.mean(vel_jitter))
    else:
        jerk = 0.0

    motion_energy = float(np.mean(vel_frame_norm))
    signal_strength = float(np.mean(raw_frame_norm) + np.mean(rel_frame_norm))

    return {
        "raw_coverage": float(np.mean(raw_frame_norm > 5e-4)),
        "rel_coverage": float(np.mean(rel_frame_norm > 5e-4)),
        "left_coverage": float(np.mean(left_frame_norm > 5e-4)),
        "right_coverage": float(np.mean(right_frame_norm > 5e-4)),
        "signal_strength": signal_strength,
        "motion_energy": motion_energy,
        "jerk": jerk,
        "zero_ratio": float(np.mean(np.abs(seq) <= EPS)),
    }


def _median_and_mad(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 1.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return median, max(mad, 1e-6)


def _robust_similarity(value: float, median: float, mad: float) -> float:
    z = abs(value - median) / max(mad, 1e-6)
    return float(1.0 / (1.0 + z))


def _score_quality(metrics: dict[str, float], class_stats: dict[str, tuple[float, float]]) -> float:
    coverage = 0.5 * metrics["raw_coverage"] + 0.5 * metrics["rel_coverage"]
    presence = max(metrics["left_coverage"], metrics["right_coverage"])

    signal_med, signal_mad = class_stats["signal_strength"]
    motion_med, motion_mad = class_stats["motion_energy"]
    jerk_med, jerk_mad = class_stats["jerk"]

    signal_score = _robust_similarity(metrics["signal_strength"], signal_med, signal_mad)
    motion_score = _robust_similarity(metrics["motion_energy"], motion_med, motion_mad)
    smooth_score = _robust_similarity(metrics["jerk"], jerk_med, jerk_mad)

    score = (
        0.35 * coverage
        + 0.10 * presence
        + 0.20 * signal_score
        + 0.20 * motion_score
        + 0.15 * smooth_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _build_candidates(file_paths: list[str]) -> tuple[list[SampleCandidate], list[str]]:
    metrics_list: list[dict[str, float]] = []
    loaded: list[tuple[int, str, np.ndarray, dict[str, float]]] = []
    unreadable_paths: list[str] = []

    for idx, path in enumerate(file_paths):
        try:
            seq = _load_sequence(path)
            metrics = _compute_metrics(seq)
            loaded.append((idx, path, seq, metrics))
            metrics_list.append(metrics)
        except Exception as exc:
            print(f"    [WARN] Skipping unreadable file: {path} ({exc})")
            unreadable_paths.append(path)

        if (idx + 1) % 200 == 0:
            print(f"    Loaded {idx + 1}/{len(file_paths)} files...")

    if not loaded:
        return [], unreadable_paths

    class_stats = {
        "signal_strength": _median_and_mad([m["signal_strength"] for m in metrics_list]),
        "motion_energy": _median_and_mad([m["motion_energy"] for m in metrics_list]),
        "jerk": _median_and_mad([m["jerk"] for m in metrics_list]),
    }

    candidates: list[SampleCandidate] = []
    for idx, path, seq, metrics in loaded:
        candidates.append(
            SampleCandidate(
                original_index=idx,
                path=path,
                sequence=seq,
                quality_score=_score_quality(metrics, class_stats),
                raw_coverage=metrics["raw_coverage"],
                rel_coverage=metrics["rel_coverage"],
                left_coverage=metrics["left_coverage"],
                right_coverage=metrics["right_coverage"],
                signal_strength=metrics["signal_strength"],
                motion_energy=metrics["motion_energy"],
                jerk=metrics["jerk"],
                zero_ratio=metrics["zero_ratio"],
            )
        )

    return candidates, unreadable_paths


def _estimate_keep_from_curve(scores: list[float], coverage_fraction: float) -> tuple[int, float | None]:
    if not scores:
        return 0, None
    ranked = sorted((float(score) for score in scores), reverse=True)
    if coverage_fraction <= 0.0:
        return 1, float(ranked[0])
    if coverage_fraction >= 1.0:
        return len(ranked), float(ranked[-1])

    total = float(sum(ranked))
    if total <= 1e-12:
        return len(ranked), float(ranked[-1])

    cumulative = np.cumsum(ranked)
    target = coverage_fraction * total
    keep_count = int(np.searchsorted(cumulative, target, side="left") + 1)
    keep_count = max(1, min(keep_count, len(ranked)))
    return keep_count, float(ranked[keep_count - 1])


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.size == 0:
        return embeddings.reshape(0, 0)
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = embeddings / norms
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    return normalized.astype(np.float32)


def _hash_paths_for_cache(paths: list[str], checkpoint_path: str | None, embedding_mode: str) -> str:
    digest = hashlib.sha1()
    digest.update(embedding_mode.encode("utf-8"))
    if checkpoint_path and os.path.isfile(checkpoint_path):
        stat = os.stat(checkpoint_path)
        digest.update(os.path.abspath(checkpoint_path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    for path in paths:
        try:
            stat = os.stat(path)
            digest.update(os.path.abspath(path).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
        except OSError:
            digest.update(os.path.abspath(path).encode("utf-8"))
            digest.update(b"missing_or_corrupted")
    return digest.hexdigest()


def _class_cache_dir(cache_root: str, class_name: str) -> str:
    return os.path.join(cache_root, _normalize_class_token(class_name))


def _cache_file_paths(cache_root: str, class_name: str, cache_key: str) -> tuple[str, str]:
    class_dir = _class_cache_dir(cache_root, class_name)
    return (
        os.path.join(class_dir, f"{cache_key}.npz"),
        os.path.join(class_dir, f"{cache_key}.json"),
    )


def _stale_cache(metadata: dict[str, Any], cache_key: str) -> bool:
    """Check if cached data is too old or corrupted."""
    if metadata.get("cache_key") != cache_key:
        return True
    generated_at = metadata.get("generated_at")
    if generated_at:
        try:
            generated = datetime.fromisoformat(generated_at)
            # Make comparison timezone-naive if the generated timestamp is naive
            now = datetime.now()
            if generated.tzinfo is not None:
                generated = generated.replace(tzinfo=None)
            age_hours = (now - generated).total_seconds() / 3600.0
            if age_hours > DEFAULT_CACHE_MAX_AGE_HOURS:
                return True
        except (ValueError, TypeError):
            return True
    return False


def _load_embedding_cache(cache_root: str, class_name: str, cache_key: str) -> dict[str, Any] | None:
    npz_path, json_path = _cache_file_paths(cache_root, class_name, cache_key)
    if not os.path.isfile(npz_path) or not os.path.isfile(json_path):
        try:
            EMBEDDING_CACHE_STATS["misses"] += 1
        except Exception:
            pass
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if _stale_cache(metadata, cache_key):
            try:
                EMBEDDING_CACHE_STATS["stale"] += 1
            except Exception:
                pass
            # Remove stale cache files silently
            try:
                os.remove(npz_path)
                os.remove(json_path)
            except OSError:
                pass
            return None
        with np.load(npz_path, allow_pickle=False) as data:
            embeddings = data["embeddings"].astype(np.float32)
            paths = data["paths"].astype(object).tolist()

        # Validate internal consistency
        if embeddings.shape[0] != len(paths):
            try:
                os.remove(npz_path)
                os.remove(json_path)
            except OSError:
                pass
            return None

        try:
            EMBEDDING_CACHE_STATS["hits"] += 1
        except Exception:
            pass
        return {"metadata": metadata, "embeddings": embeddings, "paths": paths}
    except Exception:
        # Corrupted cache → remove and recompute
        try:
            EMBEDDING_CACHE_STATS["stale"] += 1
        except Exception:
            pass
        try:
            os.remove(npz_path)
            os.remove(json_path)
        except OSError:
            pass
        return None


def _save_embedding_cache(
    cache_dir: str,
    class_name: str,
    cache_key: str,
    paths: list[str],
    embeddings: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    npz_path, json_path = _cache_file_paths(cache_dir, class_name, cache_key)
    try:
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez_compressed(
            npz_path,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            paths=np.asarray(paths, dtype=object),
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except OSError as e:
        print(f"    [WARN] Failed to save embedding cache: {e}")
    try:
        EMBEDDING_CACHE_STATS["saves"] += 1
    except Exception:
        pass


def _safe_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    embeddings = _normalize_embeddings(embeddings)
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return np.eye(embeddings.shape[0], dtype=np.float32) if embeddings.ndim == 2 and embeddings.shape[0] == 1 else np.zeros((0, 0), dtype=np.float32)
    sim = embeddings @ embeddings.T
    sim = np.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(sim, -1.0, 1.0)


def _sampled_average_pairwise_similarity(embeddings: np.ndarray, max_samples: int = 64) -> float:
    if embeddings.shape[0] < 2:
        return 0.0
    sample = embeddings[: min(max_samples, embeddings.shape[0])]
    sim = _safe_similarity_matrix(sample)
    tri = sim[np.triu_indices(sim.shape[0], k=1)]
    if tri.size == 0:
        return 0.0
    return float(np.mean(tri))


def _compute_novelty_distribution(novelty_scores: list[float]) -> dict[str, float]:
    if not novelty_scores:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "std": 0.0}
    arr = np.asarray(novelty_scores, dtype=np.float32)
    if arr.size == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "std": 0.0}
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(np.std(arr)),
    }


def _pca_coordinates(embeddings: np.ndarray, dims: int = 2) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, dims), dtype=np.float32)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vh[:dims].T
        return coords.astype(np.float32)
    except Exception:
        return np.zeros((embeddings.shape[0], dims), dtype=np.float32)


def _class_health_score(
    average_quality: float,
    average_novelty: float,
    redundancy_ratio: float,
    keep_ratio: float,
    novelty_spread: float,
) -> float:
    # Guard against NaN / Inf degenerate inputs
    vals = [average_quality, average_novelty, redundancy_ratio, keep_ratio, novelty_spread]
    vals = [0.0 if not (isinstance(v, (int, float)) and math.isfinite(v)) else v for v in vals]
    average_quality, average_novelty, redundancy_ratio, keep_ratio, novelty_spread = vals

    score = (
        34.0 * np.clip(average_quality, 0.0, 1.0)
        + 24.0 * np.clip(average_novelty, 0.0, 1.0)
        + 18.0 * np.clip(1.0 - redundancy_ratio, 0.0, 1.0)
        + 14.0 * np.clip(keep_ratio, 0.0, 1.0)
        + 10.0 * np.clip(novelty_spread, 0.0, 1.0)
    )
    return float(np.clip(score, 0.0, 100.0))


def _duplicate_mode_threshold(duplicate_mode: str, explicit_threshold: float | None = None) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)
    return float(DEFAULT_DUPLICATE_THRESHOLDS.get(duplicate_mode, DEFAULT_DUPLICATE_THRESHOLDS[DEFAULT_DUPLICATE_MODE]))


def _fallback_embeddings(sequences: np.ndarray) -> np.ndarray:
    seq = np.asarray(sequences, dtype=np.float32)
    seq = np.nan_to_num(seq, copy=False)

    base = seq[:, :, :FRAME_FEAT_DIM]
    raw = base[:, :, :RAW_FRAME_FEAT_DIM]
    rel = base[:, :, RAW_FRAME_FEAT_DIM:RAW_FRAME_FEAT_DIM + REL_DIM]
    vel = np.zeros_like(base)
    vel[:, 1:] = base[:, 1:] - base[:, :-1]
    acc = np.zeros_like(vel)
    acc[:, 1:] = vel[:, 1:] - vel[:, :-1]

    def _block_stats(block: np.ndarray) -> list[np.ndarray]:
        return [
            block.mean(axis=1),
            block.std(axis=1),
            block.min(axis=1),
            block.max(axis=1),
            np.percentile(block, 25, axis=1),
            np.percentile(block, 75, axis=1),
        ]

    def _motion_stats(block: np.ndarray) -> list[np.ndarray]:
        norm = np.linalg.norm(block, axis=2)
        return [
            norm.mean(axis=1)[:, None],
            norm.std(axis=1)[:, None],
            norm.max(axis=1)[:, None],
        ]

    features: list[np.ndarray] = []
    for block in (raw, rel, base, vel, acc):
        features.extend(_block_stats(block))
    features.extend(_motion_stats(vel))
    features.extend(_motion_stats(acc))

    first_last = np.concatenate([
        base[:, 0, :],
        base[:, -1, :],
        base[:, -1, :] - base[:, 0, :],
    ], axis=1)
    center = base.mean(axis=1)
    spread = np.concatenate([np.ptp(raw, axis=1), np.ptp(rel, axis=1)], axis=1)
    temporal_half = base[:, base.shape[1] // 2, :] - base[:, 0, :]
    late_delta = base[:, -1, :] - base[:, base.shape[1] // 2, :]
    flattened = seq.reshape(seq.shape[0], -1)
    flattened_centered = flattened - flattened.mean(axis=1, keepdims=True)

    features.extend([first_last, center, spread, temporal_half, late_delta, flattened, flattened_centered])
    embedding = np.concatenate([np.asarray(feat, dtype=np.float32) for feat in features], axis=1)
    return _normalize_embeddings(embedding)


class DiversityEmbedder:
    def __init__(self, checkpoint_path: str | None = None, batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE):
        self.batch_size = max(1, int(batch_size))
        self.checkpoint_path = checkpoint_path
        self.embedding_mode = "fallback"
        self.model: Any = None
        self.device = None
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        if torch is None or SignLanguageGRU is None:
            return

        checkpoint_path = self.checkpoint_path or _resolve_default_checkpoint()
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = _extract_state_dict(checkpoint)
            num_classes = _infer_num_classes(state_dict)
            if num_classes is None:
                return

            model = SignLanguageGRU(num_classes=num_classes)
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model.load_state_dict(state_dict, strict=False)

            model.eval()
            self.model = model
            self.device = torch.device("cpu")
            self.embedding_mode = f"model:{os.path.basename(checkpoint_path)}"
        except Exception as exc:
            print(f"[Embed] Falling back to handcrafted embeddings: {exc}")
            self.model = None
            self.device = None
            self.embedding_mode = "fallback"

    def encode(self, sequences: np.ndarray) -> np.ndarray:
        if sequences.ndim != 3:
            raise ValueError(f"Expected 3D batch (N, T, D), got shape {sequences.shape}")

        if self.model is not None and torch is not None:
            try:
                return self._encode_with_model(sequences)
            except Exception as exc:
                print(f"[Embed] Model embeddings failed; using fallback features: {exc}")
                self.model = None
                self.embedding_mode = "fallback"

        return _fallback_embeddings(sequences)

    def _encode_with_model(self, sequences: np.ndarray) -> np.ndarray:
        assert torch is not None
        assert self.model is not None

        capture: list[np.ndarray] = []

        def _capture_pre_fc(_module, inputs):
            capture.append(inputs[0].detach().cpu().numpy())

        handle = None
        if hasattr(self.model, "fc") and len(self.model.fc) > 0:
            handle = self.model.fc[0].register_forward_pre_hook(_capture_pre_fc)

        self.model.to(self.device)
        batch_outputs: list[np.ndarray] = []
        for start in range(0, len(sequences), self.batch_size):
            end = min(start + self.batch_size, len(sequences))
            batch = torch.from_numpy(np.asarray(sequences[start:end], dtype=np.float32)).to(self.device)
            with torch.inference_mode():
                _ = self.model(batch)
            if not capture:
                raise RuntimeError("Failed to capture penultimate embeddings")
            batch_outputs.append(capture.pop(0))

        if handle is not None:
            handle.remove()

        embeddings = np.concatenate(batch_outputs, axis=0).astype(np.float32)
        fallback = _fallback_embeddings(sequences)
        combined = np.concatenate([embeddings, fallback], axis=1).astype(np.float32)
        return _normalize_embeddings(combined)


def _select_hybrid_subset(
    candidates: list[SampleCandidate],
    embeddings: np.ndarray,
    budget: int,
    duplicate_threshold: float,
    quality_power: float,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
) -> tuple[list[SelectedSampleRecord], dict[str, Any]]:
    shortlist_size = len(candidates)
    if not candidates or budget <= 0:
        return [], {
            "duplicates_removed": 0,
            "avg_quality": 0.0,
            "avg_novelty": 0.0,
            "avg_pairwise_similarity": 0.0,
            "duplicate_suppression_pct": 0.0,
            "novelty_stats": {"p10": 0.0, "p50": 0.0, "p90": 0.0, "std": 0.0},
            "effective_duplicate_threshold": duplicate_threshold,
            "selected_indices": [],
            "selected_embeddings": np.zeros((0, 0), dtype=np.float32),
            "novelty_weight": 0.5,
            "adaptive_relaxed": False,
        }

    # Sanitize quality scores.
    qualities = np.asarray([candidate.quality_score for candidate in candidates], dtype=np.float32)
    qualities = np.clip(np.nan_to_num(qualities, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    # If all qualities are degenerate, default to a uniform distribution.
    if not np.any(np.isfinite(qualities)) or np.max(qualities) <= 0.0:
        qualities = np.full(shortlist_size, 0.5, dtype=np.float32)

    # Compute similarity matrix; if too few candidates, skip similarity logic.
    if shortlist_size < 2 or embeddings.size == 0 or embeddings.shape[0] < 2:
        sim = np.eye(shortlist_size, dtype=np.float32) if shortlist_size > 0 else np.zeros((0, 0), dtype=np.float32)
    else:
        sim = _safe_similarity_matrix(embeddings)
        if sim.shape != (shortlist_size, shortlist_size):
            sim = np.zeros((shortlist_size, shortlist_size), dtype=np.float32)

    selected_indices: list[int] = []
    selected_mask = np.zeros(shortlist_size, dtype=bool)
    max_sim_to_selected = np.zeros(shortlist_size, dtype=np.float32)
    suppression_counts = np.zeros(shortlist_size, dtype=np.int32)
    novelty_scores = np.zeros(shortlist_size, dtype=np.float32)
    final_scores = np.zeros(shortlist_size, dtype=np.float32)
    nearest_similarities = np.zeros(shortlist_size, dtype=np.float32)
    selected_duplicate_threshold = float(duplicate_threshold)
    adaptive_relaxed = False
    # Class-adaptive novelty weight: start at 0.5 (standard), reduce to 0.2 if
    # the diversity pipeline is suppressing too aggressively for this class.
    novelty_weight = 0.5

    def _score_all(current_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        novelty = np.clip(1.0 - max_sim_to_selected, 0.0, 1.0)
        # Use adaptive novelty_weight: lower weight = quality dominates more.
        scores = np.power(qualities, quality_power) * (1.0 - novelty_weight + novelty_weight * novelty)
        scores[current_mask] = -np.inf
        return scores, novelty

    def _pick_next(threshold: float) -> bool:
        current_mask = selected_mask.copy()
        eligible = (~current_mask) & (max_sim_to_selected <= threshold)
        if not np.any(eligible):
            return False
        # Compute scores only for eligible candidates (incremental optimization).
        eligible_scores = np.full(shortlist_size, -np.inf, dtype=np.float32)
        eligible_scores[eligible] = (
            np.power(qualities[eligible], quality_power)
            * (1.0 - novelty_weight + novelty_weight * np.clip(1.0 - max_sim_to_selected[eligible], 0.0, 1.0))
        )
        chosen = int(np.argmax(eligible_scores))
        if chosen < 0 or not np.isfinite(eligible_scores[chosen]):
            return False

        nearest_similarity = float(max_sim_to_selected[chosen]) if selected_indices else 0.0
        selected_indices.append(chosen)
        selected_mask[chosen] = True
        novelty_scores[chosen] = float(np.clip(1.0 - max_sim_to_selected[chosen], 0.0, 1.0))
        final_scores[chosen] = float(eligible_scores[chosen])
        nearest_similarities[chosen] = nearest_similarity

        # Incrementally update max-sim and suppression counts only for affected entries.
        new_sims = sim[chosen]
        suppress_mask = (~selected_mask) & (new_sims > threshold)
        suppression_counts[chosen] += int(np.count_nonzero(suppress_mask))
        max_sim_to_selected[:] = np.maximum(max_sim_to_selected, new_sims)
        max_sim_to_selected[selected_mask] = 1.0
        return True

    # Initial strict selection pass.
    selected_duplicate_threshold = float(duplicate_threshold)
    while len(selected_indices) < min(budget, shortlist_size):
        if not _pick_next(selected_duplicate_threshold):
            break

    # -- Class-adaptive novelty reduction --
    # If after the initial pass the fill ratio is below 0.70, reduce the novelty
    # weight so that quality dominates the scoring. This retains more samples
    # from repetitive classes without weakening the duplicate threshold globally.
    min_fill_target = max(1, min(budget, int(math.ceil(float(min_fill_ratio) * budget))))
    current_fill = len(selected_indices) / max(1, budget)
    if current_fill < 0.70 and budget > 10:
        # Reduce novelty weight progressively: the worse the fill, the more we
        # let quality drive. Range: 0.2 (very repetitive) to 0.45 (mild).
        reduction = max(0.05, min(0.30, (0.70 - current_fill) * 0.6))
        novelty_weight = max(0.20, 0.5 - reduction)
        adaptive_relaxed = True
        # Re-score remaining candidates with reduced novelty weight.
        while len(selected_indices) < min_fill_target:
            if not _pick_next(selected_duplicate_threshold):
                break

    # Gradually relax if the class still underfills too much.
    relax_step = 0.0025
    relax_cap = 0.9995
    while len(selected_indices) < min_fill_target and selected_duplicate_threshold < relax_cap:
        selected_duplicate_threshold = min(relax_cap, selected_duplicate_threshold + relax_step)
        progress = False
        while len(selected_indices) < min_fill_target:
            if not _pick_next(selected_duplicate_threshold):
                break
            progress = True
        if not progress:
            break

    # Last resort: if still underfilled and no eligible candidates remain, pick best-quality regardless.
    if len(selected_indices) < min_fill_target and shortlist_size > 0:
        remaining = [i for i in range(shortlist_size) if not selected_mask[i]]
        remaining_qualities = [qualities[i] for i in remaining]
        while len(selected_indices) < min_fill_target and remaining:
            best_i = int(np.argmax(remaining_qualities))
            idx = remaining.pop(best_i)
            remaining_qualities.pop(best_i)
            if idx >= 0:
                selected_indices.append(idx)
                selected_mask[idx] = True
                novelty_scores[idx] = 0.0
                final_scores[idx] = float(qualities[idx])
                nearest_similarities[idx] = float(max_sim_to_selected[idx])

    selected_embeddings = embeddings[np.asarray(selected_indices, dtype=np.int64)] if selected_indices else np.zeros((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32)
    selected_records: list[SelectedSampleRecord] = []
    effective_threshold = float(min(final_scores[selected_indices])) if selected_indices else 0.0

    for rank, idx in enumerate(selected_indices, start=1):
        candidate = candidates[idx]
        selected_records.append(
            SelectedSampleRecord(
                class_name="",
                original_index=candidate.original_index,
                path=candidate.path,
                quality_score=float(candidate.quality_score),
                novelty_score=float(novelty_scores[idx]),
                final_score=float(final_scores[idx]),
                suppressed_duplicate_count=int(suppression_counts[idx]),
                nearest_similarity=float(nearest_similarities[idx]),
                selected_rank=rank,
                curve_budget=budget,
                shortlist_rank=idx + 1,
                shortlist_size=shortlist_size,
                curve_threshold=0.0,
                effective_threshold=effective_threshold,
                effective_duplicate_threshold=selected_duplicate_threshold,
            )
        )

    duplicates_removed = int(np.sum(suppression_counts[selected_indices])) if selected_indices else 0
    avg_quality = float(np.mean([record.quality_score for record in selected_records])) if selected_records else 0.0
    avg_novelty = float(np.mean([record.novelty_score for record in selected_records])) if selected_records else 0.0
    novelty_stats = _compute_novelty_distribution([record.novelty_score for record in selected_records])
    avg_pairwise_similarity = _sampled_average_pairwise_similarity(selected_embeddings)
    duplicate_suppression_pct = float(duplicates_removed / max(1, shortlist_size))

    return selected_records, {
        "duplicates_removed": duplicates_removed,
        "avg_quality": avg_quality,
        "avg_novelty": avg_novelty,
        "avg_pairwise_similarity": avg_pairwise_similarity,
        "duplicate_suppression_pct": duplicate_suppression_pct,
        "novelty_stats": novelty_stats,
        "effective_duplicate_threshold": selected_duplicate_threshold,
        "selected_indices": selected_indices,
        "selected_embeddings": selected_embeddings,
        "selected_mask": selected_mask,
        "suppression_counts": suppression_counts,
        "nearest_similarities": nearest_similarities,
        "final_scores": final_scores,
        "novelty_weight": novelty_weight,
        "adaptive_relaxed": adaptive_relaxed,
    }


def filter_quality_class_folder(
    class_dir: str,
    auto_keep: bool,
    keep_limit: int,
    quality_coverage: float,
    shortlist_multiplier: float,
    duplicate_mode: str,
    duplicate_threshold: float | None,
    quality_power: float,
    embedder: DiversityEmbedder,
    min_fill_ratio: float,
    embedding_cache: bool,
    max_keep_per_class: int | None = None,
    min_class_samples: int | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    export_viz_data: bool = False,
    viz_pca_dims: int = 2,
    dry_run: bool = False,
    min_score: float = DEFAULT_MIN_SCORE,
    archive_root: str | None = None,
) -> tuple[ClassSummary, list[SelectedSampleRecord], dict[str, Any]]:
    files = _list_npy_files(class_dir)
    total = len(files)
    class_name = os.path.basename(class_dir)

    candidates, unreadable_paths = _build_candidates(files)
    if not candidates:
        print(f"[{class_name}] no valid samples found")
        summary = ClassSummary(
            class_name=class_name,
            total_samples=total,
            adaptive_budget=0,
            curve_threshold=0.0,
            shortlist_size=0,
            duplicates_removed=0,
            final_kept=0,
            average_quality=0.0,
            average_novelty=0.0,
            effective_threshold=0.0,
            redundancy_ratio=0.0,
            average_pairwise_similarity=0.0,
            duplicate_suppression_pct=0.0,
            novelty_p10=0.0,
            novelty_p50=0.0,
            novelty_p90=0.0,
            keep_ratio=0.0,
            health_score=0.0,
            dry_run=dry_run,
            embedding_mode="fallback",
            duplicate_mode=duplicate_mode,
            duplicate_threshold=_duplicate_mode_threshold(duplicate_mode, duplicate_threshold),
            min_fill_ratio=min_fill_ratio,
            fill_ratio=0.0,
            avg_final_similarity=0.0,
            effective_duplicate_threshold=_duplicate_mode_threshold(duplicate_mode, duplicate_threshold),
            novelty_weight=0.5,
            adaptive_relaxed=False,
            removed_by_quality=0,
            removed_by_duplicate=0,
            removed_by_novelty=0,
            collapse_signer=False,
            collapse_motion=False,
            collapse_overcompression=False,
            collapse_reason="none",
        )
        return summary, [], {"rows": [], "embeddings": np.zeros((0, 0), dtype=np.float32), "pca_coords": np.zeros((0, 0), dtype=np.float32), "paths": []}

    if min_score > 0.0:
        candidates = [candidate for candidate in candidates if candidate.quality_score >= min_score] or candidates

    qualities = [candidate.quality_score for candidate in sorted(candidates, key=lambda item: item.quality_score, reverse=True)]
    curve_budget, curve_threshold = _estimate_keep_from_curve(qualities, quality_coverage)
    if not auto_keep:
        budget = max(0, min(keep_limit, len(candidates)))
    else:
        budget = min(curve_budget, len(candidates))

    # Enforce minimum samples per class safeguard.
    if min_class_samples is not None:
        try:
            min_class_samples = int(min_class_samples)
            if min_class_samples > 0 and budget < min_class_samples:
                print(f"    [INFO] Raising budget for class '{class_name}' from {budget} to min_class_samples={min_class_samples} to protect rare classes")
            budget = max(budget, min_class_samples)
        except Exception:
            pass

    # Enforce per-class cap to limit CPU/memory when processing large corpora.
    if max_keep_per_class is not None:
        try:
            max_keep_per_class = int(max_keep_per_class)
            # If the cap conflicts with the minimum, honor the minimum and adjust the cap upward
            if min_class_samples is not None and max_keep_per_class < min_class_samples:
                print(f"    [WARN] --max-keep-per-class={max_keep_per_class} is less than --min-class-samples={min_class_samples}; raising cap to honor minimum")
                max_keep_per_class = min_class_samples
            if budget > max_keep_per_class:
                print(f"    [INFO] Capping budget for class '{class_name}' from {budget} to {max_keep_per_class} to respect --max-keep-per-class")
            budget = min(budget, max_keep_per_class)
        except Exception:
            pass

    shortlist_size = min(len(candidates), max(1, int(math.ceil(shortlist_multiplier * budget))))
    shortlist_size = min(shortlist_size, len(candidates))
    shortlisted = sorted(candidates, key=lambda item: item.quality_score, reverse=True)[:shortlist_size]

    effective_duplicate_threshold = _duplicate_mode_threshold(duplicate_mode, duplicate_threshold)
    cache_key = _hash_paths_for_cache(
        [candidate.path for candidate in shortlisted],
        embedder.checkpoint_path,
        f"{embedder.embedding_mode}|{duplicate_mode}|{effective_duplicate_threshold:.4f}|{len(shortlisted)}",
    )
    embeddings = None
    if embedding_cache:
        cached = _load_embedding_cache(cache_dir, class_name, cache_key)
        if cached is not None and cached.get("embeddings") is not None:
            embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
            # Validate lazy reload: if shape doesn't match shortlisted count, recompute.
            if embeddings.shape[0] != len(shortlisted):
                embeddings = None
    if embeddings is None:
        if len(shortlisted) == 0:
            embeddings = np.zeros((0, 0), dtype=np.float32)
        else:
            embeddings = embedder.encode(np.stack([candidate.sequence for candidate in shortlisted], axis=0).astype(np.float32))
        if embedding_cache and embeddings.size > 0:
            _save_embedding_cache(
                cache_dir=cache_dir,
                class_name=class_name,
                cache_key=cache_key,
                paths=[candidate.path for candidate in shortlisted],
                embeddings=embeddings,
                metadata={
                    "cache_key": cache_key,
                    "class_name": class_name,
                    "embedding_mode": embedder.embedding_mode,
                    "duplicate_mode": duplicate_mode,
                    "effective_duplicate_threshold": effective_duplicate_threshold,
                    "shortlist_size": len(shortlisted),
                    "generated_at": datetime.now().isoformat(),
                },
            )

    selected_records, analytics = _select_hybrid_subset(
        shortlisted,
        embeddings,
        budget=budget,
        duplicate_threshold=effective_duplicate_threshold,
        quality_power=quality_power,
        min_fill_ratio=min_fill_ratio,
    )

    for record in selected_records:
        record.class_name = class_name
        record.curve_threshold = float(curve_threshold or 0.0)
        record.embedding_cache_key = cache_key

    selected_indices = analytics.get("selected_indices", [])
    pca_coords = _pca_coordinates(embeddings, dims=viz_pca_dims) if export_viz_data and viz_pca_dims > 0 and embeddings.size > 0 else None
    viz_rows: list[dict[str, Any]] = []
    if export_viz_data:
        selected_index_set = set(selected_indices)
        for shortlist_rank, candidate in enumerate(shortlisted, start=1):
            row: dict[str, Any] = {
                "class_name": class_name,
                "original_index": candidate.original_index,
                "path": candidate.path,
                "quality_score": float(candidate.quality_score),
                "shortlist_rank": shortlist_rank,
                "selected": shortlist_rank - 1 in selected_index_set,
                "embedding_cache_key": cache_key,
            }
            if shortlist_rank - 1 in selected_index_set:
                selected_pos = selected_indices.index(shortlist_rank - 1)
                row["novelty_score"] = float(selected_records[selected_pos].novelty_score)
                row["final_score"] = float(selected_records[selected_pos].final_score)
            if pca_coords is not None and shortlist_rank - 1 < len(pca_coords):
                row["pca_x"] = float(pca_coords[shortlist_rank - 1, 0])
                row["pca_y"] = float(pca_coords[shortlist_rank - 1, 1]) if pca_coords.shape[1] > 1 else 0.0
            viz_rows.append(row)

    selected_paths = {record.path for record in selected_records}
    delete_set = {candidate.path for candidate in candidates if candidate.path not in selected_paths}
    delete_set.update(unreadable_paths)

    deleted = 0
    for path in delete_set:
        if _safe_delete(path, class_dir, dry_run, archive_root=archive_root):
            deleted += 1

    final_kept = len(selected_records)
    fill_ratio = float(final_kept / max(1, budget))
    redundancy_ratio = float(analytics.get("duplicate_suppression_pct", 0.0))
    avg_pairwise_similarity = float(analytics.get("avg_pairwise_similarity", 0.0))
    novelty_stats = analytics.get("novelty_stats", {"p10": 0.0, "p50": 0.0, "p90": 0.0, "std": 0.0})
    novelty_spread = float(max(0.0, novelty_stats.get("p90", 0.0) - novelty_stats.get("p10", 0.0)))
    keep_ratio = fill_ratio
    health_score = _class_health_score(
        average_quality=float(analytics.get("avg_quality", 0.0)),
        average_novelty=float(analytics.get("avg_novelty", 0.0)),
        redundancy_ratio=redundancy_ratio,
        keep_ratio=keep_ratio,
        novelty_spread=novelty_spread,
    )
    effective_threshold = float(min((record.final_score for record in selected_records), default=0.0))

    novelty_weight = float(analytics.get("novelty_weight", 0.5))
    adaptive_relaxed = bool(analytics.get("adaptive_relaxed", False))

    # --- Removal breakdown ---
    # removed_by_quality: candidates below quality threshold or min_score filter
    removed_by_quality = total - len(candidates) + sum(1 for c in candidates if c.quality_score < curve_threshold) if curve_threshold else 0
    # removed_by_duplicate: duplicates suppressed during selection
    removed_by_duplicate = int(analytics.get("duplicates_removed", 0))
    # removed_by_novelty: remaining deletions = total - kept - quality_removed - duplicate_removed
    removed_by_novelty = max(0, total - final_kept - removed_by_quality - removed_by_duplicate)

    # --- Lightweight embedding collapse detection ---
    collapse_reason = ""
    collapse_signer = False
    collapse_motion = False
    collapse_overcompression = False
    # Signer collapse: very low novelty_std (< 0.01) suggests samples produce nearly identical embeddings
    if novelty_stats.get("std", 0.0) < 0.01:
        collapse_signer = True
        collapse_reason += "signer-collapse "
    # Motion collapse: very high pairwise similarity (> 0.92) suggests near-identical trajectories
    if avg_pairwise_similarity > 0.92:
        collapse_motion = True
        collapse_reason += "motion-collapse "
    # Over-compression: high redundancy (> 1.0) AND low novelty_p10 (< 0.01) means embeddings are too compressed
    if redundancy_ratio > 1.0 and novelty_stats.get("p10", 0.0) < 0.01:
        collapse_overcompression = True
        collapse_reason += "over-compression "
    if not collapse_reason:
        collapse_reason = "none"

    summary = ClassSummary(
        class_name=class_name,
        total_samples=total,
        adaptive_budget=budget,
        curve_threshold=float(curve_threshold or 0.0),
        shortlist_size=shortlist_size,
        duplicates_removed=int(analytics.get("duplicates_removed", 0)),
        final_kept=final_kept,
        average_quality=float(analytics.get("avg_quality", 0.0)),
        average_novelty=float(analytics.get("avg_novelty", 0.0)),
        effective_threshold=effective_threshold,
        redundancy_ratio=redundancy_ratio,
        average_pairwise_similarity=avg_pairwise_similarity,
        duplicate_suppression_pct=float(analytics.get("duplicate_suppression_pct", 0.0)),
        novelty_p10=float(novelty_stats.get("p10", 0.0)),
        novelty_p50=float(novelty_stats.get("p50", 0.0)),
        novelty_p90=float(novelty_stats.get("p90", 0.0)),
        keep_ratio=keep_ratio,
        health_score=health_score,
        dry_run=dry_run,
        embedding_mode=embedder.embedding_mode,
        duplicate_mode=duplicate_mode,
        duplicate_threshold=effective_duplicate_threshold,
        min_fill_ratio=min_fill_ratio,
        fill_ratio=fill_ratio,
        avg_final_similarity=avg_pairwise_similarity,
        effective_duplicate_threshold=effective_duplicate_threshold,
        novelty_weight=novelty_weight,
        adaptive_relaxed=adaptive_relaxed,
        removed_by_quality=removed_by_quality,
        removed_by_duplicate=removed_by_duplicate,
        removed_by_novelty=removed_by_novelty,
        collapse_signer=collapse_signer,
        collapse_motion=collapse_motion,
        collapse_overcompression=collapse_overcompression,
        collapse_reason=collapse_reason,
    )

    relax_tag = f"novelty_w={novelty_weight:.2f}" + (" ADAPT" if adaptive_relaxed else "")
    collapse_tag = f"[{collapse_reason.strip()}]" if collapse_reason != "none" else ""
    print(
        f"[{class_name}] total={total} budget={budget} shortlist={shortlist_size} "
        f"duplicate_mode={duplicate_mode} eff_dup_thr={effective_duplicate_threshold:.3f} "
        f"duplicates_removed={summary.duplicates_removed} final_kept={final_kept} fill_ratio={fill_ratio:.3f} "
        f"avg_quality={summary.average_quality:.3f} avg_novelty={summary.average_novelty:.3f} "
        f"avg_pairwise_similarity={avg_pairwise_similarity:.3f} health={health_score:.1f} "
        f"effective_threshold={effective_threshold:.4f} redundancy_ratio={redundancy_ratio:.3f} "
        f"{relax_tag} {collapse_tag} ({'dry-run' if dry_run else 'applied'})"
    )

    viz_bundle = {
        "rows": viz_rows,
        "embeddings": np.asarray(embeddings, dtype=np.float32) if export_viz_data else np.zeros((0, 0), dtype=np.float32),
        "pca_coords": pca_coords if pca_coords is not None else np.zeros((0, 0), dtype=np.float32),
        "paths": [candidate.path for candidate in shortlisted] if export_viz_data else [],
        "class_name": class_name,
    }

    return summary, selected_records, viz_bundle


def filter_quality_processed(
    root_dir: str = ROOT_DIR,
    auto_keep: bool = True,
    keep_limit: int = 500,
    quality_coverage: float = DEFAULT_QUALITY_COVERAGE,
    shortlist_multiplier: float = DEFAULT_SHORTLIST_MULTIPLIER,
    duplicate_mode: str = DEFAULT_DUPLICATE_MODE,
    duplicate_threshold: float | None = None,
    quality_power: float = DEFAULT_QUALITY_POWER,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
    embedding_checkpoint: str | None = None,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    embedding_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    export_viz_data: bool = False,
    viz_pca_dims: int = 2,
    dry_run: bool = False,
    class_only: str | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    max_keep_per_class: int | None = None,
    max_total_kept: int | None = None,
    min_class_samples: int | None = None,
    report_dir: str = DEFAULT_REPORT_DIR,
    report_prefix: str = DEFAULT_REPORT_PREFIX,
    archive_root: str | None = None,
) -> dict[str, Any]:
    if quality_coverage <= 0.0 or quality_coverage > 1.0:
        raise ValueError("quality_coverage must be in the range (0, 1]")
    if shortlist_multiplier < 1.0:
        raise ValueError("shortlist_multiplier must be >= 1.0")
    if min_fill_ratio <= 0.0 or min_fill_ratio > 1.0:
        raise ValueError("min_fill_ratio must be in the range (0, 1]")
    if quality_power <= 0.0:
        raise ValueError("quality_power must be > 0")
    if viz_pca_dims < 0:
        raise ValueError("viz_pca_dims must be >= 0")
    if duplicate_mode not in DEFAULT_DUPLICATE_THRESHOLDS:
        available = ", ".join(DEFAULT_DUPLICATE_THRESHOLDS)
        raise ValueError(f"duplicate_mode must be one of {available}, got '{duplicate_mode}'")

    if archive_root is None:
        archive_root = f"{root_dir}_del"
    archive_root = os.path.abspath(archive_root)

    rng = random.Random(DEFAULT_SEED)
    class_dirs = _list_class_dirs(root_dir)

    # If a global cap on total kept samples is provided, translate to a
    # per-class cap to avoid excessive CPU/embeddings computations.
    num_classes = max(1, len(class_dirs))
    if max_total_kept is not None:
        try:
            per_class_cap = max(1, int(math.floor(float(max_total_kept) / float(num_classes))))
            # Ensure the per-class cap will at least allow the requested minimum samples
            if min_class_samples is not None:
                try:
                    per_class_cap = max(per_class_cap, int(min_class_samples))
                except Exception:
                    pass
            if keep_limit > per_class_cap:
                print(f"[INFO] Adjusting --keep-per-class from {keep_limit} to {per_class_cap} to respect --max-total-kept={max_total_kept}")
            keep_limit = min(keep_limit, per_class_cap)
        except Exception:
            pass

    # Also enforce an explicit per-class cap if provided.
    if max_keep_per_class is not None:
        try:
            if min_class_samples is not None and int(max_keep_per_class) < int(min_class_samples):
                print(f"[WARN] --max-keep-per-class={max_keep_per_class} is less than --min-class-samples={min_class_samples}; raising cap to honor minimum")
                max_keep_per_class = int(min_class_samples)
            if keep_limit > int(max_keep_per_class):
                print(f"[INFO] Enforcing --max-keep-per-class={max_keep_per_class}; reducing keep-per-class from {keep_limit} to {max_keep_per_class}")
            keep_limit = min(keep_limit, int(max_keep_per_class))
        except Exception:
            pass

    if class_only:
        class_token = _normalize_class_token(class_only)
        class_dirs = [
            directory
            for directory in class_dirs
            if _normalize_class_token(os.path.basename(directory)) == class_token
            or class_token in _normalize_class_token(os.path.basename(directory))
        ]
        if not class_dirs:
            available = ", ".join(os.path.basename(directory) for directory in _list_class_dirs(root_dir))
            raise ValueError(f"Class '{class_only}' not found in {os.path.abspath(root_dir)}. Available: {available}")

    del rng  # deterministic workflow; reserved for future tie-breaking if needed

    embedder = DiversityEmbedder(checkpoint_path=embedding_checkpoint, batch_size=embedding_batch_size)
    effective_duplicate_threshold = _duplicate_mode_threshold(duplicate_mode, duplicate_threshold)

    print("=" * 100)
    print(
        f"Hybrid quality filter started | ROOT_DIR={os.path.abspath(root_dir)} | "
        f"MODE={'AUTO' if auto_keep else 'FIXED'} | QUALITY_COVERAGE={quality_coverage:.2f} | "
        f"KEEP_PER_CLASS={keep_limit} | SHORTLIST_MULTIPLIER={shortlist_multiplier:.1f} | "
        f"DUPLICATE_MODE={duplicate_mode} | DUPLICATE_THRESHOLD={effective_duplicate_threshold:.3f} | "
        f"MIN_FILL_RATIO={min_fill_ratio:.2f} | QUALITY_POWER={quality_power:.1f} | "
        f"EMBEDDINGS={embedder.embedding_mode} | CACHE={'on' if embedding_cache else 'off'} | "
        f"ARCHIVE_ROOT={archive_root} | DRY_RUN={dry_run}"
    )
    print("=" * 100)

    summaries: list[ClassSummary] = []
    all_records: list[SelectedSampleRecord] = []
    viz_rows: list[dict[str, Any]] = []
    viz_embeddings: list[np.ndarray] = []
    viz_pca_coords: list[np.ndarray] = []
    grand_total = 0
    grand_kept = 0
    grand_deleted = 0
    grand_budget = 0

    for class_dir in class_dirs:
        # Wrap per-class processing: a single corrupt class should never crash the
        # entire filtering pipeline.
        try:
            summary, records, viz_bundle = filter_quality_class_folder(
                class_dir=class_dir,
                auto_keep=auto_keep,
                keep_limit=keep_limit,
                quality_coverage=quality_coverage,
                shortlist_multiplier=shortlist_multiplier,
                duplicate_mode=duplicate_mode,
                duplicate_threshold=duplicate_threshold,
                quality_power=quality_power,
                embedder=embedder,
                min_fill_ratio=min_fill_ratio,
                embedding_cache=embedding_cache,
                max_keep_per_class=max_keep_per_class,
                min_class_samples=min_class_samples,
                cache_dir=cache_dir,
                export_viz_data=export_viz_data,
                viz_pca_dims=viz_pca_dims,
                dry_run=dry_run,
                min_score=min_score,
                archive_root=archive_root,
            )
            summaries.append(summary)
            all_records.extend(records)
            if export_viz_data and viz_bundle:
                viz_rows.extend(viz_bundle.get("rows", []))
                if isinstance(viz_bundle.get("embeddings"), np.ndarray) and viz_bundle["embeddings"].size:
                    viz_embeddings.append(viz_bundle["embeddings"])
                if isinstance(viz_bundle.get("pca_coords"), np.ndarray) and viz_bundle["pca_coords"].size:
                    viz_pca_coords.append(viz_bundle["pca_coords"])
            grand_total += summary.total_samples
            grand_kept += summary.final_kept
            grand_deleted += summary.total_samples - summary.final_kept
            grand_budget += summary.adaptive_budget
        except Exception as exc:
            class_name = os.path.basename(class_dir)
            print(f"[{class_name}] SKIPPED due to error: {exc}")
            print(f"    Continuing pipeline with remaining classes...")

    print("-" * 100)
    print(f"TOTAL: total={grand_total} adaptive_budget={grand_budget} kept={grand_kept} deleted={grand_deleted}")
    if dry_run:
        print("NOTE: DRY_RUN=True, no files were actually deleted.")
    print("=" * 100)

    # Class rankings: most redundant, most diverse, healthiest, and least healthy.
    health_sorted = sorted(summaries, key=lambda s: s.health_score, reverse=True)
    class_rankings = {
        "most_redundant": [
            {"class_name": item.class_name, "redundancy_ratio": item.redundancy_ratio, "health_score": item.health_score, "average_pairwise_similarity": item.average_pairwise_similarity}
            for item in sorted(summaries, key=lambda s: (s.redundancy_ratio, -s.average_pairwise_similarity), reverse=True)[:10]
        ],
        "most_diverse": [
            {"class_name": item.class_name, "average_novelty": item.average_novelty, "health_score": item.health_score}
            for item in sorted(summaries, key=lambda s: (s.average_novelty, -s.redundancy_ratio), reverse=True)[:10]
        ],
        "healthiest": [
            {"class_name": item.class_name, "health_score": item.health_score, "fill_ratio": item.fill_ratio}
            for item in health_sorted[:10]
        ],
        "least_healthy": [
            {"class_name": item.class_name, "health_score": item.health_score, "redundancy_ratio": item.redundancy_ratio}
            for item in health_sorted[-10:] if item.health_score < 100.0
        ],
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if export_viz_data:
        viz_npz_path = os.path.join(report_dir, f"{report_prefix}_{stamp}_viz.npz")
        viz_json_path = os.path.join(report_dir, f"{report_prefix}_{stamp}_viz.json")
        combined_embeddings = np.concatenate(viz_embeddings, axis=0) if viz_embeddings else np.zeros((0, 0), dtype=np.float32)
        combined_coords = np.concatenate(viz_pca_coords, axis=0) if viz_pca_coords else np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(
            viz_npz_path,
            embeddings=combined_embeddings.astype(np.float32),
            pca_coords=combined_coords.astype(np.float32),
        )
        with open(viz_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rows": viz_rows,
                    "embedding_shape": list(combined_embeddings.shape),
                    "pca_shape": list(combined_coords.shape),
                    "generated_at": stamp,
                },
                f,
                indent=2,
            )
        print(f"Saved visualization data: {viz_npz_path}")
        print(f"Saved visualization metadata: {viz_json_path}")

    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, f"{report_prefix}_{stamp}.json")
    csv_path = os.path.join(report_dir, f"{report_prefix}_{stamp}.csv")

    report = {
        "metadata": {
            "root_dir": os.path.abspath(root_dir),
            "auto_keep": auto_keep,
            "keep_limit": keep_limit,
            "quality_coverage": quality_coverage,
            "shortlist_multiplier": shortlist_multiplier,
            "duplicate_mode": duplicate_mode,
            "duplicate_threshold": effective_duplicate_threshold,
            "quality_power": quality_power,
            "min_fill_ratio": min_fill_ratio,
            "max_keep_per_class": max_keep_per_class,
            "max_total_kept": max_total_kept,
            "embedding_checkpoint": os.path.abspath(embedding_checkpoint) if embedding_checkpoint else None,
            "embedding_batch_size": embedding_batch_size,
            "embedding_cache": embedding_cache,
            "cache_dir": os.path.abspath(cache_dir),
            "archive_root": archive_root,
            "embedding_cache_stats": dict(EMBEDDING_CACHE_STATS),
            "export_viz_data": export_viz_data,
            "viz_pca_dims": viz_pca_dims,
            "min_score": min_score,
            "dry_run": dry_run,
            "generated_at": stamp,
        },
        "classes": [asdict(summary) for summary in summaries],
        "pinned_to_min_fill": [s.class_name for s in summaries if s.fill_ratio < s.min_fill_ratio],
        "class_rankings": class_rankings,
        "kept_samples": [asdict(record) for record in all_records],
        "totals": {
            "total_samples": grand_total,
            "adaptive_budget": grand_budget,
            "kept_samples": grand_kept,
            "deleted_samples": grand_deleted,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if all_records:
        fieldnames = list(asdict(all_records[0]).keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in all_records:
                writer.writerow(asdict(record))
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["no_samples"])

    # Export lightweight per-class analytics CSV
    analytics_csv = os.path.join(report_dir, f"{report_prefix}_{stamp}_class_analytics.csv")
    try:
        with open(analytics_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "class_name",
                "total_samples",
                "adaptive_budget",
                "final_kept",
                "fill_ratio",
                "min_fill_ratio",
                "duplicates_removed",
                "redundancy_ratio",
                "average_pairwise_similarity",
                "health_score",
                "collapse_reason",
                "adaptive_relaxed",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in summaries:
                writer.writerow({
                    "class_name": s.class_name,
                    "total_samples": s.total_samples,
                    "adaptive_budget": s.adaptive_budget,
                    "final_kept": s.final_kept,
                    "fill_ratio": s.fill_ratio,
                    "min_fill_ratio": s.min_fill_ratio,
                    "duplicates_removed": s.duplicates_removed,
                    "redundancy_ratio": s.redundancy_ratio,
                    "average_pairwise_similarity": s.average_pairwise_similarity,
                    "health_score": s.health_score,
                    "collapse_reason": s.collapse_reason,
                    "max_keep_per_class": max_keep_per_class,
                    "max_total_kept": max_total_kept,
                    "min_class_samples": min_class_samples,
                    "adaptive_relaxed": s.adaptive_relaxed,
                })
        print(f"Saved per-class analytics CSV: {analytics_csv}")
    except Exception:
        pass

    print(f"Saved JSON report: {json_path}")
    print(f"Saved CSV summary: {csv_path}")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid quality + diversity filter for processed .npy samples")
    parser.add_argument("--root", default=ROOT_DIR, help="Root folder containing class subfolders")
    parser.add_argument("--auto-keep", dest="auto_keep", action="store_true", default=True, help="Use the quality curve to estimate the per-class budget")
    parser.add_argument("--fixed-keep", dest="auto_keep", action="store_false", help="Use --keep-per-class as a fixed cap instead of the quality curve")
    parser.add_argument("--keep-per-class", type=int, default=500, help="Fixed keep count when --fixed-keep is used")
    parser.add_argument("--quality-coverage", type=float, default=DEFAULT_QUALITY_COVERAGE, help="Fraction of cumulative quality to preserve")
    parser.add_argument("--shortlist-multiplier", type=float, default=DEFAULT_SHORTLIST_MULTIPLIER, help="Top-K multiplier for shortlist size")
    parser.add_argument(
        "--duplicate-mode",
        choices=list(DEFAULT_DUPLICATE_THRESHOLDS.keys()),
        default=DEFAULT_DUPLICATE_MODE,
        help=f"Duplicate suppression aggressiveness: {', '.join(DEFAULT_DUPLICATE_THRESHOLDS.keys())} "
             f"(thresholds: {', '.join(f'{k}={v}' for k, v in DEFAULT_DUPLICATE_THRESHOLDS.items())})",
    )
    parser.add_argument(
        "--duplicate-threshold", type=float, default=None,
        help="Override cosine similarity threshold. When set, overrides --duplicate-mode.",
    )
    parser.add_argument(
        "--min-fill-ratio", type=float, default=DEFAULT_MIN_FILL_RATIO,
        help=f"Minimum fraction of adaptive budget to fill even if duplicates are suppressed (default: {DEFAULT_MIN_FILL_RATIO})",
    )
    parser.add_argument("--quality-power", type=float, default=DEFAULT_QUALITY_POWER, help="Exponent applied to quality score in final ranking")
    parser.add_argument("--embedding-checkpoint", default=None, help="Optional trained classifier checkpoint for penultimate-layer embeddings")
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE, help="Batch size for embedding extraction")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Optional safety floor for quality scores")
    parser.add_argument("--export-viz-data", action="store_true", help="Export PCA-ready embeddings and t-SNE-ready sample metadata")
    parser.add_argument("--viz-pca-dims", type=int, default=2, help="PCA reduction dimensions for visualization data (default: 2)")
    parser.add_argument("--no-embedding-cache", dest="embedding_cache", action="store_false", default=True, help="Disable embedding cache")
    parser.add_argument("--max-keep-per-class", type=int, default=None, help="Hard cap on kept samples per class to limit CPU/embedding work")
    parser.add_argument("--max-total-kept", type=int, default=None, help="Hard cap on total kept samples across all classes; translated to a per-class cap")
    parser.add_argument("--min-class-samples", type=int, default=DEFAULT_MIN_CLASS_SAMPLES, help="Minimum samples to keep per class to protect rare classes")
    parser.add_argument("--archive-root", default=None, help="Directory where deleted samples are copied before removal (default: <root>_del)")
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR, help="Directory for JSON and CSV reports")
    parser.add_argument("--report-prefix", default=DEFAULT_REPORT_PREFIX, help="Report filename prefix")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without changing files")
    parser.add_argument("--class", dest="class_only", default=None, help="Only process one class folder")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    filter_quality_processed(
        root_dir=args.root,
        auto_keep=args.auto_keep,
        keep_limit=args.keep_per_class,
        quality_coverage=args.quality_coverage,
        shortlist_multiplier=args.shortlist_multiplier,
        duplicate_mode=args.duplicate_mode,
        duplicate_threshold=args.duplicate_threshold,
        quality_power=args.quality_power,
        min_fill_ratio=args.min_fill_ratio,
        embedding_checkpoint=args.embedding_checkpoint,
        embedding_batch_size=args.embedding_batch_size,
        embedding_cache=args.embedding_cache,
        export_viz_data=args.export_viz_data,
        viz_pca_dims=args.viz_pca_dims,
        dry_run=args.dry_run,
        class_only=args.class_only,
        min_score=args.min_score,
        max_keep_per_class=args.max_keep_per_class,
        max_total_kept=args.max_total_kept,
        min_class_samples=args.min_class_samples,
        report_dir=args.report_dir,
        report_prefix=args.report_prefix,
        archive_root=args.archive_root,
    )


if __name__ == "__main__":
    main()