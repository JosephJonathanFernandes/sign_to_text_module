"""Filter CVAE synthetic landmark sequences using a trained quality discriminator.

The filter keeps samples when:
- heuristic checks pass
- discriminator score >= threshold

Accepted files are copied or moved to an output directory that mirrors the
class folder layout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from quality_discriminator import (
    DiscriminatorConfig,
    FilterHeuristicConfig,
    QualityDiscriminator,
    LandmarkLabelEncoder,
    pad_or_truncate_sequence,
    quality_heuristic_score,
    save_json,
)


def load_config_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yml", ".yaml")):
            try:
                import yaml  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("YAML config requested but PyYAML is not installed") from exc
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


@dataclass
class FilterConfig:
    source_root: str = "generated"
    processed_root: str = "processed"
    output_root: str = "filtered_generated"
    checkpoint_path: str = "models/quality_discriminator.pt"
    threshold: float = 0.8
    seq_len: int = 20
    feature_dim: int = 506
    batch_size: int = 128
    class_aware: bool = False
    include_classes: list[str] | None = None
    dry_run: bool = False
    move_files: bool = False
    overwrite: bool = False
    use_processed_fallback: bool = True

    min_motion_variance: float = 1e-5
    min_feature_std: float = 0.002
    max_frame_jump: float = 2.5
    max_frame_drift: float = 4.0
    min_active_ratio: float = 0.05


class SyntheticOnlyDataset(Dataset):
    def __init__(self, source_root: str, *, include_classes: set[str] | None, seq_len: int, feature_dim: int, synthetic_prefixes: tuple[str, ...] | None = ("cvae_",)):
        self.source_root = source_root
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.synthetic_prefixes = synthetic_prefixes
        self.include_classes = include_classes

        if not os.path.isdir(source_root):
            raise FileNotFoundError(f"Source root not found: {os.path.abspath(source_root)}")

        self.class_names = self._discover_classes(source_root)
        self.label_encoder = LandmarkLabelEncoder(self.class_names)
        self.samples: list[dict[str, Any]] = []
        self._scan()

        if not self.samples:
            raise RuntimeError("No synthetic samples found")

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        import re
        s = name.strip().lower()
        s = re.sub(r"^\d+\.?\s*", "", s)
        s = s.replace(" ", "_")
        return s

    def _allow_class(self, class_name: str) -> bool:
        if not self.include_classes:
            return True
        allow = {self._normalize_class_name(c) for c in self.include_classes}
        return self._normalize_class_name(class_name) in allow

    def _discover_classes(self, root: str) -> list[str]:
        classes = []
        for name in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, name)) and self._allow_class(name):
                classes.append(name)
        return classes

    def _is_synthetic_file(self, fname: str) -> bool:
        lower = fname.lower()
        if not lower.endswith(".npy"):
            return False
        if self.synthetic_prefixes is None:
            return True
        return lower.startswith(self.synthetic_prefixes)

    def _scan(self) -> None:
        for class_name in self.class_names:
            class_dir = os.path.join(self.source_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not self._is_synthetic_file(fname):
                    continue
                path = os.path.join(class_dir, fname)
                self.samples.append(
                    {
                        "path": path,
                        "class_name": class_name,
                        "class_idx": self.label_encoder.encode(class_name),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        seq = np.load(sample["path"], allow_pickle=False).astype(np.float32)
        seq = pad_or_truncate_sequence(seq, self.seq_len, self.feature_dim)
        return {
            "x": torch.from_numpy(seq),
            "y": torch.tensor(sample["class_idx"], dtype=torch.long),
            "path": sample["path"],
            "class_name": sample["class_name"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter CVAE synthetic samples with a quality discriminator")
    parser.add_argument("--config", default=None, help="Optional JSON or YAML config file")
    parser.add_argument("--source-root", default="generated")
    parser.add_argument("--processed-root", default="processed")
    parser.add_argument("--output-root", default="filtered_generated")
    parser.add_argument("--checkpoint", default="models/quality_discriminator.pt")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--feature-dim", type=int, default=506)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--class-aware", action="store_true")
    parser.add_argument("--include-class", action="append", dest="include_classes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--move-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-processed-fallback", action="store_true")

    parser.add_argument("--min-motion-variance", type=float, default=1e-5)
    parser.add_argument("--min-feature-std", type=float, default=0.002)
    parser.add_argument("--max-frame-jump", type=float, default=2.5)
    parser.add_argument("--max-frame-drift", type=float, default=4.0)
    parser.add_argument("--min-active-ratio", type=float, default=0.05)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[QualityDiscriminator, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = DiscriminatorConfig(**ckpt["model_config"])
    model = QualityDiscriminator(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def resolve_source_root(args: argparse.Namespace) -> str:
    if os.path.isdir(args.source_root):
        return args.source_root
    if not args.no_processed_fallback:
        return args.processed_root
    raise FileNotFoundError(f"Source root not found: {args.source_root}")


def build_heuristic_cfg(args: argparse.Namespace) -> FilterHeuristicConfig:
    return FilterHeuristicConfig(
        enabled=True,
        min_motion_variance=args.min_motion_variance,
        min_feature_std=args.min_feature_std,
        max_frame_jump=args.max_frame_jump,
        max_frame_drift=args.max_frame_drift,
        min_active_ratio=args.min_active_ratio,
    )


@torch.no_grad()
def score_batch(model: QualityDiscriminator, batch: dict[str, Any], device: torch.device) -> np.ndarray:
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    logits = model(x, y if model.use_class_conditioning else None)
    return torch.sigmoid(logits).detach().cpu().numpy()


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.config:
        cfg_data = load_config_file(args.config)
        for key, value in cfg_data.items():
            if hasattr(args, key):
                setattr(args, key, value)
    include_set = set(args.include_classes) if args.include_classes else None
    source_root = resolve_source_root(args)
    os.makedirs(args.output_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    heuristic_cfg = build_heuristic_cfg(args)

    dataset = SyntheticOnlyDataset(
        source_root,
        include_classes=include_set,
        seq_len=ckpt.get("train_config", {}).get("seq_len", args.seq_len),
        feature_dim=ckpt.get("train_config", {}).get("feature_dim", args.feature_dim),
        synthetic_prefixes=None if os.path.abspath(source_root) != os.path.abspath(args.processed_root) else ("cvae_",),
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    accepted_by_class = Counter()
    rejected_by_reason = Counter()
    rejected_by_score = Counter()
    total_seen = 0
    total_kept = 0
    manifest: list[dict[str, Any]] = []

    for batch in loader:
        probs = score_batch(model, batch, device)
        for i, prob in enumerate(probs):
            src_path = batch["path"][i]
            class_name = batch["class_name"][i]
            total_seen += 1

            seq = np.load(src_path, allow_pickle=False).astype(np.float32)
            seq = pad_or_truncate_sequence(seq, args.seq_len, args.feature_dim)
            ok, reason = quality_heuristic_score(seq, heuristic_cfg)
            if not ok:
                rejected_by_reason[reason] += 1
                manifest.append({"path": src_path, "class_name": class_name, "score": float(prob), "keep": False, "reason": reason})
                continue

            keep = float(prob) >= args.threshold
            if keep:
                total_kept += 1
                accepted_by_class[class_name] += 1
                dst_path = os.path.join(args.output_root, class_name, os.path.basename(src_path))
                if args.dry_run:
                    manifest.append({"path": src_path, "class_name": class_name, "score": float(prob), "keep": True, "reason": "threshold"})
                    continue

                if os.path.exists(dst_path) and not args.overwrite:
                    stem, ext = os.path.splitext(dst_path)
                    suffix = 1
                    candidate = f"{stem}_{suffix}{ext}"
                    while os.path.exists(candidate):
                        suffix += 1
                        candidate = f"{stem}_{suffix}{ext}"
                    dst_path = candidate
                ensure_parent(dst_path)
                if args.move_files:
                    shutil.move(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                manifest.append({"path": src_path, "class_name": class_name, "score": float(prob), "keep": True, "reason": "threshold"})
            else:
                rejected_by_score["below_threshold"] += 1
                manifest.append({"path": src_path, "class_name": class_name, "score": float(prob), "keep": False, "reason": "below_threshold"})

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "source_root": os.path.abspath(source_root),
        "output_root": os.path.abspath(args.output_root),
        "threshold": args.threshold,
        "total_seen": total_seen,
        "total_kept": total_kept,
        "accepted_by_class": dict(accepted_by_class),
        "rejected_by_reason": dict(rejected_by_reason),
        "rejected_by_score": dict(rejected_by_score),
        "dry_run": args.dry_run,
        "move_files": args.move_files,
        "heuristic_config": asdict(heuristic_cfg),
        "model_config": ckpt.get("model_config", {}),
    }

    report_path = os.path.join(args.output_root, "quality_filter_report.json")
    save_json(report_path, report)

    manifest_path = os.path.join(args.output_root, "quality_filter_manifest.json")
    save_json(manifest_path, {"items": manifest})

    print("=" * 80)
    print(f"Seen: {total_seen}")
    print(f"Kept: {total_kept}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")
    if rejected_by_reason:
        print(f"Rejected by heuristic: {dict(rejected_by_reason)}")
    if rejected_by_score:
        print(f"Rejected by score: {dict(rejected_by_score)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
