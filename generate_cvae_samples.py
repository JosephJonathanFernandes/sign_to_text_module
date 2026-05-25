"""Generate class-balanced synthetic landmark sequences with a trained CVAE.

Policy highlights:
- Top-up classes below target count.
- Keep synthetic ratio controlled per class.
- Save as cvae_XXXX.npy under processed/<class_name>/.
- Apply quality filtering before saving.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cvae_landmarks import (
    CVAEModelConfig,
    LandmarkCVAE,
    ProcessedLandmarkDataset,
    QualityFilterConfig,
    filter_synthetic_sequence,
    load_json,
    pad_or_truncate_sequence,
    save_json,
)


@dataclass
class GenerateConfig:
    processed_root: str = "processed"
    checkpoint_path: str = "models/cvae_landmarks.pt"
    seq_len: int = 20
    feature_dim: int = 506

    target_per_class: int = 850
    max_ratio_synthetic: float = 0.30
    max_generate_per_class: int = 500
    batch_size: int = 128
    temperature: float = 1.0
    seed: int = 42

    quality_enabled: bool = True
    min_motion_magnitude: float = 0.002
    min_feature_std: float = 0.002
    max_frame_jump: float = 2.5

    include_classes: list[str] | None = None
    dry_run: bool = False
    output_root: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CVAE synthetic samples for low-count classes")
    p.add_argument("--processed-root", default="processed")
    p.add_argument("--checkpoint", default="models/cvae_landmarks.pt")
    p.add_argument("--target-per-class", type=int, default=850)
    p.add_argument("--max-ratio-synthetic", type=float, default=0.30)
    p.add_argument("--max-generate-per-class", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-class", action="append", dest="include_classes")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-root", default=None)

    p.add_argument("--quality-disabled", action="store_true")
    p.add_argument("--min-motion-magnitude", type=float, default=0.002)
    p.add_argument("--min-feature-std", type=float, default=0.002)
    p.add_argument("--max-frame-jump", type=float, default=2.5)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> GenerateConfig:
    cfg = GenerateConfig()
    cfg.processed_root = args.processed_root
    cfg.checkpoint_path = args.checkpoint
    cfg.target_per_class = args.target_per_class
    cfg.max_ratio_synthetic = args.max_ratio_synthetic
    cfg.max_generate_per_class = args.max_generate_per_class
    cfg.batch_size = args.batch_size
    cfg.temperature = args.temperature
    cfg.seed = args.seed
    cfg.include_classes = args.include_classes
    cfg.dry_run = args.dry_run
    cfg.output_root = args.output_root

    cfg.quality_enabled = not args.quality_disabled
    cfg.min_motion_magnitude = args.min_motion_magnitude
    cfg.min_feature_std = args.min_feature_std
    cfg.max_frame_jump = args.max_frame_jump
    return cfg


def count_real_samples(processed_root: str, include_classes: set[str] | None = None) -> dict[str, int]:
    counts = {}
    for class_name in sorted(os.listdir(processed_root)):
        class_dir = os.path.join(processed_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        if include_classes and class_name not in include_classes:
            continue
        n = 0
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(".npy"):
                continue
            if fname.lower().startswith("cvae_"):
                continue
            n += 1
        counts[class_name] = n
    return counts


def count_existing_synthetic(processed_root: str, include_classes: set[str] | None = None) -> dict[str, int]:
    counts = {}
    for class_name in sorted(os.listdir(processed_root)):
        class_dir = os.path.join(processed_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        if include_classes and class_name not in include_classes:
            continue
        n = 0
        for fname in os.listdir(class_dir):
            if fname.lower().startswith("cvae_") and fname.lower().endswith(".npy"):
                n += 1
        counts[class_name] = n
    return counts


def next_cvae_index(class_dir: str) -> int:
    max_idx = 0
    for fname in os.listdir(class_dir):
        if not (fname.lower().startswith("cvae_") and fname.lower().endswith(".npy")):
            continue
        stem = os.path.splitext(fname)[0]
        try:
            idx = int(stem.split("_")[1])
            max_idx = max(max_idx, idx)
        except (ValueError, IndexError):
            continue
    return max_idx + 1


@torch.no_grad()
def compute_class_latent_stats(
    model: LandmarkCVAE,
    dataset: ProcessedLandmarkDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    by_class_mu = defaultdict(list)

    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mu, _ = model.encode(x, y)
        mu_np = mu.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        for i in range(mu_np.shape[0]):
            by_class_mu[int(y_np[i])].append(mu_np[i])

    class_mean: dict[int, np.ndarray] = {}
    class_std: dict[int, np.ndarray] = {}
    class_count: dict[int, int] = {}

    for class_id, arrs in by_class_mu.items():
        arr = np.asarray(arrs, dtype=np.float32)
        class_mean[class_id] = arr.mean(axis=0)
        class_std[class_id] = arr.std(axis=0) + 1e-4
        class_count[class_id] = int(arr.shape[0])

    return class_mean, class_std, class_count


def build_quality_config(cfg: GenerateConfig) -> QualityFilterConfig:
    return QualityFilterConfig(
        enabled=cfg.quality_enabled,
        min_motion_magnitude=cfg.min_motion_magnitude,
        min_feature_std=cfg.min_feature_std,
        max_frame_jump=cfg.max_frame_jump,
    )


def resolve_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[LandmarkCVAE, dict[str, Any]]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {os.path.abspath(checkpoint_path)}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = CVAEModelConfig(**ckpt["model_config"])
    model = LandmarkCVAE(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def planned_generation_count(
    real_count: int,
    existing_synth: int,
    target_per_class: int,
    max_ratio_synthetic: float,
    max_generate_per_class: int,
) -> tuple[int, int, int, bool]:
    if real_count >= target_per_class:
        return 0, 0, 0, False

    needed = target_per_class - real_count

    # Keep synthetic/(real+synthetic) <= ratio, accounting for existing synthetic files.
    max_total_synth = int((max_ratio_synthetic / max(1e-6, 1.0 - max_ratio_synthetic)) * real_count)
    remaining_synth_budget = max(0, max_total_synth - existing_synth)

    n_generate = min(needed, max_generate_per_class, remaining_synth_budget)
    ratio_limited = needed > n_generate
    return n_generate, needed, remaining_synth_budget, ratio_limited


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    include_set = set(cfg.include_classes) if cfg.include_classes else None
    output_root = cfg.output_root or cfg.processed_root
    os.makedirs(output_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = resolve_model_from_checkpoint(cfg.checkpoint_path, device)

    dataset = ProcessedLandmarkDataset(
        cfg.processed_root,
        seq_len=ckpt.get("seq_len", cfg.seq_len),
        feature_dim=ckpt.get("feature_dim", cfg.feature_dim),
        include_classes=include_set,
        exclude_prefixes=("cvae_",),
    )

    class_mean, class_std, class_latent_count = compute_class_latent_stats(
        model,
        dataset,
        device,
        cfg.batch_size,
    )
    global_mean = np.stack(list(class_mean.values()), axis=0).mean(axis=0)
    global_std = np.stack(list(class_std.values()), axis=0).mean(axis=0)

    real_counts = count_real_samples(cfg.processed_root, include_set)
    synth_counts = count_existing_synthetic(cfg.processed_root, include_set)
    quality_cfg = build_quality_config(cfg)

    by_reason = Counter()
    saved_per_class = Counter()
    plan: dict[str, dict[str, Any]] = {}

    for class_name in dataset.label_encoder.classes:
        if include_set and class_name not in include_set:
            continue

        real_n = real_counts.get(class_name, 0)
        synth_n = synth_counts.get(class_name, 0)
        n_generate, needed, synth_budget, ratio_limited = planned_generation_count(
            real_n,
            synth_n,
            cfg.target_per_class,
            cfg.max_ratio_synthetic,
            cfg.max_generate_per_class,
        )

        plan[class_name] = {
            "real_count": real_n,
            "existing_synthetic": synth_n,
            "needed_for_target": needed,
            "remaining_synth_budget": synth_budget,
            "planned_generate": n_generate,
            "ratio_limited": ratio_limited,
        }

        if n_generate <= 0:
            if ratio_limited and needed > 0:
                print(
                    f"[WARN] {class_name}: needs {needed} but synth ratio cap reached "
                    f"(existing synth={synth_n}, real={real_n}, max_ratio={cfg.max_ratio_synthetic:.2f})."
                )
            continue

        class_id = dataset.label_encoder.encode(class_name)
        mu = class_mean.get(class_id, global_mean)
        std = class_std.get(class_id, global_std)
        latent_obs = class_latent_count.get(class_id, 0)

        if ratio_limited:
            print(
                f"[WARN] {class_name}: top-up requested={needed}, generating={n_generate} due to ratio cap. "
                f"(real={real_n}, existing_synth={synth_n})"
            )

        class_out_dir = os.path.join(output_root, class_name)
        os.makedirs(class_out_dir, exist_ok=True)
        next_idx = next_cvae_index(class_out_dir)

        generated = 0
        attempts = 0
        max_attempts = n_generate * 8

        while generated < n_generate and attempts < max_attempts:
            attempts += 1

            z = np.random.randn(model.cfg.z_dim).astype(np.float32)
            z = mu + (std * cfg.temperature) * z
            z_t = torch.from_numpy(z).unsqueeze(0).to(device)
            y_t = torch.tensor([class_id], dtype=torch.long, device=device)

            seq = model.sample(y_t, z_t, seq_len=model.cfg.seq_len)[0].detach().cpu().numpy()
            seq = pad_or_truncate_sequence(seq, model.cfg.seq_len, model.cfg.feature_dim)

            ok, reason = filter_synthetic_sequence(seq, quality_cfg)
            if not ok:
                by_reason[reason] += 1
                continue

            if not cfg.dry_run:
                out_name = f"cvae_{next_idx:04d}.npy"
                out_path = os.path.join(class_out_dir, out_name)
                np.save(out_path, seq.astype(np.float32))
            next_idx += 1
            generated += 1
            saved_per_class[class_name] += 1

        if generated < n_generate:
            print(
                f"[WARN] {class_name}: saved {generated}/{n_generate}. "
                f"latent_obs={latent_obs}, attempts={attempts}, temp={cfg.temperature}"
            )

    summary = {
        "config": asdict(cfg),
        "checkpoint_path": os.path.abspath(cfg.checkpoint_path),
        "output_root": os.path.abspath(output_root),
        "dry_run": cfg.dry_run,
        "saved_total": int(sum(saved_per_class.values())),
        "saved_per_class": dict(saved_per_class),
        "rejected_by_reason": dict(by_reason),
        "plan": plan,
    }

    report_path = os.path.join(output_root, "cvae_generation_report.json")
    save_json(report_path, summary)

    print("=" * 80)
    print(f"Saved synthetic total: {summary['saved_total']}")
    print(f"Report: {report_path}")
    if by_reason:
        print(f"Rejected samples by reason: {dict(by_reason)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
