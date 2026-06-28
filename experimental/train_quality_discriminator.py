"""Train a lightweight quality discriminator for landmark sequence filtering.

Default task: predict P(sample_is_real) for a sequence.
Optional mode: class-aware realism scoring by conditioning on gesture class.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from quality_discriminator import (
    DiscriminatorConfig,
    FilterHeuristicConfig,
    QualityDiscriminator,
    SequenceQualityDataset,
    binary_metrics,
    load_json,
    save_json,
)


try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None


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
class TrainQDConfig:
    real_root: str = "processed"
    synthetic_root: str = "generated"
    seq_len: int = 20
    feature_dim: int = 506
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.2
    seed: int = 42
    patience: int = 6
    threshold: float = 0.8
    class_aware: bool = False
    include_classes: list[str] | None = None
    max_real_per_class: int | None = None
    max_fake_per_class: int | None = None

    hard_negative_mining: bool = False
    hard_negative_min_score: float = 0.45
    hard_negative_top_k_per_class: int = 25
    hard_negative_finetune_epochs: int = 5
    hard_negative_finetune_lr: float = 3e-4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True

    checkpoints_dir: str = "checkpoints/quality_discriminator"
    models_dir: str = "models"
    logs_dir: str = "logs/quality_discriminator"

    min_motion_variance: float = 1e-5
    min_feature_std: float = 0.002
    max_frame_jump: float = 2.5
    max_frame_drift: float = 4.0
    min_active_ratio: float = 0.05

    config_path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train landmark quality discriminator")
    parser.add_argument("--config", default=None, help="Optional JSON or YAML config file")
    parser.add_argument("--real-root", default="processed")
    parser.add_argument("--synthetic-root", default="generated")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--feature-dim", type=int, default=506)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--class-aware", action="store_true")
    parser.add_argument("--include-class", action="append", dest="include_classes")
    parser.add_argument("--max-real-per-class", type=int, default=None)
    parser.add_argument("--max-fake-per-class", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoints-dir", default="checkpoints/quality_discriminator")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--logs-dir", default="logs/quality_discriminator")

    parser.add_argument("--hard-negative-mining", action="store_true")
    parser.add_argument("--hard-negative-min-score", type=float, default=0.45)
    parser.add_argument("--hard-negative-top-k-per-class", type=int, default=25)
    parser.add_argument("--hard-negative-finetune-epochs", type=int, default=5)
    parser.add_argument("--hard-negative-finetune-lr", type=float, default=3e-4)

    parser.add_argument("--min-motion-variance", type=float, default=1e-5)
    parser.add_argument("--min-feature-std", type=float, default=0.002)
    parser.add_argument("--max-frame-jump", type=float, default=2.5)
    parser.add_argument("--max-frame-drift", type=float, default=4.0)
    parser.add_argument("--min-active-ratio", type=float, default=0.05)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> TrainQDConfig:
    cfg = TrainQDConfig()
    if args.config:
        data = load_config_file(args.config)
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.config_path = args.config

    cfg.real_root = args.real_root
    cfg.synthetic_root = args.synthetic_root
    cfg.seq_len = args.seq_len
    cfg.feature_dim = args.feature_dim
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.val_ratio = args.val_ratio
    cfg.seed = args.seed
    cfg.patience = args.patience
    cfg.threshold = args.threshold
    cfg.class_aware = args.class_aware
    cfg.include_classes = args.include_classes
    cfg.max_real_per_class = args.max_real_per_class
    cfg.max_fake_per_class = args.max_fake_per_class
    cfg.use_amp = not args.no_amp
    if args.device:
        cfg.device = args.device
    cfg.checkpoints_dir = args.checkpoints_dir
    cfg.models_dir = args.models_dir
    cfg.logs_dir = args.logs_dir
    cfg.hard_negative_mining = args.hard_negative_mining
    cfg.hard_negative_min_score = args.hard_negative_min_score
    cfg.hard_negative_top_k_per_class = args.hard_negative_top_k_per_class
    cfg.hard_negative_finetune_epochs = args.hard_negative_finetune_epochs
    cfg.hard_negative_finetune_lr = args.hard_negative_finetune_lr
    cfg.min_motion_variance = args.min_motion_variance
    cfg.min_feature_std = args.min_feature_std
    cfg.max_frame_jump = args.max_frame_jump
    cfg.max_frame_drift = args.max_frame_drift
    cfg.min_active_ratio = args.min_active_ratio
    return cfg


def split_indices_by_class_and_source(dataset: SequenceQualityDataset, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, bool], list[int]] = {}
    for idx, sample in enumerate(dataset.samples):
        key = (sample["class_name"], bool(sample["is_real"]))
        buckets.setdefault(key, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for key, indices in buckets.items():
        indices = indices.copy()
        rng.shuffle(indices)
        if len(indices) <= 1:
            n_val = 0
        else:
            n_val = max(1, int(round(len(indices) * val_ratio)))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    if not val_idx and train_idx:
        val_idx.append(train_idx.pop())
    if not train_idx:
        raise RuntimeError("Training split is empty")
    if not val_idx:
        raise RuntimeError("Validation split is empty")

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_sampler(dataset: SequenceQualityDataset, indices: list[int]) -> WeightedRandomSampler:
    class_source_counts: dict[tuple[str, bool], int] = {}
    for idx in indices:
        sample = dataset.samples[idx]
        key = (sample["class_name"], bool(sample["is_real"]))
        class_source_counts[key] = class_source_counts.get(key, 0) + 1

    weights = []
    for idx in indices:
        sample = dataset.samples[idx]
        key = (sample["class_name"], bool(sample["is_real"]))
        weights.append(1.0 / max(1, class_source_counts[key]))

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def make_loader(dataset: SequenceQualityDataset, indices: list[int], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    subset = Subset(dataset, indices)
    if shuffle:
        sampler = build_sampler(dataset, indices)
        return DataLoader(subset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


def build_writer(logs_dir: str):
    if SummaryWriter is None:
        return None
    os.makedirs(logs_dir, exist_ok=True)
    return SummaryWriter(logs_dir)


@torch.no_grad()
def predict_scores(
    model: QualityDiscriminator,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    scores = []
    targets = []
    meta = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        target = batch["target"].cpu().numpy()
        logits = model(x, y if model.use_class_conditioning else None)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        scores.append(probs)
        targets.append(target)
        meta.extend(
            [
                {
                    "path": p,
                    "class_name": c,
                    "is_real": bool(r),
                    "score": float(s),
                }
                for p, c, r, s in zip(batch["path"], batch["class_name"], batch["is_real"], probs)
            ]
        )
    return np.concatenate(scores, axis=0), np.concatenate(targets, axis=0), meta


def train_one_epoch(
    model: QualityDiscriminator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
    pos_weight: torch.Tensor,
) -> float:
    model.train()
    running = 0.0
    count = 0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = use_amp and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(x, y if model.use_class_conditioning else None)
            loss = criterion(logits, target)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        running += float(loss.detach().cpu()) * bs
        count += bs

    return running / max(1, count)


@torch.no_grad()
def validate_one_epoch(
    model: QualityDiscriminator,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    threshold: float,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    losses = []
    all_scores = []
    all_targets = []
    meta = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        target = batch["target"].to(device)
        amp_enabled = use_amp and device.type == "cuda"

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(x, y if model.use_class_conditioning else None)
            loss = criterion(logits, target)
        losses.append(float(loss.detach().cpu()) * x.size(0))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_scores.append(probs)
        all_targets.append(target.detach().cpu().numpy())
        meta.extend(
            [
                {
                    "path": p,
                    "class_name": c,
                    "is_real": bool(r),
                    "score": float(s),
                }
                for p, c, r, s in zip(batch["path"], batch["class_name"], batch["is_real"], probs)
            ]
        )

    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = binary_metrics(targets.astype(np.int32), scores, threshold=threshold)
    avg_loss = sum(losses) / max(1, len(targets))
    return avg_loss, metrics, scores, targets, meta


@torch.no_grad()
def mine_hard_negatives(
    model: QualityDiscriminator,
    dataset: SequenceQualityDataset,
    train_indices: list[int],
    device: torch.device,
    batch_size: int,
    min_score: float,
    top_k_per_class: int,
) -> list[int]:
    fake_indices = [idx for idx in train_indices if not dataset.samples[idx]["is_real"]]
    if not fake_indices:
        return []

    loader = DataLoader(Subset(dataset, fake_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    scores, _, meta = predict_scores(model, loader, device)

    per_class: dict[str, list[tuple[float, int]]] = {}
    for local_idx, item in enumerate(meta):
        if float(item["score"]) < min_score:
            continue
        per_class.setdefault(item["class_name"], []).append((float(item["score"]), fake_indices[local_idx]))

    mined = []
    for class_name, entries in per_class.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        mined.extend([idx for _, idx in entries[:top_k_per_class]])
    return sorted(set(mined))


def save_checkpoint(path: str, model: QualityDiscriminator, optimizer: torch.optim.Optimizer, cfg: TrainQDConfig, model_cfg: DiscriminatorConfig, heuristic_cfg: FilterHeuristicConfig, epoch: int, val_metrics: dict[str, float]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_config": asdict(cfg),
            "model_config": asdict(model_cfg),
            "heuristic_config": asdict(heuristic_cfg),
            "val_metrics": val_metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    include_set = set(cfg.include_classes) if cfg.include_classes else None
    dataset = SequenceQualityDataset(
        real_root=cfg.real_root,
        synthetic_root=cfg.synthetic_root,
        seq_len=cfg.seq_len,
        feature_dim=cfg.feature_dim,
        class_aware=cfg.class_aware,
        include_classes=include_set,
        max_real_per_class=cfg.max_real_per_class,
        max_fake_per_class=cfg.max_fake_per_class,
    )

    train_idx, val_idx = split_indices_by_class_and_source(dataset, cfg.val_ratio, cfg.seed)
    train_loader = make_loader(dataset, train_idx, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(dataset, val_idx, cfg.batch_size, shuffle=False, seed=cfg.seed)

    model_cfg = DiscriminatorConfig(
        seq_len=cfg.seq_len,
        feature_dim=dataset.feature_dim,
        hidden_size=64,
        num_layers=1,
        bidirectional=True,
        dropout=0.15,
        class_embed_dim=16,
        class_aware=cfg.class_aware,
    )
    heuristic_cfg = FilterHeuristicConfig(
        enabled=True,
        min_motion_variance=cfg.min_motion_variance,
        min_feature_std=cfg.min_feature_std,
        max_frame_jump=cfg.max_frame_jump,
        max_frame_drift=cfg.max_frame_drift,
        min_active_ratio=cfg.min_active_ratio,
    )

    device = torch.device(cfg.device)
    model = QualityDiscriminator(model_cfg).to(device)

    # Balance real/fake classes in the loss using dataset prevalence.
    n_real = sum(1 for idx in train_idx if dataset.samples[idx]["is_real"])
    n_fake = sum(1 for idx in train_idx if not dataset.samples[idx]["is_real"])
    pos_weight = torch.tensor([max(1.0, float(n_fake) / max(1, n_real))], device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    writer = build_writer(cfg.logs_dir)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = cfg.patience
    start = time.time()
    best_path = os.path.join(cfg.checkpoints_dir, "best.pt")
    last_path = os.path.join(cfg.checkpoints_dir, "last.pt")

    def run_phase(epochs: int, current_lr: float, phase_name: str) -> None:
        nonlocal best_val_loss, best_epoch, patience_left, optimizer
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, cfg.use_amp, pos_weight)
            val_loss, metrics, scores, targets, meta = validate_one_epoch(model, val_loader, device, cfg.use_amp, cfg.threshold)

            if writer is not None:
                writer.add_scalar(f"loss/{phase_name}_train", train_loss, epoch)
                writer.add_scalar(f"loss/{phase_name}_val", val_loss, epoch)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"metrics/{phase_name}_{key}", value, epoch)

            print(
                f"[{phase_name} {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"acc={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f} prec={metrics['precision']:.4f} rec={metrics['recall']:.4f}"
            )

            save_checkpoint(last_path, model, optimizer, cfg, model_cfg, heuristic_cfg, epoch, metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_left = cfg.patience
                save_checkpoint(best_path, model, optimizer, cfg, model_cfg, heuristic_cfg, epoch, metrics)
            else:
                patience_left -= 1

            if patience_left <= 0:
                print(f"Early stopping in phase {phase_name} at epoch {epoch}")
                break

            if writer is not None and epoch % 1 == 0:
                writer.add_histogram(f"scores/{phase_name}_real", scores[targets >= 0.5], epoch)
                writer.add_histogram(f"scores/{phase_name}_fake", scores[targets < 0.5], epoch)

    run_phase(cfg.epochs, cfg.lr, "base")

    if cfg.hard_negative_mining:
        mined = mine_hard_negatives(
            model=model,
            dataset=dataset,
            train_indices=train_idx,
            device=device,
            batch_size=cfg.batch_size,
            min_score=cfg.hard_negative_min_score,
            top_k_per_class=cfg.hard_negative_top_k_per_class,
        )
        if mined:
            augmented_train_idx = sorted(set(train_idx + mined))
            train_loader = make_loader(dataset, augmented_train_idx, cfg.batch_size, shuffle=True, seed=cfg.seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.hard_negative_finetune_lr, weight_decay=cfg.weight_decay)
            n_real = sum(1 for idx in augmented_train_idx if dataset.samples[idx]["is_real"])
            n_fake = sum(1 for idx in augmented_train_idx if not dataset.samples[idx]["is_real"])
            pos_weight = torch.tensor([max(1.0, float(n_fake) / max(1, n_real))], device=device)
            print(f"Hard negative mining added {len(mined)} fake samples; starting finetune phase.")
            run_phase(cfg.hard_negative_finetune_epochs, cfg.hard_negative_finetune_lr, "hardneg")

    elapsed = time.time() - start

    model_path = os.path.join(cfg.models_dir, "quality_discriminator.pt")
    torch.save(
        {
            "model_state_dict": torch.load(best_path, map_location="cpu")["model_state_dict"],
            "model_config": asdict(model_cfg),
            "train_config": asdict(cfg),
            "heuristic_config": asdict(heuristic_cfg),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "classes": dataset.label_encoder.classes,
        },
        model_path,
    )

    metadata_path = os.path.join(cfg.models_dir, "quality_discriminator_metadata.json")
    save_json(
        metadata_path,
        {
            "model_path": os.path.abspath(model_path),
            "best_checkpoint": os.path.abspath(best_path),
            "last_checkpoint": os.path.abspath(last_path),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "elapsed_seconds": round(elapsed, 2),
            "threshold": cfg.threshold,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "num_classes": dataset.label_encoder.num_classes,
            "classes": dataset.label_encoder.classes,
            "model_config": asdict(model_cfg),
            "heuristic_config": asdict(heuristic_cfg),
            "train_config": asdict(cfg),
        },
    )

    if writer is not None:
        writer.close()

    print("=" * 80)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
