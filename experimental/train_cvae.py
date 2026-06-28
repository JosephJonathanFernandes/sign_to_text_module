"""Train a class-conditional VAE for landmark sequence augmentation.

Outputs:
- checkpoints/cvae_landmarks/best.pt
- checkpoints/cvae_landmarks/last.pt
- models/cvae_landmarks.pt
- models/cvae_metadata.json
- logs/cvae_landmarks/
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cvae_landmarks import (
    CVAEModelConfig,
    LandmarkCVAE,
    LossConfig,
    ProcessedLandmarkDataset,
    cvae_loss,
    per_class_stratified_split,
    save_json,
)


try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None


@dataclass
class TrainConfig:
    processed_root: str = "processed"
    seq_len: int = 20
    feature_dim: int = 506
    z_dim: int = 64
    class_embed_dim: int = 32
    encoder_hidden_dim: int = 256
    encoder_layers: int = 2
    encoder_bidirectional: bool = True
    decoder_hidden_dim: int = 256
    decoder_layers: int = 2
    dropout: float = 0.1

    batch_size: int = 32
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.2
    seed: int = 42
    early_stopping_patience: int = 12

    beta_kl: float = 1e-3
    velocity_weight: float = 0.5

    include_classes: list[str] | None = None
    include_prefixes: list[str] | None = None
    exclude_prefixes: list[str] | None = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True

    checkpoints_dir: str = "checkpoints/cvae_landmarks"
    models_dir: str = "models"
    logs_dir: str = "logs/cvae_landmarks"

    latent_pca_every: int = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CVAE for landmark sequence generation")
    p.add_argument("--processed-root", default="processed")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--feature-dim", type=int, default=506)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta-kl", type=float, default=1e-3)
    p.add_argument("--velocity-weight", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--include-class", action="append", dest="include_classes")
    p.add_argument("--include-prefix", action="append", dest="include_prefixes")
    p.add_argument("--exclude-prefix", action="append", dest="exclude_prefixes")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoints-dir", default="checkpoints/cvae_landmarks")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--logs-dir", default="logs/cvae_landmarks")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    cfg.processed_root = args.processed_root
    cfg.seq_len = args.seq_len
    cfg.feature_dim = args.feature_dim
    cfg.z_dim = args.z_dim
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.learning_rate = args.lr
    cfg.val_ratio = args.val_ratio
    cfg.seed = args.seed
    cfg.beta_kl = args.beta_kl
    cfg.velocity_weight = args.velocity_weight
    cfg.early_stopping_patience = args.patience
    cfg.include_classes = args.include_classes
    cfg.include_prefixes = args.include_prefixes
    cfg.exclude_prefixes = args.exclude_prefixes
    cfg.use_amp = not args.no_amp
    if args.device:
        cfg.device = args.device
    cfg.checkpoints_dir = args.checkpoints_dir
    cfg.models_dir = args.models_dir
    cfg.logs_dir = args.logs_dir
    return cfg


def build_writer(logs_dir: str):
    if SummaryWriter is None:
        return None
    os.makedirs(logs_dir, exist_ok=True)
    return SummaryWriter(logs_dir)


@torch.no_grad()
def maybe_log_latent_pca(model: LandmarkCVAE, loader: DataLoader, device: torch.device, writer, epoch: int) -> None:
    if writer is None:
        return

    zs = []
    ys = []
    max_batches = 8
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mu, _ = model.encode(x, y)
        zs.append(mu.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())

    if not zs:
        return

    z = np.concatenate(zs, axis=0)
    y = np.concatenate(ys, axis=0)
    z_mean = z.mean(axis=0, keepdims=True)
    z_center = z - z_mean

    # PCA via SVD; no sklearn dependency required.
    _, _, vt = np.linalg.svd(z_center, full_matrices=False)
    comp = vt[:2]
    z2 = z_center @ comp.T

    writer.add_histogram("latent/z_pc1", z2[:, 0], epoch)
    writer.add_histogram("latent/z_pc2", z2[:, 1], epoch)
    writer.add_scalar("latent/pc1_var", float(np.var(z2[:, 0])), epoch)
    writer.add_scalar("latent/pc2_var", float(np.var(z2[:, 1])), epoch)
    writer.add_scalar("latent/class_count_seen", float(len(np.unique(y))), epoch)


def train_one_epoch(
    model: LandmarkCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_cfg: LossConfig,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    meter = {"total": 0.0, "recon": 0.0, "kl": 0.0, "velocity": 0.0, "n": 0}

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = use_amp and device.type == "cuda"

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            recon, mu, logvar = model(x, y)
            losses = cvae_loss(
                x,
                recon,
                mu,
                logvar,
                beta_kl=loss_cfg.beta_kl,
                velocity_weight=loss_cfg.velocity_weight,
            )

        if scaler is not None and amp_enabled:
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            optimizer.step()

        batch_size = x.size(0)
        meter["n"] += batch_size
        for k in ("total", "recon", "kl", "velocity"):
            meter[k] += float(losses[k].detach().cpu()) * batch_size

    for k in ("total", "recon", "kl", "velocity"):
        meter[k] /= max(1, meter["n"])
    return meter


@torch.no_grad()
def validate_one_epoch(
    model: LandmarkCVAE,
    loader: DataLoader,
    loss_cfg: LossConfig,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    meter = {"total": 0.0, "recon": 0.0, "kl": 0.0, "velocity": 0.0, "n": 0}

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        amp_enabled = use_amp and device.type == "cuda"

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            recon, mu, logvar = model(x, y)
            losses = cvae_loss(
                x,
                recon,
                mu,
                logvar,
                beta_kl=loss_cfg.beta_kl,
                velocity_weight=loss_cfg.velocity_weight,
            )

        batch_size = x.size(0)
        meter["n"] += batch_size
        for k in ("total", "recon", "kl", "velocity"):
            meter[k] += float(losses[k].detach().cpu()) * batch_size

    for k in ("total", "recon", "kl", "velocity"):
        meter[k] /= max(1, meter["n"])
    return meter


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = ProcessedLandmarkDataset(
        cfg.processed_root,
        seq_len=cfg.seq_len,
        feature_dim=cfg.feature_dim,
        include_classes=set(cfg.include_classes) if cfg.include_classes else None,
        include_prefixes=tuple(cfg.include_prefixes) if cfg.include_prefixes else None,
        exclude_prefixes=tuple(cfg.exclude_prefixes) if cfg.exclude_prefixes else None,
    )

    labels = [dataset.label_encoder.encode(c) for c, _, _ in dataset.samples]
    train_idx, val_idx = per_class_stratified_split(labels, val_ratio=cfg.val_ratio, seed=cfg.seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model_cfg = CVAEModelConfig(
        seq_len=cfg.seq_len,
        feature_dim=dataset.feature_dim,
        num_classes=dataset.label_encoder.num_classes,
        z_dim=cfg.z_dim,
        class_embed_dim=cfg.class_embed_dim,
        encoder_hidden_dim=cfg.encoder_hidden_dim,
        encoder_layers=cfg.encoder_layers,
        encoder_bidirectional=cfg.encoder_bidirectional,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        decoder_layers=cfg.decoder_layers,
        dropout=cfg.dropout,
    )

    device = torch.device(cfg.device)
    model = LandmarkCVAE(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    loss_cfg = LossConfig(beta_kl=cfg.beta_kl, velocity_weight=cfg.velocity_weight)

    writer = build_writer(cfg.logs_dir)

    best_val = float("inf")
    best_epoch = -1
    wait = 0
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, loss_cfg, device, scaler, cfg.use_amp)
        val_stats = validate_one_epoch(model, val_loader, loss_cfg, device, cfg.use_amp)

        if writer is not None:
            for key in ("total", "recon", "kl", "velocity"):
                writer.add_scalar(f"loss/train_{key}", train_stats[key], epoch)
                writer.add_scalar(f"loss/val_{key}", val_stats[key], epoch)

            writer.add_scalar("meta/lr", optimizer.param_groups[0]["lr"], epoch)
            if epoch % max(1, cfg.latent_pca_every) == 0:
                maybe_log_latent_pca(model, val_loader, device, writer, epoch)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_total={train_stats['total']:.6f} val_total={val_stats['total']:.6f} "
            f"train_recon={train_stats['recon']:.6f} val_recon={val_stats['recon']:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(cfg),
            "label_encoder": dataset.label_encoder.to_dict(),
            "feature_dim": dataset.feature_dim,
            "seq_len": cfg.seq_len,
            "val_total": val_stats["total"],
            "val_recon": val_stats["recon"],
            "val_kl": val_stats["kl"],
            "val_velocity": val_stats["velocity"],
        }
        torch.save(ckpt, os.path.join(cfg.checkpoints_dir, "last.pt"))

        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            best_epoch = epoch
            wait = 0
            torch.save(ckpt, os.path.join(cfg.checkpoints_dir, "best.pt"))
        else:
            wait += 1

        if wait >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (best epoch={best_epoch}, best val={best_val:.6f})")
            break

    elapsed = time.time() - start
    best_path = os.path.join(cfg.checkpoints_dir, "best.pt")
    best_ckpt = torch.load(best_path, map_location="cpu")

    final_model_path = os.path.join(cfg.models_dir, "cvae_landmarks.pt")
    torch.save(
        {
            "model_state_dict": best_ckpt["model_state_dict"],
            "model_config": best_ckpt["model_config"],
            "label_encoder": best_ckpt["label_encoder"],
            "feature_dim": best_ckpt["feature_dim"],
            "seq_len": best_ckpt["seq_len"],
            "best_epoch": best_epoch,
            "best_val_total": best_val,
        },
        final_model_path,
    )

    metadata_path = os.path.join(cfg.models_dir, "cvae_metadata.json")
    save_json(
        metadata_path,
        {
            "model_path": os.path.abspath(final_model_path),
            "best_checkpoint": os.path.abspath(best_path),
            "best_epoch": best_epoch,
            "best_val_total": best_val,
            "elapsed_seconds": round(elapsed, 2),
            "train_size": len(train_set),
            "val_size": len(val_set),
            "num_classes": dataset.label_encoder.num_classes,
            "classes": dataset.label_encoder.classes,
            "model_config": asdict(model_cfg),
            "loss_config": asdict(loss_cfg),
            "train_config": asdict(cfg),
        },
    )

    if writer is not None:
        writer.close()

    print("=" * 80)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val total loss: {best_val:.6f}")
    print(f"Saved model: {final_model_path}")
    print(f"Saved metadata: {metadata_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
