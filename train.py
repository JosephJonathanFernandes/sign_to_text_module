"""
Training and validation pipeline for ISL word recognition.
CPU-optimized with LR scheduling, early stopping, and augmentation.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    VAL_SPLIT, RANDOM_SEED, MODEL_SAVE_PATH,
    WEIGHT_DECAY, LABEL_SMOOTHING, PATIENCE,
    SCHEDULER_PATIENCE, GRAD_CLIP,
)
from dataset import ISLDataset
from model import SignLanguageGRU


def create_data_loaders() -> tuple:
    """
    Create train (with augmentation) and val datasets,
    split from the same pool of .npy files.

    Returns:
        (train_loader, val_loader, num_classes)
    """
    # Load full dataset (no augmentation) to get indices
    full_ds = ISLDataset(augment=False)
    total = len(full_ds)
    val_size = int(total * VAL_SPLIT)
    train_size = total - val_size

    gen = torch.Generator().manual_seed(RANDOM_SEED)
    train_sub, val_sub = random_split(
        full_ds, [train_size, val_size], generator=gen
    )

    # Wrap train subset with augmentation
    train_ds = _AugmentedSubset(full_ds, train_sub.indices)
    val_ds = _PlainSubset(full_ds, val_sub.indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"[Data] Train: {train_size} (aug) | Val: {val_size}")
    return train_loader, val_loader, full_ds.num_classes


class _AugmentedSubset(torch.utils.data.Dataset):
    """Wraps a subset of ISLDataset with augmentation on."""

    def __init__(self, parent: ISLDataset, indices: list):
        self.parent = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        import numpy as np
        fpath, label = self.parent.samples[self.indices[idx]]
        seq = np.load(fpath).astype(np.float32)
        seq = ISLDataset._augment(seq)
        return (
            torch.from_numpy(seq),
            torch.tensor(label, dtype=torch.long),
        )


class _PlainSubset(torch.utils.data.Dataset):
    """Wraps a subset of ISLDataset without augmentation."""

    def __init__(self, parent: ISLDataset, indices: list):
        self.parent = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        import numpy as np
        fpath, label = self.parent.samples[self.indices[idx]]
        seq = np.load(fpath).astype(np.float32)
        return (
            torch.from_numpy(seq),
            torch.tensor(label, dtype=torch.long),
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple:
    """Train for one epoch. Returns (avg_loss, accuracy%)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )

        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple:
    """Validate model. Returns (avg_loss, accuracy%)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(sequences)
        loss = criterion(logits, labels)

        running_loss += loss.item() * sequences.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
) -> SignLanguageGRU:
    """
    Full training loop with:
      - Label smoothing
      - AdamW with weight decay
      - ReduceLROnPlateau scheduler
      - Early stopping
      - Gradient clipping
      - Best model checkpointing
    """
    model = SignLanguageGRU(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=SCHEDULER_PATIENCE,
    )

    # Parameter count
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print(f"[Model] Params: {total_p:,} | Trainable: {train_p:,}")
    print(f"[Model] Architecture:\n{model}\n")

    best_val_acc = 0.0
    no_improve = 0

    print(
        f" {'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | "
        f"{'VaLoss':>7} | {'VaAcc':>6} | {'LR':>8} | {'T':>4}"
    )
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        va_loss, va_acc = validate(
            model, val_loader, criterion
        )

        # Step scheduler based on val accuracy
        scheduler.step(va_acc)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": va_acc,
                "num_classes": num_classes,
            }, MODEL_SAVE_PATH)
            marker = " *"
        else:
            no_improve += 1

        print(
            f" {epoch:>3} | {tr_loss:>7.4f} | "
            f"{tr_acc:>5.1f}% | {va_loss:>7.4f} | "
            f"{va_acc:>5.1f}% | {cur_lr:.1e} | "
            f"{elapsed:>3.1f}s{marker}"
        )

        # Early stopping
        if no_improve >= PATIENCE:
            print(
                f"\n[Early Stop] No improvement for "
                f"{PATIENCE} epochs. Stopping."
            )
            break

    print(f"\n[Train] Best val accuracy: {best_val_acc:.2f}%")
    print(f"[Train] Model saved to: {MODEL_SAVE_PATH}")

    # Load best weights
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE,
                       weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    return model

    # Load best weights
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
