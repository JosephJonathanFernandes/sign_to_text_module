"""
Training and validation pipeline for ISL word recognition.
CPU-optimized with LR scheduling, early stopping, and augmentation.
Supports both single-split and K-fold cross-validation with ensemble.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold, ShuffleSplit
from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    VAL_SPLIT, RANDOM_SEED, MODEL_SAVE_PATH,
    WEIGHT_DECAY, LABEL_SMOOTHING, PATIENCE,
    SCHEDULER_PATIENCE, GRAD_CLIP,
    ENSEMBLE_DIR, NUM_FOLDS,
)
from dataset import ISLDataset
from model import SignLanguageGRU


def create_data_loaders() -> tuple:
    """
    Create train (with augmentation) and val datasets using
    **stratified** split to ensure every class is represented.

    Returns:
        (train_loader, val_loader, num_classes, class_weights)
    """
    full_ds = ISLDataset(augment=False, min_samples=2)
    total = len(full_ds)

    # Extract labels for stratification
    labels = np.array([lbl for _, lbl in full_ds.samples])

    # Use stratified split if all classes have >=2 samples, else plain shuffle
    class_counts = np.bincount(labels, minlength=full_ds.num_classes)
    min_class_count = class_counts[class_counts > 0].min()
    if min_class_count >= 2:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        train_idx, val_idx = next(splitter.split(np.zeros(total), labels))
    else:
        splitter = ShuffleSplit(
            n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        train_idx, val_idx = next(splitter.split(np.zeros(total)))
        print("[Data] Using non-stratified split (some classes have <2 samples)")

    # Wrap with augmentation for train, plain for val
    train_ds = _AugmentedSubset(full_ds, train_idx.tolist())
    val_ds = _PlainSubset(full_ds, val_idx.tolist())

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

    # Compute class weights (inverse frequency) for balanced loss
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=full_ds.num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * full_ds.num_classes
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    print(f"[Data] Train: {len(train_idx)} (aug, stratified) | Val: {len(val_idx)}")
    print(f"[Data] Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")
    return train_loader, val_loader, full_ds.num_classes, class_weights


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
    class_weights: torch.Tensor = None,
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
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
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
                "classes": full_ds.classes,
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


# ── K-Fold Cross-Validation Training ──────────────────────────────


def _train_fold(
    full_ds: ISLDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_classes: int,
    fold: int,
    save_path: str,
) -> float:
    """
    Train a single fold. Returns best validation accuracy.
    """
    labels = np.array([lbl for _, lbl in full_ds.samples])

    train_ds = _AugmentedSubset(full_ds, train_idx.tolist())
    val_ds = _PlainSubset(full_ds, val_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Class weights from this fold's training set
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    model = SignLanguageGRU(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5,
    )

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = validate(model, val_loader, criterion)
        scheduler.step(epoch)
        elapsed = time.time() - t0

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": va_acc,
                "num_classes": num_classes,
                "classes": full_ds.classes,
                "fold": fold,
            }, save_path)
            marker = " *"
        else:
            no_improve += 1

        if epoch % 10 == 0 or marker:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"   F{fold} Ep {epoch:>3} | "
                f"tr {tr_acc:>5.1f}% | va {va_acc:>5.1f}% | "
                f"lr {cur_lr:.1e} | {elapsed:.1f}s{marker}"
            )

        if no_improve >= PATIENCE:
            print(f"   F{fold} Early stop at epoch {epoch}")
            break

    return best_val_acc


def train_kfold() -> list:
    """
    Train NUM_FOLDS models using Stratified K-Fold CV.
    Saves each fold model to ENSEMBLE_DIR/fold_N.pth.
    Returns list of per-fold best validation accuracies.
    """
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)

    full_ds = ISLDataset(augment=False, min_samples=2)
    num_classes = full_ds.num_classes
    labels = np.array([lbl for _, lbl in full_ds.samples])

    # Use stratified K-fold if possible, else plain K-fold
    class_counts = np.bincount(labels, minlength=num_classes)
    min_class_count = class_counts[class_counts > 0].min()
    if min_class_count >= NUM_FOLDS:
        kf = StratifiedKFold(
            n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED
        )
        split_iter = kf.split(np.zeros(len(labels)), labels)
        print("[KFold] Using stratified K-fold")
    else:
        kf = KFold(
            n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED
        )
        split_iter = kf.split(np.zeros(len(labels)))
        print(f"[KFold] Using plain K-fold (min class count={min_class_count} < {NUM_FOLDS})")

    fold_accs = []
    all_val_correct = 0
    all_val_total = 0

    print(f"\n{'='*60}")
    print(f"  K-Fold Cross-Validation ({NUM_FOLDS} folds)")
    print(f"  Dataset: {len(full_ds)} samples, {num_classes} classes")
    print(f"{'='*60}\n")

    total_p = sum(
        p.numel() for p in SignLanguageGRU(num_classes).parameters()
    )
    print(f"[Model] Params per fold: {total_p:,}\n")

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        save_path = os.path.join(ENSEMBLE_DIR, f"fold_{fold}.pth")
        print(f"--- Fold {fold} ---  train={len(train_idx)}  val={len(val_idx)}")

        best_acc = _train_fold(
            full_ds, train_idx, val_idx, num_classes, fold, save_path,
        )
        fold_accs.append(best_acc)
        all_val_correct += int(round(best_acc * len(val_idx) / 100))
        all_val_total += len(val_idx)

        print(f"   Fold {fold} best val acc: {best_acc:.2f}%\n")

    avg_acc = np.mean(fold_accs)
    overall_acc = 100.0 * all_val_correct / all_val_total
    print(f"{'='*60}")
    print(f"  K-Fold Results")
    print(f"  Per-fold accuracies: {[f'{a:.1f}%' for a in fold_accs]}")
    print(f"  Mean accuracy:       {avg_acc:.2f}%")
    print(f"  Overall accuracy:    {overall_acc:.2f}%  ({all_val_correct}/{all_val_total})")
    print(f"  Models saved to:     {ENSEMBLE_DIR}/")
    print(f"{'='*60}")

    return fold_accs
