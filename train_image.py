"""
Training pipeline for image-based ISL letter/number recognition.
Supports single-split and K-fold cross-validation.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import (
    StratifiedShuffleSplit, StratifiedKFold,
)
from config_image import (
    DEVICE, IMG_BATCH_SIZE, IMG_NUM_EPOCHS,
    IMG_LEARNING_RATE, IMG_WEIGHT_DECAY,
    IMG_LABEL_SMOOTHING, IMG_PATIENCE,
    IMG_GRAD_CLIP, IMG_VAL_SPLIT, IMG_RANDOM_SEED,
    IMG_MODEL_PATH, IMG_ENSEMBLE_DIR, IMG_NUM_FOLDS,
    IMG_NUM_WORKERS,
)
from dataset_image import ISLImageDataset
from model_image import SignImageCNN


# ── Subset wrappers ──────────────────────────────────────────────


class _AugSubset(torch.utils.data.Dataset):
    """Subset with augmentation enabled."""

    def __init__(self, parent: ISLImageDataset, indices: list):
        self.parent = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fpath, label = self.parent.samples[self.indices[idx]]
        import cv2
        img = cv2.imread(fpath)
        if img is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        if img.shape[:2] != (128, 128):
            img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ISLImageDataset._augment(img)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return (
            torch.from_numpy(img),
            torch.tensor(label, dtype=torch.long),
        )


class _PlainSubset(torch.utils.data.Dataset):
    """Subset without augmentation."""

    def __init__(self, parent: ISLImageDataset, indices: list):
        self.parent = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fpath, label = self.parent.samples[self.indices[idx]]
        import cv2
        img = cv2.imread(fpath)
        if img is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        if img.shape[:2] != (128, 128):
            img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return (
            torch.from_numpy(img),
            torch.tensor(label, dtype=torch.long),
        )


# ── Training helpers ─────────────────────────────────────────────


def _train_one_epoch(model, loader, criterion, optimizer):
    """Train one epoch. Returns (loss, accuracy%)."""
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), IMG_GRAD_CLIP
        )
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def _validate(model, loader, criterion):
    """Validate. Returns (loss, accuracy%)."""
    model.eval()
    running_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


# ── Single-split training ────────────────────────────────────────


def train_image_model():
    """Train a single image model with stratified split."""
    full_ds = ISLImageDataset(augment=False)
    labels = np.array([lbl for _, lbl in full_ds.samples])
    num_classes = full_ds.num_classes

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=IMG_VAL_SPLIT,
        random_state=IMG_RANDOM_SEED,
    )
    train_idx, val_idx = next(
        splitter.split(np.zeros(len(labels)), labels)
    )

    train_ds = _AugSubset(full_ds, train_idx.tolist())
    val_ds = _PlainSubset(full_ds, val_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=IMG_BATCH_SIZE,
        shuffle=True, num_workers=IMG_NUM_WORKERS,
        persistent_workers=IMG_NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=IMG_BATCH_SIZE,
        shuffle=False, num_workers=IMG_NUM_WORKERS,
        persistent_workers=IMG_NUM_WORKERS > 0,
    )

    # Class weights
    train_labels = labels[train_idx]
    counts = np.bincount(train_labels, minlength=num_classes)
    cw = 1.0 / (counts + 1e-6)
    cw = cw / cw.sum() * num_classes
    class_weights = torch.FloatTensor(cw).to(DEVICE)

    model = SignImageCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=IMG_LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=IMG_LEARNING_RATE,
        weight_decay=IMG_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=IMG_NUM_EPOCHS, eta_min=1e-5,
    )

    total_p = sum(p.numel() for p in model.parameters())
    print(f"[ImageModel] Params: {total_p:,}")
    print(
        f"[Data] Train: {len(train_idx)} | "
        f"Val: {len(val_idx)}"
    )
    print(
        f" {'Ep':>3} | {'TrLoss':>7} | {'TrAcc':>6} | "
        f"{'VaLoss':>7} | {'VaAcc':>6} | {'LR':>8}"
    )
    print("-" * 58)

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, IMG_NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer,
        )
        va_loss, va_acc = _validate(
            model, val_loader, criterion,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "val_acc": va_acc,
                "epoch": epoch,
            }, IMG_MODEL_PATH)
            marker = " *"
        else:
            no_improve += 1

        print(
            f" {epoch:>3} | {tr_loss:>7.4f} | "
            f"{tr_acc:>5.1f}% | {va_loss:>7.4f} | "
            f"{va_acc:>5.1f}% | {lr:.1e} | "
            f"{dt:.1f}s{marker}"
        )

        if no_improve >= IMG_PATIENCE:
            print(f"\n[Early Stop] at epoch {epoch}")
            break

    print(f"\n[Train] Best val accuracy: {best_val_acc:.2f}%")


# ── K-Fold training ──────────────────────────────────────────────


def _train_one_fold(full_ds, train_idx, val_idx,
                    num_classes, fold, save_path):
    """Train one fold. Returns best val accuracy."""
    labels = np.array([lbl for _, lbl in full_ds.samples])

    train_ds = _AugSubset(full_ds, train_idx.tolist())
    val_ds = _PlainSubset(full_ds, val_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=IMG_BATCH_SIZE,
        shuffle=True, num_workers=IMG_NUM_WORKERS,
        persistent_workers=IMG_NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=IMG_BATCH_SIZE,
        shuffle=False, num_workers=IMG_NUM_WORKERS,
        persistent_workers=IMG_NUM_WORKERS > 0,
    )

    # Class weights from fold's train set
    train_labels = labels[train_idx]
    counts = np.bincount(train_labels, minlength=num_classes)
    cw = 1.0 / (counts + 1e-6)
    cw = cw / cw.sum() * num_classes
    class_weights = torch.FloatTensor(cw).to(DEVICE)

    model = SignImageCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=IMG_LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=IMG_LEARNING_RATE,
        weight_decay=IMG_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=IMG_NUM_EPOCHS, eta_min=1e-5,
    )

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, IMG_NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer,
        )
        va_loss, va_acc = _validate(
            model, val_loader, criterion,
        )
        scheduler.step()
        dt = time.time() - t0

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "val_acc": va_acc,
                "epoch": epoch,
                "fold": fold,
            }, save_path)
            marker = " *"
        else:
            no_improve += 1

        if epoch % 5 == 0 or marker:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"   F{fold} Ep {epoch:>3} | "
                f"tr {tr_acc:>5.1f}% | "
                f"va {va_acc:>5.1f}% | "
                f"lr {lr:.1e} | {dt:.1f}s{marker}"
            )

        if no_improve >= IMG_PATIENCE:
            print(f"   F{fold} Early stop at epoch {epoch}")
            break

    return best_val_acc


def train_image_kfold():
    """
    K-fold CV training for image model.
    Saves fold models to IMG_ENSEMBLE_DIR.
    """
    os.makedirs(IMG_ENSEMBLE_DIR, exist_ok=True)

    full_ds = ISLImageDataset(augment=False)
    num_classes = full_ds.num_classes
    labels = np.array([lbl for _, lbl in full_ds.samples])

    skf = StratifiedKFold(
        n_splits=IMG_NUM_FOLDS, shuffle=True,
        random_state=IMG_RANDOM_SEED,
    )

    fold_accs = []
    print(f"\n{'='*60}")
    print(
        f"  Image K-Fold CV ({IMG_NUM_FOLDS} folds) "
        f"| {len(full_ds)} images, {num_classes} classes"
    )
    print(f"{'='*60}\n")

    total_p = sum(
        p.numel()
        for p in SignImageCNN(num_classes).parameters()
    )
    print(f"[Model] Params per fold: {total_p:,}\n")

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels)
    ):
        save_path = os.path.join(
            IMG_ENSEMBLE_DIR, f"fold_{fold}.pth"
        )
        print(
            f"--- Fold {fold} ---  "
            f"train={len(tr_idx)}  val={len(va_idx)}"
        )

        best_acc = _train_one_fold(
            full_ds, tr_idx, va_idx,
            num_classes, fold, save_path,
        )
        fold_accs.append(best_acc)
        print(f"   Fold {fold} best: {best_acc:.2f}%\n")

    avg = np.mean(fold_accs)
    print(f"{'='*60}")
    print(f"  Per-fold: {[f'{a:.1f}%' for a in fold_accs]}")
    print(f"  Mean accuracy: {avg:.2f}%")
    print(f"  Models saved to: {IMG_ENSEMBLE_DIR}/")
    print(f"{'='*60}")

    return fold_accs
