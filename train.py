"""
Training and validation pipeline for ISL word recognition.
CPU-optimized with LR scheduling, early stopping, and augmentation.
Supports both single-split and K-fold cross-validation with ensemble.
Uses weighted CrossEntropyLoss + Mixup for handling class imbalance.
"""

import os
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import get_config

cfg = get_config()

# Convenience references for training
DEVICE = cfg.hardware.torch_device
BATCH_SIZE = cfg.training.batch_size
NUM_EPOCHS = cfg.training.num_epochs
LEARNING_RATE = cfg.training.learning_rate
VAL_SPLIT = cfg.training.val_split
RANDOM_SEED = cfg.training.random_seed
MODEL_SAVE_PATH = cfg.paths.model_save_path
WEIGHT_DECAY = cfg.training.weight_decay
LABEL_SMOOTHING = cfg.training.label_smoothing
PATIENCE = cfg.training.patience
SCHEDULER_PATIENCE = cfg.training.scheduler_patience
GRAD_CLIP = cfg.training.grad_clip
ENSEMBLE_DIR = cfg.paths.ensemble_dir
NUM_FOLDS = cfg.paths.num_folds
USE_CLASS_WEIGHTS = cfg.training.use_class_weights
CLASS_WEIGHT_POWER = cfg.training.class_weight_power
LR_SCHEDULER = cfg.training.lr_scheduler
LR_DECAY_FACTOR = cfg.training.lr_decay_factor
LR_MIN = cfg.training.lr_min
WARMUP_EPOCHS = cfg.training.warmup_epochs
USE_FOCAL_LOSS = cfg.training.use_focal_loss
FOCAL_ALPHA = cfg.training.focal_alpha
FOCAL_GAMMA = cfg.training.focal_gamma
USE_MIXUP = cfg.training.use_mixup
USE_CUTMIX = cfg.training.use_cutmix
MIXUP_ALPHA = cfg.training.mixup_alpha
MIXUP_PROB = cfg.training.mixup_prob
from dataset import ISLDataset
from model import SignLanguageGRU


def _compute_inverse_class_weights(
    labels: np.ndarray,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights with configurable power:
        w_c = (1 / count_c) ^ power
    Normalized so average weight ~= 1 for stable optimization.
    
    power=1.0: full inverse frequency (strong weighting)
    power=0.7: smooth weighting (moderate)
    power=0.0: uniform weights
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    # Apply power transformation for smoother weighting
    class_weights = (1.0 / (class_counts.astype(float) + 1e-6)) ** CLASS_WEIGHT_POWER
    # Normalize to have mean weight = 1
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.FloatTensor(class_weights).to(DEVICE)


# ── Mixup Utility ─────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.3):
    """Mixup: interpolate random training pairs."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    """Loss for mixup: weighted sum of both targets."""
    return (lam * criterion(logits, y_a)
            + (1 - lam) * criterion(logits, y_b))


def _is_augmented_sample(name: str) -> bool:
    """Return True for derived samples such as aug/merge variants."""
    name = name.lower()
    return any(tag in name for tag in ("_aug", "_merge", "_mrg"))


def _validation_priority(file_path: str) -> tuple[int, str]:
    """Rank samples so validation prefers MVI and minimizes webcam originals."""
    name = os.path.basename(file_path).lower()
    is_mvi = name.startswith("mvi")
    is_webcam = "webcam" in name
    is_aug = _is_augmented_sample(name)

    if is_mvi and is_aug:
        return (0, name)
    if is_mvi:
        return (1, name)
    if not is_webcam:
        return (2, name)
    if is_webcam and is_aug:
        return (3, name)
    if is_webcam:
        return (4, name)
    return (5, name)


def _training_priority(file_path: str) -> tuple[int, str]:
    """Rank samples so training prefers webcam, then other, then MVI."""
    name = os.path.basename(file_path).lower()
    is_mvi = name.startswith("mvi")
    is_webcam = "webcam" in name
    is_aug = _is_augmented_sample(name)

    if is_webcam and is_aug:
        return (0, name)
    if is_webcam:
        return (1, name)
    if not is_mvi:
        return (2, name)
    if is_mvi and is_aug:
        return (3, name)
    if is_mvi:
        return (4, name)
    return (5, name)


def _source_aware_split(
    samples: list[tuple[str, int]],
    labels: np.ndarray,
    val_split: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
        Split indices per class while biasing training toward webcam-derived files.

        The goal is to keep class coverage from stratification while selecting the
        training subset in this priority order:
            1. Webcam augmented
            2. Webcam original
            3. Other (non-webcam) files
            4. MVI augmented
            5. MVI original

        Validation receives the complement of the selected training indices, so
        the final train/val split remains strictly disjoint.
    """
    rng = np.random.default_rng(random_seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for cls in np.unique(labels):
        cls_indices = np.flatnonzero(labels == cls).tolist()
        if len(cls_indices) <= 1:
            train_indices.extend(cls_indices)
            continue

        target_val = int(round(len(cls_indices) * val_split))
        target_val = max(1, min(len(cls_indices) - 1, target_val))
        target_train = len(cls_indices) - target_val

        grouped: dict[int, list[int]] = defaultdict(list)
        for idx in cls_indices:
            grouped[_training_priority(samples[idx][0])[0]].append(idx)

        for bucket in grouped.values():
            rng.shuffle(bucket)

        chosen_train: list[int] = []
        for priority in range(5):
            if len(chosen_train) >= target_train:
                break
            bucket = grouped.get(priority, [])
            remaining = target_train - len(chosen_train)
            chosen_train.extend(bucket[:remaining])

        chosen_set = set(chosen_train)
        train_indices.extend(chosen_train)
        val_indices.extend(idx for idx in cls_indices if idx not in chosen_set)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def _build_source_aware_folds(
    samples: list[tuple[str, int]],
    labels: np.ndarray,
    num_folds: int,
    random_seed: int,
) -> list[np.ndarray]:
    """
    Build K disjoint validation folds by partitioning each class into
    priority-ordered chunks.

    Validation folds are disjoint, each class stays approximately balanced
    across folds, and higher-priority MVI samples are assigned earlier in the
    fold order so the training complement naturally keeps more webcam data.
    Returns a list of `num_folds` arrays containing validation indices.
    """
    rng = np.random.default_rng(random_seed)
    folds: list[list[int]] = [[] for _ in range(num_folds)]

    for cls in np.unique(labels):
        cls_indices = np.flatnonzero(labels == cls).tolist()
        if not cls_indices:
            continue

        buckets: dict[int, list[int]] = defaultdict(list)
        for idx in cls_indices:
            priority = _validation_priority(samples[idx][0])[0]
            buckets[priority].append(idx)

        ordered: list[int] = []
        for priority in range(6):
            bucket = buckets.get(priority, [])
            if bucket:
                rng.shuffle(bucket)
                ordered.extend(bucket)

        base = len(ordered) // num_folds
        remainder = len(ordered) % num_folds
        cursor = 0
        for fold in range(num_folds):
            fold_size = base + (1 if fold < remainder else 0)
            if fold_size:
                folds[fold].extend(ordered[cursor:cursor + fold_size])
            cursor += fold_size

    return [np.asarray(fold, dtype=np.int64) for fold in folds]


# ── Focal Loss (for hard sample mining) ──────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard samples.
    Reduces loss weight for easy samples, focuses on hard negatives.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet)
    """
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for rare class samples (0-1).
                   0 = no weighting, 0.25 = moderate, 0.5 = heavy
            gamma: Focusing parameter (0-5).
                   0 = standard CE, 2.0 = strong focus on hard samples
            weight: Class weights (from imbalanced dataset)
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits from model
            targets: (N,) ground truth class indices
        """
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none'
        )
        
        # Get probability of true class
        p_t = torch.exp(-ce_loss)
        
        # Focal weight: (1 - p_t) ^ gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Applied focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_data_loaders() -> tuple:
    """
    Create train (with augmentation + oversampling) and val datasets
    using a class-aware split that keeps validation MVI-heavy.

    Returns:
        (train_loader, val_loader, num_classes, class_weights, full_ds)
    """
    # Load without oversampling first for splitting
    full_ds = ISLDataset(augment=False, min_samples=2, oversample=False)
    total = len(full_ds)

    # Extract labels for source-aware stratified splitting
    labels = np.array([lbl for _, lbl in full_ds.samples])

    train_idx, val_idx = _source_aware_split(
        full_ds.samples,
        labels,
        VAL_SPLIT,
        RANDOM_SEED,
    )

    # Wrap with augmentation + oversampling for train, plain for val
    train_ds = _BalancedAugSubset(full_ds, train_idx.tolist())
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

    # Compute class weights (inverse frequency)
    train_labels = labels[train_idx]
    class_weights = _compute_inverse_class_weights(
        train_labels, full_ds.num_classes,
    )

    print(
        f"[Data] Train: {len(train_ds)} (balanced+aug) "
        f"| Val: {len(val_idx)}"
    )
    print(
        "[Data] Class weights: "
        f"{[f'{w:.2f}' for w in class_weights.tolist()]}"
    )
    return (
        train_loader,
        val_loader,
        full_ds.num_classes,
        class_weights,
        full_ds,
    )


class _BalancedAugSubset(torch.utils.data.Dataset):
    """
    Wraps a subset of ISLDataset with:
      - Balanced oversampling (minority classes repeated to match majority)
      - Data augmentation on every sample
    """

    def __init__(self, parent: ISLDataset, indices: list):
        self.parent = parent

        # Group indices by class
        class_indices = {}
        for i in indices:
            _, label = parent.samples[i]
            class_indices.setdefault(label, []).append(i)

        # Oversample to match the largest class
        max_count = max(len(v) for v in class_indices.values())
        self.balanced_indices = []
        for cls, idxs in sorted(class_indices.items()):
            n = len(idxs)
            repeats = max_count // n
            remainder = max_count % n
            oversampled = idxs * repeats + idxs[:remainder]
            self.balanced_indices.extend(oversampled)

        original = len(indices)
        balanced = len(self.balanced_indices)
        per_class = {}
        for i in self.balanced_indices:
            _, lbl = parent.samples[i]
            per_class[lbl] = per_class.get(lbl, 0) + 1
        print(f"[BalancedAug] {original} -> {balanced} samples (oversampled)")
        print(f"[BalancedAug] Per class: {dict(sorted(per_class.items()))}")

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        import numpy as np
        real_idx = self.balanced_indices[idx]
        fpath, label = self.parent.samples[real_idx]
        seq = np.load(fpath).astype(np.float32)
        seq, proximity = ISLDataset._prepare_sequence(seq, augment=True)
        return (
            torch.from_numpy(seq),
            torch.from_numpy(proximity),
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
        seq, proximity = ISLDataset._prepare_sequence(seq, augment=False)
        return (
            torch.from_numpy(seq),
            torch.from_numpy(proximity),
            torch.tensor(label, dtype=torch.long),
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    use_mixup: bool = True,
) -> tuple:
    """Train one epoch with optional Mixup; return (loss, accuracy%)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, proximity, labels in loader:
        sequences = sequences.to(DEVICE)
        proximity = proximity.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # Apply Mixup or CutMix if enabled
        if use_mixup and USE_MIXUP and np.random.rand() < MIXUP_PROB:
            # Mixup: interpolate random training pairs
            mixed_x, y_a, y_b, lam = mixup_data(sequences, labels, alpha=MIXUP_ALPHA)
            logits = model(mixed_x, proximity=proximity)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            # Accuracy on original labels (approximate - weighted sum)
            preds = logits.argmax(dim=1)
            correct += (lam * (preds == y_a).float().sum().item()
                        + (1 - lam) * (preds == y_b).float().sum().item())
        else:
            # Standard training
            logits = model(sequences, proximity=proximity)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )

        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
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

    for sequences, proximity, labels in loader:
        sequences = sequences.to(DEVICE)
        proximity = proximity.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(sequences, proximity=proximity)
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
    classes_list: list = None,
) -> SignLanguageGRU:
    """
        Full training loop with:
            - Weighted CrossEntropyLoss (inverse class frequency)
      - Mixup augmentation
      - AdamW with weight decay
      - ReduceLROnPlateau scheduler
      - Early stopping
      - Gradient clipping
      - Best model checkpointing
    """
    model = SignLanguageGRU(num_classes=num_classes).to(DEVICE)

    # Select loss function (Focal Loss for hard sample mining, else CE)
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            weight=class_weights,
            reduction='mean',
        )
    else:
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
                "classes": classes_list or [],
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
    ckpt = torch.load(
        MODEL_SAVE_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
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
    Train a single fold with balanced oversampling + weighted CE loss.
    Returns best validation accuracy.
    """
    labels = np.array([lbl for _, lbl in full_ds.samples])

    # Balanced oversampling for training split
    train_ds = _BalancedAugSubset(full_ds, train_idx.tolist())
    val_ds = _PlainSubset(full_ds, val_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Class weights from this fold's training set (inverse frequency)
    train_labels = labels[train_idx]
    class_weights = _compute_inverse_class_weights(
        train_labels, num_classes,
    )

    model = SignLanguageGRU(num_classes=num_classes).to(DEVICE)
    
    # Select loss function (Focal Loss for hard sample mining, else CE)
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            weight=class_weights,
            reduction='mean',
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
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
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
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
    Train NUM_FOLDS models using disjoint source-aware K-fold CV.
    Saves each fold model to ENSEMBLE_DIR/fold_N.pth.
    Returns list of per-fold best validation accuracies.
    """
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)

    full_ds = ISLDataset(augment=False, min_samples=2)
    num_classes = full_ds.num_classes
    labels = np.array([lbl for _, lbl in full_ds.samples])

    # Build source-aware folds that spread high-priority (MVI) samples across
    # every fold so each fold's validation set is MVI-heavy.
    folds = _build_source_aware_folds(
        full_ds.samples, labels, NUM_FOLDS, RANDOM_SEED
    )
    split_iter = []
    for fold in range(NUM_FOLDS):
        val_idx = folds[fold]
        train_idx = np.setdiff1d(np.arange(len(labels)), val_idx)
        split_iter.append((train_idx, val_idx))
    print(f"[KFold] Using source-aware K-fold (num_folds={NUM_FOLDS})")

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
        print(
            f"--- Fold {fold} ---  "
            f"train={len(train_idx)}  val={len(val_idx)}"
        )

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
    print("  K-Fold Results")
    print(f"  Per-fold accuracies: {[f'{a:.1f}%' for a in fold_accs]}")
    print(f"  Mean accuracy:       {avg_acc:.2f}%")
    print(
        f"  Overall accuracy:    {overall_acc:.2f}%  "
        f"({all_val_correct}/{all_val_total})"
    )
    print(f"  Models saved to:     {ENSEMBLE_DIR}/")
    print(f"{'='*60}")

    return fold_accs
