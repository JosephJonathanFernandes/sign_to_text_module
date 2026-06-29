"""
Training and validation pipeline for ISL word recognition.
CPU-optimized with LR scheduling, early stopping, and augmentation.
Supports both single-split and K-fold cross-validation with ensemble.
Uses weighted CrossEntropyLoss + Mixup for handling class imbalance.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_config
from src.utils.pipeline_logger import PipelineLogger

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
KFOLD_MANIFEST_PATH = os.path.join(ENSEMBLE_DIR, "kfold_manifest.json")
NUM_FOLDS = cfg.paths.num_folds
USE_CLASS_WEIGHTS = cfg.training.use_class_weights
CLASS_WEIGHT_POWER = cfg.training.class_weight_power
LR_SCHEDULER = cfg.training.lr_scheduler
LR_DECAY_FACTOR = cfg.training.lr_decay_factor
LR_MIN = cfg.training.lr_min
USE_FOCAL_LOSS = cfg.training.use_focal_loss
FOCAL_ALPHA = cfg.training.focal_alpha
FOCAL_GAMMA = cfg.training.focal_gamma
USE_MIXUP = cfg.training.use_mixup
USE_CUTMIX = cfg.training.use_cutmix
MIXUP_ALPHA = cfg.training.mixup_alpha
MIXUP_PROB = cfg.training.mixup_prob
from src.preprocessing.dataset import ISLDataset
from src.training.model import SignLanguageGRU


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


def _sample_label(sample) -> int:
    """Return the label field from a dataset sample.

    Supports both legacy (path, label) and weighted (path, label, weight)
    sample tuples.
    """
    return int(sample[1])


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_kfold_manifest() -> dict:
    if not os.path.exists(KFOLD_MANIFEST_PATH):
        return {}
    try:
        with open(KFOLD_MANIFEST_PATH, encoding="utf-8") as manifest_file:
            return json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_kfold_manifest(manifest: dict) -> None:
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    manifest["updated_at"] = _now_iso()
    with open(KFOLD_MANIFEST_PATH, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)
        manifest_file.write("\n")


def _init_kfold_manifest(start_fold: int, num_folds: int, dataset_size: int, num_classes: int) -> dict:
    manifest = _load_kfold_manifest()
    manifest.update(
        {
            "run_started_at": manifest.get("run_started_at") or _now_iso(),
            "start_fold": start_fold,
            "num_folds": num_folds,
            "dataset_size": dataset_size,
            "num_classes": num_classes,
            "status": "in_progress",
            "completed_folds": manifest.get("completed_folds", []),
            "folds": manifest.get("folds", {}),
        }
    )
    return manifest


def _record_kfold_fold(manifest: dict, fold: int, best_acc: float, save_path: str) -> None:
    completed_folds = manifest.setdefault("completed_folds", [])
    if fold not in completed_folds:
        completed_folds.append(fold)
        completed_folds.sort()
    folds = manifest.setdefault("folds", {})
    folds[str(fold)] = {
        "status": "complete",
        "best_val_acc": round(float(best_acc), 3),
        "checkpoint": save_path,
        "updated_at": _now_iso(),
    }
    _save_kfold_manifest(manifest)


def _finalize_kfold_manifest(manifest: dict, fold_accs: list[float]) -> None:
    manifest["status"] = "complete"
    manifest["fold_accuracies"] = [round(float(x), 3) for x in fold_accs]
    _save_kfold_manifest(manifest)


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


def _disjoint_stratified_split(
    samples: list[tuple[str, int]],
    labels: np.ndarray,
    val_split: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split indices per class into disjoint train/val sets.

    This keeps the split stratified by class, but otherwise random within each
    class. Train and validation remain strictly disjoint; no source bias is used.
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

        rng.shuffle(cls_indices)

        chosen_train = cls_indices[:target_train]
        chosen_set = set(chosen_train)
        train_indices.extend(chosen_train)
        val_indices.extend(idx for idx in cls_indices if idx not in chosen_set)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def _build_disjoint_folds(
    samples: list[tuple[str, int]],
    labels: np.ndarray,
    num_folds: int,
    random_seed: int,
) -> list[np.ndarray]:
    """
    Build K disjoint validation folds by partitioning each class randomly.

    Validation folds are disjoint and each class stays approximately balanced
    across folds, but there is no source-based priority ordering.
    Returns a list of `num_folds` arrays containing validation indices.
    """
    rng = np.random.default_rng(random_seed)
    folds: list[list[int]] = [[] for _ in range(num_folds)]

    for cls in np.unique(labels):
        cls_indices = np.flatnonzero(labels == cls).tolist()
        if not cls_indices:
            continue

        rng.shuffle(cls_indices)

        base = len(cls_indices) // num_folds
        remainder = len(cls_indices) % num_folds
        cursor = 0
        for fold in range(num_folds):
            fold_size = base + (1 if fold < remainder else 0)
            if fold_size:
                folds[fold].extend(cls_indices[cursor:cursor + fold_size])
            cursor += fold_size

    return [np.asarray(fold, dtype=np.int64) for fold in folds]


def _resolve_phase_neg_root(
    phase: str | None,
    neg_root: str | None = None,
) -> str | None:
    """Resolve the reject source for a training phase.

    Phase 1 uses `processed_negatives` when provided by the caller.
    Phase 2 prefers `processed_negatives_del` unless the caller passes an
    explicit override.
    """
    if neg_root:
        return neg_root

    phase_name = (phase or "phase1").strip().lower()
    if phase_name == "phase2":
        default_neg_del = os.path.join(os.path.dirname(cfg.paths.processed_dir), "processed_negatives_del")
        if os.path.isdir(default_neg_del):
            return default_neg_del

    return None


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


def create_data_loaders(
    neg_root: str | None = None,
    archived_root: str | None = None,
    archived_weight: float = 0.25,
    include_archived: bool = False,
    phase: str = "phase1",
) -> tuple:
    """
    Create train (with augmentation + oversampling) and val datasets
    using a class-aware split that keeps validation MVI-heavy.

    Returns:
        (train_loader, val_loader, num_classes, class_weights, full_ds)
    """
    # Only include archived samples if explicitly requested via include_archived
    if include_archived:
        if archived_root is None:
            default_arch = os.path.join(os.path.dirname(cfg.paths.processed_dir), "processed_del")
            if os.path.isdir(default_arch):
                archived_root = default_arch

    # If caller passed None explicitly for archived_weight, use the function default
    if archived_weight is None:
        archived_weight = 0.25

    # Resolve the phase-specific reject source without changing the label name.
    neg_root = _resolve_phase_neg_root(phase, neg_root=neg_root)

    # Load without oversampling first for splitting
    full_ds = ISLDataset(
        augment=False,
        min_samples=2,
        oversample=False,
        neg_root=neg_root,
        archived_root=archived_root,
        archived_weight=archived_weight,
    )
    total = len(full_ds)

    # Extract labels for stratified splitting
    # samples are (path, label, weight)
    labels = np.array([s[1] for s in full_ds.samples])

    train_idx, val_idx = _disjoint_stratified_split(
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
            # parent.samples entries are (path, label, weight, domain_idx)
            _, label, _, _ = parent.samples[i]
            class_indices.setdefault(label, []).append(i)

        # Oversample to match the largest non-reject class.
        # Keep reject at its natural count so a large negatives bucket does not
        # force every sign class to be duplicated up to that size.
        reject_label = getattr(parent, "neg_label", "__reject__")
        reject_idx = parent.class_to_idx.get(reject_label)

        non_reject_counts = [
            len(v) for cls, v in class_indices.items() if cls != reject_idx
        ]
        max_count = max(non_reject_counts) if non_reject_counts else max(len(v) for v in class_indices.values())
        self.balanced_indices = []
        for cls, idxs in sorted(class_indices.items()):
            n = len(idxs)
            if cls == reject_idx:
                self.balanced_indices.extend(idxs)
                continue
            repeats = max_count // n
            remainder = max_count % n
            oversampled = idxs * repeats + idxs[:remainder]
            self.balanced_indices.extend(oversampled)

        original = len(indices)
        balanced = len(self.balanced_indices)
        per_class = {}
        for i in self.balanced_indices:
            _, lbl, _, _ = parent.samples[i]
            per_class[lbl] = per_class.get(lbl, 0) + 1
        print(f"[BalancedAug] {original} -> {balanced} samples (oversampled)")
        print(f"[BalancedAug] Per class: {dict(sorted(per_class.items()))}")

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        import numpy as np
        real_idx = self.balanced_indices[idx]
        fpath, label, weight, domain_idx = self.parent.samples[real_idx]
        seq = np.load(fpath).astype(np.float32)
        seq, proximity = ISLDataset._prepare_sequence(seq, augment=True)
        return (
            torch.from_numpy(seq),
            torch.from_numpy(proximity),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
            torch.tensor(domain_idx, dtype=torch.long),
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
        fpath, label, weight, domain_idx = self.parent.samples[self.indices[idx]]
        seq = np.load(fpath).astype(np.float32)
        seq, proximity = ISLDataset._prepare_sequence(seq, augment=False)
        return (
            torch.from_numpy(seq),
            torch.from_numpy(proximity),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
            torch.tensor(domain_idx, dtype=torch.long),
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    use_mixup: bool = True,
    domain_criterion: nn.Module = None,
    epoch: int = 1,
    total_epochs: int = 1,
) -> tuple:
    """Train one epoch with optional Mixup; return (loss, accuracy%)."""
    model.train()
    running_loss = 0.0
    correct = 0
    domain_correct = 0
    total = 0

    total_steps = len(loader)
    for i, (sequences, proximity, labels, weights, domains) in enumerate(loader):
        sequences = sequences.to(DEVICE)
        proximity = proximity.to(DEVICE)
        labels = labels.to(DEVICE)
        weights = weights.to(DEVICE)
        domains = domains.to(DEVICE)

        optimizer.zero_grad()
        
        # Calculate lambda_val for GRL smoothly over steps
        p = float(epoch - 1 + i / total_steps) / total_epochs
        lambda_val = 2. / (1. + np.exp(-10. * p)) - 1.

        # Apply Mixup or CutMix if enabled
        if use_mixup and USE_MIXUP and np.random.rand() < MIXUP_PROB:
            # Mixup: interpolate random training pairs
            mixed_x, y_a, y_b, lam = mixup_data(sequences, labels, alpha=MIXUP_ALPHA)
            outputs = model(mixed_x, proximity=proximity, lambda_val=lambda_val)
            sign_logits = outputs["sign_logits"]
            # Per-sample losses (reduction='none')
            loss_a = criterion(sign_logits, y_a)
            loss_b = criterion(sign_logits, y_b)
            per_sample = lam * loss_a + (1 - lam) * loss_b
            # Apply sample weights and reduce
            sign_loss = (per_sample * weights).mean()

            domain_loss = 0.0
            if domain_criterion is not None and outputs["domain_logits"] is not None:
                domain_loss = domain_criterion(outputs["domain_logits"], domains)
                domain_preds = outputs["domain_logits"].argmax(dim=1)
                domain_correct += (domain_preds == domains).sum().item()

            loss = sign_loss + domain_loss

            # Accuracy on original labels (approximate - weighted sum)
            preds = sign_logits.argmax(dim=1)
            correct += (lam * (preds == y_a).float().sum().item()
                        + (1 - lam) * (preds == y_b).float().sum().item())
        else:
            # Standard training: get per-sample losses then weight
            outputs = model(sequences, proximity=proximity, lambda_val=lambda_val)
            sign_logits = outputs["sign_logits"]
            per_sample = criterion(sign_logits, labels)
            sign_loss = (per_sample * weights).mean()

            domain_loss = 0.0
            if domain_criterion is not None and outputs["domain_logits"] is not None:
                domain_loss = domain_criterion(outputs["domain_logits"], domains)
                domain_preds = outputs["domain_logits"].argmax(dim=1)
                domain_correct += (domain_preds == domains).sum().item()

            loss = sign_loss + domain_loss

            preds = sign_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )

        optimizer.step()

        running_loss += sign_loss.item() * sequences.size(0)
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total, 100.0 * domain_correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    domain_criterion: nn.Module = None,
) -> tuple:
    """Validate model. Returns (avg_loss, accuracy%)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    domain_correct = 0
    total = 0

    for sequences, proximity, labels, weights, domains in loader:
        sequences = sequences.to(DEVICE)
        proximity = proximity.to(DEVICE)
        labels = labels.to(DEVICE)
        weights = weights.to(DEVICE)
        domains = domains.to(DEVICE)

        outputs = model(sequences, proximity=proximity)
        sign_logits = outputs["sign_logits"]
        per_sample = criterion(sign_logits, labels)
        loss = (per_sample * weights).mean()

        running_loss += loss.item() * sequences.size(0)
        preds = sign_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

        if domain_criterion is not None and outputs["domain_logits"] is not None:
            domain_preds = outputs["domain_logits"].argmax(dim=1)
            domain_correct += (domain_preds == domains).sum().item()

        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total, 100.0 * domain_correct / total


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    class_weights: torch.Tensor = None,
    classes_list: list = None,
    pipeline_log: PipelineLogger | None = None,
    *,
    num_domains: int = 0,
    epochs: int | None = None,
    pretrained_checkpoint: str | None = None,
    lr: float | None = None,
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
    if pipeline_log is not None:
        pipeline_log.event(
            "train_loop_start",
            num_classes=num_classes,
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            class_weights=[float(x) for x in class_weights.tolist()] if class_weights is not None else None,
            use_focal_loss=USE_FOCAL_LOSS,
        )

    # Allow caller to override number of epochs and learning rate for fine-tuning
    epochs = int(epochs) if epochs is not None else NUM_EPOCHS
    effective_lr = float(lr) if lr is not None else LEARNING_RATE

    model = SignLanguageGRU(num_classes=num_classes, num_domains=num_domains).to(DEVICE)

    # If a pretrained checkpoint is supplied, load weights before training
    if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
        try:
            ck = torch.load(pretrained_checkpoint, map_location=DEVICE)
            model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
            print(f"[Train] Loaded weights from: {pretrained_checkpoint}")
        except Exception as e:
            print(f"[Train] Warning: failed to load checkpoint {pretrained_checkpoint}: {e}")

    # Select loss function (Focal Loss for hard sample mining, else CE)
    # Create criterion that returns per-sample losses (reduction='none'),
    # we will apply per-sample weights from the dataset before taking the mean.
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            weight=class_weights,
            reduction='none',
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
            reduction='none',
        )

    domain_criterion = None
    if num_domains > 0:
        domain_criterion = nn.CrossEntropyLoss()

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
        f"{'VaLoss':>7} | {'VaAcc':>6} | {'LR':>8} | {'T':>4} | {'DomAcc':>6}"
    )
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_dom_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            domain_criterion=domain_criterion, epoch=epoch, total_epochs=epochs
        )
        va_loss, va_acc, va_dom_acc = validate(
            model, val_loader, criterion, domain_criterion=domain_criterion
        )

        # Step scheduler based on val accuracy
        scheduler.step(va_acc)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if pipeline_log is not None:
            pipeline_log.event(
                "train_epoch_end",
                epoch=epoch,
                train_loss=round(float(tr_loss), 6),
                train_acc=round(float(tr_acc), 3),
                val_loss=round(float(va_loss), 6),
                val_acc=round(float(va_acc), 3),
                lr=round(float(cur_lr), 10),
                elapsed_sec=round(float(elapsed), 3),
                best_val_acc=round(float(best_val_acc), 3),
            )

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
            if pipeline_log is not None:
                pipeline_log.event(
                    "train_best_checkpoint",
                    epoch=epoch,
                    val_acc=round(float(va_acc), 3),
                    model_path=MODEL_SAVE_PATH,
                )
            marker = " *"
        else:
            no_improve += 1

        dom_str = f" | {tr_dom_acc:>5.1f}%" if num_domains > 0 else ""
        print(
            f" {epoch:>3} | {tr_loss:>7.4f} | "
            f"{tr_acc:>5.1f}% | {va_loss:>7.4f} | "
            f"{va_acc:>5.1f}% | {cur_lr:.1e} | "
            f"{elapsed:>3.1f}s{marker}{dom_str}"
        )

        # Early stopping
        if no_improve >= PATIENCE:
            print(
                f"\n[Early Stop] No improvement for "
                f"{PATIENCE} epochs. Stopping."
            )
            if pipeline_log is not None:
                pipeline_log.event(
                    "train_early_stop",
                    epoch=epoch,
                    patience=PATIENCE,
                    best_val_acc=round(float(best_val_acc), 3),
                )
            break

    print(f"\n[Train] Best val accuracy: {best_val_acc:.2f}%")
    print(f"[Train] Model saved to: {MODEL_SAVE_PATH}")
    if pipeline_log is not None:
        pipeline_log.event(
            "train_loop_end",
            best_val_acc=round(float(best_val_acc), 3),
            model_path=MODEL_SAVE_PATH,
        )

    # Load best weights
    ckpt = torch.load(
        MODEL_SAVE_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    return model


# ── K-Fold Cross-Validation Training ──────────────────────────────


def _train_fold(
    full_ds: ISLDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_classes: int,
    fold: int,
    save_path: str,
    pipeline_log: PipelineLogger | None = None,
    *,
    epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> float:
    """
    Train a single fold with balanced oversampling + weighted CE loss.
    Returns best validation accuracy.
    """
    labels = np.array([_sample_label(sample) for sample in full_ds.samples])

    # Balanced oversampling for training split
    train_ds = _BalancedAugSubset(full_ds, train_idx.tolist())
    if pipeline_log is not None:
        train_counts = defaultdict(int)
        for idx in train_idx:
            train_counts[int(labels[idx])] += 1
        val_counts = defaultdict(int)
        for idx in val_idx:
            val_counts[int(labels[idx])] += 1
        pipeline_log.event(
            "kfold_fold_start",
            fold=fold,
            train_size=int(len(train_idx)),
            val_size=int(len(val_idx)),
            train_class_counts=dict(train_counts),
            val_class_counts=dict(val_counts),
        )
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

    best_val_acc = 0.0
    print(f"[FOLD {fold}] Starting training (epochs={epochs}, lr={learning_rate})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=LR_MIN)

    no_improve = 0
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = validate(model, val_loader, criterion)
        scheduler.step(epoch)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if pipeline_log is not None:
            pipeline_log.event(
                "kfold_fold_epoch_end",
                fold=fold,
                epoch=epoch,
                train_loss=round(float(tr_loss), 6),
                train_acc=round(float(tr_acc), 3),
                val_loss=round(float(va_loss), 6),
                val_acc=round(float(va_acc), 3),
                lr=round(float(cur_lr), 10),
                elapsed_sec=round(float(elapsed), 3),
                best_val_acc=round(float(best_val_acc), 3),
            )

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
            if pipeline_log is not None:
                pipeline_log.event(
                    "kfold_fold_best_checkpoint",
                    fold=fold,
                    epoch=epoch,
                    val_acc=round(float(va_acc), 3),
                    model_path=save_path,
                )
            marker = " *"
        else:
            no_improve += 1

        if epoch % 10 == 0 or marker:
            print(f"   F{fold} Ep {epoch:>3} | tr {tr_acc:>5.1f}% | va {va_acc:>5.1f}% | lr {cur_lr:.1e} | {elapsed:.1f}s{marker}")

        if no_improve >= PATIENCE:
            print(f"   F{fold} Early stop at epoch {epoch}")
            if pipeline_log is not None:
                pipeline_log.event(
                    "kfold_fold_early_stop",
                    fold=fold,
                    epoch=epoch,
                    patience=PATIENCE,
                    best_val_acc=round(float(best_val_acc), 3),
                )
            break

    if pipeline_log is not None:
        pipeline_log.event(
            "kfold_fold_end",
            fold=fold,
            best_val_acc=round(float(best_val_acc), 3),
            model_path=save_path,
        )

    return best_val_acc


def train_kfold(
    pipeline_log: PipelineLogger | None = None,
    start_fold: int = 0,
    *,
    neg_root: str | None = None,
    archived_weight: float | None = None,
    epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> list:
    """
    Train NUM_FOLDS models using disjoint K-fold CV.
    Saves each fold model to ENSEMBLE_DIR/fold_N.pth.
    Returns list of per-fold best validation accuracies.
    
    Args:
        pipeline_log: Optional logger for pipeline events
        start_fold: Starting fold index (0-indexed). Default: 0
    """
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)

    full_ds = ISLDataset(augment=False, min_samples=2, neg_root=neg_root)
    num_classes = full_ds.num_classes
    labels = np.array([_sample_label(sample) for sample in full_ds.samples])
    manifest = _init_kfold_manifest(
        start_fold=start_fold,
        num_folds=NUM_FOLDS,
        dataset_size=len(full_ds),
        num_classes=num_classes,
    )

    # Build disjoint folds with class-balanced random partitioning.
    folds = _build_disjoint_folds(
        full_ds.samples, labels, NUM_FOLDS, RANDOM_SEED
    )
    split_iter = []
    for fold in range(NUM_FOLDS):
        val_idx = folds[fold]
        train_idx = np.setdiff1d(np.arange(len(labels)), val_idx)
        split_iter.append((train_idx, val_idx))
    print(f"[KFold] Using disjoint K-fold (num_folds={NUM_FOLDS})")

    fold_accs = []
    all_val_correct = 0
    all_val_total = 0

    print(f"\n{'='*60}")
    print(f"  K-Fold Cross-Validation ({NUM_FOLDS} folds)")
    print(f"  Dataset: {len(full_ds)} samples, {num_classes} classes")
    print(f"{'='*60}\n")
    if pipeline_log is not None:
        pipeline_log.event(
            "kfold_start",
            num_folds=NUM_FOLDS,
            dataset_size=int(len(full_ds)),
            num_classes=int(num_classes),
        )

    total_p = sum(
        p.numel() for p in SignLanguageGRU(num_classes).parameters()
    )
    print(f"[Model] Params per fold: {total_p:,}\n")

    for fold, (train_idx, val_idx) in enumerate(split_iter[start_fold:], start=start_fold):
        save_path = os.path.join(ENSEMBLE_DIR, f"fold_{fold}.pth")
        print(
            f"--- Fold {fold} ---  "
            f"train={len(train_idx)}  val={len(val_idx)}"
        )

        best_acc = _train_fold(
            full_ds,
            train_idx,
            val_idx,
            num_classes,
            fold,
            save_path,
            pipeline_log=pipeline_log,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        fold_accs.append(best_acc)
        _record_kfold_fold(manifest, fold, best_acc, save_path)
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
    if pipeline_log is not None:
        pipeline_log.event(
            "kfold_end",
            fold_accuracies=[round(float(x), 3) for x in fold_accs],
            mean_accuracy=round(float(avg_acc), 3),
            overall_accuracy=round(float(overall_acc), 3),
        )
    _finalize_kfold_manifest(manifest, fold_accs)
    return fold_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training runner (single split or K-fold)")
    parser.add_argument('--kfold', type=int, default=0, help='Run K-fold cross-validation with this many folds (overrides config)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--neg-root', type=str, default=None, help='Path to processed_negatives root to include as reject class')
    args = parser.parse_args()

    globals()['NUM_EPOCHS'] = int(args.epochs)
    globals()['LEARNING_RATE'] = float(args.lr)

    # If K-fold requested, run k-fold pipeline
    if args.kfold and args.kfold > 0:
        # Override module NUM_FOLDS for this run
        globals()['NUM_FOLDS'] = int(args.kfold)
        print(f"[Main] Running K-Fold with {globals()['NUM_FOLDS']} folds")
        fold_accs = train_kfold(
            pipeline_log=None,
            start_fold=0,
            neg_root=args.neg_root,
            epochs=args.epochs,
            learning_rate=args.lr,
        )
        # Summary
        avg = float(np.mean(fold_accs)) if fold_accs else 0.0
        std = float(np.std(fold_accs)) if fold_accs else 0.0
        best_idx = int(np.argmax(fold_accs)) if fold_accs else -1
        print(f"K-Fold summary: mean={avg:.2f}% std={std:.2f}% best_fold={best_idx} best_acc={fold_accs[best_idx] if best_idx>=0 else None}")
        raise SystemExit(0)

    # Single-run training
    train_loader, val_loader, num_classes, class_weights, full_ds = create_data_loaders(neg_root=args.neg_root)
    model = train(
        train_loader, val_loader, num_classes, class_weights,
        classes_list=full_ds.classes, num_domains=len(full_ds.domains)
    )
    print(f"[Done] Training complete. Model saved to {MODEL_SAVE_PATH}")
