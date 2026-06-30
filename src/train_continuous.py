import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Adjust paths to allow imports if run directly from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import get_config
cfg = get_config()

# Monkey-patch config to save the continuous model in a separate file
cfg.paths.model_save_path = os.path.join(
    os.path.dirname(cfg.paths.model_save_path), 
    "sign_language_continuous.pth"
)

from src.training.train import (
    train, 
    _compute_inverse_class_weights, 
    _disjoint_stratified_split
)
from src.preprocessing.dataset import ISLDataset
from src.augmentations1.boundary_noise import apply_boundary_noise
from src.augmentations1.transition_generator import generate_transition_dataset
from src.config.continuous_signing import (
    BOUNDARY_NOISE_PROB,
    BOUNDARY_EDGE_FRAMES,
)

class ContinuousDataset(Dataset):
    """
    Wraps the ISLDataset to apply boundary noise on-the-fly and 
    serve pre-generated synthetic transition samples for the reject class.
    """
    def __init__(self, base_dataset, indices, transitions=None, is_train=True):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transitions = transitions or []
        self.is_train = is_train

    def __len__(self):
        return len(self.indices) + len(self.transitions)

    def __getitem__(self, idx):
        if idx < len(self.indices):
            # Fetch from original dataset
            real_idx = self.indices[idx]
            seq_t, prox_t, lbl_t, weight_t, domain_t = self.base_dataset[real_idx]
            
            if self.is_train:
                seq_np = seq_t.numpy()
                lbl = lbl_t.item()
                # Apply standard spatial and temporal augmentations first!
                seq_np = ISLDataset._augment(seq_np)
                
                # Apply continuous-specific boundary noise augmentation
                seq_np = apply_boundary_noise(
                    sequence=seq_np, 
                    label=lbl, 
                    dataset=self.base_dataset, 
                    edge_frames=BOUNDARY_EDGE_FRAMES, 
                    probability=BOUNDARY_NOISE_PROB
                )
                
                # Because we mutated the sequence, we MUST re-extract proximity
                proximity = ISLDataset._extract_proximity(seq_np)
                seq_t = torch.from_numpy(seq_np)
                prox_t = torch.from_numpy(proximity)
                
            return seq_t, prox_t, lbl_t, weight_t, domain_t
        else:
            # Fetch from synthetic transition samples
            trans_idx = idx - len(self.indices)
            seq_np, lbl = self.transitions[trans_idx]
            
            # Align input size, apply standard augmentations if training, and compute proximity
            seq_np, proximity = ISLDataset._prepare_sequence(seq_np, augment=self.is_train)
            
            return (
                torch.from_numpy(seq_np),
                torch.from_numpy(proximity),
                torch.tensor(lbl, dtype=torch.long),
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(0, dtype=torch.long)
            )


def main():
    parser = argparse.ArgumentParser(description="Train Continuous Sign Language Model")
    parser.add_argument("--archived-weight", type=float, default=0.25, help="Weight for archived samples")
    args = parser.parse_args()

    print("[Continuous] Loading base dataset (Phase 1)...")
    # Load base dataset without internal augmentation (we apply it in ContinuousDataset)
    neg_root_p1 = os.path.join(os.path.dirname(cfg.paths.processed_dir), "processed_negatives")
    full_ds = ISLDataset(augment=False, neg_root=neg_root_p1)
    
    # Extract labels for stratified split
    if getattr(full_ds, 'use_hdf5', False):
        full_ds._ensure_open()
        labels = np.array(full_ds.h5["labels"])
    else:
        labels = np.array([s[1] for s in full_ds.samples])
    
    print("[Continuous] Splitting base dataset into train/val...")
    train_idx, val_idx = _disjoint_stratified_split(
        full_ds.samples if not getattr(full_ds, 'use_hdf5', False) else None,
        labels,
        cfg.training.val_split,
        cfg.training.random_seed,
    )
    
    print("[Continuous] Generating synthetic transition dataset...")
    transitions = generate_transition_dataset(full_ds)
    
    print(f"[Continuous] Generated {len(transitions)} transition samples.")
    
    # Split transitions into train and val
    trans_split = int(len(transitions) * (1 - cfg.training.val_split))
    train_transitions = transitions[:trans_split]
    val_transitions = transitions[trans_split:]
    
    # Check if a new class was added for reject
    reject_idx = full_ds.class_to_idx.get("__reject__", full_ds.num_classes)
    num_classes = max(full_ds.num_classes, reject_idx + 1)
    
    train_ds = ContinuousDataset(full_ds, train_idx.tolist(), train_transitions, is_train=True)
    val_ds = ContinuousDataset(full_ds, val_idx.tolist(), val_transitions, is_train=False)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.training.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Compute inverse frequency class weights to handle imbalance (including transitions)
    all_train_labels = np.concatenate([
        labels[train_idx],
        np.array([lbl for _, lbl in train_transitions])
    ])
    class_weights = _compute_inverse_class_weights(all_train_labels, num_classes)
    
    classes_list = full_ds.classes.copy()
    if reject_idx == full_ds.num_classes:
        classes_list.append("__reject__")
        
    print(f"[Continuous] Starting Phase 1 training with {num_classes} classes...")
    print(f"[Continuous] Will save model to: {cfg.paths.model_save_path}")
    
    # Run Phase 1 train loop
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        class_weights=class_weights,
        classes_list=classes_list
    )

    # --- PHASE 2 ---
    finetune_epochs = getattr(cfg.training, 'finetune_archived_epochs', 0)
    if finetune_epochs and int(finetune_epochs) > 0:
        processed_del = os.path.join(os.path.dirname(cfg.paths.processed_dir), "processed_del")
        neg_del = os.path.join(os.path.dirname(cfg.paths.processed_dir), "processed_negatives_del")
        
        if os.path.isdir(processed_del) or os.path.isdir(neg_del):
            print(f"\n[Continuous Phase 2] Loading archived datasets...")
            full_ds_p2 = ISLDataset(
                augment=False, 
                neg_root=neg_root_p1,
                archived_root=processed_del, 
                archived_neg_root=neg_del,
                archived_weight=args.archived_weight
            )
            
            # Map Phase 1 train files to Phase 2 indices to keep validation strictly clean
            clean_train_paths = set(full_ds.samples[i][0] for i in train_idx)
            
            train_idx_p2 = []
            for i, s in enumerate(full_ds_p2.samples):
                if s[0] in clean_train_paths or processed_del in s[0] or neg_del in s[0]:
                    train_idx_p2.append(i)
                    
            print(f"[Continuous Phase 2] Fine-tuning on {len(train_idx_p2)} samples (including archived) for {finetune_epochs} epochs.")
            
            train_ds_p2 = ContinuousDataset(full_ds_p2, train_idx_p2, train_transitions, is_train=True)
            train_loader_p2 = DataLoader(
                train_ds_p2, 
                batch_size=cfg.training.batch_size, 
                shuffle=True, 
                num_workers=0
            )
            
            # Recompute weights for Phase 2
            p2_labels = np.array([s[1] for s in full_ds_p2.samples])
            all_train_labels_p2 = np.concatenate([
                p2_labels[train_idx_p2],
                np.array([lbl for _, lbl in train_transitions])
            ])
            class_weights_p2 = _compute_inverse_class_weights(all_train_labels_p2, num_classes)
            
            ft_lr = getattr(cfg.training, 'finetune_archived_lr', None)
            if ft_lr is not None:
                ft_lr = float(ft_lr)
                
            train(
                train_loader=train_loader_p2,
                val_loader=val_loader,  # Re-use strictly clean Phase 1 val_loader!
                num_classes=num_classes,
                class_weights=class_weights_p2,
                classes_list=classes_list,
                epochs=int(finetune_epochs),
                pretrained_checkpoint=cfg.paths.model_save_path,
                lr=ft_lr
            )
        else:
            print('\n[Continuous Phase 2] Archived folders not found; skipping fine-tune.')

if __name__ == "__main__":
    main()
