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
                # Apply boundary noise augmentation
                seq_np = apply_boundary_noise(
                    sequence=seq_np, 
                    label=lbl, 
                    dataset=self.base_dataset, 
                    edge_frames=BOUNDARY_EDGE_FRAMES, 
                    probability=BOUNDARY_NOISE_PROB
                )
                seq_t = torch.from_numpy(seq_np)
                
            return seq_t, prox_t, lbl_t, weight_t, domain_t
        else:
            # Fetch from synthetic transition samples
            trans_idx = idx - len(self.indices)
            seq_np, lbl = self.transitions[trans_idx]
            
            # Align input size and compute proximity if needed
            seq_np, proximity = ISLDataset._prepare_sequence(seq_np, augment=False)
            
            return (
                torch.from_numpy(seq_np),
                torch.from_numpy(proximity),
                torch.tensor(lbl, dtype=torch.long),
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(0, dtype=torch.long)
            )


def main():
    parser = argparse.ArgumentParser(description="Train Continuous Sign Language Model")
    args = parser.parse_args()

    print("[Continuous] Loading base dataset...")
    # Load base dataset without internal augmentation (we apply it in ContinuousDataset)
    full_ds = ISLDataset(augment=False)
    
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
        
    print(f"[Continuous] Starting training with {num_classes} classes...")
    print(f"[Continuous] Will save model to: {cfg.paths.model_save_path}")
    
    # Run the existing train loop
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        class_weights=class_weights,
        classes_list=classes_list
    )

if __name__ == "__main__":
    main()
