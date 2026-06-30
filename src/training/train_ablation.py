import os
import sys
import copy
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.core.config import get_config
from src.preprocessing.dataset import ISLDataset
from src.training.train import train_one_epoch, validate
from src.training.model import SignLanguageGRU

cfg = get_config()
DEVICE = cfg.hardware.torch_device
NUM_EPOCHS = cfg.training.num_epochs

def run_ablation():
    print("="*60)
    print("DANN ABLATION STUDY: DOMAIN HOLDOUT")
    print("="*60)
    
    # 1. Load Dataset
    print("[1] Loading Full Dataset via HDF5")
    full_ds = ISLDataset(augment=False, oversample=False)
    print(f"Loaded {len(full_ds)} samples.")
    
    # Identify domains
    if 'webcam' not in full_ds.domain_to_idx:
        print("ERROR: 'webcam' domain not found. Found:", full_ds.domains)
        return
        
    webcam_idx = full_ds.domain_to_idx['webcam']
    unknown_idx = full_ds.domain_to_idx.get('unknown', -1)
    
    # 2. Split dataset
    print(f"[2] Splitting Dataset (webcam_idx={webcam_idx}, unknown_idx={unknown_idx})")
    
    # We must access h5 directly since use_hdf5=True doesn't expose self.samples
    # Actually, ISLDataset __getitem__ can be used, but we need the domains of all samples.
    # Fortunately, domains are stored in HDF5.
    import h5py
    with h5py.File(full_ds.h5_path, 'r') as f:
        domains_all = f['domains'][:]
        labels_all = f['labels'][:]
        
    train_val_indices = []
    test_indices = []
    
    for i, d in enumerate(domains_all):
        if d == webcam_idx:
            test_indices.append(i)
        elif d != unknown_idx:
            train_val_indices.append(i)
            
    print(f"Train/Val pool (MVI+cvae): {len(train_val_indices)} samples")
    print(f"Test pool (webcam): {len(test_indices)} samples")
    
    # Split train/val deterministically (e.g., 80/20)
    from sklearn.model_selection import train_test_split
    # We stratify by label to ensure all classes are present
    tv_labels = labels_all[train_val_indices]
    
    # Handle classes with only 1 sample in tv_labels safely
    unique, counts = np.unique(tv_labels, return_counts=True)
    rare_classes = unique[counts < 2]
    
    train_idx, val_idx = train_test_split(
        train_val_indices, 
        test_size=0.2, 
        random_state=42, 
        stratify=tv_labels if len(rare_classes) == 0 else None
    )
    
    # Wrap in Subset and DataLoader
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(full_ds, test_indices), batch_size=cfg.training.batch_size, shuffle=False)
    
    nc = len(full_ds.classes)
    num_domains = len(full_ds.domains)
    print(f"[3] Configuration: {nc} classes, {num_domains} domains")
    
    models = ["pure_baseline", "dann_off", "full_dann"]
    seeds = [42, 123, 777]
    
    os.makedirs("checkpoints/ablation", exist_ok=True)
    
    for seed in seeds:
        for model_type in models:
            print(f"\n--- Training {model_type.upper()} | Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            if model_type == "pure_baseline":
                model = SignLanguageGRU(num_classes=nc, num_domains=0).to(DEVICE)
                use_grl = False
            elif model_type == "dann_off":
                model = SignLanguageGRU(num_classes=nc, num_domains=num_domains).to(DEVICE)
                use_grl = False
            else: # full_dann
                model = SignLanguageGRU(num_classes=nc, num_domains=num_domains).to(DEVICE)
                use_grl = True
                
            criterion = nn.CrossEntropyLoss().to(DEVICE)
            domain_criterion = nn.CrossEntropyLoss().to(DEVICE) if model_type != "pure_baseline" else None
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
            
            best_val_acc = 0.0
            
            # Fast training for ablation
            for epoch in range(1, NUM_EPOCHS + 1):
                # For pure_baseline/dann_off, we pass epoch=1, total=1 but they don't use GRL anyway
                tr_loss, tr_acc, tr_dom_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer,
                    domain_criterion=domain_criterion, 
                    epoch=epoch if use_grl else 1, 
                    total_epochs=NUM_EPOCHS if use_grl else 1
                )
                
                va_loss, va_acc, va_dom_acc = validate(
                    model, val_loader, criterion, domain_criterion=domain_criterion
                )
                
                # Check performance on unseen domain
                test_loss, test_acc, test_dom_acc = validate(
                    model, test_loader, criterion, domain_criterion=domain_criterion
                )
                
                if va_acc > best_val_acc:
                    best_val_acc = va_acc
                    torch.save(model.state_dict(), f"checkpoints/ablation/{model_type}_{seed}.pt")
                    
                print(f"Ep {epoch:>2}/{NUM_EPOCHS} | ValAcc {va_acc:>5.1f}% | UnseenAcc {test_acc:>5.1f}% | UnseenDomAcc {test_dom_acc:>5.1f}%")

if __name__ == "__main__":
    run_ablation()
