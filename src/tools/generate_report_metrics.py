import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import json
import h5py

sys.path.insert(0, str(Path.cwd()))
from src.core.config import get_config
from src.preprocessing.dataset import ISLDataset
from src.training.model import SignLanguageGRU
from collections import defaultdict

cfg = get_config()
DEVICE = cfg.hardware.torch_device

def load_data():
    full_ds = ISLDataset(augment=False, oversample=False)
    webcam_idx = full_ds.domain_to_idx['webcam']
    unknown_idx = full_ds.domain_to_idx.get('unknown', -1)
    
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
            
    test_loader = DataLoader(Subset(full_ds, test_indices), batch_size=cfg.training.batch_size, shuffle=False)
    val_loader = DataLoader(Subset(full_ds, train_val_indices), batch_size=cfg.training.batch_size, shuffle=False)
    
    return val_loader, test_loader, len(full_ds.classes), len(full_ds.domains), full_ds.classes

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_dom_preds = []
    all_domains = []
    
    with torch.no_grad():
        for sequences, proximity, labels, weights, domains in loader:
            outputs = model(sequences.to(DEVICE), proximity=proximity.to(DEVICE))
            preds = outputs["sign_logits"].argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
            if outputs["domain_logits"] is not None:
                d_preds = outputs["domain_logits"].argmax(dim=1).cpu().numpy()
                all_dom_preds.extend(d_preds)
                all_domains.extend(domains.numpy())
                
    return np.array(all_labels), np.array(all_preds), np.array(all_domains), np.array(all_dom_preds)

def generate_metrics():
    val_loader, test_loader, nc, num_domains, classes = load_data()
    
    model_path = "models/model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Could not find main model at {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model_nc = checkpoint.get("num_classes", nc)
    model_classes = checkpoint.get("classes", classes)
    
    print(f"Evaluating canonical model: {model_path} (Trained on {model_nc} classes)")
    model = SignLanguageGRU(num_classes=model_nc, num_domains=0).to(DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate Val
    val_t, val_p, _, _ = evaluate(model, val_loader)
    val_acc = (val_t == val_p).mean() * 100
    
    # Evaluate Test (Unseen)
    t_t, t_p, _, _ = evaluate(model, test_loader)
    test_acc = (t_t == t_p).mean() * 100
    macro_f1 = f1_score(t_t, t_p, average='macro', zero_division=0) * 100
    weighted_f1 = f1_score(t_t, t_p, average='weighted', zero_division=0) * 100
    bal_acc = balanced_accuracy_score(t_t, t_p) * 100
    
    print("\n" + "="*80)
    print("FINAL MODEL RESULTS")
    print("="*80)
    print(f"Validation Accuracy:  {val_acc:.2f}%")
    print(f"Unseen Data Accuracy: {test_acc:.2f}%")
    print(f"Macro F1 Score:       {macro_f1:.2f}%")
    print(f"Weighted F1 Score:    {weighted_f1:.2f}%")
    print(f"Balanced Accuracy:    {bal_acc:.2f}%")
    
    print("\n" + "="*80)
    print("TOP-10 CONFUSION PAIRS (Unseen Data)")
    print("="*80)
    cm = confusion_matrix(t_t, t_p)
    confusion_pairs = []
    for i in range(model_nc):
        for j in range(model_nc):
            if i != j and cm[i, j] > 0:
                true_name = model_classes[t_t[i]] if t_t[i] < len(model_classes) else str(t_t[i])
                pred_name = model_classes[t_p[j]] if t_p[j] < len(model_classes) else str(t_p[j])
                confusion_pairs.append((cm[i, j], true_name, pred_name))
                
    pairs = sorted(confusion_pairs, key=lambda x: x[0], reverse=True)[:10]
    for count, true_cls, pred_cls in pairs:
        print(f"  {true_cls} → {pred_cls} : {count} misclassifications")

if __name__ == "__main__":
    import os
    generate_metrics()
