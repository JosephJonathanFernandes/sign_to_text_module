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
from config import get_config
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
    
    models = ["pure_baseline", "dann_off", "full_dann"]
    seeds = [42, 123, 777]
    
    results = defaultdict(lambda: defaultdict(list))
    confusion_pairs = defaultdict(list)
    
    for model_type in models:
        for seed in seeds:
            ckpt_path = f"checkpoints/ablation/{model_type}_{seed}.pt"
            if not os.path.exists(ckpt_path):
                print(f"Missing {ckpt_path}, skipping...")
                continue
                
            use_domain = (model_type != "pure_baseline")
            model = SignLanguageGRU(num_classes=nc, num_domains=num_domains if use_domain else 0).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            
            # Evaluate Val
            val_t, val_p, _, _ = evaluate(model, val_loader)
            val_acc = (val_t == val_p).mean() * 100
            
            # Evaluate Test (Unseen)
            t_t, t_p, d_t, d_p = evaluate(model, test_loader)
            test_acc = (t_t == t_p).mean() * 100
            macro_f1 = f1_score(t_t, t_p, average='macro') * 100
            weighted_f1 = f1_score(t_t, t_p, average='weighted') * 100
            bal_acc = balanced_accuracy_score(t_t, t_p) * 100
            
            dom_acc = 0.0
            if use_domain and len(d_t) > 0:
                dom_acc = (d_t == d_p).mean() * 100
                
            results[model_type]['val_acc'].append(val_acc)
            results[model_type]['test_acc'].append(test_acc)
            results[model_type]['macro_f1'].append(macro_f1)
            results[model_type]['weighted_f1'].append(weighted_f1)
            results[model_type]['bal_acc'].append(bal_acc)
            results[model_type]['dom_acc'].append(dom_acc)
            
            # Track confusion matrix for the first seed of each model_type
            if seed == 42:
                cm = confusion_matrix(t_t, t_p)
                for i in range(nc):
                    for j in range(nc):
                        if i != j and cm[i, j] > 0:
                            confusion_pairs[model_type].append((cm[i, j], classes[t_t[i]], classes[t_p[j]]))
                            
    # Print Table
    print("\n" + "="*80)
    print("FINAL ABLATION RESULTS (Mean ± Std)")
    print("="*80)
    header = f"{'Model':<15} | {'Val Acc':<12} | {'Unseen Acc':<12} | {'Macro F1':<12} | {'Balanced Acc':<12} | {'Domain Acc':<12}"
    print(header)
    print("-" * len(header))
    for m in models:
        r = results[m]
        if not r['val_acc']: continue
        print(f"{m:<15} | {np.mean(r['val_acc']):.1f} ± {np.std(r['val_acc']):.1f} | {np.mean(r['test_acc']):.1f} ± {np.std(r['test_acc']):.1f} | {np.mean(r['macro_f1']):.1f} ± {np.std(r['macro_f1']):.1f} | {np.mean(r['bal_acc']):.1f} ± {np.std(r['bal_acc']):.1f} | {np.mean(r['dom_acc']):.1f} ± {np.std(r['dom_acc']):.1f}")
        
    print("\n" + "="*80)
    print("TOP-10 CONFUSION PAIRS (Unseen Domain, Seed 42)")
    print("="*80)
    for m in models:
        print(f"\n{m.upper()}:")
        pairs = sorted(confusion_pairs[m], key=lambda x: x[0], reverse=True)[:10]
        for count, true_cls, pred_cls in pairs:
            print(f"  {true_cls} → {pred_cls} : {count}")

if __name__ == "__main__":
    import os
    generate_metrics()
