"""
Quick grid-search over archived weights.
Runs short training for each weight and records validation accuracy and per-class recall.
Outputs CSV to `tools/grid_search_results.csv`.

Usage: python tools/grid_search_archived.py
"""

import os
import sys
import csv
import numpy as np
import torch

# Ensure project root is on sys.path so local modules import correctly
# (__file__ is in src/tools, so project root is two directories up)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.train import create_data_loaders, train
from src.preprocessing.dataset import ISLDataset

WEIGHTS = [0.0, 0.05, 0.1, 0.25, 0.5]
EPOCHS = 2
NEG_ROOT = "processed_negatives_del"
INCLUDE_ARCHIVED = True
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "grid_search_results.csv")

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_per_class_recall(model, full_ds, val_indices, classes):
    # Compute predictions for each val index by loading from full_ds
    model.eval()
    preds = []
    trues = []
    archived_flags = []
    with torch.no_grad():
        for idx in val_indices:
            fpath, label, weight = full_ds.samples[idx]
            seq = np.load(fpath).astype(np.float32)
            seq, proximity = ISLDataset._prepare_sequence(seq, augment=False)
            seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
            prox_t = torch.from_numpy(proximity).unsqueeze(0).to(device)
            logits = model(seq_t, proximity=prox_t)
            pred = int(logits.argmax(dim=1).item())
            preds.append(pred)
            trues.append(int(label))
            archived_flags.append('processed_del' in fpath.replace('\\','/'))

    preds = np.array(preds)
    trues = np.array(trues)
    archived_flags = np.array(archived_flags)

    num_classes = len(classes)
    per_class_recall = []
    for c in range(num_classes):
        mask = trues == c
        if mask.sum() == 0:
            per_class_recall.append(None)
        else:
            per_class_recall.append(float((preds[mask] == trues[mask]).sum() / mask.sum()))

    # Archived subset recall
    if archived_flags.sum() > 0:
        arch_recall = float((preds[archived_flags] == trues[archived_flags]).sum() / archived_flags.sum())
    else:
        arch_recall = None
    non_arch_mask = ~archived_flags
    if non_arch_mask.sum() > 0:
        non_arch_recall = float((preds[non_arch_mask] == trues[non_arch_mask]).sum() / non_arch_mask.sum())
    else:
        non_arch_recall = None

    return per_class_recall, arch_recall, non_arch_recall


def run_once(arch_w):
    print(f"\n=== Running archived_weight={arch_w} ===")
    # Build data loaders including archived samples
    tl, vl, nc, cw, full_ds = create_data_loaders(neg_root=NEG_ROOT, archived_weight=arch_w, include_archived=INCLUDE_ARCHIVED)

    # Extract val indices from val loader
    val_ds = vl.dataset
    val_indices = val_ds.indices

    # Train for a few epochs
    model = train(
        tl, vl, nc, cw, classes_list=full_ds.classes, epochs=EPOCHS, 
        num_domains=len(full_ds.domains)
    )

    # Evaluate per-class recall and archived vs non-archived recall
    pcr, arch_r, non_arch_r = evaluate_per_class_recall(model, full_ds, val_indices, full_ds.classes)

    # Get final val accuracy by running validate directly
    from src.training.train import validate, nn
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.0, reduction='none')
    va_loss, va_acc = validate(model, vl, crit)

    return {
        'archived_weight': arch_w,
        'val_acc': va_acc,
        'archived_recall': arch_r,
        'non_archived_recall': non_arch_r,
        'per_class_recall': pcr,
    }


def main():
    results = []
    for w in WEIGHTS:
        try:
            res = run_once(w)
            results.append(res)
        except Exception as e:
            print(f"Run failed for weight={w}: {e}")
            results.append({'archived_weight': w, 'error': str(e)})

    # Write CSV summary
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['archived_weight', 'val_acc', 'archived_recall', 'non_archived_recall', 'per_class_recall'])
        for r in results:
            writer.writerow([
                r.get('archived_weight'),
                r.get('val_acc'),
                r.get('archived_recall'),
                r.get('non_archived_recall'),
                ' | '.join(['NA' if v is None else f"{v:.3f}" for v in (r.get('per_class_recall') or [])]) if isinstance(r.get('per_class_recall'), list) else r.get('per_class_recall')
            ])

    print(f"\nGrid search complete. Results saved to: {RESULTS_CSV}")

if __name__ == '__main__':
    main()
