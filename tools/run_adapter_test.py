"""
Offline adapter training & validation test.

Usage:
    python tools/run_adapter_test.py

This script:
- Loads the merged ensemble
- Loads pseudo-labelled sequences from `pseudo_data/` (class subfolders)
- Re-runs `merged_ensemble_predict` to compute ensemble probs per sequence
- Trains `AdapterTrainer` on these probs with a small validation split
- Prints before/after avg max-prob and top-1 accuracy on validation
"""

import os
import numpy as np
import torch
from collections import Counter

# Ensure project root is on path so imports work when running from tools/
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adapter_model import AdapterTrainer
from ensemble import load_merged_ensemble_10_2, merged_ensemble_predict

ROOT = os.path.dirname(os.path.dirname(__file__))
PSEUDO_DIR = os.path.join(ROOT, 'pseudo_data')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_sequences(pseudo_dir, per_class_cap=50):
    samples = []  # (class_name, seq)
    if not os.path.isdir(pseudo_dir):
        print('[run_adapter_test] No pseudo_data directory found')
        return samples
    for cls in sorted(os.listdir(pseudo_dir)):
        cls_dir = os.path.join(pseudo_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in sorted(os.listdir(cls_dir)) if f.endswith('.npy')]
        for f in files[:per_class_cap]:
            path = os.path.join(cls_dir, f)
            try:
                seq = np.load(path)
                samples.append((cls, seq))
            except Exception as e:
                print('Failed to load', path, e)
    return samples


def compute_ensemble_probs(main_models, fallback_models, seq):
    try:
        res = merged_ensemble_predict(main_models, fallback_models, seq, use_tta=False)
        probs = np.array(res['probs'], dtype=np.float32)
        return probs
    except Exception as e:
        print('[run_adapter_test] ensemble predict failed:', e)
        return None


def main():
    print('[run_adapter_test] Loading ensemble...')
    main_models, fallback_models, classes, _ = load_merged_ensemble_10_2()
    num_classes = len(classes)
    print(f'[run_adapter_test] Ensemble loaded: {len(main_models)} main + fallback, {num_classes} classes')

    samples = load_sequences(PSEUDO_DIR, per_class_cap=50)
    if not samples:
        print('[run_adapter_test] No samples to train on. Exiting.')
        return

    # Map class names -> indices
    class_to_idx = {name: i for i, name in enumerate(classes)}

    probs_list = []
    labels = []
    for cls_name, seq in samples:
        if cls_name not in class_to_idx:
            continue
        probs = compute_ensemble_probs(main_models, fallback_models, seq)
        if probs is None:
            continue
        probs_list.append(probs)
        labels.append(class_to_idx[cls_name])

    if len(probs_list) < 20:
        print('[run_adapter_test] Not enough valid samples after filtering:', len(probs_list))
        return

    probs_arr = np.stack(probs_list, axis=0)
    labels_arr = np.array(labels, dtype=np.int64)

    # Shuffle and split
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(probs_arr))
    probs_arr = probs_arr[perm]
    labels_arr = labels_arr[perm]

    n = len(probs_arr)
    n_train = int(n * 0.8)
    train_probs = probs_arr[:n_train]
    train_labels = labels_arr[:n_train]
    val_probs = probs_arr[n_train:]
    val_labels = labels_arr[n_train:]

    print(f'[run_adapter_test] Samples: total={n}, train={len(train_probs)}, val={len(val_probs)}')
    # Print class distribution in training data
    from collections import Counter
    train_dist = Counter(train_labels.tolist())
    print('[run_adapter_test] Train class distribution (idx:count):')
    for k, v in train_dist.most_common():
        name = classes[k] if k < len(classes) else f'class_{k}'
        print(f'  {k:3d} {name:20s}: {v}')

    # Adapter trainer
    trainer = AdapterTrainer(num_classes=num_classes, device=str(DEVICE), learning_rate=1e-4, hidden_dim=128)

    before_conf = trainer.evaluate_confidence(val_probs)[0]
    print(f'[run_adapter_test] Before avg max-prob (val): {before_conf:.4f}')

    # Show a few validation sample top-3 before training
    print('\n[run_adapter_test] Sample val predictions BEFORE training:')
    for i in range(min(5, len(val_probs))):
        p = val_probs[i]
        top3 = np.argsort(-p)[:3]
        print(f'  idx={i} true={classes[val_labels[i]]} top3=', [ (classes[t], float(p[t])) for t in top3 ])

    # Train
    result = trainer.train(train_probs.tolist(), train_labels.tolist(), epochs=10, batch_size=8, verbose=True)
    print('[run_adapter_test] Training result:', result.get('history', {}))

    after_conf = trainer.evaluate_confidence(val_probs)[1]
    print(f'[run_adapter_test] After avg max-prob (val): {after_conf:.4f}')

    # Optional: compute top-1 accuracy before/after
    from numpy import argmax
    # before preds
    before_preds = np.argmax(val_probs, axis=1)
    before_acc = (before_preds == val_labels).mean()
    # after preds
    logits = trainer.model(torch.from_numpy(val_probs.astype(np.float32)).to(trainer.device))
    after_probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
    after_preds = np.argmax(after_probs, axis=1)
    after_acc = (after_preds == val_labels).mean()

    print(f'[run_adapter_test] Val Top-1 Acc before: {before_acc:.4f}, after: {after_acc:.4f}')

    # Show a few validation sample top-3 after training
    print('\n[run_adapter_test] Sample val predictions AFTER training:')
    for i in range(min(5, len(val_probs))):
        p_before = val_probs[i]
        top3_before = np.argsort(-p_before)[:3]
        logits = trainer.model(torch.from_numpy(p_before.astype(np.float32)).to(trainer.device).unsqueeze(0))
        probs_after = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()[0]
        top3_after = np.argsort(-probs_after)[:3]
        print(f'  idx={i} true={classes[val_labels[i]]}')
        print(f'    before: ', [ (classes[t], float(p_before[t])) for t in top3_before ])
        print(f'    after : ', [ (classes[t], float(probs_after[t])) for t in top3_after ])

    # Save the adapter model if it improved avg confidence
    if after_conf >= before_conf:
        save_path = os.path.join('adapter_weights', 'adapter_test_improved.pt')
        trainer.save_model(save_path)
        print('[run_adapter_test] Adapter saved to', save_path)
    else:
        print('[run_adapter_test] Adapter did not improve confidence; not saved')

if __name__ == '__main__':
    main()
