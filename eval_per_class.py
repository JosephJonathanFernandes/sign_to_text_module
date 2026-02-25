"""Quick per-class evaluation of the ensemble."""
import numpy as np
from collections import defaultdict
from dataset import ISLDataset
from ensemble import load_ensemble, ensemble_predict

ds = ISLDataset("processed", augment=False)
models, classes, num_classes = load_ensemble()

correct = defaultdict(int)
total = defaultdict(int)
misclassified = []

# Only evaluate samples whose class is in the ensemble's classes list
class_to_idx = {c: i for i, c in enumerate(classes)}

for i in range(len(ds)):
    seq, label = ds[i]
    true_cls = ds.classes[label]
    if true_cls not in class_to_idx:
        continue  # skip single-sample classes not in model
    seq_np = np.array(seq, dtype=np.float32)
    pred_idx, conf, _ = ensemble_predict(models, seq_np)
    pred_cls = classes[pred_idx]
    total[true_cls] += 1
    if pred_idx == label:
        correct[true_cls] += 1
    else:
        misclassified.append((i, true_cls, pred_cls, conf))

print("Per-class accuracy:")
print(f"{'Class':<12} {'Correct':>7} {'Total':>7} {'Acc':>7}  {'Videos':>6}")
print("-" * 50)
for cls in classes:
    c = correct[cls]
    t = total[cls]
    acc = c / t * 100 if t > 0 else 0
    print(f"{cls:<12} {c:>7} {t:>7} {acc:>6.1f}%  {t:>6}")

print(f"\nMisclassified ({len(misclassified)}):")
for idx, true_c, pred_c, conf in misclassified:
    print(f"  #{idx:>3}: true={true_c:<12} pred={pred_c:<12} conf={conf:.1%}")
