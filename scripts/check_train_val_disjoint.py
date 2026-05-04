import importlib
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train

train = importlib.reload(train)

ds = train.create_data_loaders()[4]
labels = np.array([lbl for _, lbl in ds.samples])
folds = train._build_source_aware_folds(
    ds.samples,
    labels,
    train.NUM_FOLDS,
    train.RANDOM_SEED,
)

all_sets = [set(map(int, fold.tolist())) for fold in folds]
overlap = sum(
    len(all_sets[i] & all_sets[j])
    for i in range(len(all_sets))
    for j in range(i + 1, len(all_sets))
)
print(f"OVERLAP {overlap}")

for i, val_idx in enumerate(folds):
    source_counts = Counter()
    detail_counts = Counter()
    for idx in val_idx:
        path, _ = ds.samples[idx]
        name = os.path.basename(path).lower()
        source = 'mvi' if name.startswith('mvi') else 'webcam' if 'webcam' in name else 'other'
        source_counts[source] += 1
        detail_counts[
            source + '_' + ('aug' if any(tag in name for tag in ('_aug', '_merge', '_mrg')) else 'orig')
        ] += 1
    print(f"Fold {i} VAL counts: {dict(source_counts)}")
    print(f"Fold {i} detail: {dict(detail_counts)}")
