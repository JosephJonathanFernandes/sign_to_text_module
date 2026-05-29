"""Validate .npy files under processed/ and processed_negatives/.

Checks performed per file:
 - Can be loaded with numpy.load
 - Is a numeric ndarray
 - Has ndim == 2 (NUM_FRAMES, feat_dim)
 - num_frames > 0 and feat_dim > 0
 - No NaN or Inf values

Writes a short JSON report to `logs/npy_validation_report.json` and
prints a summary to stdout.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path.cwd()
PROCESSED = ROOT / "processed"
NEG = ROOT / "processed_negatives"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
REPORT_PATH = LOG_DIR / "npy_validation_report.json"

roots = [PROCESSED]
if NEG.exists() and NEG.is_dir():
    roots.append(NEG)

report = {
    "checked_roots": [str(p) for p in roots],
    "total_files": 0,
    "ok_files": 0,
    "bad_files": [],
}

for root in roots:
    if not root.exists():
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith('.npy'):
                continue
            report['total_files'] += 1
            fpath = os.path.join(dirpath, fn)
            entry = {"path": fpath}
            try:
                arr = np.load(fpath, allow_pickle=False)
            except Exception as e:
                entry['error'] = f"load_error: {repr(e)}"
                report['bad_files'].append(entry)
                continue
            # Basic checks
            if not isinstance(arr, np.ndarray):
                entry['error'] = "not_ndarray"
                report['bad_files'].append(entry)
                continue
            if arr.size == 0:
                entry['error'] = "empty_array"
                report['bad_files'].append(entry)
                continue
            if arr.ndim != 2:
                entry['error'] = f"bad_ndim: {arr.ndim}"
                report['bad_files'].append(entry)
                continue
            if arr.shape[0] <= 0 or arr.shape[1] <= 0:
                entry['error'] = f"invalid_shape: {arr.shape}"
                report['bad_files'].append(entry)
                continue
            # Finite check (float or int)
            try:
                if not np.all(np.isfinite(arr)):
                    entry['error'] = "contains_nonfinite"
                    report['bad_files'].append(entry)
                    continue
            except Exception as e:
                entry['error'] = f"finite_check_error: {repr(e)}"
                report['bad_files'].append(entry)
                continue
            # Passed
            report['ok_files'] += 1

# Write report
with open(REPORT_PATH, 'w', encoding='utf-8') as outf:
    json.dump(report, outf, indent=2)

# Print concise summary
print(f"Checked roots: {', '.join(report['checked_roots'])}")
print(f"Total .npy files: {report['total_files']}")
print(f"OK files: {report['ok_files']}")
print(f"Bad files: {len(report['bad_files'])}")
if report['bad_files']:
    print('\nSample bad files:')
    for b in report['bad_files'][:20]:
        print(' -', b.get('path'), '->', b.get('error'))

print(f"Report written to: {REPORT_PATH}")

if report['bad_files']:
    sys.exit(2)
else:
    sys.exit(0)
