"""
build_consolidated_hdf5.py
==========================
Read-only consolidation of all five dataset sources into a single
assets/dataset_consolidated.h5 file.

Groups written:
  main_positive      <- assets/dataset.h5         (89,883 samples, re-exported)
  refining_adapter   <- assets/pseudo_data/        (adapter/LoRA refining stage)
  refining_phase2    <- assets/processed_del/      (Phase-2 refining, culled positives)
  refining_negative  <- assets/processed_negatives_del/  (Phase-2 reject class)
  negative           <- assets/processed_negatives/      (base reject class)

SAFETY: assets/dataset.h5 and all .npy directories are opened read-only.
        Nothing is moved, deleted, or overwritten.

Output:
  assets/dataset_consolidated.h5
  assets/manifest_consolidated.csv
  assets/dataset_consolidated.h5.sha256

Usage:
  python src/tools/build_consolidated_hdf5.py [--dry-run] [--out-dir assets]
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import time

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Source paths (relative to project root)
# ---------------------------------------------------------------------------
ASSETS_DIR    = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
SRC_H5        = os.path.join(ASSETS_DIR, "dataset.h5")
SRC_ADAPTER   = os.path.join(ASSETS_DIR, "pseudo_data")
SRC_PHASE2    = os.path.join(ASSETS_DIR, "processed_del")
SRC_NEG_PHASE2= os.path.join(ASSETS_DIR, "processed_negatives_del")
SRC_NEG       = os.path.join(ASSETS_DIR, "processed_negatives")

EXPECTED_SHAPE = (20, 506)
GZIP_LEVEL     = 4
CHUNK          = (1, 20, 506)
BATCH_SIZE     = 500

NOTES = (
    "Five-group consolidation of all ISL training data sources.\n\n"
    "main_positive: Re-exported from assets/dataset.h5 (lzf-compressed, 300 classes). "
    "Filenames are synthetic (hdf5_sample_NNNNNN) because the original compile step "
    "(src/tools/compile_hdf5.py) did not store source filenames in dataset.h5. "
    "To recover real filenames, re-run compile_hdf5.py scan logic against "
    "assets/processed/ -- do NOT attempt np.allclose matching (augmented near-duplicates "
    "make 1:1 matching unreliable).\n\n"
    "refining_adapter: Live pseudo-labeled samples saved by the real-time inference "
    "pipeline (src/inference/pseudo_buffer.py) and user-correction sessions. Used for "
    "adapter/LoRA fine-tuning. Class labels = subdirectory names.\n\n"
    "refining_phase2: Archived/culled positive samples (originally in processed/, "
    "later migrated to processed_del/) reused as supplementary data in Phase-2 training "
    "via ISLDataset(archived_root=...) with weight=0.25. 300 classes, perfect alignment "
    "with dataset.h5 class vocabulary.\n\n"
    "refining_negative: Archived reject/noise samples paired with refining_phase2. "
    "Used as Phase-2 archived_neg_root. 23 noise categories.\n\n"
    "negative: Base reject/__reject__ class loaded in Phase-1 training and all "
    "robustness evaluations via ISLDataset(neg_root=...). 23 noise categories. "
    "Same category set as refining_negative."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of_array(arr):
    return hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()

def sha256_of_file(path, chunk_bytes=8*1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def collect_npy_files(root):
    """Walk root; return list of (filepath, class_name) sorted deterministically."""
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fname in sorted(filenames):
            if fname.endswith(".npy"):
                rel = os.path.relpath(dirpath, root)
                cls = rel.split(os.sep)[0]
                if cls == ".":
                    cls = "__unknown__"
                entries.append((os.path.join(dirpath, fname), cls))
    return entries

# ---------------------------------------------------------------------------
# Group writers
# ---------------------------------------------------------------------------

def write_group_from_npy(h5_group, files, manifest_rows, group_name, source_dir):
    n = len(files)
    if n == 0:
        print(f"  [WARN] {group_name}: 0 files -- skipping.")
        return {}

    data_ds = h5_group.create_dataset(
        "data", shape=(n, *EXPECTED_SHAPE), dtype=np.float32,
        chunks=CHUNK, compression="gzip", compression_opts=GZIP_LEVEL)
    fn_ds = h5_group.create_dataset(
        "filenames", shape=(n,), dtype=h5py.string_dtype(), chunks=(1,))

    class_counts = {}
    skipped = 0
    for i, (fpath, cls) in enumerate(files):
        try:
            arr = np.load(fpath).astype(np.float32)
            if arr.shape != EXPECTED_SHAPE:
                raise ValueError(f"Bad shape {arr.shape}")
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(fpath)}: {e}")
            skipped += 1
            continue
        data_ds[i] = arr
        fn_ds[i]   = os.path.basename(fpath)
        class_counts[cls] = class_counts.get(cls, 0) + 1
        manifest_rows.append({
            "filename":   os.path.basename(fpath),
            "group":      group_name,
            "class":      cls,
            "source_dir": source_dir,
            "sha256":     sha256_of_array(arr),
        })
        if (i + 1) % 1000 == 0:
            print(f"  {group_name}: {i+1}/{n}...")
    if skipped:
        print(f"  [WARN] {group_name}: {skipped} files skipped.")
    print(f"  {group_name}: done. {n - skipped} samples, {len(class_counts)} classes.")
    return class_counts


def write_group_from_h5(dst_group, src_h5_path, manifest_rows):
    with h5py.File(src_h5_path, "r") as src:
        n             = int(src.attrs["sample_count"])
        labels_raw    = src["labels"][:]
        weights_raw   = src["weights"][:]
        domains_raw   = src["domains"][:]
        class_mapping = json.loads(src["class_names"][()])
        idx_to_class  = {v: k for k, v in class_mapping.items()}
        features_ds   = src["features"]

        data_ds = dst_group.create_dataset(
            "data", shape=(n, *EXPECTED_SHAPE), dtype=np.float32,
            chunks=CHUNK, compression="gzip", compression_opts=GZIP_LEVEL)
        fn_ds = dst_group.create_dataset(
            "filenames", shape=(n,), dtype=h5py.string_dtype(), chunks=(1,))
        # Preserve per-sample metadata
        dst_group.create_dataset("labels",  data=labels_raw,  dtype=np.int32)
        dst_group.create_dataset("weights", data=weights_raw, dtype=np.float32)
        dst_group.create_dataset("domains", data=domains_raw, dtype=np.int32)

        class_counts = {}
        start = 0
        while start < n:
            end   = min(start + BATCH_SIZE, n)
            batch = features_ds[start:end]
            for j in range(end - start):
                i    = start + j
                arr  = batch[j].astype(np.float32)
                cls  = idx_to_class.get(int(labels_raw[i]), "__unknown__")
                fname = f"hdf5_sample_{i:06d}"
                data_ds[i] = arr
                fn_ds[i]   = fname
                class_counts[cls] = class_counts.get(cls, 0) + 1
                manifest_rows.append({
                    "filename":   fname,
                    "group":      "main_positive",
                    "class":      cls,
                    "source_dir": src_h5_path,
                    "sha256":     sha256_of_array(arr),
                })
            if end % 5000 == 0 or end == n:
                print(f"  main_positive: {end}/{n}...")
            start = end

    print(f"  main_positive: done. {n} samples, {len(class_counts)} classes.")
    return class_counts, n

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--out-dir",  default=ASSETS_DIR)
    args = parser.parse_args()

    out_dir    = os.path.normpath(args.out_dir)
    out_h5     = os.path.join(out_dir, "dataset_consolidated.h5")
    out_csv    = os.path.join(out_dir, "manifest_consolidated.csv")
    out_sha256 = os.path.join(out_dir, "dataset_consolidated.h5.sha256")

    print("=" * 70)
    print("ISL Dataset Consolidation")
    print(f"  Output: {out_h5}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)

    for label, path in [
        ("dataset.h5",             SRC_H5),
        ("pseudo_data/",           SRC_ADAPTER),
        ("processed_del/",         SRC_PHASE2),
        ("processed_negatives_del/", SRC_NEG_PHASE2),
        ("processed_negatives/",   SRC_NEG),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing source: {label} -> {path}")
            sys.exit(1)
    print("All source paths validated. OK")

    print("\nScanning .npy directories...")
    t0 = time.time()
    npy_adapter = collect_npy_files(SRC_ADAPTER)
    npy_phase2  = collect_npy_files(SRC_PHASE2)
    npy_neg_p2  = collect_npy_files(SRC_NEG_PHASE2)
    npy_neg     = collect_npy_files(SRC_NEG)
    with h5py.File(SRC_H5, "r") as f:
        n_main = int(f.attrs["sample_count"])
    total = n_main + len(npy_adapter) + len(npy_phase2) + len(npy_neg_p2) + len(npy_neg)

    print(f"  main_positive:     {n_main:>6}  (dataset.h5)")
    print(f"  refining_adapter:  {len(npy_adapter):>6}  (pseudo_data/)")
    print(f"  refining_phase2:   {len(npy_phase2):>6}  (processed_del/)")
    print(f"  refining_negative: {len(npy_neg_p2):>6}  (processed_negatives_del/)")
    print(f"  negative:          {len(npy_neg):>6}  (processed_negatives/)")
    print(f"  TOTAL:             {total:>6}")
    print(f"  Scan time: {time.time()-t0:.1f}s")

    if args.dry_run:
        print("\n[DRY RUN] Stopping before write. No files created.")
        return

    if os.path.exists(out_h5):
        print(f"\n[ERROR] Output already exists: {out_h5}")
        print("  Remove it manually first to rebuild. Aborting.")
        sys.exit(1)

    manifest_rows = []
    t_write = time.time()

    with h5py.File(out_h5, "w") as dst:

        print("\n[1/5] main_positive ...")
        grp = dst.create_group("main_positive")
        cc_main, n_main_actual = write_group_from_h5(grp, SRC_H5, manifest_rows)

        print("\n[2/5] refining_adapter ...")
        grp = dst.create_group("refining_adapter")
        cc_adapter = write_group_from_npy(grp, npy_adapter, manifest_rows, "refining_adapter", SRC_ADAPTER)

        print("\n[3/5] refining_phase2 ...")
        grp = dst.create_group("refining_phase2")
        cc_phase2 = write_group_from_npy(grp, npy_phase2, manifest_rows, "refining_phase2", SRC_PHASE2)

        print("\n[4/5] refining_negative ...")
        grp = dst.create_group("refining_negative")
        cc_neg_p2 = write_group_from_npy(grp, npy_neg_p2, manifest_rows, "refining_negative", SRC_NEG_PHASE2)

        print("\n[5/5] negative ...")
        grp = dst.create_group("negative")
        cc_neg = write_group_from_npy(grp, npy_neg, manifest_rows, "negative", SRC_NEG)

        print("\nWriting root attributes...")
        now = datetime.datetime.utcnow().isoformat() + "Z"
        dst.attrs["creation_date"]               = now
        dst.attrs["feature_dimension"]           = 506
        dst.attrs["sequence_length"]             = 20
        dst.attrs["total_samples"]               = total
        dst.attrs["gzip_compression_level"]      = GZIP_LEVEL
        dst.attrs["chunk_shape"]                 = str(CHUNK)
        dst.attrs["source_main_positive"]        = SRC_H5
        dst.attrs["source_refining_adapter"]     = SRC_ADAPTER
        dst.attrs["source_refining_phase2"]      = SRC_PHASE2
        dst.attrs["source_refining_negative"]    = SRC_NEG_PHASE2
        dst.attrs["source_negative"]             = SRC_NEG
        dst.attrs["count_main_positive"]         = n_main
        dst.attrs["count_refining_adapter"]      = len(npy_adapter)
        dst.attrs["count_refining_phase2"]       = len(npy_phase2)
        dst.attrs["count_refining_negative"]     = len(npy_neg_p2)
        dst.attrs["count_negative"]              = len(npy_neg)
        dst.attrs["classes_main_positive"]       = json.dumps(cc_main)
        dst.attrs["classes_refining_adapter"]    = json.dumps(cc_adapter)
        dst.attrs["classes_refining_phase2"]     = json.dumps(cc_phase2)
        dst.attrs["classes_refining_negative"]   = json.dumps(cc_neg_p2)
        dst.attrs["classes_negative"]            = json.dumps(cc_neg)
        dst.attrs["notes"]                       = NOTES

    h5_size = os.stat(out_h5).st_size
    print(f"\nHDF5 written: {h5_size/1e9:.3f} GB  ({(time.time()-t_write)/60:.1f} min)")

    print(f"\nWriting manifest CSV ({len(manifest_rows):,} rows)...")
    fieldnames = ["filename", "group", "class", "source_dir", "sha256"]
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"  Manifest: {os.stat(out_csv).st_size/1e6:.1f} MB")

    print("\nComputing SHA256 of HDF5 file...")
    h5_hash = sha256_of_file(out_h5)
    with open(out_sha256, "w", encoding="utf-8") as sf:
        sf.write(f"{h5_hash}  dataset_consolidated.h5\n")
    print(f"  SHA256: {h5_hash}")
    print(f"  Written to: {out_sha256}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"  Samples:    {total:,}")
    print(f"  HDF5:       {out_h5}  ({h5_size/1e9:.3f} GB)")
    print(f"  Manifest:   {out_csv}")
    print(f"  SHA256:     {out_sha256}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print("  assets/dataset.h5 opened READ-ONLY -- not modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
