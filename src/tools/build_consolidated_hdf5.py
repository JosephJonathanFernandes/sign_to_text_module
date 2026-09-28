"""
build_consolidated_hdf5.py
==========================
Read-only consolidation of all five dataset sources into a single
assets/dataset_consolidated.h5 file.

Groups written:
  main_positive      <- assets/dataset.h5         (re-exported from source)
  refining_adapter   <- assets/pseudo_data/        (adapter fine-tuning stage)
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
# Paths (resolved to absolute for I/O, stored as relative for attrs/CSV)
# ---------------------------------------------------------------------------
PROJECT_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS_DIR    = os.path.join(PROJECT_ROOT, "assets")

SRC_H5        = os.path.join(ASSETS_DIR, "dataset.h5")
SRC_ADAPTER   = os.path.join(ASSETS_DIR, "pseudo_data")
SRC_PHASE2    = os.path.join(ASSETS_DIR, "processed_del")
SRC_NEG_PHASE2= os.path.join(ASSETS_DIR, "processed_negatives_del")
SRC_NEG       = os.path.join(ASSETS_DIR, "processed_negatives")


def _rel(abs_path):
    """Return a project-root-relative path with forward slashes."""
    try:
        rel = os.path.relpath(abs_path, PROJECT_ROOT)
    except ValueError:
        # Different drive on Windows — fall back to basename
        rel = os.path.basename(abs_path)
    return rel.replace("\\", "/")


# Relative source paths stored in the file
REL_SRC_H5        = _rel(SRC_H5)
REL_SRC_ADAPTER   = _rel(SRC_ADAPTER)
REL_SRC_PHASE2    = _rel(SRC_PHASE2)
REL_SRC_NEG_PHASE2= _rel(SRC_NEG_PHASE2)
REL_SRC_NEG       = _rel(SRC_NEG)

EXPECTED_SHAPE = (20, 506)
GZIP_LEVEL     = 4
CHUNK          = (1, 20, 506)
BATCH_SIZE     = 500

NOTES = (
    "Consolidation of all training data sources of the Vaksetu ISL recognition project.\n"
    "Each sample is a (20, 506) float32 sequence: 20 frames x 506 features. Per frame, 253\n"
    "base features (left/right hand landmarks wrist-centred and scale-normalised; the same\n"
    "landmarks relative to the face; a hand-to-face proximity scalar) plus 253 frame-to-frame\n"
    "velocity features. Features derive from MediaPipe landmarks; no video or images are\n"
    "included.\n\n"
    "main_positive: primary training set exported from dataset.h5 (300 classes). Per-sample\n"
    "labels (indices into class_names), weights (all 1.0, placeholder) and domains\n"
    "(cvae = synthetic sequences from a conditional VAE, webcam = live webcam recordings,\n"
    "MVI = features extracted from videos of the INCLUDE dataset (Sridhar et al., ACM MM 2020, doi 10.1145/3394171.3413528, CC BY 4.0), unknown = other; see\n"
    "domain_names). Filenames are synthetic (hdf5_sample_NNNNNN) because source filenames\n"
    "were not stored when the set was compiled.\n\n"
    "refining_adapter: sequences collected during deployment and used to train a small MLP\n"
    "output adapter on top of the frozen base ensemble (not LoRA). Labels are either\n"
    "user-corrected (feedback endpoint) or model-predicted with confidence >= 0.85; the two\n"
    "are not distinguishable in the files, so this group may contain label noise.\n\n"
    "refining_phase2: positive samples removed from the primary pool by a quality/diversity\n"
    "filter (composite quality score plus diversity selection) and used in Phase-2\n"
    "fine-tuning. Domain is derived from filename prefix; webcam-prefixed files include augmented variants. Groups contain exact duplicate arrays (see overlap_stats).\n\n"
    "negative: reject class (23 categories: transitions, incomplete signs, idle, tracking\n"
    "failures, non-signing movement, and confusable sign categories) used in Phase-1 training.\n\n"
    "refining_negative: lower-quality reject samples removed from the same negative pool by\n"
    "the same filter, used in Phase-2 fine-tuning.\n\n"
    "Groups were not curated to be mutually exclusive; see attrs overlap_stats and\n"
    "domain_counts_*. Signer metadata and consent records are not part of this dataset."
)

LABEL_ENCODING_NOTE = (
    "main_positive/labels contains integer indices into main_positive/class_names "
    "(a JSON mapping of class_name -> integer_index, copied from dataset.h5). "
    "All .npy-sourced groups (refining_adapter, refining_phase2, refining_negative, "
    "negative) do not have a labels dataset; the class is encoded in the filenames "
    "column as '<class>/<filename>.npy', where <class> is the subdirectory name under "
    "the source directory."
)

# ---------------------------------------------------------------------------
# Domain helpers (mirrors compile_hdf5.py L72-85)
# ---------------------------------------------------------------------------

def _domain_from_filename(fname):
    """Infer domain string from filename prefix."""
    base = os.path.basename(fname)
    if base.startswith("cvae_"):
        return "cvae"
    if base.startswith("webcam_"):
        return "webcam"
    if base.startswith("MVI_"):
        return "MVI"
    return "unknown"


def _is_augmented(fname):
    base = os.path.basename(fname)
    return "_aug_" in base or "_merge_" in base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of_array(arr):
    return hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()


def sha256_of_file(path, chunk_bytes=8 * 1024 * 1024):
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

def write_group_from_npy(h5_group, files, manifest_rows, group_name, abs_source_dir):
    """Write a group from .npy files.  Aborts on any bad file (fail-loud)."""
    n = len(files)
    if n == 0:
        print(f"  [WARN] {group_name}: 0 files -- skipping.")
        return {}

    # ---------- pre-scan: fail loud ----------
    bad_files = []
    for fpath, cls in files:
        try:
            arr = np.load(fpath, allow_pickle=False).astype(np.float32)
        except Exception as e:
            bad_files.append((fpath, f"load error: {e}"))
            continue
        if arr.shape != EXPECTED_SHAPE:
            bad_files.append((fpath, f"bad shape {arr.shape}"))
            continue
        if np.isnan(arr).any():
            bad_files.append((fpath, "contains NaN"))
            continue
        if np.isinf(arr).any():
            bad_files.append((fpath, "contains Inf"))
            continue

    if bad_files:
        print(f"\n[ABORT] {group_name}: {len(bad_files)} bad file(s) detected:")
        for fp, reason in bad_files:
            print(f"  {fp}: {reason}")
        sys.exit(2)
    # -----------------------------------------

    rel_src = _rel(abs_source_dir)
    data_ds = h5_group.create_dataset(
        "data", shape=(n, *EXPECTED_SHAPE), dtype=np.float32,
        chunks=CHUNK, compression="gzip", compression_opts=GZIP_LEVEL)
    fn_ds = h5_group.create_dataset(
        "filenames", shape=(n,), dtype=h5py.string_dtype(), chunks=(1,))

    class_counts = {}
    domain_counts = {}
    for i, (fpath, cls) in enumerate(files):
        arr = np.load(fpath, allow_pickle=False).astype(np.float32)
        data_ds[i] = arr
        # Manifest filename includes class subfolder to avoid basename collisions
        rel_fname = f"{cls}/{os.path.basename(fpath)}"
        fn_ds[i] = rel_fname
        class_counts[cls] = class_counts.get(cls, 0) + 1
        domain = _domain_from_filename(fpath)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        augmented = _is_augmented(fpath)
        manifest_rows.append({
            "filename":     rel_fname,
            "group":        group_name,
            "class":        cls,
            "source_dir":   rel_src,
            "domain":       domain,
            "is_augmented": int(augmented),
            "sha256":       sha256_of_array(arr),
        })
        if (i + 1) % 1000 == 0:
            print(f"  {group_name}: {i+1}/{n}...")

    print(f"  {group_name}: done. {n} samples, {len(class_counts)} classes.")
    return class_counts, domain_counts


def write_group_from_h5(dst_group, src_h5_path, manifest_rows):
    """Write main_positive group from dataset.h5 (read-only)."""
    with h5py.File(src_h5_path, "r") as src:
        n             = int(src.attrs["sample_count"])
        labels_raw    = src["labels"][:]
        weights_raw   = src["weights"][:]
        domains_raw   = src["domains"][:]
        class_names_json  = src["class_names"][()]
        domain_names_json = src["domain_names"][()]
        class_mapping     = json.loads(class_names_json)
        domain_mapping    = json.loads(domain_names_json)   # domain_str -> int
        idx_to_class      = {v: k for k, v in class_mapping.items()}
        idx_to_domain     = {v: k for k, v in domain_mapping.items()}
        features_ds       = src["features"]

        data_ds = dst_group.create_dataset(
            "data", shape=(n, *EXPECTED_SHAPE), dtype=np.float32,
            chunks=CHUNK, compression="gzip", compression_opts=GZIP_LEVEL)
        fn_ds = dst_group.create_dataset(
            "filenames", shape=(n,), dtype=h5py.string_dtype(), chunks=(1,))
        # Preserve per-sample metadata
        dst_group.create_dataset("labels",  data=labels_raw,  dtype=np.int32)
        dst_group.create_dataset("weights", data=weights_raw, dtype=np.float32)
        dst_group.create_dataset("domains", data=domains_raw, dtype=np.int32)
        # Copy class/domain mappings
        dst_group.create_dataset(
            "class_names",  data=class_names_json  if isinstance(class_names_json,  str)
                                 else class_names_json.decode())
        dst_group.create_dataset(
            "domain_names", data=domain_names_json if isinstance(domain_names_json, str)
                                 else domain_names_json.decode())

        class_counts  = {}
        domain_counts = {}
        rel_src = _rel(src_h5_path)

        start = 0
        while start < n:
            end   = min(start + BATCH_SIZE, n)
            batch = features_ds[start:end]
            for j in range(end - start):
                i   = start + j
                arr = batch[j].astype(np.float32)
                cls = idx_to_class.get(int(labels_raw[i]), "__unknown__")
                domain_str = idx_to_domain.get(int(domains_raw[i]), "unknown")
                fname = f"hdf5_sample_{i:06d}"
                data_ds[i] = arr
                fn_ds[i]   = fname
                class_counts[cls]  = class_counts.get(cls, 0) + 1
                domain_counts[domain_str] = domain_counts.get(domain_str, 0) + 1
                manifest_rows.append({
                    "filename":     fname,
                    "group":        "main_positive",
                    "class":        cls,
                    "source_dir":   rel_src,
                    "domain":       domain_str,
                    "is_augmented": int(_is_augmented(fname)),
                    "sha256":       sha256_of_array(arr),
                })
            if end % 5000 == 0 or end == n:
                print(f"  main_positive: {end}/{n}...")
            start = end

    print(f"  main_positive: done. {n} samples, {len(class_counts)} classes.")
    return class_counts, domain_counts, n


# ---------------------------------------------------------------------------
# Cross-group overlap computation
# ---------------------------------------------------------------------------

GROUP_ORDER = [
    "main_positive",
    "refining_adapter",
    "refining_phase2",
    "refining_negative",
    "negative",
]


def compute_overlap_stats(manifest_rows):
    """
    Build cross-group overlap stats from sha256 hashes already in manifest_rows.
    Returns a dict:
      within_<group>: number of duplicate hashes within that group
      <groupA>_x_<groupB>: number of hashes from A that also appear in B
    """
    # Build per-group hash sets (and lists for within-group dup counting)
    by_group = {g: [] for g in GROUP_ORDER}
    for row in manifest_rows:
        by_group[row["group"]].append(row["sha256"])

    stats = {}
    # Within-group duplicates
    for g in GROUP_ORDER:
        hashes = by_group[g]
        uniq = set(hashes)
        stats[f"within_{g}"] = len(hashes) - len(uniq)

    # Cross-group: for each ordered pair
    for i, ga in enumerate(GROUP_ORDER):
        set_a = set(by_group[ga])
        for gb in GROUP_ORDER[i + 1:]:
            set_b = set(by_group[gb])
            key = f"{ga}_x_{gb}"
            stats[key] = len(set_a & set_b)

    return stats


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
    print("ISL Dataset Consolidation  (v2)")
    print(f"  Output:   {out_h5}")
    print(f"  Dry run:  {args.dry_run}")
    print("=" * 70)

    for label, path in [
        ("dataset.h5",               SRC_H5),
        ("pseudo_data/",             SRC_ADAPTER),
        ("processed_del/",           SRC_PHASE2),
        ("processed_negatives_del/", SRC_NEG_PHASE2),
        ("processed_negatives/",     SRC_NEG),
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

    print(f"  main_positive:     {n_main:>6}  (dataset.h5 -> {REL_SRC_H5})")
    print(f"  refining_adapter:  {len(npy_adapter):>6}  ({REL_SRC_ADAPTER})")
    print(f"  refining_phase2:   {len(npy_phase2):>6}  ({REL_SRC_PHASE2})")
    print(f"  refining_negative: {len(npy_neg_p2):>6}  ({REL_SRC_NEG_PHASE2})")
    print(f"  negative:          {len(npy_neg):>6}  ({REL_SRC_NEG})")
    print(f"  TOTAL:             {total:>6}")
    print(f"  Scan time: {time.time()-t0:.1f}s")

    if args.dry_run:
        print("\n[DRY RUN] Stopping before write. No files created.")
        return

    if os.path.exists(out_h5):
        print(f"\n[ERROR] Output already exists: {out_h5}")
        print("  Remove or rename it first to rebuild. Aborting.")
        sys.exit(1)

    manifest_rows = []
    t_write = time.time()

    with h5py.File(out_h5, "w") as dst:

        print("\n[1/5] main_positive ...")
        grp = dst.create_group("main_positive")
        cc_main, dc_main, n_main_actual = write_group_from_h5(grp, SRC_H5, manifest_rows)

        print("\n[2/5] refining_adapter ...")
        grp = dst.create_group("refining_adapter")
        cc_adapter, dc_adapter = write_group_from_npy(
            grp, npy_adapter, manifest_rows, "refining_adapter", SRC_ADAPTER)

        print("\n[3/5] refining_phase2 ...")
        grp = dst.create_group("refining_phase2")
        cc_phase2, dc_phase2 = write_group_from_npy(
            grp, npy_phase2, manifest_rows, "refining_phase2", SRC_PHASE2)

        print("\n[4/5] refining_negative ...")
        grp = dst.create_group("refining_negative")
        cc_neg_p2, dc_neg_p2 = write_group_from_npy(
            grp, npy_neg_p2, manifest_rows, "refining_negative", SRC_NEG_PHASE2)

        print("\n[5/5] negative ...")
        grp = dst.create_group("negative")
        cc_neg, dc_neg = write_group_from_npy(
            grp, npy_neg, manifest_rows, "negative", SRC_NEG)

        # ---- cross-group overlap (computed from manifest hashes) ----
        print("\nComputing cross-group overlap stats...")
        overlap_stats = compute_overlap_stats(manifest_rows)
        for k, v in sorted(overlap_stats.items()):
            print(f"  {k}: {v}")

        print("\nWriting root attributes...")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dst.attrs["creation_date"]           = now
        dst.attrs["feature_dimension"]       = 506
        dst.attrs["sequence_length"]         = 20
        dst.attrs["total_samples"]           = total
        dst.attrs["gzip_compression_level"]  = GZIP_LEVEL
        dst.attrs["chunk_shape"]             = str(CHUNK)
        # Relative source paths only
        dst.attrs["source_main_positive"]        = REL_SRC_H5
        dst.attrs["source_refining_adapter"]     = REL_SRC_ADAPTER
        dst.attrs["source_refining_phase2"]      = REL_SRC_PHASE2
        dst.attrs["source_refining_negative"]    = REL_SRC_NEG_PHASE2
        dst.attrs["source_negative"]             = REL_SRC_NEG
        # Counts
        dst.attrs["count_main_positive"]     = n_main
        dst.attrs["count_refining_adapter"]  = len(npy_adapter)
        dst.attrs["count_refining_phase2"]   = len(npy_phase2)
        dst.attrs["count_refining_negative"] = len(npy_neg_p2)
        dst.attrs["count_negative"]          = len(npy_neg)
        # Class counts
        dst.attrs["classes_main_positive"]      = json.dumps(cc_main)
        dst.attrs["classes_refining_adapter"]   = json.dumps(cc_adapter)
        dst.attrs["classes_refining_phase2"]    = json.dumps(cc_phase2)
        dst.attrs["classes_refining_negative"]  = json.dumps(cc_neg_p2)
        dst.attrs["classes_negative"]           = json.dumps(cc_neg)
        # Domain counts per group
        dst.attrs["domain_counts_main_positive"]      = json.dumps(dc_main)
        dst.attrs["domain_counts_refining_adapter"]   = json.dumps(dc_adapter)
        dst.attrs["domain_counts_refining_phase2"]    = json.dumps(dc_phase2)
        dst.attrs["domain_counts_refining_negative"]  = json.dumps(dc_neg_p2)
        dst.attrs["domain_counts_negative"]           = json.dumps(dc_neg)
        # Overlap + notes
        dst.attrs["overlap_stats"]           = json.dumps(overlap_stats)
        dst.attrs["label_encoding_note"]     = LABEL_ENCODING_NOTE
        dst.attrs["notes"]                   = NOTES

    h5_size = os.stat(out_h5).st_size
    print(f"\nHDF5 written: {h5_size/1e9:.3f} GB  ({(time.time()-t_write)/60:.1f} min)")

    # ---- manifest CSV ----
    print(f"\nWriting manifest CSV ({len(manifest_rows):,} rows)...")
    fieldnames = ["filename", "group", "class", "source_dir", "domain", "is_augmented", "sha256"]
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"  Manifest: {os.stat(out_csv).st_size/1e6:.1f} MB")

    # ---- SHA256 of the HDF5 file ----
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
