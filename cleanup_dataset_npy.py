"""Diversity-based cleanup for .npy datasets organized as root/class/*.npy.

File groups:
1) Original: no special tag
2) Augmented: filename contains "_aug_"
3) Merged: filename contains "_merge_" or "_mrg_"

Behavior:
- Keep all originals.
- Keep only top-K diverse augmented and merged files per class using
  greedy farthest point sampling (FPS) on flattened vectors.
- Remove near-duplicates first (distance threshold) before FPS.
- Delete only files inside class folders (safe checks).
"""

import os
import random
import numpy as np


# ------------------------------
# Configurable settings
# ------------------------------
ROOT_DIR = "processed"
MAX_AUG_PER_CLASS = 35
MAX_MERGE_PER_CLASS = 25
DRY_RUN = False
SEED = 42

# Near-duplicate threshold on L2-normalized flattened vectors
DUPLICATE_EPS = 1e-3


def is_augmented(name_lower: str) -> bool:
    return "_aug_" in name_lower


def is_merged(name_lower: str) -> bool:
    return "_merge_" in name_lower or "_mrg_" in name_lower


def safe_delete(path: str, class_dir_abs: str, dry_run: bool) -> bool:
    """Delete file only if it is a .npy under the class directory."""
    try:
        path_abs = os.path.abspath(path)
        class_abs = os.path.abspath(class_dir_abs)

        if not os.path.isfile(path_abs):
            return False
        if not path_abs.lower().endswith(".npy"):
            return False
        if not path_abs.startswith(class_abs + os.sep):
            return False

        if not dry_run:
            os.remove(path_abs)
        return True
    except Exception:
        return False


def _load_flattened_vectors(paths: list[str], tag: str) -> tuple[list[str], np.ndarray]:
    """Load paths to flattened matrix with zero-padding for variable lengths."""
    flat_list = []
    valid_paths = []

    for i, p in enumerate(paths):
        try:
            arr = np.load(p, allow_pickle=False)
            vec = np.asarray(arr, dtype=np.float32).reshape(-1)
            flat_list.append(vec)
            valid_paths.append(p)
        except Exception:
            print(f"    [WARN] Skipping unreadable file ({tag}): {p}")

        if (i + 1) % 200 == 0:
            print(f"    [{tag}] Loaded {i + 1}/{len(paths)} files...")

    if not flat_list:
        return [], np.zeros((0, 0), dtype=np.float32)

    max_len = max(v.shape[0] for v in flat_list)
    mat = np.zeros((len(flat_list), max_len), dtype=np.float32)
    for i, v in enumerate(flat_list):
        mat[i, : v.shape[0]] = v

    # L2-normalize rows for scale robustness
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    mat = mat / norms

    return valid_paths, mat


def _remove_near_duplicates(mat: np.ndarray, eps: float) -> tuple[list[int], list[int]]:
    """Greedy near-duplicate filtering on normalized vectors.

    Returns:
      keep_idx: indices kept as unique
      dup_idx: indices considered near-duplicates
    """
    n = mat.shape[0]
    if n == 0:
        return [], []

    keep_idx = []
    dup_idx = []

    for i in range(n):
        if not keep_idx:
            keep_idx.append(i)
            continue

        kept = mat[np.asarray(keep_idx)]
        d = np.linalg.norm(kept - mat[i], axis=1)
        if np.min(d) <= eps:
            dup_idx.append(i)
        else:
            keep_idx.append(i)

    return keep_idx, dup_idx


def _fps_select(mat: np.ndarray, k: int, rng: random.Random) -> list[int]:
    """Greedy farthest point sampling (FPS) indices over rows of mat."""
    n = mat.shape[0]
    if n == 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))

    first = rng.randrange(n)
    selected = [first]

    min_dist = np.linalg.norm(mat - mat[first], axis=1)
    min_dist[first] = -1.0

    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        d = np.linalg.norm(mat - mat[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
        min_dist[np.asarray(selected)] = -1.0

    return selected


def _select_diverse_subset(
    file_paths: list[str],
    keep_limit: int,
    rng: random.Random,
    tag: str,
) -> tuple[set[str], set[str], int]:
    """Select diverse files with duplicate filtering + FPS.

    Returns:
      keep_set: selected files to keep
      delete_set: files to delete
      duplicate_removed_count: number of near-duplicate files removed
    """
    if not file_paths:
        return set(), set(), 0

    loaded_paths, mat = _load_flattened_vectors(file_paths, tag)
    if not loaded_paths:
        return set(), set(), 0

    keep_unique_idx, dup_idx = _remove_near_duplicates(mat, DUPLICATE_EPS)
    unique_mat = mat[np.asarray(keep_unique_idx)] if keep_unique_idx else np.zeros((0, 0), dtype=np.float32)

    fps_count = min(keep_limit, unique_mat.shape[0])
    chosen_local = _fps_select(unique_mat, fps_count, rng)

    selected_global_idx = [keep_unique_idx[i] for i in chosen_local]
    keep_set = {loaded_paths[i] for i in selected_global_idx}
    delete_set = {p for p in loaded_paths if p not in keep_set}

    return keep_set, delete_set, len(dup_idx)


def clean_class_folder(class_dir: str, rng: random.Random) -> dict:
    """Clean one class folder and return summary stats."""
    original_files = []
    aug_files = []
    merge_files = []
    class_abs = os.path.abspath(class_dir)

    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        if not os.path.isfile(fpath) or not fname.lower().endswith(".npy"):
            continue

        name_lower = fname.lower()
        if is_merged(name_lower):
            merge_files.append(fpath)
        elif is_augmented(name_lower):
            aug_files.append(fpath)
        else:
            original_files.append(fpath)

    print(f"  [Class={os.path.basename(class_dir)}] originals={len(original_files)} aug={len(aug_files)} merge={len(merge_files)}")

    kept_aug, del_aug, aug_dup_removed = _select_diverse_subset(
        aug_files, MAX_AUG_PER_CLASS, rng, tag="aug"
    )
    kept_merge, del_merge, merge_dup_removed = _select_diverse_subset(
        merge_files, MAX_MERGE_PER_CLASS, rng, tag="merge"
    )

    deleted_count = 0
    for path in sorted(del_aug | del_merge):
        if safe_delete(path, class_abs, DRY_RUN):
            deleted_count += 1

    return {
        "class_name": os.path.basename(class_dir),
        "original_count": len(original_files),
        "kept_aug_count": len(kept_aug),
        "kept_merge_count": len(kept_merge),
        "deleted_count": deleted_count,
        "total_aug_found": len(aug_files),
        "total_merge_found": len(merge_files),
        "aug_dup_removed": aug_dup_removed,
        "merge_dup_removed": merge_dup_removed,
    }


def clean_dataset(root_dir: str = ROOT_DIR, seed: int = SEED) -> None:
    """Clean all class folders under root_dir and print per-class summary."""
    rng = random.Random(seed)

    root_abs = os.path.abspath(root_dir)
    if not os.path.isdir(root_abs):
        raise FileNotFoundError(f"Root directory not found: {root_abs}")

    class_dirs = [
        os.path.join(root_abs, d)
        for d in sorted(os.listdir(root_abs))
        if os.path.isdir(os.path.join(root_abs, d))
    ]

    print("=" * 90)
    print(f"Diversity cleanup started | ROOT_DIR={root_abs}")
    print(
        f"DRY_RUN={DRY_RUN} | MAX_AUG={MAX_AUG_PER_CLASS} | "
        f"MAX_MERGE={MAX_MERGE_PER_CLASS} | DUPLICATE_EPS={DUPLICATE_EPS} | SEED={seed}"
    )
    print("=" * 90)

    grand_original = 0
    grand_kept_aug = 0
    grand_kept_merge = 0
    grand_deleted = 0

    for class_dir in class_dirs:
        summary = clean_class_folder(class_dir, rng)
        grand_original += summary["original_count"]
        grand_kept_aug += summary["kept_aug_count"]
        grand_kept_merge += summary["kept_merge_count"]
        grand_deleted += summary["deleted_count"]

        print(
            f"[{summary['class_name']}] "
            f"original={summary['original_count']} | "
            f"kept_aug={summary['kept_aug_count']}/{summary['total_aug_found']} "
            f"(dup_removed={summary['aug_dup_removed']}) | "
            f"kept_merge={summary['kept_merge_count']}/{summary['total_merge_found']} "
            f"(dup_removed={summary['merge_dup_removed']}) | "
            f"deleted={summary['deleted_count']}"
        )

    print("-" * 90)
    print(
        f"TOTAL: original={grand_original} | kept_aug={grand_kept_aug} | "
        f"kept_merge={grand_kept_merge} | deleted={grand_deleted}"
    )
    if DRY_RUN:
        print("NOTE: DRY_RUN=True, no files were actually deleted.")
    print("=" * 90)


if __name__ == "__main__":
    clean_dataset(ROOT_DIR, SEED)
