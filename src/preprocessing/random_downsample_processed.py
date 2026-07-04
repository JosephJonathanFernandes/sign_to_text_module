"""Randomly downsample class folders inside processed/ to a fixed threshold.

This script keeps at most ``threshold`` .npy files per class folder and deletes
the rest using a random selection with no ordering bias.

Pure webcam captures (``webcam_*.npy``) are protected and never deleted. If a
class already has more protected webcam files than the threshold, that class
will remain above threshold by design.

The default threshold is 500 because the smallest processed class currently has
253 samples, so 500 trims the large classes substantially while leaving the
smallest classes untouched.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass


ROOT_DIR = os.path.join("assets", "processed")
DEFAULT_THRESHOLD = 555


@dataclass
class ClassSummary:
    class_name: str
    total_files: int
    kept_files: int
    deleted_files: int
    dry_run: bool


def _list_class_dirs(root_dir: str) -> list[str]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {os.path.abspath(root_dir)}")

    return sorted(
        os.path.join(root_dir, entry)
        for entry in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, entry))
    )


def _list_npy_files(class_dir: str) -> list[str]:
    return [
        os.path.join(class_dir, entry)
        for entry in os.listdir(class_dir)
        if entry.lower().endswith(".npy") and os.path.isfile(os.path.join(class_dir, entry))
    ]


def _is_webcam_sample(path: str) -> bool:
    """Return True for pure webcam-captured samples (excludes aug/merge)."""
    basename = os.path.basename(path).lower()
    return basename.startswith("webcam_") and "_aug_" not in basename and "_merge_" not in basename


def _safe_delete(path: str, class_dir: str, dry_run: bool) -> bool:
    path_abs = os.path.abspath(path)
    class_abs = os.path.abspath(class_dir)

    if not path_abs.lower().endswith(".npy"):
        return False
    if not path_abs.startswith(class_abs + os.sep):
        return False
    if not os.path.isfile(path_abs):
        return False

    if not dry_run:
        os.remove(path_abs)
    return True


def downsample_class_folder(
    class_dir: str,
    threshold: int,
    rng: random.Random,
    dry_run: bool = False,
) -> ClassSummary:
    files = _list_npy_files(class_dir)
    total = len(files)
    class_name = os.path.basename(class_dir)

    webcam_files = [path for path in files if _is_webcam_sample(path)]
    non_webcam_files = [path for path in files if not _is_webcam_sample(path)]

    if total <= threshold:
        print(f"[{class_name}] total={total} kept={total} deleted=0 (already at or below threshold)")
        return ClassSummary(class_name, total, total, 0, dry_run)

    protected_count = len(webcam_files)
    # Keep all webcam samples; fill the rest from non-webcam files.
    non_webcam_quota = max(0, threshold - protected_count)
    kept_non_webcam = min(non_webcam_quota, len(non_webcam_files))

    keep_set = set(webcam_files)
    if kept_non_webcam > 0:
        keep_set.update(rng.sample(non_webcam_files, kept_non_webcam))

    delete_files = [path for path in non_webcam_files if path not in keep_set]
    rng.shuffle(delete_files)

    deleted = 0
    for path in delete_files:
        if _safe_delete(path, class_dir, dry_run):
            deleted += 1

    final_kept = total - deleted
    extra_note = ""
    if protected_count > threshold:
        extra_note = f" | protected_webcam={protected_count} (> threshold, retained)"

    print(
        f"[{class_name}] total={total} kept={final_kept} deleted={deleted} "
        f"({'dry-run' if dry_run else 'applied'}){extra_note}"
    )
    return ClassSummary(class_name, total, final_kept, deleted, dry_run)


def downsample_processed(
    root_dir: str = ROOT_DIR,
    threshold: int = DEFAULT_THRESHOLD,
    seed: int | None = None,
    dry_run: bool = False,
    class_only: str | None = None,
) -> list[ClassSummary]:
    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")

    rng = random.SystemRandom() if seed is None else random.Random(seed)
    class_dirs = _list_class_dirs(root_dir)

    if class_only:
        class_dirs = [d for d in class_dirs if os.path.basename(d) == class_only]
        if not class_dirs:
            available = ", ".join(os.path.basename(d) for d in _list_class_dirs(root_dir))
            raise ValueError(f"Class '{class_only}' not found in {os.path.abspath(root_dir)}. Available: {available}")

    print("=" * 90)
    print(f"Random downsample started | ROOT_DIR={os.path.abspath(root_dir)} | THRESHOLD={threshold} | DRY_RUN={dry_run}")
    print("=" * 90)

    summaries: list[ClassSummary] = []
    grand_total = 0
    grand_kept = 0
    grand_deleted = 0

    for class_dir in class_dirs:
        summary = downsample_class_folder(class_dir, threshold=threshold, rng=rng, dry_run=dry_run)
        summaries.append(summary)
        grand_total += summary.total_files
        grand_kept += summary.kept_files
        grand_deleted += summary.deleted_files

    print("-" * 90)
    print(f"TOTAL: total={grand_total} kept={grand_kept} deleted={grand_deleted}")
    if dry_run:
        print("NOTE: DRY_RUN=True, no files were actually deleted.")
    print("=" * 90)

    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly downsample processed class folders to a fixed threshold")
    parser.add_argument("--root", default=ROOT_DIR, help="Root folder containing class subfolders")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="Maximum .npy files to keep per class")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible random selection")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without changing files")
    parser.add_argument("--class", dest="class_only", default=None, help="Only process one class folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    downsample_processed(
        root_dir=args.root,
        threshold=args.threshold,
        seed=args.seed,
        dry_run=args.dry_run,
        class_only=args.class_only,
    )