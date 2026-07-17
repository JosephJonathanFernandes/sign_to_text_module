"""Balance processed class folders to a fixed sample count.

This script keeps webcam captures as the preferred source for duplication when a
class is below target, then trims excess files when a class is above target.

It is intended for dataset preparation, not training-time oversampling.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import uuid
from dataclasses import dataclass


ROOT_DIR = os.path.join("assets", "processed")
TARGET_SAMPLES = 300
WEBCAM_PREFIX = "webcam_"
DUPLICATE_PREFIX = "webcam_dup_"


@dataclass
class ClassSummary:
    class_name: str
    total_files: int
    added_files: int
    removed_files: int
    final_files: int
    dry_run: bool


def _list_class_dirs(root_dir: str) -> list[str]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {os.path.abspath(root_dir)}")

    return sorted(
        os.path.join(root_dir, entry) for entry in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, entry))
    )


def _list_npy_files(class_dir: str) -> list[str]:
    return sorted(
        os.path.join(class_dir, entry)
        for entry in os.listdir(class_dir)
        if entry.lower().endswith(".npy") and os.path.isfile(os.path.join(class_dir, entry))
    )


def _is_webcam_sample(path: str) -> bool:
    """Return True for pure webcam-captured samples (excludes aug/merge)."""
    basename = os.path.basename(path).lower()
    return basename.startswith("webcam_") and "_aug_" not in basename and "_merge_" not in basename


def _is_duplicate_webcam_sample(path: str) -> bool:
    name = os.path.basename(path).lower()
    return name.startswith(DUPLICATE_PREFIX) or "_dup_" in name


def _classify_removals(files: list[str]) -> list[str]:
    duplicate_webcam = [path for path in files if _is_webcam_sample(path) and _is_duplicate_webcam_sample(path)]
    non_webcam = [path for path in files if not _is_webcam_sample(path)]
    original_webcam = [path for path in files if _is_webcam_sample(path) and not _is_duplicate_webcam_sample(path)]
    return duplicate_webcam + non_webcam + original_webcam


def _build_duplicate_name(source_path: str, class_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    stem = stem.replace(" ", "_")
    stem = stem[:36].rstrip("_")
    while True:
        candidate = f"{DUPLICATE_PREFIX}{stem}_{uuid.uuid4().hex[:8]}.npy"
        candidate_path = os.path.join(class_dir, candidate)
        if not os.path.exists(candidate_path):
            return candidate_path


def balance_class_folder(
    class_dir: str,
    target: int,
    rng: random.Random,
    dry_run: bool = False,
) -> ClassSummary:
    files = _list_npy_files(class_dir)
    total = len(files)
    class_name = os.path.basename(class_dir)

    if total >= target:
        msg = "already balanced" if total == target else "above target, keeping all"
        print(f"[{class_name}] total={total} final={total} added=0 removed=0 ({msg})")
        return ClassSummary(class_name, total, 0, 0, total, dry_run)

    added = 0
    removed = 0

    if total < target:
        webcam_sources = [path for path in files if _is_webcam_sample(path)]
        source_pool = webcam_sources if webcam_sources else files
        if not source_pool:
            print(f"[WARN] Class '{class_name}' has no .npy files to duplicate. Skipping.")
            return ClassSummary(class_name, total, 0, 0, total, dry_run)

        needed = target - total
        for _ in range(needed):
            source = rng.choice(source_pool)
            dst_path = _build_duplicate_name(source, class_dir)
            if not dry_run:
                shutil.copy2(source, dst_path)
            added += 1

        final_total = total + added
        print(
            f"[{class_name}] total={total} final={final_total} added={added} removed=0 "
            f"({'dry-run' if dry_run else 'applied'})"
        )
        return ClassSummary(class_name, total, added, 0, final_total, dry_run)


def balance_processed_dataset(
    root_dir: str = ROOT_DIR,
    target: int = TARGET_SAMPLES,
    seed: int | None = None,
    dry_run: bool = False,
    class_only: str | None = None,
) -> list[ClassSummary]:
    if target <= 0:
        raise ValueError("target must be greater than 0")

    rng = random.SystemRandom() if seed is None else random.Random(seed)
    class_dirs = _list_class_dirs(root_dir)

    if class_only:
        class_dirs = [d for d in class_dirs if os.path.basename(d) == class_only]
        if not class_dirs:
            available = ", ".join(os.path.basename(d) for d in _list_class_dirs(root_dir))
            raise ValueError(f"Class '{class_only}' not found in {os.path.abspath(root_dir)}. Available: {available}")

    print("=" * 90)
    print(
        f"Balance processed dataset started | ROOT_DIR={os.path.abspath(root_dir)} | TARGET={target} | DRY_RUN={dry_run}"
    )
    print("=" * 90)

    summaries: list[ClassSummary] = []
    grand_total_before = 0
    grand_total_after = 0
    grand_added = 0
    grand_removed = 0

    for class_dir in class_dirs:
        before = len(_list_npy_files(class_dir))
        summary = balance_class_folder(class_dir, target=target, rng=rng, dry_run=dry_run)
        summaries.append(summary)
        grand_total_before += before
        grand_total_after += summary.final_files
        grand_added += summary.added_files
        grand_removed += summary.removed_files

    print("-" * 90)
    print(f"TOTAL: before={grand_total_before} after={grand_total_after} added={grand_added} removed={grand_removed}")
    if dry_run:
        print("NOTE: DRY_RUN=True, no files were changed.")
    print("=" * 90)

    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balance processed class folders to a fixed target count")
    parser.add_argument("--root", default=ROOT_DIR, help="Root folder containing class subfolders")
    parser.add_argument("--target", type=int, default=TARGET_SAMPLES, help="Target number of .npy files per class")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible selection")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without modifying files")
    parser.add_argument("--class", dest="class_only", default=None, help="Only balance one class folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    balance_processed_dataset(
        root_dir=args.root,
        target=args.target,
        seed=args.seed,
        dry_run=args.dry_run,
        class_only=args.class_only,
    )
