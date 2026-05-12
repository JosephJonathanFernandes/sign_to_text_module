#!/usr/bin/env python3
"""
Resume K-fold training from a specific fold.

Usage:
    python train_kfold_resume.py                    # Start from fold 0
    python train_kfold_resume.py --start-fold 2    # Resume from fold 2
    python train_kfold_resume.py --start-fold 3    # Resume from fold 3
    python train_kfold_resume.py --list             # Show completed folds
"""

import os
import sys
import json
import argparse
from pathlib import Path
from config import get_config
from pipeline_logger import setup_pipeline_logger


def _manifest_path(cfg) -> str:
    return os.path.join(cfg.paths.ensemble_dir, "kfold_manifest.json")


def _load_manifest(cfg) -> dict:
    manifest_file = _manifest_path(cfg)
    if not os.path.exists(manifest_file):
        return {}
    try:
        with open(manifest_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _checkpoint_completed_folds(cfg) -> list[int]:
    ensemble_dir = cfg.paths.ensemble_dir
    completed = []
    for fold_index in range(cfg.paths.num_folds):
        fold_path = os.path.join(ensemble_dir, f"fold_{fold_index}.pth")
        if os.path.exists(fold_path):
            completed.append(fold_index)
    return completed


def _bootstrap_manifest_if_needed(cfg) -> dict:
    manifest = _load_manifest(cfg)
    if manifest:
        return manifest

    completed_folds = _checkpoint_completed_folds(cfg)
    if not completed_folds:
        return {}

    ensemble_dir = cfg.paths.ensemble_dir
    manifest = {
        "run_started_at": None,
        "start_fold": min(completed_folds) if completed_folds else 0,
        "num_folds": cfg.paths.num_folds,
        "dataset_size": None,
        "num_classes": None,
        "status": "complete" if len(completed_folds) == cfg.paths.num_folds else "inferred",
        "completed_folds": completed_folds,
        "folds": {},
        "bootstrapped_from_checkpoints": True,
        "updated_at": None,
    }
    for fold_index in completed_folds:
        fold_path = os.path.join(ensemble_dir, f"fold_{fold_index}.pth")
        manifest["folds"][str(fold_index)] = {
            "status": "complete",
            "checkpoint": fold_path,
        }
    with open(_manifest_path(cfg), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def list_completed_folds():
    """Show which folds have already been trained."""
    cfg = get_config()
    ensemble_dir = cfg.paths.ensemble_dir
    manifest = _bootstrap_manifest_if_needed(cfg)
    
    if not os.path.isdir(ensemble_dir):
        print(f"[INFO] Ensemble directory not found: {ensemble_dir}")
        print("[INFO] No folds have been trained yet.")
        return
    
    print(f"\n{'='*60}")
    print(f"  Fold Status")
    print(f"{'='*60}")
    
    completed = []
    manifest_completed = sorted({int(fold) for fold in manifest.get("completed_folds", [])})
    manifest_folds = manifest.get("folds", {})

    for i in range(cfg.paths.num_folds):
        fold_path = os.path.join(ensemble_dir, f"fold_{i}.pth")
        if i in manifest_completed:
            checkpoint = manifest_folds.get(str(i), {}).get("checkpoint", fold_path)
            size = os.path.getsize(checkpoint) / (1024 * 1024) if os.path.exists(checkpoint) else 0.0
            print(f"  [✓] fold_{i}.pth ({size:.1f} MB)")
            completed.append(i)
        elif os.path.exists(fold_path):
            size = os.path.getsize(fold_path) / (1024 * 1024)  # MB
            print(f"  [~] fold_{i}.pth exists, but not recorded in manifest ({size:.1f} MB)")
        else:
            print(f"  [ ] fold_{i}.pth (not trained)")
    
    if completed:
        next_fold = max(completed) + 1
        if next_fold < cfg.paths.num_folds:
            print(f"\nNext fold to train: {next_fold}")
        else:
            print(f"\nAll {cfg.paths.num_folds} folds completed!")
    else:
        print(f"\nStart training from fold 0")

    if manifest:
        print(f"\nManifest: {_manifest_path(cfg)}")
        print(f"Manifest status: {manifest.get('status', 'unknown')}")
        if manifest.get("bootstrapped_from_checkpoints"):
            print("Manifest source: inferred from existing checkpoints")
    
    print(f"{'='*60}\n")


def train_kfold_from_fold(start_fold: int):
    """Train k-fold starting from a specific fold."""
    cfg = get_config()
    num_folds = cfg.paths.num_folds
    
    # Validate start_fold
    if start_fold < 0 or start_fold >= num_folds:
        print(f"[ERROR] start_fold must be between 0 and {num_folds - 1}")
        sys.exit(1)
    
    # Check which folds already exist
    ensemble_dir = cfg.paths.ensemble_dir
    os.makedirs(ensemble_dir, exist_ok=True)
    
    manifest = _load_manifest(cfg)
    completed = sorted({int(fold) for fold in manifest.get("completed_folds", []) if int(fold) < start_fold})
    if not completed:
        for i in range(start_fold):
            fold_path = os.path.join(ensemble_dir, f"fold_{i}.pth")
            if os.path.exists(fold_path):
                completed.append(i)
    
    if len(completed) < start_fold:
        missing_folds = [i for i in range(start_fold) if i not in completed]
        print(f"[WARNING] Missing folds {missing_folds} that come before fold {start_fold}")
        print(f"[WARNING] Starting anyway, but results may be incomplete.\n")
    
    # Import and run training with start_fold
    from train import train_kfold
    
    pipeline_log = setup_pipeline_logger("kfold_resume")
    
    with pipeline_log.capture_stdio():
        pipeline_log.event(
            "kfold_resume_start",
            start_fold=start_fold,
            num_folds=num_folds,
        )
        
        print("=" * 60)
        print(f"  K-Fold Training (Resume)")
        print(f"  Starting from fold {start_fold} of {num_folds}")
        print("=" * 60)
        print()
        
        fold_accs = train_kfold(
            pipeline_log=pipeline_log,
            start_fold=start_fold
        )
        
        pipeline_log.event(
            "kfold_resume_end",
            fold_accuracies=fold_accs,
            folds=len(fold_accs),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resume K-fold training from a specific fold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_kfold_resume.py --list           # Show fold status
  python train_kfold_resume.py                  # Train all folds from 0
  python train_kfold_resume.py --start-fold 2  # Resume from fold 2
  python train_kfold_resume.py --start-fold 3  # Resume from fold 3
        """
    )
    
    parser.add_argument(
        "--list", action="store_true",
        help="List completed folds and exit"
    )
    
    parser.add_argument(
        "--start-fold", type=int, default=0,
        help="Start training from this fold (0-indexed, default: 0)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_completed_folds()
        sys.exit(0)
    
    train_kfold_from_fold(args.start_fold)
