#!/usr/bin/env python3
"""
Augmentation pipeline orchestrator.

Runs augmentation, merge, and cleanup in sequence:
1. Augments landmarks N times
2. Merges augmentations with 4 different modes (splice, blend, hand_swap, hybrid), N times each
3. Cleans up the dataset once

Usage:
    python augment_pipeline.py --class old
    python augment_pipeline.py --class old --augment-iterations 4 --merge-iterations 4
    
This will run:
    - Augmentation: 4 times
    - Merge: 4 modes × 4 iterations = 16 times total
    - Cleanup: 1 time
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from config import get_config

cfg = get_config()


def run_command(cmd, step_name, iteration=None):
    """Run a command and report results."""
    iter_suffix = f" (iteration {iteration})" if iteration is not None else ""
    print(f"\n{'=' * 70}")
    print(f"Running: {step_name}{iter_suffix}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ {step_name}{iter_suffix} completed successfully in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {step_name}{iter_suffix} failed after {elapsed:.1f}s")
        print(f"  Exit code: {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run augmentation pipeline: augment → merge → cleanup"
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        required=False,
        default=None,
        help="Class name to augment (e.g., 'old', 'young'). If not provided, runs for ALL classes."
    )
    parser.add_argument(
        "--augment-iterations",
        type=int,
        default=10,
        help="Number of times to run augmentation (default: 10)"
    )
    parser.add_argument(
        "--merge-iterations",
        type=int,
        default=10,
        help="Number of times to run merge (default: 10)"
    )
    parser.add_argument(
        "--augment-n",
        type=int,
        default=10,
        help="Number of augmentations per iteration (default: 10)"
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup step"
    )
    
    args = parser.parse_args()
    
    # Get class list
    if args.class_name:
        classes = [args.class_name.strip()]
    else:
        # Get all classes from processed directory
        processed_dir = cfg.paths.processed_dir
        classes = sorted([
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ])
        if not classes:
            print(f"Error: No classes found in {processed_dir}")
            sys.exit(1)
    
    print(f"\n{'*' * 70}")
    print(f"Augmentation Pipeline")
    print(f"{'*' * 70}")
    print(f"Classes to process: {len(classes)} - {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
    print(f"Augment iterations: {args.augment_iterations}")
    print(f"Merge iterations: {args.merge_iterations}")
    print(f"Augment N: {args.augment_n}")
    print(f"{'*' * 70}\n")
    
    failed_steps = []
    total_classes = len(classes)
    
    # Process each class
    for class_idx, class_name in enumerate(classes, 1):
        print(f"\n{'#' * 70}")
        print(f"Processing Class {class_idx}/{total_classes}: {class_name}")
        print(f"{'#' * 70}\n")
        
        # Phase 1: Augmentation
        print(f"\n### Phase 1: Augmentation ({args.augment_iterations} iterations) ###")
        for i in range(1, args.augment_iterations + 1):
            cmd = [
                sys.executable,
                "main.py",
                "--augment-landmarks",
                "--augment-landmarks-cls",
                class_name,
                "--augment-landmarks-n",
                str(args.augment_n)
            ]
            if not run_command(cmd, f"[{class_name}] Augmentation", iteration=i):
                failed_steps.append(f"{class_name}: Augmentation iteration {i}")
        
        # Phase 2: Merge (all 4 types)
        merge_modes = ["splice", "blend", "hand_swap", "hybrid"]
        print(f"\n### Phase 2: Merge (4 modes × {args.merge_iterations} iterations = {4 * args.merge_iterations} total) ###")
        for mode in merge_modes:
            for i in range(1, args.merge_iterations + 1):
                cmd = [
                    sys.executable,
                    "merge_augmentations.py",
                    "processed",
                    "--output_dir",
                    "processed",
                    "--n",
                    "2",
                    "--mode",
                    mode,
                    "--class",
                    class_name
                ]
                if not run_command(cmd, f"[{class_name}] Merge ({mode})", iteration=i):
                    failed_steps.append(f"{class_name}: Merge {mode} iteration {i}")
        
        # Phase 3: Cleanup (per class)
        if not args.skip_cleanup:
            print(f"\n### Phase 3: Cleanup ###")
            cmd = [
                sys.executable,
                "cleanup_dataset_npy.py",
                "--class",
                class_name
            ]
            if not run_command(cmd, f"[{class_name}] Cleanup"):
                failed_steps.append(f"{class_name}: Cleanup")
        else:
            print(f"\n### Phase 3: Cleanup (skipped) ###")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 70}")
    
    if failed_steps:
        print(f"✗ Pipeline completed with {len(failed_steps)} failure(s):")
        for step in failed_steps:
            print(f"  - {step}")
        sys.exit(1)
    else:
        print(f"✓ Pipeline completed successfully for {total_classes} class(es)!")
        sys.exit(0)


if __name__ == "__main__":
    main()
