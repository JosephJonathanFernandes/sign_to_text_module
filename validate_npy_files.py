"""
Validates all .npy files in the processed directory and identifies corrupted ones.
Can optionally remove corrupt files or attempt repairs.
"""

import os
import numpy as np
from pathlib import Path
from config import get_config

cfg = get_config()
PROCESSED_DIR = cfg.paths.processed_dir


def validate_npy_file(fpath):
    """
    Try to load a .npy file and validate it.
    Returns: (is_valid, error_msg, shape_info)
    """
    try:
        data = np.load(fpath)
        if data.size == 0:
            return False, "Empty file", None
        return True, None, data.shape
    except Exception as e:
        return False, str(e), None


def scan_processed_dir(remove_corrupt=False, verbose=True):
    """
    Scan all .npy files in processed/ and identify corrupted ones.
    """
    if not os.path.isdir(PROCESSED_DIR):
        print(f"ERROR: {PROCESSED_DIR} not found!")
        return

    corrupt_files = []
    valid_count = 0
    total_count = 0

    # Walk through all class directories
    for class_name in sorted(os.listdir(PROCESSED_DIR)):
        class_dir = os.path.join(PROCESSED_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\n[{class_name}]", end=" ")
        class_corrupt = 0

        for fname in sorted(os.listdir(class_dir)):
            if not fname.endswith(".npy"):
                continue

            fpath = os.path.join(class_dir, fname)
            total_count += 1
            is_valid, error_msg, shape_info = validate_npy_file(fpath)

            if is_valid:
                valid_count += 1
                print(".", end="", flush=True)
            else:
                class_corrupt += 1
                corrupt_files.append((fpath, error_msg))
                print("X", end="", flush=True)

                if remove_corrupt:
                    try:
                        os.remove(fpath)
                        print(f" DELETED {fname}", end="", flush=True)
                    except Exception as e:
                        print(f" FAILED TO DELETE {fname}: {e}", end="", flush=True)

        if class_corrupt > 0:
            print(f" [{class_corrupt} corrupt]")
        else:
            print()

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {valid_count}/{total_count} files valid")
    
    if corrupt_files:
        print(f"\n⚠️  Found {len(corrupt_files)} CORRUPT FILES:")
        for fpath, error_msg in corrupt_files[:20]:  # Show first 20
            print(f"  {fpath}")
            print(f"    Error: {error_msg}")
        if len(corrupt_files) > 20:
            print(f"  ... and {len(corrupt_files) - 20} more")
        
        if remove_corrupt:
            print(f"\n✓ Removed {len(corrupt_files)} corrupt files")
    else:
        print("\n✓ All files are valid!")

    return corrupt_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate .npy files in processed/")
    parser.add_argument("--remove", action="store_true", 
                        help="Remove corrupt files")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output")
    
    args = parser.parse_args()
    
    corrupt_files = scan_processed_dir(
        remove_corrupt=args.remove,
        verbose=args.verbose
    )
    
    if corrupt_files and not args.remove:
        print("\nTo remove corrupt files, run with --remove flag:")
        print("  python validate_npy_files.py --remove")
