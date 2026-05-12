"""
Simple utility to find and remove corrupt .npy files.
Doesn't require config imports - works standalone.
"""

import os
import numpy as np
import sys
from pathlib import Path


def validate_npy_file(fpath):
    """Try to load a .npy file. Returns True if valid."""
    try:
        data = np.load(fpath)
        return data.size > 0
    except Exception:
        return False


def find_corrupt_files(processed_dir="processed", remove=False):
    """
    Scan processed/ directory for corrupt .npy files.
    """
    if not os.path.isdir(processed_dir):
        print(f"ERROR: {processed_dir}/ not found!")
        return []

    corrupt_files = []

    # Walk through all class directories
    for class_name in sorted(os.listdir(processed_dir)):
        class_dir = os.path.join(processed_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"[{class_name}] ", end="", flush=True)

        for fname in sorted(os.listdir(class_dir)):
            if not fname.endswith(".npy"):
                continue

            fpath = os.path.join(class_dir, fname)
            
            if not validate_npy_file(fpath):
                corrupt_files.append(fpath)
                print("!", end="", flush=True)
                
                if remove:
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        print(f"(FAILED TO DELETE)", end="", flush=True)
            else:
                print(".", end="", flush=True)

        print()

    # Summary
    print("\n" + "=" * 60)
    if corrupt_files:
        print(f"Found {len(corrupt_files)} CORRUPT files:")
        for fpath in corrupt_files[:10]:
            print(f"  {fpath}")
        if len(corrupt_files) > 10:
            print(f"  ... and {len(corrupt_files) - 10} more")
        
        if remove:
            print(f"\n[OK] Removed {len(corrupt_files)} corrupt files")
        else:
            print("\nTo remove these files, run:")
            print("  python find_corrupt_npy.py --remove")
    else:
        print("[OK] No corrupt files found!")

    return corrupt_files


if __name__ == "__main__":
    remove = "--remove" in sys.argv or "-r" in sys.argv
    corrupt = find_corrupt_files("processed", remove=remove)
    sys.exit(len(corrupt) > 0)
