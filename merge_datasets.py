"""
Merge converted friend's dataset with existing dataset.
Combines overlapping classes and creates unified structure.
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict

# Paths
YOUR_DATASET = Path("Dataset")
FRIEND_DATASET = Path("friend_dataset_converted")
PROCESSED_DIR = Path("processed")

def merge_datasets():
    """Merge friend's dataset into existing processed dataset."""
    
    if not FRIEND_DATASET.exists():
        print(f"❌ Friend dataset not found: {FRIEND_DATASET}")
        return False
    
    if not PROCESSED_DIR.exists():
        print(f"❌ Processed dataset not found: {PROCESSED_DIR}")
        return False
    
    # Count before merge
    before_counts = {}
    for class_dir in PROCESSED_DIR.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.npy")))
            if count > 0:
                before_counts[class_dir.name] = count
    
    print("=" * 70)
    print("MERGING FRIEND'S DATASET INTO EXISTING DATASET")
    print("=" * 70)
    
    print("\n📊 BEFORE MERGE:")
    total_before = sum(before_counts.values())
    for cls, count in sorted(before_counts.items()):
        print(f"  {cls:15} : {count:3} samples")
    print(f"  {'TOTAL':15} : {total_before:3} samples")
    
    # Merge friend's classes
    friend_merged = defaultdict(int)
    
    for class_dir in FRIEND_DATASET.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            
            # Create or get target directory
            target_dir = PROCESSED_DIR / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all .npy files
            npy_files = list(class_dir.glob("*.npy"))
            for npy_file in npy_files:
                dest_file = target_dir / npy_file.name
                shutil.copy2(npy_file, dest_file)
                friend_merged[class_name] += 1
                print(f"  ✓ Merged {class_name}/{npy_file.name}")
    
    # Report results
    print("\n📊 AFTER MERGE:")
    after_counts = {}
    for class_dir in PROCESSED_DIR.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.npy")))
            if count > 0:
                after_counts[class_dir.name] = count
    
    total_after = sum(after_counts.values())
    for cls in sorted(set(list(before_counts.keys()) + list(after_counts.keys()))):
        before = before_counts.get(cls, 0)
        after = after_counts.get(cls, 0)
        added = after - before
        status = "→ MERGED" if added > 0 else ""
        print(f"  {cls:15} : {before:3} + {added:3} = {after:3}  {status}")
    
    print(f"  {'TOTAL':15} : {total_before:3} + {sum(friend_merged.values()):3} = {total_after:3}")
    
    print(f"\n✅ Merge complete! Dataset now has {total_after} samples in {len(after_counts)} classes.")
    return True

if __name__ == "__main__":
    merge_datasets()
