import numpy as np
from pathlib import Path

def check_npy_files_for_zeros(dataset_dir="processed"):
    """Check if .npy files contain all zeros."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Dataset directory '{dataset_dir}' not found.")
        return
    
    all_zero_files = []
    non_zero_files = []
    
    # Find all .npy files recursively
    npy_files = list(dataset_path.rglob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {dataset_dir}")
        return
    
    print(f"Checking {len(npy_files)} .npy files...\n")
    
    for npy_file in sorted(npy_files):
        try:
            data = np.load(npy_file)
            is_all_zero = np.all(data == 0)
            
            if is_all_zero:
                all_zero_files.append(str(npy_file))
                print(f"✓ ALL ZEROS: {npy_file.relative_to(dataset_path)}")
            else:
                non_zero_files.append(str(npy_file))
                non_zero_count = np.count_nonzero(data)
                print(f"✗ NON-ZERO:  {npy_file.relative_to(dataset_path)} ({non_zero_count} non-zero values)")
        except Exception as e:
            print(f"ERROR loading {npy_file}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Total files: {len(npy_files)}")
    print(f"All zero files: {len(all_zero_files)}")
    print(f"Non-zero files: {len(non_zero_files)}")
    
    if all_zero_files:
        print(f"\n⚠️  Files with all zeros:")
        for file in all_zero_files:
            print(f"  - {file}")

if __name__ == "__main__":
    check_npy_files_for_zeros()
