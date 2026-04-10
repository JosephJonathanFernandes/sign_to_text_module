import numpy as np
from pathlib import Path
from collections import defaultdict

def get_classes_summary(dataset_dir="Dataset"):
    """Get summary of all classes and their files."""
    dataset_path = Path(dataset_dir)
    
    class_info = defaultdict(list)
    
    # Find all .npy files
    npy_files = list(dataset_path.rglob("*.npy"))
    
    # Group by class (folder name)
    for npy_file in npy_files:
        class_name = npy_file.parent.name
        class_info[class_name].append(npy_file)
    
    # Also include empty class folders
    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir() and class_folder.name not in class_info:
            class_info[class_folder.name] = []
    
    print(f"\n{'='*70}")
    print(f"Dataset Summary: {len(class_info)} classes, {len(npy_files)} total files")
    print(f"{'='*70}\n")
    
    # Sort by class name
    for class_name in sorted(class_info.keys()):
        files = class_info[class_name]
        status = "" if len(files) > 0 else " (empty)"
        print(f"{class_name:30s} -> {len(files):3d} files{status}")
    
    print(f"\n{'='*70}\n")

def get_class_files(class_name, dataset_dir="Dataset"):
    """Get all files in a specific class."""
    dataset_path = Path(dataset_dir)
    class_path = dataset_path / class_name
    
    if not class_path.exists():
        print(f"Class folder '{class_name}' not found!")
        get_classes_summary(dataset_dir)
        return
    
    npy_files = sorted(list(class_path.glob("*.npy")))
    
    print(f"\n{'='*70}")
    print(f"Class: {class_name}")
    print(f"Total files: {len(npy_files)}")
    print(f"{'='*70}\n")
    
    total_non_zero = 0
    all_zero_files = []
    
    for i, npy_file in enumerate(npy_files):
        data = np.load(npy_file)
        non_zero = np.count_nonzero(data)
        total_non_zero += non_zero
        
        if non_zero == 0:
            all_zero_files.append(npy_file.name)
            status = "✗ ALL ZEROS"
        else:
            status = "✓ Valid   "
        
        print(f"[{i:2d}] {npy_file.name:30s} shape: {str(data.shape):20s} non-zero: {non_zero:6d} {status}")
    
    print(f"\n{'='*70}")
    print(f"Summary for '{class_name}':")
    print(f"  Total files: {len(npy_files)}")
    print(f"  Valid files: {len(npy_files) - len(all_zero_files)}")
    print(f"  Zero files: {len(all_zero_files)}")
    print(f"  Total non-zero values: {total_non_zero:,}")
    
    if all_zero_files:
        print(f"\n  ⚠️  Files with all zeros:")
        for file in all_zero_files:
            print(f"    - {file}")
    
    print(f"{'='*70}\n")

def main():
    import sys
    
    # Default to processed folder if it exists
    default_dir = "processed" if Path("processed").exists() else "Dataset"
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python check_class_files.py list          # List all classes")
        print("  python check_class_files.py check <class> # Check files in a class")
        print()
        print("Example:")
        print("  python check_class_files.py list")
        print("  python check_class_files.py check 'blind'")
        print("  python check_class_files.py check 'good'")
        print()
        get_classes_summary(default_dir)
        return
    
    command = sys.argv[1]
    
    if command == "list":
        get_classes_summary(default_dir)
    
    elif command == "check":
        if len(sys.argv) < 3:
            print("Usage: python check_class_files.py check <class_name>")
            get_classes_summary(default_dir)
            return
        
        class_name = sys.argv[2]
        get_class_files(class_name, default_dir)
    
    else:
        print(f"Unknown command: {command}")
        print("\nUsage:")
        print("  python check_class_files.py list")
        print("  python check_class_files.py check <class_name>")

if __name__ == "__main__":
    main()
