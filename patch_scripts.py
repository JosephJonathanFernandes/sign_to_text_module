import sys
import numpy as np

def patch_robustness():
    filepath = 'src/tools/evaluate_robustness.py'
    with open(filepath, 'r') as f:
        content = f.read()
    
    if 'import argparse' not in content:
        content = content.replace('import os\n', 'import os\nimport argparse\n')
        
    content = content.replace('def evaluate_baseline():', 'def evaluate_baseline(val_only=False):')
    
    main_block = """if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-only', action='store_true')
    args = parser.parse_args()
    evaluate_baseline(val_only=args.val_only)"""
    
    content = content.replace('if __name__ == "__main__":\n    evaluate_baseline()', main_block)
    
    robust_filter = """
    if val_only:
        val_idx = set(np.load("results_v2/val_idx.npy"))
        test_indices = [i for i in test_indices if i in val_idx]
        valid_sign_indices = [i for i in valid_sign_indices if i in val_idx]
        reject_indices = [i for i in reject_indices if i in val_idx]
        print(f"Filtered to VAL ONLY. Now {len(test_indices)} samples.")
    """
    if 'print(f"Test set (natural distribution)' in content:
        content = content.replace('    print(f"Test set (natural distribution) includes {len(test_indices)} samples.")', robust_filter + '\n    print(f"Test set (natural distribution) includes {len(test_indices)} samples.")')

    # Also, we need to save the outputs in results_v2!
    # Let's override plt.savefig
    content = content.replace('plt.savefig(os.path.join("diagrams",', 'plt.savefig(os.path.join("results_v2",')
    content = content.replace('with open("diagrams/', 'with open("results_v2/')

    with open(filepath, 'w') as f:
        f.write(content)

def patch_metrics():
    filepath = 'src/tools/generate_report_metrics.py'
    with open(filepath, 'r') as f:
        content = f.read()
    
    if 'import argparse' not in content:
        content = content.replace('import os\n', 'import os\nimport argparse\n')

    content = content.replace('def main():', 'def main(val_only=False):')
    
    main_block = """if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-only', action='store_true')
    args = parser.parse_args()
    main(val_only=args.val_only)"""
    
    content = content.replace('if __name__ == "__main__":\n    main()', main_block)

    metrics_filter = """
    if val_only:
        val_idx = set(np.load("results_v2/val_idx.npy"))
        test_indices = [i for i in test_indices if i in val_idx]
        val_test_indices = [i for i in val_test_indices if i in val_idx]
        print(f"Filtered to VAL ONLY. Now test={len(test_indices)} val={len(val_test_indices)} samples.")
    """
    if 'print(f"Found {len(test_indices)} test/unseen samples' in content:
        content = content.replace('    print(f"Found {len(test_indices)} test/unseen samples', metrics_filter + '\n    print(f"Found {len(test_indices)} test/unseen samples')
    
    content = content.replace('os.path.join("benchmarks",', 'os.path.join("results_v2",')

    with open(filepath, 'w') as f:
        f.write(content)

patch_robustness()
patch_metrics()
print("Patched.")
