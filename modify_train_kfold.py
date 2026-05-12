#!/usr/bin/env python3
"""Script to modify train.py to support start_fold parameter."""

import re

def modify_train_py():
    # Read train.py
    with open('train.py', 'r') as f:
        lines = f.readlines()
    
    # Find and modify the function signature
    for i, line in enumerate(lines):
        if 'def train_kfold(pipeline_log: PipelineLogger | None = None) -> list:' in line:
            lines[i] = line.replace(
                'def train_kfold(pipeline_log: PipelineLogger | None = None) -> list:',
                'def train_kfold(pipeline_log: PipelineLogger | None = None, start_fold: int = 0) -> list:'
            )
            print(f'[✓] Updated function signature at line {i+1}')
            break
    
    # Find and modify the docstring
    for i, line in enumerate(lines):
        if 'Train NUM_FOLDS models using disjoint source-aware K-fold CV.' in line:
            # Find the closing """ of the docstring
            for j in range(i, min(i+10, len(lines))):
                if '"""' in lines[j] and j > i:
                    # Insert before the closing """
                    docstring_end = j
                    # Check if it already has Args section
                    if 'Args:' not in ''.join(lines[i:j]):
                        insert_text = '''    
    Args:
        pipeline_log: Optional logger for pipeline events
        start_fold: Starting fold index (0-indexed). Default: 0
'''
                        lines.insert(j, insert_text)
                        print(f'[✓] Updated docstring at line {i+1}')
                    break
            break
    
    # Find and modify the training loop
    for i, line in enumerate(lines):
        if 'for fold, (train_idx, val_idx) in enumerate(split_iter):' in line:
            # Check if this is in train_kfold (look for nearby context)
            context = ''.join(lines[max(0, i-10):i])
            if 'fold_accs = []' in context and 'all_val_correct = 0' in context:
                lines[i] = line.replace(
                    'for fold, (train_idx, val_idx) in enumerate(split_iter):',
                    'for fold, (train_idx, val_idx) in enumerate(split_iter[start_fold:], start=start_fold):'
                )
                print(f'[✓] Updated training loop at line {i+1}')
                break
    
    # Write back
    with open('train.py', 'w') as f:
        f.writelines(lines)
    
    print('[✓] train.py modified successfully!')
    print('\nNow you can use:')
    print('  python train_kfold_resume.py --list        # Show fold status')
    print('  python train_kfold_resume.py --start-fold 2  # Resume from fold 2')

if __name__ == '__main__':
    modify_train_py()
