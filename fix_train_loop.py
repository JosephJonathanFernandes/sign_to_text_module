#!/usr/bin/env python3
"""Script to complete the train.py modifications."""

def fix_train_loop():
    with open('train.py', 'r') as f:
        content = f.read()
    
    # Find and replace the specific loop in train_kfold
    # We need to find the occurrence after "Params per fold"
    old_line = 'for fold, (train_idx, val_idx) in enumerate(split_iter):'
    new_line = 'for fold, (train_idx, val_idx) in enumerate(split_iter[start_fold:], start=start_fold):'
    
    # Find the position after "Params per fold" marker
    marker = 'print(f"[Model] Params per fold: {total_p:,}\\n")'
    if marker in content:
        # Find the loop after this marker
        marker_pos = content.find(marker)
        loop_pos = content.find(old_line, marker_pos)
        
        if loop_pos > marker_pos:
            # Replace only this occurrence
            content = content[:loop_pos] + new_line + content[loop_pos + len(old_line):]
            print(f'[✓] Updated training loop')
        else:
            print('[!] Could not find loop after marker')
    else:
        print('[!] Could not find marker')
    
    with open('train.py', 'w') as f:
        f.write(content)
    
    print('[✓] train.py loop updated!')

if __name__ == '__main__':
    fix_train_loop()
