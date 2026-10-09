import os
import argparse
import numpy as np
import torch

from src.core.config import get_config
from src.inference.ensemble import load_ensemble, ensemble_predict

def evaluate_clean_npy_test_set(input_dir: str):
    """
    Evaluates the model on a clean directory of .npy recordings.
    
    Expected directory structure:
    input_dir/
       ├── sign_name_1/
       │   ├── file1.npy
       │   └── file2.npy
       ├── sign_name_2/
       │   ├── file1.npy
       │   └── file2.npy
    """
    print(f"--- Starting Clean NPY Evaluation ---")
    print(f"Reading features from: {input_dir}")
    
    cfg = get_config()
    
    classes_found = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    if not classes_found:
        print(f"Error: No class directories found in {input_dir}")
        return
        
    print(f"Found {len(classes_found)} class directories.")
    
    all_features = []
    all_labels = []
    all_fpaths = []
    
    for cls_name in classes_found:
        cls_dir = os.path.join(input_dir, cls_name)
        
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith('.npy')]
        
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            seq = np.load(fpath)
            
            if seq.shape[0] != cfg.frame_features.num_frames:
                print(f"Warning: {fpath} resulted in shape {seq.shape}. Expected {cfg.frame_features.num_frames} frames. Skipping.")
                continue
                
            all_features.append(seq)
            all_labels.append(cls_name)
            all_fpaths.append(fpath)
            
    print(f"Loaded {len(all_features)} independent samples.")
    
    # 2. Evaluation
    print("\n--- Starting Model Evaluation ---")
    print("Loading baseline model...")
    models, classes, num_classes = load_ensemble()
    
    # Create reverse lookup for true labels
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    correct = 0
    total = 0
    unknown_classes = 0
    
    print("\n[Results]")
    for seq, true_label_str, fpath in zip(all_features, all_labels, all_fpaths):
        # We need the true label index
        clean_true_label = true_label_str.split(". ", 1)[-1].strip().lower()
        
        if clean_true_label not in class_to_idx:
            print(f"Warning: Found label '{clean_true_label}' which is not in the model's classes. Skipping.")
            unknown_classes += 1
            continue
            
        true_idx = class_to_idx[clean_true_label]
        
        # Predict
        pred_idx, conf, _ = ensemble_predict(models, seq)
        
        is_correct = (pred_idx == true_idx)
        if is_correct:
            correct += 1
        total += 1
        
        status = "HIT " if is_correct else "MISS"
        print(f"[{status}] {os.path.basename(fpath).ljust(25)} | True: {clean_true_label.ljust(15)} | Pred: {classes[pred_idx].ljust(15)} | Conf: {conf:.4f}")
        
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\nFinal Clean Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    else:
        print("\nNo valid samples evaluated.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a clean independent test set of .npy files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to .npy files grouped by sign.")
    args = parser.parse_args()
    
    evaluate_clean_npy_test_set(args.input_dir)
