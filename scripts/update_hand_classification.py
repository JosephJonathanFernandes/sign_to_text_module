import os
import json
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.config import get_config

def main():
    cfg = get_config()
    processed_dir = cfg.paths.processed_dir
    json_path = os.path.join(cfg.paths.base_dir, "..", "..", "data", "hand_sign_classification.json")
    json_path = os.path.abspath(json_path)
    
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Feature dimensions
    ldim = cfg.landmarks.landmark_dim_per_hand  # 63
    left_start, left_end = 0, ldim
    right_start, right_end = ldim, ldim * 2

    two_hands_classes = []
    one_hand_classes = []
    threshold = 0.20

    print("Analyzing dataset...")
    
    classes = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    
    for cls in sorted(classes):
        cls_dir = os.path.join(processed_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
        
        if not files:
            continue
            
        total_frames = 0
        two_hands_frames = 0
        
        for f in files:
            file_path = os.path.join(cls_dir, f)
            try:
                seq = np.load(file_path) # shape: (num_frames, feature_dim)
                total_frames += seq.shape[0]
                
                # Check presence of left and right hands
                # A hand is present if its raw coordinates are not all zeros
                left_present = np.any(seq[:, left_start:left_end] != 0, axis=1)
                right_present = np.any(seq[:, right_start:right_end] != 0, axis=1)
                
                # Frames where both hands are present
                both_present = left_present & right_present
                two_hands_frames += np.sum(both_present)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        if total_frames > 0:
            ratio = two_hands_frames / total_frames
            if ratio >= threshold:
                two_hands_classes.append(cls)
                print(f"[TWO HANDS] {cls:15s} (Ratio: {ratio:.1%})")
            else:
                one_hand_classes.append(cls)
                print(f"[ONE HAND]  {cls:15s} (Ratio: {ratio:.1%})")
                
    # Update JSON
    if "axes" in data and "hand_count" in data["axes"]:
        data["axes"]["hand_count"]["one_hand"] = one_hand_classes
        data["axes"]["hand_count"]["two_hands"] = two_hands_classes
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\nSuccessfully updated {json_path}")
        print(f"One-hand signs: {len(one_hand_classes)}")
        print(f"Two-hand signs: {len(two_hands_classes)}")
    else:
        print("JSON structure not as expected. Missing axes.hand_count.")

if __name__ == "__main__":
    main()
