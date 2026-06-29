import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure src module can be imported when script is run directly
sys.path.insert(0, str(Path.cwd()))

from collections import defaultdict
from src.core.config import get_config

def generate_heuristics():
    cfg = get_config()
    processed_dir = Path(cfg.paths.processed_dir)
    
    if not processed_dir.exists():
        print(f"Error: {processed_dir} does not exist.")
        return

    # Derive mapping dynamically from feature extraction logic
    # As per src/shared/feature_extractor.py, features are concatenated in order:
    # [left_norm (63), right_norm (63), left_rel (63), right_rel (63), proximity (1)]
    # We will use this layout.
    LANDMARK_DIM = 63
    LANDMARK_SCHEMA = {
        "left_raw": [0, LANDMARK_DIM],
        "right_raw": [LANDMARK_DIM, LANDMARK_DIM * 2],
        "left_relative": [LANDMARK_DIM * 2, LANDMARK_DIM * 3],
        "right_relative": [LANDMARK_DIM * 3, LANDMARK_DIM * 4]
    }
    
    all_files = list(processed_dir.rglob("*.npy"))
    if not all_files:
        print(f"No .npy files found in {processed_dir}")
        return

    print(f"Found {len(all_files)} .npy files. Processing heuristics...")

    video_heuristics = defaultdict(list)
    keypoints_data = []

    for fpath in all_files:
        class_name = fpath.parent.name
        # Remove any leading number from class name e.g., "01. hello" -> "hello"
        if '.' in class_name:
            class_name = class_name.split('.', 1)[1].strip()
            
        sample_id = fpath.stem
        try:
            tensor = np.load(fpath) # expected shape e.g., (20, 506) or (20, 253)
        except Exception as e:
            continue
            
        if len(tensor.shape) != 2:
            continue
            
        num_frames = tensor.shape[0]
        
        l_hand_present = 0
        r_hand_present = 0
        
        l_movements = []
        r_movements = []
        
        l_locs = []
        r_locs = []
        
        prev_l_centroid = None
        prev_r_centroid = None
        
        l_orientations = []
        r_orientations = []

        # Iterate over frames to extract heuristics and generate keypoints
        for frame_idx in range(num_frames):
            frame = tensor[frame_idx]
            
            # Extract hands
            l_raw = frame[LANDMARK_SCHEMA["left_raw"][0] : LANDMARK_SCHEMA["left_raw"][1]]
            r_raw = frame[LANDMARK_SCHEMA["right_raw"][0] : LANDMARK_SCHEMA["right_raw"][1]]
            
            l_rel = frame[LANDMARK_SCHEMA["left_relative"][0] : LANDMARK_SCHEMA["left_relative"][1]]
            r_rel = frame[LANDMARK_SCHEMA["right_relative"][0] : LANDMARK_SCHEMA["right_relative"][1]]
            
            # Check presence using tolerance
            l_present = np.abs(l_raw).sum() > 1e-6
            r_present = np.abs(r_raw).sum() > 1e-6
            
            if l_present:
                l_hand_present += 1
                # Centroid movement
                l_reshaped = l_raw.reshape(21, 3)
                centroid = l_reshaped.mean(axis=0)
                if prev_l_centroid is not None:
                    l_movements.append(np.linalg.norm(centroid - prev_l_centroid))
                prev_l_centroid = centroid
                
                # Location (relative to face nose)
                l_rel_reshaped = l_rel.reshape(21, 3)
                centroid_rel_y = l_rel_reshaped.mean(axis=0)[1]
                # Y is positive downwards
                if centroid_rel_y > 1.0:
                    l_locs.append("neutral_space")
                elif centroid_rel_y > -0.2:
                    l_locs.append("upper_space")
                else:
                    l_locs.append("face_near")
                
                # Add to CSV
                row = [sample_id, class_name, frame_idx, "left"] + l_raw.tolist()
                keypoints_data.append(row)
                
            if r_present:
                r_hand_present += 1
                # Centroid movement
                r_reshaped = r_raw.reshape(21, 3)
                centroid = r_reshaped.mean(axis=0)
                if prev_r_centroid is not None:
                    r_movements.append(np.linalg.norm(centroid - prev_r_centroid))
                prev_r_centroid = centroid
                
                # Location
                r_rel_reshaped = r_rel.reshape(21, 3)
                centroid_rel_y = r_rel_reshaped.mean(axis=0)[1]
                if centroid_rel_y > 1.0:
                    r_locs.append("neutral_space")
                elif centroid_rel_y > -0.2:
                    r_locs.append("upper_space")
                else:
                    r_locs.append("face_near")
                
                # Add to CSV
                row = [sample_id, class_name, frame_idx, "right"] + r_raw.tolist()
                keypoints_data.append(row)
                
        # Aggregate Video-Level Heuristics
        presence_threshold = num_frames * 0.3
        two_hands = l_hand_present > presence_threshold and r_hand_present > presence_threshold
        hand_count = "two_hands" if two_hands else "one_hand"
        
        # Movement
        avg_mov = 0.0
        mov_list = []
        if two_hands:
            mov_list = l_movements + r_movements
        elif l_hand_present > presence_threshold:
            mov_list = l_movements
        elif r_hand_present > presence_threshold:
            mov_list = r_movements
            
        if mov_list:
            avg_mov = float(np.mean(mov_list))
        
        if avg_mov < 0.01:
            movement = "static_hold"
        elif avg_mov < 0.03:
            movement = "small_motion"
        elif avg_mov < 0.07:
            movement = "medium_motion"
        else:
            movement = "large_motion"
            
        # Location (majority vote)
        all_locs = l_locs + r_locs
        location = "unknown"
        if all_locs:
            location = max(set(all_locs), key=all_locs.count)
            
        # Symmetry (placeholder simple heuristic based on movement differences if 2 hands)
        symmetry = "unknown"
        if two_hands and l_movements and r_movements:
            min_len = min(len(l_movements), len(r_movements))
            diff = np.mean(np.abs(np.array(l_movements[:min_len]) - np.array(r_movements[:min_len])))
            symmetry = "symmetric" if diff < 0.02 else "asymmetric"
        
        # Orientation is complex without real depth/world coordinates from standard normalized hands.
        # We'll set a placeholder based on hand to face relation or just "unknown" for strict purity.
        orientation = "unknown"
        
        video_heuristics[class_name].append({
            "hand_count": hand_count,
            "movement": movement,
            "location": location,
            "orientation": orientation,
            "symmetry": symmetry
        })

    # Aggregate to class level
    final_classification = {
        "description": "ISL classification generated from dataset observations",
        "signs": {}
    }
    confidence_stats = {}
    candidate_map = {"similar_pairs": []} # placeholder
    
    for cls_name, heuristics in video_heuristics.items():
        n = len(heuristics)
        if n == 0:
            continue
            
        counts = defaultdict(lambda: defaultdict(int))
        for h in heuristics:
            counts["hand_count"][h["hand_count"]] += 1
            counts["movement"][h["movement"]] += 1
            counts["location"][h["location"]] += 1
            counts["orientation"][h["orientation"]] += 1
            counts["symmetry"][h["symmetry"]] += 1
            
        majority = {}
        confidence = {}
        for prop in ["hand_count", "movement", "location", "orientation", "symmetry"]:
            best_val = max(counts[prop], key=counts[prop].get)
            majority[prop] = best_val
            confidence[f"{prop}_confidence"] = round(counts[prop][best_val] / n, 2)
            
        final_classification["signs"][cls_name] = majority
        confidence_stats[cls_name] = confidence
        
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / "hand_sign_classification.json", "w") as f:
        json.dump(final_classification, f, indent=2)
        
    with open(out_dir / "confidence_statistics.json", "w") as f:
        json.dump(confidence_stats, f, indent=2)
        
    with open(out_dir / "candidate_map.json", "w") as f:
        json.dump(candidate_map, f, indent=2)
        
    # Write CSV
    csv_headers = ["sample_id", "class_name", "frame_number", "hand"] + [f"{axis}{i+1}" for i in range(21) for axis in ['x', 'y', 'z']]
    df = pd.DataFrame(keypoints_data, columns=csv_headers)
    df.to_csv(out_dir / "keypoints.csv", index=False)
    
    print("Done! Generated:")
    print(" - data/hand_sign_classification.json")
    print(" - data/confidence_statistics.json")
    print(" - data/candidate_map.json")
    print(" - data/keypoints.csv")

if __name__ == "__main__":
    generate_heuristics()
