import os
import sys
import json
import csv
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Ensure src module can be imported when script is run directly
sys.path.insert(0, str(Path.cwd()))
import argparse
from src.core.config import get_config

def get_finger_state(raw):
    state = ""
    # thumb (4 vs 2 or wrist)
    if np.linalg.norm(raw[4] - raw[0]) > np.linalg.norm(raw[2] - raw[0]):
        state += "1"
    else:
        state += "0"
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if np.linalg.norm(raw[tip] - raw[0]) > np.linalg.norm(raw[pip] - raw[0]):
            state += "1"
        else:
            state += "0"
    return state

def get_hand_shape(raw, finger_state):
    # check pinch
    if np.linalg.norm(raw[4] - raw[8]) < 0.05:
        return "pinch"
    
    if finger_state == "00000": return "fist"
    if finger_state == "11111": return "open_palm"
    if finger_state in ["01111", "10111", "00111"]: return "spread"
    if finger_state in ["01000", "11000", "01100"]: return "pointing"
    
    # Check curved: tips are closer to wrist than pips but not fully closed?
    # Simple heuristic: if not fist and not open, maybe curved if all tips are closer than pips but not zero distance.
    return "unknown"

def get_orientation(raw, hand_type):
    v1 = raw[5] - raw[0]
    v2 = raw[17] - raw[0]
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-6: return "unknown"
    normal = normal / norm_len
    
    # For left hand, cross product might point opposite to right hand
    if hand_type == "left":
        normal = -normal
        
    nx, ny, nz = normal
    abs_nx, abs_ny, abs_nz = abs(nx), abs(ny), abs(nz)
    
    if abs_ny > abs_nx and abs_ny > abs_nz:
        return "palm_down" if ny > 0 else "palm_up"
    elif abs_nx > abs_ny and abs_nx > abs_nz:
        return "palm_right" if nx > 0 else "palm_left"
    else:
        return "forward"

def get_motion_direction(trajectory):
    if len(trajectory) < 2: return "static"
    diff = trajectory[-1] - trajectory[0]
    dist = np.linalg.norm(diff)
    if dist < 0.05: return "static"
    
    dx, dy, dz = diff
    abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
    if abs_dx > abs_dy and abs_dx > abs_dz:
        return "left_to_right" if dx > 0 else "right_to_left"
    elif abs_dy > abs_dx and abs_dy > abs_dz:
        return "down" if dy > 0 else "up"
    else:
        return "backward" if dz > 0 else "forward"

def get_trajectory_pattern(trajectory):
    if len(trajectory) < 3: return "static"
    
    # Smooth the trajectory to reduce noise accumulation in path_len
    window = 3
    smoothed = []
    for i in range(len(trajectory)):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(trajectory), i + window // 2 + 1)
        smoothed.append(np.mean(trajectory[start_idx:end_idx], axis=0))
    trajectory = smoothed

    start = trajectory[0]
    end = trajectory[-1]
    dist = np.linalg.norm(end - start)
    path_len = sum(np.linalg.norm(trajectory[i] - trajectory[i-1]) for i in range(1, len(trajectory)))
    
    if path_len < 0.15: return "static"
    ratio = path_len / (dist + 1e-6)
    
    if ratio < 1.5: return "linear"
    elif ratio < 2.5: return "arc"
    else: 
        if dist < path_len * 0.3 and path_len > 0.4:
            traj_arr = np.array(trajectory)
            ranges = traj_arr.max(axis=0) - traj_arr.min(axis=0)
            sorted_ranges = np.sort(ranges)[::-1]
            if sorted_ranges[0] > 0.1 and sorted_ranges[1] > 0.05:
                return "circular"
        if path_len > 0.3:
            return "zigzag"
    return "unknown"

def generate_heuristics(class_only=None):
    cfg = get_config()
    processed_dir = Path(cfg.paths.processed_dir)
    
    if not processed_dir.exists():
        print(f"Error: {processed_dir} does not exist.")
        return

    LANDMARK_DIM = 63
    LANDMARK_SCHEMA = {
        "left_raw": [0, LANDMARK_DIM],
        "right_raw": [LANDMARK_DIM, LANDMARK_DIM * 2],
        "left_relative": [LANDMARK_DIM * 2, LANDMARK_DIM * 3],
        "right_relative": [LANDMARK_DIM * 3, LANDMARK_DIM * 4]
    }
    
    if class_only:
        class_path = processed_dir / class_only
        if not class_path.exists():
            print(f"Error: Class directory {class_path} not found.")
            return
        all_files = list(class_path.rglob("*.npy"))
    else:
        all_files = list(processed_dir.rglob("*.npy"))
        
    if not all_files:
        print(f"No .npy files found in {processed_dir}")
        return

    print(f"Found {len(all_files)} .npy files. Processing heuristics...")

    video_heuristics = defaultdict(list)
    
    # We will stream keypoints to a temp CSV to avoid OOM
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "keypoints.csv"
    
    csv_headers = ["sample_id", "class_name", "frame_number", "hand"] + [f"{axis}{i+1}" for i in range(21) for axis in ['x', 'y', 'z']]
    temp_csv_file = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', dir=out_dir)
    temp_csv_path = Path(temp_csv_file.name)
    csv_writer = csv.writer(temp_csv_file)
    csv_writer.writerow(csv_headers)
    
    # If class_only is provided, copy all old rows except for this class
    if class_only and csv_path.exists():
        try:
            with open(csv_path, 'r', newline='') as f_in:
                reader = csv.reader(f_in)
                header = next(reader, None)
                if header:
                    # find class_name index
                    try:
                        class_idx = header.index("class_name")
                        for row in reader:
                            if len(row) > class_idx and row[class_idx] != class_only:
                                csv_writer.writerow(row)
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Warning: Could not copy old CSV entries: {e}")
            
    # For percentiles
    all_movements = []
    all_speeds = []
    
    # Pass 1: Extract features and collect stats for percentiles
    sample_data_list = []

    for fpath in all_files:
        class_name = fpath.parent.name
        if '.' in class_name:
            class_name = class_name.split('.', 1)[1].strip()
            
        sample_id = fpath.stem
        try:
            tensor = np.load(fpath)
        except Exception:
            continue
            
        if len(tensor.shape) != 2:
            continue
            
        num_frames = tensor.shape[0]
        l_hand_present = 0
        r_hand_present = 0
        
        l_movements = []
        r_movements = []
        l_vels = []
        r_vels = []
        l_locs = []
        r_locs = []
        l_orients = []
        r_orients = []
        l_fstates = []
        r_fstates = []
        l_hshapes = []
        r_hshapes = []
        l_traj = []
        r_traj = []
        
        prev_l_centroid = None
        prev_r_centroid = None

        for frame_idx in range(num_frames):
            frame = tensor[frame_idx]
            
            l_raw = frame[LANDMARK_SCHEMA["left_raw"][0] : LANDMARK_SCHEMA["left_raw"][1]]
            r_raw = frame[LANDMARK_SCHEMA["right_raw"][0] : LANDMARK_SCHEMA["right_raw"][1]]
            l_rel = frame[LANDMARK_SCHEMA["left_relative"][0] : LANDMARK_SCHEMA["left_relative"][1]]
            r_rel = frame[LANDMARK_SCHEMA["right_relative"][0] : LANDMARK_SCHEMA["right_relative"][1]]
            
            l_present = np.abs(l_raw).sum() > 1e-6
            r_present = np.abs(r_raw).sum() > 1e-6
            
            if l_present:
                l_hand_present += 1
                l_reshaped = l_raw.reshape(21, 3)
                centroid = l_reshaped.mean(axis=0)
                l_traj.append(centroid)
                if prev_l_centroid is not None:
                    vel = centroid - prev_l_centroid
                    l_vels.append(vel)
                    l_movements.append(np.linalg.norm(vel))
                prev_l_centroid = centroid
                
                # Location
                l_rel_reshaped = l_rel.reshape(21, 3)
                centroid_rel_y = l_rel_reshaped.mean(axis=0)[1]
                if centroid_rel_y > 1.0:
                    l_locs.append("neutral_space")
                elif centroid_rel_y > -0.2:
                    l_locs.append("upper_space")
                else:
                    l_locs.append("face_near")
                    
                # Orientation
                l_orients.append(get_orientation(l_reshaped, "left"))
                
                # Finger state & shape
                fstate = get_finger_state(l_reshaped)
                l_fstates.append(fstate)
                l_hshapes.append(get_hand_shape(l_reshaped, fstate))
                
                row = [sample_id, class_name, frame_idx, "left"] + l_raw.tolist()
                csv_writer.writerow(row)
                
            if r_present:
                r_hand_present += 1
                r_reshaped = r_raw.reshape(21, 3)
                centroid = r_reshaped.mean(axis=0)
                r_traj.append(centroid)
                if prev_r_centroid is not None:
                    vel = centroid - prev_r_centroid
                    r_vels.append(vel)
                    r_movements.append(np.linalg.norm(vel))
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
                    
                r_orients.append(get_orientation(r_reshaped, "right"))
                fstate = get_finger_state(r_reshaped)
                r_fstates.append(fstate)
                r_hshapes.append(get_hand_shape(r_reshaped, fstate))
                
                row = [sample_id, class_name, frame_idx, "right"] + r_raw.tolist()
                csv_writer.writerow(row)
                
        # Flush regularly
        if len(sample_data_list) % 500 == 0:
            temp_csv_file.flush()

        presence_threshold = num_frames * 0.3
        two_hands = (l_hand_present > presence_threshold) and (r_hand_present > presence_threshold)
        
        if two_hands:
            hand_count = "two_hands"
            dom = "both"
            mov_list = l_movements + r_movements
            vel_list = [np.linalg.norm(v) for v in l_vels] + [np.linalg.norm(v) for v in r_vels]
        elif l_hand_present > presence_threshold:
            hand_count = "one_hand"
            dom = "left"
            mov_list = l_movements
            vel_list = [np.linalg.norm(v) for v in l_vels]
        elif r_hand_present > presence_threshold:
            hand_count = "one_hand"
            dom = "right"
            mov_list = r_movements
            vel_list = [np.linalg.norm(v) for v in r_vels]
        else:
            hand_count = "unknown"
            dom = "unknown"
            mov_list = []
            vel_list = []
            
        avg_mov = float(np.mean(mov_list)) if mov_list else 0.0
        avg_speed = float(np.mean(vel_list)) if vel_list else 0.0
        
        if hand_count != "unknown":
            all_movements.append(avg_mov)
            all_speeds.append(avg_speed)
            
        # Get majority votes for frame-wise features
        def get_majority(lst, conf_thresh=0.0):
            if not lst: return "unknown"
            counts = pd.Series(lst).value_counts()
            best = counts.index[0]
            if counts.iloc[0] / len(lst) >= conf_thresh:
                return best
            return "unknown"

        location = get_majority(l_locs + r_locs)
        orientation = get_majority(l_orients + r_orients, conf_thresh=0.7)
        finger_state = get_majority(l_fstates + r_fstates)
        hand_shape = get_majority(l_hshapes + r_hshapes)
        
        # Motion direction & Trajectory pattern
        if dom == "both":
            md = get_motion_direction(r_traj) # default to right hand for motion direction if both
            tp = get_trajectory_pattern(r_traj)
        elif dom == "left":
            md = get_motion_direction(l_traj)
            tp = get_trajectory_pattern(l_traj)
        elif dom == "right":
            md = get_motion_direction(r_traj)
            tp = get_trajectory_pattern(r_traj)
        else:
            md = "unknown"
            tp = "unknown"
            
        if tp == "circular":
            md = "circular"
        elif tp == "static":
            md = "static"
            
        # Symmetry
        symmetry = "unknown"
        if dom == "both" and l_vels and r_vels:
            min_len = min(len(l_vels), len(r_vels))
            if min_len >= 2:
                sims = []
                for i in range(min_len):
                    ln = np.linalg.norm(l_vels[i])
                    rn = np.linalg.norm(r_vels[i])
                    if ln > 1e-5 and rn > 1e-5:
                        sims.append(np.abs(np.dot(l_vels[i], r_vels[i]) / (ln * rn)))
                if sims and np.mean(sims) > 0.6:
                    symmetry = "symmetric"
                else:
                    symmetry = "asymmetric"

        sample_data_list.append({
            "class_name": class_name,
            "hand_count": hand_count,
            "avg_mov": avg_mov,
            "avg_speed": avg_speed,
            "location": location,
            "orientation": orientation,
            "symmetry": symmetry,
            "finger_state": finger_state,
            "hand_shape": hand_shape,
            "motion_direction": md,
            "trajectory_pattern": tp,
            "dominant_hand": dom
        })

    # Compute percentiles for movement and speed
    if all_movements:
        p_mov_33 = np.percentile(all_movements, 33)
        p_mov_66 = np.percentile(all_movements, 66)
    else:
        p_mov_33, p_mov_66 = 0.01, 0.03
        
    if all_speeds:
        p_speed_33 = np.percentile(all_speeds, 33)
        p_speed_66 = np.percentile(all_speeds, 66)
    else:
        p_speed_33, p_speed_66 = 0.01, 0.03

    # Pass 2: Assign threshold-based features and aggregate
    for d in sample_data_list:
        if d["hand_count"] == "unknown":
            d["movement"] = "unknown"
            d["motion_speed"] = "unknown"
        else:
            if d["avg_mov"] < 1e-4: d["movement"] = "static_hold"
            elif d["avg_mov"] < p_mov_33: d["movement"] = "small_motion"
            elif d["avg_mov"] < p_mov_66: d["movement"] = "medium_motion"
            else: d["movement"] = "large_motion"
            
            if d["avg_speed"] < 1e-4: d["motion_speed"] = "static"
            elif d["avg_speed"] < p_speed_33: d["motion_speed"] = "slow"
            elif d["avg_speed"] < p_speed_66: d["motion_speed"] = "medium"
            else: d["motion_speed"] = "fast"
            
        video_heuristics[d["class_name"]].append(d)

    final_classification = {
        "description": "ISL classification generated from dataset observations",
        "signs": {}
    }
    confidence_stats = {}
    
    features = [
        "hand_count", "movement", "location", "orientation", "symmetry", 
        "finger_state", "hand_shape", "motion_direction", "trajectory_pattern", 
        "dominant_hand", "motion_speed"
    ]
    
    candidate_map_raw = defaultdict(list)

    for cls_name, heuristics in video_heuristics.items():
        n = len(heuristics)
        if n == 0: continue
            
        majority = {}
        confidence = {}
        
        for prop in features:
            vals = [h[prop] for h in heuristics]
            counts = pd.Series(vals).value_counts()
            best_val = counts.index[0]
            conf = counts.iloc[0] / n
            
            if conf < 0.5: # Majority vote
                best_val = "unknown"
                
            majority[prop] = best_val
            confidence[f"{prop}_agreement"] = f"{conf*100:.1f}%"
            
        final_classification["signs"][cls_name] = majority
        confidence_stats[cls_name] = confidence
        
        # Build candidate map string: one_hand|small_motion|face_near|pointing
        cmap_key = f"{majority['hand_count']}|{majority['movement']}|{majority['location']}|{majority['hand_shape']}"
        candidate_map_raw[cmap_key].append(cls_name)

    candidate_map = dict(candidate_map_raw)
        
    hsc_path = out_dir / "hand_sign_classification.json"
    cs_path = out_dir / "confidence_statistics.json"
    cm_path = out_dir / "candidate_map.json"
    
    # Update logic for appending new class info
    if class_only:
        if hsc_path.exists():
            with open(hsc_path, "r") as f:
                existing_hsc = json.load(f)
            existing_hsc["signs"].update(final_classification["signs"])
            final_classification = existing_hsc
            
        if cs_path.exists():
            with open(cs_path, "r") as f:
                existing_cs = json.load(f)
            existing_cs.update(confidence_stats)
            confidence_stats = existing_cs
            
        # Candidate map needs to be fully rebuilt from the merged final_classification
        new_cmap_raw = defaultdict(list)
        for cname, props in final_classification["signs"].items():
            cmap_key = f"{props['hand_count']}|{props['movement']}|{props['location']}|{props['hand_shape']}"
            new_cmap_raw[cmap_key].append(cname)
        candidate_map = dict(new_cmap_raw)

    with open(hsc_path, "w") as f:
        json.dump(final_classification, f, indent=2)
        
    with open(cs_path, "w") as f:
        json.dump(confidence_stats, f, indent=2)
        
    with open(cm_path, "w") as f:
        json.dump(candidate_map, f, indent=2)
        
    # Close and swap CSV
    temp_csv_file.close()
    try:
        if csv_path.exists():
            csv_path.unlink()
        temp_csv_path.rename(csv_path)
    except Exception as e:
        print(f"Warning: Could not replace {csv_path} with {temp_csv_path}: {e}")
    
    print("Done! Generated:")
    print(" - data/hand_sign_classification.json")
    print(" - data/confidence_statistics.json")
    print(" - data/candidate_map.json")
    print(" - data/keypoints.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset heuristics and classifications")
    parser.add_argument("--class", dest="class_only", default=None, help="Only process and append info for this specific class")
    args = parser.parse_args()
    
    generate_heuristics(class_only=args.class_only)
