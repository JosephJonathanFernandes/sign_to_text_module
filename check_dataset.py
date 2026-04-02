"""
Dataset compatibility checker for ISL sign language recognition.

Validates if a dataset is compatible with the current implementation.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from config import (
    NUM_FRAMES, FRAME_FEAT_DIM,
    VIDEO_EXTENSIONS,
    INPUT_SIZE,  # Includes velocity if USE_VELOCITY=True
)


def check_video_dataset(dataset_path: str) -> dict:
    """
    Check if video dataset is compatible.
    
    Expected structure:
        dataset/
            class1/
                video1.mp4
                video2.mp4
            class2/
                video3.mp4
    """
    print(f"\n{'='*70}")
    print(f"Checking Video Dataset: {dataset_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(dataset_path):
        return {'compatible': False, 'error': f"Path not found: {dataset_path}"}
    
    issues = []
    classes = {}
    total_videos = 0
    
    # Scan directory structure
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        videos = []
        for file in os.listdir(class_path):
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                videos.append(file)
        
        if videos:
            classes[class_name] = videos
            total_videos += len(videos)
    
    print(f"✓ Classes found: {len(classes)}")
    for cls_name, vids in sorted(classes.items()):
        print(f"  - {cls_name:20s}: {len(vids):3d} videos")
    
    if total_videos == 0:
        issues.append("No video files found!")
    
    print(f"\n✓ Total videos: {total_videos}")
    
    # Check video properties (sample first video)
    if total_videos > 0:
        print(f"\nSampling first video to check properties...")
        for cls_name, vids in classes.items():
            sample_path = os.path.join(dataset_path, cls_name, vids[0])
            cap = cv2.VideoCapture(sample_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                print(f"✓ Sample: {vids[0]}")
                print(f"  - Resolution: {width}x{height}")
                print(f"  - FPS: {fps:.1f}")
                print(f"  - Frames: {frame_count}")
                
                if frame_count < NUM_FRAMES:
                    issues.append(
                        f"Video has {frame_count} frames, need at least {NUM_FRAMES}"
                    )
                break
    
    compatible = len(issues) == 0
    status = "✅ COMPATIBLE" if compatible else "⚠️ INCOMPATIBLE"
    
    print(f"\n{status}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  ⚠️ {issue}")
    
    return {
        'compatible': compatible,
        'classes': len(classes),
        'total_videos': total_videos,
        'class_details': classes,
        'issues': issues,
    }


def check_processed_dataset(dataset_path: str) -> dict:
    """
    Check if pre-processed .npy dataset is compatible.
    
    Expected structure:
        processed/
            class1/
                sample1.npy
                sample2.npy
            class2/
                sample3.npy
    """
    print(f"\n{'='*70}")
    print(f"Checking Processed .npy Dataset: {dataset_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(dataset_path):
        return {'compatible': False, 'error': f"Path not found: {dataset_path}"}
    
    issues = []
    classes = {}
    total_samples = 0
    
    # Scan directory structure
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        samples = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        
        if samples:
            classes[class_name] = samples
            total_samples += len(samples)
    
    print(f"✓ Classes found: {len(classes)}")
    for cls_name, smps in sorted(classes.items()):
        print(f"  - {cls_name:20s}: {len(smps):3d} samples")
    
    if total_samples == 0:
        issues.append("No .npy files found!")
    
    print(f"\n✓ Total samples: {total_samples}")
    
    # Check sample shape (sample first .npy)
    if total_samples > 0:
        print(f"\nChecking sample shapes...")
        incompatible_shapes = set()
        
        for cls_name, smps in classes.items():
            sample_path = os.path.join(dataset_path, cls_name, smps[0])
            try:
                data = np.load(sample_path)
                shape = data.shape
                print(f"✓ Sample: {smps[0]}")
                print(f"  - Shape: {shape}")
                
                # Check compatibility
                expected_shape = (NUM_FRAMES, INPUT_SIZE)
                if shape != expected_shape:
                    incompatible_shapes.add(f"{shape} (expected {expected_shape})")
                    
            except Exception as e:
                issues.append(f"Error loading {sample_path}: {str(e)}")
        
        if incompatible_shapes:
            for wrong_shape in incompatible_shapes:
                issues.append(f"Incompatible shape: {wrong_shape}")
    
    compatible = len(issues) == 0
    status = "✅ COMPATIBLE" if compatible else "⚠️ INCOMPATIBLE"
    
    print(f"\n{status}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  ⚠️ {issue}")
    
    return {
        'compatible': compatible,
        'classes': len(classes),
        'total_samples': total_samples,
        'class_details': classes,
        'issues': issues,
    }


def check_csv_landmarks(dataset_path: str) -> dict:
    """
    Check if raw landmarks CSV format is compatible.
    
    Expected format per line:
        x1,y1,z1,...,x21,y1,z21 (right hand) + x1,y1,z1,...,x21,y1,z21 (left hand)
        = 126 coordinates total (can add face-relative + velocity after)
    """
    print(f"\n{'='*70}")
    print(f"Checking CSV Landmarks Dataset: {dataset_path}")
    print(f"{'='*70}")
    
    if not os.path.isfile(dataset_path):
        return {'compatible': False, 'error': f"File not found: {dataset_path}"}
    
    issues = []
    
    # Sample first few lines
    try:
        with open(dataset_path, 'r') as f:
            sample_lines = [next(f) for _ in range(min(5, 100))]
        
        print(f"✓ File readable")
        print(f"✓ Sample lines: {len(sample_lines)}")
        
        coords_count = None
        for idx, line in enumerate(sample_lines[:1]):
            values = line.strip().split(',')
            coords_count = len(values)
            print(f"  Line {idx}: {coords_count} values")
            
            # Check if compatible with raw hands (126 coords)
            if coords_count < 126:
                issues.append(
                    f"Line has {coords_count} values, need at least 126 (for 2 hands)"
                )
        
        compatible = len(issues) == 0
        status = "✅ COMPATIBLE" if compatible else "⚠️ INCOMPATIBLE"
        
        print(f"\n{status}")
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  ⚠️ {issue}")
        
        return {
            'compatible': compatible,
            'sample_lines': len(sample_lines),
            'coords_per_line': coords_count,
            'issues': issues,
        }
        
    except Exception as e:
        return {
            'compatible': False,
            'error': f"Error reading file: {str(e)}"
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python check_dataset.py <path> [type]")
        print("  type: 'video', 'processed', or 'csv' (auto-detected if omitted)")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    dataset_type = sys.argv[2].lower() if len(sys.argv) > 2 else None
    
    # Auto-detect type
    if dataset_type is None:
        if dataset_path.endswith('.csv'):
            dataset_type = 'csv'
        elif os.path.isdir(dataset_path):
            # Check if it has .npy files
            has_npy = any(
                f.endswith('.npy') 
                for root, dirs, files in os.walk(dataset_path)
                for f in files
            )
            dataset_type = 'processed' if has_npy else 'video'
    
    print(f"Detecting dataset type: {dataset_type}")
    
    if dataset_type == 'video':
        result = check_video_dataset(dataset_path)
    elif dataset_type == 'processed':
        result = check_processed_dataset(dataset_path)
    elif dataset_type == 'csv':
        result = check_csv_landmarks(dataset_path)
    else:
        print(f"Unknown type: {dataset_type}")
        sys.exit(1)
    
    print(f"\nFinal result: {result}\n")
