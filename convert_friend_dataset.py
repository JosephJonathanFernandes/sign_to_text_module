"""
Convert friend's dataset format to your implementation format.

Friend's format: (40-63 frames, 1612 features)
Your format:    (20 frames, 506 features)

This converter:
1. Resamples frames to 20 (uniform sampling)
2. Extracts/transforms features to 506 dimensions
3. Saves compatible .npy files
"""

import os
import numpy as np
from pathlib import Path
import sys

SOURCE_DIR = "friend_dataset"
OUTPUT_DIR = "friend_dataset_converted"
TARGET_FRAMES = 20
TARGET_FEATURES = 506


def resample_frames(data: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Resample variable-length sequences to target frame count.
    Uses linear interpolation.
    
    Args:
        data: Shape (N_frames, N_features)
        target_frames: Target number of frames
    
    Returns:
        Resampled data: Shape (target_frames, N_features)
    """
    n_frames = data.shape[0]
    n_features = data.shape[1]
    
    # Create index mapping
    old_indices = np.linspace(0, n_frames - 1, n_frames)
    new_indices = np.linspace(0, n_frames - 1, target_frames)
    
    # Interpolate each feature
    resampled = np.zeros((target_frames, n_features), dtype=np.float32)
    for feat_idx in range(n_features):
        resampled[:, feat_idx] = np.interp(
            new_indices, 
            old_indices, 
            data[:, feat_idx]
        )
    
    return resampled


def transform_features(data: np.ndarray, target_features: int) -> np.ndarray:
    """
    Transform feature dimension to target size.
    
    Strategies (in order):
    1. If target < current: PCA or simple truncation
    2. If target > current: Pad with zeros or repeat
    3. If similar: Apply learned transformation matrix
    
    For now: Use simple truncation/padding with scaling
    """
    current_features = data.shape[1]
    n_frames = data.shape[0]
    
    if current_features == target_features:
        return data
    
    elif current_features > target_features:
        # Truncate - keep first target_features
        print(f"  Truncating {current_features} -> {target_features} features")
        return data[:, :target_features].astype(np.float32)
    
    else:
        # Pad with zeros
        print(f"  Padding {current_features} -> {target_features} features")
        padded = np.zeros((n_frames, target_features), dtype=np.float32)
        padded[:, :current_features] = data
        return padded


def convert_dataset():
    """Convert entire friend dataset to compatible format."""
    
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        return False
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Converting Friend's Dataset")
    print(f"{'='*70}")
    print(f"Source:  {SOURCE_DIR}")
    print(f"Target:  {OUTPUT_DIR}")
    print(f"Frames:  variable → {TARGET_FRAMES}")
    print(f"Features: 1612 → {TARGET_FEATURES}")
    print(f"{'='*70}\n")
    
    total_converted = 0
    total_errors = 0
    
    # Process each class
    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        output_class_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        
        print(f"📁 {class_name}")
        
        # Process each sample
        samples = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        for sample_file in samples:
            try:
                sample_path = os.path.join(class_path, sample_file)
                
                # Load
                data = np.load(sample_path)
                orig_shape = data.shape
                
                # Transform
                # 1. Resample frames
                resampled = resample_frames(data, TARGET_FRAMES)
                
                # 2. Transform features
                transformed = transform_features(resampled, TARGET_FEATURES)
                
                # Save
                output_path = os.path.join(output_class_path, sample_file)
                np.save(output_path, transformed)
                
                print(f"  ✓ {sample_file}: {orig_shape} → {transformed.shape}")
                total_converted += 1
                
            except Exception as e:
                print(f"  ❌ {sample_file}: {str(e)}")
                total_errors += 1
    
    print(f"\n{'='*70}")
    print(f"✅ Conversion Complete")
    print(f"  Converted: {total_converted}")
    print(f"  Errors:    {total_errors}")
    print(f"  Output:    {OUTPUT_DIR}/")
    print(f"{'='*70}\n")
    
    if total_errors == 0:
        print("✨ All samples converted successfully!")
        print(f"\nNext steps:")
        print(f"1. Backup original: Dataset_original -> Dataset_backup")
        print(f"2. Replace: move friend_dataset_converted/* into Dataset/")
        print(f"3. Run preprocessing or training\n")
        return True
    
    return False


if __name__ == "__main__":
    success = convert_dataset()
    sys.exit(0 if success else 1)
