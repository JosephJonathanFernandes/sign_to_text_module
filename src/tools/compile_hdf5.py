import os
import sys
import json
import hashlib
import time
import numpy as np
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import get_config

def get_dir_hash(directory):
    """Compute a lightweight hash based on file names, sizes, and modification times."""
    hasher = hashlib.md5()
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith('.npy'):
                fpath = os.path.join(root, f)
                stat = os.stat(fpath)
                hasher.update(f"{f}_{stat.st_size}_{stat.st_mtime}".encode('utf-8'))
    return hasher.hexdigest()

def compile_hdf5():
    print("--- HDF5 Dataset Compiler ---")
    cfg = get_config()
    
    root_dir = cfg.paths.processed_dir
    assets_dir = os.path.dirname(root_dir)
    h5_path = os.path.join(assets_dir, "dataset.h5")
    report_path = os.path.join(assets_dir, "validation_report.json")
    
    sequence_length = cfg.frame_features.input_sequence_dim if hasattr(cfg.frame_features, 'input_sequence_dim') else cfg.NUM_FRAMES
    # Wait, the config uses cfg.frame_features.input_sequence_dim for dimension, but sequence length is NUM_FRAMES.
    # Let's be precise: sequence_length is NUM_FRAMES, feature_dimension is INPUT_SIZE.
    sequence_length = getattr(cfg, 'NUM_FRAMES', 20)
    feature_dimension = getattr(cfg, 'INPUT_SIZE', 506)
    
    if hasattr(cfg, 'frame_features'):
        feature_dimension = getattr(cfg.frame_features, 'input_sequence_dim', feature_dimension)
    
    print(f"Target Shape per sample: ({sequence_length}, {feature_dimension})")
    
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} does not exist.")
        return

    # Discover classes
    class_dirs = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    
    class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}
    
    report = {
        "compilation_time": time.time(),
        "total_files_scanned": 0,
        "valid_files": 0,
        "rejected_files": 0,
        "rejection_reasons": {},
        "class_distribution": {},
        "target_shape": [sequence_length, feature_dimension]
    }
    
    features_list = []
    labels_list = []
    domain_list = []
    
    domain_to_idx = {}
    domains_list_str = []
    
    def _get_domain_idx(filename: str) -> int:
        if filename.startswith("webcam_"):
            d_str = "webcam"
        elif filename.startswith("MVI_"):
            d_str = "MVI"
        elif filename.startswith("cvae_"):
            d_str = "cvae"
        else:
            d_str = "unknown"
            
        if d_str not in domain_to_idx:
            domain_to_idx[d_str] = len(domains_list_str)
            domains_list_str.append(d_str)
        return domain_to_idx[d_str]
    
    for cls_name in class_dirs:
        cls_dir = os.path.join(root_dir, cls_name)
        cls_idx = class_to_idx[cls_name]
        valid_count = 0
        
        for fname in os.listdir(cls_dir):
            if not fname.endswith(".npy"):
                continue
                
            report["total_files_scanned"] += 1
            fpath = os.path.join(cls_dir, fname)
            
            try:
                data = np.load(fpath)
                
                # Validation checks
                if data.size == 0:
                    raise ValueError("Empty file")
                
                # Check shape
                if len(data.shape) != 2:
                    raise ValueError(f"Invalid rank: {len(data.shape)} != 2")
                    
                if data.shape[0] != sequence_length:
                    raise ValueError(f"Invalid sequence length: {data.shape[0]} != {sequence_length}")
                    
                # Check features dim. In dataset.py, it pads/truncates. We should enforce it or pad it here.
                if data.shape[1] > feature_dimension:
                    data = data[:, :feature_dimension]
                elif data.shape[1] < feature_dimension:
                    pad = np.zeros((data.shape[0], feature_dimension - data.shape[1]), dtype=np.float32)
                    data = np.concatenate([data, pad], axis=1)
                
                # Check NaNs / Infs
                if np.isnan(data).any():
                    raise ValueError("Contains NaN")
                if np.isinf(data).any():
                    raise ValueError("Contains Inf")
                
                features_list.append(data.astype(np.float32))
                labels_list.append(cls_idx)
                domain_list.append(_get_domain_idx(fname))
                valid_count += 1
                report["valid_files"] += 1
                
            except Exception as e:
                report["rejected_files"] += 1
                reason = str(e)
                report["rejection_reasons"][reason] = report["rejection_reasons"].get(reason, 0) + 1
                
        report["class_distribution"][cls_name] = valid_count
        
    if not features_list:
        print("Error: No valid files found.")
        return
        
    print(f"Scanned {report['total_files_scanned']} files.")
    print(f"Valid: {report['valid_files']}, Rejected: {report['rejected_files']}")
    
    # Compile arrays
    features = np.stack(features_list)
    labels = np.array(labels_list, dtype=np.int32)
    domain_indices = np.array(domain_list, dtype=np.int32)
    weights = np.ones_like(labels, dtype=np.float32)  # Default weights
    
    N = features.shape[0]
    print("Computing dataset hash...")
    dataset_hash = get_dir_hash(root_dir)
    print(f"Dataset hash computed: {dataset_hash}")
    
    print(f"Writing to {h5_path}...")
    with h5py.File(h5_path, 'w') as f:
        print("Creating features dataset...")
        # Create datasets
        f.create_dataset(
            'features',
            data=features,
            dtype='float32',
            compression='lzf',
            chunks=(1, sequence_length, feature_dimension)
        )
        print("Features dataset created.")
        
        print("Creating labels & weights datasets...")
        f.create_dataset('labels', data=labels, dtype='int32')
        f.create_dataset('weights', data=weights, dtype='float32')
        f.create_dataset('domains', data=domain_indices, dtype='int32')
        
        # Serialize class and domain names as JSON string
        f.create_dataset('class_names', data=json.dumps(class_to_idx))
        f.create_dataset('domain_names', data=json.dumps(domain_to_idx))
        
        # Metadata Fingerprinting
        f.attrs['dataset_version'] = "1.0"
        f.attrs['compiler_version'] = "1.0"
        f.attrs['feature_schema_version'] = "1.0"
        f.attrs['creation_date'] = time.time()
        f.attrs['dataset_hash'] = dataset_hash
        f.attrs['sequence_length'] = sequence_length
        f.attrs['feature_dimension'] = feature_dimension
        f.attrs['num_classes'] = len(class_dirs)
        f.attrs['sample_count'] = N
        print("Metadata attributes set.")
        
    print("Writing validation report...")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Successfully compiled {N} samples into {h5_path}")
    print(f"Validation report saved to {report_path}")

if __name__ == "__main__":
    compile_hdf5()
