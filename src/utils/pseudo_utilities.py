"""
Utility functions for pseudo-label collection and adapter training.

Helper functions for working with pseudo-labeled data and adapter models.
"""

import os
import numpy as np
from collections import defaultdict


def load_pseudo_data(pseudo_data_dir: str = "pseudo_data/") -> dict:
    """
    Load all pseudo-labeled data from disk.
    
    Args:
        pseudo_data_dir: Directory containing pseudo-labeled samples
    
    Returns:
        {
            'class_name': {
                'sequences': [seq1, seq2, ...],
                'metadata': {'count': int, 'files': [str, ...]},
            },
            ...
        }
    """
    data = defaultdict(lambda: {'sequences': [], 'metadata': {'files': []}})
    
    if not os.path.exists(pseudo_data_dir):
        return dict(data)
    
    for class_dir in os.listdir(pseudo_data_dir):
        class_path = os.path.join(pseudo_data_dir, class_dir)
        
        if not os.path.isdir(class_path):
            continue
        
        for filename in os.listdir(class_path):
            if filename.endswith('.npy'):
                filepath = os.path.join(class_path, filename)
                try:
                    seq = np.load(filepath)
                    data[class_dir]['sequences'].append(seq)
                    data[class_dir]['metadata']['files'].append(filename)
                except Exception as e:
                    print(f"[Error] Could not load {filepath}: {e}")
    
    # Update counts
    for class_name in data:
        data[class_name]['metadata']['count'] = len(data[class_name]['sequences'])
    
    return dict(data)


def print_pseudo_data_summary(pseudo_data_dir: str = "pseudo_data/"):
    """Print summary of pseudo-labeled data on disk."""
    data = load_pseudo_data(pseudo_data_dir)
    
    if not data:
        print(f"[Pseudo] No data found in {pseudo_data_dir}")
        return
    
    total = sum(info['metadata']['count'] for info in data.values())
    
    print(f"\n[Pseudo] Summary: {total} samples in {len(data)} classes")
    for class_name, info in sorted(data.items()):
        count = info['metadata']['count']
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {class_name:20s}: {count:3d} samples ({pct:5.1f}%)")


def get_pseudo_dataset_stats(pseudo_buffer) -> dict:
    """Get statistics about pseudo-buffer."""
    return {
        'total_samples': pseudo_buffer.get_total_samples(),
        'class_counts': pseudo_buffer.get_class_counts(),
        'distribution': pseudo_buffer.get_distribution(),
        'is_imbalanced': pseudo_buffer.check_class_imbalance(),
    }


def prepare_adapter_training_data(
    pseudo_buffer,
    classes: list,
    ensemble_models,
    ensemble_fallback,
    preprocess_fn,
) -> tuple:
    """
    Prepare training data for adapter from pseudo-buffer.
    
    Re-runs ensemble predictions on pseudo-labeled sequences to get
    the exact ensemble outputs used for adapter training.
    
    Args:
        pseudo_buffer: PseudoLabelBuffer instance
        classes: List of class names
        ensemble_models: Main ensemble models
        ensemble_fallback: Fallback ensemble models
        preprocess_fn: Preprocessing function for sequences
    
    Returns:
        (ensemble_probs_list, class_indices_list)
    """
    from src.inference.ensemble import ensemble_predict
    ensemble_probs_list = []
    class_indices_list = []
    
    for class_name, sequences in pseudo_buffer.buffer.items():
        try:
            class_idx = classes.index(class_name)
        except ValueError:
            print(f"[Warning] Class '{class_name}' not in class list")
            continue
        
        for seq in sequences:
            # Preprocess sequence
            seq_processed = preprocess_fn(seq)
            
            # Run ensemble
            pred_idx, conf, probs = ensemble_predict(
                ensemble_models if ensemble_models else ensemble_fallback,
                seq_processed,
                use_tta=False,
            )
            
            ensemble_probs_list.append(probs)
            class_indices_list.append(class_idx)
    
    return ensemble_probs_list, class_indices_list


def merge_pseudo_datasets(
    source_dirs: list,
    target_dir: str = "pseudo_data_merged/",
) -> int:
    """
    Merge multiple pseudo-dataset directories.
    
    Args:
        source_dirs: List of source directories
        target_dir: Target directory for merged data
    
    Returns:
        Total number of samples merged
    """
    os.makedirs(target_dir, exist_ok=True)
    total = 0
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"[Warning] Source dir not found: {source_dir}")
            continue
        
        for class_dir in os.listdir(source_dir):
            source_class_path = os.path.join(source_dir, class_dir)
            target_class_path = os.path.join(target_dir, class_dir)
            
            if not os.path.isdir(source_class_path):
                continue
            
            os.makedirs(target_class_path, exist_ok=True)
            
            for filename in os.listdir(source_class_path):
                if filename.endswith('.npy'):
                    src_file = os.path.join(source_class_path, filename)
                    dst_file = os.path.join(target_class_path, filename)
                    
                    try:
                        seq = np.load(src_file)
                        np.save(dst_file, seq)
                        total += 1
                    except Exception as e:
                        print(f"[Error] Could not copy {src_file}: {e}")
    
    print(f"[Merge] Merged {total} samples to {target_dir}")
    return total


def clean_pseudo_data(
    pseudo_data_dir: str = "pseudo_data/",
    min_sequence_length: int = 10,
    max_sequence_length: int = 300,
) -> int:
    """
    Clean pseudo-data by removing invalid sequences.
    
    Args:
        pseudo_data_dir: Directory with pseudo-labeled data
        min_sequence_length: Minimum frames per sequence
        max_sequence_length: Maximum frames per sequence
    
    Returns:
        Number of files removed
    """
    removed = 0
    
    for class_dir in os.listdir(pseudo_data_dir):
        class_path = os.path.join(pseudo_data_dir, class_dir)
        
        if not os.path.isdir(class_path):
            continue
        
        for filename in os.listdir(class_path):
            if filename.endswith('.npy'):
                filepath = os.path.join(class_path, filename)
                
                try:
                    seq = np.load(filepath)
                    
                    # Check sequence length
                    if seq.shape[0] < min_sequence_length or seq.shape[0] > max_sequence_length:
                        os.remove(filepath)
                        removed += 1
                        print(f"  Removed {filename} (invalid length: {seq.shape[0]})")
                    
                    # Check for NaN/Inf
                    if np.isnan(seq).any() or np.isinf(seq).any():
                        os.remove(filepath)
                        removed += 1
                        print(f"  Removed {filename} (contains NaN/Inf)")
                
                except Exception as e:
                    print(f"  Error checking {filename}: {e}")
    
    print(f"\n[Clean] Removed {removed} invalid sequences")
    return removed


def export_adapter_metrics(adapter_manager, output_file: str = "adapter_metrics.txt"):
    """
    Export adapter training metrics to file.
    
    Args:
        adapter_manager: AdapterTrainingManager instance
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("=== Adapter Training Metrics ===\n\n")
        
        f.write(f"Total trainings: {len(adapter_manager.performance_log)}\n")
        f.write(f"Is training: {adapter_manager.is_training}\n\n")
        
        f.write("Performance Log:\n")
        for entry in adapter_manager.performance_log:
            f.write(f"\n  Timestamp: {entry['timestamp']}\n")
            f.write(f"  Samples: {entry['num_samples']}\n")
            f.write(f"  Epochs: {entry['epochs']}\n")
            f.write(f"  Status: {entry['status']}\n")
            if entry['validation_passed'] is not None:
                f.write(f"  Validation: {'Passed' if entry['validation_passed'] else 'Failed'}\n")
    
    print(f"[Export] Adapter metrics saved to {output_file}")


def export_pseudo_buffer_stats(pseudo_buffer, output_file: str = "pseudo_stats.txt"):
    """
    Export pseudo-buffer statistics to file.
    
    Args:
        pseudo_buffer: PseudoLabelBuffer instance
        output_file: Output file path
    """
    stats = get_pseudo_dataset_stats(pseudo_buffer)
    
    with open(output_file, 'w') as f:
        f.write("=== Pseudo-Label Buffer Statistics ===\n\n")
        
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"Is imbalanced: {stats['is_imbalanced']}\n\n")
        
        f.write("Class Distribution:\n")
        for class_name, info in sorted(stats['distribution'].items()):
            f.write(f"  {class_name:20s}: {info['count']:3d} "
                   f"({info['percentage']:5.1f}%) | "
                   f"Conf: {info['mean_confidence']:.2f}\n")
    
    print(f"[Export] Pseudo-buffer stats saved to {output_file}")
