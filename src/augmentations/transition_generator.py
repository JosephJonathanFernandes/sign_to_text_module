import random
import numpy as np

def create_transition_sample(signA, signB):
    """
    Generate synthetic transition examples for reject class.
    Build transition sample using: last 10 frames of signA + first 10 frames of signB.
    Also adds an interpolation/crossfade over 5 frames in the middle.

    Args:
        signA: np.ndarray (20, feature_dim)
        signB: np.ndarray (20, feature_dim)

    Returns:
        np.ndarray: synthetic transition sequence of shape (20, feature_dim)
    """
    # Ensure sequences have at least 10 frames
    if signA.shape[0] < 10 or signB.shape[0] < 10:
        # Fallback to simple concatenation if sequences are too short
        min_len = min(signA.shape[0], signB.shape[0]) // 2
        partA = signA[-min_len:]
        partB = signB[:min_len]
        return np.concatenate([partA, partB], axis=0)

    partA = signA[-10:].copy()
    partB = signB[:10].copy()
    
    transition = np.concatenate([partA, partB], axis=0)
    
    # Optional interpolation/crossfade over 5 frames at the boundary
    # The boundary is at index 10 (frames 9 and 10)
    # We will blend frames 8, 9, 10, 11, 12 using a linear alpha
    alpha = np.linspace(0, 1, 5).reshape(-1, 1)
    
    # Take 5 frames from A and 5 frames from B for the blending region
    fade_A = signA[-5:]
    fade_B = signB[:5]
    
    blended = (1 - alpha) * fade_A + alpha * fade_B
    transition[8:13] = blended
    
    return transition

def generate_transition_dataset(dataset):
    """
    Randomly select pairs from different classes, generate multiple transition samples,
    and assign them to the 'reject' class.

    Args:
        dataset: The ISLDataset containing training samples

    Returns:
        List of tuples: [(transition_sequence, reject_label_index), ...]
    """
    # Import from the newly created config
    from src.config.continuous_signing import TRANSITION_SAMPLES_PER_CLASS
    
    num_classes = dataset.num_classes
    reject_idx = dataset.class_to_idx.get("__reject__", num_classes)
    transitions = []
    
    use_hdf5 = getattr(dataset, 'use_hdf5', False)
    
    if use_hdf5:
        dataset._ensure_open()
        labels = dataset.h5["labels"][:]
        
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
            
        for cls in class_indices.keys():
            other_classes = [c for c in class_indices.keys() if c != cls and c != reject_idx]
            if not other_classes:
                continue
                
            for _ in range(TRANSITION_SAMPLES_PER_CLASS):
                idxA = random.choice(class_indices[cls])
                other_cls = random.choice(other_classes)
                idxB = random.choice(class_indices[other_cls])
                
                try:
                    signA = dataset.h5["features"][idxA].copy().astype(np.float32)
                    signB = dataset.h5["features"][idxB].copy().astype(np.float32)
                    
                    trans_seq = create_transition_sample(signA, signB)
                    transitions.append((trans_seq, reject_idx))
                except Exception:
                    pass
    else:
        samples = dataset.samples
        class_samples = {}
        for fpath, label, _, _ in samples:
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(fpath)
            
        for cls in class_samples.keys():
            other_classes = [c for c in class_samples.keys() if c != cls and c != reject_idx]
            if not other_classes:
                continue
                
            for _ in range(TRANSITION_SAMPLES_PER_CLASS):
                pathA = random.choice(class_samples[cls])
                other_cls = random.choice(other_classes)
                pathB = random.choice(class_samples[other_cls])
                
                try:
                    signA = np.load(pathA).astype(np.float32)
                    signB = np.load(pathB).astype(np.float32)
                    
                    trans_seq = create_transition_sample(signA, signB)
                    transitions.append((trans_seq, reject_idx))
                except Exception:
                    pass
                    
    return transitions
