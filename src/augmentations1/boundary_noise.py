import random
import numpy as np

def apply_boundary_noise(sequence, label, dataset, edge_frames=3, probability=0.5):
    """
    Implement temporal boundary contamination augmentation.
    Randomly replaces the first N and last N frames of a sequence 
    with real landmark frames from a different class to simulate 
    transitioning into/out of a sign.

    Args:
        sequence: np.ndarray of shape (20, feature_dim)
        label: int, the true label of the current sequence
        dataset: The ISLDataset containing other samples
        edge_frames: int, number of frames to contaminate at each boundary
        probability: float, probability of applying the noise

    Returns:
        np.ndarray: The augmented sequence
    """
    if random.random() > probability:
        return sequence

    num_samples = len(dataset.samples)
    if num_samples == 0:
        return sequence

    # Randomly choose a donor sequence from a different class
    max_attempts = 10
    donor_fpath = None
    
    for _ in range(max_attempts):
        idx = random.randint(0, num_samples - 1)
        # sample format is (fpath, label, weight, domain_idx)
        donor_label = dataset.samples[idx][1]
        
        if donor_label != label:
            donor_fpath = dataset.samples[idx][0]
            break
            
    if not donor_fpath:
        return sequence
        
    try:
        # Load donor sequence
        donor_sequence = np.load(donor_fpath).astype(np.float32)
        
        # Ensure donor is properly sized
        donor_frames = donor_sequence.shape[0]
        if donor_frames < edge_frames * 2:
            return sequence
            
        # Copy to avoid modifying original
        augmented = sequence.copy()
        
        # Replace first N frames using donor's first N frames
        augmented[:edge_frames] = donor_sequence[:edge_frames]
        
        # Replace last N frames using donor's last N frames
        augmented[-edge_frames:] = donor_sequence[-edge_frames:]
        
        return augmented
        
    except Exception:
        # If any loading fails, fallback to returning the original sequence
        return sequence
