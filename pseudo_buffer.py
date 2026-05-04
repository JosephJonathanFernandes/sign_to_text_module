"""
Pseudo-label buffer for collecting high-confidence predictions during live inference.

This module enables safe, continuous collection of pseudo-labeled training samples
without modifying the base ensemble models.

Features:
- High-confidence prediction collection
- Per-class capacity enforcement
- Periodic disk storage
- Class distribution logging
"""

import os
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import json


class PseudoLabelBuffer:
    """Safe buffer for collecting pseudo-labeled sequences."""
    
    def __init__(
        self,
        save_dir: str = "pseudo_data/",
        pseudo_threshold: float = 0.85,
        min_buffer: int = 20,
        per_class_cap: int = 50,
        auto_save: bool = True,
    ):
        """
        Args:
            save_dir: Directory to save pseudo-labeled samples
            pseudo_threshold: Minimum confidence to collect sample
            min_buffer: Minimum samples in buffer before auto-save
            per_class_cap: Maximum samples per class
            auto_save: Whether to auto-save when min_buffer is reached
        """
        self.save_dir = save_dir
        self.pseudo_threshold = pseudo_threshold
        self.min_buffer = min_buffer
        self.per_class_cap = per_class_cap
        self.auto_save = auto_save
        
        # In-memory buffer: {class_name: [seq1, seq2, ...]}
        self.buffer = defaultdict(list)
        
        # Metadata: {class_name: [confidence_values]}
        self.confidences = defaultdict(list)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[PseudoBuffer] Initialized:")
        print(f"  Save dir: {save_dir}")
        print(f"  Threshold: {pseudo_threshold:.0%}")
        print(f"  Per-class cap: {per_class_cap}")
        print(f"  Auto-save at: {min_buffer} samples")
    
    def add_sample(
        self,
        class_name: str,
        seq: np.ndarray,
        confidence: float,
    ) -> bool:
        """
        Add a pseudo-labeled sample to buffer.
        
        Args:
            class_name: Class label (e.g., "Hello", "Thank_you")
            seq: Sequence array (NUM_FRAMES, feat_dim)
            confidence: Prediction confidence (0-1)
        
        Returns:
            True if sample was added, False if rejected
        """
        # Check confidence threshold
        if confidence < self.pseudo_threshold:
            return False
        
        # Check per-class cap
        if len(self.buffer[class_name]) >= self.per_class_cap:
            return False
        
        # Add to buffer
        self.buffer[class_name].append(seq.copy())
        self.confidences[class_name].append(float(confidence))
        
        return True
    
    def get_total_samples(self) -> int:
        """Return total samples across all classes."""
        return sum(len(samples) for samples in self.buffer.values())
    
    def get_class_counts(self) -> dict:
        """Return count per class."""
        return {cls: len(samples) for cls, samples in self.buffer.items()}
    
    def get_distribution(self) -> dict:
        """Return class distribution with statistics."""
        counts = self.get_class_counts()
        total = self.get_total_samples()
        
        distribution = {}
        for cls, count in counts.items():
            mean_conf = (
                np.mean(self.confidences[cls])
                if self.confidences[cls] else 0.0
            )
            distribution[cls] = {
                "count": count,
                "percentage": (count / total * 100) if total > 0 else 0,
                "mean_confidence": float(mean_conf),
            }
        
        return distribution
    
    def print_distribution(self):
        """Print class distribution to console."""
        dist = self.get_distribution()
        total = self.get_total_samples()
        
        print(f"\n[PseudoBuffer] Distribution ({total} total samples):")
        for cls, stats in sorted(dist.items()):
            print(
                f"  {cls:15s}: {stats['count']:3d} samples "
                f"({stats['percentage']:5.1f}%) | "
                f"Conf: {stats['mean_confidence']:.2f}"
            )
    
    def should_save(self) -> bool:
        """Check if buffer should be saved."""
        return self.get_total_samples() >= self.min_buffer
    
    def save(self, verbose: bool = True) -> int:
        """
        Save buffer to disk.
        
        Args:
            verbose: Print save information
        
        Returns:
            Number of samples saved
        """
        if not self.buffer:
            if verbose:
                print("[PseudoBuffer] Buffer empty, nothing to save")
            return 0
        
        total_saved = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for class_name, sequences in self.buffer.items():
            class_dir = os.path.join(self.save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Save each sequence
            for idx, seq in enumerate(sequences):
                filename = f"{class_name}_{timestamp}_{idx:03d}.npy"
                filepath = os.path.join(class_dir, filename)
                np.save(filepath, seq)
                total_saved += 1
        
        if verbose:
            self.print_distribution()
            print(f"[PseudoBuffer] ✓ Saved {total_saved} samples to {self.save_dir}")
        
        return total_saved
    
    def clear(self):
        """Clear buffer (keeping confidences for statistics)."""
        self.buffer.clear()
        # Keep confidences for history
    
    def keep_recent(self, n_per_class: int = 5):
        """
        Keep only recent samples per class (useful after auto-save).
        
        Args:
            n_per_class: Number of recent samples to keep per class
        """
        for class_name in self.buffer:
            if len(self.buffer[class_name]) > n_per_class:
                self.buffer[class_name] = self.buffer[class_name][-n_per_class:]
                self.confidences[class_name] = self.confidences[class_name][-n_per_class:]
    
    def get_all_samples(self) -> list:
        """
        Return all samples as list of dicts.
        Useful for adapter training.
        
        Returns:
            [{
                'class': str,
                'sequence': np.ndarray,
                'confidence': float,
            }, ...]
        """
        samples = []
        
        for class_name, sequences in self.buffer.items():
            for seq, conf in zip(sequences, self.confidences[class_name]):
                samples.append({
                    'class': class_name,
                    'sequence': seq,
                    'confidence': conf,
                })
        
        return samples
    
    def get_buffer_copy(self) -> dict:
        """Return copy of buffer for external use (e.g., adapter training)."""
        return {
            cls: [seq.copy() for seq in sequences]
            for cls, sequences in self.buffer.items()
        }
    
    def check_class_imbalance(self, imbalance_ratio: float = 0.7) -> bool:
        """
        Check if buffer has class imbalance (one class dominates).
        
        Args:
            imbalance_ratio: If any class >= this ratio of total, consider imbalanced
        
        Returns:
            True if imbalanced, False otherwise
        """
        counts = self.get_class_counts()
        total = self.get_total_samples()
        
        if total == 0:
            return False
        
        max_count = max(counts.values())
        if max_count / total >= imbalance_ratio:
            return True
        
        return False
