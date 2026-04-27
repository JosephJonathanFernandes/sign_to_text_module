"""
Temporal Post-Processor for Real-Time Sign Language Recognition Inference.

Implements confidence-weighted smoothing and anti-flicker stabilization to improve
robustness against noisy per-frame predictions from neural networks.

Core Components:
    1. ConfidenceSmoother: Smooths probability vectors using confidence-weighted averaging
    2. StablePredictor: Stabilizes class predictions using patience and hysteresis
    3. TemporalPostProcessor: Integration wrapper combining both components

Key Features:
    - Real-time inference efficiency (deque-based buffer)
    - Handles edge cases (empty buffer, first prediction)
    - Optional exponential decay weighting for older frames
    - Hysteresis to prevent unnecessary class switching
    - Minimal dependencies (numpy, collections.deque only)

Usage Example:
    >>> processor = TemporalPostProcessor(
    ...     window_size=10,
    ...     patience=3,
    ...     delta=0.1,
    ...     enable_decay=True
    ... )
    >>> for frame_probs in inference_stream:
    ...     stable_class = processor.update(frame_probs)
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional


class ConfidenceSmoother:
    """
    Smooths raw model probability vectors using confidence-weighted averaging.
    
    Maintains a fixed-size buffer of recent predictions. Each prediction is weighted
    by its confidence (max probability), resulting in less noisy estimates over time.
    
    Attributes:
        window_size (int): Number of frames to keep in buffer (default: 10)
        decay_factor (float): Exponential decay weight for older frames (0-1).
                             If 0: uniform weighting. Higher values favor recent frames.
        buffer (deque): Stores tuples of (probability_vector, confidence)
    """
    
    def __init__(self, window_size: int = 10, decay_factor: float = 0.0):
        """
        Initialize the confidence smoother.
        
        Args:
            window_size (int): Maximum number of frames to buffer. Default: 10.
            decay_factor (float): Exponential decay for older frames in [0, 1).
                                 0.0 = uniform weighting (default)
                                 0.5 = older frames weighted as 0.5^age
                                 Helps recent frames have more influence.
        """
        if not (0 <= window_size):
            raise ValueError(f"window_size must be non-negative, got {window_size}")
        if not (0 <= decay_factor < 1):
            raise ValueError(f"decay_factor must be in [0, 1), got {decay_factor}")
        
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.buffer = deque(maxlen=window_size)
    
    def update(self, probs: np.ndarray) -> np.ndarray:
        """
        Update buffer with new prediction and return smoothed probabilities.
        
        Args:
            probs (np.ndarray): Probability vector from model, shape (num_classes,).
                               Should sum to ~1.0 (softmax output).
        
        Returns:
            np.ndarray: Smoothed probability vector, same shape as input.
                       Returns original probs if buffer is empty.
        
        Raises:
            ValueError: If probs shape is incompatible or contains NaN/Inf.
        """
        probs = np.asarray(probs, dtype=np.float32)
        
        if probs.ndim != 1:
            raise ValueError(f"probs must be 1D, got shape {probs.shape}")
        if np.isnan(probs).any() or np.isinf(probs).any():
            raise ValueError("probs contains NaN or Inf values")
        
        # Confidence = max probability in this frame
        confidence = np.max(probs)
        
        # Add to buffer
        self.buffer.append((probs.copy(), confidence))
        
        # If buffer is empty or has only one element, return original
        if len(self.buffer) == 0:
            return probs.copy()
        
        # Compute confidence-weighted smoothed probabilities
        smoothed = self._compute_weighted_average()
        
        # Renormalize to ensure it sums to 1.0 (numerical stability)
        smoothed = smoothed / (np.sum(smoothed) + 1e-8)
        
        return smoothed
    
    def _compute_weighted_average(self) -> np.ndarray:
        """
        Compute confidence-weighted average of buffered predictions.
        
        Applies optional exponential decay to older frames:
            weight_i = confidence_i * (decay_factor ^ age_i)
        
        Returns:
            np.ndarray: Weighted average probability vector.
        """
        if len(self.buffer) == 0:
            return np.zeros(1, dtype=np.float32)
        
        total_weight = 0.0
        weighted_sum = None
        
        # Iterate from oldest (index 0) to newest (index -1)
        for age, (probs, conf) in enumerate(self.buffer):
            # Apply exponential decay if enabled
            decay = self.decay_factor ** age
            weight = conf * decay
            total_weight += weight
            
            if weighted_sum is None:
                weighted_sum = probs * weight
            else:
                weighted_sum += probs * weight
        
        if total_weight < 1e-8:
            # Fallback if all confidences are near zero
            return np.mean([p for p, _ in self.buffer], axis=0)
        
        return weighted_sum / total_weight
    
    def reset(self) -> None:
        """Clear the buffer (useful between video clips or resets)."""
        self.buffer.clear()
    
    def get_buffer_size(self) -> int:
        """Return current number of frames in buffer."""
        return len(self.buffer)


class StablePredictor:
    """
    Produces stable class predictions by filtering noisy per-frame predictions.
    
    Uses two mechanisms to reduce flicker/jitter:
        1. Patience: Requires multiple consecutive frames of the same candidate class
        2. Hysteresis: Only switches if new confidence exceeds current by delta threshold
    
    Attributes:
        patience (int): Minimum consecutive frames before switching classes (default: 3)
        delta (float): Confidence margin required for hysteresis (default: 0.1)
        current_class (int): Currently predicted class
        current_confidence (float): Confidence of current prediction (0-1)
        candidate_class (int): Class being evaluated for switch
        candidate_count (int): How many consecutive frames voting for candidate
    """
    
    def __init__(self, patience: int = 3, delta: float = 0.1):
        """
        Initialize the stable predictor.
        
        Args:
            patience (int): Minimum consecutive matching frames to confirm switch.
                          Default: 3. Higher = more stable but slower to adapt.
            delta (float): Confidence margin for hysteresis (0-1).
                          Default: 0.1. Higher = harder to switch classes.
        
        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if not (0 <= delta <= 1):
            raise ValueError(f"delta must be in [0, 1], got {delta}")
        
        self.patience = patience
        self.delta = delta
        
        # State: initialized to None (no prediction yet)
        self.current_class = None
        self.current_confidence = 0.0
        self.candidate_class = None
        self.candidate_count = 0
    
    def update(self, pred_class: int, confidence: float) -> Optional[int]:
        """
        Update predictor with new frame's prediction and return stable class.
        
        Args:
            pred_class (int): Predicted class index (from argmax of probabilities)
            confidence (float): Confidence of that prediction (0-1, typically max prob)
        
        Returns:
            int or None: Stable predicted class, or None if not yet initialized.
        
        Raises:
            ValueError: If confidence not in [0, 1] or pred_class is negative.
        """
        if not isinstance(pred_class, (int, np.integer)) or pred_class < 0:
            raise ValueError(f"pred_class must be non-negative int, got {pred_class}")
        if not (0 <= confidence <= 1):
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        
        pred_class = int(pred_class)
        confidence = float(confidence)
        
        # First prediction ever
        if self.current_class is None:
            self.current_class = pred_class
            self.current_confidence = confidence
            return self.current_class
        
        # If same as current, reset candidate counter
        if pred_class == self.current_class:
            self.candidate_class = None
            self.candidate_count = 0
            return self.current_class
        
        # If same as candidate, increment counter
        if pred_class == self.candidate_class:
            self.candidate_count += 1
        else:
            # New candidate class
            self.candidate_class = pred_class
            self.candidate_count = 1
        
        # Check if we should switch
        if self._should_switch(confidence):
            self.current_class = self.candidate_class
            self.current_confidence = confidence
            self.candidate_class = None
            self.candidate_count = 0
        
        return self.current_class
    
    def _should_switch(self, new_confidence: float) -> bool:
        """
        Determine if we should switch to the candidate class.
        
        Checks:
            1. Patience: candidate_count >= patience
            2. Hysteresis: new_confidence > current_confidence + delta
        
        Args:
            new_confidence (float): Confidence of candidate class
        
        Returns:
            bool: True if both conditions are met.
        """
        # Must have enough consecutive frames
        if self.candidate_count < self.patience:
            return False
        
        # Must exceed hysteresis threshold
        if new_confidence <= self.current_confidence + self.delta:
            return False
        
        return True
    
    def reset(self) -> None:
        """Reset to initial state (useful between video clips)."""
        self.current_class = None
        self.current_confidence = 0.0
        self.candidate_class = None
        self.candidate_count = 0
    
    def get_state(self) -> dict:
        """
        Return current internal state for debugging/logging.
        
        Returns:
            dict: Contains current_class, current_confidence, candidate info, counts.
        """
        return {
            "current_class": self.current_class,
            "current_confidence": self.current_confidence,
            "candidate_class": self.candidate_class,
            "candidate_count": self.candidate_count,
        }


class TemporalPostProcessor:
    """
    Complete temporal post-processing pipeline for real-time inference.
    
    Combines confidence smoothing and stable prediction into a single module
    for streamlined inference. Processes raw model outputs and returns stable
    class predictions.
    
    Workflow:
        1. Input: raw_probs (numpy array from model, shape [num_classes])
        2. Apply confidence-weighted smoothing
        3. Compute argmax → predicted_class and confidence
        4. Pass through stability filter (patience + hysteresis)
        5. Output: stable_class (int)
    
    Attributes:
        smoother (ConfidenceSmoother): Probability smoothing component
        predictor (StablePredictor): Stability filtering component
    """
    
    def __init__(
        self,
        window_size: int = 10,
        patience: int = 3,
        delta: float = 0.1,
        enable_decay: bool = False,
        decay_factor: float = 0.3,
    ):
        """
        Initialize the temporal post-processor.
        
        Args:
            window_size (int): Number of frames for smoothing buffer. Default: 10.
            patience (int): Frames needed to confirm class switch. Default: 3.
            delta (float): Confidence margin for hysteresis. Default: 0.1.
            enable_decay (bool): Use exponential decay for older frames. Default: False.
            decay_factor (float): Decay weight if enable_decay=True. Default: 0.3.
                                 Used only if enable_decay=True.
        """
        decay = decay_factor if enable_decay else 0.0
        
        self.smoother = ConfidenceSmoother(
            window_size=window_size,
            decay_factor=decay
        )
        self.predictor = StablePredictor(patience=patience, delta=delta)
        self._last_smoothed_confidence = 0.0  # Cache for confidence retrieval
    
    def update(self, raw_probs: np.ndarray) -> Optional[int]:
        """
        Process one frame's raw probabilities and return stable prediction.
        
        Step-by-step:
            1. Smooth probabilities using confidence weighting
            2. Extract best class and its confidence via argmax
            3. Feed into stability predictor
            4. Return stabilized class
        
        Args:
            raw_probs (np.ndarray): Model output, shape (num_classes,).
                                   Should be softmax of logits.
        
        Returns:
            int or None: Stable predicted class index, or None if first frame
                        and predictor not yet initialized.
        
        Raises:
            ValueError: If raw_probs has invalid shape or contains NaN/Inf.
        """
        # Step 1: Smooth
        smoothed_probs = self.smoother.update(raw_probs)
        
        # Step 2: Predict
        pred_class = int(np.argmax(smoothed_probs))
        confidence = float(smoothed_probs[pred_class])
        self._last_smoothed_confidence = confidence  # Cache for retrieval
        
        # Step 3: Stabilize
        stable_class = self.predictor.update(pred_class, confidence)
        
        return stable_class
    
    def update_with_confidence(self, raw_probs: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Process one frame's raw probabilities and return both stable class and confidence.
        
        Combines smoothing, class prediction, and stability filtering.
        Use this when you need both the stable class AND its confidence for inference.
        
        Args:
            raw_probs (np.ndarray): Model output, shape (num_classes,).
                                   Should be softmax of logits.
        
        Returns:
            Tuple[Optional[int], float]: (stable_class_idx, confidence)
                - stable_class_idx: Stabilized class index or None if not yet stabilized
                - confidence: Confidence of the smoothed class (0.0-1.0)
        
        Raises:
            ValueError: If raw_probs has invalid shape or contains NaN/Inf.
        """
        # Step 1: Smooth
        smoothed_probs = self.smoother.update(raw_probs)
        
        # Step 2: Predict
        pred_class = int(np.argmax(smoothed_probs))
        confidence = float(smoothed_probs[pred_class])
        self._last_smoothed_confidence = confidence  # Cache for retrieval
        
        # Step 3: Stabilize
        stable_class = self.predictor.update(pred_class, confidence)
        
        return stable_class, confidence
    
    def get_last_confidence(self) -> float:
        """Return the confidence of the last smoothed prediction."""
        return self._last_smoothed_confidence
    
    def reset(self) -> None:
        """Reset both smoother and predictor (useful between video clips)."""
        self.smoother.reset()
        self.predictor.reset()
    
    def get_state(self) -> dict:
        """
        Return complete state for debugging/monitoring.
        
        Returns:
            dict: Includes smoother buffer size, predictor internal state.
        """
        return {
            "buffer_size": self.smoother.get_buffer_size(),
            "predictor_state": self.predictor.get_state(),
        }


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    # Example: simulate inference stream
    np.random.seed(42)
    
    # Create processor with tuned parameters
    processor = TemporalPostProcessor(
        window_size=10,
        patience=3,
        delta=0.1,
        enable_decay=True,
        decay_factor=0.3,
    )
    
    num_classes = 60
    num_frames = 50
    
    print("Temporal Post-Processor Demo")
    print("=" * 60)
    print(f"Classes: {num_classes}, Frames: {num_frames}")
    print(f"Window: 10, Patience: 3, Delta: 0.1, Decay: 0.3\n")
    
    # Simulate noisy predictions (true class with noise)
    true_class = 5
    
    print(f"{'Frame':>5} {'True':>5} {'Raw Pred':>10} {'Smooth':>10} {'Stable':>10}")
    print("-" * 60)
    
    for frame_idx in range(num_frames):
        # Simulate some flicker/noise
        if frame_idx < 10:
            # First 10 frames: true class 5
            true_class = 5
        elif frame_idx < 25:
            # Noisy phase
            true_class = 5 if frame_idx % 3 == 0 else (4 if frame_idx % 3 == 1 else 6)
        else:
            # Switch to true class 10 with some noise
            true_class = 10 if frame_idx % 4 == 0 else (9 if frame_idx % 4 == 1 else 11)
        
        # Create noisy softmax
        logits = np.random.randn(num_classes) * 0.5
        logits[true_class] += 2.0  # Boost true class
        probs = np.exp(logits)
        probs /= probs.sum()
        
        raw_pred = np.argmax(probs)
        raw_conf = probs[raw_pred]
        
        stable_class = processor.update(probs)
        
        if frame_idx % 5 == 0 or frame_idx < 5:
            state = processor.get_state()
            print(
                f"{frame_idx:5d} {true_class:5d} {raw_pred:10d} "
                f"{raw_conf:10.3f} {stable_class:10d}"
            )
    
    print("\nFinal state:", processor.get_state())
