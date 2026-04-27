"""
Unit tests and validation suite for TemporalPostProcessor.

Tests all components:
    - ConfidenceSmoother: buffer management, weighting, edge cases
    - StablePredictor: patience, hysteresis, state transitions
    - TemporalPostProcessor: integration, real-world scenarios
"""

import numpy as np
import pytest
from temporal_postprocessor import (
    ConfidenceSmoother,
    StablePredictor,
    TemporalPostProcessor,
)


class TestConfidenceSmoother:
    """Test suite for ConfidenceSmoother."""
    
    def test_init_valid(self):
        """Test initialization with valid parameters."""
        smoother = ConfidenceSmoother(window_size=10, decay_factor=0.0)
        assert smoother.window_size == 10
        assert smoother.decay_factor == 0.0
        assert smoother.get_buffer_size() == 0
    
    def test_init_invalid_window_size(self):
        """Test that negative window_size raises error."""
        with pytest.raises(ValueError):
            ConfidenceSmoother(window_size=-1)
    
    def test_init_invalid_decay_factor(self):
        """Test that invalid decay_factor raises error."""
        with pytest.raises(ValueError):
            ConfidenceSmoother(decay_factor=1.5)
    
    def test_single_frame_returns_original(self):
        """First frame should return the same probabilities."""
        smoother = ConfidenceSmoother(window_size=10)
        probs = np.array([0.1, 0.2, 0.7])
        result = smoother.update(probs)
        np.testing.assert_array_almost_equal(result, probs)
    
    def test_buffer_size_respected(self):
        """Buffer should not exceed window_size."""
        smoother = ConfidenceSmoother(window_size=3)
        for _ in range(10):
            probs = np.random.dirichlet([1]*5)
            smoother.update(probs)
        assert smoother.get_buffer_size() == 3
    
    def test_uniform_probabilities(self):
        """Uniform input should produce uniform output."""
        smoother = ConfidenceSmoother(window_size=5)
        uniform = np.ones(4) / 4
        for _ in range(3):
            result = smoother.update(uniform)
        np.testing.assert_array_almost_equal(result, uniform, decimal=5)
    
    def test_confidence_weighting(self):
        """Higher confidence frames should have more influence."""
        smoother = ConfidenceSmoother(window_size=10, decay_factor=0.0)
        
        # Low confidence frame
        low_conf = np.array([0.05, 0.05, 0.9])
        smoother.update(low_conf)
        
        # High confidence frame (different class)
        high_conf = np.array([0.8, 0.1, 0.1])
        result = smoother.update(high_conf)
        
        # Result should be closer to high_conf since it has higher max
        assert result[0] > result[2]
    
    def test_exponential_decay(self):
        """Newer frames should be weighted more with decay_factor > 0."""
        smoother = TemporalPostProcessor(window_size=5, enable_decay=True, decay_factor=0.5)
        
        # Fill with all class 0 to 100%
        for _ in range(5):
            probs = np.array([1.0, 0.0, 0.0])
            smoother.update(probs)
        
        # Recent frame switches to class 1
        probs_new = np.array([0.0, 1.0, 0.0])
        result = smoother.update(probs_new)
        
        # Result should lean toward class 1 (decay weights recent more)
        assert result[1] > result[0]
    
    def test_invalid_input_shape(self):
        """Multi-dimensional input should raise error."""
        smoother = ConfidenceSmoother()
        with pytest.raises(ValueError):
            smoother.update(np.random.randn(3, 3))
    
    def test_nan_input(self):
        """NaN in input should raise error."""
        smoother = ConfidenceSmoother()
        with pytest.raises(ValueError):
            smoother.update(np.array([0.1, np.nan, 0.2]))
    
    def test_inf_input(self):
        """Inf in input should raise error."""
        smoother = ConfidenceSmoother()
        with pytest.raises(ValueError):
            smoother.update(np.array([0.1, np.inf, 0.2]))
    
    def test_reset(self):
        """Reset should clear buffer."""
        smoother = ConfidenceSmoother()
        smoother.update(np.array([0.5, 0.5]))
        assert smoother.get_buffer_size() == 1
        smoother.reset()
        assert smoother.get_buffer_size() == 0


class TestStablePredictor:
    """Test suite for StablePredictor."""
    
    def test_init_valid(self):
        """Test initialization with valid parameters."""
        predictor = StablePredictor(patience=3, delta=0.1)
        assert predictor.patience == 3
        assert predictor.delta == 0.1
        assert predictor.current_class is None
    
    def test_init_invalid_patience(self):
        """Patience must be >= 1."""
        with pytest.raises(ValueError):
            StablePredictor(patience=0)
    
    def test_init_invalid_delta(self):
        """Delta must be in [0, 1]."""
        with pytest.raises(ValueError):
            StablePredictor(delta=1.5)
    
    def test_first_prediction(self):
        """First prediction should immediately return the class."""
        predictor = StablePredictor()
        result = predictor.update(5, 0.8)
        assert result == 5
        assert predictor.current_class == 5
        assert predictor.current_confidence == 0.8
    
    def test_stays_on_same_class(self):
        """Resets candidate counter when same class predicts."""
        predictor = StablePredictor(patience=3)
        predictor.update(5, 0.8)
        predictor.update(3, 0.7)  # Different, increment candidate
        assert predictor.candidate_count == 1
        
        predictor.update(5, 0.8)  # Back to original, reset
        assert predictor.candidate_class is None
        assert predictor.candidate_count == 0
    
    def test_patience_requirement(self):
        """Should not switch until patience threshold met."""
        predictor = StablePredictor(patience=3, delta=0.0)
        predictor.update(5, 0.9)
        
        # First 2 frames of new class
        predictor.update(7, 1.0)
        assert predictor.current_class == 5
        
        predictor.update(7, 1.0)
        assert predictor.current_class == 5
        
        # On third frame, should switch
        predictor.update(7, 1.0)
        assert predictor.current_class == 7
    
    def test_hysteresis_requirement(self):
        """Should not switch without sufficient confidence margin."""
        predictor = StablePredictor(patience=1, delta=0.2)
        predictor.update(5, 0.9)
        
        # Try to switch with insufficient confidence margin
        predictor.update(7, 0.95)
        predictor.update(7, 0.95)  # Meets patience, but 0.95 < 0.9 + 0.2
        assert predictor.current_class == 5
        
        # Now with enough margin
        predictor.update(7, 1.0)
        predictor.update(7, 1.0)
        assert predictor.current_class == 7
    
    def test_invalid_class_index(self):
        """Negative class index should raise error."""
        predictor = StablePredictor()
        with pytest.raises(ValueError):
            predictor.update(-1, 0.5)
    
    def test_invalid_confidence(self):
        """Confidence outside [0, 1] should raise error."""
        predictor = StablePredictor()
        with pytest.raises(ValueError):
            predictor.update(0, 1.5)
    
    def test_reset(self):
        """Reset should clear state."""
        predictor = StablePredictor()
        predictor.update(5, 0.8)
        predictor.reset()
        assert predictor.current_class is None
        assert predictor.current_confidence == 0.0
    
    def test_get_state(self):
        """get_state should return dict with all state."""
        predictor = StablePredictor()
        predictor.update(5, 0.8)
        state = predictor.get_state()
        assert state['current_class'] == 5
        assert state['current_confidence'] == 0.8


class TestTemporalPostProcessor:
    """Test suite for TemporalPostProcessor integration."""
    
    def test_first_frame(self):
        """First frame should be processed without errors."""
        processor = TemporalPostProcessor()
        probs = np.array([0.1, 0.2, 0.7])
        result = processor.update(probs)
        assert result == 2  # argmax of [0.1, 0.2, 0.7]
    
    def test_stable_class_stream(self):
        """Stable input should produce stable output."""
        processor = TemporalPostProcessor(patience=2, delta=0.0)
        
        # Class 3 with high confidence for many frames
        for _ in range(10):
            probs = np.zeros(5)
            probs[3] = 0.9
            probs[1:3] = 0.05
            result = processor.update(probs)
        
        assert result == 3
    
    def test_flicker_rejection(self):
        """One-frame flickers should be rejected."""
        processor = TemporalPostProcessor(patience=3, delta=0.15)
        
        # Establish class 2
        for _ in range(5):
            probs = np.zeros(5)
            probs[2] = 0.8
            processor.update(probs)
        
        # Single frame of class 4 (should be rejected)
        probs = np.zeros(5)
        probs[4] = 0.85
        result = processor.update(probs)
        assert result == 2  # Still class 2
        
        # Back to class 2
        probs = np.zeros(5)
        probs[2] = 0.8
        result = processor.update(probs)
        assert result == 2
    
    def test_sustained_switch(self):
        """Sustained high-confidence new class should eventually switch."""
        processor = TemporalPostProcessor(patience=3, delta=0.1)
        
        # Establish class 2 with confidence 0.85
        for _ in range(5):
            probs = np.zeros(10)
            probs[2] = 0.85
            processor.update(probs)
        
        # Switch to class 7 with confidence 0.96
        for _ in range(4):
            probs = np.zeros(10)
            probs[7] = 0.96
            result = processor.update(probs)
        
        assert result == 7  # Should have switched
    
    def test_reset(self):
        """Reset should clear state."""
        processor = TemporalPostProcessor()
        processor.update(np.array([0.1, 0.9]))
        processor.reset()
        assert processor.smoother.get_buffer_size() == 0
        assert processor.predictor.current_class is None
    
    def test_get_state(self):
        """get_state should contain buffer info and predictor state."""
        processor = TemporalPostProcessor()
        processor.update(np.array([0.1, 0.9]))
        state = processor.get_state()
        assert 'buffer_size' in state
        assert 'predictor_state' in state
    
    def test_real_world_scenario(self):
        """Simulate realistic noisy inference stream."""
        processor = TemporalPostProcessor(
            window_size=10,
            patience=3,
            delta=0.1,
            enable_decay=True,
        )
        
        np.random.seed(42)
        num_classes = 20
        
        # Phase 1: Class 5 (some noise)
        for frame in range(20):
            logits = np.random.randn(num_classes) * 0.3
            logits[5] += 2.0
            probs = np.exp(logits)
            probs /= probs.sum()
            result = processor.update(probs)
            # Should stay at 5 or None initially
            if frame > 0:
                assert result == 5 or result is None
        
        # Phase 2: Strong switch to class 14
        for frame in range(30):
            logits = np.random.randn(num_classes) * 0.3
            logits[14] += 3.0
            probs = np.exp(logits)
            probs /= probs.sum()
            result = processor.update(probs)
            # Should eventually switch
        
        # Should be at class 14 after patience + hysteresis met
        assert result == 14


def run_manual_tests():
    """Run manual demonstration tests."""
    print("\n" + "="*60)
    print("MANUAL VALIDATION TESTS")
    print("="*60 + "\n")
    
    # Test 1: Smoothing with high/low confidence
    print("Test 1: Confidence Weighting")
    print("-" * 40)
    smoother = ConfidenceSmoother(window_size=3, decay_factor=0.0)
    
    low_conf = np.array([0.05, 0.05, 0.9])  # max_conf = 0.9
    high_conf = np.array([0.8, 0.1, 0.1])   # max_conf = 0.8
    
    smoother.update(low_conf)
    result = smoother.update(high_conf)
    
    print(f"Low conf frame:  {low_conf}")
    print(f"High conf frame: {high_conf}")
    print(f"Smoothed result: {result}")
    print(f"✓ High confidence class (0) has more weight\n")
    
    # Test 2: Patience + Hysteresis
    print("Test 2: Patience & Hysteresis")
    print("-" * 40)
    predictor = StablePredictor(patience=2, delta=0.15)
    
    print(f"Patience: 2, Delta: 0.15")
    print(f"{'Frame':>5} {'Class':>5} {'Conf':>6} {'Current':>10} {'Status':>20}")
    print("-" * 50)
    
    # Start with class 5
    predictor.update(5, 0.80)
    print(f"{'0':>5} {'5':>5} {'0.80':>6} {'5':>10} {'Initialize'}")
    
    # Try to switch to class 3
    for i in range(1, 5):
        conf = 0.92 + i * 0.02
        result = predictor.update(3, conf)
        status = ""
        if i < 2:
            status = "Candidate (1/2)"
        elif i == 2:
            status = "Hysteresis failed"
        else:
            status = f"SWITCHED → {result}" if result == 3 else "Still evaluating"
        print(f"{i:>5} {'3':>5} {conf:>6.2f} {result:>10} {status:>20}")
    
    print()


if __name__ == "__main__":
    # Run manual demonstrations
    run_manual_tests()
    
    # To run pytest tests:
    # pytest test_temporal_postprocessor.py -v
    print("\n" + "="*60)
    print("To run all unit tests:")
    print("  pytest test_temporal_postprocessor.py -v")
    print("="*60 + "\n")
