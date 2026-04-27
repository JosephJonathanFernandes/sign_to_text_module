"""
Unit tests for HandSelector module.

Tests all components:
    - Face center computation
    - Hand center computation
    - Distance-based filtering
    - ROI-based filtering
    - Hand selection logic
    - Left/right assignment
    - Edge cases
    - Integration tests
"""

import numpy as np
import pytest
from hand_selector import (
    HandSelector,
    compute_face_center,
    compute_hand_centers,
)


class TestFaceCenterComputation:
    """Tests for face center computation."""
    
    def test_face_center_from_centered_landmarks(self):
        """Face center should be accurate when landmarks are centered."""
        face_landmarks = np.zeros((468, 3))
        
        # Place nose, eyes at known positions
        face_landmarks[1] = [320, 240, 0]  # Nose
        face_landmarks[33] = [300, 240, 0]  # Left eye
        face_landmarks[263] = [340, 240, 0]  # Right eye
        
        center = compute_face_center(face_landmarks)
        
        # Average should be around (320, 240)
        assert abs(center[0] - 320) < 0.1
        assert abs(center[1] - 240) < 0.1
    
    def test_face_center_uses_only_xy(self):
        """Face center should ignore z-coordinate."""
        face_landmarks = np.zeros((468, 3))
        
        face_landmarks[1] = [100, 100, 999]  # High z value
        face_landmarks[33] = [100, 100, 999]
        face_landmarks[263] = [100, 100, 999]
        
        center = compute_face_center(face_landmarks)
        
        # Should be [100, 100], not affected by z
        np.testing.assert_array_almost_equal(center, [100, 100])
    
    def test_face_center_asymmetric_eyes(self):
        """Face center should handle asymmetric eye positions."""
        face_landmarks = np.zeros((468, 3))
        
        face_landmarks[1] = [200, 200, 0]  # Nose
        face_landmarks[33] = [150, 150, 0]  # Left eye (closer)
        face_landmarks[263] = [250, 250, 0]  # Right eye (farther)
        
        center = compute_face_center(face_landmarks)
        
        # Should be average of three points
        expected = np.mean([[200, 200], [150, 150], [250, 250]], axis=0)
        np.testing.assert_array_almost_equal(center, expected)


class TestHandCenterComputation:
    """Tests for hand center computation."""
    
    def test_single_hand_center(self):
        """Hand center should be average of 21 landmarks."""
        hand_landmarks_list = [np.ones((21, 3)) * 50]
        
        centers = compute_hand_centers(hand_landmarks_list)
        
        assert len(centers) == 1
        np.testing.assert_array_almost_equal(centers[0], [50.0, 50.0])
    
    def test_multiple_hands_centers(self):
        """Should compute center for each hand independently."""
        hand_landmarks_list = [
            np.ones((21, 3)) * 100,
            np.ones((21, 3)) * 200,
            np.ones((21, 3)) * 300,
        ]
        
        centers = compute_hand_centers(hand_landmarks_list)
        
        assert len(centers) == 3
        np.testing.assert_array_almost_equal(centers[0], [100.0, 100.0])
        np.testing.assert_array_almost_equal(centers[1], [200.0, 200.0])
        np.testing.assert_array_almost_equal(centers[2], [300.0, 300.0])
    
    def test_hand_center_ignores_z(self):
        """Hand center should use only x, y coordinates."""
        hand_landmarks_list = [np.column_stack([
            np.arange(21) * 10,
            np.arange(21) * 5,
            np.arange(21) * 999,  # z values (ignored)
        ])]
        
        centers = compute_hand_centers(hand_landmarks_list)
        
        # Should be mean of x and y only
        expected_x = np.mean(np.arange(21) * 10)
        expected_y = np.mean(np.arange(21) * 5)
        np.testing.assert_array_almost_equal(centers[0], [expected_x, expected_y])
    
    def test_empty_hand_landmarks(self):
        """Empty hand list should return empty centers."""
        centers = compute_hand_centers([])
        assert len(centers) == 0


class TestDistanceFiltering:
    """Tests for distance-based hand filtering."""
    
    def test_single_hand_within_threshold(self):
        """Hand within threshold should be included."""
        selector = HandSelector(distance_threshold=100.0, use_roi_filtering=False)
        
        face_center = (320, 240)
        hand_centers = [(320, 250)]  # Distance = 10
        
        filtered = selector._filter_hands_by_distance(hand_centers, face_center, 100.0)
        
        assert filtered == [0]
    
    def test_single_hand_outside_threshold(self):
        """Hand outside threshold should be filtered out."""
        selector = HandSelector(distance_threshold=100.0, use_roi_filtering=False)
        
        face_center = (320, 240)
        hand_centers = [(320, 400)]  # Distance = 160
        
        filtered = selector._filter_hands_by_distance(hand_centers, face_center, 100.0)
        
        assert filtered == []
    
    def test_multiple_hands_mixed(self):
        """Should filter some hands and keep others."""
        selector = HandSelector(distance_threshold=100.0, use_roi_filtering=False)
        
        face_center = (320, 240)
        hand_centers = [
            (320, 250),  # Distance = 10 (inside)
            (320, 300),  # Distance = 60 (inside)
            (320, 400),  # Distance = 160 (outside)
            (420, 240),  # Distance = 100 (on boundary)
        ]
        
        filtered = selector._filter_hands_by_distance(hand_centers, face_center, 100.0)
        
        assert sorted(filtered) == [0, 1, 3]


class TestROIFiltering:
    """Tests for ROI-based hand filtering."""
    
    def test_hand_inside_roi(self):
        """Hand inside ROI should be included."""
        selector = HandSelector(
            roi_width_ratio=0.5,
            roi_height_ratio=0.5,
            use_roi_filtering=True,
        )
        
        face_center = (320, 240)
        hand_centers = [(320, 240)]  # At face center
        frame_shape = (480, 640)
        
        filtered = selector._filter_hands_by_roi(hand_centers, face_center, frame_shape)
        
        assert filtered == [0]
    
    def test_hand_outside_roi(self):
        """Hand outside ROI should be filtered out."""
        selector = HandSelector(
            roi_width_ratio=0.5,
            roi_height_ratio=0.5,
            use_roi_filtering=True,
        )
        
        face_center = (320, 240)
        # Hand far outside ROI
        hand_centers = [(10, 10)]
        frame_shape = (480, 640)
        
        filtered = selector._filter_hands_by_roi(hand_centers, face_center, frame_shape)
        
        assert filtered == []
    
    def test_roi_boundaries(self):
        """Test roi at exact boundaries."""
        selector = HandSelector(
            roi_width_ratio=0.5,
            roi_height_ratio=0.5,
            use_roi_filtering=True,
        )
        
        face_center = (320, 240)
        frame_shape = (480, 640)
        
        # ROI bounds:
        # x: 320 ± 320*0.5/2 = 320 ± 80 → [240, 400]
        # y: 240 ± 240*0.5/2 = 240 ± 60 → [180, 300]
        
        hand_centers = [
            (240, 240),  # Left boundary
            (400, 240),  # Right boundary
            (320, 180),  # Top boundary
            (320, 300),  # Bottom boundary
            (239, 240),  # Just outside left
            (401, 240),  # Just outside right
        ]
        
        filtered = selector._filter_hands_by_roi(hand_centers, face_center, frame_shape)
        
        # First 4 should be inside, last 2 outside
        assert sorted(filtered) == [0, 1, 2, 3]
    
    def test_multiple_roi_ratios(self):
        """ROI size should scale with ratio parameters."""
        selector1 = HandSelector(roi_width_ratio=0.5, roi_height_ratio=0.5)
        selector2 = HandSelector(roi_width_ratio=0.3, roi_height_ratio=0.3)
        
        face_center = (320, 240)
        frame_shape = (480, 640)
        hand_centers = [(380, 240)]  # 60 pixels to right
        
        filtered1 = selector1._filter_hands_by_roi(hand_centers, face_center, frame_shape)
        filtered2 = selector2._filter_hands_by_roi(hand_centers, face_center, frame_shape)
        
        # Smaller ROI should exclude the hand
        assert 0 in filtered1
        assert 0 not in filtered2


class TestHandSelection:
    """Tests for selecting closest hands."""
    
    def test_select_all_two_hands(self):
        """With 2 hands, both should be selected."""
        selector = HandSelector()
        
        face_center = (320, 240)
        hand_centers = [(200, 240), (440, 240)]
        hand_indices = [0, 1]
        
        selected = selector._select_hands(hand_centers, hand_indices, face_center, max_hands=2)
        
        assert len(selected) == 2
        assert 0 in selected
        assert 1 in selected
    
    def test_select_single_hand(self):
        """With 1 hand, that hand should be selected."""
        selector = HandSelector()
        
        face_center = (320, 240)
        hand_centers = [(320, 240)]
        hand_indices = [0]
        
        selected = selector._select_hands(hand_centers, hand_indices, face_center, max_hands=2)
        
        assert selected == [0]
    
    def test_select_closest_of_three(self):
        """With 3 hands, 2 closest should be selected."""
        selector = HandSelector()
        
        face_center = (320, 240)
        hand_centers = [
            (320, 250),  # Distance = 10 (closest)
            (400, 300),  # Distance ≈ 94 (second closest)
            (100, 100),  # Distance ≈ 311 (farthest)
        ]
        hand_indices = [0, 1, 2]
        
        selected = selector._select_hands(hand_centers, hand_indices, face_center, max_hands=2)
        
        assert len(selected) == 2
        assert 0 in selected
        assert 1 in selected


class TestHandSideAssignment:
    """Tests for assigning left/right to hands."""
    
    def test_single_left_hand(self):
        """Hand on left side should be labeled left."""
        selector = HandSelector()
        
        face_center = (320, 240)
        hand_centers = [(200, 240)]
        hand_landmarks = [np.ones((21, 3))]
        
        result = selector._assign_hand_sides(
            hand_landmarks, hand_centers, [0], face_center
        )
        
        assert result['left'] is not None
        assert result['right'] is None
    
    def test_single_right_hand(self):
        """Hand on right side should be labeled right."""
        selector = HandSelector()
        
        face_center = (320, 240)
        hand_centers = [(440, 240)]
        hand_landmarks = [np.ones((21, 3))]
        
        result = selector._assign_hand_sides(
            hand_landmarks, hand_centers, [0], face_center
        )
        
        assert result['left'] is None
        assert result['right'] is not None
    
    def test_two_hands_left_and_right(self):
        """Two hands on opposite sides should be labeled correctly."""
        selector = HandSelector()
        
        face_center = (320, 240)
        hand_centers = [(200, 240), (440, 240)]
        hand_landmarks = [np.ones((21, 3)), np.ones((21, 3)) * 2]
        
        result = selector._assign_hand_sides(
            hand_landmarks, hand_centers, [0, 1], face_center
        )
        
        assert result['left'] is not None
        assert result['right'] is not None
        # Check that they're different
        assert not np.array_equal(result['left'], result['right'])
    
    def test_two_hands_same_side(self):
        """Two hands on same side: first one selected."""
        selector = HandSelector()
        
        face_center = (320, 240)
        # Both on left side
        hand_centers = [(200, 240), (220, 240)]
        hand_landmarks = [np.ones((21, 3)) * 1, np.ones((21, 3)) * 2]
        
        result = selector._assign_hand_sides(
            hand_landmarks, hand_centers, [0, 1], face_center
        )
        
        assert result['left'] is not None
        assert result['right'] is None
        # Should be first one
        np.testing.assert_array_equal(result['left'], hand_landmarks[0])


class TestProcessHandsIntegration:
    """Integration tests for full process_hands pipeline."""
    
    def test_process_hands_normal_case(self):
        """Normal case: face and two hands."""
        selector = HandSelector(enable_debugging=False)
        
        face_landmarks = np.zeros((468, 3))
        face_landmarks[1] = [320, 240, 0]
        face_landmarks[33] = [300, 240, 0]
        face_landmarks[263] = [340, 240, 0]
        
        hand_landmarks = [
            np.ones((21, 3)) * 100,  # Left
            np.ones((21, 3)) * 200,  # Right
        ]
        
        result = selector.process_hands(
            face_landmarks, hand_landmarks, (480, 640)
        )
        
        assert result['face_center'] is not None
        assert result['left_hand'] is not None
        assert result['right_hand'] is not None
    
    def test_process_hands_no_face(self):
        """No face detected: should return empty."""
        selector = HandSelector(enable_debugging=False)
        
        hand_landmarks = [np.ones((21, 3)) * 100]
        
        result = selector.process_hands(None, hand_landmarks, (480, 640))
        
        assert result['left_hand'] is None
        assert result['right_hand'] is None
        assert result['face_center'] is None
    
    def test_process_hands_no_hands(self):
        """No hands detected: should return empty."""
        selector = HandSelector(enable_debugging=False)
        
        face_landmarks = np.zeros((468, 3))
        face_landmarks[1] = [320, 240, 0]
        
        result = selector.process_hands(face_landmarks, None, (480, 640))
        
        assert result['left_hand'] is None
        assert result['right_hand'] is None
    
    def test_process_hands_empty_hands_list(self):
        """Empty hands list: should return empty."""
        selector = HandSelector(enable_debugging=False)
        
        face_landmarks = np.zeros((468, 3))
        face_landmarks[1] = [320, 240, 0]
        
        result = selector.process_hands(face_landmarks, [], (480, 640))
        
        assert result['left_hand'] is None
        assert result['right_hand'] is None
    
    def test_process_hands_filtering(self):
        """Hands outside threshold should be filtered."""
        selector = HandSelector(
            distance_threshold=100.0,
            use_roi_filtering=False,
            enable_debugging=False,
        )
        
        face_landmarks = np.zeros((468, 3))
        face_landmarks[1] = [320, 240, 0]
        face_landmarks[33] = [300, 240, 0]
        face_landmarks[263] = [340, 240, 0]
        
        hand_landmarks = [
            np.ones((21, 3)) * 30 + [320, 240, 0],  # Very close, should be selected
            np.ones((21, 3)) * 30 + [10, 10, 0],    # Far away, should be filtered
        ]
        
        result = selector.process_hands(
            face_landmarks, hand_landmarks, (480, 640)
        )
        
        # First hand should be selected, second should be filtered
        assert len(result['filtered_hands']) == 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_hand_at_exact_threshold(self):
        """Hand at exact threshold distance should be included."""
        selector = HandSelector(distance_threshold=100.0, use_roi_filtering=False)
        
        face_center = (320, 240)
        hand_centers = [(320, 340)]  # Distance = 100
        
        filtered = selector._filter_hands_by_distance(hand_centers, face_center, 100.0)
        
        assert 0 in filtered
    
    def test_zero_distance(self):
        """Hand at same position as face should be included."""
        selector = HandSelector(distance_threshold=100.0, use_roi_filtering=False)
        
        face_center = (320, 240)
        hand_centers = [(320, 240)]  # Distance = 0
        
        filtered = selector._filter_hands_by_distance(hand_centers, face_center, 100.0)
        
        assert 0 in filtered
    
    def test_large_number_of_hands(self):
        """Should handle many hands and select closest 2."""
        selector = HandSelector(enable_debugging=False)
        
        face_landmarks = np.zeros((468, 3))
        face_landmarks[1] = [320, 240, 0]
        face_landmarks[33] = [300, 240, 0]
        face_landmarks[263] = [340, 240, 0]
        
        # Create 10 hands at varying distances
        hand_landmarks = [np.ones((21, 3)) * (i * 30 + 200) for i in range(10)]
        
        result = selector.process_hands(
            face_landmarks, hand_landmarks, (480, 640)
        )
        
        # Should select at most 2
        num_selected = sum([
            1 if result['left_hand'] is not None else 0,
            1 if result['right_hand'] is not None else 0,
        ])
        assert num_selected <= 2


def run_manual_tests():
    """Manual demonstration tests."""
    print("\n" + "="*60)
    print("MANUAL VALIDATION TESTS")
    print("="*60 + "\n")
    
    selector = HandSelector(
        distance_threshold=300.0,
        roi_width_ratio=0.5,
        roi_height_ratio=0.5,
        use_roi_filtering=True,
        enable_debugging=True,
    )
    
    # Create realistic test case
    face_landmarks = np.zeros((468, 3))
    face_landmarks[1] = [320, 240, 0]  # Nose
    face_landmarks[33] = [300, 220, 0]  # Left eye
    face_landmarks[263] = [340, 220, 0]  # Right eye
    
    print("Test 1: Normal case with two hands")
    print("-" * 40)
    hand_landmarks = [
        np.random.rand(21, 3) * 30 + 250,  # Left hand
        np.random.rand(21, 3) * 30 + 390,  # Right hand
    ]
    
    result = selector.process_hands(face_landmarks, hand_landmarks, (480, 640))
    print(f"Face center: {result['face_center']}")
    print(f"Selected indices: {result['selected_hand_indices']}")
    print(f"Left hand: {result['left_hand'] is not None}")
    print(f"Right hand: {result['right_hand'] is not None}\n")


if __name__ == "__main__":
    run_manual_tests()
    print("\n" + "="*60)
    print("To run all unit tests:")
    print("  pytest test_hand_selector.py -v")
    print("="*60 + "\n")
