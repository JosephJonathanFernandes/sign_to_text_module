"""
Unit tests for src/shared/feature_extractor.py

This is the Single Source of Truth for feature extraction.
Tests verify:
  - Output shape is always (253,) float32
  - Null/missing inputs produce zero vectors (not errors)
  - Feature ordering: [left_norm(63) | right_norm(63) | left_rel(63) | right_rel(63) | proximity(1)]
  - Proximity is a finite scalar
  - Zero-drift: same inputs always produce the same output
"""

import numpy as np
import pytest


pytestmark = pytest.mark.unit

EXPECTED_DIM = 253
LANDMARK_DIM = 63


class TestOutputShape:
    def test_all_none_returns_253(self):
        from src.shared.feature_extractor import build_single_frame_features
        result = build_single_frame_features(None, None, None)
        assert result.shape == (EXPECTED_DIM,)

    def test_all_none_dtype_is_float32(self):
        from src.shared.feature_extractor import build_single_frame_features
        result = build_single_frame_features(None, None, None)
        assert result.dtype == np.float32

    def test_all_none_is_zeros(self):
        from src.shared.feature_extractor import build_single_frame_features
        result = build_single_frame_features(None, None, None)
        # all spatial dims should be zero when no landmarks
        assert np.allclose(result[:252], 0.0)

    def test_with_landmarks_returns_253(self, synthetic_landmarks):
        from src.shared.feature_extractor import build_single_frame_features
        left, right, face = synthetic_landmarks
        result = build_single_frame_features(left, right, face)
        assert result.shape == (EXPECTED_DIM,)

    def test_with_zero_landmarks_returns_253(self, zero_landmarks):
        from src.shared.feature_extractor import build_single_frame_features
        left, right, face = zero_landmarks
        result = build_single_frame_features(left, right, face)
        assert result.shape == (EXPECTED_DIM,)


class TestFeatureLayout:
    """Verify the exact feature layout matches the documented contract."""

    def test_left_norm_block_is_first_63(self, synthetic_landmarks):
        from src.shared.feature_extractor import build_single_frame_features, normalize_hand_landmarks
        left, right, face = synthetic_landmarks
        result = build_single_frame_features(left, right, face)
        expected_left_norm = normalize_hand_landmarks(left)
        np.testing.assert_allclose(result[:63], expected_left_norm, rtol=1e-5)

    def test_right_norm_block_is_second_63(self, synthetic_landmarks):
        from src.shared.feature_extractor import build_single_frame_features, normalize_hand_landmarks
        left, right, face = synthetic_landmarks
        result = build_single_frame_features(left, right, face)
        expected_right_norm = normalize_hand_landmarks(right)
        np.testing.assert_allclose(result[63:126], expected_right_norm, rtol=1e-5)

    def test_proximity_is_last_element(self, synthetic_landmarks):
        from src.shared.feature_extractor import build_single_frame_features
        left, right, face = synthetic_landmarks
        result = build_single_frame_features(left, right, face)
        proximity = result[252]
        assert np.isfinite(proximity), "Proximity must be a finite scalar"

    def test_proximity_none_inputs_is_valid(self):
        from src.shared.feature_extractor import build_single_frame_features
        result = build_single_frame_features(None, None, None)
        proximity = result[252]
        assert np.isfinite(proximity)


class TestNormalization:
    def test_normalize_zero_vector_returns_zeros(self):
        from src.shared.feature_extractor import normalize_hand_landmarks
        zero_hand = np.zeros(63, dtype=np.float32)
        result = normalize_hand_landmarks(zero_hand)
        np.testing.assert_array_equal(result, np.zeros(63, dtype=np.float32))

    def test_normalize_wrong_length_returns_zeros(self):
        from src.shared.feature_extractor import normalize_hand_landmarks
        wrong_length = np.ones(42, dtype=np.float32)
        result = normalize_hand_landmarks(wrong_length)
        np.testing.assert_array_equal(result, np.zeros(63, dtype=np.float32))

    def test_normalize_none_returns_zeros(self):
        from src.shared.feature_extractor import normalize_hand_landmarks
        result = normalize_hand_landmarks(None)
        np.testing.assert_array_equal(result, np.zeros(63, dtype=np.float32))

    def test_normalize_non_trivial_hand_is_unit_scale(self):
        from src.shared.feature_extractor import normalize_hand_landmarks
        rng = np.random.default_rng(1)
        hand = rng.uniform(0.1, 0.9, 63).astype(np.float32)
        result = normalize_hand_landmarks(hand)
        # After centering and scaling, max distance from origin should be ≤ 1
        reshaped = result.reshape(21, 3)
        dists = np.linalg.norm(reshaped, axis=1)
        assert dists.max() <= 1.0 + 1e-5


class TestZeroDrift:
    """Same inputs must always produce identical outputs."""

    def test_deterministic_with_landmarks(self, synthetic_landmarks):
        from src.shared.feature_extractor import build_single_frame_features
        left, right, face = synthetic_landmarks
        r1 = build_single_frame_features(left.copy(), right.copy(), face.copy())
        r2 = build_single_frame_features(left.copy(), right.copy(), face.copy())
        np.testing.assert_array_equal(r1, r2)

    def test_deterministic_none_inputs(self):
        from src.shared.feature_extractor import build_single_frame_features
        r1 = build_single_frame_features(None, None, None)
        r2 = build_single_frame_features(None, None, None)
        np.testing.assert_array_equal(r1, r2)

    def test_no_nan_in_output(self, synthetic_landmarks):
        from src.shared.feature_extractor import build_single_frame_features
        left, right, face = synthetic_landmarks
        result = build_single_frame_features(left, right, face)
        assert not np.any(np.isnan(result)), "Feature vector must not contain NaN"

    def test_no_inf_in_output(self, synthetic_landmarks):
        from src.shared.feature_extractor import build_single_frame_features
        left, right, face = synthetic_landmarks
        result = build_single_frame_features(left, right, face)
        assert not np.any(np.isinf(result)), "Feature vector must not contain Inf"
