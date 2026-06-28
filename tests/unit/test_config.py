"""
Unit tests for src/core/config.py

Tests:
  - Config instantiation and validation
  - Feature dimension computations (spatial + velocity)
  - Computed properties correctness
  - Config hash reproducibility
"""

import pytest


pytestmark = pytest.mark.unit


class TestConfigInstantiation:
    def test_get_config_returns_instance(self):
        from src.core.config import get_config
        cfg = get_config()
        assert cfg is not None

    def test_config_is_singleton(self):
        from src.core.config import get_config
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_config_validate_passes(self, config):
        # validate() raises AssertionError on invalid config
        config.validate()  # must not raise


class TestFeatureDimensions:
    """Verify the 253/506 feature dimension contract is enforced by config."""

    def test_landmark_dim_per_hand(self, config):
        # 21 landmarks × 3 coords = 63
        assert config.landmarks.landmark_dim_per_hand == 63

    def test_raw_frame_features_dim(self, config):
        # 63 × 2 hands = 126
        assert config.landmarks.raw_frame_features_dim == 126

    def test_spatial_dim_with_face_relative(self, config):
        # face_relative=True → relative_frame_features_dim == 126
        assert config.spatial.relative_frame_features_dim == 126

    def test_proximity_dim(self, config):
        # face_relative=True → proximity_dim == 1
        assert config.spatial.proximity_dim == 1

    def test_frame_features_dim(self, config):
        # raw(126) + relative(126) + proximity(1) = 253
        assert config.frame_features.frame_features_dim == 253

    def test_input_sequence_dim_with_velocity(self, config):
        # use_velocity=True → 253 × 2 = 506
        assert config.frame_features.use_velocity is True
        assert config.frame_features.input_sequence_dim == 506

    def test_num_frames(self, config):
        assert config.preprocessing.num_frames == 20

    def test_proximity_index(self, config):
        # proximity is the last element of the spatial block: index 252
        assert config.frame_features.proximity_index == 252





class TestModelConfig:
    def test_model_config_validate(self, config):
        config.model.validate()  # must not raise

    def test_hidden_size(self, config):
        assert config.model.hidden_size == 64

    def test_num_layers(self, config):
        assert config.model.num_layers == 3

    def test_bidirectional(self, config):
        assert config.model.bidirectional is True


class TestHardwareConfig:
    def test_hardware_validate(self, config):
        config.hardware.validate()

    def test_device_type(self, config):
        assert config.hardware.device_type in ("cpu", "cuda")

    def test_num_threads_positive(self, config):
        assert config.hardware.num_threads >= 1
