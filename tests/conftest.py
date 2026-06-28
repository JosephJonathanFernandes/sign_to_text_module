"""
Shared pytest fixtures for all test tiers.

Usage:
    All fixtures are auto-discovered by pytest.
    Import from conftest in any test file automatically.
"""

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────
# Config fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config():
    """Return a validated Config instance."""
    from src.core.config import get_config
    cfg = get_config()
    cfg.validate()
    return cfg


@pytest.fixture(scope="session")
def num_frames(config):
    return config.preprocessing.num_frames  # 20


@pytest.fixture(scope="session")
def input_size(config):
    return config.frame_features.input_sequence_dim  # 506


@pytest.fixture(scope="session")
def spatial_dim(config):
    return config.frame_features.frame_features_dim  # 253


# ─────────────────────────────────────────────────────────────────
# Data fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_sequence(num_frames, input_size):
    """A valid all-zeros sequence of shape (num_frames, input_size)."""
    return np.zeros((num_frames, input_size), dtype=np.float32)


@pytest.fixture
def dummy_sequence_batch(num_frames, input_size):
    """A batch of 4 valid zero sequences."""
    return np.zeros((4, num_frames, input_size), dtype=np.float32)


@pytest.fixture
def random_sequence(num_frames, input_size):
    """A random sequence of correct shape — simulates real data."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((num_frames, input_size)).astype(np.float32)


@pytest.fixture
def null_landmarks():
    """All-None landmark inputs (missing detection)."""
    return None, None, None


@pytest.fixture
def zero_landmarks():
    """Valid-shape zero landmark arrays."""
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    face = np.zeros(264 * 3, dtype=np.float32)
    return left, right, face


@pytest.fixture
def synthetic_landmarks():
    """Plausible synthetic landmark arrays for testing feature extraction."""
    rng = np.random.default_rng(0)
    left = rng.uniform(0.1, 0.9, 63).astype(np.float32)
    right = rng.uniform(0.1, 0.9, 63).astype(np.float32)
    # Face raw needs to be large enough for index 263 (right eye)
    face = rng.uniform(0.1, 0.9, 264 * 3).astype(np.float32)
    return left, right, face
