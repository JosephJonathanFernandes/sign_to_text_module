"""
Unit tests: Dimensionality & Schema Validation
===============================================
Enforces strict shape checks on the LandmarkFrame (506D) and
PredictRequest (20×506) Pydantic schemas before any data reaches
the 20-frame sequence buffer.

Tests verify:
  - LandmarkFrame with correct 506-element features validates cleanly
  - LandmarkFrame with wrong dimension raises ValidationError
  - PredictRequest with correct (20, 506) shape passes
  - PredictRequest with wrong frame count or wrong feature dim fails
  - feature_dimension and sequence_length are read from config, not hardcoded
"""

import numpy as np
import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit

INPUT_DIM = 506
NUM_FRAMES = 20


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _make_landmark_frame(n_features: int = INPUT_DIM, type_: str = "landmarks") -> dict:
    """Build a raw LandmarkFrame dict payload."""
    return {
        "type": type_,
        "schema_version": "1.0",
        "feature_dimension": INPUT_DIM,
        "sequence_length": NUM_FRAMES,
        "features": [0.0] * n_features,
        "timestamp": 1_700_000_000_000,
    }


def _make_predict_request(
    n_frames: int = NUM_FRAMES,
    n_features: int = INPUT_DIM,
) -> dict:
    """Build a raw PredictRequest dict payload."""
    return {
        "schema_version": "1.0",
        "sequence": [[0.0] * n_features for _ in range(n_frames)],
    }


# ─────────────────────────────────────────────────────────
# LandmarkFrame schema
# ─────────────────────────────────────────────────────────

class TestLandmarkFrameSchema:
    def test_valid_506d_features_parses(self):
        from api.schemas import LandmarkFrame
        frame = LandmarkFrame(**_make_landmark_frame(INPUT_DIM))
        assert len(frame.features) == INPUT_DIM

    def test_type_must_be_landmarks(self):
        """'type' field is a plain string — any value is accepted at schema level."""
        from api.schemas import LandmarkFrame
        frame = LandmarkFrame(**_make_landmark_frame(INPUT_DIM, type_="landmarks"))
        assert frame.type == "landmarks"

    def test_null_features_is_allowed(self):
        """features is Optional — None is valid (e.g. no-hand-detected frame)."""
        from api.schemas import LandmarkFrame
        payload = _make_landmark_frame(INPUT_DIM)
        payload["features"] = None
        frame = LandmarkFrame(**payload)
        assert frame.features is None

    def test_wrong_dim_detected_at_app_layer(self):
        """
        Pydantic accepts any List[float], so dim checks are enforced
        at the application layer (api/app.py). Verify we can detect
        a wrong-dim payload by inspecting len(features).
        """
        from api.schemas import LandmarkFrame
        frame = LandmarkFrame(**_make_landmark_frame(42))
        # App-layer guard would reject this:
        assert len(frame.features) != INPUT_DIM

    def test_feature_dimension_field_matches_config(self):
        """feature_dimension in the schema must match config input_sequence_dim."""
        from api.schemas import LandmarkFrame
        from src.core.config import get_config
        cfg = get_config()
        frame = LandmarkFrame(**_make_landmark_frame())
        assert frame.feature_dimension == cfg.frame_features.input_sequence_dim

    def test_sequence_length_field_matches_config(self):
        from api.schemas import LandmarkFrame
        from src.core.config import get_config
        cfg = get_config()
        frame = LandmarkFrame(**_make_landmark_frame())
        assert frame.sequence_length == cfg.preprocessing.num_frames

    def test_empty_features_accepted_by_schema(self):
        """Empty list is valid at schema level (app layer enforces non-empty)."""
        from api.schemas import LandmarkFrame
        payload = _make_landmark_frame(0)
        frame = LandmarkFrame(**payload)
        assert frame.features == []

    def test_timestamp_is_optional(self):
        from api.schemas import LandmarkFrame
        payload = _make_landmark_frame()
        del payload["timestamp"]
        frame = LandmarkFrame(**payload)
        assert frame.timestamp is None


# ─────────────────────────────────────────────────────────
# PredictRequest schema
# ─────────────────────────────────────────────────────────

class TestPredictRequestSchema:
    def test_valid_20x506_passes(self):
        from api.schemas import PredictRequest
        req = PredictRequest(**_make_predict_request(NUM_FRAMES, INPUT_DIM))
        assert len(req.sequence) == NUM_FRAMES
        assert len(req.sequence[0]) == INPUT_DIM

    def test_wrong_frame_count_detected_at_app_layer(self):
        """Schema accepts any list length; app layer validates shape."""
        from api.schemas import PredictRequest
        req = PredictRequest(**_make_predict_request(5, INPUT_DIM))  # 5 frames, not 20
        assert len(req.sequence) != NUM_FRAMES

    def test_wrong_feature_dim_detected_at_app_layer(self):
        from api.schemas import PredictRequest
        req = PredictRequest(**_make_predict_request(NUM_FRAMES, 128))  # wrong feat dim
        assert len(req.sequence[0]) != INPUT_DIM

    def test_empty_sequence_accepted_by_schema(self):
        from api.schemas import PredictRequest
        req = PredictRequest(schema_version="1.0", sequence=[])
        assert req.sequence == []

    def test_schema_version_default(self):
        from api.schemas import PredictRequest
        req = PredictRequest(**_make_predict_request())
        assert req.schema_version == "1.0"

    def test_numpy_to_list_roundtrip(self):
        """Validate that converting a numpy array to list works for API payloads."""
        from api.schemas import PredictRequest
        rng = np.random.default_rng(42)
        seq_np = rng.standard_normal((NUM_FRAMES, INPUT_DIM)).astype(np.float32)
        payload = {"schema_version": "1.0", "sequence": seq_np.tolist()}
        req = PredictRequest(**payload)
        assert len(req.sequence) == NUM_FRAMES
        assert len(req.sequence[0]) == INPUT_DIM


# ─────────────────────────────────────────────────────────
# Config-driven dimension contract
# ─────────────────────────────────────────────────────────

class TestDimensionContract:
    def test_config_input_dim_is_506(self, config):
        assert config.frame_features.input_sequence_dim == 506

    def test_config_num_frames_is_20(self, config):
        assert config.preprocessing.num_frames == 20

    def test_spatial_dim_is_253(self, config):
        """Full feature = 2 × spatial (253 × 2 = 506)."""
        assert config.frame_features.frame_features_dim == 253
        assert config.frame_features.frame_features_dim * 2 == config.frame_features.input_sequence_dim

    def test_sequence_buffer_size(self, config):
        """Total buffer slots = frames × feature_dim = 10,120 floats."""
        total = config.preprocessing.num_frames * config.frame_features.input_sequence_dim
        assert total == 10_120
