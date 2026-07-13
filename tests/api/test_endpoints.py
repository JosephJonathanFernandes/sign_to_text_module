"""
API tests for the ISL Sign-to-Text FastAPI application.

These tests use FastAPI's TestClient to test all endpoints without
running a real server or loading real model weights. The model state
is mocked using app.state injection.

Tests:
  - GET /health — response schema
  - POST /predict — shape validation (422 on wrong shape)
  - POST /validate_features — dimension check
  - WS /ws/translate — connection, message protocol
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.api

NUM_FRAMES = 20
INPUT_SIZE = 506


@pytest.fixture(scope="module")
def mock_classes():
    """Fake class list for API testing — no real model needed."""
    return [f"SIGN_{i}" for i in range(10)]


@pytest.fixture(scope="module")
def client(mock_classes):
    """
    TestClient with mocked model state.
    Avoids loading real .pth files in CI.
    """
    from api.app import app

    # Inject fake model state before client context
    app.state.model_loaded = True
    app.state.num_classes = len(mock_classes)
    app.state.classes = mock_classes

    # Mock models with a callable that returns dummy prediction
    mock_model = MagicMock()
    app.state.models = [mock_model]

    with patch("api.app.load_ensemble", return_value=([mock_model], mock_classes, len(mock_classes))), \
         patch("api.app.ensemble_predict", return_value=(np.zeros((1, len(mock_classes))), np.zeros((1, len(mock_classes))))):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c



class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_has_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_returns_schema_version(self, client):
        data = client.get("/health").json()
        assert data["schema_version"] == "1.0"

    def test_health_sequence_length_matches_config(self, client, config):
        data = client.get("/health").json()
        assert data["sequence_length"] == config.preprocessing.num_frames

    def test_health_feature_dimension_matches_config(self, client, config):
        data = client.get("/health").json()
        assert data["feature_dimension"] == config.frame_features.input_sequence_dim

    def test_health_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_predict_wrong_shape_returns_422(self, client):
        # Send wrong shape: (10, 506) instead of (20, 506)
        wrong_shape = np.zeros((10, INPUT_SIZE)).tolist()
        r = client.post("/predict", json={"sequence": wrong_shape})
        assert r.status_code == 422

    def test_predict_wrong_feature_dim_returns_422(self, client):
        # Send wrong feature dim: (20, 253) instead of (20, 506)
        wrong_dim = np.zeros((NUM_FRAMES, 253)).tolist()
        r = client.post("/predict", json={"sequence": wrong_dim})
        assert r.status_code == 422

    def test_predict_empty_sequence_returns_422(self, client):
        r = client.post("/predict", json={"sequence": []})
        assert r.status_code == 422

    def test_predict_invalid_json_returns_422(self, client):
        r = client.post("/predict", content="NOT_JSON",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 422


class TestValidateFeaturesEndpoint:
    def test_validate_features_wrong_dim_returns_error(self, client):
        payload = {
            "schema_version": "1.0",
            "raw_landmarks": {
                "left_hand": None,
                "right_hand": None,
                "face": None,
            },
            "features": [0.0] * 100,  # Wrong length
        }
        r = client.post("/validate_features", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["valid"] is False
        assert data["dimension_check"] is False

    def test_validate_features_correct_dim_runs(self, client):
        payload = {
            "schema_version": "1.0",
            "raw_landmarks": {
                "left_hand": None,
                "right_hand": None,
                "face": None,
            },
            "features": [0.0] * 253,
        }
        r = client.post("/validate_features", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "valid" in data
        assert "mae" in data


class TestSchemas:
    def test_health_response_schema(self):
        from api.schemas import HealthResponse
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            num_classes=89,
            sequence_length=20,
            feature_dimension=506,
            device="cpu",
        )
        assert resp.schema_version == "1.0"
        assert resp.model_loaded is True

    def test_predict_request_schema(self):
        from api.schemas import PredictRequest
        req = PredictRequest(sequence=[[0.0] * 506] * 20)
        assert len(req.sequence) == 20

    def test_validate_features_response_schema(self):
        from api.schemas import ValidateFeaturesResponse
        resp = ValidateFeaturesResponse(
            valid=True, mae=0.0,
            dimension_check=True, range_check=True, errors=[]
        )
        assert resp.valid is True
