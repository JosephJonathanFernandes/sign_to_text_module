"""
Pydantic request/response schemas for the ISL Sign-to-Text API.

All sequence shape constraints are derived dynamically from config —
never hardcoded — so retraining with different dimensions cannot
silently break the API.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    schema_version: str = "1.0"
    model_loaded: bool
    num_classes: int
    sequence_length: int       # NUM_FRAMES — read from config
    feature_dimension: int     # INPUT_SIZE — read from config
    device: str                # e.g. "cpu"


# ─────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Single-shot inference request.

    sequence: list of frames, each frame is a feature vector.
    Expected shape: (sequence_length, feature_dimension)
    Shape is validated at runtime against config values.
    """
    schema_version: str = "1.0"
    sequence: List[List[float]]


class RawLandmarksDict(BaseModel):
    left_hand: Optional[List[float]] = None
    right_hand: Optional[List[float]] = None
    face: Optional[List[float]] = None


class ValidateFeaturesRequest(BaseModel):
    schema_version: str = "1.0"
    raw_landmarks: RawLandmarksDict
    features: List[float]


class ValidateFeaturesResponse(BaseModel):
    valid: bool
    mae: float
    dimension_check: bool
    range_check: bool
    errors: List[str]


class Top5Entry(BaseModel):
    word: str
    confidence: float


class DebugInfo(BaseModel):
    top5: List[Top5Entry]
    raw_confidence: float   # confidence before temporal smoothing


class PredictResponse(BaseModel):
    predicted_word: str
    confidence: float
    debug: Optional[DebugInfo] = None   # only when DEBUG=true


# ─────────────────────────────────────────────
# WebSocket /ws/translate — incoming messages
# ─────────────────────────────────────────────

class LandmarkFrame(BaseModel):
    """
    Single-frame feature message from the frontend.

    type: must be "landmarks"
    features: flat float list of length INPUT_SIZE (506 by default)
    timestamp: optional unix millisecond timestamp for latency tracking
    """
    type: str
    schema_version: str = "1.0"
    feature_dimension: int = 506
    sequence_length: int = 20
    features: Optional[List[float]] = None
    timestamp: Optional[int] = None


# ─────────────────────────────────────────────
# WebSocket /ws/translate — outgoing messages
# ─────────────────────────────────────────────

class PredictionMessage(BaseModel):
    """Per-frame streaming prediction sent back on each inference."""
    type: str                           # "prediction"
    word: Optional[str]                 # None if confidence < threshold
    confidence: float
    sentence_so_far: str
    debug: Optional[DebugInfo] = None   # only when DEBUG=true


class TranslationMessage(BaseModel):
    """Final sentence sent on stop signal."""
    type: str                           # "translation"
    text: str                           # NLP post-processed full sentence
    words: List[str]                    # raw committed word list


class ErrorMessage(BaseModel):
    type: str = "error"
    message: str


class EmergencyAlert(BaseModel):
    """
    Emitted over the WebSocket when an emergency sign is detected with
    sufficient confidence, after temporal smoothing.

    The frontend should use this to show a red alert banner and/or
    trigger navigator.vibrate() on supported devices.

    severity values:
        "critical" — immediate danger (help, fire, danger, emergency, police)
        "warning"  — medical/assistance needed (stop, hospital, doctor)
    """
    type: str = "emergency_alert"
    word: str           # uppercase, e.g. "HELP"
    confidence: float   # smoothed confidence from temporal post-processor
    severity: str       # "critical" | "warning" — drives banner color and vibration pattern
    timestamp: int      # unix milliseconds
    session_id: str = ""  # WebSocket session UUID
