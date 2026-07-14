"""
Per-WebSocket inference session.

Each WebSocket connection owns one InferenceSession that holds:
  - A sliding deque buffer (maxlen = NUM_FRAMES from config)
  - An independent TemporalPostProcessor (smoothing + stability)
  - An independent SentenceBuilder (word accumulation + NLP)
  - A flood-protection counter (pending_count)

Sessions are keyed by UUID hex strings, never by WebSocket object IDs.
"""

import uuid
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.inference.temporal_postprocessor import TemporalPostProcessor
from src.inference.sentence_builder import SentenceBuilder

if TYPE_CHECKING:
    from api.emergency import EmergencyConfig, EmergencySessionState


@dataclass
class InferenceSession:
    """
    Isolated state for a single WebSocket client.

    Attributes:
        session_id:      UUID hex string (e.g., "a3f9c1...") — used as dict key
        buffer:          Sliding deque of shape-(INPUT_SIZE,) numpy arrays
        postprocessor:   TemporalPostProcessor — smoothing + patience + hysteresis
        sentence_builder: SentenceBuilder — word commit + NLP post-processing
        pending_count:   Number of inference calls currently in-flight (flood guard)
        created_at:      Unix timestamp of session creation
    """
    session_id: str
    buffer: deque
    postprocessor: TemporalPostProcessor
    sentence_builder: SentenceBuilder
    emergency: "EmergencySessionState"
    pending_count: int = 0
    created_at: float = field(default_factory=time.time)

    def reset(self) -> None:
        """
        Reset all state for session reuse after a stop/clear signal.

        - Clears the sliding buffer (retains maxlen)
        - Resets temporal smoother and stability predictor
        - Replaces SentenceBuilder to clear all word and stability tracking
        - Resets emergency edge-detection state
        - Resets flood counter
        """
        self.buffer.clear()
        self.postprocessor.reset()
        self.sentence_builder = _make_sentence_builder()
        self.emergency.reset()
        self.pending_count = 0


# ─────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────

def _make_postprocessor() -> TemporalPostProcessor:
    """Create a TemporalPostProcessor with production-tuned parameters."""
    return TemporalPostProcessor(
        window_size=8,
        patience=3,
        delta=0.12,
        enable_decay=True,
        decay_factor=0.3,
    )


def _make_sentence_builder() -> SentenceBuilder:
    """Create a SentenceBuilder with production-tuned parameters."""
    return SentenceBuilder(
        confidence_threshold=0.60,
        stability_frames=8,
        ambiguity_margin_threshold=0.05,
        ambiguity_delay_frames=4,
        auto_sentence_timeout=60,
    )


def create_session(num_frames: int, emergency_config: "EmergencyConfig" = None) -> InferenceSession:
    """
    Factory: create a fresh InferenceSession with correctly-sized buffer.

    Args:
        num_frames:        Sequence length from config (e.g., 20). Sets deque maxlen.
        emergency_config:  Shared EmergencyConfig loaded at startup. If None,
                           a default config is created (for testing only).

    Returns:
        InferenceSession ready for use.
    """
    from api.emergency import EmergencyConfig, EmergencySessionState
    cfg = emergency_config or EmergencyConfig.from_config()
    return InferenceSession(
        session_id=uuid.uuid4().hex,
        buffer=deque(maxlen=num_frames),
        postprocessor=_make_postprocessor(),
        sentence_builder=_make_sentence_builder(),
        emergency=EmergencySessionState(cfg),
        created_at=time.time(),
    )
