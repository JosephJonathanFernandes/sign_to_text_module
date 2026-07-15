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
import numpy as np
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
        buffer:          Pre-allocated circular numpy buffer (NUM_FRAMES, INPUT_SIZE)
        postprocessor:   TemporalPostProcessor — smoothing + patience + hysteresis
        sentence_builder: SentenceBuilder — word commit + NLP post-processing
        pending_count:   Number of inference calls currently in-flight (flood guard)
        created_at:      Unix timestamp of session creation
        write_idx:       Current index to write to in the circular buffer
        frames_received: Total frames received
    """
    session_id: str
    buffer: np.ndarray
    postprocessor: TemporalPostProcessor
    sentence_builder: SentenceBuilder
    emergency: "EmergencySessionState"
    created_at: float = field(default_factory=time.time)
    write_idx: int = 0
    frames_received: int = 0
    idle_frames: int = 0
    pending_count: int = 0
    # Frame-to-frame landmark jump detection
    prev_frame: np.ndarray = field(default=None)      # Raw coords of last accepted frame
    landmark_jump_count: int = 0                       # Consecutive high-jump frames seen

    def reset(self) -> None:
        """
        Reset all state for session reuse after a stop/clear signal.

        - Clears the circular buffer
        - Resets temporal smoother and stability predictor
        - Replaces SentenceBuilder to clear all word and stability tracking
        - Resets emergency edge-detection state
        - Resets flood counter
        """
        self.buffer.fill(0.0)
        self.postprocessor.reset()
        self.sentence_builder = _make_sentence_builder()
        self.emergency.reset()
        self.pending_count = 0
        self.write_idx = 0
        self.frames_received = 0
        self.idle_frames = 0
        self.prev_frame = None
        self.landmark_jump_count = 0

    def append_frame(self, frame: np.ndarray) -> None:
        self.buffer[self.write_idx] = frame
        self.write_idx = (self.write_idx + 1) % len(self.buffer)
        self.frames_received += 1

    def get_sequence(self) -> np.ndarray:
        if self.frames_received < len(self.buffer):
            # Not full yet
            return None
        if self.write_idx == 0:
            return self.buffer
        return np.concatenate((self.buffer[self.write_idx:], self.buffer[:self.write_idx]))


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


def create_session(num_frames: int, input_size: int, emergency_config: "EmergencyConfig" = None) -> InferenceSession:
    """
    Factory: create a fresh InferenceSession with correctly-sized buffer.

    Args:
        num_frames:        Sequence length from config (e.g., 20). Sets circular buffer size.
        input_size:        Feature dimension (e.g., 506).
        emergency_config:  Shared EmergencyConfig loaded at startup. If None,
                           a default config is created (for testing only).

    Returns:
        InferenceSession ready for use.
    """
    from api.emergency import EmergencyConfig, EmergencySessionState
    cfg = emergency_config or EmergencyConfig.from_config()
    sid = uuid.uuid4().hex
    return InferenceSession(
        session_id=sid,
        buffer=np.zeros((num_frames, input_size), dtype=np.float32),
        postprocessor=_make_postprocessor(),
        sentence_builder=_make_sentence_builder(),
        emergency=EmergencySessionState(cfg, session_id=sid),
        created_at=time.time(),
    )
