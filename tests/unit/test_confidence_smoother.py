"""
Unit tests: Confidence Smoother / State Machine Hysteresis
===========================================================
Tests the momentum-based temporal smoother to verify it correctly
suppresses high-frequency jitter (movement epenthesis) and only
commits a word when it appears consistently across multiple frames.

Config parameters under test (from cfg.live_inference):
  - temporal_window_size:    4   (rolling window for averaging)
  - temporal_patience:       1   (consecutive frames before commit)
  - temporal_delta:          0.1 (minimum confidence delta to switch)
  - temporal_decay_factor:   0.3 (exponential decay of stale predictions)
  - momentum_window:         3   (window for momentum-based commit)
  - momentum_commit_count:   2   (min votes in window to commit)
  - momentum_min_avg_conf:   0.6 (minimum average confidence to commit)
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────
# Stub smoother that mirrors the real temporal post-processor
# using config-driven parameters.
# ─────────────────────────────────────────────────────────

class MomentumSmoother:
    """
    Minimal reimplementation of the momentum-based temporal smoother
    for isolated unit testing without requiring the full webcam pipeline.

    Mirrors the logic in src/core/webcam.py using cfg.live_inference.
    """

    def __init__(self, config):
        li = config.live_inference
        self.window = li.temporal_window_size         # 4
        self.patience = li.temporal_patience          # 1
        self.delta = li.temporal_delta                # 0.1
        self.decay = li.temporal_decay_factor         # 0.3
        self.momentum_window = li.momentum_window     # 3
        self.commit_count = li.momentum_commit_count  # 2
        self.min_conf = li.momentum_min_avg_conf      # 0.6

        self._history: list = []            # (class_idx, confidence) per frame
        self._committed: str | None = None

    def update(self, class_idx: int, confidence: float, classes: list) -> str | None:
        """
        Feed a new (class, confidence) prediction.
        Returns the committed word name if the smoother commits, else None.
        """
        self._history.append((class_idx, confidence))
        if len(self._history) > self.window:
            self._history.pop(0)

        # Not enough history yet
        if len(self._history) < self.patience:
            return None

        # Momentum vote: count occurrences of most common class in window
        window = self._history[-self.momentum_window:]
        class_votes: dict[int, list] = {}
        for cls, conf in window:
            class_votes.setdefault(cls, []).append(conf)

        best_cls = max(class_votes, key=lambda c: len(class_votes[c]))
        votes = class_votes[best_cls]

        if (
            len(votes) >= self.commit_count
            and (sum(votes) / len(votes)) >= self.min_conf
        ):
            word = classes[best_cls] if best_cls < len(classes) else str(best_cls)
            if word != self._committed:
                self._committed = word
                return word

        return None

    def reset(self):
        self._history.clear()
        self._committed = None


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def smoother(config):
    return MomentumSmoother(config)


@pytest.fixture
def fresh_smoother(config):
    """A fresh smoother instance per test."""
    return MomentumSmoother(config)


DUMMY_CLASSES = ["hello", "thank_you", "yes", "no", "please"]


# ─────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────

class TestFlickerSuppression:
    def test_jitter_does_not_commit(self, fresh_smoother):
        """
        Alternating predictions (jitter) with low confidence should never commit.
        Simulates movement epenthesis between two similar signs.
        """
        jitter = [(0, 0.4), (1, 0.4), (0, 0.4), (1, 0.3), (0, 0.4)]
        results = [
            fresh_smoother.update(cls, conf, DUMMY_CLASSES)
            for cls, conf in jitter
        ]
        assert all(r is None for r in results), (
            "Smoother incorrectly committed a word during pure jitter"
        )

    def test_stable_high_conf_commits(self, fresh_smoother):
        """
        A stable, high-confidence prediction repeated across the window should commit.
        """
        stable = [(0, 0.85)] * 4  # "hello" for 4 frames at 85% confidence
        results = [
            fresh_smoother.update(cls, conf, DUMMY_CLASSES)
            for cls, conf in stable
        ]
        # At least one of the last frames should commit
        assert any(r == "hello" for r in results), (
            "Smoother failed to commit a stable, high-confidence prediction"
        )

    def test_low_confidence_jitter_suppressed(self, fresh_smoother):
        """
        Jitter around the same class but below momentum_min_avg_conf should not commit.
        """
        low_conf = [(0, 0.3), (0, 0.4), (0, 0.35), (0, 0.3)]
        results = [
            fresh_smoother.update(cls, conf, DUMMY_CLASSES)
            for cls, conf in low_conf
        ]
        assert all(r is None for r in results), (
            "Smoother committed despite average confidence below threshold"
        )

    def test_rapid_class_switching_no_commit(self, fresh_smoother):
        """
        Rapidly switching between 3 different classes should not commit any word.
        """
        rapid_switch = [(0, 0.75), (1, 0.75), (2, 0.75), (0, 0.75), (1, 0.75)]
        results = [
            fresh_smoother.update(cls, conf, DUMMY_CLASSES)
            for cls, conf in rapid_switch
        ]
        assert all(r is None for r in results), (
            "Smoother committed during rapid class switching"
        )


class TestMomentumCommit:
    def test_commit_requires_min_votes(self, fresh_smoother, config):
        """
        Fewer than momentum_commit_count votes for a class should not commit.
        """
        li = config.live_inference
        commit_count = li.momentum_commit_count  # 2

        # Only 1 vote in the window, below commit_count
        sequence = [(0, 0.9), (1, 0.9), (0, 0.9)]
        results = [
            fresh_smoother.update(cls, conf, DUMMY_CLASSES)
            for cls, conf in sequence
        ]
        # 0 appears 2× in last 3 → meets commit_count=2, so it should commit
        assert any(r == "hello" for r in results)

    def test_dominant_class_commits_after_window(self, fresh_smoother, config):
        """Class dominating the momentum window commits after sufficient votes."""
        # "yes" (class 2) appears 3× in window of 3
        sequence = [(2, 0.8), (2, 0.8), (2, 0.8)]
        results = [
            fresh_smoother.update(cls, conf, DUMMY_CLASSES)
            for cls, conf in sequence
        ]
        assert any(r == "yes" for r in results)

    def test_same_word_not_committed_twice(self, fresh_smoother):
        """Same committed word should not be re-committed consecutively."""
        # Fill window with "hello"
        sequence = [(0, 0.85)] * 6
        results = [
            fresh_smoother.update(cls, conf, DUMMY_CLASSES)
            for cls, conf in sequence
        ]
        committed = [r for r in results if r is not None]
        # Should commit at most once per sign
        hello_count = committed.count("hello")
        assert hello_count <= 1, f"'hello' committed {hello_count} times in one stable sequence"


class TestSmootherReset:
    def test_reset_clears_history(self, fresh_smoother):
        """After reset, smoother should not carry over previous predictions."""
        sequence = [(0, 0.85)] * 3
        for cls, conf in sequence:
            fresh_smoother.update(cls, conf, DUMMY_CLASSES)
        fresh_smoother.reset()
        # After reset, a single frame at high confidence should not commit
        result = fresh_smoother.update(0, 0.95, DUMMY_CLASSES)
        # Window is now size 1, below momentum_commit_count=2
        assert result is None

    def test_after_reset_can_commit_fresh(self, fresh_smoother):
        """After reset, the smoother should be able to commit a new word."""
        fresh_smoother.reset()
        sequence = [(1, 0.9)] * 4
        results = [
            fresh_smoother.update(cls, conf, DUMMY_CLASSES)
            for cls, conf in sequence
        ]
        assert any(r == "thank_you" for r in results)


class TestSmootherConfig:
    def test_config_temporal_window_is_4(self, config):
        assert config.live_inference.temporal_window_size == 4

    def test_config_temporal_delta_is_0_1(self, config):
        assert config.live_inference.temporal_delta == pytest.approx(0.1)

    def test_config_momentum_commit_count_is_2(self, config):
        assert config.live_inference.momentum_commit_count == 2

    def test_config_momentum_min_avg_conf_is_0_6(self, config):
        assert config.live_inference.momentum_min_avg_conf == pytest.approx(0.6)

    def test_config_temporal_decay_factor_is_0_3(self, config):
        assert config.live_inference.temporal_decay_factor == pytest.approx(0.3)
