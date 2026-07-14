"""
Emergency Sign Notification System
====================================
Implements the Notifier abstraction with three concrete backends:

  NtfyNotifier   — pushes to ntfy.sh (internet required, no app needed)
  TelegramNotifier — sends Telegram bot message
  LocalNotifier  — writes to log file (works fully offline, always enabled)

Add new notifiers by subclassing BaseNotifier and registering in
build_notifiers(). No other file needs to change.

Configuration is read entirely from data/emergency_config.json.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger("sign_to_text.emergency")


# ─────────────────────────────────────────────────────────────────────────────
# NotificationEvent — unified payload consumed by every notifier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NotificationEvent:
    """
    Immutable payload passed to every notifier.

    Using a structured event instead of loose (word, confidence) args means
    adding fields (session_id, severity, location) in future never requires
    changing notifier method signatures — just add the field here.
    """
    word: str                   # uppercase display name e.g. "HELP"
    confidence: float           # smoothed confidence from temporal post-processor
    timestamp_ms: int           # unix milliseconds
    session_id: str = ""        # WebSocket session UUID (for multi-user tracing)
    severity: str = "urgent"    # reserved for future severity tiers

# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "data" / "emergency_config.json"


def load_emergency_config() -> dict:
    """Load emergency_config.json. Returns defaults if file is missing."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    logger.warning("[Emergency] emergency_config.json not found — using defaults")
    return {
        "emergency": {
            "confidence_threshold": 0.75,
            "cooldown_seconds": 10.0,
            "words": ["help", "stop", "danger", "fire"],
        },
        "notifiers": {
            "ntfy": {"enabled": False},
            "telegram": {"enabled": False},
            "local": {"enabled": True, "log_to_file": "logs/emergency_alerts.log"},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseNotifier(ABC):
    """Interface all notifiers must implement."""

    @abstractmethod
    async def send(self, event: NotificationEvent) -> bool:
        """
        Send an alert. Returns True on success, False on failure.
        Must never raise — catch and log internally.
        """


# ─────────────────────────────────────────────────────────────────────────────
# NtfyNotifier — push to ntfy.sh (or self-hosted ntfy)
# ─────────────────────────────────────────────────────────────────────────────

class NtfyNotifier(BaseNotifier):
    """
    Sends push notifications via ntfy.sh.

    The receiver just opens https://ntfy.sh/<topic> in a browser
    and taps Subscribe — no app required. Vibration is automatic.

    Set base_url to your own ntfy server for offline/intranet demos.
    """

    def __init__(self, topic: str, base_url: str, priority: str, tags: str):
        self.topic = topic
        self.base_url = base_url.rstrip("/")
        self.priority = priority
        self.tags = tags

    async def send(self, event: NotificationEvent) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.base_url}/{self.topic}",
                    content=f"\U0001f6a8 Emergency sign detected: {event.word} ({event.confidence:.0%} confidence)",
                    headers={
                        "Title": f"ISL Alert: {event.word}",
                        "Priority": self.priority,
                        "Tags": self.tags,
                    },
                )
            logger.info(f"[NtfyNotifier] Sent alert for '{event.word}' \u2014 status {resp.status_code}")
            return resp.status_code < 300
        except ImportError:
            logger.warning("[NtfyNotifier] httpx not installed \u2014 skipping")
            return False
        except Exception as exc:
            logger.error(f"[NtfyNotifier] Failed to send alert: {exc}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# TelegramNotifier — Telegram Bot API
# ─────────────────────────────────────────────────────────────────────────────

class TelegramNotifier(BaseNotifier):
    """
    Sends a message via a Telegram Bot.

    Setup:
      1. Create a bot via @BotFather → get bot_token
      2. Send a message to your bot → get chat_id from API
      3. Set in emergency_config.json
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    async def send(self, event: NotificationEvent) -> bool:
        if not self.bot_token or not self.chat_id:
            logger.warning("[TelegramNotifier] bot_token or chat_id not set")
            return False
        try:
            import httpx
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json={
                    "chat_id": self.chat_id,
                    "text": f"\U0001f6a8 *Emergency detected*: {event.word}\nConfidence: {event.confidence:.0%}",
                    "parse_mode": "Markdown",
                })
            logger.info(f"[TelegramNotifier] Sent alert for '{event.word}' \u2014 status {resp.status_code}")
            return resp.status_code < 300
        except Exception as exc:
            logger.error(f"[TelegramNotifier] Failed to send alert: {exc}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# LocalNotifier — file log (always works, no internet required)
# ─────────────────────────────────────────────────────────────────────────────

class LocalNotifier(BaseNotifier):
    """
    Writes emergency events to a local log file.

    Always works — even without internet. Useful as a fallback and
    for generating an audit trail during the demo.
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    async def send(self, event: NotificationEvent) -> bool:
        try:
            from datetime import datetime
            entry = (
                f"[{datetime.now().isoformat()}] "
                f"EMERGENCY: {event.word} | "
                f"confidence={event.confidence:.4f} | "
                f"session={event.session_id or 'unknown'}\n"
            )
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry)
            logger.info(f"[LocalNotifier] Logged emergency: '{event.word}'")
            return True
        except Exception as exc:
            logger.error(f"[LocalNotifier] Failed to write log: {exc}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# EmergencyConfig — stateless, loaded once at startup, shared across sessions
# ─────────────────────────────────────────────────────────────────────────────

class EmergencyConfig:
    """
    Holds the emergency word list, thresholds, and notifier instances.

    Loaded once at startup via EmergencyConfig.from_config().
    Shared across all WebSocket sessions — contains no per-session state.
    """

    def __init__(
        self,
        words: frozenset[str],
        confidence_threshold: float,
        cooldown_seconds: float,
        notifiers: list[BaseNotifier],
        severity_map: dict[str, str] = None,
    ):
        self.words = words
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.notifiers = notifiers
        # severity_map: word (lowercase) -> "critical" | "warning" | "info"
        # Words not in the map fall back to "warning"
        self._severity_map: dict[str, str] = severity_map or {}

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "EmergencyConfig":
        """Build from emergency_config.json. Call once at startup."""
        cfg = config or load_emergency_config()
        ecfg = cfg.get("emergency", {})
        ncfg = cfg.get("notifiers", {})

        words = frozenset(w.lower() for w in ecfg.get("words", []))
        confidence_threshold = float(ecfg.get("confidence_threshold", 0.75))
        cooldown = float(ecfg.get("cooldown_seconds", 10.0))
        severity_map = {k.lower(): v for k, v in ecfg.get("severity", {}).items()}

        notifiers: list[BaseNotifier] = []

        # LocalNotifier — first, always available
        local_cfg = ncfg.get("local", {})
        if local_cfg.get("enabled", True):
            notifiers.append(LocalNotifier(local_cfg.get("log_to_file", "logs/emergency_alerts.log")))

        # NtfyNotifier
        ntfy_cfg = ncfg.get("ntfy", {})
        if ntfy_cfg.get("enabled", False):
            notifiers.append(NtfyNotifier(
                topic=ntfy_cfg.get("topic", "isl-emergency-demo"),
                base_url=ntfy_cfg.get("base_url", "https://ntfy.sh"),
                priority=ntfy_cfg.get("priority", "urgent"),
                tags=ntfy_cfg.get("tags", "rotating_light"),
            ))

        # TelegramNotifier
        tg_cfg = ncfg.get("telegram", {})
        if tg_cfg.get("enabled", False):
            notifiers.append(TelegramNotifier(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", tg_cfg.get("bot_token", "")),
                chat_id=os.getenv("TELEGRAM_CHAT_ID", tg_cfg.get("chat_id", "")),
            ))

        logger.info(
            f"[EmergencyConfig] Loaded {len(words)} emergency words, "
            f"{len(notifiers)} notifier(s), "
            f"threshold={confidence_threshold}, cooldown={cooldown}s"
        )
        return cls(words, confidence_threshold, cooldown, notifiers, severity_map)

    def is_emergency(self, word: str, confidence: float) -> bool:
        """True if word is in the emergency list AND meets the confidence threshold."""
        return word.lower() in self.words and confidence >= self.confidence_threshold

    def get_severity(self, word: str) -> str:
        """Return the severity tier for a word. Defaults to 'warning' if not mapped."""
        return self._severity_map.get(word.lower(), "warning")


# ─────────────────────────────────────────────────────────────────────────────
# EmergencySessionState — per-session, owns edge-detection and cooldown state
# ─────────────────────────────────────────────────────────────────────────────

class EmergencySessionState:
    """
    Tracks emergency alert state for a single WebSocket session.

    Attach one instance to each InferenceSession so that concurrent connections
    never share cooldown timers or previous-word state.

    Edge-triggered detection:
        An alert fires only on the RISING EDGE — when the predicted word
        transitions from a non-emergency (or different) sign into an emergency
        sign. Holding the same sign continuously produces at most one alert
        per cooldown period.

        Previous  Current   Action
        ────────  ───────   ──────
        None      HELP      → Alert
        HELP      HELP      → Suppress (same sign, no edge)
        HELP      None      → (resets edge state)
        None      HELP      → Alert again (new occurrence)

    Usage:
        # At session creation:
        state = EmergencySessionState(config)

        # In the WS handler, after temporal smoothing:
        payload = state.check(word, smoothed_conf)
        if payload:
            await websocket.send_json(payload)
            asyncio.create_task(state.dispatch_notifications(word, smoothed_conf))
    """

    def __init__(self, config: EmergencyConfig, session_id: str = ""):
        self._config = config
        self._session_id = session_id
        self._previous_word: Optional[str] = None
        self._last_alert: dict[str, float] = {}
        # Tracked task set prevents 'Task exception was never retrieved' warnings
        self._background_tasks: Set["asyncio.Task"] = set()

    def _is_cooldown_active(self, word: str) -> bool:
        last = self._last_alert.get(word.lower(), 0.0)
        return (time.time() - last) < self._config.cooldown_seconds

    def check(self, word: Optional[str], confidence: float) -> Optional[dict]:
        """
        Synchronous edge-detection check. Call on every temporally-smoothed
        prediction (word may be None when confidence < threshold).

        Returns a WebSocket-ready emergency_alert dict on a rising edge,
        or None if the sign is held, not an emergency, or on cooldown.

        After receiving a non-None result, the caller should also call
        dispatch_notifications() as a fire-and-forget task.
        """
        prev = self._previous_word
        self._previous_word = word  # always update edge state

        if word is None:
            return None  # below confidence threshold — reset edge

        if not self._config.is_emergency(word, confidence):
            return None

        # ── Edge detection: suppress if the same emergency sign is held ───────
        if word.lower() == (prev or "").lower():
            return None  # same sign still active — no rising edge

        # ── Cooldown check ────────────────────────────────────────────────────
        if self._is_cooldown_active(word):
            logger.info(f"[EmergencySessionState] Cooldown active for '{word}', suppressing")
            return None

        self._last_alert[word.lower()] = time.time()
        severity = self._config.get_severity(word)
        logger.info(
            f"[EmergencySessionState] Rising edge detected: '{word}' "
            f"({confidence:.0%}) severity={severity}"
        )

        return {
            "type": "emergency_alert",
            "word": word.upper(),
            "confidence": round(float(confidence), 4),
            "severity": severity,
            "timestamp": int(time.time() * 1000),
            "session_id": self._session_id,
        }

    def dispatch_as_task(self, word: str, confidence: float) -> None:
        """
        Schedule notification dispatch as a tracked background task.

        Use this instead of bare asyncio.create_task() — tasks are stored in
        _background_tasks so exceptions are retrieved and logged rather than
        silently dropped (avoids 'Task exception was never retrieved' warnings).

        In the WS handler:
            session.emergency.dispatch_as_task(word, conf)
        """
        import asyncio
        task = asyncio.create_task(self._dispatch(word, confidence))
        self._background_tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: "asyncio.Task") -> None:
        """Callback: remove task from set and log any unexpected exception."""
        self._background_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                f"[EmergencySessionState] Notification task raised: {task.exception()}"
            )

    async def _dispatch(self, word: str, confidence: float) -> None:
        """Internal coroutine that fans out to all notifiers with a unified event."""
        import asyncio
        event = NotificationEvent(
            word=word.upper(),
            confidence=round(float(confidence), 4),
            timestamp_ms=int(time.time() * 1000),
            session_id=self._session_id,
            severity=self._config.get_severity(word),
        )
        results = await asyncio.gather(
            *[n.send(event) for n in self._config.notifiers],
            return_exceptions=True,
        )
        success_count = sum(1 for r in results if r is True)
        logger.info(
            f"[EmergencySessionState] Notifications dispatched for '{word}' — "
            f"{success_count}/{len(self._config.notifiers)} succeeded"
        )

    # Keep the awaitable form for test code that wants to await directly
    async def dispatch_notifications(self, word: str, confidence: float) -> None:
        """Awaitable version of _dispatch. Use dispatch_as_task() in production."""
        await self._dispatch(word, confidence)

    def reset(self) -> None:
        """Call when the session is reset (stop/clear signal). Clears edge state."""
        self._previous_word = None


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compat alias — app.py calls EmergencyDetector.from_config()
# ─────────────────────────────────────────────────────────────────────────────

# Keep the old name so app.py import doesn't need to change yet.
EmergencyDetector = EmergencyConfig
