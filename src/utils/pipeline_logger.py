"""Shared pipeline logger for inference, training, and k-fold runs.

This module creates a per-run log file and can mirror stdout/stderr into the
same logger so existing print-based code is captured without invasive rewrites.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator


LOG_DIR = "logs"


class _StreamToLogger(io.TextIOBase):
    """File-like stream that forwards text to a logger line by line."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self.logger.log(self.level, line)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.rstrip("\r"))
        self._buffer = ""


@dataclass
class PipelineLogger:
    """Helper around a configured logger and its log file path."""

    logger: logging.Logger
    log_path: str

    def event(self, name: str, **fields) -> None:
        """Emit a structured JSON event line."""
        payload = {
            "event": name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **fields,
        }
        self.logger.info("%s", json.dumps(payload, ensure_ascii=True, default=str))

    @contextlib.contextmanager
    def capture_stdio(self) -> Iterator[None]:
        """Mirror stdout/stderr into this logger while preserving console output."""
        import sys

        stdout_logger = _StreamToLogger(self.logger, logging.INFO)
        stderr_logger = _StreamToLogger(self.logger, logging.ERROR)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_logger
        sys.stderr = stderr_logger
        try:
            yield
        finally:
            stdout_logger.flush()
            stderr_logger.flush()
            sys.stdout = old_stdout
            sys.stderr = old_stderr


_configured_handlers: dict[str, logging.Logger] = {}


def _build_log_path(run_name: str, log_dir: str = LOG_DIR) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = run_name.replace(" ", "_").replace(os.sep, "_")
    return os.path.join(log_dir, f"{safe_name}_{timestamp}.log")


def setup_pipeline_logger(run_name: str, log_dir: str = LOG_DIR) -> PipelineLogger:
    """Create or reuse a logger for a given pipeline run."""
    log_path = _build_log_path(run_name, log_dir=log_dir)
    logger_name = f"sign_to_text.{run_name}.{os.path.basename(log_path)}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger_name not in _configured_handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        _configured_handlers[logger_name] = logger

    logger.info("Logger initialized for %s", run_name)
    logger.info("Log file: %s", log_path)
    return PipelineLogger(logger=logger, log_path=log_path)
