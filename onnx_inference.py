"""ONNX Runtime inference wrapper for ISL models.

Provides a unified inference interface with automatic fallback to PyTorch.
Includes timing, profiling, and error handling hooks.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class InferenceStats:
    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    errors: int = 0
    fallback_count: int = 0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(1, self.total_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_time_ms": round(self.total_time_ms, 3),
            "avg_time_ms": round(self.avg_time_ms, 3),
            "min_time_ms": round(self.min_time_ms, 3) if self.min_time_ms != float("inf") else None,
            "max_time_ms": round(self.max_time_ms, 3),
            "errors": self.errors,
            "fallback_count": self.fallback_count,
        }


class ONNXModelWrapper:
    """Unified ONNX inference wrapper with PyTorch fallback."""

    def __init__(
        self,
        onnx_path: str,
        pytorch_model: Any = None,
        pytorch_checkpoint: str | None = None,
        device: str = "cpu",
        fallback_to_pytorch: bool = True,
        enable_profiling: bool = False,
    ):
        self.onnx_path = onnx_path
        self.pytorch_model = pytorch_model
        self.pytorch_checkpoint = pytorch_checkpoint
        self.device = torch.device(device)
        self.fallback_to_pytorch = fallback_to_pytorch
        self.enable_profiling = enable_profiling
        self.stats = InferenceStats()

        self.session = None
        self.onnx_available = False
        self.pytorch_loaded = pytorch_model is not None

        self._init_onnx()
        self._init_pytorch()

    def _init_onnx(self) -> None:
        """Initialize ONNX Runtime session."""
        try:
            import onnxruntime as rt
        except ImportError:
            logger.warning("onnxruntime not installed; ONNX inference disabled")
            return

        if not os.path.isfile(self.onnx_path):
            logger.warning(f"ONNX model not found: {self.onnx_path}")
            return

        try:
            self.session = rt.InferenceSession(
                self.onnx_path,
                providers=["CPUExecutionProvider"],
            )
            self.onnx_available = True
            logger.info(f"ONNX Runtime session loaded: {self.onnx_path}")
        except Exception as e:
            logger.warning(f"Failed to load ONNX session: {str(e)}")
            self.onnx_available = False

    def _init_pytorch(self) -> None:
        """Load PyTorch model if not already provided."""
        if self.pytorch_model is not None:
            self.pytorch_loaded = True
            return

        if self.pytorch_checkpoint is None:
            return

        try:
            from model import ISLModel

            ckpt = torch.load(self.pytorch_checkpoint, map_location=self.device)
            model_dict = ckpt.get("model_state_dict", ckpt)
            model = ISLModel()
            if isinstance(model_dict, dict):
                model.load_state_dict(model_dict, strict=False)
            self.pytorch_model = model.to(self.device).eval()
            self.pytorch_loaded = True
            logger.info(f"PyTorch model loaded: {self.pytorch_checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to load PyTorch checkpoint: {str(e)}")
            self.pytorch_loaded = False

    @torch.no_grad()
    def infer_pytorch(self, input_seq: np.ndarray, proximity: np.ndarray) -> np.ndarray:
        """Inference using PyTorch model."""
        if not self.pytorch_loaded:
            raise RuntimeError("PyTorch model not available")

        input_t = torch.from_numpy(input_seq).to(self.device)
        prox_t = torch.from_numpy(proximity).to(self.device)

        logits = self.pytorch_model(input_t, prox_t)
        output = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        return output

    def infer_onnx(self, input_seq: np.ndarray, proximity: np.ndarray) -> np.ndarray:
        """Inference using ONNX Runtime."""
        if not self.onnx_available or self.session is None:
            raise RuntimeError("ONNX session not available")

        input_seq = input_seq.astype(np.float32)
        proximity = proximity.astype(np.float32)

        inputs = {
            "input_seq": input_seq,
            "proximity": proximity,
        }

        try:
            output = self.session.run(None, inputs)
            logits = output[0].astype(np.float32)

            from scipy.special import softmax
            probs = softmax(logits, axis=-1).astype(np.float32)
            return probs
        except Exception as e:
            logger.error(f"ONNX inference failed: {str(e)}")
            raise

    def __call__(self, input_seq: np.ndarray, proximity: np.ndarray) -> np.ndarray:
        """Unified inference with fallback."""
        if not isinstance(input_seq, np.ndarray):
            input_seq = np.asarray(input_seq, dtype=np.float32)
        if not isinstance(proximity, np.ndarray):
            proximity = np.asarray(proximity, dtype=np.float32)

        input_seq = input_seq.astype(np.float32)
        proximity = proximity.astype(np.float32)

        start = time.time()
        output = None
        used_fallback = False

        if self.onnx_available and self.session is not None:
            try:
                output = self.infer_onnx(input_seq, proximity)
            except Exception as e:
                logger.warning(f"ONNX inference failed: {str(e)}; falling back to PyTorch")
                used_fallback = True
                if not self.fallback_to_pytorch:
                    raise

        if output is None:
            if not self.pytorch_loaded:
                raise RuntimeError("No inference backend available (ONNX and PyTorch both failed)")
            output = self.infer_pytorch(input_seq, proximity)
            used_fallback = True

        elapsed_ms = (time.time() - start) * 1000.0

        if self.enable_profiling:
            self.stats.total_calls += 1
            self.stats.total_time_ms += elapsed_ms
            self.stats.min_time_ms = min(self.stats.min_time_ms, elapsed_ms)
            self.stats.max_time_ms = max(self.stats.max_time_ms, elapsed_ms)
            if used_fallback:
                self.stats.fallback_count += 1

        return output

    def get_stats(self) -> dict[str, Any]:
        """Return profiling statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset profiling statistics."""
        self.stats = InferenceStats()
