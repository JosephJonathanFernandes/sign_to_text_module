"""ONNX Runtime inference wrapper for ISL models.

Provides a unified inference interface with automatic fallback to PyTorch.
Includes timing, profiling, and error handling hooks.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

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
            from src.training.model import ISLModel

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
        if isinstance(logits, dict):
            logits = logits.get("sign_logits", logits.get("logits", logits))
        output = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        return output

    def infer_onnx(self, input_seq: np.ndarray, proximity: np.ndarray) -> np.ndarray:
        """Inference using ONNX Runtime."""
        if not self.onnx_available or self.session is None:
            raise RuntimeError("ONNX session not available")

        input_seq = input_seq.astype(np.float32)
        proximity = proximity.astype(np.float32)

        # Align input dimensions to what the ONNX session expects (robust to
        # feature-dimension mismatches caused by velocity toggles or export)
        try:
            sess_inputs = self.session.get_inputs()
            if sess_inputs:
                # Expect input_seq shape: (batch?, seq_len?, feat_dim?)
                expected_feat_dim = None
                # inspect last dimension of the first input's shape if available
                inp_shape = sess_inputs[0].shape
                if len(inp_shape) >= 1 and inp_shape[-1] is not None:
                    expected_feat_dim = int(inp_shape[-1])

                # If expected feature dim known and differs, pad/truncate
                if expected_feat_dim is not None:
                    # handle 2D arrays (seq, feat)
                    if input_seq.ndim == 2:
                        cur_feat = input_seq.shape[1]
                        if cur_feat != expected_feat_dim:
                            if cur_feat > expected_feat_dim:
                                input_seq = input_seq[:, :expected_feat_dim]
                            else:
                                pad = np.zeros((input_seq.shape[0], expected_feat_dim - cur_feat), dtype=np.float32)
                                input_seq = np.concatenate([input_seq, pad], axis=1)

                    # handle 3D arrays (batch, seq, feat)
                    elif input_seq.ndim >= 3:
                        cur_feat = input_seq.shape[-1]
                        if cur_feat != expected_feat_dim:
                            if cur_feat > expected_feat_dim:
                                input_seq = input_seq[..., :expected_feat_dim]
                            else:
                                pad_shape = list(input_seq.shape)
                                pad_shape[-1] = expected_feat_dim - cur_feat
                                pad = np.zeros(tuple(pad_shape), dtype=np.float32)
                                input_seq = np.concatenate([input_seq, pad], axis=-1)

                # If proximity is None, create a seq-shaped placeholder (we'll adapt shape later)
                if proximity is None:
                    if input_seq.ndim >= 3:
                        proximity = np.zeros((input_seq.shape[0], input_seq.shape[1]), dtype=np.float32)
                    else:
                        proximity = np.zeros((input_seq.shape[0], input_seq.shape[1]), dtype=np.float32)
                # If the session expects a batched input (3 dims) but we have (seq, feat), add batch dim
                if sess_inputs and len(inp_shape) >= 3:
                    if input_seq.ndim == 2:
                        input_seq = np.expand_dims(input_seq, axis=0)
                    # ensure proximity has batch dim as well
                    if proximity is not None and proximity.ndim == 2:
                        # proximity likely (seq,1) -> make (1,seq,1)
                        proximity = np.expand_dims(proximity, axis=0)

                # Final sanity: if proximity batch doesn't match input batch, try to broadcast when safe
                try:
                    if proximity is not None and input_seq.ndim >= 3 and proximity.ndim >= 3:
                        if proximity.shape[0] != input_seq.shape[0] and proximity.shape[0] == 1:
                            proximity = np.repeat(proximity, input_seq.shape[0], axis=0)
                except Exception:
                    pass
        except Exception:
            # Best-effort only; fall back to using provided arrays
            pass

        # Log shapes for diagnostics and adapt proximity to what the session expects
        try:
            sess_inputs = self.session.get_inputs() if self.session is not None else []
            expected = [tuple(i.shape) for i in sess_inputs]
        except Exception:
            sess_inputs = []
            expected = None

        # Determine expected proximity shape (heuristic: look for input named 'proximity' or second input)
        expected_prox_shape = None
        try:
            for i in sess_inputs:
                if 'proximity' in i.name.lower():
                    expected_prox_shape = i.shape
                    break
            if expected_prox_shape is None and len(sess_inputs) > 1:
                expected_prox_shape = sess_inputs[1].shape
        except Exception:
            expected_prox_shape = None

        # If session expects batched inputs but we were given a 2D (seq,feat) input,
        # add a batch dimension and align proximity from (seq,1) -> (1,seq) as needed.
        try:
            if input_seq.ndim == 2 and sess_inputs and len(inp_shape) >= 3:
                # add batch dim to sequence
                input_seq = np.expand_dims(input_seq, axis=0)

                if proximity is not None:
                    # If proximity is (seq,1), convert to (1,seq)
                    if proximity.ndim == 2 and proximity.shape[1] == 1:
                        proximity = np.squeeze(proximity, axis=-1)
                        proximity = np.expand_dims(proximity, axis=0)
                    # If proximity is (seq,), make it (1,seq)
                    elif proximity.ndim == 1:
                        proximity = np.expand_dims(proximity, axis=0)
                    # If proximity is (1,seq,1) or (1,seq), try to reduce to (1,seq)
                    elif proximity.ndim == 3 and proximity.shape[-1] == 1:
                        proximity = np.squeeze(proximity, axis=-1)
        except Exception:
            pass

        # If session expects 2D proximity (batch, seq) but we have (...,1), squeeze last axis
        try:
            if expected_prox_shape is not None and len(expected_prox_shape) == 2:
                # proximity could be (seq,1), (batch, seq,1) or (batch,seq)
                if proximity is not None and proximity.ndim >= 2:
                    # If last dim is singleton, squeeze it
                    if proximity.shape[-1] == 1:
                        proximity = np.squeeze(proximity, axis=-1)

                    # If proximity is (seq,) and input is batched, add batch dim
                    if proximity.ndim == 1 and input_seq.ndim >= 3:
                        proximity = np.expand_dims(proximity, axis=0)

                    # If proximity is (seq,) and input is 2D seq, expand to (1,seq)
                    if proximity.ndim == 1 and input_seq.ndim == 2:
                        proximity = np.expand_dims(proximity, axis=0)

                    # If proximity batch-size 1 but input has larger batch, broadcast
                    if proximity.ndim == 2 and input_seq.ndim >= 3 and proximity.shape[0] == 1 and input_seq.shape[0] > 1:
                        proximity = np.repeat(proximity, input_seq.shape[0], axis=0)

            # If session expects 3D proximity (batch, seq, feat) but we have 2D, expand trailing dim
            elif expected_prox_shape is not None and len(expected_prox_shape) >= 3:
                if proximity is not None and proximity.ndim == 2:
                    proximity = np.expand_dims(proximity, axis=-1)
                # broadcast batch if needed
                if proximity is not None and input_seq.ndim >= 3 and proximity.shape[0] == 1 and input_seq.shape[0] > 1:
                    proximity = np.repeat(proximity, input_seq.shape[0], axis=0)
        except Exception:
            pass

        logger.info(f"ONNX expected_inputs={expected}; passing input_seq.shape={getattr(input_seq, 'shape', None)}, proximity.shape={getattr(proximity, 'shape', None)}; expected_prox_shape={expected_prox_shape}")

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
