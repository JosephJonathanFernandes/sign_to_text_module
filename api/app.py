"""
ISL Sign-to-Text — FastAPI Application

Endpoints:
    GET  /health          Model status + config dimensions
    POST /predict         Single stateless inference (for testing)
    WS   /ws/translate    Real-time streaming translation

Design principles:
    - Model loaded ONCE in lifespan startup — never per request
    - Warmup inference burns off PyTorch JIT/cache overhead
    - All sequence dimensions read from config — never hardcoded
    - Per-WebSocket sessions keyed by UUID hex strings
    - Sliding deque window: predict on EVERY new frame once buffer full
    - Flood protection: discard frames when >MAX_PENDING inferences in-flight
    - Blocking torch calls wrapped in run_in_executor (never blocks event loop)
    - DEBUG=true env var enables top-5 probability responses
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from functools import partial
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import get_config
from src.inference.ensemble import ensemble_predict, load_ensemble

from api.inference import run_predict
from api.schemas import (
    HealthResponse, PredictRequest, PredictResponse,
    ValidateFeaturesRequest, ValidateFeaturesResponse
)
from api.session import InferenceSession, create_session
from src.shared.feature_extractor import build_single_frame_features

# ─────────────────────────────────────────────────────────────────────────────
# Config — read dynamically, NEVER hardcoded
# ─────────────────────────────────────────────────────────────────────────────

cfg = get_config()
NUM_FRAMES: int = cfg.preprocessing.num_frames          # e.g., 20
INPUT_SIZE: int = cfg.frame_features.input_sequence_dim  # e.g., 506
CONFIDENCE_THRESHOLD: float = cfg.inference.confidence_threshold  # e.g., 0.12
DEVICE: str = str(cfg.hardware.torch_device)             # e.g., "cpu"

# Max in-flight inference calls per session before frames are dropped
MAX_PENDING: int = 2

# Enable debug mode via: DEBUG=true python run_api.py
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

if DEBUG:
    print("[API] Debug mode ON — top-5 probabilities will be included in responses")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model once at startup. Run warmup inference.
    Clean up sessions on shutdown.
    """
    print("[API] Loading ensemble models...")
    models, classes, num_classes = load_ensemble()
    for m in models:
        m.eval()

    # ── Warmup inference ──────────────────────────────────────────────────────
    # PyTorch has cold-start overhead on the first forward pass (JIT, cache
    # initialization). Running one dummy pass at startup ensures all subsequent
    # real predictions are fast (20–100 ms instead of 400–1000 ms).
    print(f"[API] Running warmup inference ({NUM_FRAMES}×{INPUT_SIZE} zeros)...")
    dummy_seq = np.zeros((NUM_FRAMES, INPUT_SIZE), dtype=np.float32)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, partial(ensemble_predict, models, dummy_seq))
    print("[API] Warmup complete. Model is ready.")

    # Store in app state
    app.state.models = models
    app.state.classes = classes
    app.state.num_classes = num_classes
    app.state.model_loaded = True
    app.state.sessions: Dict[str, InferenceSession] = {}

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    app.state.sessions.clear()
    print("[API] Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ISL Sign-to-Text API",
    description=(
        "Real-time Indian Sign Language word recognition.\n\n"
        f"Input shape: `({NUM_FRAMES}, {INPUT_SIZE})` — read from config at startup.\n\n"
        "Set `DEBUG=true` to enable top-5 probability responses."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten this for production deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health() -> dict:
async def health() -> HealthResponse:
    """
    Model readiness check.

    Returns config-derived dimensions so the frontend can self-configure
    without hardcoding sequence_length or feature_dimension.
    """
    return HealthResponse(
        status="healthy",
        schema_version="1.0",
        model_loaded=getattr(app.state, "model_loaded", False),
        num_classes=getattr(app.state, "num_classes", 0),
        sequence_length=NUM_FRAMES,
        feature_dimension=INPUT_SIZE,
        device=str(cfg.hardware.torch_device)
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest) -> dict:
    """
    Stateless single-shot inference endpoint.

    Intended for local testing and integration validation.
    Does NOT use temporal smoothing or sentence building.

    Shape is validated dynamically against config:
        expected: ({NUM_FRAMES}, {INPUT_SIZE})

    Example request body:
        {"sequence": [[0.0, 0.0, ... (506 floats)] × 20 frames]}
    """
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model not yet loaded")

    # ── Shape validation from config — never hardcoded ────────────────────────
    sequence = np.array(request.sequence, dtype=np.float32)
    if sequence.shape != (NUM_FRAMES, INPUT_SIZE):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid sequence shape {tuple(sequence.shape)}. "
                f"Expected ({NUM_FRAMES}, {INPUT_SIZE}) — "
                f"sequence_length={NUM_FRAMES}, feature_dimension={INPUT_SIZE}."
            ),
        )

    # ── Run inference in thread — never blocks event loop ────────────────────
    loop = asyncio.get_running_loop()
    response_dict, _, _, _ = await loop.run_in_executor(
        None,
        partial(run_predict, app.state.models, sequence, app.state.classes, DEBUG),
    )

    return response_dict


# ─────────────────────────────────────────────────────────────────────────────
# POST /validate_features
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/validate_features", response_model=ValidateFeaturesResponse, tags=["Validation"])
async def validate_features(request: ValidateFeaturesRequest) -> dict:
    """
    Validates frontend-generated features against the backend ground-truth extractor.
    """
    errors = []
    dimension_check = True
    range_check = True

    # 1. Dimension Check
    feat_len = len(request.features)
    # The reference is a single frame without velocity (253)
    if feat_len not in [253, 506]:
        errors.append(f"Invalid feature length {feat_len}. Expected 253 or 506.")
        dimension_check = False

    # 2. Range Check (Valid normalized coordinates should ideally be in [-3.0, 3.0])
    for idx, val in enumerate(request.features):
        if not (-3.0 <= val <= 3.0) and not np.isnan(val) and not np.isinf(val):
            # We don't necessarily fail here if it's slightly off, but flag it
            pass
        if np.isnan(val):
            errors.append(f"NaN value found at index {idx}")
            range_check = False
            break

    # 3. Ground Truth MAE Calculation
    left_raw = np.array(request.raw_landmarks.left_hand, dtype=np.float32) if request.raw_landmarks.left_hand else None
    right_raw = np.array(request.raw_landmarks.right_hand, dtype=np.float32) if request.raw_landmarks.right_hand else None
    face_raw = np.array(request.raw_landmarks.face, dtype=np.float32) if request.raw_landmarks.face else None

    try:
        reference = build_single_frame_features(left_raw, right_raw, face_raw)
    except Exception as e:
        errors.append(f"Error building reference features: {str(e)}")
        return {
            "valid": False,
            "mae": -1.0,
            "dimension_check": dimension_check,
            "range_check": range_check,
            "errors": errors,
        }

    # Compare only the first 253 features (ignore velocity for this frame-by-frame check)
    incoming = np.array(request.features[:253], dtype=np.float32)
    difference = float(np.mean(np.abs(reference - incoming)))
    
    mae_tolerance = 1e-5
    valid = difference < mae_tolerance and dimension_check and range_check

    if not valid and difference >= mae_tolerance:
        errors.append(f"MAE {difference:.6f} exceeds tolerance {mae_tolerance}")

    return {
        "valid": valid,
        "mae": difference,
        "dimension_check": dimension_check,
        "range_check": range_check,
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket /ws/translate
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/translate")
async def ws_translate(websocket: WebSocket) -> None:
    """
    Real-time streaming translation endpoint.

    Incoming message types:
        {"type": "landmarks", "features": [...506 floats...], "timestamp": ...}
        {"type": "stop"}
        {"type": "clear"}

    Outgoing message types:
        {"type": "prediction", "word": "HELLO", "confidence": 0.94,
         "sentence_so_far": "HELLO HOW_ARE_YOU"}
        {"type": "translation", "text": "Hello, how are you?",
         "words": ["HELLO", "HOW_ARE_YOU"]}
        {"type": "error", "message": "..."}
        {"type": "cleared"}

    Sliding window behaviour:
        Each incoming frame is appended to a deque(maxlen=NUM_FRAMES).
        Inference runs on EVERY frame once the buffer is full — frames 1-20
        produce the first prediction, frames 2-21 produce the second, etc.

    Flood protection:
        If >MAX_PENDING inference calls are already in-flight, the incoming
        frame is discarded. This prevents queuing stale frames and keeps
        latency low when the frontend sends faster than inference can process.
    """
    await websocket.accept()

    # ── Create session with UUID — never keyed by websocket object ────────────
    session = create_session(NUM_FRAMES)
    session_id = session.session_id
    websocket.state.session_id = session_id
    app.state.sessions[session_id] = session

    short_id = session_id[:8]
    print(f"[WS] Session {short_id} connected — "
          f"total active: {len(app.state.sessions)}")

    loop = asyncio.get_running_loop()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON"}
                )
                continue

            msg_type = msg.get("type")

            # ── landmarks ─────────────────────────────────────────────────────
            if msg_type == "landmarks":
                schema_version = msg.get("schema_version", "1.0")
                feature_dimension = msg.get("feature_dimension", INPUT_SIZE)
                sequence_length = msg.get("sequence_length", NUM_FRAMES)
                
                if schema_version != "1.0":
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unsupported schema_version: {schema_version}"
                    })
                    continue
                    
                if feature_dimension != INPUT_SIZE or sequence_length != NUM_FRAMES:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Dimension mismatch. Expected {INPUT_SIZE}x{NUM_FRAMES}. Got {feature_dimension}x{sequence_length}."
                    })
                    continue

                features = msg.get("features")

                if not features or len(features) != INPUT_SIZE:
                    await websocket.send_json({
                        "type": "error",
                        "message": (
                            f"Expected {INPUT_SIZE} features, "
                            f"got {len(features) if features else 0}."
                        ),
                    })
                    continue

                # ── Flood protection ──────────────────────────────────────────
                # If too many inferences are already queued, drop this frame.
                # This keeps predictions responding to the LATEST frames,
                # not a backlog of old ones.
                if session.pending_count > MAX_PENDING:
                    session.pending_count = max(0, session.pending_count - 1)
                    continue

                # ── Append to sliding buffer ──────────────────────────────────
                frame = np.array(features, dtype=np.float32)
                session.buffer.append(frame)

                # ── Predict on every frame once buffer is full ────────────────
                # deque(maxlen=20): frames 1–20 → predict, 2–21 → predict, etc.
                if len(session.buffer) < NUM_FRAMES:
                    continue  # Still filling — not enough frames yet

                session.pending_count += 1
                sequence = np.array(list(session.buffer), dtype=np.float32)

                try:
                    # Blocking torch call — run in thread pool
                    pred_idx, confidence, all_probs = await loop.run_in_executor(
                        None,
                        partial(ensemble_predict, app.state.models, sequence),
                    )
                finally:
                    session.pending_count = max(0, session.pending_count - 1)

                # ── Temporal post-processing (per-session state) ──────────────
                stable_class, smoothed_conf = (
                    session.postprocessor.update_with_confidence(all_probs)
                )

                if stable_class is None:
                    continue  # Predictor not yet initialized (very first frame)

                word = app.state.classes[stable_class]

                # ── Sentence builder ──────────────────────────────────────────
                session.sentence_builder.update(word, smoothed_conf)

                # Build response
                response: dict = {
                    "type": "prediction",
                    "word": (
                        word.upper()
                        if smoothed_conf >= CONFIDENCE_THRESHOLD
                        else None
                    ),
                    "confidence": round(float(smoothed_conf), 4),
                    "sentence_so_far": session.sentence_builder.current_sentence,
                }

                if DEBUG:
                    top5_idx = np.argsort(all_probs)[::-1][:5]
                    response["debug"] = {
                        "top5": [
                            {
                                "word": app.state.classes[i].upper(),
                                "confidence": round(float(all_probs[i]), 4),
                            }
                            for i in top5_idx
                        ],
                        "raw_confidence": round(float(confidence), 4),
                        "stable_class": int(stable_class),
                        "pending_count": session.pending_count,
                    }

                await websocket.send_json(response)

            # ── stop ──────────────────────────────────────────────────────────
            elif msg_type == "stop":
                # Flush any sign still being held (not yet committed)
                session.sentence_builder.flush_pending_word()

                # NLP post-process and return final sentence
                final_text = session.sentence_builder.save_sentence()
                words_snapshot = list(session.sentence_builder.words)

                await websocket.send_json({
                    "type": "translation",
                    "text": final_text,
                    "words": words_snapshot,
                })

                # Reset session — ready for the next signing sequence
                session.reset()
                print(f"[WS] Session {short_id} stopped — "
                      f"'{final_text}' ({len(words_snapshot)} words)")

            # ── clear ─────────────────────────────────────────────────────────
            elif msg_type == "clear":
                session.reset()
                await websocket.send_json({"type": "cleared"})

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: '{msg_type}'. "
                               f"Expected: 'landmarks', 'stop', or 'clear'.",
                })

    except WebSocketDisconnect:
        print(f"[WS] Session {short_id} disconnected")
    except Exception as exc:
        print(f"[WS] Session {short_id} error: {exc}")
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass
    finally:
        app.state.sessions.pop(session_id, None)
        print(f"[WS] Session {short_id} cleaned up — "
              f"remaining: {len(app.state.sessions)}")
