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
import time
import logging
import psutil
import hashlib
from collections import deque
from contextlib import asynccontextmanager
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import get_config
from src.inference.ensemble import check_ood
from src.inference.onnx_ensemble import detect_and_load_models, ensemble_predict_mixed
from src.training.adapter_model import AdapterModel

from api.emergency import EmergencyDetector
from api.inference import run_predict
from api.schemas import (
    HealthResponse, PredictRequest, PredictResponse,
    ValidateFeaturesRequest, ValidateFeaturesResponse,
    FeedbackRequest
)
from api.session import InferenceSession, create_session
from api.feedback import process_feedback_async
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

# Maximum allowed WebSocket payload size (bytes) to prevent OOM
MAX_PAYLOAD_SIZE: int = 50000

# Velocity threshold below which a frame is considered idle
IDLE_VELOCITY_THRESHOLD: float = 0.15

# Directory containing continual learning adapter weights
ADAPTER_WEIGHTS_DIR: str = "adapter_weights"

# Skeleton quality thresholds
# Fraction of raw hand coords (features 0-126) that must be non-zero for frame to be valid
LANDMARK_ZERO_RATIO_THRESHOLD: float = 0.5
# Mean absolute landmark jump (normalized) above which a frame is considered a tracking glitch
LANDMARK_JUMP_THRESHOLD: float = 0.20
# Consecutive jump frames before inference is paused (mirrors idle_frames pattern)
MAX_CONSECUTIVE_JUMPS: int = 3

# Enable debug mode via: DEBUG=true python run_api.py
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
ENV: str = os.getenv("ENV", "production").lower()

if DEBUG:
    print("[API] Debug mode ON — top-5 probabilities will be included in responses")

# ─────────────────────────────────────────────────────────────────────────────
# Logging & Metrics State
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sign_to_text")

START_TIME = time.time()
TOTAL_PREDICTIONS = 0
DROPPED_FRAMES = 0
LATENCY_HISTORY = deque(maxlen=1000)
ACTIVE_SESSIONS = 0
INFERENCE_TIME_HISTORY = deque(maxlen=1000)
MOTION_SCORE_HISTORY = deque(maxlen=1000)

# Initialize psutil CPU percent (first call returns 0.0)
process = psutil.Process(os.getpid())
process.cpu_percent(interval=None)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model once at startup. Run warmup inference.
    Clean up sessions on shutdown.
    """
    logger.info("startup_started", extra={"event": "startup_started"})

    # ── Load models — auto-detect ONNX (INT8 > FP32) then PyTorch fallback ───
    # Priority: model_int8.onnx > model.onnx > model.pth
    # Searches models/ and models/ensemble/ (picks up fold checkpoints too)
    model_dirs = [
        cfg.paths.model_save_path.replace("model.pth", "").rstrip("/\\"),
        cfg.paths.ensemble_dir,
    ]
    models, load_meta = detect_and_load_models(
        ensemble_dir=model_dirs,
        max_models=cfg.live_inference.ensemble_size,
        device=DEVICE,
    )
    if not models:
        raise RuntimeError("No models found. Train with --train or --kfold first.")

    logger.info(
        "models_loaded",
        extra={
            "onnx_models": load_meta["onnx_models"],
            "pytorch_models": load_meta["pytorch_models"],
            "total": load_meta["total_models"],
            "artifacts": [
                f"{a['family']} ({a['kind']})"
                for a in load_meta["selected_artifacts"]
            ],
        },
    )

    # Read class list from model metadata (source of truth for trained models)
    import os as _os
    import json as _json
    classes = []
    
    # Try to load from model metadata first
    meta_path = _os.path.join(cfg.paths.model_save_path.replace("model.pth", "").rstrip("/\\"), "model_metadata.json")
    if _os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = _json.load(f)
                classes = meta.get("checkpoint", {}).get("classes", [])
        except Exception as e:
            logger.warning(f"Could not load classes from {meta_path}: {e}")
            
    # Fallback to processed dir
    if not classes:
        classes = sorted([
            d for d in _os.listdir(cfg.paths.processed_dir)
            if _os.path.isdir(_os.path.join(cfg.paths.processed_dir, d))
        ])
    num_classes = len(classes)

    # ── Warmup inference ──────────────────────────────────────────────────────
    logger.info("warmup_started", extra={"num_frames": NUM_FRAMES, "input_size": INPUT_SIZE})
    dummy_seq = np.zeros((NUM_FRAMES, INPUT_SIZE), dtype=np.float32)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        partial(
            ensemble_predict_mixed,
            models, dummy_seq, DEVICE,
            cfg.spatial.proximity_dim,
            cfg.frame_features.input_sequence_dim,
            cfg.frame_features.proximity_index,
        ),
    )
    logger.info("warmup_completed", extra={"event": "model_ready"})

    # Store in app state
    app.state.models = models
    app.state.classes = classes
    app.state.num_classes = num_classes
    app.state.model_loaded = True
    app.state.sessions = {}  # Dict[str, InferenceSession]

    from concurrent.futures import ThreadPoolExecutor
    app.state.inference_executor = ThreadPoolExecutor(
        max_workers=max(1, min(4, _os.cpu_count() or 1)),
        thread_name_prefix="inference"
    )

    # ── Load Adapter (if exists) ──────────────────────────────────────────────
    adapter_weights_dir = ADAPTER_WEIGHTS_DIR
    app.state.adapter_model = None
    if _os.path.exists(adapter_weights_dir):
        pt_files = sorted([f for f in _os.listdir(adapter_weights_dir) if f.endswith(".pt")])
        if pt_files:
            latest_adapter_path = _os.path.join(adapter_weights_dir, pt_files[-1])
            try:
                adapter_model = AdapterModel(num_classes, hidden_dim=cfg.training.adapter_hidden_dim).to(DEVICE)
                adapter_model.load_weights(latest_adapter_path)
                adapter_model.eval()
                app.state.adapter_model = adapter_model
                logger.info("adapter_loaded", extra={"path": latest_adapter_path})
            except Exception as e:
                logger.warning(f"Failed to load adapter weights from {latest_adapter_path}: {e}")

    # ── Emergency detector — loaded from data/emergency_config.json ──────────
    app.state.emergency_detector = EmergencyDetector.from_config()
    logger.info("emergency_detector_ready", extra={"event": "emergency_ready"})

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    app.state.sessions.clear()
    if hasattr(app.state, "inference_executor"):
        app.state.inference_executor.shutdown(wait=False)
    logger.info("shutdown_completed", extra={"event": "shutdown"})


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

# ── CORS — production-safe ───────────────────────────────────────────────────
# In development mode: allow all origins to prevent local CORS issues.
# In production: read ALLOWED_ORIGINS from environment. If unset, defaults to
# an empty list — no cross-origin requests are permitted.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
if ENV == "development":
    _allowed_origins: list[str] = ["*"]
elif _raw_origins:
    _allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
else:
    _allowed_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Status"])
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
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics", tags=["Status"])
async def metrics() -> dict:
    """
    Exposes runtime observability metrics.
    Keeps collection lightweight and in-memory.
    """
    from api.feedback import FEEDBACK_STATE

    uptime = int(time.time() - START_TIME)
    active_sessions = getattr(app.state, "sessions", {})
    
    latencies = list(LATENCY_HISTORY)
    inf_times = list(INFERENCE_TIME_HISTORY)
    if latencies:
        arr = np.array(latencies)
        avg_latency = float(np.mean(arr))
        avg_ms = round(avg_latency, 2)
        p50 = round(float(np.percentile(arr, 50)), 2)
        p95 = round(float(np.percentile(arr, 95)), 2)
        p99 = round(float(np.percentile(arr, 99)), 2)
    else:
        avg_latency = avg_ms = p50 = p95 = p99 = 0.0

    if inf_times:
        arr_inf = np.array(inf_times)
        avg_inf_time = float(np.mean(arr_inf))
    else:
        avg_inf_time = 0.0

    avg_queue_wait = max(0.0, avg_latency - avg_inf_time)

    mem_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    cpu = process.cpu_percent(interval=None)
    fps = TOTAL_PREDICTIONS / uptime if uptime > 0 else 0.0

    # Read model version if available
    model_version = "unknown"
    metadata_path = os.path.join("assets", "model_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                model_version = metadata.get("model_version", "unknown")
        except Exception:
            pass

    return {
        "uptime_seconds": uptime,
        "active_sessions": len(active_sessions),
        "total_predictions": TOTAL_PREDICTIONS,
        "dropped_frames": DROPPED_FRAMES,
        "fps_processed": round(fps, 2),
        "avg_latency_ms": avg_ms,
        "avg_inference_time_ms": round(avg_inf_time, 2),
        "avg_queue_wait_ms": round(avg_queue_wait, 2),
        "p50_inference_ms": p50,
        "p95_inference_ms": p95,
        "p99_inference_ms": p99,
        "process_memory_mb": mem_mb,
        "cpu_percent": cpu,
        "model_version": model_version,
        "api_version": "v1",
        "current_model": f"Ensemble ({len(getattr(app.state, 'models', []))} models)",
        "active_adapter_version": FEEDBACK_STATE.get("active_adapter_version"),
        "total_feedback_received": FEEDBACK_STATE.get("total_feedback_received", 0),
        "training_in_progress": FEEDBACK_STATE.get("training_in_progress", False),
        "pending_feedback_samples": FEEDBACK_STATE.get("pending_feedback_count", 0),
        "adapter_success_runs": FEEDBACK_STATE.get("success_runs", 0),
        "adapter_failed_runs": FEEDBACK_STATE.get("failed_runs", 0),
    }


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
    t0 = time.perf_counter()
    response_dict, _, _, _ = await loop.run_in_executor(
        None,
        partial(run_predict, app.state.models, sequence, app.state.classes, DEBUG),
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    global TOTAL_PREDICTIONS
    TOTAL_PREDICTIONS += 1
    LATENCY_HISTORY.append(latency_ms)

    logger.info(
        "predict_stateless_completed",
        extra={
            "latency_ms": latency_ms,
            "confidence": response_dict.get("confidence", 0.0)
        }
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
    if not dimension_check:
        difference = -1.0
        valid = False
    else:
        incoming = np.array(request.features[:253], dtype=np.float32)
        difference = float(np.mean(np.abs(reference - incoming)))
        
        mae_tolerance = 1e-5
        valid = difference < mae_tolerance and range_check

        if not valid and difference >= mae_tolerance:
            errors.append(f"MAE {difference:.6f} exceeds tolerance {mae_tolerance}")

    return {
        "valid": valid,
        "mae": difference,
        "dimension_check": dimension_check,
        "range_check": range_check,
        "errors": list(errors)
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /feedback
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/feedback", status_code=202)
async def submit_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit a corrected sequence from the frontend.
    This sequence is saved to the pseudo_data directory.
    A background task is spawned to re-train the active learning adapter.
    """
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded yet")
        
    correct_word = req.correct_word.lower()
    if correct_word not in app.state.classes:
        raise HTTPException(status_code=400, detail=f"Unknown class: {correct_word}")
        
    sequence_length = len(req.sequence)
    if sequence_length != NUM_FRAMES:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {NUM_FRAMES} frames, got {sequence_length}"
        )
        
    # Dispatch the I/O and training work to the background
    background_tasks.add_task(
        process_feedback_async,
        app,
        req.sequence,
        correct_word,
        req.session_id
    )
    
    return {"status": "accepted", "message": f"Feedback for '{correct_word}' received. Training adapter in background."}




# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET TRANSLATE ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
def _run_inference_with_adapter(models, sequence, device, prox_dim, seq_dim, prox_idx, adapter):
    import time
    t_start = time.perf_counter()
    pred_idx, confidence, all_probs = ensemble_predict_mixed(
        models, sequence, device, prox_dim, seq_dim, prox_idx
    )
    if adapter is not None:
        import torch
        import torch.nn.functional as F
        import numpy as np
        with torch.no_grad():
            probs_tensor = torch.from_numpy(all_probs.astype(np.float32)).unsqueeze(0).to(device)
            adapted_logits = adapter(probs_tensor)
            adapted_probs = F.softmax(adapted_logits, dim=1).cpu().numpy()[0]
            all_probs = adapted_probs
            pred_idx = int(np.argmax(all_probs))
            confidence = float(all_probs[pred_idx])
            
    t_end = time.perf_counter()
    return pred_idx, confidence, all_probs, (t_end - t_start) * 1000.0


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
    session = create_session(NUM_FRAMES, INPUT_SIZE, emergency_config=app.state.emergency_detector)
    session_id = session.session_id
    websocket.state.session_id = session_id
    app.state.sessions[session_id] = session

    short_id = session_id[:8]
    logger.info(
        "websocket_connected",
        extra={
            "session_id": short_id,
            "active_sessions": len(app.state.sessions)
        }
    )

    loop = asyncio.get_running_loop()
    
    global TOTAL_PREDICTIONS, DROPPED_FRAMES, ACTIVE_SESSIONS
    ACTIVE_SESSIONS += 1

    try:
        while True:
            raw = await websocket.receive_text()

            if len(raw) > MAX_PAYLOAD_SIZE:
                raise ValueError(f"Payload too large: {len(raw)} bytes")

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
                if session.pending_count > MAX_PENDING:
                    DROPPED_FRAMES += 1
                    continue

                # ── Append to circular buffer (zero-copy) ─────────────────────
                frame = np.array(features, dtype=np.float32)

                if not np.isfinite(frame).all():
                    logger.warning("websocket_invalid_frame", extra={"session_id": short_id, "error": "NaN or Infinity detected"})
                    continue

                # ── Skeleton Quality Gate ─────────────────────────────────────
                # 1. Zero-ratio check: >50% of raw hand coordinates are zero
                #    means MediaPipe did not detect either hand reliably.
                raw_coords = frame[:126]
                zero_ratio = float(np.mean(np.abs(raw_coords) < 1e-6))
                if zero_ratio > LANDMARK_ZERO_RATIO_THRESHOLD:
                    logger.debug("skeleton_quality_zero", extra={"session_id": short_id, "zero_ratio": round(zero_ratio, 3)})
                    continue

                # 2. Landmark jump check: sudden large coordinate shift between
                #    consecutive frames usually indicates a MediaPipe tracking glitch.
                #    Uses session-level counter so 1-2 glitchy frames don't interrupt
                #    a running sign (same pattern as idle_frames).
                #    IMPORTANT: prev_frame is only updated on ACCEPTED frames so that
                #    each bad frame is compared against the last known-good position,
                #    not against the previous bad frame (which would mask continued drift).
                if session.prev_frame is not None:
                    jump = float(np.mean(np.abs(raw_coords - session.prev_frame[:126])))
                    if jump > LANDMARK_JUMP_THRESHOLD:
                        session.landmark_jump_count += 1
                        if session.landmark_jump_count >= MAX_CONSECUTIVE_JUMPS:
                            logger.warning("skeleton_quality_jump", extra={"session_id": short_id, "jump": round(jump, 4), "consecutive": session.landmark_jump_count})
                            session.landmark_jump_count = 0
                        continue  # prev_frame intentionally NOT updated here
                    else:
                        session.landmark_jump_count = 0
                session.prev_frame = frame.copy()  # Only reached on accepted frames

                session.append_frame(frame)

                sequence = session.get_sequence()
                if sequence is None:
                    continue  # Still filling — not enough frames yet

                # ── Motion-Triggered Inference ────────────────────────────────
                # Features 253-505 are velocity (frame-to-frame delta)
                velocity = frame[253:506]
                vel_sum = float(np.sum(np.abs(velocity)))
                MOTION_SCORE_HISTORY.append(vel_sum)
                
                if vel_sum < IDLE_VELOCITY_THRESHOLD:  # IDLE_THRESHOLD
                    session.idle_frames += 1
                else:
                    session.idle_frames = 0
                
                if session.idle_frames >= 3:
                    # Feed explicit idle state
                    idle_class_idx = app.state.classes.index("...") if "..." in app.state.classes else 0
                    idle_probs = np.zeros(app.state.num_classes, dtype=np.float32)
                    idle_probs[idle_class_idx] = 1.0
                    
                    stable_class, smoothed_conf = session.postprocessor.update_with_confidence(idle_probs)
                    if stable_class is None:
                        continue
                    word = app.state.classes[stable_class]
                    session.sentence_builder.update(word, smoothed_conf)
                    
                    response = {
                        "type": "prediction",
                        "word": word.upper() if smoothed_conf >= CONFIDENCE_THRESHOLD else None,
                        "confidence": round(float(smoothed_conf), 4),
                        "sentence_so_far": session.sentence_builder.current_sentence,
                    }
                    if DEBUG:
                        response["debug"] = {
                            "top5": [], "raw_confidence": 1.0, 
                            "stable_class": int(stable_class), "pending_count": session.pending_count,
                            "idle": True
                        }
                    await websocket.send_json(response)
                    continue

                session.pending_count += 1

                # Blocking inference — run in dedicated thread pool
                try:
                    t0 = time.perf_counter()
                    active_adapter = getattr(app.state, "adapter_model", None)
                    
                    pred_idx, confidence, all_probs, inf_time_ms = await loop.run_in_executor(
                        app.state.inference_executor,
                        partial(
                            _run_inference_with_adapter,
                            app.state.models,
                            sequence,
                            DEVICE,
                            cfg.spatial.proximity_dim,
                            cfg.frame_features.input_sequence_dim,
                            cfg.frame_features.proximity_index,
                            active_adapter
                        ),
                    )
                    t1 = time.perf_counter()
                    latency_ms = (t1 - t0) * 1000.0

                    TOTAL_PREDICTIONS += 1
                    INFERENCE_TIME_HISTORY.append(inf_time_ms)
                    LATENCY_HISTORY.append(latency_ms)

                    logger.info(
                        "prediction_completed",
                        extra={
                            "session_id": short_id,
                            "latency_ms": round(latency_ms, 2),
                            "confidence": round(float(confidence), 4),
                            "word": app.state.classes[pred_idx]
                        }
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

                if logger.isEnabledFor(logging.DEBUG):
                    locked_idx = getattr(session.postprocessor.predictor, "current_class", None)
                    locked_word = app.state.classes[locked_idx] if locked_idx is not None else "None"
                    
                    rounded_seq = np.round(np.array(sequence), decimals=3)
                    sequence_hash = hashlib.md5(rounded_seq.tobytes()).hexdigest()[:8]
                    
                    frame_delta = None
                    if len(sequence) >= 2:
                        frame_delta = float(np.mean(np.abs(np.array(sequence[-1]) - np.array(sequence[-2]))))

                    raw_word = app.state.classes[pred_idx]
                    logger.debug("temporal_debug", extra={
                        "raw_prediction": raw_word,
                        "raw_confidence": float(confidence),
                        "stable_prediction": word,
                        "stable_confidence": float(smoothed_conf),
                        "current_locked_word": locked_word,
                        "current_locked_conf": float(session.postprocessor.predictor.current_confidence),
                        "sequence_hash": sequence_hash,
                        "frame_delta": frame_delta
                    })

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

                # ── Emergency detection — after temporal smoothing ─────────────
                # check() is synchronous (no I/O) — edge detection + cooldown
                # dispatch_as_task() is fire-and-forget with task tracking —
                # exceptions are caught and logged, never silently dropped.
                emergency_word = word if smoothed_conf >= CONFIDENCE_THRESHOLD else None
                alert = session.emergency.check(emergency_word, smoothed_conf)
                if alert:
                    await websocket.send_json(alert)
                    session.emergency.dispatch_as_task(word, smoothed_conf)

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
                logger.info(
                    "session_stopped",
                    extra={
                        "session_id": short_id,
                        "sentence": final_text,
                        "word_count": len(words_snapshot)
                    }
                )

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
        logger.info("websocket_disconnected", extra={"session_id": short_id})
    except ValueError as exc:
        logger.warning("websocket_validation_error", extra={"session_id": short_id, "error": str(exc)})
        try:
            await websocket.close(code=1008, reason="Policy Violation: Invalid Data")
        except Exception:
            pass
    except Exception as exc:
        logger.error("websocket_error", extra={"session_id": short_id, "error": str(exc)})
        try:
            await websocket.close(code=1011, reason="Internal Error")
        except Exception:
            pass
    finally:
        ACTIVE_SESSIONS = max(0, ACTIVE_SESSIONS - 1)
        app.state.sessions.pop(session_id, None)
        logger.info(
            "session_cleanup",
            extra={
                "session_id": short_id,
                "active_sessions": len(app.state.sessions)
            }
        )
