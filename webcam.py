"""Live webcam ISL recognition with continuous automatic translation.

Automatically translates sign sequences into sentences in real-time.
No keyboard input needed - signs are recognized and sentences build automatically.
Sentences auto-complete after ~2 seconds of no new signs.

Controls:
    Q/ESC  - Quit (only control needed)

════════════════════════════════════════════════════════════════════════════════════
PHASE 3: LIVE INFERENCE OPTIMIZATION
════════════════════════════════════════════════════════════════════════════════════
- Dynamic ensemble size (1, 3, or 5 models via LIVE_ENSEMBLE_SIZE)
- Configurable TTA (disabled by default for better latency via LIVE_USE_TTA)
- Optional latency reporting via PRINT_LATENCY_STATS
- All settings controlled through config.live_inference
"""

import cv2
import numpy as np
from collections import Counter, deque
import mediapipe as mp
import torch
import os
import time
import threading

from profiling import get_profiler, profile_section, start_frame, end_frame, record_inference

from config import get_config
from pseudo_buffer import PseudoLabelBuffer
from adapter_model import AdapterModel, AdapterTrainer
from adapter_training import AdapterTrainingManager
from onnx_ensemble_integration import (
    load_ensemble_with_onnx,
    ensemble_predict_with_onnx,
    check_onnx_models_available,
)

cfg = get_config()

# Convenience references for webcam inference
NUM_FRAMES = cfg.preprocessing.num_frames
NUM_HANDS = cfg.landmarks.num_hands
DEBUG_DRAW_FACE_CENTER = cfg.preprocessing.debug_draw_face_center
USE_VELOCITY = cfg.frame_features.use_velocity
CONFIDENCE_THRESHOLD = cfg.inference.confidence_threshold
PREDICTION_SMOOTHING_WINDOW = cfg.inference.prediction_smoothing_window
MOTION_GATING_ENABLED = cfg.motion.enabled
MOTION_THRESHOLD = cfg.get_motion_threshold_pixels()
MOTION_SMOOTHING = cfg.motion.motion_smoothing
IDLE_CONFIDENCE_THRESHOLD = cfg.motion.idle_confidence_threshold
DYNAMIC_THRESHOLD_ENABLED = cfg.motion.dynamic_threshold_enabled
MOTION_BOOST_FACTOR = cfg.motion.motion_boost_factor
STABILITY_BOOST_FACTOR = cfg.motion.stability_boost_factor
DYNAMIC_THRESHOLD_MIN = cfg.motion.dynamic_threshold_min
TRANSITION_HYSTERESIS = cfg.inference.transition_hysteresis
AMBIGUITY_MARGIN_THRESHOLD = cfg.inference.ambiguity_margin_threshold
AMBIGUITY_DELAY_FRAMES = cfg.inference.ambiguity_delay_frames
TRANSITION_MOVEMENT_THRESHOLD = 0.02
TRANSITION_STABLE_FRAMES = 3

# ════════════════════════════════════════════════════════════════════════════════════
# PHASE 3: Live inference optimization configuration
# ════════════════════════════════════════════════════════════════════════════════════
LIVE_ENSEMBLE_SIZE = cfg.live_inference.ensemble_size
LIVE_USE_TTA = cfg.live_inference.use_tta
PRINT_LATENCY_STATS = cfg.live_inference.print_latency_stats
LIVE_TEMPORAL_SMOOTHING_ENABLED = cfg.live_inference.temporal_smoothing_enabled
LIVE_TEMPORAL_WINDOW = cfg.live_inference.temporal_window_size
LIVE_TEMPORAL_PATIENCE = cfg.live_inference.temporal_patience
LIVE_TEMPORAL_DELTA = cfg.live_inference.temporal_delta
LIVE_TEMPORAL_DECAY = cfg.live_inference.temporal_decay_factor
HAND_FORCE_REDETECT_INTERVAL = cfg.preprocessing.forced_hand_redetect_interval
HAND_DRIFT_JUMP_RATIO = 0.08
HAND_DRIFT_MIN_JUMP_PX = 40

# ── Pseudo-Label Collection (PART A) ──
PSEUDO_BUFFER_ENABLED = True  # Toggle pseudo-label collection
PSEUDO_THRESHOLD = 0.85  # Minimum confidence to collect
MIN_BUFFER_SIZE = 20  # Minimum samples before auto-save
PER_CLASS_CAP = 50  # Max samples per class
PSEUDO_SAVE_DIR = "pseudo_data/"
AUTO_SAVE_PSEUDO = True  # Auto-save when MIN_BUFFER_SIZE is reached
PSEUDO_SAVE_INTERVAL = 50  # Save every N predictions

# ── Adapter Model (PART B) ──
ADAPTER_ENABLED = True  # Toggle adaptive learning (enabled with safety checks)
ADAPTER_LEARNING_RATE = 1e-4
ADAPTER_EPOCHS = 10
ADAPTER_BATCH_SIZE = 8
ADAPTER_HIDDEN_DIM = 128
ADAPTER_TRAIN_MIN_SAMPLES = cfg.live_inference.adapter_train_min_samples
ADAPTER_MIN_CLASSES = cfg.live_inference.adapter_min_classes
ADAPTER_MIN_SAMPLES_PER_CLASS = cfg.live_inference.adapter_min_samples_per_class
ADAPTER_WEIGHTS_DIR = "adapter_weights/"
ADAPTER_TRAINING_INTERVAL = cfg.live_inference.adapter_training_interval
ADAPTER_POLL_INTERVAL = 50  # Poll adapter_weights/ for new models every N predictions
ADAPTER_MIN_SAVED_SAMPLES = cfg.live_inference.adapter_min_saved_samples

# Device for adapter
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Adapter] Using device: {DEVICE}")

from preprocess import (
    _normalize_landmarks,
    _add_velocity,
    create_landmarker,
    create_face_landmarker,
    extract_landmarks_with_face_relative,
)
from sentence_builder import SentenceBuilder
from temporal_postprocessor import TemporalPostProcessor
from hand_selector import HandSelector


# ── Hand landmark drawing connections ──
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
BLUE = (255, 120, 0)
ORANGE = (0, 165, 255)


def _landmarks_to_numpy(landmarks) -> np.ndarray:
    """Convert MediaPipe landmarks to numpy array (N, 2 or 3 dimensions)."""
    if landmarks is None:
        return None
    return np.array([[lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0.0] for lm in landmarks], dtype=np.float32)


def _bbox_iou(a, b):
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _landmarks_to_bbox(hand_landmarks, w, h, pad=14):
    """Convert normalized hand landmarks to a padded pixel bbox."""
    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w - 1, max(xs) + pad)
    y2 = min(h - 1, max(ys) + pad)
    return x1, y1, x2, y2


def _bbox_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_contains_point(box, pt):
    x1, y1, x2, y2 = box
    px, py = pt
    return x1 <= px <= x2 and y1 <= py <= y2


def _wrist_point_px(hand_landmarks, w, h):
    """Return wrist landmark (id=0) in pixels."""
    wrist = hand_landmarks[0]
    return int(wrist.x * w), int(wrist.y * h)


def _calculate_hand_motion(wrist_pos, wrist_history, motion_magnitude):
    """Calculate exponential moving average of hand motion velocity.
    
    Args:
        wrist_pos: Current wrist position (x, y)
        wrist_history: Deque of recent wrist positions
        motion_magnitude: Previous motion EMA value
    
    Returns:
        Updated motion magnitude (normalized motion velocity)
    """
    if not wrist_history:
        return motion_magnitude
    
    last_pos = wrist_history[-1]
    dx = wrist_pos[0] - last_pos[0]
    dy = wrist_pos[1] - last_pos[1]
    current_motion = (dx**2 + dy**2)**0.5
    
    # Exponential moving average
    new_motion = MOTION_SMOOTHING * current_motion + (1 - MOTION_SMOOTHING) * motion_magnitude
    return new_motion


def _detect_hand_drift(hand_infos, wrist_history, frame_shape):
    """Detect suspicious wrist jumps that usually indicate stale tracking."""
    if not hand_infos or len(wrist_history) == 0:
        return False

    current_wrists = []
    for info in hand_infos:
        landmarks = info.get("landmarks")
        if landmarks:
            current_wrists.append(_wrist_point_px(landmarks, frame_shape[1], frame_shape[0]))

    if not current_wrists:
        return False

    current_center = (
        sum(point[0] for point in current_wrists) / len(current_wrists),
        sum(point[1] for point in current_wrists) / len(current_wrists),
    )
    previous_center = wrist_history[-1]
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    jump = (dx * dx + dy * dy) ** 0.5

    frame_diag = (frame_shape[1] ** 2 + frame_shape[0] ** 2) ** 0.5
    jump_threshold = max(HAND_DRIFT_MIN_JUMP_PX, frame_diag * HAND_DRIFT_JUMP_RATIO)
    return jump > jump_threshold


def _calculate_dynamic_threshold(motion_magnitude, stability_counter, is_transition):
    """Calculate adaptive confidence threshold based on motion and stability.
    
    Args:
        motion_magnitude: Current hand motion velocity
        stability_counter: Frames stable at current prediction
        is_transition: Whether in transition detection phase
    
    Returns:
        Effective confidence threshold to use
    """
    if not DYNAMIC_THRESHOLD_ENABLED:
        return CONFIDENCE_THRESHOLD
    
    threshold = CONFIDENCE_THRESHOLD
    
    # Boost threshold temporarily during transitions (require high confidence)
    if is_transition:
        threshold += TRANSITION_HYSTERESIS
    
    # Reduce threshold when motion is detected (high motion = easier to detect)
    if motion_magnitude > MOTION_THRESHOLD:
        motion_ratio = min(motion_magnitude / (MOTION_THRESHOLD * 2), 1.0)
        threshold -= MOTION_BOOST_FACTOR * motion_ratio
    
    # Reduce threshold as sign becomes more stable
    if stability_counter > 2:
        stability_ratio = min(stability_counter / 8.0, 1.0)
        threshold -= STABILITY_BOOST_FACTOR * stability_ratio
    
    # Floor to minimum threshold
    return max(threshold, DYNAMIC_THRESHOLD_MIN)


def _is_motion_gating_active(motion_magnitude, frames_in_motion):
    """Determine if we should gate (suppress) predictions based on motion.
    
    Args:
        motion_magnitude: Current hand motion EMA
        frames_in_motion: Consecutive frames with motion
    
    Returns:
        True if should suppress predictions (no motion detected)
    """
    if not MOTION_GATING_ENABLED:
        return False
    
    # Consider motion active if recent motion magnitude exceeds threshold
    # OR if we've seen motion recently (momentum)
    has_current_motion = motion_magnitude > MOTION_THRESHOLD
    has_recent_motion = frames_in_motion > 0
    
    # Gate (suppress) when NO motion at all
    return not (has_current_motion or has_recent_motion)


def _compute_transition_movement(current_landmarks, previous_landmarks):
    """Compute mean absolute frame-to-frame landmark movement."""
    if previous_landmarks is None:
        return 0.0
    return float(np.mean(np.abs(current_landmarks - previous_landmarks)))


def _log_adapter_skip(pipeline_log, trigger_label: str, reason: str, **details):
    print(f"[Adapter] Skipping training ({trigger_label}): {reason}")
    if pipeline_log is not None:
        pipeline_log.event(
            "adapter_training_skipped",
            trigger=trigger_label,
            reason=reason,
            **details,
        )


def _compute_adaptive_thresholds(elapsed_seconds: float, base_interval: int, base_min_saved: int):
    """
    Auto-tune adapter training thresholds based on session duration.
    
    If session is short (< 2 min), aggressively lower thresholds to enable training before exit.
    If session is long, use normal thresholds.
    
    Args:
        elapsed_seconds: Time since session start
        base_interval: Normal training interval
        base_min_saved: Normal minimum saved samples
    
    Returns:
        (adaptive_interval, adaptive_min_saved, is_short_session)
    """
    is_short_session = elapsed_seconds < 120  # < 2 minutes
    
    if not is_short_session:
        return base_interval, base_min_saved, False
    
    # For short sessions, reduce thresholds based on elapsed time
    # At 30 sec: interval=50, min_saved=20
    # At 60 sec: interval=75, min_saved=30
    # At 120 sec: return to normal
    
    progress_ratio = elapsed_seconds / 120.0  # 0 to 1
    
    # Interval: from 50 to base_interval
    adaptive_interval = int(50 + (base_interval - 50) * progress_ratio)
    
    # Min saved: from 20 to base_min_saved
    adaptive_min_saved = int(20 + (base_min_saved - 20) * progress_ratio)
    
    return adaptive_interval, adaptive_min_saved, True


def _get_adaptive_hand_detection_interval(motion_magnitude: float) -> int:
    """Return hand detection interval that expands only when motion is low."""
    base_interval = max(1, int(cfg.preprocessing.hand_detection_interval))
    if not cfg.preprocessing.adaptive_hand_interval_enabled:
        return base_interval

    threshold = MOTION_THRESHOLD * float(cfg.preprocessing.low_motion_threshold_ratio)
    if motion_magnitude > threshold:
        return base_interval

    expanded = int(round(base_interval * float(cfg.preprocessing.low_motion_interval_multiplier)))
    return max(base_interval, min(expanded, int(cfg.preprocessing.max_adaptive_hand_interval)))



def _detect_person_boxes(frame, hog_detector):
    """Detect person boxes and apply a lightweight NMS by IoU."""
    rects, weights = hog_detector.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    candidates = []
    for (x, y, w, h), conf in zip(rects, weights):
        if conf < 0.3:
            continue
        candidates.append((x, y, x + w, y + h, float(conf)))

    candidates.sort(key=lambda t: t[4], reverse=True)
    kept = []
    for cand in candidates:
        cbox = cand[:4]
        if any(_bbox_iou(cbox, k[:4]) > 0.45 for k in kept):
            continue
        kept.append(cand)

    return kept


def _assign_hand_to_person(hand_box, person_boxes):
    """Assign hand box to a person by containment-first then nearest center."""
    if len(person_boxes) == 1:
        # If only one person is visible, assign all hands to that person.
        return 0

    hand_center = _bbox_center(hand_box)

    containing = []
    for idx, p in enumerate(person_boxes):
        pbox = p[:4]
        if _box_contains_point(pbox, hand_center):
            containing.append((idx, pbox))

    candidates = containing if containing else [
        (idx, p[:4]) for idx, p in enumerate(person_boxes)
    ]
    if not candidates:
        return None

    hx, hy = hand_center
    best_idx = None
    best_dist = float("inf")
    for idx, pbox in candidates:
        px, py = _bbox_center(pbox)
        dist = (hx - px) ** 2 + (hy - py) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    return best_idx


def _draw_landmarks(frame, hand_landmarks, w, h):
    """Draw hand landmarks and connections on the frame."""
    points = []
    for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 4, GREEN, -1)

    for i, j in HAND_CONNECTIONS:
        if i < len(points) and j < len(points):
            cv2.line(
                frame, points[i], points[j], (0, 200, 0), 2
            )


def _extract_frame_landmarks(
    landmarker,
    force_landmarker,
    holistic,
    frame,
    hand_cache,
    face_cache,
    frame_idx,
    frame_timestamp_ms=None,
    hand_detect_interval=None,
    face_detect_interval=None,
    force_detection=False,
):
    """
    Extract frame vector with shared preprocess feature logic.

    Optimized for real-time: face/hand detection can run every N frames
    with cached reuse between intervals.
    
    Args:
        hand_detect_interval: Run full hand detection every N frames (None = use config default)
        face_detect_interval: Run face detection every N frames (None = use config default)
    """
    if hand_detect_interval is None:
        hand_detect_interval = cfg.preprocessing.hand_detection_interval

    # Use config default if not provided
    if face_detect_interval is None:
        face_detect_interval = cfg.preprocessing.face_detection_interval
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def _next_video_timestamp_ms(cache_dict: dict, cache_key: str, ts_ms: int | None) -> int | None:
        """Return a strictly increasing timestamp for MediaPipe VIDEO mode."""
        if ts_ms is None:
            return None

        current_ts = int(ts_ms)
        last_ts = cache_dict.get(cache_key)
        if last_ts is not None and current_ts <= int(last_ts):
            current_ts = int(last_ts) + 1
        cache_dict[cache_key] = current_ts
        return current_ts

    # Hand detection every N frames; reuse cached result between detects.
    # To avoid visibly "stuck" landmarks during subtle hand motion, we cap
    # cache age more aggressively while hands are currently tracked.
    cached_result = hand_cache.get('result')
    cached_frame_idx = int(hand_cache.get('frame_idx', -10**9))
    cache_age_frames = frame_idx - cached_frame_idx
    has_cached_hands = bool(
        cached_result is not None
        and getattr(cached_result, "hand_landmarks", None)
    )

    # When hands are visible, refresh often for responsiveness.
    # When no hands are tracked, keep the normal interval for efficiency.
    max_cache_age_frames = 2 if has_cached_hands else max(1, int(hand_detect_interval))

    should_detect_hands = (
        force_detection
        or frame_idx % max(1, int(HAND_FORCE_REDETECT_INTERVAL)) == 0
        or frame_idx % max(1, int(hand_detect_interval)) == 0
        or cached_result is None
        or cache_age_frames >= max_cache_age_frames
    )
    if should_detect_hands:
        if force_detection:
            if force_landmarker is not None:
                result = force_landmarker.detect(mp_image)
            else:
                result = landmarker.detect(mp_image)
        elif frame_timestamp_ms is not None and hasattr(landmarker, "detect_for_video"):
            hand_ts_ms = _next_video_timestamp_ms(
                hand_cache,
                "last_video_timestamp_ms",
                frame_timestamp_ms,
            )
            result = landmarker.detect_for_video(mp_image, hand_ts_ms)
        else:
            result = landmarker.detect(mp_image)
        hand_cache['result'] = result
        hand_cache['frame_idx'] = frame_idx
    else:
        result = hand_cache.get('result')

    # Face detection every N frames to speed up real-time inference
    face_landmarks = None
    if holistic is not None and frame_idx % face_detect_interval == 0:
        if frame_timestamp_ms is not None and hasattr(holistic, "detect_for_video"):
            face_ts_ms = _next_video_timestamp_ms(
                face_cache,
                "last_video_timestamp_ms",
                frame_timestamp_ms,
            )
            face_result = holistic.detect_for_video(mp_image, face_ts_ms)
        else:
            face_result = holistic.detect(mp_image)
        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            face_cache['landmarks'] = face_landmarks
            face_cache['frame_idx'] = frame_idx
    else:
        # Reuse cached face landmarks from recent frames
        face_landmarks = face_cache.get('landmarks')

    landmarks_vec = extract_landmarks_with_face_relative(
        frame=frame,
        hand_result=result,
        face_landmarks=face_landmarks,
    )

    hand_infos = []
    for hand, handedness_list in zip(
        result.hand_landmarks,
        result.handedness,
    ):
        label = handedness_list[0].display_name  # "Right" or "Left"

        hand_infos.append({
            "label": label,
            "landmarks": hand,
        })

    face_center = None
    if face_landmarks is not None:
        nose = face_landmarks[1]
        face_center = (
            int(nose.x * frame.shape[1]),
            int(nose.y * frame.shape[0]),
        )

    return landmarks_vec, hand_infos, face_center, face_landmarks


def run_webcam(pipeline_log=None, model_artifact_path: str | None = None):
    """
        Main webcam loop for continuous word recognition.

        Pipeline per frame:
            1) Extract landmarks.
            2) Append to rolling window (size = NUM_FRAMES).
            3) Once full, normalize exactly like training + optional velocity.
            4) Predict and smooth with majority vote over recent predictions.
    """

    # ── Lazy model loading ──
    word_models = word_models_fallback = word_classes = None
    word_use_onnx = False

    def ensure_word_models():
        nonlocal word_models, word_models_fallback, word_classes, word_use_onnx
        if word_models is None:
            if model_artifact_path:
                print(f"Loading quantized model bundle: {model_artifact_path}")
                from ensemble import load_ensemble

                word_models, word_classes, _ = load_ensemble(model_artifact_path=model_artifact_path)
                word_models_fallback = []
            else:
                # Prefer ONNX ensemble if available
                try:
                    if check_onnx_models_available([cfg.paths.ensemble_dir, cfg.paths.base_dir]):
                        print("Loading ONNX ensemble...")
                        word_models, word_classes, _ = load_ensemble_with_onnx()
                        word_models_fallback = []
                        word_use_onnx = True
                    else:
                        print("Loading merged 10+2 ensemble...")
                        from ensemble import load_merged_ensemble_10_2
                        word_models, word_models_fallback, word_classes, _ = load_merged_ensemble_10_2()
                        word_use_onnx = False
                except Exception:
                    # Fallback to PyTorch ensemble loader
                    print("ONNX ensemble load failed, falling back to PyTorch ensemble")
                    from ensemble import load_merged_ensemble_10_2
                    word_models, word_models_fallback, word_classes, _ = load_merged_ensemble_10_2()
                    word_use_onnx = False
            if pipeline_log is not None:
                pipeline_log.event(
                    "ensemble_loaded",
                    mode="webcam",
                    main_models=len(word_models) if word_models else 0,
                    fallback_models=len(word_models_fallback) if word_models_fallback else 0,
                    classes=len(word_classes) if word_classes else 0,
                )
        return word_models, word_models_fallback, word_classes

    try:
        ensure_word_models()
    except FileNotFoundError:
        print("[WARN] No word model found  -- train first")
        if pipeline_log is not None:
            pipeline_log.event("ensemble_missing", mode="webcam")

    # ── Landmarker — optimized for webcam (high conf, face skipping) ──
    landmarker = create_landmarker(num_hands=NUM_HANDS, for_webcam=True)
    force_landmarker = create_landmarker(num_hands=NUM_HANDS, for_webcam=True, force_image_mode=True)
    holistic = create_face_landmarker(for_webcam=True)
    # Force-disable HOG person detection in live webcam mode to save CPU/time.
    # This avoids creating the HOG detector object and running detectMultiScale.
    HOG_ENABLED = False  # Set to True to re-enable HOG during live runs
    hog_detector = None
    if HOG_ENABLED and not cfg.preprocessing.disable_hog_detection:
        try:
            hog_detector = cv2.HOGDescriptor()
            hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception:
            hog_detector = None
    # Threaded camera capture to avoid blocking the main inference loop
    class CameraThread:
        def __init__(self, src=0):
            self.cap = cv2.VideoCapture(src)
            self.lock = threading.Lock()
            self.ret = False
            self.frame = None
            self.running = False
            if self.cap.isOpened():
                # Pre-warm a few frames to avoid the first-frame stall
                for _ in range(3):
                    self.cap.read()
                self.running = True
                self.thread = threading.Thread(target=self._reader, daemon=True)
                self.thread.start()

        def _reader(self):
            while self.running:
                ret, frame = self.cap.read()
                with self.lock:
                    self.ret, self.frame = ret, frame
                if not ret:
                    # Avoid tight loop if camera fails
                    time.sleep(0.01)

        def read(self):
            with self.lock:
                return self.ret, self.frame

        def isOpened(self):
            return self.cap.isOpened()

        def release(self):
            self.running = False
            try:
                if hasattr(self, 'thread'):
                    self.thread.join(timeout=1.0)
            except Exception:
                pass
            try:
                self.cap.release()
            except Exception:
                pass

    camera = CameraThread(0)
    if not camera.isOpened():
        print("[ERROR] Cannot open webcam.")
        if pipeline_log is not None:
            pipeline_log.event("webcam_open_failed")
        return
    # Preferred live resolution; can be reduced for performance
    camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State ──
    hand_cache = {}  # Cache hand landmarks between detection frames
    face_cache = {}  # Cache face landmarks between frames
    frame_idx = 0
    sequence_buffer = deque(maxlen=NUM_FRAMES)
    prediction_history = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)
    prediction_text = "Show a sign"
    confidence_text = ""
    prob_lines = []
    no_hand_frames = 0
    invalid_pair_frames = 0
    
    # ── Motion Tracking (for gating & dynamic thresholds) ──
    wrist_history = deque(maxlen=3)  # Recent wrist positions for motion calculation
    motion_magnitude = 0.0  # Exponential moving average of hand motion
    frames_in_motion = 0  # Consecutive frames with motion detected
    effective_threshold = CONFIDENCE_THRESHOLD  # Dynamically adjusted threshold
    last_output_prediction = None  # For transition hysteresis
    prediction_stability_counter = 0  # Frames with stable prediction
    previous_landmarks_vec = None
    transition_state = "STABLE"
    transition_stable_frames = 0
    transition_movement = 0.0
    
    # ── Sentence Builder (continuous translation) ──
    sentence_builder = SentenceBuilder(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        stability_frames=6,  # Reduced from 12 for faster response (~0.2s at 30fps)
        ambiguity_margin_threshold=AMBIGUITY_MARGIN_THRESHOLD,
        ambiguity_delay_frames=AMBIGUITY_DELAY_FRAMES,
        auto_sentence_timeout=75  # ~2.5 seconds at 30fps
    )
    last_displayed_word = None

    # ── Temporal Post-Processor (confidence-weighted smoothing + anti-flicker) ──
    from ensemble import load_merged_ensemble_10_2
    if LIVE_TEMPORAL_SMOOTHING_ENABLED:
        try:
            temporal_postprocessor = TemporalPostProcessor(
                window_size=LIVE_TEMPORAL_WINDOW,
                patience=LIVE_TEMPORAL_PATIENCE,
                delta=LIVE_TEMPORAL_DELTA,
                enable_decay=True,
                decay_factor=LIVE_TEMPORAL_DECAY,
            )
            temporal_postprocessor_enabled = True
        except Exception as e:
            print(f"[WARN] Could not initialize TemporalPostProcessor: {e}")
            temporal_postprocessor = None
            temporal_postprocessor_enabled = False
    else:
        temporal_postprocessor = None
        temporal_postprocessor_enabled = False

    # ── Prediction Momentum (majority + confidence commit) ──
    class PredictionMomentum:
        """Simple momentum buffer: commit when a class appears >= commit_count

        Keeps recent (idx, conf) tuples in a circular buffer and commits a
        prediction when majority agreement, average confidence, and minimum
        occurrences are satisfied.
        """
        def __init__(self, window: int = 5, commit_count: int = 3, min_avg_conf: float = 0.6):
            from collections import Counter

            self.window = window
            self.commit_count = commit_count
            self.min_avg_conf = min_avg_conf
            self._hist = deque(maxlen=window)

        def push(self, idx: int, conf: float) -> None:
            self._hist.append((int(idx), float(conf)))

        def get_commit(self):
            """Return (idx, avg_conf) if commit conditions met, else None."""
            if len(self._hist) < self.commit_count:
                return None
            counts = Counter([h[0] for h in self._hist])
            most, cnt = counts.most_common(1)[0]
            if cnt < self.commit_count:
                return None
            confs = [h[1] for h in self._hist if h[0] == most]
            avg_conf = sum(confs) / len(confs)
            if avg_conf < self.min_avg_conf:
                return None
            return int(most), float(avg_conf)

        def clear(self):
            self._hist.clear()

    # Initialize momentum buffer using config values
    pm_window = cfg.live_inference.momentum_window
    pm_commit = cfg.live_inference.momentum_commit_count
    pm_min_conf = cfg.live_inference.momentum_min_avg_conf
    prediction_momentum = PredictionMomentum(window=pm_window, commit_count=pm_commit, min_avg_conf=pm_min_conf)

    # ── Hand Selector (single-person hand filtering via MediaPipe face landmarks) ──
    hand_selector = HandSelector(
        distance_threshold=300,  # Pixel distance threshold for hand-to-face
        roi_width_ratio=0.5,  # 50% of frame width (centered at face)
        roi_height_ratio=0.5,  # 50% of frame height (centered at face)
        use_roi_filtering=True,  # Use ROI-based filtering (more reliable than pure distance)
        enable_debugging=False  # Set to True for debug logging
    )

    # ── PART A: Pseudo-Label Collection ──
    pseudo_buffer = None
    prediction_count_since_save = 0
    # Locks for background operations
    _pseudo_lock = threading.Lock()
    _adapter_lock = threading.Lock()
    _adapter_training_in_progress = False
    
    if PSEUDO_BUFFER_ENABLED:
        pseudo_buffer = PseudoLabelBuffer(
            save_dir=PSEUDO_SAVE_DIR,
            pseudo_threshold=PSEUDO_THRESHOLD,
            min_buffer=MIN_BUFFER_SIZE,
            per_class_cap=PER_CLASS_CAP,
            auto_save=AUTO_SAVE_PSEUDO,
        )

    def _background_save_buffer(buffer_copy, verbose=True):
        try:
            total_saved = 0
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for class_name, sequences in buffer_copy.items():
                class_dir = os.path.join(PSEUDO_SAVE_DIR, class_name)
                os.makedirs(class_dir, exist_ok=True)
                for idx, seq in enumerate(sequences):
                    filename = f"{class_name}_{timestamp}_{idx:03d}.npy"
                    filepath = os.path.join(class_dir, filename)
                    try:
                        np.save(filepath, seq)
                        total_saved += 1
                    except Exception:
                        continue

            if verbose:
                print(f"[PseudoBuffer] (bg) ✓ Saved {total_saved} samples to {PSEUDO_SAVE_DIR}")
            if pipeline_log is not None:
                pipeline_log.event(
                    "pseudo_buffer_saved_bg",
                    saved_samples=int(total_saved),
                    total_saved_on_disk=pseudo_buffer.get_saved_sample_count() if pseudo_buffer is not None else 0,
                )
        except Exception as e:
            print(f"[PseudoBuffer] Background save failed: {e}")

    def _background_adapter_training(trigger_label: str):
        nonlocal _adapter_training_in_progress
        # Prevent concurrent adapter training
        if not _adapter_lock.acquire(blocking=False):
            return
        try:
            if _adapter_training_in_progress:
                return
            _adapter_training_in_progress = True
            try:
                _attempt_adapter_training(trigger_label)
                if pipeline_log is not None:
                    pipeline_log.event("adapter_training_requested_bg", trigger=trigger_label)
            except Exception as e:
                print(f"[Adapter] Background training failed: {e}")
        finally:
            _adapter_training_in_progress = False
            try:
                _adapter_lock.release()
            except Exception:
                pass
    
    # ── PART B: Adapter Model ──
    adapter_model = None
    adapter_trainer = None
    adapter_manager = None
    prediction_count_since_training = 0
    session_start_time = time.time()  # Track session duration for auto-tuning
    
    if ADAPTER_ENABLED:
        try:
            # Get number of classes from models
            main_models, fallback_models, word_classes = ensure_word_models()
            num_classes = len(word_classes)
            
            # Create adapter
            adapter_model = AdapterModel(num_classes, hidden_dim=ADAPTER_HIDDEN_DIM).to(DEVICE)
            adapter_trainer = AdapterTrainer(
                num_classes=num_classes,
                device=DEVICE,
                learning_rate=ADAPTER_LEARNING_RATE,
                hidden_dim=ADAPTER_HIDDEN_DIM,
            )
            adapter_manager = AdapterTrainingManager(
                adapter_trainer=adapter_trainer,
                num_classes=num_classes,
                device=DEVICE,
                enable_adaptation=ADAPTER_ENABLED,
                adapter_weights_dir=ADAPTER_WEIGHTS_DIR,
            )
            
            # Load existing adapter weights if available
            latest_adapter = None
            if os.path.exists(ADAPTER_WEIGHTS_DIR):
                files = sorted([
                    f for f in os.listdir(ADAPTER_WEIGHTS_DIR)
                    if f.endswith('.pt')
                ])
                if files:
                    latest_adapter = os.path.join(ADAPTER_WEIGHTS_DIR, files[-1])
                    adapter_trainer.load_model(latest_adapter)
            
            print(f"[Adapter] Initialized: {num_classes} classes, {DEVICE}")
            if latest_adapter:
                print(f"[Adapter] Loaded weights from {latest_adapter}")
            if pipeline_log is not None:
                pipeline_log.event(
                    "adapter_initialized",
                    enabled=ADAPTER_ENABLED,
                    num_classes=num_classes,
                    device=str(DEVICE),
                    loaded_weights=latest_adapter,
                )
        
        except Exception as e:
            print(f"[WARN] Could not initialize adapter: {e}")
            adapter_model = None
            adapter_trainer = None
            adapter_manager = None

    def _attempt_adapter_training(trigger_label: str) -> bool:
        """Collect saved pseudo-data and start adapter training if safe."""
        if adapter_manager is None or adapter_trainer is None or word_classes is None:
            _log_adapter_skip(
                pipeline_log,
                trigger_label,
                "adapter not initialized",
            )
            return False

        if pseudo_buffer is None:
            _log_adapter_skip(
                pipeline_log,
                trigger_label,
                "pseudo-buffer is unavailable",
            )
            return False

        disk_sample_count = pseudo_buffer.get_saved_sample_count()
        
        # Compute adaptive min_saved based on trigger and session duration
        min_saved_threshold = ADAPTER_MIN_SAVED_SAMPLES
        if trigger_label in ("cleanup", "periodic check"):
            elapsed_secs = time.time() - session_start_time
            _, adaptive_min_saved, _ = _compute_adaptive_thresholds(
                elapsed_secs,
                ADAPTER_TRAINING_INTERVAL,
                ADAPTER_MIN_SAVED_SAMPLES,
            )
            min_saved_threshold = adaptive_min_saved
        
        if disk_sample_count < min_saved_threshold:
            _log_adapter_skip(
                pipeline_log,
                trigger_label,
                f"only {disk_sample_count} saved samples on disk",
                disk_samples=int(disk_sample_count),
                min_required=int(min_saved_threshold),
            )
            return False

        saved_samples = pseudo_buffer.load_saved_samples()
        if not saved_samples:
            _log_adapter_skip(
                pipeline_log,
                trigger_label,
                "no valid saved pseudo-data sequences could be loaded",
                disk_samples=int(disk_sample_count),
            )
            return False

        ensemble_probs_list = []
        class_indices_list = []

        for class_name, seq_stored in saved_samples:
            try:
                class_idx = word_classes.index(class_name)
            except ValueError:
                continue

            try:
                res = merged_ensemble_predict(
                    main_models, fallback_models, seq_stored, use_tta=LIVE_USE_TTA
                )
                probs_for_seq = np.array(res["probs"], dtype=np.float32)
                ensemble_probs_list.append(probs_for_seq)
                class_indices_list.append(class_idx)
            except Exception:
                continue

        if not ensemble_probs_list:
            _log_adapter_skip(
                pipeline_log,
                trigger_label,
                "no valid saved pseudo-data sequences could be scored",
                disk_samples=int(disk_sample_count),
            )
            return False

        class_id_to_name = {i: name for i, name in enumerate(word_classes)}

        per_class_indices = {}
        for idx, class_idx in enumerate(class_indices_list):
            per_class_indices.setdefault(class_idx, []).append(idx)

        train_mask = np.ones(len(ensemble_probs_list), dtype=bool)
        validation_indices = []
        eligible_class_count = 0

        for class_idx, indices in per_class_indices.items():
            class_count = len(indices)
            if class_count < ADAPTER_MIN_SAMPLES_PER_CLASS:
                continue

            eligible_class_count += 1
            holdout = min(
                max(1, class_count // 5),
                max(0, class_count - ADAPTER_MIN_SAMPLES_PER_CLASS),
            )
            if holdout <= 0:
                continue

            validation_indices.extend(indices[:holdout])
            for idx in indices[:holdout]:
                train_mask[idx] = False

        if eligible_class_count < ADAPTER_MIN_CLASSES:
            _log_adapter_skip(
                pipeline_log,
                trigger_label,
                f"only {eligible_class_count} classes reached the per-class minimum",
                class_count=int(eligible_class_count),
                min_required=int(ADAPTER_MIN_CLASSES),
                min_samples_per_class=int(ADAPTER_MIN_SAMPLES_PER_CLASS),
            )
            return False

        train_probs = [
            probs for i, probs in enumerate(ensemble_probs_list)
            if train_mask[i]
        ]
        train_targets = [
            class_idx for i, class_idx in enumerate(class_indices_list)
            if train_mask[i]
        ]
        validation_probs = (
            np.array([ensemble_probs_list[i] for i in validation_indices], dtype=np.float32)
            if validation_indices else None
        )

        if len(train_probs) < ADAPTER_TRAIN_MIN_SAMPLES:
            _log_adapter_skip(
                pipeline_log,
                trigger_label,
                f"only {len(train_probs)} balanced samples after holdout",
                train_samples=int(len(train_probs)),
                min_required=int(ADAPTER_TRAIN_MIN_SAMPLES),
                disk_samples=int(disk_sample_count),
                class_count=int(len(set(class_indices_list))),
            )
            return False

        started = adapter_manager.trigger_training_with_probs(
            ensemble_probs_list=train_probs,
            class_indices_list=train_targets,
            classes=word_classes,
            class_id_to_name=class_id_to_name,
            validation_probs=validation_probs,
            epochs=ADAPTER_EPOCHS,
            batch_size=ADAPTER_BATCH_SIZE,
            min_classes=ADAPTER_MIN_CLASSES,
            min_samples_per_class=ADAPTER_MIN_SAMPLES_PER_CLASS,
            use_class_weights=cfg.training.adapter_use_class_weights,
            class_weight_power=cfg.training.adapter_class_weight_power,
            class_weight_clip_min=cfg.training.adapter_class_weight_clip_min,
            class_weight_clip_max=cfg.training.adapter_class_weight_clip_max,
        )

        if not started:
            _log_adapter_skip(
                pipeline_log,
                trigger_label,
                "adapter manager rejected the request",
                train_samples=int(len(train_probs)),
                val_samples=int(len(validation_indices)),
                disk_samples=int(disk_sample_count),
            )
            return False

        print(
            f"[Adapter] Training requested from {trigger_label} "
            f"({len(train_probs)} train / {len(validation_indices)} val samples)"
        )
        if pipeline_log is not None:
            pipeline_log.event(
                "adapter_training_requested",
                trigger=trigger_label,
                raw_samples=int(len(ensemble_probs_list)),
                train_samples=int(len(train_probs)),
                val_samples=int(len(validation_indices)),
                disk_samples=int(disk_sample_count),
                class_count=int(len(set(class_indices_list))),
            )
        return True

    # ── Initialize session timing and profiler ──
    session_start_time = time.time()
    profiler = get_profiler()
    
    print("\n=== ISL Sign Language Recognition (Continuous, Automatic Translation) ===")
    print(f"  Sliding window: {NUM_FRAMES} frames")
    print(f"  Base confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Word stability: {sentence_builder.stability_frames} frames")
    print(f"  Auto-sentence timeout: {sentence_builder.auto_sentence_timeout} frames (~{sentence_builder.auto_sentence_timeout/30:.1f}s)")
    print(f"  ✓ Real-time profiling ENABLED (time.perf_counter, report every 100 frames)")
    if cfg.preprocessing.adaptive_hand_interval_enabled:
        print(
            "  ✓ Adaptive hand detect interval ENABLED "
            f"(base: {cfg.preprocessing.hand_detection_interval}, "
            f"low-motion max: {cfg.preprocessing.max_adaptive_hand_interval}, "
            f"hard re-detect: every {HAND_FORCE_REDETECT_INTERVAL} frames)"
        )
    if temporal_postprocessor_enabled:
        print(
            f"  ✓ Temporal Smoothing ENABLED "
            f"(window: {LIVE_TEMPORAL_WINDOW}, decay: {LIVE_TEMPORAL_DECAY}, anti-flicker: delta={LIVE_TEMPORAL_DELTA:.2f})"
        )
    else:
        print("  ✗ Temporal Smoothing DISABLED")
    if MOTION_GATING_ENABLED:
        print(f"  ✓ Motion gating ENABLED (motion threshold: {MOTION_THRESHOLD:.1f}px)")
    if DYNAMIC_THRESHOLD_ENABLED:
        print(f"  ✓ Dynamic thresholds ENABLED (motion boost: {MOTION_BOOST_FACTOR:.0%}, stability boost: {STABILITY_BOOST_FACTOR:.0%})")
    if PSEUDO_BUFFER_ENABLED:
        print(f"  ✓ Pseudo-label Collection ENABLED (threshold: {PSEUDO_THRESHOLD:.0%})")
    if ADAPTER_ENABLED and adapter_model is not None:
        print(
            "  ✓ Adapter Learning ENABLED "
            f"(every {ADAPTER_TRAINING_INTERVAL} preds, "
            f"{ADAPTER_MIN_SAVED_SAMPLES}+ saved, "
            f"{ADAPTER_TRAIN_MIN_SAMPLES}+ train, "
            f"{ADAPTER_MIN_CLASSES}+ classes, "
            f"{ADAPTER_MIN_SAMPLES_PER_CLASS}+ per class)"
        )
        print(
            "    + Auto-tuning ENABLED: thresholds lower for short sessions (< 2 min)"
        )
    print(f"  ➜ Just sign! No keyboard input needed (Q/ESC to quit)")
    print("=======================================================================")

    if pipeline_log is not None:
        pipeline_log.event(
            "inference_start",
            num_frames=NUM_FRAMES,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            pseudo_buffer_enabled=PSEUDO_BUFFER_ENABLED,
            adapter_enabled=ADAPTER_ENABLED,
            adapter_training_interval=ADAPTER_TRAINING_INTERVAL,
            adapter_train_min_samples=ADAPTER_TRAIN_MIN_SAMPLES,
            adapter_min_saved_samples=ADAPTER_MIN_SAVED_SAMPLES,
            adapter_min_classes=ADAPTER_MIN_CLASSES,
            adapter_min_samples_per_class=ADAPTER_MIN_SAMPLES_PER_CLASS,
        )



    while True:
        # ═══════════════════════════════════════════════════════════════════════════
        # MARK FRAME START (for total frame time measurement)
        # ═══════════════════════════════════════════════════════════════════════════
        start_frame()
        
        # ─────────────────────────────────────────────────────────────────────────
        # [SECTION 1] Frame Capture
        # ─────────────────────────────────────────────────────────────────────────
        with profile_section("frame_capture"):
            ret, frame = camera.read()
            if not ret or frame is None:
                # If camera temporarily fails, sleep briefly and continue
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)

        frame_timestamp_ms = int(time.perf_counter() * 1000)
        
        h, w = frame.shape[:2]
        hand_detect_interval = _get_adaptive_hand_detection_interval(motion_magnitude)

        # ─────────────────────────────────────────────────────────────────────────
        # [SECTION 2] Landmark Extraction (Hand + Face)
        # ─────────────────────────────────────────────────────────────────────────
        with profile_section("hand_detection"):
            # Respect runtime override: do not run HOG detection in live mode when forced off.
            if not HOG_ENABLED or cfg.preprocessing.disable_hog_detection or hog_detector is None:
                people = []  # OPTIMIZATION: Skip HOG detection entirely in live mode
            else:
                people = _detect_person_boxes(frame, hog_detector)
        
        with profile_section("landmark_extraction"):
            landmarks_vec, hand_infos, face_center, face_landmarks = _extract_frame_landmarks(
                landmarker,
                force_landmarker,
                holistic,
                frame,
                hand_cache,
                face_cache,
                frame_idx,
                frame_timestamp_ms=frame_timestamp_ms,
                hand_detect_interval=hand_detect_interval,
            )

        if _detect_hand_drift(hand_infos, wrist_history, frame.shape):
            hand_cache['result'] = None
            with profile_section("hand_redetect"):
                landmarks_vec, hand_infos, face_center, face_landmarks = _extract_frame_landmarks(
                    landmarker,
                    force_landmarker,
                    holistic,
                    frame,
                    hand_cache,
                    face_cache,
                    frame_idx,
                    frame_timestamp_ms=frame_timestamp_ms,
                    hand_detect_interval=hand_detect_interval,
                    force_detection=True,
                )

        if DEBUG_DRAW_FACE_CENTER and face_center is not None:
            cv2.circle(frame, face_center, 6, (255, 0, 255), -1)
            cv2.putText(
                frame, "Nose",
                (face_center[0] + 8, face_center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1,
            )

        for pid, (x1, y1, x2, y2, conf) in enumerate(people):
            cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 2)
            cv2.putText(
                frame, f"P{pid} {conf:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2,
            )

        # ─────────────────────────────────────────────────────────────────────────
        # [SECTION 3] Hand Selection (Face-Based Filtering)
        # ─────────────────────────────────────────────────────────────────────────
        with profile_section("hand_selection"):
            filtered_hand_infos = []
            
            # OPTIMIZATION: Skip hand selector for single-hand case
            # Single-hand signs don't benefit from multi-hand filtering
            if hand_infos and len(hand_infos) <= 1:
                # Fast path: single hand (common case, ~90% of signs)
                filtered_hand_infos = hand_infos
            elif face_landmarks is not None and hand_infos:
                # Multi-hand case: use hand_selector for face-relative filtering
                # Convert MediaPipe landmarks to numpy arrays for hand_selector
                face_lms_np = _landmarks_to_numpy(face_landmarks)  # (468, 3)
                hand_lms_list = [_landmarks_to_numpy(info["landmarks"]) for info in hand_infos]  # List of (21, 3)
                
                # Call hand_selector with correct format
                hand_selector_result = hand_selector.process_hands(
                    face_lms_np, hand_lms_list, (h, w)
                )
                
                # Reconstruct filtered_hand_infos from selected hand indices
                selected_indices = hand_selector_result.get('selected_hand_indices', [])
                for idx in selected_indices:
                    if idx < len(hand_infos):
                        filtered_hand_infos.append(hand_infos[idx])
            else:
                # Fallback: if no face detected or no hands, use all hands (backwards compat)
                filtered_hand_infos = hand_infos

        left_owner_ids = []
        right_owner_ids = []
        hand_labels = []
        wrist_points = []
        for info in filtered_hand_infos:
            hand = info["landmarks"]
            label = info["label"]
            _draw_landmarks(frame, hand, w, h)

            hand_box = _landmarks_to_bbox(hand, w, h)
            # With HandSelector, hands are already filtered to the signer, so owner tracking is simplified
            color = CYAN if label == "Left" else ORANGE
            hand_labels.append(label)
            wrist_points.append(_wrist_point_px(hand, w, h))

            x1, y1, x2, y2 = hand_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

            if label == "Left":
                left_owner_ids.append(0)  # Simplified: assume all filtered hands from signer (person 0)
            elif label == "Right":
                right_owner_ids.append(0)
        
        # ── Motion Tracking ──
        # Use first wrist point if available, or track motion for all hands
        if wrist_points:
            # Average wrist position if both hands present
            avg_wrist = (
                (sum(p[0] for p in wrist_points) / len(wrist_points),
                 sum(p[1] for p in wrist_points) / len(wrist_points))
            )
            motion_magnitude = _calculate_hand_motion(
                avg_wrist, wrist_history, motion_magnitude
            )
            wrist_history.append(avg_wrist)
            
            # Track motion momentum
            if motion_magnitude > MOTION_THRESHOLD:
                frames_in_motion = 5  # Reset momentum counter
            else:
                frames_in_motion = max(0, frames_in_motion - 1)
        else:
            frames_in_motion = max(0, frames_in_motion - 1)

        matched_person_id = None
        for left_id in left_owner_ids:
            if left_id is None:
                continue
            if left_id in right_owner_ids:
                matched_person_id = left_id
                break

        has_left = "Left" in hand_labels
        has_right = "Right" in hand_labels
        two_hand_mode = has_left and has_right

        if two_hand_mode:
            if matched_person_id is not None:
                same_person_pair = True
            elif len(people) <= 1:
                # If detector sees <=1 person, trust two-hand presence.
                same_person_pair = True
                if len(people) == 1:
                    matched_person_id = 0
            elif len(wrist_points) >= 2:
                # Final fallback: nearby wrists likely belong to same signer.
                (x1, y1), (x2, y2) = wrist_points[0], wrist_points[1]
                wrist_dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                same_person_pair = wrist_dist < (0.45 * w)
            else:
                same_person_pair = False
        else:
            # One-hand sign path should stay valid.
            same_person_pair = False

        hands_visible = len(filtered_hand_infos) > 0
        valid_for_prediction = hands_visible and (
            (not two_hand_mode) or same_person_pair
        )

        # ── Continuous sliding-window inference ──
        if valid_for_prediction:
            no_hand_frames = 0
            invalid_pair_frames = 0
            sequence_buffer.append(landmarks_vec.copy())
        else:
            if not hands_visible:
                no_hand_frames += 1
            else:
                invalid_pair_frames += 1

            # After ~0.5s with no valid signer-pair, reset
            if no_hand_frames > 15 or invalid_pair_frames > 15:
                sequence_buffer.clear()
                prediction_history.clear()
                prediction_text = "Show a sign"
                confidence_text = ""
                prob_lines = []
            else:
                prediction_history.clear()
                prediction_text = "MOVING"
                confidence_text = "Transition suppressed"
            previous_landmarks_vec = None
            transition_state = "MOVING"
            transition_stable_frames = 0

        if valid_for_prediction:
            transition_movement = _compute_transition_movement(landmarks_vec, previous_landmarks_vec)
            if previous_landmarks_vec is None:
                if transition_state == "MOVING":
                    transition_stable_frames = min(transition_stable_frames + 1, TRANSITION_STABLE_FRAMES)
                    if transition_stable_frames >= TRANSITION_STABLE_FRAMES:
                        transition_state = "STABLE"
                else:
                    transition_state = "STABLE"
                    transition_stable_frames = TRANSITION_STABLE_FRAMES
            elif transition_movement > TRANSITION_MOVEMENT_THRESHOLD:
                if transition_state != "MOVING":
                    prediction_momentum.clear()
                    prediction_history.clear()
                transition_state = "MOVING"
                transition_stable_frames = 0
            else:
                transition_stable_frames += 1
                if transition_stable_frames >= TRANSITION_STABLE_FRAMES:
                    transition_state = "STABLE"

            previous_landmarks_vec = landmarks_vec.copy()

        # ─────────────────────────────────────────────────────────────────────────
        # [SECTION 4] Model Inference (triggered when buffer is full)
        # ─────────────────────────────────────────────────────────────────────────
        if valid_for_prediction and len(sequence_buffer) == NUM_FRAMES:
            record_inference()  # Record that an inference occurred
            
            with profile_section("preprocessing"):
                seq = np.array(sequence_buffer, dtype=np.float32)
                with profile_section("normalization"):
                    seq = _normalize_landmarks(seq)
                
                with profile_section("velocity_features"):
                    if USE_VELOCITY:
                        seq = _add_velocity(seq)

            with profile_section("model_inference"):
                result = {
                    "pred_idx": -1,
                    "confidence": 0.0,
                    "probs": np.array([], dtype=np.float32),
                }
                try:
                    main_models, fallback_models, classes = ensure_word_models()
                    # PHASE 3: Use ONNX ensemble if available, else existing PyTorch merged ensemble
                    if word_use_onnx:
                        idx, conf, probs = ensemble_predict_with_onnx(main_models, seq, use_tta=LIVE_USE_TTA)
                        result = {
                            "main": (int(idx), float(conf), probs),
                            "fallback": None,
                            "pred_idx": int(idx),
                            "confidence": float(conf),
                            "probs": probs,
                        }
                        predicted = classes[idx] if idx < len(classes) else "?"
                    else:
                        from ensemble import merged_ensemble_predict
                        # PHASE 3: Use config-driven TTA setting (disabled by default for live inference)
                        result = merged_ensemble_predict(
                            main_models, fallback_models, seq, use_tta=LIVE_USE_TTA,
                        )
                        idx = result['pred_idx']
                        conf = result['confidence']
                        probs = result['probs']
                        predicted = classes[idx] if idx < len(classes) else "?"
                    
                except Exception as e:
                    print(f"[ERROR] Inference failed: {e}")
                    idx = -1
                    conf = 0.0
                    probs = np.zeros(len(classes)) if classes else []
                    predicted = "ERROR"
                    result = {
                        "pred_idx": int(idx),
                        "confidence": float(conf),
                        "probs": np.array(probs, dtype=np.float32) if len(probs) else np.array([], dtype=np.float32),
                    }
            
            # ─────────────────────────────────────────────────────────────────────────
            # [SECTION 5] Post-Processing (Temporal Smoothing + Momentum)
            # ─────────────────────────────────────────────────────────────────────────
            
            # Convert probabilities to numpy array if not already
            probs_array = np.array(probs) if not isinstance(probs, np.ndarray) else probs
            confidence_gap = None
            if probs_array is not None and len(probs_array) >= 2:
                top_two = np.sort(np.asarray(probs_array, dtype=np.float32))[-2:]
                confidence_gap = float(top_two[1] - top_two[0])
            
            with profile_section("temporal_smoothing"):
                if temporal_postprocessor_enabled:
                    # Use smooth_raw_prediction for confidence smoothing ONLY (no class lock)
                    # This allows the existing prediction_history smoothing to work properly
                    idx, conf = temporal_postprocessor.smooth_raw_prediction(probs_array)
                    predicted = classes[idx] if idx < len(classes) else "?"
            
            # ── PART B: Adapter Model Application ──
            # DISABLED: Adapter model is producing near-uniform distributions
            # (converting 0.94 confidence to 0.02). Model needs retraining or investigation.
            # Temporarily disabled to verify ensemble baseline works correctly.
            original_probs = probs_array.copy()
            # if adapter_model is not None:
            #     try:
            #         with torch.no_grad():
            #             probs_tensor = torch.from_numpy(
            #                 probs_array.astype(np.float32)
            #             ).unsqueeze(0).to(DEVICE)
            #             adapted_probs_tensor = adapter_model(probs_tensor)
            #             adapted_probs = adapted_probs_tensor.squeeze(0).cpu().numpy()
            #             probs_array = adapted_probs
            #             probs = probs_array
            #             idx = int(np.argmax(probs_array))
            #             conf = float(probs_array[idx])
            #             predicted = classes[idx] if idx < len(classes) else "?"
            #     except Exception as e:
            #         print(f"[Adapter] Error: {e}")
            
            # ── Dynamic Threshold Calculation ──
            is_transition = (last_output_prediction is not None and 
                            predicted != last_output_prediction)
            effective_threshold = _calculate_dynamic_threshold(
                motion_magnitude, prediction_stability_counter, is_transition
            )
            
            # Debug: Log confidence details for low-confidence predictions
            if conf < 0.15:
                print(f"[DEBUG] Low confidence: word={predicted}, conf={conf:.4f} ({conf:.1%}), "
                      f"ensemble_conf={result['confidence']:.4f}, "
                      f"top_prob={np.max(probs):.4f}, threshold={effective_threshold:.4f}")
            
            # ── Motion Gating ──
            motion_gated = _is_motion_gating_active(motion_magnitude, frames_in_motion)
            
            # ── Transition Hysteresis ──
            meets_threshold = conf >= effective_threshold
            if last_output_prediction is not None and is_transition:
                # Require extra confidence to switch predictions
                meets_threshold = conf >= (effective_threshold + TRANSITION_HYSTERESIS)

            if transition_state != "STABLE":
                prediction_momentum.clear()
                prediction_text = "MOVING"
                confidence_text = (
                    f"Suppressed: {transition_movement:.4f} | State: {transition_state}"
                )
                if pipeline_log is not None:
                    pipeline_log.event(
                        "prediction_suppressed",
                        reason="transition_movement",
                        transition_state=transition_state,
                        transition_movement=round(float(transition_movement), 4),
                        stable_frames=int(transition_stable_frames),
                    )
            elif meets_threshold and not motion_gated:
                # ─────────────────────────────────────────────────────────────
                # [SECTION 6] Prediction Momentum (Majority Voting)
                # ─────────────────────────────────────────────────────────────
                with profile_section("prediction_momentum"):
                    prediction_momentum.push(idx, conf)
                    commit = prediction_momentum.get_commit()
                
                if commit is None:
                    # Tentative prediction; don't accept yet
                    prediction_text = f"... {predicted}?"
                    confidence_text = f"Tentative: {conf:.1%} | Motion: {motion_magnitude:.1f}"
                    # Log tentative event for analysis
                    if pipeline_log is not None:
                        pipeline_log.event(
                            "prediction_tentative",
                            predicted=predicted,
                            pred_idx=int(idx),
                            confidence=round(float(conf), 4),
                            motion=round(float(motion_magnitude), 2),
                            momentum_window=prediction_momentum.window,
                            momentum_count=prediction_momentum.commit_count,
                        )
                else:
                    committed_idx, avg_conf = commit
                    committed_predicted = classes[committed_idx] if committed_idx < len(classes) else "?"

                    if committed_predicted == last_output_prediction:
                        prediction_stability_counter += 1
                    else:
                        # Word changed: clear history for instant switching
                        prediction_stability_counter = 1
                        prediction_history.clear()
                        last_output_prediction = committed_predicted

                    prediction_history.append(committed_predicted.upper())
                    prediction_text = Counter(prediction_history).most_common(1)[0][0]
                    confidence_text = (
                        f"Conf: {avg_conf:.1%} | Motion: {motion_magnitude:.1f} | Stable: {prediction_stability_counter}"
                    )
                    # Log committed prediction
                    if pipeline_log is not None:
                        pipeline_log.event(
                            "prediction_committed",
                            predicted=committed_predicted,
                            committed_idx=int(committed_idx),
                            avg_conf=round(float(avg_conf), 4),
                            motion=round(float(motion_magnitude), 2),
                            stable_frames=int(prediction_stability_counter),
                        )

                if pipeline_log is not None:
                    pipeline_log.event(
                        "prediction_accepted",
                        predicted=predicted,
                        confidence=round(float(conf), 4),
                            effective_threshold=round(float(effective_threshold), 4),
                            motion=round(float(motion_magnitude), 2),
                            stable_frames=prediction_stability_counter,
                            transition=bool(is_transition),
                            transition_state=transition_state,
                            transition_movement=round(float(transition_movement), 4),
                        )
                    
                    # ── PART A: Pseudo-Label Collection ──
                    # Collect high-confidence prediction as pseudo-labeled sample
                    if pseudo_buffer is not None and conf >= PSEUDO_THRESHOLD:
                        # Protect pseudo-buffer operations with lock
                        with _pseudo_lock:
                            collected = pseudo_buffer.add_sample(
                                class_name=predicted,
                                seq=seq,
                                confidence=conf,
                            )
                        if collected:
                            prediction_count_since_save = 0
                            if pipeline_log is not None:
                                pipeline_log.event(
                                    "pseudo_sample_collected",
                                    class_name=predicted,
                                    confidence=round(float(conf), 4),
                                    buffer_total=pseudo_buffer.get_total_samples(),
                                )
                    
                    # Periodically save pseudo-buffer
                    prediction_count_since_save += 1
                    if pseudo_buffer is not None and AUTO_SAVE_PSEUDO:
                        # Check and swap buffer for background save
                        should = False
                        with _pseudo_lock:
                            should = pseudo_buffer.should_save()
                        if should and prediction_count_since_save >= PSEUDO_SAVE_INTERVAL:
                            # Create a copy to save in background and clear main buffer
                            with _pseudo_lock:
                                buffer_copy = pseudo_buffer.get_buffer_copy()
                                pseudo_buffer.clear()
                            # Spawn background thread to save copied buffer
                            t = threading.Thread(target=_background_save_buffer, args=(buffer_copy, True), daemon=True)
                            t.start()
                            prediction_count_since_save = 0
                    
                    # ── Adapter Training Trigger ──
                    # Periodically check if we should train adapter
                    if adapter_manager is not None:
                        prediction_count_since_training += 1
                        
                        # Compute adaptive thresholds based on session duration
                        elapsed_secs = time.time() - session_start_time
                        adaptive_interval, adaptive_min_saved, is_short_session = _compute_adaptive_thresholds(
                            elapsed_secs,
                            ADAPTER_TRAINING_INTERVAL,
                            ADAPTER_MIN_SAVED_SAMPLES,
                        )
                        
                        if is_short_session and adaptive_interval != ADAPTER_TRAINING_INTERVAL:
                            print(f"[Adapter] Short session detected ({elapsed_secs:.0f}s): lowering interval {ADAPTER_TRAINING_INTERVAL} → {adaptive_interval}")
                        
                        if prediction_count_since_training >= adaptive_interval:
                            # Run adapter training check asynchronously to avoid blocking
                            t = threading.Thread(target=_background_adapter_training, args=("periodic check",), daemon=True)
                            t.start()
                            prediction_count_since_training = 0
                
                else:
                    # Prediction rejected
                    if motion_gated:
                        reason = "Motion gated"
                    else:
                        reason = f"Low conf (>{effective_threshold:.0%})"
                    
                    prediction_history.clear()
                    prediction_stability_counter = 0
                    prediction_text = "..."
                    confidence_text = (
                        f"Rejected: {reason} | Conf: {conf:.1%}"
                    )

                    if pipeline_log is not None:
                        pipeline_log.event(
                            "prediction_rejected",
                            predicted=predicted,
                            confidence=round(float(conf), 4),
                            effective_threshold=round(float(effective_threshold), 4),
                            motion_gated=bool(motion_gated),
                            reason=reason,
                        )

                # ─────────────────────────────────────────────────────────────
                # [SECTION 7] Sentence Builder (Continuous Translation)
                # ─────────────────────────────────────────────────────────────
                with profile_section("sentence_builder"):
                    result = sentence_builder.update(
                        prediction_text,
                        conf,
                        confidence_gap=confidence_gap,
                    )
                
                added_word = result.get('added_word')
                completed_sentence = result.get('completed_sentence')
                
                if added_word and added_word != last_displayed_word:
                    print(f"📝 Added: {added_word}")
                    last_displayed_word = added_word
                
                if completed_sentence:
                    print(f"✅ Sentence: {completed_sentence}")
                    if pipeline_log is not None:
                        pipeline_log.event(
                            "sentence_completed",
                            sentence=completed_sentence,
                            completed_count=len(sentence_builder.completed_sentences),
                        )

                top5 = sorted(
                    enumerate(probs), key=lambda x: -x[1],
                )[:5]
                prob_lines = [
                    f"{classes[i]}: {probs[i]:.1%}"
                    for i, _ in top5
                ]

        # ─────────────────────────────────────────────────────────────────────────
        # [SECTION 8] Rendering (All cv2 drawing operations)
        # ─────────────────────────────────────────────────────────────────────────
        with profile_section("rendering"):
            # ── Prediction panel ──
            overlay = frame.copy()
            panel_h = 140
            cv2.rectangle(overlay, (0, h - panel_h), (280, h), BLACK, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(
                frame, prediction_text,
                (10, h - panel_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2,
            )
            if confidence_text:
                cv2.putText(
                    frame, confidence_text,
                    (10, h - panel_h + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
                )
            for idx, line in enumerate(prob_lines[:6]):
                cv2.putText(
                    frame, line,
                    (10, h - panel_h + 67 + idx * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1,
                )

            # ── Full translation display (top of screen) ──
            display_info = sentence_builder.get_display_text()
            current_sentence = display_info['sentence'] if display_info['sentence'] else "(signing...)"
            completed_sentences = sentence_builder.completed_sentences
            
            # Build full translation text: completed sentences + current
            full_translation_parts = completed_sentences + [current_sentence]
            full_translation = " ".join(full_translation_parts).strip()
            if not full_translation or full_translation == "(signing...)":
                full_translation = "👂 Listening to your signs..."
            
            # Main translation display (top of screen - prominent)
            overlay_top = frame.copy()
            top_panel_h = 80
            cv2.rectangle(overlay_top, (10, 10), (w - 10, 10 + top_panel_h), BLACK, -1)
            cv2.addWeighted(overlay_top, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(
                frame, "Real-time Translation:",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1,
            )
            
            # Wrap and display full translation
            max_chars_per_line = 90
            lines = []
            remaining = full_translation
            while len(remaining) > max_chars_per_line:
                lines.append(remaining[:max_chars_per_line])
                remaining = remaining[max_chars_per_line:]
            if remaining:
                lines.append(remaining)
            
            for idx, line in enumerate(lines[:2]):  # Show up to 2 lines
                y_offset = 50 + idx * 18
                cv2.putText(
                    frame, line,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1,
                )
            
            if len(lines) > 2:
                cv2.putText(
                    frame, f"... (+{len(lines)-2} more lines)",
                    (20, 50 + 2*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, YELLOW, 1,
                )

            cv2.putText(
                frame,
                f"Sens: {sentence_builder.frames_since_last_word}/{sentence_builder.auto_sentence_timeout}  Sentences: {len(completed_sentences)}",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 1,
            )

            if not hands_visible:
                pair_status = "Same person: waiting"
                pair_color = WHITE
            elif not two_hand_mode:
                pair_status = "Single-hand sign mode"
                pair_color = GREEN
            elif same_person_pair:
                if matched_person_id is None:
                    pair_status = "Same person: YES"
                else:
                    pair_status = f"Same person: YES (P{matched_person_id})"
                pair_color = GREEN
            else:
                pair_status = "Same person: NO"
                pair_color = RED

            cv2.putText(
                frame, pair_status,
                (w - 250, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, pair_color, 2,
            )

            n_hands = len(hand_infos)
            status = (
                f"{n_hands} hand{'s' if n_hands != 1 else ''} OK"
                if n_hands else "Show hand"
            )
            color = GREEN if valid_for_prediction else RED
            cv2.putText(
                frame, status,
                (w - 120, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        # ─────────────────────────────────────────────────────────────────────────
        # [SECTION 9] Display (cv2.imshow + cv2.waitKey)
        # ─────────────────────────────────────────────────────────────────────────
        with profile_section("display"):
            cv2.imshow("ISL Sign Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
        
        # ── Keyboard controls (minimal) ──
        if key == ord("q") or key == 27:  # Q/ESC - Quit
            break

        # ─────────────────────────────────────────────────────────────────────────
        # MARK FRAME END (complete frame timing)
        # ─────────────────────────────────────────────────────────────────────────
        end_frame()
        
        # Print profiling report every 100 frames
        if frame_idx % 100 == 0:
            profiler = get_profiler()
            profiler.print_report()

        frame_idx += 1

    # Cleanup camera thread and release resources
    try:
        camera.release()
    except Exception:
        pass
    landmarker.close()
    if holistic is not None:
        holistic.close()
    cv2.destroyAllWindows()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL PROFILING REPORT (printed on exit)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("PROFILING SESSION COMPLETE - FINAL REPORT")
    print("="*80)
    profiler = get_profiler()
    profiler.print_report(title="FINAL SESSION STATISTICS")
    print("\n" + "─"*80)
    print("Detailed Section Breakdown:")
    print("─"*80)
    profiler.print_section_details()
    print("\n" + "="*80)
    
    # ── Cleanup: Save pseudo-buffer and shutdown adapter ──
    if pseudo_buffer is not None and pseudo_buffer.get_total_samples() > 0:
        print(f"\n[Cleanup] Saving pseudo-buffer with {pseudo_buffer.get_total_samples()} samples...")
        pseudo_buffer.save(verbose=True)
        if pipeline_log is not None:
            pipeline_log.event(
                "pseudo_buffer_saved_on_cleanup",
                remaining_buffer_samples=int(pseudo_buffer.get_total_samples()),
                saved_on_disk=int(pseudo_buffer.get_saved_sample_count()),
            )

    if ADAPTER_ENABLED and adapter_manager is not None:
        if _attempt_adapter_training("cleanup"):
            if pipeline_log is not None:
                pipeline_log.event("adapter_training_requested_on_cleanup")
    
    if adapter_manager is not None:
        print("[Cleanup] Shutting down adapter manager...")
        adapter_manager.shutdown()
        if pipeline_log is not None:
            pipeline_log.event("adapter_shutdown")

    flushed_word = sentence_builder.flush_pending_word()
    if flushed_word:
        print(f"[Cleanup] Flushed pending word: {flushed_word}")
        if pipeline_log is not None:
            pipeline_log.event("pending_word_flushed", word=flushed_word)
    
    # Show final translation summary
    all_parts = sentence_builder.completed_sentences.copy()
    if sentence_builder.current_sentence.strip():
        all_parts.append(sentence_builder.current_sentence.strip())
    
    if all_parts:
        full_text = " ".join(all_parts)
        print(f"\n{'='*70}")
        print(f"📝 FINAL TRANSLATION ({len(sentence_builder.completed_sentences)} completed + current)")
        print(f"{'='*70}")
        print(f"{full_text}")
        print(f"{'='*70}\n")
    else:
        print("\nNo translation recorded.")
        if pipeline_log is not None:
            pipeline_log.event("no_translation_recorded")
    print("Webcam closed.")
    if pipeline_log is not None:
        pipeline_log.event("inference_stop")


if __name__ == "__main__":
    run_webcam()
