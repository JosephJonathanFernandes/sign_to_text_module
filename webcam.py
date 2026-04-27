"""Live webcam ISL recognition with continuous automatic translation.

Automatically translates sign sequences into sentences in real-time.
No keyboard input needed - signs are recognized and sentences build automatically.
Sentences auto-complete after ~2 seconds of no new signs.

Controls:
    Q/ESC  - Quit (only control needed)
"""

import cv2
import numpy as np
from collections import Counter, deque
import mediapipe as mp

from config import (
    NUM_FRAMES, NUM_HANDS,
    DEBUG_DRAW_FACE_CENTER,
    USE_VELOCITY, CONFIDENCE_THRESHOLD,
    PREDICTION_SMOOTHING_WINDOW,
    MOTION_GATING_ENABLED, MOTION_THRESHOLD,
    MOTION_SMOOTHING, IDLE_CONFIDENCE_THRESHOLD,
    DYNAMIC_THRESHOLD_ENABLED, MOTION_BOOST_FACTOR,
    STABILITY_BOOST_FACTOR, DYNAMIC_THRESHOLD_MIN,
    TRANSITION_HYSTERESIS,
)
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
    holistic,
    frame,
    face_cache,
    frame_idx,
    face_detect_interval=3,
):
    """
    Extract frame vector with shared preprocess feature logic.

    Optimized for real-time: face detection runs every N frames
    (cached between). Hand detection runs every frame (critical).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Hand detection every frame (fast and critical)
    result = landmarker.detect(mp_image)

    # Face detection every N frames to speed up real-time inference
    face_landmarks = None
    if holistic is not None and frame_idx % face_detect_interval == 0:
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

    return landmarks_vec, hand_infos, face_center


def run_webcam():
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

    def ensure_word_models():
        nonlocal word_models, word_models_fallback, word_classes
        if word_models is None:
            print("Loading merged 10+2 ensemble...")
            from ensemble import load_merged_ensemble_10_2
            word_models, word_models_fallback, word_classes, _ = load_merged_ensemble_10_2()
        return word_models, word_models_fallback, word_classes

    try:
        ensure_word_models()
    except FileNotFoundError:
        print("[WARN] No word model found  -- train first")

    # ── Landmarker — optimized for webcam (high conf, face skipping) ──
    landmarker = create_landmarker(num_hands=NUM_HANDS, for_webcam=True)
    holistic = create_face_landmarker(for_webcam=True)
    hog_detector = cv2.HOGDescriptor()
    hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State ──
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
    
    # ── Sentence Builder (continuous translation) ──
    sentence_builder = SentenceBuilder(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        stability_frames=6,  # Reduced from 12 for faster response (~0.2s at 30fps)
        auto_sentence_timeout=75  # ~2.5 seconds at 30fps
    )
    last_displayed_word = None

    # ── Temporal Post-Processor (confidence-weighted smoothing + anti-flicker) ──
    from ensemble import load_merged_ensemble_10_2
    try:
        temporal_postprocessor = TemporalPostProcessor(
            window_size=7,  # Frames for confidence averaging
            patience=3,  # Frames to confirm transition
            delta=0.1,  # Confidence margin for transitions
            enable_decay=True,  # Use exponential decay for older frames
            decay_factor=0.3  # Decay weight for older predictions
        )
        temporal_postprocessor_enabled = True
    except Exception as e:
        print(f"[WARN] Could not initialize TemporalPostProcessor: {e}")
        temporal_postprocessor = None
        temporal_postprocessor_enabled = False

    # ── Hand Selector (single-person hand filtering via MediaPipe face landmarks) ──
    hand_selector = HandSelector(
        distance_threshold=300,  # Pixel distance threshold for hand-to-face
        roi_width_ratio=0.5,  # 50% of frame width (centered at face)
        roi_height_ratio=0.5,  # 50% of frame height (centered at face)
        use_roi_filtering=True,  # Use ROI-based filtering (more reliable than pure distance)
        enable_debugging=False  # Set to True for debug logging
    )

    print("\n=== ISL Sign Language Recognition (Continuous, Automatic Translation) ===")
    print(f"  Sliding window: {NUM_FRAMES} frames")
    print(f"  Base confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Word stability: {sentence_builder.stability_frames} frames")
    print(f"  Auto-sentence timeout: {sentence_builder.auto_sentence_timeout} frames (~{sentence_builder.auto_sentence_timeout/30:.1f}s)")
    if temporal_postprocessor_enabled:
        print(f"  ✓ TemporalPostProcessor ENABLED (window: 7 frames, patience: 3)")
    if MOTION_GATING_ENABLED:
        print(f"  ✓ Motion gating ENABLED (motion threshold: {MOTION_THRESHOLD:.1f}px)")
    if DYNAMIC_THRESHOLD_ENABLED:
        print(f"  ✓ Dynamic thresholds ENABLED (motion boost: {MOTION_BOOST_FACTOR:.0%}, stability boost: {STABILITY_BOOST_FACTOR:.0%})")
    print(f"  ➜ Just sign! No keyboard input needed (Q/ESC to quit)")
    print("=======================================================================")


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Detect people + hands
        people = _detect_person_boxes(frame, hog_detector)
        landmarks_vec, hand_infos, face_center = _extract_frame_landmarks(
            landmarker,
            holistic,
            frame,
            face_cache,
            frame_idx,
            face_detect_interval=3,
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

        # ── Hand Selection via HandSelector (face-based single-person filtering) ──
        filtered_hand_infos = []
        if face_landmarks is not None and hand_infos:
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

        if valid_for_prediction and len(sequence_buffer) == NUM_FRAMES:
            seq = np.array(sequence_buffer, dtype=np.float32)
            seq = _normalize_landmarks(seq)
            if USE_VELOCITY:
                seq = _add_velocity(seq)

            try:
                from ensemble import merged_ensemble_predict
                main_models, fallback_models, classes = ensure_word_models()
                result = merged_ensemble_predict(
                    main_models, fallback_models, seq, use_tta=False,
                )
                idx = result['pred_idx']
                conf = result['confidence']
                probs = result['probs']
                predicted = classes[idx] if idx < len(classes) else "?"
                
                # ── TemporalPostProcessor (confidence-weighted smoothing + anti-flicker) ──
                # Convert probabilities to numpy array if not already
                probs_array = np.array(probs) if not isinstance(probs, np.ndarray) else probs
                
                if temporal_postprocessor_enabled:
                    # Use update_with_confidence to get both stable class and smoothed confidence
                    stable_class_idx, smoothed_conf = temporal_postprocessor.update_with_confidence(
                        probs_array
                    )
                    # Only update if stable class is not None (wait for stabilization)
                    if stable_class_idx is not None:
                        idx = stable_class_idx
                        conf = smoothed_conf
                        predicted = classes[idx] if idx < len(classes) else "?"
                
                # ── Dynamic Threshold Calculation ──
                is_transition = (last_output_prediction is not None and 
                                predicted != last_output_prediction)
                effective_threshold = _calculate_dynamic_threshold(
                    motion_magnitude, prediction_stability_counter, is_transition
                )
                
                # ── Motion Gating ──
                motion_gated = _is_motion_gating_active(motion_magnitude, frames_in_motion)
                
                # ── Transition Hysteresis ──
                meets_threshold = conf >= effective_threshold
                if last_output_prediction is not None and is_transition:
                    # Require extra confidence to switch predictions
                    meets_threshold = conf >= (effective_threshold + TRANSITION_HYSTERESIS)

                if meets_threshold and not motion_gated:
                    # Valid prediction detected
                    if predicted == last_output_prediction:
                        prediction_stability_counter += 1
                    else:
                        prediction_stability_counter = 1
                        last_output_prediction = predicted
                    
                    prediction_history.append(predicted.upper())
                    prediction_text = Counter(
                        prediction_history
                    ).most_common(1)[0][0]
                    confidence_text = (
                        f"Conf: {conf:.1%} | Motion: {motion_magnitude:.1f} | Stable: {prediction_stability_counter}"
                    )
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

                # Update sentence builder (continuous translation)
                result = sentence_builder.update(prediction_text, conf)
                added_word = result.get('added_word')
                completed_sentence = result.get('completed_sentence')
                
                if added_word and added_word != last_displayed_word:
                    print(f"📝 Added: {added_word}")
                    last_displayed_word = added_word
                
                if completed_sentence:
                    print(f"✅ Sentence: {completed_sentence}")

                top5 = sorted(
                    enumerate(probs), key=lambda x: -x[1],
                )[:5]
                prob_lines = [
                    f"{classes[i]}: {probs[i]:.1%}"
                    for i, _ in top5
                ]
            except FileNotFoundError:
                prediction_text = "No word model"
                confidence_text = ""
                prob_lines = []

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

        cv2.imshow("ISL Sign Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        
        # ── Keyboard controls (minimal) ──
        if key == ord("q") or key == 27:  # Q/ESC - Quit
            break

        frame_idx += 1

    cap.release()
    landmarker.close()
    if holistic is not None:
        holistic.close()
    cv2.destroyAllWindows()
    
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
    print("Webcam closed.")


if __name__ == "__main__":
    run_webcam()
