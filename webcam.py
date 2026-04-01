"""
Live webcam ISL recognition -- rolling-window prediction via landmarks.

Uses a continuous sliding window of NUM_FRAMES and predicts every frame.
Stability is improved with majority voting over recent predictions.

Controls:
    Q/ESC  - Quit
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
)
from preprocess import (
    _normalize_landmarks,
    _add_velocity,
    create_landmarker,
    create_face_landmarker,
    extract_landmarks_with_face_relative,
)


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
    word_models = word_classes = None

    def ensure_word_models():
        nonlocal word_models, word_classes
        if word_models is None:
            print("Loading word models...")
            from ensemble import load_ensemble
            word_models, word_classes, _ = load_ensemble()
        return word_models, word_classes

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

    print("\n=== ISL Sign Language Recognition ===")
    print(f"  Sliding window: {NUM_FRAMES} frames")
    print(f"  Smoothing window: {PREDICTION_SMOOTHING_WINDOW} predictions")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print("  Q/ESC  - Quit")
    print("=====================================\n")

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

        left_owner_ids = []
        right_owner_ids = []
        hand_labels = []
        wrist_points = []
        for info in hand_infos:
            hand = info["landmarks"]
            label = info["label"]
            _draw_landmarks(frame, hand, w, h)

            hand_box = _landmarks_to_bbox(hand, w, h)
            owner = _assign_hand_to_person(hand_box, people)
            color = CYAN if label == "Left" else ORANGE
            hand_labels.append(label)
            wrist_points.append(_wrist_point_px(hand, w, h))

            x1, y1, x2, y2 = hand_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            owner_txt = f"P{owner}" if owner is not None else "P?"
            cv2.putText(
                frame, f"{label} {owner_txt}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

            if label == "Left":
                left_owner_ids.append(owner)
            elif label == "Right":
                right_owner_ids.append(owner)

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

        hands_visible = len(hand_infos) > 0
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
                from ensemble import ensemble_predict
                models, classes = ensure_word_models()
                idx, conf, probs = ensemble_predict(
                    models, seq, use_tta=False,
                )
                predicted = classes[idx] if idx < len(classes) else "?"

                if conf >= CONFIDENCE_THRESHOLD:
                    prediction_history.append(predicted.upper())
                    prediction_text = Counter(
                        prediction_history
                    ).most_common(1)[0][0]
                    confidence_text = (
                        f"Conf: {conf:.1%} | Smooth: {len(prediction_history)}"
                    )
                else:
                    prediction_history.clear()
                    prediction_text = "..."
                    confidence_text = (
                        f"Low conf: {conf:.1%} (< {CONFIDENCE_THRESHOLD:.0%})"
                    )

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

        cv2.putText(
            frame,
            f"Window: {len(sequence_buffer)}/{NUM_FRAMES} | Q: Quit",
            (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1,
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
        if key == ord("q") or key == 27:
            break

        frame_idx += 1

    cap.release()
    landmarker.close()
    if holistic is not None:
        holistic.close()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    run_webcam()
