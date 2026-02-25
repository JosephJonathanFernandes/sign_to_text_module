"""
Live webcam ISL recognition -- rolling-window prediction via landmarks.

Uses a continuous sliding window of NUM_FRAMES and predicts every frame.
Stability is improved with majority voting over recent predictions.

Controls:
    Q/ESC  - Quit
"""

import cv2
import numpy as np
import time
from collections import Counter, deque
import mediapipe as mp

from config import (
    NUM_FRAMES, NUM_LANDMARKS, NUM_COORDS, NUM_HANDS,
    LANDMARK_DIM, FRAME_FEAT_DIM,
    USE_VELOCITY, CONFIDENCE_THRESHOLD,
    PREDICTION_SMOOTHING_WINDOW,
)
from preprocess import _normalize_landmarks, _add_velocity, create_landmarker


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


def _draw_landmarks(frame, hand_landmarks, w, h):
    """Draw hand landmarks and connections on the frame."""
    points = []
    for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 4, GREEN, -1)

    for i, j in HAND_CONNECTIONS:
        if i < len(points) and j < len(points):
            cv2.line(frame, points[i], points[j], (0, 200, 0), 2)


def _extract_frame_landmarks(landmarker, frame):
    """Extract FRAME_FEAT_DIM-dim landmark vector from a frame (both hands)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    landmarks_vec = np.zeros(FRAME_FEAT_DIM, dtype=np.float32)
    # Return all detected hands for drawing; first hand for status check
    all_hands = result.hand_landmarks  # list of hand landmark lists
    hand_lm = all_hands[0] if all_hands else None

    hand_slots = {"Right": 0, "Left": 1}
    for hand, handedness_list in zip(
        result.hand_landmarks,
        result.handedness,
    ):
        label = handedness_list[0].display_name  # "Right" or "Left"
        slot = hand_slots.get(label, 0)
        start = slot * LANDMARK_DIM
        coords = []
        for lm in hand:
            coords.extend([lm.x, lm.y, lm.z])
        landmarks_vec[start:start + LANDMARK_DIM] = np.array(
            coords, dtype=np.float32
        )

    return landmarks_vec, all_hands, hand_lm


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

    # ── Landmarker — uses the same settings as preprocess.py ──
    landmarker = create_landmarker(num_hands=NUM_HANDS)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State ──
    sequence_buffer = deque(maxlen=NUM_FRAMES)
    prediction_history = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)
    prediction_text = "Show a sign"
    confidence_text = ""
    prob_lines = []

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
        now = time.time()

        # Detect hand
        landmarks_vec, all_hands, hand_lm = _extract_frame_landmarks(landmarker, frame)
        for hand in all_hands:
            _draw_landmarks(frame, hand, w, h)

        # ── Continuous sliding-window inference ──
        sequence_buffer.append(landmarks_vec.copy())
        if len(sequence_buffer) == NUM_FRAMES:
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

        n_hands = len(all_hands)
        status = f"{n_hands} hand{'s' if n_hands != 1 else ''} OK" if n_hands else "Show hand"
        color = GREEN if n_hands else RED
        cv2.putText(
            frame, status,
            (w - 120, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )

        cv2.imshow("ISL Sign Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    run_webcam()
