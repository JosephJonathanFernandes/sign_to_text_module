"""
Live webcam ISL recognition -- automatic dual mode.

The system automatically detects what you're signing:
  - STATIC hand pose held steady ~1s -> letter/number (CNN)
  - Press SPACE to record a gesture   -> word (BiGRU)

Controls:
    SPACE  - Record a word gesture (30 frames)
    Q/ESC  - Quit
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from config import (
    NUM_FRAMES, NUM_LANDMARKS, NUM_COORDS,
    HAND_LANDMARKER_MODEL, USE_VELOCITY,
)
from config_image import IMG_SIZE
from preprocess import _normalize_landmarks, _add_velocity


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
ORANGE = (0, 165, 255)

# Auto-detection thresholds
STILL_THRESHOLD = 0.8       # seconds hand must be still for letter
MOVE_THRESH = 0.04           # normalised landmark movement threshold
LETTER_COOLDOWN = 2.0        # seconds between auto-letter fires


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
    """Extract 63-dim landmark vector from a frame."""
    feat_dim = NUM_LANDMARKS * NUM_COORDS
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    landmarks_vec = np.zeros(feat_dim, dtype=np.float32)
    hand_lm = None

    if result.hand_landmarks:
        hand_lm = result.hand_landmarks[0]
        coords = []
        for lm in hand_lm:
            coords.extend([lm.x, lm.y, lm.z])
        landmarks_vec = np.array(coords, dtype=np.float32)

    return landmarks_vec, hand_lm


def _crop_hand_region(frame, hand_landmarks, w, h):
    """Crop the hand region from frame for letter prediction."""
    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    pad_x = int((x2 - x1) * 0.3)
    pad_y = int((y2 - y1) * 0.3)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    size = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    return crop


def run_webcam():
    """
    Main webcam loop with automatic detection.

    - Hold hand still ~1s  -> auto letter/number classification (CNN)
    - Press SPACE           -> record 30-frame gesture -> word (BiGRU)
    """

    # ── Lazy model loading ──
    word_models = word_classes = None
    letter_models = letter_classes = None

    def ensure_word_models():
        nonlocal word_models, word_classes
        if word_models is None:
            print("Loading word models...")
            from ensemble import load_ensemble
            word_models, word_classes, _ = load_ensemble()
        return word_models, word_classes

    def ensure_letter_models():
        nonlocal letter_models, letter_classes
        if letter_models is None:
            print("Loading letter models...")
            from ensemble_image import load_image_ensemble
            letter_models, letter_classes, _ = load_image_ensemble()
        return letter_models, letter_classes

    # Try pre-loading both
    try:
        ensure_word_models()
    except FileNotFoundError:
        print("[WARN] No word model found  -- train first")
    try:
        ensure_letter_models()
    except FileNotFoundError:
        print("[WARN] No letter model found -- train first")

    # ── Landmarker ──
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State ──
    recording = False
    recorded_frames = []
    prediction_text = "Show a sign"
    confidence_text = ""
    prob_lines = []
    detected_type = ""          # "LETTER" or "WORD"

    # Still-detection state
    prev_landmarks = None
    still_start = None
    next_letter_time = 0.0      # cooldown timestamp

    print("\n=== ISL Sign Language Recognition ===")
    print("  Auto-detects letters (hold hand still)")
    print("  SPACE  - Record word gesture")
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
        landmarks_vec, hand_lm = _extract_frame_landmarks(landmarker, frame)
        if hand_lm:
            _draw_landmarks(frame, hand_lm, w, h)

        # ── Auto letter detection (hand held still) ──
        if hand_lm and not recording:
            if prev_landmarks is not None:
                diff = np.linalg.norm(landmarks_vec - prev_landmarks)
                if diff < MOVE_THRESH:
                    if still_start is None:
                        still_start = now
                    elif (now - still_start >= STILL_THRESHOLD
                          and now >= next_letter_time):
                        # Classify as letter
                        still_start = None
                        next_letter_time = now + LETTER_COOLDOWN
                        try:
                            models, classes = ensure_letter_models()
                            crop = _crop_hand_region(frame, hand_lm, w, h)
                            crop_r = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                            from ensemble_image import (
                                preprocess_image,
                                image_ensemble_predict,
                            )
                            img_chw = preprocess_image(crop_r)
                            idx, conf, probs = image_ensemble_predict(
                                models, img_chw,
                            )
                            pc = classes[idx] if idx < len(classes) else "?"
                            detected_type = "LETTER"
                            prediction_text = pc
                            confidence_text = f"Conf: {conf:.1%}"
                            top5 = sorted(
                                enumerate(probs), key=lambda x: -x[1],
                            )[:5]
                            prob_lines = [
                                f"{classes[i]}: {p:.1%}" for i, p in top5
                            ]
                            print(f"  [LETTER] {pc} ({conf:.1%})")
                        except FileNotFoundError:
                            pass
                else:
                    still_start = None
            prev_landmarks = landmarks_vec.copy()
        else:
            if not recording:
                prev_landmarks = None
                still_start = None

        # ── WORD recording (SPACE triggered) ──
        if recording:
            recorded_frames.append(landmarks_vec.copy())
            progress = len(recorded_frames)

            cv2.circle(frame, (30, 30), 12, RED, -1)
            cv2.putText(
                frame, f"Recording: {progress}/{NUM_FRAMES}",
                (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2,
            )
            bar_w = 200
            bar_x = w - bar_w - 20
            filled = int(bar_w * progress / NUM_FRAMES)
            cv2.rectangle(frame, (bar_x, 50), (bar_x + bar_w, 70), WHITE, 2)
            cv2.rectangle(frame, (bar_x, 50), (bar_x + filled, 70), GREEN, -1)

            if progress >= NUM_FRAMES:
                recording = False
                seq = np.array(
                    recorded_frames[:NUM_FRAMES], dtype=np.float32,
                )
                seq = _normalize_landmarks(seq)
                if USE_VELOCITY:
                    seq = _add_velocity(seq)
                try:
                    from ensemble import ensemble_predict
                    models, classes = ensure_word_models()
                    idx, conf, probs = ensemble_predict(
                        models, seq, use_tta=True,
                    )
                    pc = classes[idx] if idx < len(classes) else "?"
                    detected_type = "WORD"
                    prediction_text = pc.upper()
                    confidence_text = f"Conf: {conf:.1%}"
                    prob_lines = [
                        f"{c}: {probs[i]:.1%}"
                        for i, c in enumerate(classes)
                    ]
                    print(f"  [WORD] {pc} ({conf:.1%})")
                except FileNotFoundError:
                    prediction_text = "No word model"
                    confidence_text = ""
                    prob_lines = []
                recorded_frames = []

        # ── Still-detection progress indicator ──
        if still_start is not None and not recording:
            elapsed = now - still_start
            pct = min(elapsed / STILL_THRESHOLD, 1.0)
            bar_w = 150
            cv2.rectangle(
                frame, (w - bar_w - 20, 15), (w - 20, 35), WHITE, 2,
            )
            cv2.rectangle(
                frame, (w - bar_w - 20, 15),
                (w - bar_w - 20 + int(bar_w * pct), 35), ORANGE, -1,
            )
            cv2.putText(
                frame, "Detecting letter...",
                (w - bar_w - 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ORANGE, 1,
            )

        # ── Prediction panel ──
        overlay = frame.copy()
        panel_h = 160
        cv2.rectangle(overlay, (0, h - panel_h), (280, h), BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if detected_type:
            badge_color = CYAN if detected_type == "WORD" else ORANGE
            cv2.putText(
                frame, detected_type,
                (10, h - panel_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, badge_color, 2,
            )

        cv2.putText(
            frame, prediction_text,
            (10, h - panel_h + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2,
        )
        if confidence_text:
            cv2.putText(
                frame, confidence_text,
                (10, h - panel_h + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
            )
        for idx, line in enumerate(prob_lines[:6]):
            cv2.putText(
                frame, line,
                (10, h - panel_h + 82 + idx * 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1,
            )

        if not recording:
            cv2.putText(
                frame, "SPACE: Record word | Q: Quit",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1,
            )

        status = "Hand OK" if hand_lm else "Show hand"
        color = GREEN if hand_lm else RED
        cv2.putText(
            frame, status,
            (w - 120, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )

        cv2.imshow("ISL Sign Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" ") and not recording:
            recording = True
            recorded_frames = []
            prediction_text = "Recording..."
            confidence_text = ""
            prob_lines = []
            detected_type = ""

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    run_webcam()
