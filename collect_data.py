"""
Data collection tool -- record new training samples via webcam.

Usage:
    python collect_data.py                      # interactive class menu
    python collect_data.py --class happy        # go straight to recording 'happy'
    python collect_data.py --class happy --n 10 # record 10 samples of 'happy'

Each recording:
  1.  3-second countdown (get your hand ready)
  2.  ~3-second capture window (90 frames at ~30 fps)
  3.  Uniform sub-sample to 30 frames -> normalize -> save .npy

Saved to: processed/<class>/webcam_<timestamp>.npy
These samples are identical in format to preprocessed video samples,
so you can retrain immediately after collecting.
"""

import os
import sys
import cv2
import time
import numpy as np
import mediapipe as mp

from config import (
    NUM_FRAMES, NUM_LANDMARKS, NUM_COORDS, NUM_HANDS,
    LANDMARK_DIM, FRAME_FEAT_DIM,
    USE_VELOCITY, PROCESSED_DIR,
    WEBCAM_RECORD_FRAMES, WEBCAM_COUNTDOWN,
)
from preprocess import _normalize_landmarks, _add_velocity, create_landmarker

# Colors for OpenCV
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _draw_hand(frame, hand_lm, w, h):
    pts = []
    for lm in hand_lm:
        px, py = int(lm.x * w), int(lm.y * h)
        pts.append((px, py))
        cv2.circle(frame, (px, py), 4, GREEN, -1)
    for i, j in HAND_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            cv2.line(frame, pts[i], pts[j], (0, 200, 0), 2)


def _extract_landmarks(landmarker, frame):
    feat_dim = FRAME_FEAT_DIM
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    vec = np.zeros(feat_dim, dtype=np.float32)
    all_hands = result.hand_landmarks
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
        vec[start:start + LANDMARK_DIM] = np.array(
            coords, dtype=np.float32
        )

    return vec, all_hands, hand_lm


def _existing_classes():
    """Return sorted list of classes already in processed/."""
    if not os.path.isdir(PROCESSED_DIR):
        return []
    return sorted([
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ])


def _count_samples(cls_name):
    """Count .npy files for a class."""
    d = os.path.join(PROCESSED_DIR, cls_name)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".npy")])


def _choose_class():
    """Interactive class selection menu."""
    existing = _existing_classes()

    print("\n" + "=" * 50)
    print("  DATA COLLECTION -- Choose a class")
    print("=" * 50)

    if existing:
        print("\nExisting classes (samples):")
        for i, cls in enumerate(existing, 1):
            n = _count_samples(cls)
            print(f"  {i:>3}. {cls:<15} ({n} samples)")
        print(f"\n  Enter a number (1-{len(existing)}) to add to existing,")
        print("  or type a NEW class name to create it.")
    else:
        print("\n  No existing classes found.")
        print("  Type a class name to create it.")

    print("  Type 'q' to quit.\n")
    choice = input("Class: ").strip()

    if not choice or choice.lower() == "q":
        return None

    # Check if it's a number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(existing):
            return existing[idx]
    except ValueError:
        pass

    # Treat as a new class name
    return choice.lower().strip()


def record_samples(cls_name, num_samples=None):
    """
    Open webcam and record gesture samples for the given class.

    Args:
        cls_name: class label (e.g. 'happy')
        num_samples: if set, auto-quit after this many. Otherwise loop.
    """
    save_dir = os.path.join(PROCESSED_DIR, cls_name)
    os.makedirs(save_dir, exist_ok=True)

    existing_count = _count_samples(cls_name)
    print(f"\n[Collect] Class: '{cls_name}'")
    print(f"[Collect] Existing samples: {existing_count}")
    print(f"[Collect] Recording: {WEBCAM_RECORD_FRAMES} frames "
          f"(~{WEBCAM_RECORD_FRAMES / 30:.0f}s), sub-sampled to "
          f"{NUM_FRAMES}")
    if num_samples:
        print(f"[Collect] Target: {num_samples} new samples")
    print()

    # MediaPipe landmarker (shared settings)
    landmarker = create_landmarker(num_hands=NUM_HANDS)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    saved = 0
    state = "IDLE"          # IDLE -> COUNTDOWN -> RECORDING -> SAVED -> IDLE
    countdown_start = 0.0
    recorded_frames = []

    print("Controls:")
    print("  SPACE  - Start recording a sample")
    print("  Q/ESC  - Finish and quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()

        # Detect hand
        vec, all_hands, hand_lm = _extract_landmarks(landmarker, frame)
        for hand in all_hands:
            _draw_hand(frame, hand, w, h)

        # ── State machine ──

        if state == "COUNTDOWN":
            elapsed = now - countdown_start
            remaining = WEBCAM_COUNTDOWN - elapsed
            if remaining <= 0:
                state = "RECORDING"
                recorded_frames = []
            else:
                cd = int(remaining) + 1
                cv2.putText(
                    frame, str(cd),
                    (w // 2 - 40, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, YELLOW, 6,
                )
                cv2.putText(
                    frame, "Get ready to sign!",
                    (w // 2 - 130, h // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2,
                )

        elif state == "RECORDING":
            recorded_frames.append(vec.copy())
            progress = len(recorded_frames)
            total = WEBCAM_RECORD_FRAMES

            # Red dot + progress
            cv2.circle(frame, (30, 30), 12, RED, -1)
            cv2.putText(
                frame, f"Recording: {progress}/{total}",
                (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2,
            )
            # Progress bar
            bar_w = 200
            bar_x = w - bar_w - 20
            filled = int(bar_w * progress / total)
            cv2.rectangle(frame, (bar_x, 50), (bar_x + bar_w, 70), WHITE, 2)
            cv2.rectangle(
                frame, (bar_x, 50), (bar_x + filled, 70), GREEN, -1,
            )
            secs_left = max(0, (total - progress) / 30)
            cv2.putText(
                frame, f"{secs_left:.1f}s",
                (bar_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
            )

            if progress >= total:
                # Process and save
                raw = np.array(recorded_frames, dtype=np.float32)
                indices = np.linspace(
                    0, len(raw) - 1, NUM_FRAMES, dtype=int,
                )
                seq = raw[indices]  # (NUM_FRAMES, FRAME_FEAT_DIM)
                seq = _normalize_landmarks(seq)
                if USE_VELOCITY:
                    seq = _add_velocity(seq)

                # Check quality: count frames with hands detected
                hand_frames = np.sum(np.any(raw != 0, axis=1))
                hand_pct = hand_frames / len(raw)

                if hand_pct < 0.5:
                    print(f"  [SKIP] Only {hand_pct:.0%} frames had hand "
                          f"detected. Try again.")
                    state = "IDLE"
                else:
                    ts = int(time.time() * 1000)
                    fname = f"webcam_{ts}.npy"
                    np.save(os.path.join(save_dir, fname), seq)
                    saved += 1
                    total_now = existing_count + saved
                    print(f"  [SAVED] #{saved} -> {fname}  "
                          f"(hand: {hand_pct:.0%}, total: {total_now})")
                    state = "SAVED"
                    save_time = now

                    if num_samples and saved >= num_samples:
                        # Show saved feedback briefly
                        cv2.putText(
                            frame,
                            f"Saved! ({saved}/{num_samples} done)",
                            (w // 2 - 120, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2,
                        )
                        cv2.imshow("Data Collection", frame)
                        cv2.waitKey(1500)
                        break

                recorded_frames = []

        elif state == "SAVED":
            # Show green feedback for 1 second
            cv2.putText(
                frame,
                f"Saved! ({saved} new, "
                f"{existing_count + saved} total)",
                (w // 2 - 150, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2,
            )
            if now - save_time > 1.0:
                state = "IDLE"

        # ── HUD (always visible) ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (320, h), BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(
            frame, f"Class: {cls_name.upper()}",
            (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2,
        )
        cv2.putText(
            frame,
            f"Saved: {saved} new | Total: {existing_count + saved}",
            (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
        )

        n_hands = len(all_hands)
        status = (
            f"{n_hands} hand{'s' if n_hands != 1 else ''} OK"
            if n_hands else "Show hand"
        )
        color = GREEN if n_hands else RED
        cv2.putText(
            frame, status,
            (w - 120, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )

        if state == "IDLE":
            cv2.putText(
                frame,
                "SPACE: Record | Q: Quit",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1,
            )

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" ") and state == "IDLE":
            state = "COUNTDOWN"
            countdown_start = time.time()
            print(f"  [REC] Countdown {WEBCAM_COUNTDOWN}s...")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    total_final = existing_count + saved
    print(f"\n[Collect] Done! Saved {saved} new samples for '{cls_name}'.")
    print(f"[Collect] Total samples for '{cls_name}': {total_final}")
    return saved


def collect_interactive():
    """Interactive loop: pick class -> record -> pick again."""
    print("\n" + "=" * 50)
    print("  ISL DATA COLLECTION TOOL")
    print("=" * 50)

    # Show current stats
    existing = _existing_classes()
    if existing:
        total = sum(_count_samples(c) for c in existing)
        print(f"\nCurrent data: {len(existing)} classes, {total} samples")

    while True:
        cls = _choose_class()
        if cls is None:
            break

        n_str = input(
            f"How many samples to record for '{cls}'? "
            f"(Enter for unlimited): "
        ).strip()
        n = int(n_str) if n_str.isdigit() else None

        record_samples(cls, num_samples=n)

        cont = input("\nRecord another class? (y/n): ").strip().lower()
        if cont != "y":
            break

    # Final stats
    existing = _existing_classes()
    if existing:
        print("\n" + "=" * 50)
        print("  FINAL DATASET STATS")
        print("=" * 50)
        total = 0
        for cls in existing:
            n = _count_samples(cls)
            total += n
            print(f"  {cls:<15} {n:>4} samples")
        print(f"  {'TOTAL':<15} {total:>4} samples")
        print("\nTo retrain: python main.py --kfold")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Record training data via webcam"
    )
    parser.add_argument(
        "--cls", type=str, default=None,
        help="Class name to record (e.g. 'happy')",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of samples to record (default: unlimited)",
    )
    args = parser.parse_args()

    if args.cls:
        record_samples(args.cls.lower().strip(), num_samples=args.n)
    else:
        collect_interactive()
