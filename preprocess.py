"""
Preprocessing: Extract hand landmarks from videos using MediaPipe.
Converts each video into a (NUM_FRAMES, 63) numpy array and saves as .npy.
Uses the new MediaPipe Tasks API (HandLandmarker).
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from config import (
    DATASET_DIR, PROCESSED_DIR, NUM_FRAMES,
    NUM_LANDMARKS, NUM_COORDS, NUM_HANDS,
    LANDMARK_DIM, FRAME_FEAT_DIM,
    VIDEO_EXTENSIONS,
    HAND_LANDMARKER_MODEL, USE_VELOCITY,
)


def create_landmarker(num_hands: int = NUM_HANDS) -> HandLandmarker:
    """
    Create a HandLandmarker instance — single source of truth for all
    MediaPipe settings used during preprocessing, training and webcam
    inference so behaviour is always identical.
    """
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=HAND_LANDMARKER_MODEL
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return HandLandmarker.create_from_options(options)


# Internal alias kept for backward compat
_create_landmarker = create_landmarker


def extract_hand_landmarks(
    video_path: str,
    num_frames: int = NUM_FRAMES,
) -> np.ndarray:
    """
    Extract 21 hand landmarks (x, y, z) from a video.

    Steps:
        1. Read all frames from the video.
        2. Uniformly sample `num_frames` frames.
        3. Run MediaPipe HandLandmarker on each frame.
        4. Flatten 21 landmarks x 3 coords = 63 features.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.

    Returns:
        np.ndarray of shape (num_frames, 63).
        Zero-padded if no hand detected in a frame.
    """
    feat_dim = FRAME_FEAT_DIM             # 126 — both hands
    zero = np.zeros((num_frames, feat_dim), dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return zero

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total = len(frames)
    if total == 0:
        print(f"  [WARN] No frames in: {video_path}")
        return zero

    # Uniformly sample frame indices
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    sampled = [frames[i] for i in indices]

    # Extract landmarks with HandLandmarker (Tasks API)
    # Slot 0 = right hand, slot 1 = left hand  (zeros when absent)
    sequence = np.zeros(
        (num_frames, feat_dim), dtype=np.float32
    )
    landmarker = create_landmarker()

    for i, frame in enumerate(sampled):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb
        )
        result = landmarker.detect(mp_image)

        # Fill right-hand slot (0) and left-hand slot (1)
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
            sequence[i, start:start + LANDMARK_DIM] = np.array(
                coords, dtype=np.float32
            )
        # else: slot stays zero-padded

    landmarker.close()

    # -- Normalize: center on wrist, scale by hand size --
    sequence = _normalize_landmarks(sequence)

    # -- Append velocity (frame-to-frame deltas) --
    if USE_VELOCITY:
        sequence = _add_velocity(sequence)

    return sequence


def _normalize_landmarks(sequence: np.ndarray) -> np.ndarray:
    """
    Normalize each hand slot independently per frame:
      1. Center on wrist (landmark 0) so position-invariant.
      2. Scale by max distance from wrist so size-invariant.
    Handles both hands: slot 0 = right (index 0..62),
                        slot 1 = left  (index 63..125).
    """
    num_frames = sequence.shape[0]
    out = np.zeros_like(sequence)

    for i in range(num_frames):
        for slot in range(NUM_HANDS):
            start = slot * LANDMARK_DIM
            end = start + LANDMARK_DIM
            hand = sequence[i, start:end].reshape(NUM_LANDMARKS, NUM_COORDS)

            # Skip all-zero slots (hand not detected)
            if np.all(hand == 0):
                continue

            # Center on wrist (landmark 0)
            wrist = hand[0].copy()
            hand = hand - wrist

            # Scale by max Euclidean distance from wrist
            dists = np.linalg.norm(hand, axis=1)
            max_dist = dists.max()
            if max_dist > 1e-6:
                hand = hand / max_dist

            out[i, start:end] = hand.flatten()

    return out


def _add_velocity(sequence: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame velocity (deltas) and concatenate with position.

    Args:
        sequence: (num_frames, 63) normalized landmark positions.

    Returns:
        (num_frames, 126) array with [position | velocity] per frame.
    """
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    # First frame velocity stays zero
    return np.concatenate([sequence, velocity], axis=1).astype(np.float32)


def preprocess_dataset() -> dict:
    """
    Walk through the Dataset directory, extract landmarks from every video,
    and save the resulting numpy arrays into the processed/ directory.

    Returns:
        dict mapping class names to number of videos processed.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Discover classes from folder names
    class_folders = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if not class_folders:
        raise FileNotFoundError(f"No class folders found in {DATASET_DIR}")

    print(f"[Preprocess] Found {len(class_folders)} classes: {class_folders}")
    stats = {}

    for cls_folder in class_folders:
        # Strip leading number prefix like "1. " for clean label
        label = cls_folder.split(". ", 1)[-1].strip().lower()
        cls_path = os.path.join(DATASET_DIR, cls_folder)
        save_dir = os.path.join(PROCESSED_DIR, label)
        os.makedirs(save_dir, exist_ok=True)

        # Get video files
        videos = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]
        print(f"  Class '{label}': {len(videos)} videos")

        count = 0
        for vid_file in videos:
            vid_path = os.path.join(cls_path, vid_file)
            sequence = extract_hand_landmarks(vid_path)

            # Save as .npy
            npy_name = os.path.splitext(vid_file)[0] + ".npy"
            save_path = os.path.join(save_dir, npy_name)
            np.save(save_path, sequence)
            count += 1

        stats[label] = count

    print(f"[Preprocess] Done. Saved to: {PROCESSED_DIR}")
    return stats


if __name__ == "__main__":
    stats = preprocess_dataset()
    total = sum(stats.values())
    print(f"\nTotal videos processed: {total}")
    for cls, n in stats.items():
        print(f"  {cls}: {n}")
