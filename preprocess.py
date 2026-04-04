"""
Preprocessing: Extract hand landmarks from videos using MediaPipe.
Per frame feature layout:
    [left_raw, right_raw, left_relative_to_face, right_relative_to_face]
with optional velocity appended after sequence-level normalization.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from config import (
    DATASET_DIR, PROCESSED_DIR, NUM_FRAMES,
    WEBCAM_WIDTH, WEBCAM_HEIGHT, CROP_TO_WEBCAM_SIZE,
    NUM_LANDMARKS, NUM_COORDS, NUM_HANDS,
    LANDMARK_DIM, RAW_FRAME_FEAT_DIM, FRAME_FEAT_DIM,
    PROXIMITY_FEAT_DIM,
    VIDEO_EXTENSIONS,
    HAND_LANDMARKER_MODEL, FACE_LANDMARKER_MODEL,
    USE_VELOCITY,
    USE_FACE_RELATIVE,
    FACE_NOSE_INDEX, FACE_LEFT_EYE_INDEX, FACE_RIGHT_EYE_INDEX,
)


_FACE_WARNING_SHOWN = False


def create_landmarker(
    num_hands: int = NUM_HANDS,
    for_webcam: bool = False,
) -> HandLandmarker:
    """
    Create a HandLandmarker instance — single source of truth for all
    MediaPipe settings used during preprocessing, training and webcam
    inference so behaviour is always identical.

    Args:
        num_hands: Maximum number of hands to detect
        for_webcam: If True, use higher confidence thresholds for speed
    """
    # For webcam: raise confidence to filter false positives and speed up
    min_conf = 0.5 if for_webcam else 0.3
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=HAND_LANDMARKER_MODEL
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
    )
    return HandLandmarker.create_from_options(options)


def create_face_landmarker(for_webcam: bool = False) -> FaceLandmarker | None:
    """
    Create FaceLandmarker (Tasks API) for face-anchor extraction.

    Args:
        for_webcam: If True, use higher confidence thresholds for speed
    """
    global _FACE_WARNING_SHOWN
    try:
        # For webcam: raise confidence to speed up, reduce false positives
        min_conf = 0.6 if for_webcam else 0.3
        
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=FACE_LANDMARKER_MODEL
            ),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_conf,
            min_face_presence_confidence=min_conf,
        )
        return FaceLandmarker.create_from_options(options)
    except Exception:
        if not _FACE_WARNING_SHOWN:
            print(
                "[WARN] FaceLandmarker unavailable; "
                "face-relative features will be zeroed."
            )
            _FACE_WARNING_SHOWN = True
        return None


def create_holistic():
    """Backward-compat alias kept for older imports."""
    return create_face_landmarker()


# Internal alias kept for backward compat
_create_landmarker = create_landmarker


def _crop_frame_to_webcam_size(frame, target_w=WEBCAM_WIDTH, target_h=WEBCAM_HEIGHT):
    """
    Center-crop frame to webcam size (640x480) for training/inference consistency.
    
    If frame is smaller than target, it's resized instead to maintain aspect ratio.
    This ensures all frames processed by MediaPipe have consistent geometry.
    
    Args:
        frame: Input frame (h, w, 3)
        target_w: Target width (640)
        target_h: Target height (480)
    
    Returns:
        Cropped/resized frame (target_h, target_w, 3)
    """
    if frame is None:
        return None
    
    h, w = frame.shape[:2]
    
    # If frame is smaller than target, resize to fit
    if w < target_w or h < target_h:
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Center-crop to target size
    x_offset = (w - target_w) // 2
    y_offset = (h - target_h) // 2
    
    return frame[y_offset:y_offset + target_h, x_offset:x_offset + target_w]


def _extract_face_anchor(face_landmarks):
    """
    Return face center and scale from keypoints.
    center = nose tip (index 1)
    scale  = distance(left eye index 33, right eye index 263)
    """
    if face_landmarks is None or len(face_landmarks) <= FACE_RIGHT_EYE_INDEX:
        return None, 1.0

    nose = face_landmarks[FACE_NOSE_INDEX]
    left_eye = face_landmarks[FACE_LEFT_EYE_INDEX]
    right_eye = face_landmarks[FACE_RIGHT_EYE_INDEX]

    center = np.array([nose.x, nose.y, nose.z], dtype=np.float32)
    left = np.array([left_eye.x, left_eye.y, left_eye.z], dtype=np.float32)
    right = np.array([right_eye.x, right_eye.y, right_eye.z], dtype=np.float32)

    scale = float(np.linalg.norm(left - right))
    if scale < 1e-6:
        scale = 1.0
    return center, scale


def compute_face_relative_features(
    face_landmarks,
    hand_landmarks,
) -> np.ndarray:
    """
    Convert hand landmarks into face-relative coordinates.

    If face or hand is missing, returns all zeros (63 dims).
    """
    if hand_landmarks is None or len(hand_landmarks) == 0:
        return np.zeros(LANDMARK_DIM, dtype=np.float32)

    face_center, scale = _extract_face_anchor(face_landmarks)
    if face_center is None:
        return np.zeros(LANDMARK_DIM, dtype=np.float32)

    out = np.zeros(LANDMARK_DIM, dtype=np.float32)
    for i, lm in enumerate(hand_landmarks):
        base = i * NUM_COORDS
        out[base] = (lm.x - face_center[0]) / scale
        out[base + 1] = (lm.y - face_center[1]) / scale
        out[base + 2] = (lm.z - face_center[2]) / scale
    return out


def extract_landmarks_with_face_relative(
    frame,
    landmarker=None,
    face_landmarker=None,
    hand_result=None,
    face_landmarks=None,
):
    """
    Extract per-frame features in fixed order:
            [left_raw, right_raw, left_relative, right_relative, proximity]

    Returns:
      vec: np.ndarray(FRAME_FEAT_DIM,)
    """
    rgb = None
    if hand_result is None:
        if frame is None or landmarker is None:
            raise ValueError(
                "frame and landmarker are required when hand_result "
                "is not provided"
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = landmarker.detect(mp_image)

    if USE_FACE_RELATIVE and face_landmarks is None:
        if frame is None or face_landmarker is None:
            # No face available: fallback handled by zero relative features.
            face_landmarks = None
        else:
            if rgb is None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_result = face_landmarker.detect(mp_image)
            if face_result.face_landmarks:
                face_landmarks = face_result.face_landmarks[0]

    left_raw = np.zeros(LANDMARK_DIM, dtype=np.float32)
    right_raw = np.zeros(LANDMARK_DIM, dtype=np.float32)
    left_rel = np.zeros(LANDMARK_DIM, dtype=np.float32)
    right_rel = np.zeros(LANDMARK_DIM, dtype=np.float32)

    for hand, handedness_list in zip(
        hand_result.hand_landmarks,
        hand_result.handedness,
    ):
        label = handedness_list[0].display_name  # "Right" or "Left"
        coords = np.array(
            [c for lm in hand for c in (lm.x, lm.y, lm.z)],
            dtype=np.float32,
        )

        if label == "Left":
            left_raw = coords
            if USE_FACE_RELATIVE:
                left_rel = compute_face_relative_features(face_landmarks, hand)
        else:
            right_raw = coords
            if USE_FACE_RELATIVE:
                right_rel = compute_face_relative_features(
                    face_landmarks, hand
                )

    if USE_FACE_RELATIVE:
        face_ok = face_landmarks is not None
        left_present = bool(np.any(left_raw != 0.0))
        right_present = bool(np.any(right_raw != 0.0))

        # Missing face or both hands: keep a neutral distance to avoid
        # over-emphasizing these frames in proximity-biased attention.
        proximity = 1.0
        if face_ok and (left_present or right_present):
            d_left = np.inf
            d_right = np.inf
            if left_present:
                d_left = float(np.linalg.norm(left_rel))
            if right_present:
                d_right = float(np.linalg.norm(right_rel))
            proximity = float(min(d_left, d_right))
            if not np.isfinite(proximity):
                proximity = 1.0

        if PROXIMITY_FEAT_DIM:
            prox_vec = np.array([proximity], dtype=np.float32)
            return np.concatenate(
                [left_raw, right_raw, left_rel, right_rel, prox_vec]
            ).astype(np.float32)

        return np.concatenate(
            [left_raw, right_raw, left_rel, right_rel]
        ).astype(np.float32)

    return np.concatenate([left_raw, right_raw]).astype(np.float32)


def extract_hand_landmarks(
    video_path: str,
    num_frames: int = NUM_FRAMES,
) -> np.ndarray:
    """
    Extract hand landmarks from a video using the configured frame feature
    layout from config.py.

    Steps:
        1. Read all frames from the video.
        2. Uniformly sample `num_frames` frames.
        3. Run MediaPipe HandLandmarker on each frame.
        4. Build per-frame feature vector.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.

    Returns:
        np.ndarray of shape (num_frames, FRAME_FEAT_DIM) before velocity,
        or (num_frames, FRAME_FEAT_DIM * 2) when velocity is enabled.
        Zero-padded if no hand detected in a frame.
    """
    feat_dim = FRAME_FEAT_DIM
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
        # Crop frames to webcam size for consistency
        if CROP_TO_WEBCAM_SIZE:
            frame = _crop_frame_to_webcam_size(frame)
        frames.append(frame)
    cap.release()

    total = len(frames)
    if total == 0:
        print(f"  [WARN] No frames in: {video_path}")
        return zero

    # Uniformly sample frame indices
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    sampled = [frames[i] for i in indices]

    # Extract landmarks per frame using shared helper.
    sequence = np.zeros((num_frames, feat_dim), dtype=np.float32)
    landmarker = create_landmarker()
    face_landmarker = create_face_landmarker()

    for i, frame in enumerate(sampled):
        sequence[i] = extract_landmarks_with_face_relative(
            frame=frame,
            landmarker=landmarker,
            face_landmarker=face_landmarker,
        )

    landmarker.close()
    if face_landmarker is not None:
        face_landmarker.close()

    # -- Normalize: center on wrist, scale by hand size --
    sequence = _normalize_landmarks(sequence)

    # -- Append velocity (frame-to-frame deltas) --
    if USE_VELOCITY:
        sequence = _add_velocity(sequence)

    return sequence


def _normalize_landmarks(sequence: np.ndarray) -> np.ndarray:
    """
    Normalize RAW hand slots independently per frame:
      1. Center on wrist (landmark 0) so position-invariant.
      2. Scale by max distance from wrist so size-invariant.
    Relative face features (if present) are left unchanged.

    RAW slot layout used here: left (0..62), right (63..125).
    """
    num_frames = sequence.shape[0]
    out = np.zeros_like(sequence)
    out[:, RAW_FRAME_FEAT_DIM:] = sequence[:, RAW_FRAME_FEAT_DIM:]

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
        sequence: (num_frames, FRAME_FEAT_DIM) normalized features.

    Returns:
        (num_frames, FRAME_FEAT_DIM * 2) array with
        [position | velocity] per frame.
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
