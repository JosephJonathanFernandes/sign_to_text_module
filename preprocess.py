"""
Preprocessing: Extract hand landmarks from videos using MediaPipe.
Per frame feature layout:
    [left_raw, right_raw, left_relative_to_face, right_relative_to_face]
with optional velocity appended after sequence-level normalization.
"""

import argparse
import hashlib
import os
import random
import shutil
import time
from typing import Callable

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from pipeline_logger import PipelineLogger
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from config import get_config

cfg = get_config()

# Convenience references for preprocessing
DATASET_DIR = cfg.paths.dataset_dir
PROCESSED_DIR = cfg.paths.processed_dir
NUM_FRAMES = cfg.preprocessing.num_frames
WEBCAM_WIDTH = cfg.preprocessing.webcam_width
WEBCAM_HEIGHT = cfg.preprocessing.webcam_height
CROP_TO_WEBCAM_SIZE = cfg.preprocessing.crop_to_webcam_size
NUM_LANDMARKS = cfg.landmarks.num_landmarks
NUM_COORDS = cfg.landmarks.num_coords
NUM_HANDS = cfg.landmarks.num_hands
LANDMARK_DIM = cfg.landmarks.landmark_dim_per_hand
RAW_FRAME_FEAT_DIM = cfg.landmarks.raw_frame_features_dim
FRAME_FEAT_DIM = cfg.frame_features.frame_features_dim
PROXIMITY_FEAT_DIM = cfg.spatial.proximity_dim
VIDEO_EXTENSIONS = cfg.paths.video_extensions
HAND_LANDMARKER_MODEL = cfg.paths.hand_landmarker_model
FACE_LANDMARKER_MODEL = cfg.paths.face_landmarker_model
USE_VELOCITY = cfg.frame_features.use_velocity
USE_FACE_RELATIVE = cfg.spatial.use_face_relative
FACE_NOSE_INDEX = cfg.preprocessing.face_nose_index
FACE_LEFT_EYE_INDEX = cfg.preprocessing.face_left_eye_index
FACE_RIGHT_EYE_INDEX = cfg.preprocessing.face_right_eye_index


_FACE_WARNING_SHOWN = False
AUGMENTED_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "augmented_dataset",
)
AUGMENTABLE_VIDEO_EXTENSIONS = (".mp4", ".mov")
VIDEO_AUGMENT_WIDTH = 224
VIDEO_AUGMENT_HEIGHT = 224
# Updated maximum per-video variants: 3 spatial-only + 3 * (number of effects)
# Effects expanded to include motion/defocus blur, JPEG artifacts, gamma, white-balance,
# perspective warp and temporal_jitter. Current computed max = 3 + 3*17 = 54
VIDEO_AUGMENT_MAX_PER_VIDEO = 54
VIDEO_AUGMENT_MAX_PER_CLASS = 900

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit="item", leave=False)


def _is_augmentable_video(filename: str) -> bool:
    return filename.lower().endswith(AUGMENTABLE_VIDEO_EXTENSIONS)


def _copy_original_video(src_path: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)


def _build_square_crop(
    frame: np.ndarray,
    mode: str,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Crop a frame to a square window and resize to the webcam target size."""
    height, width = frame.shape[:2]
    crop_size = min(height, width)

    if crop_size <= 0:
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    x_slack = width - crop_size
    y_slack = height - crop_size

    if mode == "center":
        x0 = int(round(x_slack * 0.5)) if x_slack > 0 else 0
    elif mode == "left":
        x0 = int(round(x_slack * 0.15)) if x_slack > 0 else 0
    elif mode == "right":
        x0 = int(round(x_slack * 0.85)) if x_slack > 0 else 0
    else:
        raise ValueError(f"Unknown crop mode: {mode}")

    y0 = int(round(y_slack * 0.5)) if y_slack > 0 else 0

    x0 = max(0, min(x0, width - crop_size))
    y0 = max(0, min(y0, height - crop_size))

    cropped = frame[y0:y0 + crop_size, x0:x0 + crop_size]
    interpolation = cv2.INTER_LANCZOS4
    return cv2.resize(cropped, (target_width, target_height), interpolation=interpolation)


def _apply_visual_effect(
    frame: np.ndarray,
    effect_name: str,
    rng: random.Random,
) -> np.ndarray:
    """Apply one mild visual effect to a single frame."""
    if effect_name == "noise":
        sigma = rng.uniform(2.0, 5.0)
        noise_map = np.random.normal(0.0, sigma, frame.shape).astype(np.float32)
        out = frame.astype(np.float32) + noise_map
        return np.clip(out, 0, 255).astype(np.uint8)

    if effect_name == "brightness":
        alpha = rng.uniform(0.92, 1.08)
        beta = rng.randint(-8, 8)
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    if effect_name == "contrast":
        alpha = rng.uniform(0.85, 1.15)
        beta = 0
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    if effect_name == "color_jitter":
        # Add small random noise to each RGB channel independently
        b, g, r = cv2.split(frame)
        for channel in [b, g, r]:
            noise = np.random.normal(0, rng.uniform(3, 8), channel.shape).astype(np.float32)
            channel_out = channel.astype(np.float32) + noise
            channel[:] = np.clip(channel_out, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])

    if effect_name == "scale":
        # Random zoom in/out (0.85 to 1.15x)
        scale = rng.uniform(0.85, 1.15)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    if effect_name == "rotation":
        # Subtle rotation ±3 degrees
        angle = rng.uniform(-3, 3)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    if effect_name == "hue":
        # Random hue shift in HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_shift = rng.uniform(-15, 15)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
        return cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2BGR)

    if effect_name == "fog":
        # Simulate fog by adding white Gaussian noise and reducing contrast
        fog_intensity = rng.uniform(0.15, 0.35)
        fog_layer = np.random.normal(200, 30, frame.shape).astype(np.float32)
        frame_float = frame.astype(np.float32)
        # Blend frame with fog
        fogged = frame_float * (1.0 - fog_intensity) + fog_layer * fog_intensity
        return np.clip(fogged, 0, 255).astype(np.uint8)

    if effect_name == "pixel_dropout":
        # Random pixel dropout (set random pixels to 0 or blur them)
        frame_out = frame.copy()
        h, w = frame.shape[:2]
        dropout_rate = rng.uniform(0.02, 0.08)  # 2-8% of pixels
        num_pixels = int(h * w * dropout_rate)
        for _ in range(num_pixels):
            y = rng.randint(0, h - 1)
            x = rng.randint(0, w - 1)
            frame_out[y, x] = [rng.randint(0, 255) for _ in range(3)]
        return frame_out

    if effect_name == "coarse_dropout":
        # Drop random patches (like DropBlock regularization)
        frame_out = frame.copy().astype(np.float32)
        h, w = frame.shape[:2]
        block_size = rng.randint(8, 24)
        num_blocks = rng.randint(2, 6)
        for _ in range(num_blocks):
            y = rng.randint(0, max(1, h - block_size))
            x = rng.randint(0, max(1, w - block_size))
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            frame_out[y:y_end, x:x_end] = rng.randint(100, 200)
        return np.clip(frame_out, 0, 255).astype(np.uint8)

    # --- New effects added ---
    if effect_name == "motion_blur":
        # Linear motion blur kernel with random length and angle
        ksize = rng.randint(3, 15)
        if ksize % 2 == 0:
            ksize += 1
        # create a vertical kernel and rotate it
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[:, ksize // 2] = 1.0 / ksize
        angle = rng.uniform(0, 360)
        # rotate kernel via warpAffine
        M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1.0)
        kernel_rot = cv2.warpAffine(kernel, M, (ksize, ksize))
        frame_blur = cv2.filter2D(frame, -1, kernel_rot)
        return np.clip(frame_blur, 0, 255).astype(np.uint8)

    if effect_name == "defocus_blur":
        # Simulate defocus by applying a larger Gaussian blur
        k = rng.randint(3, 11)
        if k % 2 == 0:
            k += 1
        sigma = rng.uniform(0.8, 2.5)
        return cv2.GaussianBlur(frame, (k, k), sigma)

    if effect_name == "jpeg_artifact":
        # Encode/decode with low JPEG quality to introduce blocky artifacts
        quality = rng.randint(25, 65)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, encimg = cv2.imencode('.jpg', frame, encode_param)
        if not ret:
            return frame
        decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        return decimg

    if effect_name == "gamma":
        # Gamma correction (gamma <1 brighter, >1 darker)
        g = rng.uniform(0.7, 1.4)
        inv = 1.0 / g
        table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype('uint8')
        return cv2.LUT(frame, table)

    if effect_name == "white_balance":
        # Approximate white-balance by scaling individual channels
        b_mul = rng.uniform(0.92, 1.12)
        g_mul = rng.uniform(0.92, 1.12)
        r_mul = rng.uniform(0.92, 1.12)
        b, g, r = cv2.split(frame.astype(np.float32))
        b = np.clip(b * b_mul, 0, 255).astype(np.uint8)
        g = np.clip(g * g_mul, 0, 255).astype(np.uint8)
        r = np.clip(r * r_mul, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])

    if effect_name == "perspective_warp":
        # Mild perspective warp by perturbing corner points slightly
        h, w = frame.shape[:2]
        max_shift_x = int(w * 0.06)
        max_shift_y = int(h * 0.06)
        src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst_pts = src_pts.copy()
        dst_pts[0] += [rng.randint(-max_shift_x, max_shift_x), rng.randint(-max_shift_y, max_shift_y)]
        dst_pts[1] += [rng.randint(-max_shift_x, max_shift_x), rng.randint(-max_shift_y, max_shift_y)]
        dst_pts[2] += [rng.randint(-max_shift_x, max_shift_x), rng.randint(-max_shift_y, max_shift_y)]
        dst_pts[3] += [rng.randint(-max_shift_x, max_shift_x), rng.randint(-max_shift_y, max_shift_y)]
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return warped

    raise ValueError(f"Unknown effect: {effect_name}")


def _make_augmented_variants(
    augment_count: int,
    seed: int,
    target_width: int,
    target_height: int,
) -> list[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """Build all combination variants: 3 crops × 10 effects = 30 total.
    
    Systematically covers ALL combinations:
    - Variants 1-3: Spatial only (center, left, right crops)
    - Variants 4-33: All 3 crops × 10 effects (center+effect, left+effect, right+effect)
    
    Order:
    - Variants 4-13: center crop + each effect (brightness through coarse_dropout)
    - Variants 14-23: left crop + each effect
    - Variants 24-33: right crop + each effect
    """
    augment_count = max(0, min(augment_count, 33))  # Support up to 33 variants
    if augment_count <= 0:
        return []

    rng = random.Random(seed)
    variants: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
    
    # All effects in deterministic order
    all_effects = (
        "brightness", "contrast", "hue", "fog", "rotation",
        "scale", "color_jitter", "noise", "pixel_dropout", "coarse_dropout",
        # Added effects
        "motion_blur", "defocus_blur", "jpeg_artifact", "gamma", "white_balance", "perspective_warp", "temporal_jitter",
    )
    
    # All crop positions
    all_crops = ("center", "left", "right")

    # Variant 1-3: Spatial only (baseline variants)
    for crop_idx, crop_pos in enumerate(all_crops, 1):
        if augment_count >= crop_idx:
            def _spatial_only(
                frame: np.ndarray, 
                pos: str = crop_pos
            ) -> np.ndarray:
                return _build_square_crop(frame, pos, target_width, target_height)
            variants.append((f"aug{crop_idx}", _spatial_only))
    
    # Variants 4-33: All crop × effect combinations
    variant_idx = 4
    for crop_pos in all_crops:
        for effect_name in all_effects:
            if variant_idx > augment_count:
                break
            # For temporal_jitter we need a stateful per-transform closure
            if effect_name == "temporal_jitter":
                # Create a stateful jitter transform that may duplicate or blur adjacent frames
                def make_jitter(local_rng: random.Random, crop: str):
                    state = {"prev": None}

                    def _jitter(frame: np.ndarray) -> np.ndarray:
                        base = _build_square_crop(frame, crop, target_width, target_height)
                        p = local_rng.random()
                        if state["prev"] is None:
                            out = base
                        else:
                            if p < 0.06:
                                # duplicate previous (temporal duplicate)
                                out = state["prev"].copy()
                            elif p < 0.10:
                                # simulate drop by heavy blur
                                out = cv2.GaussianBlur(base, (5, 5), 0)
                            else:
                                out = base
                        state["prev"] = out.copy()
                        return out

                    return _jitter

                variants.append((f"aug{variant_idx}", make_jitter(rng, crop_pos)))
            else:
                # Create closure with current crop_pos and effect_name
                def _aug_crop_effect(
                    frame: np.ndarray,
                    crop: str = crop_pos,
                    effect: str = effect_name,
                    local_rng: random.Random = rng
                ) -> np.ndarray:
                    base = _build_square_crop(frame, crop, target_width, target_height)
                    return _apply_visual_effect(base, effect, local_rng)

                variants.append((f"aug{variant_idx}", _aug_crop_effect))
            variant_idx += 1
        
        if variant_idx > augment_count:
            break

    return variants


def _write_augmented_video(
    video_path: str,
    output_paths: list[str],
    transforms: list[Callable[[np.ndarray], np.ndarray]],
    target_width: int,
    target_height: int,
) -> None:
    """Stream a video once and write all augmented outputs frame-by-frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open for augmentation: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = []
    try:
        for output_path in output_paths:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            writers.append(
                cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (target_width, target_height),
                )
            )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for writer, transform in zip(writers, transforms):
                writer.write(transform(frame))
    finally:
        cap.release()
        for writer in writers:
            writer.release()


def augment_video_dataset(
    input_dir: str = DATASET_DIR,
    output_dir: str = AUGMENTED_DATASET_DIR,
    max_videos_per_class: int = VIDEO_AUGMENT_MAX_PER_CLASS,
    max_augments_per_video: int = VIDEO_AUGMENT_MAX_PER_VIDEO,
    target_width: int = VIDEO_AUGMENT_WIDTH,
    target_height: int = VIDEO_AUGMENT_HEIGHT,
    clear_output: bool = False,
    pipeline_log: PipelineLogger | None = None,
    class_only: str | None = None,
) -> dict:
    """
    Build a controlled augmented video dataset from class folders.

    Copies every original .mp4/.mov file into output_dir and generates up to
    four additional, separate augmentations per original video until the per-
    class threshold is reached.
    
    Args:
        class_only: If set, only augment this specific class folder.
    """
    if pipeline_log:
        pipeline_log.event("augment_start", input_dir=input_dir, output_dir=output_dir, max_videos_per_class=max_videos_per_class, class_only=class_only)
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    if input_dir == output_dir:
        raise ValueError("input_dir and output_dir must be different")

    if clear_output and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    class_folders = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])
    if not class_folders:
        raise FileNotFoundError(f"No class folders found in {input_dir}")
    
    if class_only:
        # Support both exact match ("36. light") and partial match ("light")
        matching = [c for c in class_folders if c == class_only or class_only in c]
        if not matching:
            raise ValueError(f"Class '{class_only}' not found in {input_dir}. Available: {', '.join(class_folders[:5])}...")
        class_folders = matching

    print(f"[VideoAug] Found {len(class_folders)} classes")
    stats = {}

    for cls_folder in _progress(class_folders, "Classes"):
        cls_in = os.path.join(input_dir, cls_folder)
        cls_out = os.path.join(output_dir, cls_folder)
        os.makedirs(cls_out, exist_ok=True)

        videos = sorted([
            f for f in os.listdir(cls_in)
            if os.path.isfile(os.path.join(cls_in, f)) and _is_augmentable_video(f)
        ])

        originals_copied = 0
        augmented_written = 0
        remaining_aug_slots = max(0, max_videos_per_class - len(videos))

        print(f"  Class '{cls_folder}': {len(videos)} source videos")
        if len(videos) >= max_videos_per_class:
            print(
                f"    [VideoAug] Originals already meet/exceed the class cap "
                f"({max_videos_per_class}); skipping augmentation"
            )

        for index, vid_file in enumerate(_progress(videos, cls_folder)):
            src_path = os.path.join(cls_in, vid_file)
            dst_path = os.path.join(cls_out, vid_file)
            _copy_original_video(src_path, dst_path)
            originals_copied += 1

            aug_budget = min(max_augments_per_video, remaining_aug_slots)
            if aug_budget <= 0:
                continue

            stem, _ = os.path.splitext(vid_file)
            seed_bytes = f"{cls_folder}/{vid_file}/{index}".encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "big")
            variants = _make_augmented_variants(
                aug_budget,
                seed=seed,
                target_width=target_width,
                target_height=target_height,
            )
            if not variants:
                continue

            output_paths = [os.path.join(cls_out, f"{stem}_{name}.mp4") for name, _ in variants]
            transforms = [transform for _, transform in variants]
            _write_augmented_video(
                src_path,
                output_paths,
                transforms,
                target_width,
                target_height,
            )
            augmented_written += len(variants)
            remaining_aug_slots -= len(variants)

        stats[cls_folder] = {
            "originals_copied": originals_copied,
            "augmented_written": augmented_written,
            "total_output": originals_copied + augmented_written,
        }
        print(
            f"    [VideoAug] copied={originals_copied}, "
            f"augmented={augmented_written}, total={originals_copied + augmented_written}"
        )
    if pipeline_log:
        pipeline_log.event(
            "augment_class_complete",
            class_name=cls_folder,
            originals_copied=originals_copied,
            augmented_written=augmented_written,
            total_output=originals_copied + augmented_written,
        )

    print(f"[VideoAug] Done. Saved to: {output_dir}")
    if pipeline_log:
        pipeline_log.event("augment_end", output_dir=output_dir, total_classes=len(class_folders), stats=stats)
    return stats


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


def preprocess_dataset(input_dir: str = DATASET_DIR, pipeline_log: PipelineLogger | None = None) -> dict:
    """
    Walk through a class-wise video directory, extract landmarks from every
    supported video, and save the resulting numpy arrays into processed/.

    Returns:
        dict mapping class names to number of videos processed.
    """
    if pipeline_log:
        pipeline_log.event("preprocess_start", input_dir=input_dir, output_dir=PROCESSED_DIR)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    input_dir = os.path.abspath(input_dir)

    # Discover classes from folder names
    class_folders = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    if not class_folders:
        raise FileNotFoundError(f"No class folders found in {input_dir}")

    print(f"[Preprocess] Found {len(class_folders)} classes: {class_folders}")
    print(f"[Preprocess] Reading videos from: {input_dir}")
    if pipeline_log:
        pipeline_log.event("preprocess_classes_discovered", num_classes=len(class_folders), class_list=class_folders)
    stats = {}

    for cls_folder in class_folders:
        # Strip leading number prefix like "1. " for clean label
        label = cls_folder.split(". ", 1)[-1].strip().lower()
        cls_path = os.path.join(input_dir, cls_folder)
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
            if pipeline_log:
                pipeline_log.event(
                    "preprocess_file_saved",
                    class_name=label,
                    video_file=vid_file,
                    npy_file=npy_name,
                    sequence_shape=str(sequence.shape),
                )

        stats[label] = count
        if pipeline_log:
            pipeline_log.event("preprocess_class_complete", class_name=label, video_count=count)

    print(f"[Preprocess] Done. Saved to: {PROCESSED_DIR}")
    if pipeline_log:
        total = sum(stats.values())
        pipeline_log.event("preprocess_end", output_dir=PROCESSED_DIR, total_videos=total, stats=stats)
    return stats


if __name__ == "__main__":
    stats = preprocess_dataset()
    total = sum(stats.values())
    print(f"\nTotal videos processed: {total}")
    for cls, n in stats.items():
        print(f"  {cls}: {n}")
