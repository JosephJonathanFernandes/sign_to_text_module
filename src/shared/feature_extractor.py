"""
Shared feature extraction logic.
This module is the Single Source of Truth for converting raw MediaPipe landmarks
into the 253-dimension (or 506 with velocity) feature vectors used by the ML model.

It ensures that training, testing, and frontend clients all use the exact same logic.
"""

import numpy as np

# Dimension sizes
NUM_LANDMARKS = 21
NUM_COORDS = 3
LANDMARK_DIM = NUM_LANDMARKS * NUM_COORDS  # 63
FACE_NOSE_INDEX = 1
FACE_LEFT_EYE_INDEX = 33
FACE_RIGHT_EYE_INDEX = 263

def normalize_hand_landmarks(hand_raw: np.ndarray) -> np.ndarray:
    """
    Normalize a raw 63-dim hand array.
    1. Center on wrist (landmark 0).
    2. Scale by max Euclidean distance from wrist.
    
    Args:
        hand_raw: (63,) float32 array of [x1,y1,z1, x2,y2,z2, ...]
    
    Returns:
        (63,) float32 array normalized.
    """
    if hand_raw is None or len(hand_raw) != LANDMARK_DIM:
        return np.zeros(LANDMARK_DIM, dtype=np.float32)
        
    # Check if hand is entirely zeros
    if not np.any(hand_raw):
        return np.zeros(LANDMARK_DIM, dtype=np.float32)
        
    hand_reshaped = hand_raw.reshape((NUM_LANDMARKS, NUM_COORDS)).copy()
    
    # 1. Center on wrist (landmark 0)
    wrist = hand_reshaped[0].copy()
    hand_reshaped = hand_reshaped - wrist
    
    # 2. Scale by max Euclidean distance from wrist
    dists = np.linalg.norm(hand_reshaped, axis=1)
    max_dist = dists.max()
    if max_dist > 1e-6:
        hand_reshaped = hand_reshaped / max_dist
        
    return hand_reshaped.flatten().astype(np.float32)

def extract_face_anchor(face_raw: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Extract face center and scale from flat raw face landmarks.
    Expected face_raw is an array of [x,y,z] flattened, with at least up to index 263.
    """
    if face_raw is None or len(face_raw) <= FACE_RIGHT_EYE_INDEX * NUM_COORDS:
        return None, 1.0
        
    nose_base = FACE_NOSE_INDEX * NUM_COORDS
    l_eye_base = FACE_LEFT_EYE_INDEX * NUM_COORDS
    r_eye_base = FACE_RIGHT_EYE_INDEX * NUM_COORDS
    
    center = face_raw[nose_base:nose_base+3]
    left = face_raw[l_eye_base:l_eye_base+3]
    right = face_raw[r_eye_base:r_eye_base+3]
    
    scale = float(np.linalg.norm(left - right))
    if scale < 1e-6:
        scale = 1.0
        
    return center, scale

def compute_face_relative(face_raw: np.ndarray, hand_raw: np.ndarray) -> np.ndarray:
    """
    Convert flat raw hand landmarks into face-relative coordinates.
    """
    if hand_raw is None or not np.any(hand_raw):
        return np.zeros(LANDMARK_DIM, dtype=np.float32)
        
    center, scale = extract_face_anchor(face_raw)
    if center is None:
        return np.zeros(LANDMARK_DIM, dtype=np.float32)
        
    hand_reshaped = hand_raw.reshape((NUM_LANDMARKS, NUM_COORDS))
    rel = (hand_reshaped - center) / scale
    
    return rel.flatten().astype(np.float32)

def build_single_frame_features(left_raw: np.ndarray | None, right_raw: np.ndarray | None, face_raw: np.ndarray | None) -> np.ndarray:
    """
    Build the exact 253-dimension feature vector for a single frame.
    
    Args:
        left_raw: (63,) float array or None
        right_raw: (63,) float array or None
        face_raw: (N*3,) float array or None
        
    Returns:
        (253,) float32 array
    """
    left_raw = left_raw if left_raw is not None else np.zeros(LANDMARK_DIM, dtype=np.float32)
    right_raw = right_raw if right_raw is not None else np.zeros(LANDMARK_DIM, dtype=np.float32)
    
    # 1. Normalize hands (internally centered on wrist and scaled)
    left_norm = normalize_hand_landmarks(left_raw)
    right_norm = normalize_hand_landmarks(right_raw)
    
    # 2. Face relative logic
    left_rel = np.zeros(LANDMARK_DIM, dtype=np.float32)
    right_rel = np.zeros(LANDMARK_DIM, dtype=np.float32)
    
    if face_raw is not None:
        left_rel = compute_face_relative(face_raw, left_raw)
        right_rel = compute_face_relative(face_raw, right_raw)
        
    # 3. Proximity
    proximity = 1.0
    face_ok = face_raw is not None
    left_present = np.any(left_raw)
    right_present = np.any(right_raw)
    
    if face_ok and (left_present or right_present):
        d_left = float(np.linalg.norm(left_rel)) if left_present else np.inf
        d_right = float(np.linalg.norm(right_rel)) if right_present else np.inf
        proximity = float(min(d_left, d_right))
        if not np.isfinite(proximity):
            proximity = 1.0
            
    prox_vec = np.array([proximity], dtype=np.float32)
    
    # 4. Concatenate: [left_norm (63), right_norm (63), left_rel (63), right_rel (63), proximity (1)] = 253
    vec = np.concatenate([left_norm, right_norm, left_rel, right_rel, prox_vec]).astype(np.float32)
    return vec
