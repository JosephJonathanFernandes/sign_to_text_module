import numpy as np
from src.shared.feature_extractor import build_single_frame_features

def old_logic(left_raw, right_raw, face_raw):
    # This reproduces the old _normalize_landmarks and compute_face_relative_features
    # EXACTLY as they were before the refactor.
    
    # 1. Face relative
    left_rel = np.zeros(63, dtype=np.float32)
    right_rel = np.zeros(63, dtype=np.float32)
    
    if face_raw is not None:
        nose = face_raw[1*3:1*3+3]
        l_eye = face_raw[33*3:33*3+3]
        r_eye = face_raw[263*3:263*3+3]
        scale = np.linalg.norm(l_eye - r_eye)
        if scale < 1e-6: scale = 1.0
        
        if np.any(left_raw):
            left_rel = (left_raw.reshape(21, 3) - nose) / scale
            left_rel = left_rel.flatten()
        if np.any(right_raw):
            right_rel = (right_raw.reshape(21, 3) - nose) / scale
            right_rel = right_rel.flatten()

    # 2. Proximity
    proximity = 1.0
    if face_raw is not None and (np.any(left_raw) or np.any(right_raw)):
        d_left = np.linalg.norm(left_rel) if np.any(left_raw) else np.inf
        d_right = np.linalg.norm(right_rel) if np.any(right_raw) else np.inf
        proximity = min(d_left, d_right)
        if not np.isfinite(proximity):
            proximity = 1.0
            
    # 3. Normalization of RAW
    def norm_hand(hand):
        if not np.any(hand):
            return np.zeros(63, dtype=np.float32)
        h = hand.reshape(21, 3).copy()
        wrist = h[0].copy()
        h = h - wrist
        dists = np.linalg.norm(h, axis=1)
        m = dists.max()
        if m > 1e-6:
            h = h / m
        return h.flatten()
        
    left_norm = norm_hand(left_raw)
    right_norm = norm_hand(right_raw)
    
    return np.concatenate([
        left_norm, right_norm, left_rel, right_rel, [proximity]
    ]).astype(np.float32)


def test_drift():
    # Mock some raw landmarks
    left = np.random.rand(63).astype(np.float32)
    right = np.zeros(63).astype(np.float32) # Missing right hand
    
    face = np.random.rand(264*3).astype(np.float32)
    
    old_features = old_logic(left, right, face)
    new_features = build_single_frame_features(left, right, face)
    
    mae = np.mean(np.abs(old_features - new_features))
    print(f"MAE between old and new logic: {mae}")
    
    assert mae < 1e-8, f"Drift detected! MAE={mae}"
    print("Verification passed! No drift detected.")

if __name__ == "__main__":
    test_drift()
