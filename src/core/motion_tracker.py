import numpy as np

def calculate_hand_motion(wrist_pos, wrist_history, motion_magnitude, motion_smoothing):
    """Calculate exponential moving average of hand motion velocity."""
    if not wrist_history:
        return motion_magnitude
    
    last_pos = wrist_history[-1]
    dx = wrist_pos[0] - last_pos[0]
    dy = wrist_pos[1] - last_pos[1]
    current_motion = (dx**2 + dy**2)**0.5
    
    # Exponential moving average
    new_motion = motion_smoothing * current_motion + (1 - motion_smoothing) * motion_magnitude
    return new_motion

def detect_hand_drift(hand_infos, wrist_history, frame_shape, min_jump_px, jump_ratio):
    """Detect suspicious wrist jumps that usually indicate stale tracking."""
    if not hand_infos or len(wrist_history) == 0:
        return False

    current_wrists = []
    for info in hand_infos:
        landmarks = info.get("landmarks")
        if landmarks:
            # wrist is landmark 0
            current_wrists.append((int(landmarks[0].x * frame_shape[1]), int(landmarks[0].y * frame_shape[0])))

    if not current_wrists:
        return False

    current_center = (
        sum(point[0] for point in current_wrists) / len(current_wrists),
        sum(point[1] for point in current_wrists) / len(current_wrists),
    )
    previous_center = wrist_history[-1]
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    jump = (dx * dx + dy * dy) ** 0.5

    frame_diag = (frame_shape[1] ** 2 + frame_shape[0] ** 2) ** 0.5
    jump_threshold = max(min_jump_px, frame_diag * jump_ratio)
    return jump > jump_threshold

def calculate_dynamic_threshold(motion_magnitude, stability_counter, is_transition, base_threshold, dynamic_enabled, transition_hysteresis, motion_threshold, motion_boost, stability_boost, min_threshold):
    """Calculate adaptive confidence threshold based on motion and stability."""
    if not dynamic_enabled:
        return base_threshold
    
    threshold = base_threshold
    
    # Boost threshold temporarily during transitions (require high confidence)
    if is_transition:
        threshold += transition_hysteresis
    
    # Reduce threshold when motion is detected (high motion = easier to detect)
    if motion_magnitude > motion_threshold:
        motion_ratio = min(motion_magnitude / (motion_threshold * 2), 1.0)
        threshold -= motion_boost * motion_ratio
    
    # Reduce threshold as sign becomes more stable
    if stability_counter > 2:
        stability_ratio = min(stability_counter / 8.0, 1.0)
        threshold -= stability_boost * stability_ratio
    
    # Floor to minimum threshold
    return max(threshold, min_threshold)

def is_motion_gating_active(motion_magnitude, frames_in_motion, gating_enabled, motion_threshold):
    """Determine if we should gate (suppress) predictions based on motion."""
    if not gating_enabled:
        return False
    
    # Consider motion active if recent motion magnitude exceeds threshold
    # OR if we've seen motion recently (momentum)
    has_current_motion = motion_magnitude > motion_threshold
    has_recent_motion = frames_in_motion > 0
    
    # Gate (suppress) when NO motion at all
    return not (has_current_motion or has_recent_motion)

def compute_transition_movement(current_landmarks, previous_landmarks):
    """Compute mean absolute frame-to-frame landmark movement."""
    if previous_landmarks is None:
        return 0.0
    return float(np.mean(np.abs(current_landmarks - previous_landmarks)))
