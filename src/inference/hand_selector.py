"""
Hand Selection and Filtering Module for Real-Time Sign Language Recognition.

Implements single-person hand detection and selection using MediaPipe Face and Hand
landmarks. Ensures only one person's hands are used for inference, filtering out
hands from other people in multi-person scenarios.

Core Functions:
    - compute_face_center(): Extract face center from face landmarks
    - compute_hand_centers(): Get center position for each detected hand
    - filter_hands_by_distance(): Distance-based hand filtering to face
    - filter_hands_by_roi(): Region-of-Interest based filtering
    - select_hands(): Select at most 2 hands closest to face
    - assign_hand_sides(): Assign left/right based on horizontal position
    - process_hands(): Main integration function (ready for inference loop)
    - horizontally_flip_hand(): Support for left-handed users (optional)

Usage Example:
    >>> from src.inference.hand_selector import HandSelector
    >>> selector = HandSelector(
    ...     distance_threshold=300,  # pixels
    ...     roi_width_ratio=0.5,
    ...     roi_height_ratio=0.5,
    ... )
    >>> result = selector.process_hands(
    ...     face_landmarks=face_lms,
    ...     hand_landmarks=hand_lms_list,
    ...     frame_shape=(480, 640)
    ... )
    >>> left_hand = result['left_hand']
    >>> right_hand = result['right_hand']
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class HandSelector:
    """
    Selects and filters hands from a multi-person scene using face-based anchoring.
    
    Ensures robust single-person hand selection by:
    1. Locating the face center
    2. Filtering hands based on proximity to face
    3. Selecting at most 2 hands (one per side)
    4. Assigning consistent left/right labels
    
    Attributes:
        distance_threshold (float): Max distance (pixels) from face to hand center
        roi_width_ratio (float): ROI width as ratio of frame width (0-1)
        roi_height_ratio (float): ROI height as ratio of frame height (0-1)
        use_roi_filtering (bool): Enable ROI-based filtering (vs pure distance)
        enable_debugging (bool): Log debug information
    """
    
    def __init__(
        self,
        distance_threshold: float = 300.0,
        roi_width_ratio: float = 0.5,
        roi_height_ratio: float = 0.5,
        use_roi_filtering: bool = True,
        enable_debugging: bool = False,
    ):
        """
        Initialize hand selector.
        
        Args:
            distance_threshold (float): Max Euclidean distance from face to hand center.
                                       Typical: 200-400 pixels depending on frame size.
                                       Default: 300.
            roi_width_ratio (float): ROI width as fraction of frame width. Default: 0.5.
            roi_height_ratio (float): ROI height as fraction of frame height. Default: 0.5.
            use_roi_filtering (bool): Use ROI method instead of pure distance. Default: True.
            enable_debugging (bool): Print debug logs. Default: False.
        """
        if not (0 < roi_width_ratio <= 1):
            raise ValueError(f"roi_width_ratio must be in (0, 1], got {roi_width_ratio}")
        if not (0 < roi_height_ratio <= 1):
            raise ValueError(f"roi_height_ratio must be in (0, 1], got {roi_height_ratio}")
        if distance_threshold < 0:
            raise ValueError(f"distance_threshold must be non-negative, got {distance_threshold}")
        
        self.distance_threshold = float(distance_threshold)
        self.roi_width_ratio = float(roi_width_ratio)
        self.roi_height_ratio = float(roi_height_ratio)
        self.use_roi_filtering = bool(use_roi_filtering)
        self.enable_debugging = bool(enable_debugging)
    
    def process_hands(
        self,
        face_landmarks: Optional[np.ndarray],
        hand_landmarks: Optional[List[np.ndarray]],
        frame_shape: Tuple[int, int],
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Process and select hands from detected faces and hands in frame.
        
        Complete pipeline:
        1. Validate face detection
        2. Compute face center
        3. Compute hand centers
        4. Filter hands by proximity
        5. Select at most 2 hands
        6. Assign left/right
        7. Return structured output
        
        Args:
            face_landmarks (np.ndarray or None): Face landmarks shape (468, 3) from MediaPipe.
                                                None if no face detected.
            hand_landmarks (List[np.ndarray] or None): List of hand landmark arrays,
                                                       each shape (21, 3).
                                                       None or [] if no hands detected.
            frame_shape (Tuple[int, int]): Frame dimensions (height, width) in pixels.
        
        Returns:
            Dict containing:
                'left_hand': np.ndarray (21, 3) or None
                'right_hand': np.ndarray (21, 3) or None
                'face_center': Tuple[float, float] or None
                'selected_hand_indices': List[int] - original indices of selected hands
                'filtered_hands': List of hand indices that passed filtering
        
        Examples:
            result = selector.process_hands(face_lms, hand_lms_list, (480, 640))
            if result['left_hand'] is not None:
                left_hand = result['left_hand']  # Shape (21, 3)
        """
        result = {
            'left_hand': None,
            'right_hand': None,
            'face_center': None,
            'selected_hand_indices': [],
            'filtered_hands': [],
        }
        
        # Edge case: no face detected
        if face_landmarks is None or len(face_landmarks) == 0:
            if self.enable_debugging:
                print("[HandSelector] No face detected, skipping hand selection")
            return result
        
        # Edge case: no hands detected
        if hand_landmarks is None or len(hand_landmarks) == 0:
            if self.enable_debugging:
                print("[HandSelector] No hands detected")
            return result
        
        # Step 1: Compute face center
        face_center = self._compute_face_center(face_landmarks)
        result['face_center'] = face_center
        
        if self.enable_debugging:
            print(f"[HandSelector] Face center: {face_center}")
        
        # Step 2: Compute hand centers
        hand_centers = self._compute_hand_centers(hand_landmarks)
        
        # Step 3: Filter hands by proximity or ROI
        if self.use_roi_filtering:
            filtered_indices = self._filter_hands_by_roi(
                hand_centers, face_center, frame_shape
            )
        else:
            filtered_indices = self._filter_hands_by_distance(
                hand_centers, face_center, self.distance_threshold
            )
        
        result['filtered_hands'] = filtered_indices
        
        if self.enable_debugging:
            print(f"[HandSelector] Filtered to {len(filtered_indices)} hands: {filtered_indices}")
        
        # Edge case: no hands passed filtering
        if len(filtered_indices) == 0:
            if self.enable_debugging:
                print("[HandSelector] No hands passed proximity filtering")
            return result
        
        # Step 4: Select at most 2 closest hands
        selected_indices = self._select_hands(
            hand_centers, filtered_indices, face_center, max_hands=2
        )
        result['selected_hand_indices'] = selected_indices
        
        if self.enable_debugging:
            print(f"[HandSelector] Selected {len(selected_indices)} hands: {selected_indices}")
        
        # Step 5: Assign left/right and extract landmarks
        hands_by_side = self._assign_hand_sides(
            hand_landmarks, hand_centers, selected_indices, face_center
        )
        
        result['left_hand'] = hands_by_side['left']
        result['right_hand'] = hands_by_side['right']
        
        return result
    
    @staticmethod
    def _compute_face_center(face_landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Compute face center from face landmarks.
        
        Uses the average of nose and eye positions for robustness.
        MediaPipe face landmark indices:
            - Nose: 1
            - Left eye: 33
            - Right eye: 263
        
        Args:
            face_landmarks (np.ndarray): Shape (468, 3), from MediaPipe Face Landmarker.
                                        Coordinates normalized to [0, 1] or pixels.
        
        Returns:
            Tuple[float, float]: (x, y) face center coordinates.
        """
        # Use nose and eyes for robust center estimation
        nose_idx = 1
        left_eye_idx = 33
        right_eye_idx = 263
        
        # Extract (x, y) coordinates only
        nose = face_landmarks[nose_idx, :2]
        left_eye = face_landmarks[left_eye_idx, :2]
        right_eye = face_landmarks[right_eye_idx, :2]
        
        # Average to get face center
        face_center = np.mean([nose, left_eye, right_eye], axis=0)
        
        return tuple(face_center.astype(np.float32))
    
    @staticmethod
    def _compute_hand_centers(
        hand_landmarks: List[np.ndarray],
    ) -> List[Tuple[float, float]]:
        """
        Compute center position for each detected hand.
        
        Center = average of all 21 hand landmarks (palm + fingers).
        
        Args:
            hand_landmarks (List[np.ndarray]): List of hand landmark arrays.
                                              Each shape (21, 3).
        
        Returns:
            List[Tuple[float, float]]: List of (x, y) hand centers.
        """
        hand_centers = []
        for hand_lms in hand_landmarks:
            # Extract (x, y) and compute mean
            hand_center = np.mean(hand_lms[:, :2], axis=0)
            hand_centers.append(tuple(hand_center.astype(np.float32)))
        
        return hand_centers
    
    def _filter_hands_by_distance(
        self,
        hand_centers: List[Tuple[float, float]],
        face_center: Tuple[float, float],
        threshold: float,
    ) -> List[int]:
        """
        Filter hands based on Euclidean distance to face center.
        
        Keep only hands within threshold distance from face.
        
        Args:
            hand_centers (List[Tuple]): List of (x, y) hand centers.
            face_center (Tuple): (x, y) face center.
            threshold (float): Max distance threshold in pixels.
        
        Returns:
            List[int]: Indices of hands within threshold distance.
        """
        filtered = []
        
        for idx, hand_center in enumerate(hand_centers):
            dist = np.linalg.norm(
                np.array(hand_center) - np.array(face_center)
            )
            
            if dist <= threshold:
                filtered.append(idx)
        
        return filtered
    
    def _filter_hands_by_roi(
        self,
        hand_centers: List[Tuple[float, float]],
        face_center: Tuple[float, float],
        frame_shape: Tuple[int, int],
    ) -> List[int]:
        """
        Filter hands using a rectangular Region of Interest (ROI) around face.
        
        ROI is centered at face_center with:
        - Width = roi_width_ratio * frame_width
        - Height = roi_height_ratio * frame_height
        
        Args:
            hand_centers (List[Tuple]): List of (x, y) hand centers.
            face_center (Tuple): (x, y) face center.
            frame_shape (Tuple): (height, width) in pixels.
        
        Returns:
            List[int]: Indices of hands inside ROI.
        """
        height, width = frame_shape
        
        # Compute ROI dimensions
        roi_width = width * self.roi_width_ratio
        roi_height = height * self.roi_height_ratio
        
        # Compute ROI bounds (centered at face_center)
        x_min = face_center[0] - roi_width / 2
        x_max = face_center[0] + roi_width / 2
        y_min = face_center[1] - roi_height / 2
        y_max = face_center[1] + roi_height / 2
        
        filtered = []
        
        for idx, hand_center in enumerate(hand_centers):
            x, y = hand_center
            
            # Check if hand is inside ROI
            if x_min <= x <= x_max and y_min <= y <= y_max:
                filtered.append(idx)
        
        return filtered
    
    @staticmethod
    def _select_hands(
        hand_centers: List[Tuple[float, float]],
        hand_indices: List[int],
        face_center: Tuple[float, float],
        max_hands: int = 2,
    ) -> List[int]:
        """
        Select at most max_hands hands closest to face center.
        
        Args:
            hand_centers (List[Tuple]): List of all (x, y) hand centers.
            hand_indices (List[int]): Indices of hands to consider (pre-filtered).
            face_center (Tuple): (x, y) face center.
            max_hands (int): Max number of hands to select. Default: 2.
        
        Returns:
            List[int]: Indices of selected hands (at most max_hands).
        """
        if len(hand_indices) <= max_hands:
            return hand_indices
        
        # Compute distances for filtered hands
        distances = []
        for idx in hand_indices:
            dist = np.linalg.norm(
                np.array(hand_centers[idx]) - np.array(face_center)
            )
            distances.append((dist, idx))
        
        # Sort by distance and select closest max_hands
        distances.sort(key=lambda x: x[0])
        selected = [idx for _, idx in distances[:max_hands]]
        
        return selected
    
    @staticmethod
    def _assign_hand_sides(
        hand_landmarks: List[np.ndarray],
        hand_centers: List[Tuple[float, float]],
        selected_indices: List[int],
        face_center: Tuple[float, float],
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Assign left/right labels to selected hands based on position relative to face.
        
        Logic:
            If hand_center.x < face_center.x → LEFT hand
            Else → RIGHT hand
        
        Args:
            hand_landmarks (List[np.ndarray]): All detected hand landmarks.
            hand_centers (List[Tuple]): All hand centers.
            selected_indices (List[int]): Indices of selected hands.
            face_center (Tuple): (x, y) face center.
        
        Returns:
            Dict with keys 'left' and 'right', values are np.ndarray or None.
        """
        result = {'left': None, 'right': None}
        
        for idx in selected_indices:
            hand_center = hand_centers[idx]
            hand_lms = hand_landmarks[idx]
            
            # Determine side based on x-coordinate
            if hand_center[0] < face_center[0]:
                # LEFT side
                if result['left'] is None:
                    result['left'] = hand_lms
                # If left already set, keep the leftmost (first one)
            else:
                # RIGHT side
                if result['right'] is None:
                    result['right'] = hand_lms
                # If right already set, keep the rightmost (first one)
        
        return result
    
    @staticmethod
    def horizontally_flip_hand(
        hand_landmarks: np.ndarray,
        frame_width: int,
    ) -> np.ndarray:
        """
        Horizontally flip hand landmarks (for left-handed user support).
        
        Mirrors x-coordinates and swaps symmetrical hand landmark pairs.
        
        MediaPipe hand pairs (0-indexed):
            - (5, 17): Thumb tip ↔ Pinky base
            - (6, 18): Thumb IP ↔ Pinky IP
            - (7, 19): Thumb MCP ↔ Pinky MCP
            - (8, 20): Thumb CMC ↔ Pinky CMC
            - (9, 13): Index tip ↔ Ring tip
            - (10, 14): Index IP ↔ Ring IP
            - (11, 15): Index MCP ↔ Ring MCP
            - (12, 16): Index CMC ↔ Ring CMC
        
        Args:
            hand_landmarks (np.ndarray): Shape (21, 3) hand landmarks.
            frame_width (int): Frame width in pixels (for x-coordinate flip).
        
        Returns:
            np.ndarray: Flipped landmarks, same shape as input.
        """
        flipped = hand_landmarks.copy()
        
        # Flip x-coordinates
        flipped[:, 0] = frame_width - hand_landmarks[:, 0]
        
        # Swap symmetrical landmark pairs
        pairs = [
            (5, 17), (6, 18), (7, 19), (8, 20),
            (9, 13), (10, 14), (11, 15), (12, 16),
        ]
        
        for i, j in pairs:
            flipped[[i, j]] = hand_landmarks[[j, i]]
        
        return flipped


# ============================================================================
# Visualization Utilities (Optional)
# ============================================================================

def draw_roi_box(
    image: np.ndarray,
    face_center: Tuple[float, float],
    frame_shape: Tuple[int, int],
    roi_width_ratio: float = 0.5,
    roi_height_ratio: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw ROI box on image for visualization.
    
    Args:
        image (np.ndarray): Input image.
        face_center (Tuple): (x, y) face center.
        frame_shape (Tuple): (height, width).
        roi_width_ratio (float): ROI width ratio.
        roi_height_ratio (float): ROI height ratio.
        color (Tuple): BGR color for box.
        thickness (int): Line thickness.
    
    Returns:
        np.ndarray: Image with ROI box drawn.
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping ROI visualization")
        return image
    
    height, width = frame_shape
    roi_width = width * roi_width_ratio
    roi_height = height * roi_height_ratio
    
    x_min = int(face_center[0] - roi_width / 2)
    y_min = int(face_center[1] - roi_height / 2)
    x_max = int(face_center[0] + roi_width / 2)
    y_max = int(face_center[1] + roi_height / 2)
    
    # Clip to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    return image


def draw_hand_centers(
    image: np.ndarray,
    face_center: Tuple[float, float],
    hand_centers: List[Tuple[float, float]],
    selected_indices: List[int],
    radius: int = 5,
) -> np.ndarray:
    """
    Draw hand centers and face center on image for visualization.
    
    Args:
        image (np.ndarray): Input image.
        face_center (Tuple): (x, y) face center.
        hand_centers (List[Tuple]): List of (x, y) hand centers.
        selected_indices (List[int]): Indices of selected hands.
        radius (int): Circle radius for visualization.
    
    Returns:
        np.ndarray: Image with centers drawn.
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping hand center visualization")
        return image
    
    # Draw face center (blue)
    cv2.circle(image, tuple(map(int, face_center)), radius, (255, 0, 0), -1)
    
    # Draw all hand centers (gray) and selected (green)
    for idx, hand_center in enumerate(hand_centers):
        color = (0, 255, 0) if idx in selected_indices else (128, 128, 128)
        cv2.circle(image, tuple(map(int, hand_center)), radius, color, -1)
    
    return image


# ============================================================================
# Standalone Functions (for modular use)
# ============================================================================

def compute_face_center(face_landmarks: np.ndarray) -> Tuple[float, float]:
    """Standalone function: compute face center."""
    return HandSelector._compute_face_center(face_landmarks)


def compute_hand_centers(
    hand_landmarks: List[np.ndarray],
) -> List[Tuple[float, float]]:
    """Standalone function: compute hand centers."""
    return HandSelector._compute_hand_centers(hand_landmarks)


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("Hand Selection Module - Example Usage")
    print("=" * 60)
    
    # Create selector
    selector = HandSelector(
        distance_threshold=300.0,
        roi_width_ratio=0.5,
        roi_height_ratio=0.5,
        use_roi_filtering=True,
        enable_debugging=True,
    )
    
    # Simulate face landmarks (468, 3)
    np.random.seed(42)
    face_landmarks = np.random.rand(468, 3) * 100
    face_landmarks[1] = [320, 240, 0]  # Nose at center
    face_landmarks[33] = [300, 220, 0]  # Left eye
    face_landmarks[263] = [340, 220, 0]  # Right eye
    
    # Simulate hand landmarks (multiple hands)
    hand_landmarks = [
        np.random.rand(21, 3) * 50 + 250,  # Hand 1 (left, should be selected)
        np.random.rand(21, 3) * 50 + 390,  # Hand 2 (right, should be selected)
        np.random.rand(21, 3) * 50 + 50,   # Hand 3 (far, should be filtered out)
    ]
    
    frame_shape = (480, 640)  # (height, width)
    
    print(f"\nFrame shape: {frame_shape}")
    print(f"Number of face landmarks: {face_landmarks.shape}")
    print(f"Number of hands detected: {len(hand_landmarks)}")
    
    # Process hands
    result = selector.process_hands(
        face_landmarks=face_landmarks,
        hand_landmarks=hand_landmarks,
        frame_shape=frame_shape,
    )
    
    print(f"\n--- Processing Results ---")
    print(f"Face center: {result['face_center']}")
    print(f"Selected hand indices: {result['selected_hand_indices']}")
    print(f"Left hand selected: {result['left_hand'] is not None}")
    print(f"Right hand selected: {result['right_hand'] is not None}")
    
    # Test edge cases
    print(f"\n--- Edge Case Tests ---")
    
    # No face
    result_no_face = selector.process_hands(None, hand_landmarks, frame_shape)
    print(f"No face: {result_no_face['left_hand'] is None and result_no_face['right_hand'] is None}")
    
    # No hands
    result_no_hands = selector.process_hands(face_landmarks, None, frame_shape)
    print(f"No hands: {result_no_hands['left_hand'] is None and result_no_hands['right_hand'] is None}")
    
    # Empty hands list
    result_empty_hands = selector.process_hands(face_landmarks, [], frame_shape)
    print(f"Empty hands: {result_empty_hands['left_hand'] is None and result_empty_hands['right_hand'] is None}")
    
    print("\n✓ All tests passed!")
