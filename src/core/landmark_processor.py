import numpy as np

def landmarks_to_numpy(landmarks) -> np.ndarray:
    """Convert MediaPipe landmarks to numpy array (N, 2 or 3 dimensions)."""
    if landmarks is None:
        return None
    return np.array([[lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0.0] for lm in landmarks], dtype=np.float32)


def bbox_iou(a, b):
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def landmarks_to_bbox(hand_landmarks, w, h, pad=14):
    """Convert normalized hand landmarks to a padded pixel bbox."""
    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w - 1, max(xs) + pad)
    y2 = min(h - 1, max(ys) + pad)
    return x1, y1, x2, y2


def bbox_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_contains_point(box, pt):
    x1, y1, x2, y2 = box
    px, py = pt
    return x1 <= px <= x2 and y1 <= py <= y2


def wrist_point_px(hand_landmarks, w, h):
    """Return wrist landmark (id=0) in pixels."""
    wrist = hand_landmarks[0]
    return int(wrist.x * w), int(wrist.y * h)


def detect_person_boxes(frame, hog_detector):
    """Detect person boxes and apply a lightweight NMS by IoU."""
    rects, weights = hog_detector.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    candidates = []
    for (x, y, w, h), conf in zip(rects, weights):
        if conf < 0.3:
            continue
        candidates.append((x, y, x + w, y + h, float(conf)))

    candidates.sort(key=lambda t: t[4], reverse=True)
    kept = []
    for cand in candidates:
        cbox = cand[:4]
        if any(bbox_iou(cbox, k[:4]) > 0.45 for k in kept):
            continue
        kept.append(cand)

    return kept


def assign_hand_to_person(hand_box, person_boxes):
    """Assign hand box to a person by containment-first then nearest center."""
    if len(person_boxes) == 1:
        # If only one person is visible, assign all hands to that person.
        return 0

    hand_center = bbox_center(hand_box)

    containing = []
    for idx, p in enumerate(person_boxes):
        pbox = p[:4]
        if box_contains_point(pbox, hand_center):
            containing.append((idx, pbox))

    candidates = containing if containing else [
        (idx, p[:4]) for idx, p in enumerate(person_boxes)
    ]
    if not candidates:
        return None

    hx, hy = hand_center
    best_idx = None
    best_dist = float("inf")
    for idx, pbox in candidates:
        px, py = bbox_center(pbox)
        dist = (hx - px) ** 2 + (hy - py) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    return best_idx
