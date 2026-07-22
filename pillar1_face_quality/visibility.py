"""
pillar1_face_quality/visibility.py
"""

from config import FACE_TARGET_BBOX_AREA_RATIO, FACE_LANDMARKS_EXPECTED
from pillar1_face_quality.face_detect import FaceDetectionResult


def compute_visibility_score(det: FaceDetectionResult) -> float:
    if not det.found:
        return 0.0

    h, w = det.frame_shape
    frame_area = h * w
    x1, y1, x2, y2 = det.face_bbox
    bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox_area_ratio = bbox_area / frame_area if frame_area > 0 else 0.0

    landmark_ratio = det.landmarks_found / FACE_LANDMARKS_EXPECTED
    landmark_ratio = min(landmark_ratio, 1.0)

    area_term = bbox_area_ratio / FACE_TARGET_BBOX_AREA_RATIO
    area_term = max(0.0, min(area_term, 1.0))

    visibility = det.detection_confidence * landmark_ratio * area_term
    return round(max(0.0, min(visibility, 1.0)), 4)
