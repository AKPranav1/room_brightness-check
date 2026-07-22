"""
pillar1_face_quality/clarity.py
"""

import numpy as np
import cv2
from config import CLARITY_MIN_REFERENCE, CLARITY_MAX_REFERENCE


def compute_clarity_score(face_crop_bgr: np.ndarray) -> float:
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    span = CLARITY_MAX_REFERENCE - CLARITY_MIN_REFERENCE
    normalized = (laplacian_var - CLARITY_MIN_REFERENCE) / span
    return round(max(0.0, min(normalized, 1.0)), 4)