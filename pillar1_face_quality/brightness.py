"""
pillar1_face_quality/brightness.py
"""

import math
import numpy as np
import cv2
from config import (
    SKIN_PATCH_LANDMARK_INDICES, SKIN_PATCH_HALF_SIZE_RATIO,
    SKIN_PATCH_HALF_SIZE_MIN_PX, SKIN_PATCH_HALF_SIZE_MAX_PX,
    SKIN_PATCH_MIN_VALID_PATCHES,
    SHADOW_CLIP_THRESHOLD, HIGHLIGHT_CLIP_THRESHOLD,
    MAX_UNDEREXPOSED_FRACTION, MAX_OVEREXPOSED_FRACTION,
    SKIN_MEDIAN_TARGET, SKIN_MEDIAN_ACCEPTABLE_RANGE
)

def _patch_half_size(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    diag = float(np.hypot(x2 - x1, y2 - y1))
    half = int(round(diag * SKIN_PATCH_HALF_SIZE_RATIO))
    return max(SKIN_PATCH_HALF_SIZE_MIN_PX, min(half, SKIN_PATCH_HALF_SIZE_MAX_PX))

def _sample_skin_patches(gray_full: np.ndarray, landmarks_px: list[tuple[float, float]],
                          bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    if not landmarks_px:
        return None

    h, w = gray_full.shape[:2]
    half = _patch_half_size(bbox)
    pixels = []
    valid_patches = 0

    for _name, idx in SKIN_PATCH_LANDMARK_INDICES.items():
        if idx >= len(landmarks_px):
            continue
        cx, cy = landmarks_px[idx]
        cx, cy = int(round(cx)), int(round(cy))
        x1, x2 = max(0, cx - half), min(w, cx + half)
        y1, y2 = max(0, cy - half), min(h, cy + half)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = gray_full[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        pixels.append(patch.reshape(-1))
        valid_patches += 1

    if valid_patches < SKIN_PATCH_MIN_VALID_PATCHES:
        return None

    return np.concatenate(pixels)

def compute_brightness_score(
    face_crop_bgr: np.ndarray,
    frame_bgr: np.ndarray | None = None,
    landmarks_px: list[tuple[float, float]] | None = None,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[float, dict]:
    
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return 0.0, {"mode": "no_face", "median_luminance": 0.0,
                      "underexposed_fraction": 1.0, "overexposed_fraction": 0.0,
                      "is_too_dark": True, "is_too_bright": False}

    skin_pixels = None
    if frame_bgr is not None and landmarks_px and face_bbox is not None:
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        skin_pixels = _sample_skin_patches(gray_full, landmarks_px, face_bbox)

    if skin_pixels is not None:
        mode = "skin_patch"
    else:
        gray_crop = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        flat = np.sort(gray_crop.reshape(-1))
        n = len(flat)
        trim = int(n * 0.15)
        skin_pixels = flat[trim: n - trim] if n - 2 * trim > 0 else flat
        mode = "degraded_whole_crop"

    median_lum = float(np.median(skin_pixels))
    underexposed_fraction = float(np.mean(skin_pixels < SHADOW_CLIP_THRESHOLD))
    overexposed_fraction = float(np.mean(skin_pixels > HIGHLIGHT_CLIP_THRESHOLD))

    is_too_dark = underexposed_fraction > MAX_UNDEREXPOSED_FRACTION
    is_too_bright = overexposed_fraction > MAX_OVEREXPOSED_FRACTION

    # 1. Calculate the true base score using the clean logarithmic curve
    if median_lum >= SKIN_MEDIAN_TARGET:
        base_score = 1.0
    else:
        lower_bound = max(1.0, SKIN_MEDIAN_TARGET - SKIN_MEDIAN_ACCEPTABLE_RANGE)
        if median_lum <= lower_bound:
            base_score = 0.0
        else:
            progress = (median_lum - lower_bound) / (SKIN_MEDIAN_TARGET - lower_bound)
            base_score = math.log10(progress * 9.0 + 1.0)

    # 2. Strict Architectural Gates (The Fix)
    # If the image trips the shadow or glare thresholds, fail it instantly. No inverted math.
    if is_too_dark or is_too_bright:
        score = 0.0
    else:
        score = base_score

    diagnostics = {
        "mode": mode,
        "median_luminance": round(median_lum, 2),
        "underexposed_fraction": round(underexposed_fraction, 4),
        "overexposed_fraction": round(overexposed_fraction, 4),
        "is_too_dark": is_too_dark,
        "is_too_bright": is_too_bright,
    }
    return round(max(0.0, min(score, 1.0)), 4), diagnostics