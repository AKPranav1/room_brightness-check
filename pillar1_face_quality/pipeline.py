"""
pillar1_face_quality/pipeline.py

Orchestrates Pillar 1 end-to-end.

`run_face_quality_pipeline` scores a single frame.

`run_face_quality_pipeline_burst` scores a short burst of frames (captured
~100-150ms apart client-side) and only surfaces a brightness hint if a
majority of frames in the burst agree. This is the main defense against a
webcam's auto-exposure/auto-white-balance "hunting" for a moment right after
capture starts and producing one unlucky dark-looking frame — it should not
be able to fail a user on its own.
"""

import numpy as np
from config import (
    FACE_QUALITY_WEIGHT_VISIBILITY, FACE_QUALITY_WEIGHT_BRIGHTNESS,
    FACE_QUALITY_WEIGHT_CLARITY, FACE_QUALITY_HINT_THRESHOLD,
    BRIGHTNESS_BURST_MIN_AGREEMENT_FRACTION, BRIGHTNESS_BURST_MIN_FRAMES,
)
from utils.timing import StageTimer
from pillar1_face_quality.face_detect import detect_face
from pillar1_face_quality.visibility import compute_visibility_score
from pillar1_face_quality.brightness import compute_brightness_score
from pillar1_face_quality.clarity import compute_clarity_score


def _crop_face(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def run_face_quality_pipeline(frame_bgr: np.ndarray) -> dict:
    timer = StageTimer()

    with timer.stage("face_detection"):
        det = detect_face(frame_bgr)

    with timer.stage("visibility_score"):
        visibility = compute_visibility_score(det)

    face_crop = _crop_face(frame_bgr, det.face_bbox) if det.found else None

    with timer.stage("brightness_score"):
        if face_crop is not None:
            brightness, brightness_diag = compute_brightness_score(
                face_crop,
                frame_bgr=frame_bgr,
                landmarks_px=det.landmarks_px,
                face_bbox=det.face_bbox,
            )
        else:
            brightness, brightness_diag = 0.0, {
                "mode": "no_face", "median_luminance": 0.0,
                "underexposed_fraction": 1.0, "overexposed_fraction": 0.0,
                "is_too_dark": True, "is_too_bright": False,
            }

    with timer.stage("clarity_score"):
        clarity = compute_clarity_score(face_crop) if face_crop is not None else 0.0

    composite = (
        visibility * FACE_QUALITY_WEIGHT_VISIBILITY
        + brightness * FACE_QUALITY_WEIGHT_BRIGHTNESS
        + clarity * FACE_QUALITY_WEIGHT_CLARITY
    )
    composite = round(composite, 4)

    hints = []
    if not det.found:
        hints.append("No face detected — please center your face in the frame.")
    else:
        if visibility < FACE_QUALITY_HINT_THRESHOLD:
            hints.append("Face is too small or partially out of frame — move closer / center yourself.")

        if brightness_diag["is_too_dark"]:
            hints.append("Warning: Image is too dark. Please turn on a light.")
        elif brightness_diag["is_too_bright"]:
            hints.append("Warning: Severe glare. Please face away from direct light.")

        if clarity < FACE_QUALITY_HINT_THRESHOLD:
            hints.append("Image is blurry. Please hold still or clean your camera lens.")

    return {
        "visibility_score": visibility,
        "brightness_score": brightness,
        "brightness_diagnostics": brightness_diag,
        "clarity_score": clarity,
        "composite_score": composite,
        "hints": hints,
        "stage_timings_ms": timer.as_dict(),
    }


def run_face_quality_pipeline_burst(frames_bgr: list[np.ndarray]) -> dict:
    """
    Scores a burst of frames and aggregates:
      - visibility/brightness/clarity/composite scores -> median across frames
        that had a face detected (single unlucky frame can't tank the score)
      - brightness "too dark"/"too bright" hints only fire if they agree
        across >= BRIGHTNESS_BURST_MIN_AGREEMENT_FRACTION of frames
      - other hints (no face / too small / blurry) use majority vote too
    """
    per_frame = [run_face_quality_pipeline(f) for f in frames_bgr]
    n = len(per_frame)

    faces_found = [r for r in per_frame if "No face detected — please center your face in the frame." not in r["hints"]]
    use = faces_found if faces_found else per_frame

    def _median(key):
        vals = [r[key] for r in use]
        return round(float(np.median(vals)), 4) if vals else 0.0

    visibility = _median("visibility_score")
    brightness = _median("brightness_score")
    clarity = _median("clarity_score")
    composite = _median("composite_score")

    too_dark_votes = sum(1 for r in per_frame if r["brightness_diagnostics"]["is_too_dark"])
    too_bright_votes = sum(1 for r in per_frame if r["brightness_diagnostics"]["is_too_bright"])
    no_face_votes = sum(1 for r in per_frame if "No face detected — please center your face in the frame." in r["hints"])
    small_votes = sum(1 for r in per_frame if any("too small" in h for h in r["hints"]))
    blur_votes = sum(1 for r in per_frame if any("blurry" in h for h in r["hints"]))

    agreement_needed = max(
        BRIGHTNESS_BURST_MIN_FRAMES if n >= BRIGHTNESS_BURST_MIN_FRAMES else n,
        int(np.ceil(n * BRIGHTNESS_BURST_MIN_AGREEMENT_FRACTION)),
    )

    hints = []
    if no_face_votes >= agreement_needed:
        hints.append("No face detected — please center your face in the frame.")
    else:
        if small_votes >= agreement_needed:
            hints.append("Face is too small or partially out of frame — move closer / center yourself.")
        if too_dark_votes >= agreement_needed:
            hints.append("Warning: Image is too dark. Please turn on a light.")
        elif too_bright_votes >= agreement_needed:
            hints.append("Warning: Severe glare. Please face away from direct light.")
        if blur_votes >= agreement_needed:
            hints.append("Image is blurry. Please hold still or clean your camera lens.")

    return {
        "visibility_score": visibility,
        "brightness_score": brightness,
        "clarity_score": clarity,
        "composite_score": composite,
        "hints": hints,
        "frames_analyzed": n,
        "per_frame_diagnostics": [r["brightness_diagnostics"] for r in per_frame],
        "stage_timings_ms": {"total_ms": round(sum(r["stage_timings_ms"]["total"] for r in per_frame), 2)},
    }
