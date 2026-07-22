"""
pillar1_face_quality/face_detect.py

Uses MediaPipe's new Tasks API (FaceLandmarker) instead of the legacy
`mp.solutions.face_mesh`. Google removed `mp.solutions` entirely in recent
mediapipe releases (0.10.30+) — any code still importing it will crash with
`AttributeError: module 'mediapipe' has no attribute 'solutions'` the moment
someone installs a fresh mediapipe. The Tasks API is the actively maintained
path forward, so this is the reliable long-term choice, not a workaround.

Requires a model file (~a few MB) on disk. Download once:

    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Save it to the path in config.FACE_LANDMARKER_MODEL_PATH (default:
`models/face_landmarker.task`). This is checked for existence at import time
with a clear error message — it will NOT fail silently or fall back to
guessing.
"""

import os
import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass, field
from config import FACE_LANDMARKS_EXPECTED, FACE_LANDMARKER_MODEL_PATH


def _build_landmarker() -> mp_vision.FaceLandmarker:
    if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
        raise FileNotFoundError(
            f"Face landmarker model not found at '{FACE_LANDMARKER_MODEL_PATH}'.\n"
            "Download it with:\n"
            "  curl -L -o " + FACE_LANDMARKER_MODEL_PATH + " "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        )
    base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


_landmarker = _build_landmarker()
_landmarker_lock = threading.Lock()
# FaceLandmarker's underlying C++ graph is not safe to call concurrently from
# multiple threads — FastAPI can run sync endpoint code across a threadpool,
# so this lock is what actually makes concurrent requests safe rather than
# "usually fine until it isn't."


@dataclass
class FaceDetectionResult:
    found: bool
    detection_confidence: float
    landmarks_found: int
    face_bbox: tuple[int, int, int, int]        # x1, y1, x2, y2 in pixel coords
    frame_shape: tuple[int, int]                 # H, W
    landmarks_px: list[tuple[float, float]] = field(default_factory=list)
    # ^ all landmark (x, y) in pixel coordinates, 478-point topology
    #   (face mesh + iris). Empty when found=False.


def detect_face(frame_bgr: np.ndarray) -> FaceDetectionResult:
    h, w = frame_bgr.shape[:2]
    rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    with _landmarker_lock:
        result = _landmarker.detect(mp_image)

    if not result.face_landmarks:
        return FaceDetectionResult(
            found=False, detection_confidence=0.0, landmarks_found=0,
            face_bbox=(0, 0, 0, 0), frame_shape=(h, w), landmarks_px=[]
        )

    landmarks = result.face_landmarks[0]
    landmarks_px = [(lm.x * w, lm.y * h) for lm in landmarks]
    xs = [p[0] for p in landmarks_px]
    ys = [p[1] for p in landmarks_px]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))

    return FaceDetectionResult(
        found=True,
        detection_confidence=0.9,
        landmarks_found=len(landmarks),
        face_bbox=(x1, y1, x2, y2),
        frame_shape=(h, w),
        landmarks_px=landmarks_px,
    )
