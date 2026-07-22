"""
utils/decoding.py

Every malformed-input path here must raise HTTPException(400, ...) — never let a bad
base64 string or corrupt image bubble up as an unhandled 500.
"""

import base64
import binascii
import numpy as np
import cv2
from fastapi import HTTPException


def decode_base64_image(b64_string: str, field_name: str = "image") -> np.ndarray:
    """
    Decode a base64 string into a BGR cv2 image (np.ndarray, HxWx3, uint8).
    Raises HTTPException(400) on any failure — malformed base64, empty payload,
    or bytes that don't decode into a valid image.
    """
    if not b64_string or not isinstance(b64_string, str):
        raise HTTPException(status_code=400, detail=f"{field_name}: empty or invalid payload")

    # Strip data URI prefix if present, e.g. "data:image/jpeg;base64,...."
    if "," in b64_string and b64_string.strip().startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(b64_string, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name}: not valid base64 ({e})"
        )

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail=f"{field_name}: decoded to zero bytes")

    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name}: bytes decoded but could not be interpreted as an image"
        )

    return image


def assert_same_resolution(img_a: np.ndarray, img_b: np.ndarray,
                            name_a: str, name_b: str) -> None:
    """
    Pillar 2 requires flash_off and flash_on to be the same resolution — if they aren't,
    error clearly rather than silently misaligning crops downstream.
    """
    if img_a.shape[:2] != img_b.shape[:2]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Resolution mismatch: {name_a} is {img_a.shape[1]}x{img_a.shape[0]}, "
                f"{name_b} is {img_b.shape[1]}x{img_b.shape[0]}. Both frames must match."
            )
        )
