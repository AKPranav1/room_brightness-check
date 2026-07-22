"""
schemas.py
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal


# ---- Requests ----

class FaceQualityRequest(BaseModel):
    face_frame: Optional[str] = Field(
        None, description="Base64-encoded JPEG/PNG of the user's face (single-frame mode)"
    )
    face_frames: Optional[list[str]] = Field(
        None, description="List of base64 frames captured ~100-150ms apart (burst mode, "
                           "recommended — protects against single-frame auto-exposure flicker)"
    )

    @model_validator(mode="after")
    def _require_one_of(self):
        if not self.face_frame and not self.face_frames:
            raise ValueError("Provide either 'face_frame' or 'face_frames'.")
        return self


class IdPadRequest(BaseModel):
    id_flash_off: str = Field(..., description="Base64 frame, ambient light, no flash")
    id_flash_on: str = Field(..., description="Base64 frame, captured during UI flash pulse")


# ---- Responses ----

class FaceQualityResponse(BaseModel):
    visibility_score: float
    brightness_score: float
    clarity_score: float
    composite_score: float
    hints: list[str] = []
    stage_timings_ms: dict[str, float]
    brightness_diagnostics: Optional[dict] = None
    per_frame_diagnostics: Optional[list[dict]] = None
    frames_analyzed: Optional[int] = None


class IdPadResponse(BaseModel):
    roi_pixel: Optional[list[int]] = None
    roi_normalized: Optional[list[float]] = None
    yolo_class: Optional[int] = None
    yolo_confidence: Optional[float] = None
    band: Literal["physical_id_card", "screen_detected", "ambiguous-needs_human_verification", "undetermined"]
    action: Literal["pass", "fail", "retry-once", "flag-for-review"]
    reason: Optional[str] = None
    stage_timings_ms: dict[str, float]


class ErrorResponse(BaseModel):
    detail: str
