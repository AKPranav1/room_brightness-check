"""
main.py

Standalone Pillar 1 (face quality) tester — no Pillar 2 / ID-PAD dependency,
so this folder runs on its own without your main project's yolo/onnx models.
If you later merge this into your full project, swap back to the version of
main.py that also imports pillar2_id_pad.pipeline.
"""

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from schemas import FaceQualityRequest, FaceQualityResponse
from utils.decoding import decode_base64_image
from utils.logging_calibration import log_calibration_row
from pillar1_face_quality.pipeline import run_face_quality_pipeline, run_face_quality_pipeline_burst

app = FastAPI(title="Pillar 1 — Face Quality Reliability Tester")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.post("/v1/face-quality", response_model=FaceQualityResponse)
async def face_quality_endpoint(payload: FaceQualityRequest):
    request_id = str(uuid.uuid4())

    if payload.face_frames:
        frames = [
            decode_base64_image(f, field_name=f"face_frames[{i}]")
            for i, f in enumerate(payload.face_frames)
        ]
        result = run_face_quality_pipeline_burst(frames)

        if result.get("per_frame_diagnostics"):
            result["brightness_diagnostics"] = result["per_frame_diagnostics"][0]

    else:
        frame = decode_base64_image(payload.face_frame, field_name="face_frame")
        result = run_face_quality_pipeline(frame)

    log_calibration_row(request_id, "face_quality", {
        "visibility_score": result["visibility_score"],
        "brightness_score": result["brightness_score"],
        "clarity_score": result["clarity_score"],
        "composite_score": result["composite_score"],
    })

    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
