# Pillar 1 — Face Quality Reliability Tester

A standalone FastAPI service that scores webcam face captures for **visibility**, **brightness**, and **clarity**, and returns actionable hints ("too dark", "too blurry", "move closer") so a client-side capture UI can guide the user to a usable photo before it's submitted downstream (e.g. into an ID-verification flow).

This folder is **Pillar 1** of a larger project. Pillar 2 (ID card / screen-spoof detection) is referenced in `config.py` and `schemas.py` but its pipeline is not wired into `main.py` in this build — `main.py` currently only imports the Pillar 1 pipeline, so it runs standalone with no YOLO/ONNX dependency.

---

## 1. What it does

A user's browser captures either a single webcam frame or a short **burst** of frames (5 frames, ~110ms apart) and POSTs them as base64 JPEGs to `/v1/face-quality`. The server:

1. Detects the face and 478 facial landmarks (MediaPipe FaceLandmarker).
2. Scores three independent quality dimensions (0.0–1.0 each):
   - **Visibility** — is a face present, centered, and large enough in frame?
   - **Brightness** — is the skin, specifically, well-lit (not underexposed/overexposed)?
   - **Clarity** — is the face crop in focus (Laplacian variance)?
3. Combines them into a weighted **composite score**.
4. Emits human-readable **hints** ("Warning: Image is too dark. Please turn on a light.") when a score falls below threshold.
5. In burst mode, aggregates all 5 frames by **median score** and **majority-vote hints**, so one unlucky auto-exposure frame can't fail the whole check.
6. Logs every raw score to a CSV for later threshold calibration.

The included `static/index.html` is a self-contained browser demo: it opens the webcam, lets you fire a capture, and renders the returned scores as live progress bars plus raw JSON.

---

## 2. Architecture

```mermaid
flowchart TD
    subgraph Client["Browser (static/index.html)"]
        A[getUserMedia webcam stream] --> B[captureFrame / captureBurst]
        B --> C[POST base64 JPEG frame(s)]
    end

    subgraph Server["FastAPI (main.py)"]
        C --> D["/v1/face-quality endpoint"]
        D --> E[decode_base64_image\nutils/decoding.py]
        E --> F{single frame or\nburst?}
        F -->|single| G[run_face_quality_pipeline]
        F -->|burst, N frames| H[run_face_quality_pipeline_burst]
        H --> G
    end

    subgraph Pipeline["pillar1_face_quality/pipeline.py"]
        G --> I[detect_face\nface_detect.py\nMediaPipe FaceLandmarker]
        I --> J[compute_visibility_score\nvisibility.py]
        I --> K[compute_brightness_score\nbrightness.py\nskin-patch sampling]
        I --> L[compute_clarity_score\nclarity.py\nLaplacian variance]
        J --> M[weighted composite +\nhint generation]
        K --> M
        L --> M
    end

    M --> N[log_calibration_row\nutils/logging_calibration.py\n-> logs/calibration_scores.csv]
    M --> O[FaceQualityResponse JSON]
    O --> C

    P[config.py\nall thresholds & weights] -.-> J
    P -.-> K
    P -.-> L
    P -.-> M
```

### Module map

| Path | Responsibility |
|---|---|
| `main.py` | FastAPI app, `/v1/face-quality` and `/health` routes, serves the static demo UI |
| `config.py` | Every threshold/weight/magic number in the project — the single source of truth |
| `schemas.py` | Pydantic request/response models (`FaceQualityRequest`, `FaceQualityResponse`, plus unused Pillar 2 schemas) |
| `pillar1_face_quality/face_detect.py` | Wraps MediaPipe's `FaceLandmarker` (Tasks API), returns bbox + 478 landmarks |
| `pillar1_face_quality/visibility.py` | Scores how centered/large/well-tracked the face is |
| `pillar1_face_quality/brightness.py` | Skin-patch luminance sampling + exposure gating |
| `pillar1_face_quality/clarity.py` | Blur detection via Laplacian variance |
| `pillar1_face_quality/pipeline.py` | Orchestrates the above per-frame and across a burst |
| `utils/decoding.py` | Safe base64 → OpenCV image decoding (never raises unhandled 500s) |
| `utils/logging_calibration.py` | Appends every raw score to a CSV for threshold tuning |
| `utils/timing.py` | Lightweight per-stage timer, surfaced in the API response |
| `static/index.html` | Browser demo: webcam capture, burst toggle, live scorebars, raw JSON viewer |
| `static/face-quality-burst-snippet.js` | Reference snippet for wiring burst capture into a custom UI |
| `models/face_landmarker.task` | MediaPipe model weights (**not included** — must be downloaded, see below) |

---

## 3. How the scoring actually works

### Visibility (`visibility.py`)
```
visibility = detection_confidence × landmark_ratio × area_term
```
- `landmark_ratio` = landmarks found ÷ 478 expected.
- `area_term` = face bounding-box area ÷ target area ratio (`FACE_TARGET_BBOX_AREA_RATIO = 0.025`), clamped to [0, 1]. A face that's too small in frame gets penalized.

### Brightness (`brightness.py`)
Rather than measuring brightness over the whole face crop (which is skewed by hair, background, glasses glare, etc.), it samples small patches at four **skin landmark points**: forehead, left cheek, right cheek, chin. Patch size scales with face size (bounded 4–45px). If fewer than 2 patches are valid, it falls back to a trimmed-mean of the whole face crop ("degraded" mode — surfaced to the client via a badge).

From the sampled pixels:
- `median_luminance` is compared against a target (115) with an acceptable range (±85) using a **logarithmic** falloff curve below target.
- Two **hard gates** run in parallel: if the fraction of pixels below 30 (shadow clip) exceeds 25%, or above 245 (highlight clip) exceeds 35%, the score is forced to **0.0** regardless of the median — a clearly under/over-exposed face fails outright rather than getting averaged into a passable score.

### Clarity (`clarity.py`)
Variance of the Laplacian of the grayscale face crop, linearly normalized between a reference blurry value (40) and a reference sharp value (200).

### Composite
```
composite = visibility × 1/3 + brightness × 1/3 + clarity × 1/3
```
Equal weighting by default (`config.py`), each below the 0.5 threshold triggers its own hint.

### Burst mode (`pipeline.py: run_face_quality_pipeline_burst`)
Each of the N captured frames is scored independently, then:
- Numeric scores (visibility/brightness/clarity/composite) → **median** across frames where a face was found.
- Hints (too dark / too bright / too small / blurry / no face) → **majority vote**: a hint only surfaces if it appears in at least `max(3 frames, 60% of frames)`.

This exists specifically to absorb webcam auto-exposure/auto-white-balance "hunting" in the first ~100–200ms after a stream starts, which can otherwise produce one falsely dark or blurry frame.

---

## 4. API

### `POST /v1/face-quality`
Request (single frame):
```json
{ "face_frame": "<base64 JPEG/PNG>" }
```
Request (burst, recommended):
```json
{ "face_frames": ["<base64>", "<base64>", "..."] }
```

Response:
```json
{
  "visibility_score": 0.91,
  "brightness_score": 0.74,
  "clarity_score": 0.88,
  "composite_score": 0.84,
  "hints": [],
  "stage_timings_ms": { "face_detection": 12.3, "...": "...", "total": 25.1 },
  "brightness_diagnostics": {
    "mode": "skin_patch",
    "median_luminance": 121.4,
    "underexposed_fraction": 0.01,
    "overexposed_fraction": 0.0,
    "is_too_dark": false,
    "is_too_bright": false
  },
  "frames_analyzed": 5,
  "per_frame_diagnostics": [ "...one diagnostics object per frame (burst mode only)..." ]
}
```

### `GET /health`
Returns `{"status": "ok"}` — for uptime checks / load balancer probes.

### `GET /`
Serves the browser demo at `static/index.html`.

All malformed/invalid input (bad base64, corrupt image bytes, empty payload) returns a clean `400` with a descriptive `detail` message — it never bubbles up as an unhandled `500`.

---

## 5. Setup & running it

### Requirements
- Python 3.10+ (uses `X | Y` union type hints and `dict[str, float]` generics)
- A webcam-capable browser to use the demo UI

### Install
```bash
pip install -r requirements.txt
```
Installs: `fastapi`, `uvicorn[standard]`, `pydantic`, `opencv-python-headless`, `numpy`, `mediapipe`.

### Download the required model
The FaceLandmarker model is **not checked into the repo** and must be downloaded before first run:
```bash
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
PowerShell:
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models/face_landmarker.task"
```
Expected size is a few MB. If the download comes back as a tiny HTML/error file, your network is blocking `storage.googleapis.com` — fetch it from another network and copy the file in. The app checks for this file's existence at import time and fails fast with a clear error if it's missing (no silent fallback).

### Run the server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Try it
Open **http://localhost:8000/** in a browser, allow camera access, and click **Run Face Quality Check** (burst mode is on by default). Scores render as live bars; the raw JSON response is shown below them.

Or hit the API directly:
```bash
curl -X POST http://localhost:8000/v1/face-quality \
  -H "Content-Type: application/json" \
  -d '{"face_frame": "<base64-jpeg-here>"}'
```

---

## 6. Calibration logging

Every request's raw scores are appended to `logs/calibration_scores.csv` (path/mode configurable via `CALIBRATION_LOG_PATH` / `CALIBRATION_LOG_MODE` in `config.py` — `"csv"`, `"stdout"`, or `"both"`). This is meant to be run against real reference samples (known-good / known-bad captures) so the thresholds in `config.py` can be re-tuned empirically rather than guessed — none of the shipped constants are meant to be final production values.

---

## 7. Key design notes worth knowing before you modify this

- **Every tunable number lives in `config.py`.** Nothing downstream should hardcode a threshold or weight — if you're tempted to inline a magic number in a pillar module, it belongs in config instead.
- **Brightness gating is intentionally strict and non-linear.** Median luminance alone isn't trusted — the shadow/highlight clipping fractions are hard gates that zero out the score, so a face that's *mostly* fine but has one blown-out patch (e.g. glare on glasses) won't slip through on the median alone.
- **MediaPipe Tasks API only.** The legacy `mp.solutions.face_mesh` API was removed in mediapipe 0.10.30+; `face_detect.py` deliberately uses `mediapipe.tasks` instead. Don't revert to `mp.solutions`.
- **The FaceLandmarker instance is a single global with a threading lock**, since the underlying C++ graph isn't safe for concurrent calls — FastAPI can run sync route handlers across a threadpool, so this lock is load-bearing, not defensive boilerplate.
- **`main.py` here is a Pillar-1-only build.** If merging back into the full project, swap in the version of `main.py` that also imports `pillar2_id_pad.pipeline`; `schemas.py` and `config.py` already contain the Pillar 2 (ID-PAD) definitions in anticipation of that merge.