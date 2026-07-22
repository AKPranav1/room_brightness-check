"""
config.py

Every threshold, weight, and magic number in this project lives here and ONLY here.
Nothing downstream should hardcode a numeric literal for scoring/thresholding logic.

All values below are reasonable *starting points* for a first deployment and are meant
to be re-tuned during the calibration pass described in Section 8 of the guide.
"""

# ---------------------------------------------------------------------------
# PILLAR 1 - FACE QUALITY CONSTANTS
# ---------------------------------------------------------------------------

# Visibility
FACE_TARGET_BBOX_AREA_RATIO = 0.025
FACE_LANDMARKS_EXPECTED = 478

FACE_LANDMARKER_MODEL_PATH = "models/face_landmarker.task"

# --- Robust Skin-Sampled Brightness Gate ---
SKIN_PATCH_LANDMARK_INDICES = {
    "forehead": 151,
    "left_cheek": 50,
    "right_cheek": 280,
    "chin": 152,
}
SKIN_PATCH_HALF_SIZE_RATIO = 0.08
SKIN_PATCH_HALF_SIZE_MIN_PX = 4
SKIN_PATCH_HALF_SIZE_MAX_PX = 45
SKIN_PATCH_MIN_VALID_PATCHES = 2

SHADOW_CLIP_THRESHOLD = 20
HIGHLIGHT_CLIP_THRESHOLD = 245
MAX_UNDEREXPOSED_FRACTION = 0.45
MAX_OVEREXPOSED_FRACTION = 0.35

SKIN_MEDIAN_TARGET = 115.0
SKIN_MEDIAN_ACCEPTABLE_RANGE = 85.0

# Burst mode
BRIGHTNESS_BURST_MIN_AGREEMENT_FRACTION = 0.60
BRIGHTNESS_BURST_MIN_FRAMES = 3

# Clarity (Laplacian variance)
CLARITY_MIN_REFERENCE = 40.0
CLARITY_MAX_REFERENCE = 200.0

# Composite weights (must sum to 1.0)
FACE_QUALITY_WEIGHT_VISIBILITY = 1 / 3
FACE_QUALITY_WEIGHT_BRIGHTNESS = 1 / 3
FACE_QUALITY_WEIGHT_CLARITY = 1 / 3

# Hint thresholds
FACE_QUALITY_HINT_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# PILLAR 2 - ID PAD CONSTANTS
# ---------------------------------------------------------------------------

YOLO_MODEL_PATH = "models/yolov8n_id_card.onnx"
YOLO_CONFIDENCE_MIN = 0.35
YOLO_CONFIDENCE_GAP_FOR_SINGLE_FRAME = 0.25

YOLO_2CLASS_MODEL_PATH = "models/id_pad_2class.onnx"
YOLO_2CLASS_CONF_HIGH = 0.60
YOLO_2CLASS_CONF_LOW  = 0.25
YOLO_2CLASS_CONF_GAP  = 0.25

FLASH_DELTA_MAGNITUDE_SCREEN_SUPPRESSION = True

MOIRE_STRIP_WIDTH_PX = 12
MOIRE_FFT_PEAK_FREQ_MIN = 0.15
MOIRE_FFT_PEAK_FREQ_MAX = 0.45

FUSION_WEIGHT_DELTA_MAGNITUDE = 0.35
FUSION_WEIGHT_DELTA_CONCENTRATION = 0.40
FUSION_WEIGHT_MOIRE = 0.25

FUSION_THRESHOLD_UPPER = 0.70
FUSION_THRESHOLD_LOWER = 0.35

CNN_FALLBACK_INPUT_SIZE = (224, 224)
CNN_FALLBACK_STRICT_MODE = False
CNN_FALLBACK_PLACEHOLDER_SCORE = None
CNN_FALLBACK_UPPER = 0.65
CNN_FALLBACK_LOWER = 0.35

# ---------------------------------------------------------------------------
# SHARED
# ---------------------------------------------------------------------------
CALIBRATION_LOG_MODE = "csv"
CALIBRATION_LOG_PATH = "logs/calibration_scores.csv"