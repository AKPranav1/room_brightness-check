"""
utils/logging_calibration.py

Logs EVERY raw scalar score per request (not just the final decision) so thresholds
in config.py can be tuned against real reference samples instead of guessed.
"""

import csv
import os
import time
from config import CALIBRATION_LOG_MODE, CALIBRATION_LOG_PATH

_CSV_HEADER = [
    "timestamp", "request_id", "pillar",
    "mean_delta_magnitude", "delta_concentration", "moire_score",
    "fusion_confidence", "cnn_fallback_score", "band", "action",
    "visibility_score", "brightness_score", "clarity_score", "composite_score",
]


def _ensure_csv_exists():
    os.makedirs(os.path.dirname(CALIBRATION_LOG_PATH), exist_ok=True)
    if not os.path.exists(CALIBRATION_LOG_PATH):
        with open(CALIBRATION_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)


def log_calibration_row(request_id: str, pillar: str, scores: dict) -> None:
    row = {
        "timestamp": time.time(),
        "request_id": request_id,
        "pillar": pillar,
        **scores,
    }

    if CALIBRATION_LOG_MODE in ("stdout", "both"):
        print(f"[calibration] {row}")

    if CALIBRATION_LOG_MODE in ("csv", "both"):
        _ensure_csv_exists()
        with open(CALIBRATION_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADER, extrasaction="ignore")
            writer.writerow(row)
