"""
utils/timing.py

Lightweight stage timer. Usage:

    timer = StageTimer()
    with timer.stage("yolo_localize"):
        ... do work ...
    with timer.stage("fusion"):
        ... do work ...
    result_dict["stage_timings_ms"] = timer.as_dict()
"""

import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self._timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._timings[name] = round(elapsed_ms, 2)

    def as_dict(self) -> dict[str, float]:
        self._timings["total"] = round(sum(self._timings.values()), 2)
        return dict(self._timings)
