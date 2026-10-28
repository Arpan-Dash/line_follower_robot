"""Fast, deterministic line extraction helpers used by the controller."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LineObservation:
    """The result of processing one camera frame."""

    error: float
    confidence: float
    line_x: Optional[float]
    mask: np.ndarray
    roi_top: int


def _largest_dark_run(row: np.ndarray, expected_x: float) -> Optional[Tuple[float, int]]:
    """Return the dark run closest to the last known line position."""
    dark = row > 0
    transitions = np.diff(np.pad(dark.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    if not len(starts):
        return None

    runs = []
    for start, end in zip(starts, ends):
        if end - start >= 2:
            runs.append((float((start + end - 1) / 2.0), int(end - start)))
    if not runs:
        return None
    return min(runs, key=lambda run: abs(run[0] - expected_x))


def detect_line(
    image: np.ndarray,
    previous_x: Optional[float],
    roi_top_ratio: float = 0.42,
    threshold: int = 105,
) -> LineObservation:
    """Detect a dark line and return a normalized horizontal tracking error."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    height, width = gray.shape[:2]
    roi_top = max(0, min(height - 1, int(height * roi_top_ratio)))
    roi = gray[roi_top:, :]
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # Otsu adapts to lighting, while the fixed limit prevents a pale floor from
    # becoming a false line when the frame has very little contrast.
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, fixed = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(otsu, fixed)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    expected_x = float(previous_x if previous_x is not None else width / 2.0)
    rows = np.linspace(int(mask.shape[0] * 0.15), mask.shape[0] - 2, 8).astype(int)
    points = []
    weights = []
    for row_index in rows:
        run = _largest_dark_run(mask[row_index], expected_x)
        if run is not None:
            x, run_width = run
            points.append(x)
            weights.append(float(run_width) * (1.0 + row_index / mask.shape[0]))

    if not points:
        return LineObservation(0.0, 0.0, None, mask, roi_top)

    line_x = float(np.average(points, weights=weights))
    spread = float(np.average((np.asarray(points) - line_x) ** 2, weights=weights) ** 0.5)
    coverage = min(1.0, len(points) / len(rows))
    confidence = float(np.clip(coverage * np.exp(-spread / max(10.0, width * 0.08)), 0.0, 1.0))
    error = float(np.clip((line_x - width / 2.0) / (width / 2.0), -1.0, 1.0))
    return LineObservation(error, confidence, line_x, mask, roi_top)


def make_debug_image(
    image: np.ndarray,
    observation: LineObservation,
    error: float,
    angular_command: float,
) -> np.ndarray:
    """Draw the ROI, detected line, error and controller state on the image."""
    debug = image.copy()
    height, width = debug.shape[:2]
    cv2.line(debug, (width // 2, 0), (width // 2, height), (255, 180, 0), 1)
    cv2.line(debug, (0, observation.roi_top), (width, observation.roi_top), (0, 180, 255), 1)
    if observation.line_x is not None:
        x = int(observation.line_x)
        cv2.line(debug, (x, observation.roi_top), (x, height), (0, 255, 0), 2)
    cv2.putText(
        debug,
        f"error: {error:+.3f}  confidence: {observation.confidence:.2f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug,
        f"angular: {angular_command:+.3f}",
        (8, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return debug
