from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualMatch:
    media_seconds: float
    score: float
    second_best_margin: float


def image_signature(image: np.ndarray, size: tuple[int, int] = (48, 36)) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("cannot compute signature for an empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    norm = float(np.linalg.norm(small))
    return (small / norm if norm > 1e-8 else small).ravel()


def match_signature(query: np.ndarray, candidates: np.ndarray, times: np.ndarray) -> VisualMatch:
    if len(candidates) != len(times) or len(times) < 2:
        raise ValueError("visual matching needs at least two time-indexed candidates")
    scores = candidates @ query
    order = np.argsort(scores)[::-1]
    best, second = int(order[0]), int(order[1])
    return VisualMatch(
        float(times[best]), float(scores[best]), float(scores[best] - scores[second])
    )


def build_video_signature_index(video: Path, step_s: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    stride = max(1, round(fps * step_s))
    times: list[float] = []
    signatures: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            reported = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            times.append(reported if reported > 0 else index / fps)
            signatures.append(image_signature(frame))
        index += 1
    capture.release()
    if len(signatures) < 2:
        raise OSError(f"too few decoded frames: {video}")
    return np.asarray(times), np.asarray(signatures)


def read_frame_at_time(video: Path, media_seconds: float) -> tuple[np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {video}")
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, media_seconds) * 1000.0)
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise OSError(f"cannot decode {video} at {media_seconds:.3f}s")
    decoded_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    capture.release()
    return frame, decoded_s, frame_index


def match_still(
    still: Path,
    times: np.ndarray,
    signatures: np.ndarray,
    predicted_media_s: float,
    search_radius_s: float,
) -> VisualMatch:
    image = cv2.imread(str(still), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read still image: {still}")
    mask = np.abs(times - predicted_media_s) <= search_radius_s
    if int(mask.sum()) < 2:
        raise ValueError(f"not enough candidates near {predicted_media_s:.3f}s")
    return match_signature(image_signature(image), signatures[mask], times[mask])
