from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorMeasurement:
    status: str
    confidence: float
    valid_fraction: float
    glare_fraction: float
    pixel_count: int
    hsv_h_median: float | None
    hsv_s_median: float | None
    hsv_v_median: float | None
    lab_l_median: float | None
    lab_a_median: float | None
    lab_b_median: float | None
    lab_l_std: float | None
    lab_a_std: float | None
    lab_b_std: float | None
    observed_rgb: list[int] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_color(image: np.ndarray, config: dict[str, Any]) -> tuple[ColorMeasurement, np.ndarray]:
    roi_config = config["roi"]
    color_config = config["color"]
    x, y = int(roi_config["x"]), int(roi_config["y"])
    width, height = int(roi_config["width"]), int(roi_config["height"])
    if image is None or image.ndim != 3:
        raise ValueError("color measurement requires a BGR image")
    if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError("configured ROI lies outside the image")
    roi = image[y : y + height, x : x + width]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    glare = value >= int(color_config["glare_value"])
    mask = (
        (saturation >= int(color_config["min_saturation"]))
        & (value >= int(color_config["min_value"]))
        & (value <= int(color_config["max_value"]))
    ).astype(np.uint8)
    kernel_size = int(color_config.get("morphology_kernel", 3))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Avoid support hardware and meniscus/nozzle boundaries at ROI edges.
    mask[:4, :] = 0
    mask[-8:, :] = 0
    mask[:, :4] = 0
    mask[:, -4:] = 0
    valid = mask.astype(bool)
    valid_fraction = float(valid.mean())
    glare_fraction = float(glare.mean())
    minimum = float(color_config["min_valid_fraction"])
    count = int(valid.sum())

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 2)
    colored_mask = np.zeros_like(roi)
    colored_mask[valid] = (0, 255, 0)
    overlay[y : y + height, x : x + width] = cv2.addWeighted(roi, 0.72, colored_mask, 0.28, 0)
    if count == 0:
        measurement = ColorMeasurement(
            "missing", 0.0, valid_fraction, glare_fraction, 0, *(None,) * 10
        )
        return measurement, overlay

    hsv_pixels = hsv[valid].astype(float)
    lab_pixels = lab[valid]
    bgr_pixels = roi[valid].astype(float)
    lab_pixels[:, 0] *= 100.0 / 255.0
    lab_pixels[:, 1:] -= 128.0
    median_lab = np.median(lab_pixels, axis=0)
    std_lab = np.std(lab_pixels, axis=0)
    median_hsv = np.median(hsv_pixels, axis=0)
    median_bgr = np.median(bgr_pixels, axis=0).astype(np.uint8)
    dispersion_penalty = float(np.exp(-np.linalg.norm(std_lab[1:]) / 35.0))
    confidence = float(
        np.clip(
            (valid_fraction / max(minimum * 3, 1e-6)) * dispersion_penalty * (1 - glare_fraction),
            0,
            1,
        )
    )
    status = "ok" if valid_fraction >= minimum and confidence >= 0.25 else "low_confidence"
    measurement = ColorMeasurement(
        status=status,
        confidence=confidence,
        valid_fraction=valid_fraction,
        glare_fraction=glare_fraction,
        pixel_count=count,
        hsv_h_median=float(median_hsv[0]),
        hsv_s_median=float(median_hsv[1]),
        hsv_v_median=float(median_hsv[2]),
        lab_l_median=float(median_lab[0]),
        lab_a_median=float(median_lab[1]),
        lab_b_median=float(median_lab[2]),
        lab_l_std=float(std_lab[0]),
        lab_a_std=float(std_lab[1]),
        lab_b_std=float(std_lab[2]),
        observed_rgb=[int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0])],
    )
    swatch = tuple(int(channel) for channel in median_bgr)
    cv2.rectangle(overlay, (8, 8), (72, 52), swatch, -1)
    cv2.putText(
        overlay,
        f"{status} c={confidence:.2f}",
        (82, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
    )
    return measurement, overlay


def measure_file(path: Path, config: dict[str, Any]) -> tuple[ColorMeasurement, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read image: {path}")
    return measure_color(image, config)
