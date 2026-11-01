from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel
from skimage.color import deltaE_ciede2000, rgb2lab

from mixing_copilot.config import load_yaml
from mixing_copilot.vision.color import measure_color

TANK_RGB = {"t100_ml": (1.0, 1.0, 1.0), "t200_ml": (1.0, 0.28, 0.03), "t400_ml": (0.03, 0.20, 1.0)}


class VialPrediction(BaseModel):
    vial_id: int | str
    recipe_id: int | str
    observed_rgb: list[int] | None
    observed_lab: list[float] | None
    expected_rgb: list[int] | None
    expected_lab: list[float] | None
    expected_color_source: str | None
    delta_e_2000: float | None
    mismatch_score: float | None
    measurement_confidence: float
    alignment_confidence: float | None
    evidence_image_paths: list[str]
    explanation: str
    verification_status: str = "unverified"
    ground_truth_anomaly: bool | None = None


def expected_recipe_color(recipe: dict[str, Any]) -> tuple[list[int], list[float]] | None:
    def finite_value(name: str) -> float:
        value = recipe.get(name)
        numeric = float(value) if value is not None else 0.0
        return numeric if np.isfinite(numeric) else 0.0

    total = finite_value("total_ml")
    if total <= 0:
        return None
    rgb = np.asarray(
        [
            sum(finite_value(tank) * TANK_RGB[tank][channel] for tank in TANK_RGB) / total
            for channel in range(3)
        ]
    )
    lab = rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)
    return (np.rint(rgb * 255).astype(int).tolist(), lab.astype(float).tolist())


def predict_vial(
    frames: list[str | Path | np.ndarray],
    recipe: dict[str, Any],
    previous_recipe: dict[str, Any] | None = None,
    alignment: dict[str, Any] | None = None,
    features_config: str | Path = "configs/features.yaml",
) -> VialPrediction:
    config = load_yaml(features_config)
    measurements = []
    evidence: list[str] = []
    for item in frames:
        if isinstance(item, (str, Path)):
            image = cv2.imread(str(item), cv2.IMREAD_COLOR)
            evidence.append(str(item))
        else:
            image = item
        measurement, _ = measure_color(image, config)
        if measurement.observed_rgb is not None:
            measurements.append(measurement)
    if not measurements:
        return VialPrediction(
            vial_id=recipe.get("vial", "unknown"),
            recipe_id=recipe.get("recipe_id", "unknown"),
            observed_rgb=None,
            observed_lab=None,
            expected_rgb=None,
            expected_lab=None,
            expected_color_source=None,
            delta_e_2000=None,
            mismatch_score=None,
            measurement_confidence=0,
            alignment_confidence=(alignment or {}).get("confidence"),
            evidence_image_paths=evidence,
            explanation="No reliable liquid-color pixels were available.",
        )
    observed_lab = np.median(
        [[m.lab_l_median, m.lab_a_median, m.lab_b_median] for m in measurements], axis=0
    )
    observed_rgb = np.median([m.observed_rgb for m in measurements], axis=0).astype(int).tolist()
    confidence = float(np.median([m.confidence for m in measurements]))
    expected = expected_recipe_color(recipe)
    delta_e = None
    score = None
    if expected:
        expected_rgb, expected_lab = expected
        delta_e = float(deltaE_ciede2000(observed_lab, np.asarray(expected_lab)))
        score = float(np.clip(delta_e / 50.0, 0, 1))
        source = "synthetic_tank_setpoint_weighted_srgb_unvalidated"
        explanation = (
            "Observed color differs from the documented synthetic recipe proxy by "
            f"Delta E2000={delta_e:.1f}. "
            "The score is an interpretable mismatch index, not a contamination probability."
        )
    else:
        expected_rgb, expected_lab, source = None, None, None
        explanation = (
            "Observed color was measured, but no defensible expected-color proxy was supplied."
        )
    if previous_recipe:
        explanation += f" Previous recipe context: {previous_recipe.get('recipe_id', 'unknown')}."
    return VialPrediction(
        vial_id=recipe.get("vial", "unknown"),
        recipe_id=recipe.get("recipe_id", "unknown"),
        observed_rgb=observed_rgb,
        observed_lab=observed_lab.astype(float).tolist(),
        expected_rgb=expected_rgb,
        expected_lab=expected_lab,
        expected_color_source=source,
        delta_e_2000=delta_e,
        mismatch_score=score,
        measurement_confidence=confidence,
        alignment_confidence=(alignment or {}).get("confidence"),
        evidence_image_paths=evidence,
        explanation=explanation,
    )


def write_prediction(prediction: VialPrediction, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prediction.model_dump(), indent=2) + "\n", encoding="utf-8")
