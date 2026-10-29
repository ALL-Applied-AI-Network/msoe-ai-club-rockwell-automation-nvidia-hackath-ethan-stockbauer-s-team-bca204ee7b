from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import HuberRegressor, RANSACRegressor


@dataclass(frozen=True)
class AlignmentFit:
    offset_s: float
    scale: float
    residual_median_s: float
    residual_mad_s: float
    residual_max_abs_s: float
    inlier_count: int
    anchor_count: int
    confidence: float
    method: str = "ransac_huber_affine"

    def media_to_wall(self, media_seconds: float | np.ndarray) -> float | np.ndarray:
        return self.offset_s + self.scale * media_seconds

    def wall_to_media(self, wall_seconds: float | np.ndarray) -> float | np.ndarray:
        return (wall_seconds - self.offset_s) / self.scale

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def fit_affine_alignment(
    media_seconds: np.ndarray,
    wall_seconds: np.ndarray,
    residual_threshold_s: float = 0.35,
    minimum_inliers: int = 8,
) -> tuple[AlignmentFit, np.ndarray, np.ndarray]:
    x = np.asarray(media_seconds, dtype=float).reshape(-1, 1)
    y = np.asarray(wall_seconds, dtype=float)
    if len(x) != len(y) or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("alignment requires at least three finite paired anchors")

    seed_model = RANSACRegressor(
        estimator=HuberRegressor(epsilon=1.35),
        min_samples=max(3, min(minimum_inliers, len(x))),
        residual_threshold=residual_threshold_s,
        random_state=7,
    ).fit(x, y)
    inliers = np.asarray(seed_model.inlier_mask_, dtype=bool)
    if int(inliers.sum()) < minimum_inliers:
        raise ValueError(f"only {int(inliers.sum())} alignment inliers; need {minimum_inliers}")
    model = HuberRegressor(epsilon=1.35).fit(x[inliers], y[inliers])
    prediction = model.predict(x)
    residuals = y - prediction
    inlier_residuals = residuals[inliers]
    median = float(np.median(np.abs(inlier_residuals)))
    mad = float(np.median(np.abs(inlier_residuals - np.median(inlier_residuals))))
    confidence = float(np.clip((inliers.mean()) * np.exp(-median / residual_threshold_s), 0, 1))
    fit = AlignmentFit(
        offset_s=float(model.intercept_),
        scale=float(model.coef_[0]),
        residual_median_s=median,
        residual_mad_s=mad,
        residual_max_abs_s=float(np.max(np.abs(inlier_residuals))),
        inlier_count=int(inliers.sum()),
        anchor_count=len(x),
        confidence=confidence,
    )
    return fit, residuals, inliers
