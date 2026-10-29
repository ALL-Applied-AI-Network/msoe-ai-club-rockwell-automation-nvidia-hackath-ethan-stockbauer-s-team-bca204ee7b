import numpy as np
import pytest

from mixing_copilot.alignment.model import fit_affine_alignment


def test_robust_affine_fit_rejects_outlier():
    media = np.linspace(0, 100, 30)
    wall = 0.63 + 1.216 * media
    wall[8] += 12
    fit, residuals, inliers = fit_affine_alignment(media, wall, minimum_inliers=10)
    assert fit.offset_s == pytest.approx(0.63, abs=1e-3)
    assert fit.scale == pytest.approx(1.216, abs=1e-4)
    assert not inliers[8]
    assert residuals[8] > 10
