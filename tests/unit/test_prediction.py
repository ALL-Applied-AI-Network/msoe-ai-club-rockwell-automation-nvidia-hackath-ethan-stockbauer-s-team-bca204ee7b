import numpy as np

from mixing_copilot.prediction import expected_recipe_color


def test_expected_recipe_color_uses_setpoint_weights():
    rgb, lab = expected_recipe_color({"total_ml": 20, "t100_ml": 10, "t200_ml": 10, "t400_ml": 0})
    assert np.allclose(rgb, [255, 163, 131], atol=1)
    assert len(lab) == 3
