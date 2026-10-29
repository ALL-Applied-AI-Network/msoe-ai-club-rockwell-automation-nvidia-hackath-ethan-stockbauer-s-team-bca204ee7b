import numpy as np

from mixing_copilot.vision.color import measure_color


def test_color_measurement_finds_saturated_patch():
    image = np.full((100, 100, 3), 220, dtype=np.uint8)
    image[20:80, 30:70] = (20, 80, 220)
    config = {
        "roi": {"x": 20, "y": 10, "width": 60, "height": 80},
        "color": {
            "min_saturation": 12,
            "min_value": 25,
            "max_value": 242,
            "glare_value": 238,
            "min_valid_fraction": 0.08,
            "morphology_kernel": 3,
        },
    }
    measurement, overlay = measure_color(image, config)
    assert measurement.status == "ok"
    assert measurement.observed_rgb[0] > measurement.observed_rgb[2]
    assert overlay.shape == image.shape
