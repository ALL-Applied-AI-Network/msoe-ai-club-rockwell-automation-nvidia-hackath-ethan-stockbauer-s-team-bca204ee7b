import numpy as np

from mixing_copilot.alignment.matching import image_signature, match_signature


def test_signature_matching_selects_expected_candidate():
    images = []
    for column in (5, 15, 25):
        image = np.zeros((36, 48, 3), dtype=np.uint8)
        image[:, column : column + 3] = 255
        images.append(image_signature(image))
    candidates = np.asarray(images)
    match = match_signature(images[1], candidates, np.asarray([1.0, 2.0, 3.0]))
    assert match.media_seconds == 2.0
    assert match.score > 0.99
