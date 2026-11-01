from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mixing_copilot.audit.manifest import audit_dataset


def test_audit_reports_missing_and_recipe_coverage(tmp_path: Path):
    (tmp_path / "README.md").write_text("fixture", encoding="utf-8")
    pd.DataFrame([{"video_id": 5}]).to_csv(tmp_path / "videos.csv", index=False)
    labels = tmp_path / "vial_fill" / "labels"
    labels.mkdir(parents=True)
    pd.DataFrame([{"vial": 1, "recipe_id": 5}]).to_csv(
        labels / "video5_vial_recipes.csv", index=False
    )
    report = audit_dataset(tmp_path, include_hashes=False)
    assert report["file_count"] == 3
    assert report["recipe_coverage"]["vials"] == 1
    assert report["label_caveats"]["ground_truth_anomaly"] is None
    assert report["missing_required"]


def test_opencv_fixture_available(tmp_path: Path):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    path = tmp_path / "image.jpg"
    assert cv2.imwrite(str(path), image)
    assert cv2.imread(str(path)).shape == image.shape
