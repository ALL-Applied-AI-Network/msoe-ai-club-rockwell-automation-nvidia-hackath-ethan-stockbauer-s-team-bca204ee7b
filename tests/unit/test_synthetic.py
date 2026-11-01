from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from mixing_copilot.cli import main as cli_main
from mixing_copilot.dataset.synthetic import (
    VideoMetadata,
    process_synthetic_dataset,
    select_stage_rows,
    validate_synthetic_tables,
)


def synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame, VideoMetadata]:
    fps = 10.0
    start = datetime(2026, 1, 1)
    rows = []
    recipe_specs = [
        (1, 10, 0, 2, 10.0, 0.0, 0.0, "T100"),
        (2, 11, 4, 6, 0.0, 10.0, 0.0, "T200"),
    ]
    by_frame = {
        frame: spec
        for spec in recipe_specs
        for frame in range(spec[2], spec[3] + 1)
    }
    for frame in range(8):
        spec = by_frame.get(frame)
        timestamp = start + timedelta(seconds=frame / fps)
        if spec is None:
            rows.append(
                {
                    "filename": f"frame_{frame:06d}.png",
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "video_s": frame / fps,
                    "in_fill": 0,
                    "vial": None,
                    "recipe_id": None,
                    "type": "between_fills",
                    "t100_ml": None,
                    "t200_ml": None,
                    "t400_ml": None,
                    "total_ml": None,
                    "dispense_order": None,
                    "color_purpose": None,
                    "mp4_frame": frame,
                    "render_frame": 1,
                }
            )
            continue
        vial, recipe_id, _, _, t100, t200, t400, order = spec
        rows.append(
            {
                "filename": f"frame_{frame:06d}.png",
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "video_s": frame / fps,
                "in_fill": 1,
                "vial": vial,
                "recipe_id": recipe_id,
                "type": "fixture",
                "t100_ml": t100,
                "t200_ml": t200,
                "t400_ml": t400,
                "total_ml": 10.0,
                "dispense_order": order,
                "color_purpose": "unit-test",
                "mp4_frame": frame,
                "render_frame": 100 + frame,
            }
        )
    recipes = pd.DataFrame(
        [
            {
                "vial": vial,
                "recipe_id": recipe_id,
                "type": "fixture",
                "t100_ml": t100,
                "t200_ml": t200,
                "t400_ml": t400,
                "total_ml": 10.0,
                "dispense_order": order,
                "color_purpose": "unit-test",
                "start_time": f"00:00:00.{start_frame * 100:03d}",
                "end_time": f"00:00:00.{end_frame * 100:03d}",
                "start_s": start_frame / fps,
                "end_s": end_frame / fps,
                "duration_s": (end_frame - start_frame) / fps,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
            for vial, recipe_id, start_frame, end_frame, t100, t200, t400, order in recipe_specs
        ]
    )
    return pd.DataFrame(rows), recipes, VideoMetadata(fps=fps, frame_count=8, width=640, height=480)


def test_synthetic_fixture_keeps_gap_stages_as_transition_evidence():
    labels, recipes, video = synthetic_tables()
    labels, recipes, report = validate_synthetic_tables(labels, recipes, video)
    selected = select_stage_rows(labels, recipes, video.fps)
    stages = {
        (int(recipe["vial"]), stage, int(row["mp4_frame"]))
        for recipe, stage, row in selected
    }
    assert report["frame_label_rows"] == 8
    assert report["ground_truth_anomaly"] is None
    assert (1, "early_fill", 0) in stages
    assert (1, "post_fill", 3) in stages
    assert (2, "pre_fill", 3) in stages
    assert (2, "late_fill", 6) in stages
    gap_stages = [(stage, int(row["in_fill"])) for _, stage, row in selected if not row["in_fill"]]
    assert gap_stages
    assert all(in_fill == 0 for _, in_fill in gap_stages)


def test_synthetic_fixture_rejects_noncontiguous_frame_indices():
    labels, recipes, video = synthetic_tables()
    labels.loc[2, "mp4_frame"] = 9
    with pytest.raises(ValueError, match="contiguous"):
        validate_synthetic_tables(labels, recipes, video)


def test_synthetic_fixture_rejects_timestamp_drift():
    labels, recipes, video = synthetic_tables()
    labels.loc[4, "timestamp"] = "2026-01-01 00:00:09.000000"
    with pytest.raises(ValueError, match="timestamp"):
        validate_synthetic_tables(labels, recipes, video)


def test_synthetic_fixture_rejects_recipe_boundary_on_gap_frame():
    labels, recipes, video = synthetic_tables()
    recipes.loc[1, "start_frame"] = 3
    recipes.loc[1, "start_s"] = 0.3
    recipes.loc[1, "start_time"] = "00:00:00.300"
    recipes.loc[1, "duration_s"] = 0.3
    with pytest.raises(ValueError, match="non-fill"):
        validate_synthetic_tables(labels, recipes, video)


def test_synthetic_fixture_rejects_malformed_nonblank_zero_volume():
    labels, recipes, video = synthetic_tables()
    labels["t400_ml"] = labels["t400_ml"].astype(object)
    labels.loc[(labels["in_fill"] == 1) & (labels["t400_ml"] == 0), "t400_ml"] = "bad"
    with pytest.raises(ValueError, match="malformed nonblank numeric"):
        validate_synthetic_tables(labels, recipes, video)


def test_synthetic_fixture_rejects_shifted_recipe_clocks():
    labels, recipes, video = synthetic_tables()
    recipes["start_time"] = ["01:00:00.000", "01:00:00.400"]
    recipes["end_time"] = ["01:00:00.200", "01:00:00.600"]
    with pytest.raises(ValueError, match="frame-label timestamp"):
        validate_synthetic_tables(labels, recipes, video)


def test_synthetic_fixture_accepts_recipe_clocks_across_midnight():
    labels, recipes, video = synthetic_tables()
    start = datetime(2026, 1, 1, 23, 59, 59, 800000)
    labels["timestamp"] = [
        (start + timedelta(seconds=frame / video.fps)).strftime("%Y-%m-%d %H:%M:%S.%f")
        for frame in labels["mp4_frame"]
    ]
    recipes["start_time"] = ["23:59:59.800", "00:00:00.200"]
    recipes["end_time"] = ["00:00:00.000", "00:00:00.400"]
    _, _, report = validate_synthetic_tables(labels, recipes, video)
    assert report["recipe_rows"] == 2


def _write_fixture_video(path: Path, labels: pd.DataFrame) -> None:
    size = (96, 72)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    if not writer.isOpened():
        pytest.fail("OpenCV could not create the temporary MP4 fixture")
    try:
        for row in labels.to_dict(orient="records"):
            if not row["in_fill"]:
                color = (180, 180, 180)
            elif int(row["vial"]) == 1:
                color = (20, 80, 220)
            else:
                color = (220, 80, 20)
            writer.write(np.full((size[1], size[0], 3), color, dtype=np.uint8))
    finally:
        writer.release()


def test_process_synthetic_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    labels, recipes, _ = synthetic_tables()
    video_path = tmp_path / "fixture.mp4"
    labels_path = tmp_path / "fixture_frame_labels.csv"
    recipes_path = tmp_path / "fixture_recipes.csv"
    features_config = tmp_path / "features.yaml"
    output_root = tmp_path / "output"
    _write_fixture_video(video_path, labels)
    labels.to_csv(labels_path, index=False)
    recipes.to_csv(recipes_path, index=False)
    features_config.write_text(
        yaml.safe_dump(
            {
                "roi": {"x": 20, "y": 15, "width": 40, "height": 40},
                "color": {
                    "min_saturation": 10,
                    "min_value": 20,
                    "max_value": 242,
                    "glare_value": 238,
                    "min_valid_fraction": 0.08,
                    "morphology_kernel": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    validation = process_synthetic_dataset(
        video_path,
        labels_path,
        recipes_path,
        tmp_path / "validation-only",
        "fixture_validation",
        None,
        camera_id="FillCam",
        video_id=7,
        validate_only=True,
    )
    assert validation["camera_id"] == "FillCam"
    assert validation["video_id"] == 7

    result = cli_main(
        [
            "process-synthetic",
            "--video",
            str(video_path),
            "--frame-labels",
            str(labels_path),
            "--recipes",
            str(recipes_path),
            "--dataset-id",
            "fixture_dataset",
            "--video-id",
            "7",
            "--camera-id",
            "FillCam",
            "--output-root",
            str(output_root),
            "--features-config",
            str(features_config),
        ]
    )
    assert result == 0
    assert "fixture_dataset" in capsys.readouterr().out

    interim = output_root / "data" / "interim" / "synthetic" / "fixture_dataset"
    processed = output_root / "data" / "processed" / "synthetic" / "fixture_dataset"
    qa_root = output_root / "artifacts" / "qa" / "synthetic" / "fixture_dataset" / "color"
    manifest = pd.read_csv(interim / "extracted_frames.csv")
    features = pd.read_csv(interim / "frame_features.csv")
    per_frame = pd.read_csv(processed / "per_frame.csv")
    per_vial = pd.read_csv(processed / "per_vial.csv")

    assert {"dataset_id", "video_id", "camera_id", "measurement_eligible"}.issubset(
        manifest.columns
    )
    assert pd.api.types.is_integer_dtype(manifest["video_id"])
    assert manifest["dataset_id"].eq("fixture_dataset").all()
    assert manifest["camera_id"].eq("FillCam").all()
    assert len(manifest) == 9
    assert int(manifest["measurement_eligible"].sum()) == 6
    assert set(manifest.loc[~manifest["measurement_eligible"], "stage"]) <= {
        "pre_fill",
        "post_fill",
    }
    assert features.loc[~features["measurement_eligible"], "status"].eq("not_measured").all()
    assert features.loc[~features["measurement_eligible"], "color_qa_path"].isna().all()
    assert len(list(qa_root.glob("*.jpg"))) == 6
    assert len(per_frame) == 9
    assert len(per_vial) == 2
    eligible_medians = features.loc[features["measurement_eligible"]].groupby("vial_id")[
        "pixel_count"
    ].median()
    actual_medians = per_vial.set_index("vial_id")["pixel_count_median"]
    pd.testing.assert_series_equal(actual_medians, eligible_medians, check_names=False)

    with pytest.raises(ValueError, match="CapSealCam color extraction is disabled"):
        process_synthetic_dataset(
            video_path,
            labels_path,
            recipes_path,
            tmp_path / "cap-seal",
            "cap_seal_fixture",
            features_config,
            camera_id="CapSealCam",
            video_id=7,
        )
