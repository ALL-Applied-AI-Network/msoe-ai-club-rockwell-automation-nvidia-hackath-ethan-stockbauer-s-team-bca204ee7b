from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from mixing_copilot.config import load_yaml
from mixing_copilot.vision.color import measure_file

FEATURE_COLUMNS = [
    "confidence",
    "valid_fraction",
    "glare_fraction",
    "pixel_count",
    "hsv_h_median",
    "hsv_s_median",
    "hsv_v_median",
    "lab_l_median",
    "lab_a_median",
    "lab_b_median",
    "lab_l_std",
    "lab_a_std",
    "lab_b_std",
]


def extract_features(output_root: Path, features_config_path: Path) -> pd.DataFrame:
    config = load_yaml(features_config_path)
    manifest_path = output_root / "data" / "interim" / "video5_extracted_frames.csv"
    manifest = pd.read_csv(manifest_path)
    qa_root = output_root / "artifacts" / "qa" / "color"
    qa_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        measurement, overlay = measure_file(output_root / row["image_path"], config)
        qa_path = qa_root / Path(row["image_path"]).name
        cv2.imwrite(str(qa_path), overlay)
        record = {
            **row,
            **measurement.to_dict(),
            "color_qa_path": qa_path.relative_to(output_root).as_posix(),
        }
        records.append(record)
    features = pd.DataFrame(records)
    output = output_root / "data" / "interim" / "video5_frame_features.csv"
    features.to_csv(output, index=False)
    return features


def build_datasets(dataset_root: Path, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(output_root / "data" / "interim" / "video5_frame_features.csv")
    recipes = pd.read_csv(dataset_root / "vial_fill" / "labels" / "video5_vial_recipes.csv")
    recipe_fields = [
        "vial",
        "type",
        "t100_ml",
        "t200_ml",
        "t400_ml",
        "total_ml",
        "dispense_order",
        "color_purpose",
    ]
    frame_data = features.merge(
        recipes[recipe_fields], left_on="vial_id", right_on="vial", how="left"
    ).drop(columns="vial")
    frame_data["time_within_fill_s"] = frame_data["original_wall_seconds"] - frame_data.groupby(
        "vial_id"
    )["original_wall_seconds"].transform("min")
    frame_data["previous_recipe_id"] = frame_data["recipe_id"].map(
        lambda value: value - 1 if value > 5 else pd.NA
    )
    frame_data["batch_position"] = (frame_data["vial_id"] - 1) % 3 + 1
    processed = output_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame_data.to_csv(processed / "per_frame.csv", index=False)
    frame_data.to_parquet(processed / "per_frame.parquet", index=False)

    aggregate = {
        column: ["median", "mean", "std"] for column in FEATURE_COLUMNS if column in frame_data
    }
    vial_data = frame_data.groupby("vial_id").agg(aggregate)
    vial_data.columns = [f"{column}_{stat}" for column, stat in vial_data.columns]
    vial_data = vial_data.reset_index()
    context = (
        frame_data.sort_values("corrected_media_seconds").groupby("vial_id").first().reset_index()
    )
    context_columns = [
        "vial_id",
        "video_id",
        "camera_id",
        "recipe_id",
        "recipe_transition_id",
        "previous_recipe_id",
        "batch_position",
        "alignment_confidence",
        "t100_ml",
        "t200_ml",
        "t400_ml",
        "total_ml",
        "dispense_order",
        "ground_truth_anomaly",
        "label_status",
    ]
    vial_data = context[context_columns].merge(vial_data, on="vial_id")
    vial_data.to_csv(processed / "per_vial.csv", index=False)
    vial_data.to_parquet(processed / "per_vial.parquet", index=False)
    return frame_data, vial_data
