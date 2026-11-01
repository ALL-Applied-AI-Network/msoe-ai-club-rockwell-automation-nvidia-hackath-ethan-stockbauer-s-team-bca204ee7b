from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from mixing_copilot.alignment.matching import read_frame_at_time
from mixing_copilot.alignment.model import AlignmentFit


def _stage_times(start_s: float, end_s: float, settings: dict[str, Any]) -> list[tuple[str, float]]:
    duration = end_s - start_s
    return [
        ("pre_fill", start_s - float(settings.get("pre_offset_s", 1.0))),
        ("early_fill", start_s + 0.15 * duration),
        ("mid_fill", start_s + 0.50 * duration),
        ("late_fill", start_s + 0.85 * duration),
        ("post_fill", end_s + float(settings.get("post_offset_s", 1.0))),
    ]


def _contact_sheet(rows: pd.DataFrame, image_root: Path, output: Path) -> None:
    tiles: list[np.ndarray] = []
    for _, row in rows.iterrows():
        image = cv2.imread(str(image_root / row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        tile = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (320, 28), (0, 0, 0), -1)
        label = f"v{int(row['vial_id']):02d} r{int(row['recipe_id'])} {row['stage']}"
        cv2.putText(tile, label, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return
    columns = 5
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * ((-len(tiles)) % columns))
    sheet = np.vstack(
        [np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def extract_vial_frames(
    dataset_root: Path,
    output_root: Path,
    camera: str,
    video_id: int,
    settings: dict[str, Any],
) -> pd.DataFrame:
    alignment_path = (
        output_root / "data" / "interim" / "alignment" / f"{camera}_video_{video_id}_alignment.json"
    )
    with alignment_path.open(encoding="utf-8") as handle:
        alignment = json.load(handle)
    fit = AlignmentFit(**alignment["fit"])
    video = dataset_root / camera / "videos" / f"{camera}_video_{video_id}.mp4"
    recipes_path = dataset_root / "vial_fill" / "labels" / f"video{video_id}_vial_recipes.csv"
    recipes = pd.read_csv(recipes_path)
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    duration_s = frame_count / fps

    frame_root = output_root / "data" / "interim" / "frames" / f"{camera}_video_{video_id}"
    frame_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for recipe in recipes.itertuples(index=False):
        for stage, wall_s in _stage_times(float(recipe.start_s), float(recipe.end_s), settings):
            corrected_s = float(fit.wall_to_media(wall_s))
            if corrected_s < 0 or corrected_s >= duration_s:
                continue
            name = f"video{video_id}_vial{int(recipe.vial):02d}_{stage}.jpg"
            destination = frame_root / name
            frame, decoded_s, decoded_index = read_frame_at_time(video, corrected_s)
            if not destination.is_file() and not cv2.imwrite(str(destination), frame):
                raise OSError(f"could not write extracted frame: {destination}")
            records.append(
                {
                    "video_id": video_id,
                    "camera_id": camera,
                    "vial_id": int(recipe.vial),
                    "recipe_id": int(recipe.recipe_id),
                    "recipe_transition_id": int(recipe.recipe_id - 5)
                    if recipe.recipe_id > 5
                    else None,
                    "stage": stage,
                    "original_wall_seconds": wall_s,
                    "corrected_media_seconds": corrected_s,
                    "decoded_media_seconds": decoded_s,
                    "decoded_frame_index": decoded_index,
                    "alignment_residual_seconds": decoded_s - corrected_s,
                    "alignment_confidence": fit.confidence,
                    "source_video": str(video),
                    "source_sha256": alignment["source_sha256"],
                    "image_path": destination.relative_to(output_root).as_posix(),
                    "extraction_config": "configs/dataset.yaml",
                    "ground_truth_anomaly": None,
                    "label_status": "unverified",
                }
            )
    manifest = pd.DataFrame(records).sort_values(["vial_id", "corrected_media_seconds"])
    manifest_path = output_root / "data" / "interim" / "video5_extracted_frames.csv"
    manifest.to_csv(manifest_path, index=False)
    qa_root = output_root / "artifacts" / "qa" / "contact_sheets"
    for recipe_id, rows in manifest.groupby("recipe_id"):
        _contact_sheet(rows, output_root, qa_root / f"recipe_{int(recipe_id):02d}.jpg")
    return manifest
