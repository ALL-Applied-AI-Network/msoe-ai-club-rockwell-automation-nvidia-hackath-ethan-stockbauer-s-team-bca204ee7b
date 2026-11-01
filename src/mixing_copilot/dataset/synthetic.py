from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from mixing_copilot.config import load_yaml
from mixing_copilot.dataset.build import build_datasets, extract_features
from mixing_copilot.io.files import atomic_json_dump, sha256_file

FRAME_LABEL_COLUMNS = {
    "filename",
    "timestamp",
    "video_s",
    "in_fill",
    "vial",
    "recipe_id",
    "type",
    "t100_ml",
    "t200_ml",
    "t400_ml",
    "total_ml",
    "dispense_order",
    "color_purpose",
    "mp4_frame",
    "render_frame",
}

RECIPE_COLUMNS = {
    "vial",
    "recipe_id",
    "type",
    "t100_ml",
    "t200_ml",
    "t400_ml",
    "total_ml",
    "dispense_order",
    "color_purpose",
    "start_time",
    "end_time",
    "start_s",
    "end_s",
    "duration_s",
    "start_frame",
    "end_frame",
}

VOLUME_COLUMNS = ["t100_ml", "t200_ml", "t400_ml"]
@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


def probe_video(path: Path) -> VideoMetadata:
    if not path.is_file():
        raise FileNotFoundError(f"synthetic video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open synthetic video: {path}")
    metadata = VideoMetadata(
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    if metadata.fps <= 0 or metadata.frame_count <= 0:
        raise ValueError(f"synthetic video has invalid metadata: {metadata}")
    return metadata


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")


def _validate_dataset_id(dataset_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9._-]+", dataset_id) is None or dataset_id in {".", ".."}:
        raise ValueError(
            "dataset_id may contain only letters, numbers, dots, underscores, and hyphens"
        )


def _validate_camera_id(camera_id: str) -> None:
    if camera_id not in {"FillCam", "CapSealCam"}:
        raise ValueError("camera_id must be FillCam or CapSealCam")


def _blank_values(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _integer_values(series: pd.Series, name: str, *, nullable: bool = False) -> pd.Series:
    blank = _blank_values(series)
    if not nullable and blank.any():
        raise ValueError(f"{name} cannot contain null values")
    numeric = pd.to_numeric(series.mask(blank), errors="raise")
    present = numeric.dropna()
    if not np.allclose(present, np.rint(present), atol=0, rtol=0):
        raise ValueError(f"{name} must contain integer values")
    return numeric.astype("Int64" if nullable else int)


def _numeric_or_blank(series: pd.Series, name: str) -> pd.Series:
    blank = _blank_values(series)
    try:
        numeric = pd.to_numeric(series.mask(blank), errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} contains a malformed nonblank numeric value") from error
    return numeric.astype(float)


def _seconds_of_day(value: pd.Timestamp) -> float:
    return (
        value.hour * 60 * 60
        + value.minute * 60
        + value.second
        + value.microsecond / 1_000_000
        + value.nanosecond / 1_000_000_000
    )


def _clock_distance_seconds(first: float, second: float) -> float:
    return abs((first - second + 12 * 60 * 60) % (24 * 60 * 60) - 12 * 60 * 60)


def _normalized_tables(
    frame_labels: pd.DataFrame, recipes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(frame_labels, FRAME_LABEL_COLUMNS, "frame-label CSV")
    _require_columns(recipes, RECIPE_COLUMNS, "recipe CSV")
    labels = frame_labels.copy()
    recipe_table = recipes.copy()

    labels["mp4_frame"] = _integer_values(labels["mp4_frame"], "mp4_frame")
    labels["render_frame"] = _integer_values(labels["render_frame"], "render_frame")
    labels["in_fill"] = _integer_values(labels["in_fill"], "in_fill")
    labels["vial"] = _integer_values(labels["vial"], "frame-label vial", nullable=True)
    labels["recipe_id"] = _integer_values(
        labels["recipe_id"], "frame-label recipe_id", nullable=True
    )
    labels["video_s"] = pd.to_numeric(labels["video_s"], errors="raise").astype(float)
    labels["_timestamp"] = pd.to_datetime(labels["timestamp"], errors="raise")
    for column in [*VOLUME_COLUMNS, "total_ml"]:
        labels[column] = _numeric_or_blank(labels[column], f"frame-label {column}")

    for column in ["vial", "recipe_id", "start_frame", "end_frame"]:
        recipe_table[column] = _integer_values(recipe_table[column], f"recipe {column}")
    for column in [*VOLUME_COLUMNS, "total_ml", "start_s", "end_s", "duration_s"]:
        recipe_table[column] = pd.to_numeric(recipe_table[column], errors="raise").astype(float)
    return labels, recipe_table


def validate_synthetic_tables(
    frame_labels: pd.DataFrame,
    recipes: pd.DataFrame,
    video: VideoMetadata,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    labels, recipe_table = _normalized_tables(frame_labels, recipes)
    if len(labels) != video.frame_count:
        raise ValueError(
            f"frame-label row count {len(labels)} does not match video frame count "
            f"{video.frame_count}"
        )
    expected_frames = np.arange(video.frame_count)
    actual_frames = labels["mp4_frame"].to_numpy()
    if not np.array_equal(actual_frames, expected_frames):
        raise ValueError("mp4_frame must be unique, ordered, and contiguous from zero")
    if labels["filename"].isna().any() or labels["filename"].duplicated().any():
        raise ValueError("frame-label filename values must be present and unique")
    if not labels["in_fill"].isin([0, 1]).all():
        raise ValueError("in_fill must contain only 0 or 1")
    if (labels["render_frame"] < 0).any():
        raise ValueError("render_frame cannot be negative")

    time_tolerance = max(0.0011, 0.05 / video.fps)
    expected_seconds = labels["mp4_frame"].to_numpy(dtype=float) / video.fps
    if not np.allclose(labels["video_s"], expected_seconds, atol=time_tolerance, rtol=0):
        raise ValueError("video_s is inconsistent with mp4_frame and video FPS")
    if not labels["video_s"].is_monotonic_increasing:
        raise ValueError("video_s must be monotonic")
    if not labels["_timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp must be monotonic")
    timestamp_elapsed = (
        labels["_timestamp"] - labels["_timestamp"].iloc[0]
    ).dt.total_seconds()
    if not np.allclose(
        timestamp_elapsed,
        labels["video_s"] - labels["video_s"].iloc[0],
        atol=max(0.002, 0.1 / video.fps),
        rtol=0,
    ):
        raise ValueError("timestamp progression is inconsistent with video_s")

    if recipe_table.empty or recipe_table["vial"].duplicated().any():
        raise ValueError("recipe vial identifiers must be present and unique")
    if (recipe_table["vial"] < 1).any():
        raise ValueError("recipe vial identifiers must be positive")
    volumes = recipe_table[VOLUME_COLUMNS].fillna(0)
    if (volumes < 0).any().any() or (recipe_table["total_ml"] <= 0).any():
        raise ValueError("recipe volumes must be non-negative and total_ml must be positive")
    if not np.allclose(volumes.sum(axis=1), recipe_table["total_ml"], atol=1e-6, rtol=1e-6):
        raise ValueError("recipe total_ml must equal the sum of tank volumes")

    ordered_recipes = recipe_table.sort_values("start_frame").reset_index(drop=True)
    if (ordered_recipes["start_frame"] < 0).any() or (
        ordered_recipes["end_frame"] >= video.frame_count
    ).any():
        raise ValueError("recipe frame boundaries lie outside the video")
    if (ordered_recipes["end_frame"] < ordered_recipes["start_frame"]).any():
        raise ValueError("recipe end_frame cannot precede start_frame")
    if len(ordered_recipes) > 1 and (
        ordered_recipes["start_frame"].iloc[1:].to_numpy()
        <= ordered_recipes["end_frame"].iloc[:-1].to_numpy()
    ).any():
        raise ValueError("recipe frame ranges overlap")

    if not np.allclose(
        ordered_recipes["start_s"],
        ordered_recipes["start_frame"] / video.fps,
        atol=time_tolerance,
        rtol=0,
    ):
        raise ValueError("recipe start_s is inconsistent with start_frame and video FPS")
    if not np.allclose(
        ordered_recipes["end_s"],
        ordered_recipes["end_frame"] / video.fps,
        atol=time_tolerance,
        rtol=0,
    ):
        raise ValueError("recipe end_s is inconsistent with end_frame and video FPS")
    expected_duration = (
        ordered_recipes["end_frame"] - ordered_recipes["start_frame"]
    ) / video.fps
    if not np.allclose(
        ordered_recipes["duration_s"], expected_duration, atol=time_tolerance, rtol=0
    ):
        raise ValueError("recipe duration_s is inconsistent with its frame boundaries")
    recipe_start_times = pd.to_timedelta(ordered_recipes["start_time"], errors="raise")
    recipe_end_times = pd.to_timedelta(ordered_recipes["end_time"], errors="raise")
    clock_durations = (recipe_end_times - recipe_start_times).dt.total_seconds()
    clock_durations = np.where(clock_durations < 0, clock_durations + 24 * 60 * 60, clock_durations)
    if not np.allclose(
        clock_durations, ordered_recipes["duration_s"], atol=time_tolerance, rtol=0
    ):
        raise ValueError("recipe start_time/end_time are inconsistent with duration_s")

    clock_tolerance = max(0.002, 0.1 / video.fps)
    for row_index, recipe in ordered_recipes.iterrows():
        start_frame = int(recipe["start_frame"])
        end_frame = int(recipe["end_frame"])
        frame_start_clock = _seconds_of_day(labels.iloc[start_frame]["_timestamp"])
        frame_end_clock = _seconds_of_day(labels.iloc[end_frame]["_timestamp"])
        recipe_start_clock = float(recipe_start_times.iloc[row_index].total_seconds()) % (
            24 * 60 * 60
        )
        recipe_end_clock = float(recipe_end_times.iloc[row_index].total_seconds()) % (
            24 * 60 * 60
        )
        if _clock_distance_seconds(frame_start_clock, recipe_start_clock) > clock_tolerance:
            raise ValueError(
                f"vial {recipe['vial']} start_time does not match its frame-label timestamp"
            )
        if _clock_distance_seconds(frame_end_clock, recipe_end_clock) > clock_tolerance:
            raise ValueError(
                f"vial {recipe['vial']} end_time does not match its frame-label timestamp"
            )

    covered = np.zeros(video.frame_count, dtype=bool)
    for recipe in ordered_recipes.to_dict(orient="records"):
        start, end = int(recipe["start_frame"]), int(recipe["end_frame"])
        if covered[start : end + 1].any():
            raise ValueError("recipe frame ranges overlap")
        covered[start : end + 1] = True
        rows = labels.iloc[start : end + 1]
        if not (rows["in_fill"] == 1).all():
            raise ValueError(f"vial {recipe['vial']} range contains non-fill frame labels")
        if not (rows["vial"] == recipe["vial"]).all():
            raise ValueError(f"vial {recipe['vial']} range has inconsistent vial labels")
        if not (rows["recipe_id"] == recipe["recipe_id"]).all():
            raise ValueError(f"vial {recipe['vial']} range has inconsistent recipe_id labels")
        for column in [*VOLUME_COLUMNS, "total_ml"]:
            actual = rows[column].fillna(0).to_numpy(dtype=float)
            expected = 0.0 if pd.isna(recipe[column]) else float(recipe[column])
            if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
                raise ValueError(
                    f"vial {recipe['vial']} frame labels disagree on recipe field {column}"
                )
        for column in ["type", "dispense_order", "color_purpose"]:
            actual = rows[column].fillna("").astype(str).str.strip()
            expected = "" if pd.isna(recipe[column]) else str(recipe[column]).strip()
            if not (actual == expected).all():
                raise ValueError(
                    f"vial {recipe['vial']} frame labels disagree on recipe field {column}"
                )

    gap_rows = labels.loc[~covered]
    if not (gap_rows["in_fill"] == 0).all():
        raise ValueError("frames outside recipe ranges must be labeled as between-fill gaps")
    if gap_rows[["vial", "recipe_id"]].notna().any().any():
        raise ValueError("between-fill gap rows cannot contain vial or recipe identifiers")
    if not (gap_rows["type"].fillna("").astype(str).str.strip() == "between_fills").all():
        raise ValueError("frames outside recipe ranges must have type=between_fills")
    if gap_rows[[*VOLUME_COLUMNS, "total_ml"]].notna().any().any():
        raise ValueError("between-fill gap rows cannot contain recipe volumes")

    report = {
        "schema_version": "1.0",
        "adapter": "synthetic_recipe_video_v1",
        "video": asdict(video) | {"duration_s": video.duration_s},
        "frame_label_rows": len(labels),
        "recipe_rows": len(ordered_recipes),
        "vials": [int(value) for value in ordered_recipes["vial"]],
        "recipe_ids": [int(value) for value in ordered_recipes["recipe_id"]],
        "gap_frames": int((labels["in_fill"] == 0).sum()),
        "ground_truth_anomaly": None,
        "label_status": "synthetic_unverified",
    }
    return labels, ordered_recipes, report


def select_stage_rows(
    labels: pd.DataFrame,
    recipes: pd.DataFrame,
    fps: float,
    *,
    pre_offset_s: float = 1.0,
    post_offset_s: float = 1.0,
) -> list[tuple[dict[str, Any], str, pd.Series]]:
    selected: list[tuple[dict[str, Any], str, pd.Series]] = []
    for recipe in recipes.to_dict(orient="records"):
        start, end = int(recipe["start_frame"]), int(recipe["end_frame"])
        duration = end - start
        targets = {
            "early_fill": start + round(0.15 * duration),
            "mid_fill": start + round(0.50 * duration),
            "late_fill": start + round(0.85 * duration),
        }
        for stage, target in targets.items():
            selected.append((recipe, stage, labels.iloc[target]))

        pre_rows = labels.loc[(labels["in_fill"] == 0) & (labels["mp4_frame"] < start)]
        if not pre_rows.empty:
            target = start - round(pre_offset_s * fps)
            index = (pre_rows["mp4_frame"] - target).abs().idxmin()
            selected.append((recipe, "pre_fill", labels.loc[index]))

        post_rows = labels.loc[(labels["in_fill"] == 0) & (labels["mp4_frame"] > end)]
        next_starts = recipes.loc[recipes["start_frame"] > end, "start_frame"]
        if not next_starts.empty:
            post_rows = post_rows.loc[post_rows["mp4_frame"] < int(next_starts.min())]
        if not post_rows.empty:
            target = end + round(post_offset_s * fps)
            index = (post_rows["mp4_frame"] - target).abs().idxmin()
            selected.append((recipe, "post_fill", labels.loc[index]))
    return sorted(selected, key=lambda item: (int(item[0]["vial"]), int(item[2]["mp4_frame"])))


def _validate_roi(features_config_path: Path, video: VideoMetadata) -> None:
    config = load_yaml(features_config_path)
    roi = config["roi"]
    x, y = int(roi["x"]), int(roi["y"])
    width, height = int(roi["width"]), int(roi["height"])
    if x < 0 or y < 0 or x + width > video.width or y + height > video.height:
        raise ValueError(
            f"configured color ROI {(x, y, width, height)} lies outside synthetic video "
            f"dimensions {(video.width, video.height)}"
        )


def extract_synthetic_frames(
    video_path: Path,
    output_root: Path,
    dataset_id: str,
    labels: pd.DataFrame,
    recipes: pd.DataFrame,
    video: VideoMetadata,
    *,
    camera_id: str = "FillCam",
    video_id: int = 1,
    pre_offset_s: float = 1.0,
    post_offset_s: float = 1.0,
) -> pd.DataFrame:
    _validate_dataset_id(dataset_id)
    _validate_camera_id(camera_id)
    if video_id < 1:
        raise ValueError("video_id must be positive")
    interim_root = output_root / "data" / "interim" / "synthetic" / dataset_id
    frame_root = interim_root / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(video_path)
    recipe_ids = recipes["recipe_id"].tolist()
    transition_by_vial: dict[int, int] = {}
    transition = 0
    previous: int | None = None
    for recipe in recipes.to_dict(orient="records"):
        recipe_id = int(recipe["recipe_id"])
        if previous is not None and recipe_id != previous:
            transition += 1
        transition_by_vial[int(recipe["vial"])] = transition
        previous = recipe_id
    if not recipe_ids:
        raise ValueError("synthetic recipe table is empty")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"cannot open synthetic video: {video_path}")
    records: list[dict[str, Any]] = []
    try:
        for recipe, stage, label in select_stage_rows(
            labels,
            recipes,
            video.fps,
            pre_offset_s=pre_offset_s,
            post_offset_s=post_offset_s,
        ):
            frame_index = int(label["mp4_frame"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            decoded_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if not ok or image is None:
                raise OSError(f"cannot decode synthetic frame {frame_index} from {video_path}")
            if decoded_index != frame_index:
                raise OSError(
                    f"requested synthetic frame {frame_index}, but decoder returned {decoded_index}"
                )
            vial = int(recipe["vial"])
            destination = frame_root / f"{dataset_id}_vial{vial:02d}_{stage}.jpg"
            if not cv2.imwrite(str(destination), image):
                raise OSError(f"could not write extracted frame: {destination}")
            video_s = float(label["video_s"])
            records.append(
                {
                    "video_id": video_id,
                    "dataset_id": dataset_id,
                    "camera_id": camera_id,
                    "vial_id": vial,
                    "recipe_id": int(recipe["recipe_id"]),
                    "recipe_transition_id": transition_by_vial[vial],
                    "stage": stage,
                    "original_wall_seconds": video_s,
                    "corrected_media_seconds": video_s,
                    "decoded_media_seconds": decoded_index / video.fps,
                    "decoded_frame_index": decoded_index,
                    "alignment_residual_seconds": decoded_index / video.fps - video_s,
                    "alignment_confidence": 1.0,
                    "alignment_method": "synthetic_direct_frame_index",
                    "source_video": str(video_path),
                    "source_sha256": source_hash,
                    "source_frame_label": str(label["filename"]),
                    "render_frame": int(label["render_frame"]),
                    "measurement_eligible": bool(label["in_fill"]),
                    "image_path": destination.relative_to(output_root).as_posix(),
                    "extraction_config": "synthetic_recipe_video_v1",
                    "ground_truth_anomaly": None,
                    "label_status": "synthetic_unverified",
                }
            )
    finally:
        capture.release()
    manifest = pd.DataFrame(records).sort_values(["vial_id", "corrected_media_seconds"])
    manifest.to_csv(interim_root / "extracted_frames.csv", index=False)
    return manifest


def process_synthetic_dataset(
    video_path: Path,
    frame_labels_path: Path,
    recipes_path: Path,
    output_root: Path,
    dataset_id: str,
    features_config_path: Path | None = None,
    *,
    camera_id: str = "FillCam",
    video_id: int = 1,
    validate_only: bool = False,
) -> dict[str, Any]:
    _validate_dataset_id(dataset_id)
    _validate_camera_id(camera_id)
    if video_id < 1:
        raise ValueError("video_id must be positive")
    video_path = video_path.resolve()
    frame_labels_path = frame_labels_path.resolve()
    recipes_path = recipes_path.resolve()
    output_root = output_root.resolve()
    for path in (frame_labels_path, recipes_path):
        if not path.is_file():
            raise FileNotFoundError(f"required synthetic input does not exist: {path}")
    video = probe_video(video_path)
    labels, recipes, report = validate_synthetic_tables(
        pd.read_csv(frame_labels_path), pd.read_csv(recipes_path), video
    )
    report.update(
        {
            "dataset_id": dataset_id,
            "video_id": video_id,
            "camera_id": camera_id,
            "video_path": str(video_path),
            "video_sha256": sha256_file(video_path),
            "frame_labels_path": str(frame_labels_path),
            "frame_labels_sha256": sha256_file(frame_labels_path),
            "recipes_path": str(recipes_path),
            "recipes_sha256": sha256_file(recipes_path),
        }
    )
    report_path = output_root / "reports" / "synthetic" / f"{dataset_id}_validation.json"
    atomic_json_dump(report, report_path)
    result: dict[str, Any] = {"validation_report": str(report_path), **report}
    if validate_only:
        return result

    if camera_id != "FillCam":
        raise ValueError(
            "CapSealCam color extraction is disabled until a camera-specific ROI is calibrated"
        )
    if features_config_path is None:
        raise ValueError(
            "features_config_path is required when processing synthetic color features"
        )
    features_config_path = features_config_path.resolve()
    if not features_config_path.is_file():
        raise FileNotFoundError(
            f"required feature configuration does not exist: {features_config_path}"
        )
    _validate_roi(features_config_path, video)
    interim_root = output_root / "data" / "interim" / "synthetic" / dataset_id
    manifest_path = interim_root / "extracted_frames.csv"
    feature_path = interim_root / "frame_features.csv"
    qa_root = output_root / "artifacts" / "qa" / "synthetic" / dataset_id / "color"
    processed_root = output_root / "data" / "processed" / "synthetic" / dataset_id
    manifest = extract_synthetic_frames(
        video_path,
        output_root,
        dataset_id,
        labels,
        recipes,
        video,
        camera_id=camera_id,
        video_id=video_id,
    )
    features = extract_features(
        output_root,
        features_config_path,
        manifest_path=manifest_path,
        qa_root=qa_root,
        feature_output_path=feature_path,
    )
    frame_data, vial_data = build_datasets(
        recipes_path.parent,
        output_root,
        features_path=feature_path,
        recipes_path=recipes_path,
        processed_root=processed_root,
        previous_recipe_by_vial=True,
    )
    result.update(
        {
            "manifest": str(manifest_path),
            "feature_table": str(feature_path),
            "processed_root": str(processed_root),
            "qa_root": str(qa_root),
            "extracted_frames": len(manifest),
            "feature_rows": len(features),
            "per_frame_rows": len(frame_data),
            "per_vial_rows": len(vial_data),
        }
    )
    return result
