from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from mixing_copilot.io.files import iter_files, sha256_file

CAMERAS = ("vial_fill", "tank_level", "cap_seal")


def _video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"readable": False}
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    result = {
        "readable": True,
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frame_count": frames,
        "duration_s": frames / fps if fps > 0 else None,
        "codec_fourcc": int(capture.get(cv2.CAP_PROP_FOURCC)),
    }
    capture.release()
    return result


def _csv_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "null_counts": {key: int(value) for key, value in frame.isna().sum().items()},
    }


def audit_dataset(root: Path, include_hashes: bool = True) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    required = [root / "README.md", root / "videos.csv"]
    for camera in CAMERAS:
        required.extend(
            [root / camera / "image_index.csv", root / camera / "images", root / camera / "videos"]
        )
        required.extend(
            root / camera / "videos" / f"{camera}_video_{index}.mp4" for index in range(1, 6)
        )
    required.extend(
        [
            root / "vial_fill" / "labels" / "video5_frame_labels.csv",
            root / "vial_fill" / "labels" / "video5_vial_recipes.csv",
            root / "vial_fill" / "labels" / "video5_vial_recipes.json",
        ]
    )

    files = list(iter_files(root))
    extension_counts = Counter(path.suffix.lower() or "(none)" for path in files)
    records = []
    for path in files:
        stat = path.stat()
        record = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        }
        if include_hashes:
            record["sha256"] = sha256_file(path)
        records.append(record)

    csvs = {path.relative_to(root).as_posix(): _csv_summary(path) for path in root.rglob("*.csv")}
    videos = {
        path.relative_to(root).as_posix(): _video_metadata(path)
        for path in sorted(root.rglob("*.mp4"))
    }
    recipes_path = root / "vial_fill" / "labels" / "video5_vial_recipes.csv"
    recipe_coverage: dict[str, Any] = {}
    if recipes_path.is_file():
        recipes = pd.read_csv(recipes_path)
        recipe_coverage = {
            "vials": int(recipes["vial"].nunique()),
            "recipe_ids": sorted(int(item) for item in recipes["recipe_id"].dropna().unique()),
            "vials_per_recipe": {
                str(int(key)): int(value)
                for key, value in recipes.groupby("recipe_id")["vial"].nunique().items()
            },
        }

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_root": str(root),
        "source_policy": "read_only_external_dataset",
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in records),
        "extension_counts": dict(sorted(extension_counts.items())),
        "missing_required": [str(path.relative_to(root)) for path in required if not path.exists()],
        "files": records,
        "csvs": csvs,
        "videos": videos,
        "recipe_coverage": recipe_coverage,
        "label_caveats": {
            "ground_truth_anomaly": None,
            "label_status": "unverified",
            "recipe_volumes": "setpoints_not_measured_dispense",
        },
    }
