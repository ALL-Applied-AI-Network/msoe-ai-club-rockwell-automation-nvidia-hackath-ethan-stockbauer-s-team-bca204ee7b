from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from mixing_copilot.alignment.matching import (
    build_video_signature_index,
    match_still,
    read_frame_at_time,
)
from mixing_copilot.alignment.model import fit_affine_alignment
from mixing_copilot.io.files import atomic_json_dump, sha256_file


def _annotate(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(result, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return result


def _write_qa_sheet(rows: pd.DataFrame, image_dir: Path, video: Path, output: Path) -> None:
    selected_positions = sorted(set(np.linspace(0, len(rows) - 1, min(8, len(rows)), dtype=int)))
    panels: list[np.ndarray] = []
    for position in selected_positions:
        row = rows.iloc[position]
        still = cv2.imread(str(image_dir / row["filename"]), cv2.IMREAD_COLOR)
        decoded, actual_s, _ = read_frame_at_time(video, float(row["matched_media_s"]))
        still = _annotate(still, f"still wall={row['wall_seconds']:.2f}s")
        decoded = _annotate(decoded, f"video media={actual_s:.2f}s score={row['match_score']:.3f}")
        panels.append(np.hstack([still, decoded]))
    if panels:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), np.vstack(panels))


def run_alignment(
    dataset_root: Path,
    output_root: Path,
    camera: str,
    video_id: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".tools" / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video = dataset_root / camera / "videos" / f"{camera}_video_{video_id}.mp4"
    labels_path = dataset_root / "vial_fill" / "labels" / f"video{video_id}_frame_labels.csv"
    image_dir = dataset_root / camera / "images"
    if not video.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"alignment inputs missing: {video}, {labels_path}")

    interim = output_root / "data" / "interim" / "alignment"
    qa = output_root / "artifacts" / "qa" / "alignment"
    interim.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    cache = interim / f"{camera}_video_{video_id}_signatures.npz"
    if cache.is_file():
        cached = np.load(cache)
        times, signatures = cached["times"], cached["signatures"]
    else:
        times, signatures = build_video_signature_index(
            video, step_s=float(settings.get("coarse_step_s", 0.25))
        )
        np.savez_compressed(cache, times=times, signatures=signatures)

    labels = pd.read_csv(labels_path)
    labels = labels.loc[labels["filename"].notna() & labels["video_s"].notna()].reset_index(
        drop=True
    )
    maximum = min(int(settings.get("max_anchors", 140)), len(labels))
    sample_positions = np.linspace(1, len(labels) - 2, maximum, dtype=int)
    offset = float(settings.get("bootstrap_offset_s", 0.63))
    scale = float(settings.get("bootstrap_scale", 1.216))
    radius = float(settings.get("search_radius_s", 3.0))
    records: list[dict[str, Any]] = []
    for position in sample_positions:
        row = labels.iloc[position]
        wall_s = float(row["video_s"])
        predicted_s = (wall_s - offset) / scale
        if predicted_s < times[0] or predicted_s > times[-1]:
            continue
        match = match_still(image_dir / row["filename"], times, signatures, predicted_s, radius)
        records.append(
            {
                "filename": row["filename"],
                "wall_seconds": wall_s,
                "bootstrap_media_s": predicted_s,
                "matched_media_s": match.media_seconds,
                "match_score": match.score,
                "second_best_margin": match.second_best_margin,
            }
        )
    anchors = pd.DataFrame(records)
    if len(anchors) < int(settings.get("minimum_inliers", 12)):
        raise RuntimeError(f"only {len(anchors)} visual anchors could be matched")
    fit, residuals, inliers = fit_affine_alignment(
        anchors["matched_media_s"].to_numpy(),
        anchors["wall_seconds"].to_numpy(),
        residual_threshold_s=float(settings.get("residual_threshold_s", 0.35)),
        minimum_inliers=int(settings.get("minimum_inliers", 12)),
    )
    anchors["fit_residual_s"] = residuals
    anchors["fit_inlier"] = inliers
    anchors_path = interim / f"{camera}_video_{video_id}_anchors.csv"
    anchors.to_csv(anchors_path, index=False)

    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.scatter(
        anchors["matched_media_s"], residuals, c=np.where(inliers, "#1f77b4", "#d62728"), s=18
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Matched media time (s)",
        ylabel="Wall-clock fit residual (s)",
        title="Affine alignment residuals",
    )
    fig.tight_layout()
    residual_plot = qa / f"{camera}_video_{video_id}_residuals.png"
    fig.savefig(residual_plot, dpi=150)
    plt.close(fig)
    qa_sheet = qa / f"{camera}_video_{video_id}_visual_matches.jpg"
    _write_qa_sheet(anchors.loc[inliers].reset_index(drop=True), image_dir, video, qa_sheet)

    result = {
        "schema_version": "1.0",
        "camera_id": camera,
        "video_id": video_id,
        "source_video": str(video),
        "source_sha256": sha256_file(video),
        "mapping": "wall_seconds = offset_s + scale * media_seconds",
        "bootstrap": {"offset_s": offset, "scale": scale, "status": "provisional_search_hint_only"},
        "fit": fit.to_dict(),
        "piecewise_fit": {
            "used": False,
            "reason": "not justified without residual discontinuity evidence",
        },
        "anchors_csv": str(anchors_path),
        "residual_plot": str(residual_plot),
        "visual_qa": str(qa_sheet),
    }
    manifest_path = interim / f"{camera}_video_{video_id}_alignment.json"
    atomic_json_dump(result, manifest_path)
    result["manifest"] = str(manifest_path)
    return result
